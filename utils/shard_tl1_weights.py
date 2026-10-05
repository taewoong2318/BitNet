#!/usr/bin/env python3
"""Slice a TL1 GGUF's per-layer ternary weight tensors into per-device row-tile
shards for weight-sharded tensor-parallel inference, plus a manifest describing
which device owns which tiles of which tensor.

Each TL1 tensor's packed byte blob is already laid out as `M // BM` independent,
byte-contiguous row-tiles (see `process_tl1()` in convert-hf-to-gguf-bitnet.py --
the `M // BM` axis is never moved out of the leading position through that whole
reshape/transpose chain), so slicing at tile boundaries is a pure byte-copy: no
re-quantization needed. See the implementation plan this script was generated
from for the full design rationale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from configparser import ConfigParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

TENSOR_KINDS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
TENSOR_NAME_RE = re.compile(r"^blk\.(\d+)\.(" + "|".join(TENSOR_KINDS) + r")\.weight$")


def load_kernel_config(path: Path) -> list[dict]:
    config = ConfigParser()
    config.read(str(path))
    shapes = []
    for section in config.sections():
        shapes.append({
            "m": int(config.get(section, "m")),
            "k": int(config.get(section, "k")),
            "bm": int(config.get(section, "bm")),
        })
    return shapes


def bm_for_shape(shapes: list[dict], m: int, k: int) -> int:
    for shape in shapes:
        if shape["m"] == m and shape["k"] == k:
            return shape["bm"]
    raise NotImplementedError(f"no kernel_config.ini entry for m={m} k={k}")


def parse_workers(spec: str) -> list[tuple[str, int]]:
    workers = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, port = entry.rsplit(":", 1)
        workers.append((host, int(port)))
    return workers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="Path to the source TL1 GGUF")
    ap.add_argument("--kernel-config", default=Path("include/kernel_config.ini"), type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", required=True,
                     help="Comma-separated host:port list of bitnet-tp-worker instances, "
                          "in rank order (0-indexed). Use the CM4s' local 1GbE LAN "
                          "addresses, not Tailscale hostnames -- this is hot-path traffic.")
    ap.add_argument("--orchestrator-participates", action="store_true",
                     help="Reserve one extra rank (after the network workers) for the "
                          "orchestrator's own local compute -- no shard file is written "
                          "for it, since the orchestrator already holds the full tensor.")
    args = ap.parse_args()

    workers = parse_workers(args.workers)
    if not workers:
        raise SystemExit("--workers must list at least one host:port")

    w_net = len(workers)
    world_size = w_net + (1 if args.orchestrator_participates else 0)
    self_rank = w_net if args.orchestrator_participates else None

    shapes = load_kernel_config(args.kernel_config)
    reader = gguf.GGUFReader(str(args.model))

    # rank -> { "blk.{layer}.{kind}.shard": np.uint8 array }
    shard_blobs: dict[int, dict[str, np.ndarray]] = {r: {} for r in range(w_net)}
    manifest_tensors: dict[str, dict] = {}

    n_tensors_sharded = 0
    for tensor in reader.tensors:
        if tensor.tensor_type != gguf.GGMLQuantizationType.TL1:
            continue
        match = TENSOR_NAME_RE.match(tensor.name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        kind = match.group(2)
        kind_idx = TENSOR_KINDS.index(kind)

        # dims is on-disk order [k, m] (ne[0]=k fastest, ne[1]=m), matching
        # kernel_config.ini's (m, k) kernel-selector pairs.
        k, m = int(tensor.shape[0]), int(tensor.shape[1])
        bm = bm_for_shape(shapes, m, k)
        n_tiles = m // bm
        tile_size_bytes = k * bm // 4

        blob = np.asarray(tensor.data).reshape(-1)
        if blob.nbytes != n_tiles * tile_size_bytes:
            raise ValueError(
                f"{tensor.name}: expected {n_tiles * tile_size_bytes} packed bytes, "
                f"got {blob.nbytes}")

        start = (layer_idx * len(TENSOR_KINDS) + kind_idx) % world_size
        tile_owners = [(start + i) % world_size for i in range(n_tiles)]

        for rank in range(w_net):
            owned = [i for i, owner in enumerate(tile_owners) if owner == rank]
            if not owned:
                continue
            pieces = [blob[i * tile_size_bytes:(i + 1) * tile_size_bytes] for i in owned]
            shard_blobs[rank][f"blk.{layer_idx}.{kind}.shard"] = np.concatenate(pieces)

        manifest_tensors[f"{layer_idx}:{kind}"] = {
            "m": m, "k": k, "bm": bm, "n_tiles": n_tiles, "tile_owners": tile_owners,
        }
        n_tensors_sharded += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rank, (host, port) in enumerate(workers):
        shard_path = args.out_dir / f"shard.rank{rank}.gguf"
        writer = gguf.GGUFWriter(str(shard_path), arch="bitnet-tp-shard")
        for name, data in shard_blobs[rank].items():
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.I8)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"rank {rank} ({host}:{port}): {len(shard_blobs[rank])} tensor shards -> {shard_path}")

    manifest = {
        "world_size": world_size,
        "orchestrator_participates": args.orchestrator_participates,
        "self_rank": self_rank,
        "workers": [{"rank": r, "host": h, "port": p} for r, (h, p) in enumerate(workers)],
        "tensors": manifest_tensors,
    }
    manifest_path = args.out_dir / "tp_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest for {n_tensors_sharded} tensors ({world_size}-way split) -> {manifest_path}")


if __name__ == "__main__":
    main()
