// Wire protocol between the BitNet weight-sharded-TP orchestrator (the dispatch
// code added to ggml_bitnet_mul_mat in src/ggml-bitnet-lut.cpp) and
// bitnet-tp-worker processes.
//
// All peers are identical aarch64 CM4s on the same 1GbE LAN, so multi-byte
// fields are sent native-endian with no htonl/ntohl conversion -- if a
// non-aarch64 or cross-endian peer is ever added to the fleet, conversion
// must be added here.
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>

#include <sys/socket.h>
#include <unistd.h>

#define TP_PROTO_MAGIC   0x42544C31u   // "BTL1"
#define TP_PROTO_VERSION 1u

enum tp_msg_type : uint32_t {
    TP_MSG_MATMUL_REQUEST  = 1,
    TP_MSG_MATMUL_RESPONSE = 2,
    TP_MSG_PING            = 3,
    TP_MSG_PONG            = 4,
    TP_MSG_ERROR           = 5,
    TP_MSG_BATCH_REQUEST   = 6,
    TP_MSG_BATCH_RESPONSE  = 7,
};

enum tp_tensor_kind : uint32_t {
    TP_TENSOR_ATTN_Q      = 0,
    TP_TENSOR_ATTN_K      = 1,
    TP_TENSOR_ATTN_V      = 2,
    TP_TENSOR_ATTN_OUTPUT = 3,
    TP_TENSOR_FFN_GATE    = 4,
    TP_TENSOR_FFN_UP      = 5,
    TP_TENSOR_FFN_DOWN    = 6,
    TP_TENSOR_KIND_COUNT  = 7,
};

// 16 bytes, precedes every payload in both directions.
struct tp_msg_header {
    uint32_t magic;
    uint32_t version;
    uint32_t type;        // tp_msg_type
    uint32_t payload_len; // bytes following this header
};

// Payload header for TP_MSG_MATMUL_REQUEST; raw `float activation[k]` follows
// immediately (k*4 bytes, native-endian F32, matches ggml_tensor's src1->data
// layout directly -- no copy/reformat needed on the orchestrator side).
struct tp_matmul_request {
    uint32_t seq_id;      // orchestrator-assigned, monotonic; echoed back
    uint32_t layer_idx;
    uint32_t tensor_kind; // tp_tensor_kind
    uint32_t m_full;      // ORIGINAL (unsharded) output-row count -- selects the
                           // ggml_qgemm_lut/ggml_preprocessor kernel specialization,
                           // never BM * n_local_tiles.
    uint32_t k;
};

// Payload header for TP_MSG_MATMUL_RESPONSE; raw `float result[n_rows]` follows
// immediately -- the worker's owned tiles concatenated in ascending global
// tile-index order (matches the manifest's tile_owners ordering), so the
// orchestrator can scatter-copy with no extra bookkeeping.
struct tp_matmul_response {
    uint32_t seq_id;
    uint32_t layer_idx;
    uint32_t tensor_kind;
    uint32_t n_rows;
};

// Payload for TP_MSG_ERROR -- sent instead of TP_MSG_MATMUL_RESPONSE when a
// worker can't service a request (e.g. shard/manifest mismatch).
struct tp_error_response {
    uint32_t seq_id;
    uint32_t code;
};

// Q/K/V (and separately ffn_gate/ffn_up) all read the exact same activation
// vector, but computing them as 3 (or 2) separate TP_MSG_MATMUL_REQUESTs --
// even dispatched concurrently on separate connections -- still serializes
// their GEMM compute on the worker, since there's only one shared 4-core
// tile-compute pool per worker (see tp_compute_helper in
// tools/tp-worker/main.cpp). A batch request instead ships the whole
// group's tensor kinds in one message, built from one shared activation and
// one shared LUT, so the worker can split ALL of the group's tiles across
// its cores in a single pass instead of one tensor at a time.
//
// Payload header for TP_MSG_BATCH_REQUEST; followed immediately by
// `n_items` x tp_batch_item, then raw `float activation[k]` (shared).
struct tp_batch_request_header {
    uint32_t seq_id;
    uint32_t layer_idx;
    uint32_t k;      // shared activation width for every item in the batch
    uint32_t n_items;
};

struct tp_batch_item {
    uint32_t tensor_kind;
    uint32_t m_full; // this item's own original output-row count
};

// Payload header for TP_MSG_BATCH_RESPONSE; followed by `n_items` x
// tp_batch_result_item, then the concatenated `float result[...]` for all
// items back to back, in the same order as the request's items.
struct tp_batch_response_header {
    uint32_t seq_id;
    uint32_t layer_idx;
    uint32_t n_items;
};

struct tp_batch_result_item {
    uint32_t tensor_kind;
    uint32_t n_rows;
};

inline const char * tp_tensor_kind_name(uint32_t kind) {
    switch (kind) {
        case TP_TENSOR_ATTN_Q:      return "attn_q";
        case TP_TENSOR_ATTN_K:      return "attn_k";
        case TP_TENSOR_ATTN_V:      return "attn_v";
        case TP_TENSOR_ATTN_OUTPUT: return "attn_output";
        case TP_TENSOR_FFN_GATE:    return "ffn_gate";
        case TP_TENSOR_FFN_UP:      return "ffn_up";
        case TP_TENSOR_FFN_DOWN:    return "ffn_down";
        default:                    return "unknown";
    }
}

inline bool tp_tensor_kind_from_name(const char * name, uint32_t * out_kind) {
    static const char * names[TP_TENSOR_KIND_COUNT] = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
    };
    for (uint32_t i = 0; i < TP_TENSOR_KIND_COUNT; i++) {
        if (strcmp(name, names[i]) == 0) {
            *out_kind = i;
            return true;
        }
    }
    return false;
}

// GGUF tensor names here are "blk.<layer>.<kind>.weight" -- recover (layer_idx,
// tensor_kind) straight from ggml_tensor::name with no model-loading plumbing.
inline bool tp_parse_tensor_name(const char * tensor_name, uint32_t * layer_idx, uint32_t * tensor_kind) {
    unsigned layer;
    char kind_buf[32];
    if (sscanf(tensor_name, "blk.%u.%31[^.]", &layer, kind_buf) != 2) {
        return false;
    }
    if (!tp_tensor_kind_from_name(kind_buf, tensor_kind)) {
        return false;
    }
    *layer_idx = layer;
    return true;
}

// TCP can short-read/short-write; these loop until `len` bytes are moved or the
// connection fails. Used by both the worker and the orchestrator's dispatch code.
inline bool tp_send_all(int fd, const void * buf, size_t len) {
    const uint8_t * p = (const uint8_t *) buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        sent += (size_t) n;
    }
    return true;
}

inline bool tp_recv_all(int fd, void * buf, size_t len) {
    uint8_t * p = (uint8_t *) buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            return false;
        }
        got += (size_t) n;
    }
    return true;
}

inline bool tp_send_msg(int fd, tp_msg_type type, const void * payload, uint32_t payload_len) {
    tp_msg_header hdr{TP_PROTO_MAGIC, TP_PROTO_VERSION, (uint32_t) type, payload_len};
    if (!tp_send_all(fd, &hdr, sizeof(hdr))) {
        return false;
    }
    if (payload_len == 0) {
        return true;
    }
    return tp_send_all(fd, payload, payload_len);
}

// Reads a header and validates magic/version; returns false on I/O failure or
// a bad header (caller should drop the connection in either case).
inline bool tp_recv_header(int fd, tp_msg_header * out) {
    if (!tp_recv_all(fd, out, sizeof(*out))) {
        return false;
    }
    return out->magic == TP_PROTO_MAGIC && out->version == TP_PROTO_VERSION;
}
