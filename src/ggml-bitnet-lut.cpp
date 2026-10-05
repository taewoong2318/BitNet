#include <vector>
#include <type_traits>
#include <algorithm>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
#include "bitnet-lut-kernels.h"
#endif

#if defined(GGML_BITNET_ARM_TL1)

#if defined(GGML_BITNET_TP)
// Weight-sharded tensor-parallel dispatch: when BITNET_TP_MANIFEST is set, some
// (or all) of a tensor's M-row tiles are computed by remote bitnet-tp-worker
// processes instead of locally. Row-sharding means this is a pure
// concatenation of disjoint output slices, not a numerical reduction -- see
// tools/tp-worker/main.cpp, which runs the identical ggml_preprocessor/
// ggml_qgemm_lut calls this file's own tile loop below uses.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json.hpp"
#include "tp_proto.h"

namespace {

struct tp_route {
    int m_full = 0;
    int k = 0;
    int n_tiles = 0;
    std::vector<int> local_tiles;                              // tile indices computed by this orchestrator itself
    std::vector<std::pair<int, std::vector<int>>> worker_tiles; // (rank, ascending tile indices), rank order fixed at init
};

// One persistent socket per worker, read and written directly from the
// dispatching thread (ith==0) via poll() -- no handler thread, no
// mutex/condvar. An earlier version routed each request through a
// persistent-but-separate handler thread (dispatcher -> cv_request ->
// handler -> socket -> handler -> cv_response -> dispatcher) to avoid
// spawning a fresh std::thread per call; that fixed the thread-spawn cost,
// but each request/response still paid two extra futex wake/context-switch
// hops on top of the actual network round trip. Measurement (a raw
// select()-based Python probe hitting the same workers with the same wire
// protocol, bypassing this file's dispatch code entirely) showed that gap:
// the probe's round trip was faster than this file's own gather time by
// close to the margin those two hops would cost. Doing the socket I/O
// directly here -- send to every worker in the group, then poll() for
// whichever responds first, mirroring the probe -- removes that gap.
//
// Request/response carry a *list* of (tensor_kind, m_full) items sharing
// one activation, not just one tensor: Q/K/V all read the exact same
// activation (the output of attn_norm), and so do ffn_gate/ffn_up (the
// output of ffn_norm), but ggml's graph executor calls their mul_mat ops
// one at a time. An earlier version dispatched each group member as its
// OWN request -- but measurement showed the worker's shared 4-core compute
// pool still serialized their GEMM compute one tensor at a time, since each
// arrived as a separate request. Sending the WHOLE group as one combined
// request per worker lets the worker split the group's combined tiles
// across its cores in a single pass instead (see TP_MSG_BATCH_REQUEST in
// tp_proto.h and tools/tp-worker/main.cpp).
struct tp_worker_conn {
    int fd = -1;
    std::vector<uint8_t> send_buf;                                             // reused scratch for the batched request below
    std::vector<std::pair<uint32_t, std::vector<bitnet_float_type>>> result_items; // (tensor_kind, its rows), filled per dispatch
};

// Groups of tensor kinds that share one activation and have no data
// dependency on each other, so their requests can be combined into one
// per-worker batch instead of dispatched (and compute-serialized) one at a
// time. attn_output/ffn_down each depend on their own group's results, so
// there's nothing to group them with -- irreducibly sequential.
const std::vector<uint32_t> TP_GROUP_QKV = {TP_TENSOR_ATTN_Q, TP_TENSOR_ATTN_K, TP_TENSOR_ATTN_V};
const std::vector<uint32_t> TP_GROUP_GATEUP = {TP_TENSOR_FFN_GATE, TP_TENSOR_FFN_UP};

const std::vector<uint32_t> * tp_group_members(uint32_t tensor_kind) {
    switch (tensor_kind) {
        case TP_TENSOR_ATTN_Q: case TP_TENSOR_ATTN_K: case TP_TENSOR_ATTN_V:
            return &TP_GROUP_QKV;
        case TP_TENSOR_FFN_GATE: case TP_TENSOR_FFN_UP:
            return &TP_GROUP_GATEUP;
        default:
            return nullptr;
    }
}

std::vector<std::unique_ptr<tp_worker_conn>> g_tp_worker_conns;
std::unordered_map<uint64_t, tp_route> g_tp_routes;
// Holds a group member's full result (keyed by route_key(layer,kind)) once
// fetched as part of its group's dispatch, until that member's own
// ggml_bitnet_mul_mat call comes along and consumes it -- at that point it's
// a pure memcpy, no network wait at all.
std::unordered_map<uint64_t, std::vector<bitnet_float_type>> g_tp_prefetch_cache;
bool g_tp_enabled = false;
uint32_t g_tp_seq = 0;

// Scratch for broadcasting ith==0's decisions to the other orchestrator
// threads below (whether this call's result already arrived via another
// group member's dispatch, and the seq_id to send with) so they can join
// the fan-out send loop. Safe as plain globals: ggml's CPU threadpool runs
// one graph node across all nth threads at a time, with a barrier between
// writer and readers, so there's no concurrent reentrancy to race on.
bool g_tp_scratch_consumed_from_cache = false;
uint32_t g_tp_scratch_seq_id = 0;

// Diagnostic only (BITNET_TP_DEBUG_TIMING=1): accumulates wall-clock time
// spent in the two halves of the dispatch -- fan-out (lock+notify per
// worker) and gather (lock+wait+memcpy per worker) -- to find out how much
// of the per-call cost is the orchestrator's own dispatch code vs. the
// network+worker time already measured separately with tmp_matmul_probe2.py.
bool g_tp_debug_timing = false;
std::atomic<uint64_t> g_tp_fanout_ns{0};
std::atomic<uint64_t> g_tp_gather_ns{0};
std::atomic<uint64_t> g_tp_call_count{0};

uint64_t tp_route_key(uint32_t layer_idx, uint32_t tensor_kind) {
    return (uint64_t(layer_idx) << 8) | tensor_kind;
}

// g_tp_prefetch_cache must be keyed by the batch column too: n_tokens is
// usually 1, but when it isn't (llama.cpp's multi-token capability probes
// at model load, or a batched multi-token prompt eval), ggml_bitnet_mul_mat
// loops over each column *within* the same call -- without `tok` in the
// key, column 1's cached K/V would silently overwrite column 0's before
// column 0's own K/V call ever consumed it.
uint64_t tp_prefetch_key(uint32_t tok, uint32_t layer_idx, uint32_t tensor_kind) {
    return (uint64_t(tok) << 32) | tp_route_key(layer_idx, tensor_kind);
}

int tp_connect(const std::string & host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    GGML_ASSERT(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    GGML_ASSERT(inet_pton(AF_INET, host.c_str(), &addr.sin_addr) == 1 && "bitnet-tp: invalid worker host in manifest");
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        fprintf(stderr, "bitnet-tp: failed to connect to worker %s:%d: %s\n",
                host.c_str(), port, strerror(errno));
        exit(1);
    }
    // This is a small-message request/response protocol (one activation
    // vector out, one result slice back per call) -- without TCP_NODELAY,
    // Nagle's algorithm interacting with the peer's delayed ACKs can add
    // tens of milliseconds per round trip, which dominates everything else
    // given 210 of these round trips happen per generated token.
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

// Builds and sends one batch request (one or more items sharing one
// activation) on wc->fd. Does not wait for a response -- the caller sends
// to every worker in the group first, then polls all of them together; see
// the dispatch site in ggml_bitnet_mul_mat.
bool tp_send_batch_request(tp_worker_conn * wc, uint32_t seq_id, uint32_t layer_idx, int k,
                            const std::vector<std::pair<uint32_t, uint32_t>> & items,
                            const float * activation) {
    tp_batch_request_header req{seq_id, layer_idx, (uint32_t) k, (uint32_t) items.size()};
    const size_t items_bytes = items.size() * sizeof(tp_batch_item);
    const size_t payload_len = sizeof(req) + items_bytes + (size_t) k * sizeof(float);
    tp_msg_header hdr{TP_PROTO_MAGIC, TP_PROTO_VERSION, TP_MSG_BATCH_REQUEST, (uint32_t) payload_len};

    // One send() for the whole request -- see the original single-tensor
    // version's comment (now folded into this batch form) for why.
    wc->send_buf.resize(sizeof(hdr) + payload_len);
    uint8_t * p = wc->send_buf.data();
    memcpy(p, &hdr, sizeof(hdr));                      p += sizeof(hdr);
    memcpy(p, &req, sizeof(req));                      p += sizeof(req);
    for (auto & [kind, m_full] : items) {
        tp_batch_item item{kind, m_full};
        memcpy(p, &item, sizeof(item));
        p += sizeof(item);
    }
    memcpy(p, activation, (size_t) k * sizeof(float));
    return tp_send_all(wc->fd, wc->send_buf.data(), wc->send_buf.size());
}

// Reads one complete batch response from wc->fd into wc->result_items.
// Called only after poll() has reported wc->fd readable, but still uses the
// blocking tp_recv_all helpers: readiness only guarantees *some* bytes are
// available, not the whole message, and a response this small (a header
// plus a few KB of floats) typically arrives in one segment anyway -- the
// same assumption the probe that validated this design made.
bool tp_recv_batch_response(tp_worker_conn * wc, uint32_t layer_idx) {
    wc->result_items.clear();

    tp_msg_header resp_hdr;
    if (!tp_recv_header(wc->fd, &resp_hdr)) {
        return false;
    }
    if (resp_hdr.type == TP_MSG_ERROR) {
        tp_error_response err;
        tp_recv_all(wc->fd, &err, sizeof(err));
        fprintf(stderr, "bitnet-tp: worker returned batch error code %u for layer %u\n",
                err.code, layer_idx);
        return false;
    }
    if (resp_hdr.type != TP_MSG_BATCH_RESPONSE) {
        return false;
    }
    tp_batch_response_header resp;
    if (!tp_recv_all(wc->fd, &resp, sizeof(resp))) {
        return false;
    }
    std::vector<tp_batch_result_item> result_items_wire(resp.n_items);
    if (!tp_recv_all(wc->fd, result_items_wire.data(), result_items_wire.size() * sizeof(tp_batch_result_item))) {
        return false;
    }
    for (auto & ri : result_items_wire) {
        std::vector<bitnet_float_type> buf(ri.n_rows);
        if (!tp_recv_all(wc->fd, buf.data(), (size_t) ri.n_rows * sizeof(bitnet_float_type))) {
            return false;
        }
        wc->result_items.emplace_back(ri.tensor_kind, std::move(buf));
    }
    return true;
}

// Called once from ggml_bitnet_init() (itself called exactly once per process,
// from ggml_init()'s is_first_call guard) -- blocking connects and thread
// creation here are a one-time startup cost, not a hot-path concern.
void tp_init(void) {
    const char * manifest_path = getenv("BITNET_TP_MANIFEST");
    if (manifest_path == nullptr) {
        return;
    }

    std::ifstream mf(manifest_path);
    GGML_ASSERT(mf.good() && "bitnet-tp: cannot open BITNET_TP_MANIFEST");
    nlohmann::json manifest;
    mf >> manifest;

    for (auto & w : manifest["workers"]) {
        auto wc = std::make_unique<tp_worker_conn>();
        wc->fd = tp_connect(w["host"].get<std::string>(), w["port"].get<int>());
        g_tp_worker_conns.push_back(std::move(wc));
    }
    const int n_workers = (int) g_tp_worker_conns.size();

    for (auto & [key, info] : manifest["tensors"].items()) {
        const size_t colon = key.find(':');
        const uint32_t layer_idx = (uint32_t) std::stoul(key.substr(0, colon));
        uint32_t tensor_kind;
        GGML_ASSERT(tp_tensor_kind_from_name(key.substr(colon + 1).c_str(), &tensor_kind));

        tp_route route;
        route.m_full = info["m"].get<int>();
        route.k = info["k"].get<int>();
        route.n_tiles = info["n_tiles"].get<int>();

        std::vector<std::vector<int>> by_worker(n_workers);
        int tile_idx = 0;
        for (auto & owner_json : info["tile_owners"]) {
            const int owner = owner_json.get<int>();
            // An owner index outside the worker list is this orchestrator's own
            // bucket -- the shard tool reserves exactly one such index when run
            // with --orchestrator-participates, and none otherwise.
            if (owner < n_workers) {
                by_worker[owner].push_back(tile_idx);
            } else {
                route.local_tiles.push_back(tile_idx);
            }
            tile_idx++;
        }
        for (int rank = 0; rank < n_workers; rank++) {
            if (!by_worker[rank].empty()) {
                route.worker_tiles.emplace_back(rank, std::move(by_worker[rank]));
            }
        }

        g_tp_routes.emplace(tp_route_key(layer_idx, tensor_kind), std::move(route));
    }

    g_tp_enabled = true;
    g_tp_debug_timing = getenv("BITNET_TP_DEBUG_TIMING") != nullptr;
    fprintf(stderr, "bitnet-tp: enabled, %d workers, %zu routed tensors\n", n_workers, g_tp_routes.size());
}

} // namespace
#endif // GGML_BITNET_TP

void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;

#if defined(GGML_BITNET_TP)
    tp_init();
#endif
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    // No upper bound on src1->ne[1] (batch/token count): ggml_bitnet_mul_mat
    // loops internally over each column (see "n_tokens" below), so an
    // arbitrary-width activation batch -- not just the usual single-token
    // decode step -- is handled correctly. This also covers the >1-token
    // capability-probe decodes llama.cpp's common_init_from_params()/
    // common_context_can_seq_rm() issue once at model-load time, which would
    // otherwise hit no fallback implementation for GGML_TYPE_TL1 and crash.
    return is_type_supported(src0->type) &&
           src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne00 = src0->ne[0];

    // QLUT needs 16 bytes per K element (see lut_ctor<K> in bitnet-lut-kernels.h),
    // plus one bitnet_float_type for the per-activation LUT scale.
    size_t wsize = ne00 * 16 * sizeof(int8_t) + sizeof(bitnet_float_type);
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int m = (int)src0->ne[1];
    const int k = (int)src0->ne[0];
    const int n_tokens = (int)src1->ne[1]; // usually 1 (single-token decode); see ggml_bitnet_can_mul_mat

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    if (ith == 0) {
        ggml_bitnet_transform_tensor(const_cast<struct ggml_tensor *>(src0));
    }
    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    struct bitnet_tensor_extra * extra = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(extra != nullptr);

    char * wdata = (char *)params->wdata;
    int8_t * qlut = (int8_t *)wdata;
    bitnet_float_type * lut_scales = (bitnet_float_type *)(qlut + (size_t)k * 16);

    // qgemm_lut_{M}_{K} only computes one BM-sized tile of the M output rows per
    // call (BM is baked into the generated kernel, e.g. 256 for M=2560) - loop
    // over all n_tile_num = M/BM tiles, offsetting the weight/output pointers,
    // and split the tiles across threads.
    const int n_tile_num = extra->n_tile_num;
    GGML_ASSERT(m % n_tile_num == 0);
    const int w_size = k * m / 4;
    const int w_tile_size = w_size / n_tile_num;
    const int c_tile_size = m / n_tile_num;

#if defined(GGML_BITNET_TP)
    // Route lookup depends only on src0 (the weight tensor), not on which
    // token column is being processed, so it's resolved once outside the
    // per-token loop below.
    tp_route * route = nullptr;
    uint32_t layer_idx = 0, tensor_kind = 0;
    if (g_tp_enabled && tp_parse_tensor_name(src0->name, &layer_idx, &tensor_kind)) {
        auto it = g_tp_routes.find(tp_route_key(layer_idx, tensor_kind));
        if (it != g_tp_routes.end()) {
            route = &it->second;
            GGML_ASSERT(route->m_full == m && route->k == k && route->n_tiles == n_tile_num);
        }
    }
#endif

    // Loop over each column of the activation batch. n_tokens is almost
    // always 1 (the normal single-token decode step); >1 only occurs for
    // llama.cpp's internal multi-token capability-probe decodes issued once
    // at model-load time (common_init_from_params's warmup,
    // common_context_can_seq_rm) -- ggml_bitnet_can_mul_mat has no upper
    // bound on ne[1], so those must be handled correctly here rather than
    // crashing into a nonexistent generic TL1 fallback.
    for (int tok = 0; tok < n_tokens; tok++) {
        void * src1_tok = (char *) src1->data + (size_t) tok * src1->nb[1];
        bitnet_float_type * dst_tok = (bitnet_float_type *)((char *) dst->data + (size_t) tok * dst->nb[1]);

#if defined(GGML_BITNET_TP)
        if (route != nullptr) {
            if (ith == 0) {
                auto cache_it = g_tp_prefetch_cache.find(tp_prefetch_key(tok, layer_idx, tensor_kind));
                if (cache_it != g_tp_prefetch_cache.end()) {
                    GGML_ASSERT((int) cache_it->second.size() == m);
                    memcpy(dst_tok, cache_it->second.data(), (size_t) m * sizeof(bitnet_float_type));
                    g_tp_prefetch_cache.erase(cache_it);
                    g_tp_scratch_consumed_from_cache = true;
                } else {
                    g_tp_scratch_consumed_from_cache = false;
                    g_tp_scratch_seq_id = ++g_tp_seq;
                }
            }
            // Broadcasts g_tp_scratch_consumed_from_cache (and, on a miss,
            // g_tp_scratch_seq_id) to the other threads below, which need
            // both before joining the fan-out send loop.
            if (nth > 1) {
                ggml_barrier(params->threadpool);
            }
            const bool consumed_from_cache = g_tp_scratch_consumed_from_cache;

            if (!consumed_from_cache) {
                // Q/K/V (and separately gate/up) share one activation and
                // have no data dependency on each other, but ggml calls
                // their mul_mat ops one at a time -- so when the first
                // member of a group is called, batch-dispatch the WHOLE
                // group now (one combined request per worker, see
                // tp_worker_conn's comment for why combining rather than
                // just overlapping separate requests matters) instead of
                // waiting for ggml to call the others later. Non-grouped
                // tensors (attn_output, ffn_down) are just a batch of one.
                //
                // This routing/grouping computation is pure and deterministic
                // given (layer_idx, tensor_kind, the read-only g_tp_routes
                // table) -- every thread below redoes it independently and
                // gets identical results, with no locking needed, so each
                // can index into its own copy of union_ranks when joining
                // the fan-out loop further down.
                const std::vector<uint32_t> * members = tp_group_members(tensor_kind);
                tp_route * group_routes[3] = {route, nullptr, nullptr};
                uint32_t group_kinds[3] = {tensor_kind, 0, 0};
                int n_group = 1;
                if (members != nullptr) {
                    n_group = 0;
                    for (uint32_t kd : *members) {
                        auto rit = g_tp_routes.find(tp_route_key(layer_idx, kd));
                        if (rit == g_tp_routes.end()) {
                            continue;
                        }
                        group_routes[n_group] = &rit->second;
                        group_kinds[n_group] = kd;
                        n_group++;
                    }
                }
                // This dispatch path only gathers each item's *network*
                // tiles into full_result below -- it doesn't also run the
                // orchestrator's own local-tile loop the way the plain
                // (non-grouped) path used to. That's fine for the
                // pure-orchestrator config this was built and measured
                // against (every route's local_tiles is empty there), but
                // would silently drop output for a config where the
                // orchestrator also holds a shard (--orchestrator-participates).
                // Fail loudly instead of computing wrong results if that
                // config is ever combined with this optimization.
                for (int g = 0; g < n_group; g++) {
                    GGML_ASSERT(group_routes[g]->local_tiles.empty() &&
                                 "bitnet-tp: group-batch dispatch doesn't support "
                                 "orchestrator-local tiles (--orchestrator-participates) yet");
                }

                // Build the union of workers touched by any group member,
                // and for each one, the sub-list of (kind, m_full) items it
                // actually owns tiles for -- one combined request per
                // worker, not one request per (worker, group member) pair.
                std::vector<int> union_ranks;
                std::vector<std::vector<int>> rank_members(g_tp_worker_conns.size()); // rank -> group indices
                for (int g = 0; g < n_group; g++) {
                    for (auto & [rank, tiles] : group_routes[g]->worker_tiles) {
                        (void) tiles;
                        if (rank_members[rank].empty()) {
                            union_ranks.push_back(rank);
                        }
                        rank_members[rank].push_back(g);
                    }
                }

                // Fan-out, parallelized across the orchestrator's own
                // compute threads instead of one thread looping through
                // every worker's socket: thread ith sends to
                // union_ranks[ith], union_ranks[ith+nth], ... -- a disjoint,
                // complete partition of union_ranks since every thread
                // computed the identical vector above. This was previously
                // one thread issuing union_ranks.size() sequential blocking
                // send() calls; avg_fanout scaled up close to linearly with
                // worker count as a result (measured ~0.18ms at 2 workers to
                // ~0.37ms at 5), consistent with that serial cost.
                const auto fanout_t0 = (ith == 0 && g_tp_debug_timing) ? std::chrono::steady_clock::now()
                                                                        : std::chrono::steady_clock::time_point{};
                const uint32_t seq_id = g_tp_scratch_seq_id;
                for (size_t i = (size_t) ith; i < union_ranks.size(); i += (size_t) nth) {
                    const int rank = union_ranks[i];
                    tp_worker_conn * wc = g_tp_worker_conns[rank].get();
                    std::vector<std::pair<uint32_t, uint32_t>> items;
                    for (int g : rank_members[rank]) {
                        items.emplace_back(group_kinds[g], (uint32_t) group_routes[g]->m_full);
                    }
                    const bool ok = tp_send_batch_request(wc, seq_id, layer_idx, k, items, (const float *) src1_tok);
                    GGML_ASSERT(ok && "bitnet-tp: failed to send batch request to worker");
                }
                // Every worker's request must actually be on the wire before
                // anyone starts polling for responses below.
                if (nth > 1) {
                    ggml_barrier(params->threadpool);
                }
                if (ith == 0 && g_tp_debug_timing) {
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - fanout_t0).count();
                    g_tp_fanout_ns.fetch_add((uint64_t) ns);
                }

                if (ith == 0) {
                    const auto gather_t0 = g_tp_debug_timing ? std::chrono::steady_clock::now()
                                                              : std::chrono::steady_clock::time_point{};
                    std::vector<bitnet_float_type> full_results[3];
                    for (int g = 0; g < n_group; g++) {
                        full_results[g].assign((size_t) group_routes[g]->m_full, (bitnet_float_type) 0);
                    }

                    // Every request above has already been sent to every
                    // worker in union_ranks -- now poll all of their sockets
                    // together and consume whichever responds first, instead
                    // of handing each one off to its own handler thread and
                    // waiting on a condvar. This is exactly what
                    // tmp_batch_probe.py did to measure this protocol's
                    // network-only floor; doing the same I/O directly in
                    // this thread removes the two condvar/futex hops a
                    // separate handler thread would have added on top.
                    // (Left single-threaded, unlike the fan-out above:
                    // poll() already multiplexes every worker's socket
                    // concurrently from one thread, so splitting this loop
                    // across threads too wouldn't shorten the wait on the
                    // slowest worker -- there's no equivalent of the
                    // sequential-send cost to parallelize away here.)
                    std::vector<int> pending_ranks = union_ranks;
                    std::vector<struct pollfd> pfds;
                    pfds.reserve(pending_ranks.size());
                    for (int rank : pending_ranks) {
                        pfds.push_back({g_tp_worker_conns[rank]->fd, POLLIN, 0});
                    }
                    while (!pending_ranks.empty()) {
                        // TODO: no timeout/liveness handling -- a dead or slow
                        // worker hangs here indefinitely; see the plan's "known
                        // risks" section.
                        const int n_ready = poll(pfds.data(), (nfds_t) pfds.size(), -1);
                        GGML_ASSERT(n_ready > 0 && "bitnet-tp: poll() failed waiting for worker responses");
                        for (size_t i = 0; i < pfds.size();) {
                            if (!(pfds[i].revents & POLLIN)) {
                                i++;
                                continue;
                            }
                            const int rank = pending_ranks[i];
                            tp_worker_conn * wc = g_tp_worker_conns[rank].get();
                            const bool ok = tp_recv_batch_response(wc, layer_idx);
                            GGML_ASSERT(ok && "bitnet-tp: worker batch request failed");
                            for (auto & [kind, buf] : wc->result_items) {
                                int g_idx = -1;
                                for (int g = 0; g < n_group; g++) {
                                    if (group_kinds[g] == kind) { g_idx = g; break; }
                                }
                                GGML_ASSERT(g_idx >= 0 && "bitnet-tp: worker returned an unexpected tensor kind");
                                tp_route * r = group_routes[g_idx];
                                const std::vector<int> * my_tiles = nullptr;
                                for (auto & [r2, tiles2] : r->worker_tiles) {
                                    if (r2 == rank) { my_tiles = &tiles2; break; }
                                }
                                GGML_ASSERT(my_tiles != nullptr);
                                const int item_c_tile_size = r->m_full / r->n_tiles;
                                GGML_ASSERT(buf.size() == my_tiles->size() * (size_t) item_c_tile_size);
                                for (size_t ti = 0; ti < my_tiles->size(); ti++) {
                                    const int dst_offset = (*my_tiles)[ti] * item_c_tile_size;
                                    memcpy(full_results[g_idx].data() + dst_offset, buf.data() + ti * item_c_tile_size,
                                           item_c_tile_size * sizeof(bitnet_float_type));
                                }
                            }
                            // Swap-remove: order among pending entries doesn't matter.
                            pfds[i] = pfds.back();
                            pfds.pop_back();
                            pending_ranks[i] = pending_ranks.back();
                            pending_ranks.pop_back();
                        }
                    }
                    for (int g = 0; g < n_group; g++) {
                        if (group_kinds[g] == tensor_kind) {
                            memcpy(dst_tok, full_results[g].data(),
                                   (size_t) group_routes[g]->m_full * sizeof(bitnet_float_type));
                        } else {
                            g_tp_prefetch_cache[tp_prefetch_key(tok, layer_idx, group_kinds[g])] =
                                std::move(full_results[g]);
                        }
                    }
                    if (g_tp_debug_timing) {
                        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - gather_t0).count();
                        g_tp_gather_ns.fetch_add((uint64_t) ns);
                        const uint64_t calls = g_tp_call_count.fetch_add(1) + 1;
                        if (calls <= 16) {
                            fprintf(stderr, "bitnet-tp-timing-detail: call=%llu trigger_kind=%s n_group=%d gather=%.3fms\n",
                                    (unsigned long long) calls, tp_tensor_kind_name(tensor_kind), n_group,
                                    (double) ns / 1e6);
                        }
                        if (calls % 210 == 0) { // ~once per generated token (7 tensors x 30 layers)
                            fprintf(stderr,
                                    "bitnet-tp-timing: calls=%llu avg_fanout=%.3fms avg_gather=%.3fms\n",
                                    (unsigned long long) calls,
                                    (double) g_tp_fanout_ns.load() / calls / 1e6,
                                    (double) g_tp_gather_ns.load() / calls / 1e6);
                        }
                    }
                }
            }
            if (nth > 1) {
                ggml_barrier(params->threadpool);
            }
            continue; // next token
        }
#endif

        if (ith == 0) {
            ggml_preprocessor(m, k, src1_tok, lut_scales, qlut);
        }

        // barrier
        if (nth > 1) {
            ggml_barrier(params->threadpool);
        }

        const int th_tile_num = (n_tile_num + nth - 1) / nth;
        const int th_tile_beg = ith * th_tile_num;
        const int th_tile_end = std::min((ith + 1) * th_tile_num, n_tile_num);

        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset = i_tile * w_tile_size;
            const int dst_offset = i_tile * c_tile_size;
            ggml_qgemm_lut(m, k, extra->qweights + w_offset, qlut, extra->scales, lut_scales,
                            dst_tok + dst_offset);
        }

        // Barrier before the next token's iteration (if any) starts rebuilding
        // the shared qlut/lut_scales buffer -- without this, a thread still
        // reading qlut for this token's tiles races against thread 0 already
        // overwriting it for tok+1. Harmless (just one extra sync) on the
        // last iteration / the common n_tokens==1 case.
        if (nth > 1) {
            ggml_barrier(params->threadpool);
        }
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];

    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

// bitnet_float_type is `float` on x86 (see ggml-bitnet.h), so unlike the ARM_TL1
// path there is no fp16 scratch buffer / conversion step needed here: activations
// and outputs are used in place.
void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ne00 = (int)src0->ne[0];
    const int ne01 = (int)src0->ne[1];
    const int ne10 = (int)src1->ne[0];
    const int ne11 = (int)src1->ne[1];
    const int ne0  = (int)dst->ne[0];

    const int ith = params->ith;
    const int nth = params->nth;

    struct bitnet_tensor_extra * wt = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(wt != nullptr);

    char * wdata = (char *)params->wdata;

    const int total_k = ne10;
    const int three_k = (total_k / wt->BK) * wt->BK;
    const int two_k   = total_k - three_k;

    int8_t * three_qlut = (int8_t *)wdata;
    bitnet_float_type * lut_scales = (bitnet_float_type *)(three_qlut + three_k / 3 * 16 * 2 * ne11);
    int8_t * two_qlut = (int8_t *)(lut_scales + ne11);

    if (ith == 0) {
        // Although we have done this in file `llama.cpp`,
        // we still need to do it here for non-model inference, e.g., test-backend-ops.cpp.
        ggml_bitnet_transform_tensor(const_cast<struct ggml_tensor *>(src0));
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        ggml_preprocessor(ne11, ne01, three_k, two_k, src1->data, lut_scales, three_qlut, two_qlut);
    }

    ggml_barrier(params->threadpool);

    float * act_output = (float *)dst->data;

    const int n_tile_num = wt->n_tile_num;
    GGML_ASSERT(ne0 % n_tile_num == 0);
    const int w_size        = three_k * ne01 / (2 * 3);
    const int w_tile_size   = w_size / n_tile_num;
    const int c_size        = ne01;
    const int c_tile_size   = c_size / n_tile_num;
    const int sign_size     = three_k * ne01 / 24;
    const int sign_tile_size = sign_size / n_tile_num;

    const int th_tile_num = (n_tile_num + nth - 1) / nth;
    const int th_tile_beg = ith * th_tile_num;
    const int th_tile_end = std::min((ith + 1) * th_tile_num, n_tile_num);

    uint8_t * sign = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2;
    const int two_w_size      = ne01 * two_k / (2 * 2);
    const int two_w_tile_size = two_w_size / n_tile_num;
    // The "two" region starts right after the sign region (weight_bytes + sign_bytes),
    // i.e. 5*sign_size, NOT three_k*ne01/4 -- that matched an older packer layout that
    // reserved extra alignment padding this one (preprocess_weights_tl2) doesn't add.
    uint8_t * two_A = sign + sign_size;

    // The generated qgemm_lut kernels write batch row b's tile output at
    // C[b*c_tile_size + i] (tile-local, BM-strided), but dst->data needs row
    // (done+b) at act_output[(done+b)*ne01 + tile_col_offset + i] (ne01-strided,
    // since c_tile_size == ne01 only when n_tile_num == 1). Route through a
    // per-tile scratch buffer and scatter it into place with the right stride.
    std::vector<float> tile_scratch((size_t)512 * c_tile_size);

    auto do_bs_batch = [&](int bs, int qlut_off_elems_three, int qlut_off_elems_two,
                           int lut_scales_off, int done) {
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset       = i_tile * w_tile_size;
            const int sign_offset    = i_tile * sign_tile_size;
            const int two_w_offset   = i_tile * two_w_tile_size;
            const int tile_col_offset = i_tile * c_tile_size;

            ggml_qgemm_lut(bs, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                           sign + sign_offset,
                           three_qlut + qlut_off_elems_three,
                           wt->scales,
                           lut_scales + lut_scales_off,
                           tile_scratch.data());
            ggml_qgemm_lut(bs, ne01, ne00, two_k, two_A + two_w_offset,
                           NULL,
                           two_qlut + qlut_off_elems_two,
                           wt->scales,
                           lut_scales + lut_scales_off,
                           tile_scratch.data());

            for (int b = 0; b < bs; b++) {
                memcpy(act_output + (size_t)(done + b) * ne01 + tile_col_offset,
                       tile_scratch.data() + (size_t)b * c_tile_size,
                       c_tile_size * sizeof(float));
            }
        }
    };

    if (ne11 > 1) {
        int iter = ne11;
        const int bs512_num = iter / 512; iter -= 512 * bs512_num;
        const int bs256_num = iter / 256; iter -= 256 * bs256_num;
        const int bs128_num = iter / 128; iter -= 128 * bs128_num;
        const int bs32_num  = iter / 32;  iter -= 32  * bs32_num;
        const int bs8_num   = iter / 8;   iter -= 8   * bs8_num;
        const int bs1_num   = iter / 1;

        int done = 0; // rows of ne11 processed so far
        for (int i = 0; i < bs512_num; i++, done += 512) {
            do_bs_batch(512, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs256_num; i++, done += 256) {
            do_bs_batch(256, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs128_num; i++, done += 128) {
            do_bs_batch(128, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs32_num; i++, done += 32) {
            do_bs_batch(32, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs8_num; i++, done += 8) {
            do_bs_batch(8, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs1_num; i++, done += 1) {
            do_bs_batch(1, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
    } else {
        do_bs_batch(1, 0, 0, 0, 0);
    }
}

#endif
