// bitnet-tp-worker: a standalone process that holds a row-tile shard of some of
// a BitNet TL1 model's ternary weight tensors, and answers "quantize this
// activation, run my tiles, give me the result slice" requests over TCP.
//
// Deliberately minimal: no ggml_cgraph, no bitnet_tensor_extra, no tokenizer --
// just the manifest, one mmap'd shard GGUF, and the already-exported TL1 kernel
// calls (ggml_preprocessor/ggml_qgemm_lut) that do everything needed per tile.
// See ggml_bitnet_mul_mat() in src/ggml-bitnet-lut.cpp for the reference
// single-device tile loop this mirrors.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-bitnet.h"
#include "gguf.h"
#include "json.hpp"
#include "tp_proto.h"

using json = nlohmann::json;

namespace {

struct local_tensor {
    int m_full = 0;
    int k = 0;
    int bm = 0;
    int n_local_tiles = 0;
    const uint8_t * blob = nullptr; // points into the mmap'd shard file, never freed
};

std::unordered_map<uint64_t, local_tensor> g_tensors;

uint64_t tensor_key(uint32_t layer_idx, uint32_t tensor_kind) {
    return (uint64_t(layer_idx) << 8) | tensor_kind;
}

// Compute-side thread pool: handle_connection used to run its whole tile loop
// on a single thread, leaving 3 of the CM4's 4 cores idle during the exact
// work this whole project exists to parallelize. This mirrors the
// orchestrator's own persistent-handler-thread pattern in
// ggml_bitnet_mul_mat (src/ggml-bitnet-lut.cpp) but for local compute instead
// of network dispatch. There is exactly one persistent connection per
// worker (the orchestrator opens it once at startup and never reconnects),
// so a single shared pool for the whole process is safe -- no concurrent
// requests ever compete for it.
// One ggml_qgemm_lut call's worth of work: a single BM-row tile of some
// tensor in the current request (which tensor it belongs to doesn't matter
// here -- qlut/lut_scale/dummy_scale are shared across every job in a batch,
// see tp_run_tile_jobs). Flattening a whole batch request's tiles -- across
// however many tensor-kind items it combines -- into one list of these lets
// one dispatch split ALL of them across the compute pool in a single pass,
// instead of one pass per item.
struct resolved_tile_job {
    int m_full = 0, k = 0;
    const uint8_t * blob = nullptr;
    bitnet_float_type * out = nullptr;
};

struct tp_compute_helper {
    std::thread thread;
    int ith = 0; // fixed at spawn time: 1..g_num_threads-1 (0 is handle_connection's own thread)
    std::mutex mu;
    std::condition_variable cv_go;
    std::condition_variable cv_done;
    bool has_job = false;
    bool done = false;
    bool shutdown = false;

    // Job fields, set by handle_connection under `mu` before notifying
    // cv_go; read by this helper once woken.
    const resolved_tile_job * jobs = nullptr;
    int n_jobs = 0;
    int n_active = 0; // participants this round (<= g_num_threads)
    int8_t * qlut = nullptr;
    bitnet_float_type * lut_scale = nullptr;
    bitnet_float_type * dummy_scale = nullptr;
};

int g_num_threads = 4; // matches the CM4's core count and the orchestrator's own -t 4
std::vector<std::unique_ptr<tp_compute_helper>> g_helpers; // g_num_threads - 1 entries

// The orchestrator may now hold more than one connection open to this
// worker at once (one per position in a dispatch group, e.g. Q/K/V), so
// more than one handle_connection thread can be live simultaneously. The
// compute pool above is a single shared set of g_num_threads-1 helper
// threads sized for the CM4's actual core count -- it was never meant to be
// used by two requests at once (there's nowhere for a second 4-way split to
// run without thrashing the same 4 cores), so every use of it is guarded by
// this mutex. recv()/send() on each connection's own socket still happen
// outside this lock and so still overlap across connections -- only the
// actual GEMM compute is serialized, which matches reality: there are only
// 4 cores no matter how many requests arrive concurrently.
std::mutex g_compute_pool_mutex;

// Same ceil-split formula used throughout this codebase (e.g.
// ggml_bitnet_mul_mat's th_tile_num) so a helper's tile range is computed
// identically regardless of which side of the network it runs on.
void tp_compute_tile_range(int ith, int n_active, int n_local_tiles, int * beg, int * end) {
    const int th_tile_num = (n_local_tiles + n_active - 1) / n_active;
    *beg = ith * th_tile_num;
    *end = std::min((ith + 1) * th_tile_num, n_local_tiles);
}

void tp_compute_helper_loop(tp_compute_helper * h) {
    for (;;) {
        std::unique_lock<std::mutex> lock(h->mu);
        h->cv_go.wait(lock, [h] { return h->has_job || h->shutdown; });
        if (h->shutdown) {
            return;
        }
        h->has_job = false;
        const int ith = h->ith;
        const int n_active = h->n_active;
        const int n_jobs = h->n_jobs;
        const resolved_tile_job * jobs = h->jobs;
        int8_t * qlut = h->qlut;
        bitnet_float_type * lut_scale = h->lut_scale;
        bitnet_float_type * dummy_scale = h->dummy_scale;
        lock.unlock();

        int beg, end;
        tp_compute_tile_range(ith, n_active, n_jobs, &beg, &end);
        for (int i = beg; i < end; i++) {
            const resolved_tile_job & j = jobs[i];
            ggml_qgemm_lut(j.m_full, j.k, (void *) j.blob, qlut, dummy_scale, lut_scale, j.out);
        }

        lock.lock();
        h->done = true;
        lock.unlock();
        h->cv_done.notify_one();
    }
}

// Splits `jobs` (already-flattened, possibly spanning several tensor-kind
// items sharing one activation/LUT -- see TP_MSG_BATCH_REQUEST below) across
// the compute pool in one pass and blocks until every job is done. Guarded
// by g_compute_pool_mutex: see its comment -- only one connection's request
// may use the shared helper pool at a time.
void tp_run_tile_jobs(const std::vector<resolved_tile_job> & jobs, int8_t * qlut,
                       bitnet_float_type * lut_scale, bitnet_float_type * dummy_scale) {
    std::lock_guard<std::mutex> pool_lock(g_compute_pool_mutex);
    const int n_jobs = (int) jobs.size();
    const int n_active = std::min(n_jobs, g_num_threads);
    for (int ith = 1; ith < n_active; ith++) {
        tp_compute_helper * h = g_helpers[ith - 1].get();
        {
            std::lock_guard<std::mutex> lock(h->mu);
            h->jobs = jobs.data();
            h->n_jobs = n_jobs;
            h->n_active = n_active;
            h->qlut = qlut;
            h->lut_scale = lut_scale;
            h->dummy_scale = dummy_scale;
            h->has_job = true;
        }
        h->cv_go.notify_one();
    }

    // This thread's own share (ith=0).
    {
        int beg, end;
        tp_compute_tile_range(0, n_active, n_jobs, &beg, &end);
        for (int i = beg; i < end; i++) {
            const resolved_tile_job & j = jobs[i];
            ggml_qgemm_lut(j.m_full, j.k, (void *) j.blob, qlut, dummy_scale, lut_scale, j.out);
        }
    }

    for (int ith = 1; ith < n_active; ith++) {
        tp_compute_helper * h = g_helpers[ith - 1].get();
        std::unique_lock<std::mutex> lock(h->mu);
        h->cv_done.wait(lock, [h] { return h->done; });
        h->done = false;
    }
}

// Kept mapped for the whole process lifetime -- every local_tensor::blob
// points into this region.
void * g_shard_map_addr = nullptr;
size_t g_shard_map_size = 0;

void load_shard(const std::string & shard_path, const std::string & manifest_path, int my_rank) {
    std::ifstream mf(manifest_path);
    if (!mf) {
        fprintf(stderr, "cannot open manifest %s\n", manifest_path.c_str());
        exit(1);
    }
    json manifest;
    mf >> manifest;

    int fd = open(shard_path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "cannot open shard file %s: %s\n", shard_path.c_str(), strerror(errno));
        exit(1);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "fstat failed for %s: %s\n", shard_path.c_str(), strerror(errno));
        exit(1);
    }
    void * addr = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s: %s\n", shard_path.c_str(), strerror(errno));
        exit(1);
    }
    g_shard_map_addr = addr;
    g_shard_map_size = (size_t) st.st_size;

    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context * gctx = gguf_init_from_file(shard_path.c_str(), params);
    if (!gctx) {
        fprintf(stderr, "failed to parse shard GGUF %s\n", shard_path.c_str());
        exit(1);
    }
    const size_t data_offset = gguf_get_data_offset(gctx);
    const uint8_t * base = (const uint8_t *) addr;

    size_t n_loaded = 0;
    for (auto & [key, info] : manifest["tensors"].items()) {
        // key is "{layer_idx}:{kind}"
        size_t colon = key.find(':');
        uint32_t layer_idx = (uint32_t) std::stoul(key.substr(0, colon));
        std::string kind_name = key.substr(colon + 1);
        uint32_t tensor_kind;
        if (!tp_tensor_kind_from_name(kind_name.c_str(), &tensor_kind)) {
            continue;
        }

        int n_local = 0;
        for (auto & owner : info["tile_owners"]) {
            if (owner.get<int>() == my_rank) {
                n_local++;
            }
        }
        if (n_local == 0) {
            continue;
        }

        std::string shard_tensor_name = "blk." + std::to_string(layer_idx) + "." + kind_name + ".shard";
        int64_t tid = gguf_find_tensor(gctx, shard_tensor_name.c_str());
        if (tid < 0) {
            fprintf(stderr, "manifest expects tensor %s on rank %d but it's not in %s\n",
                    shard_tensor_name.c_str(), my_rank, shard_path.c_str());
            exit(1);
        }

        local_tensor lt;
        lt.m_full = info["m"].get<int>();
        lt.k = info["k"].get<int>();
        lt.bm = info["bm"].get<int>();
        lt.n_local_tiles = n_local;
        lt.blob = base + data_offset + gguf_get_tensor_offset(gctx, tid);
        g_tensors[tensor_key(layer_idx, tensor_kind)] = lt;
        n_loaded++;
    }

    gguf_free(gctx);
    fprintf(stderr, "rank %d: loaded %zu tensor shards from %s\n", my_rank, n_loaded, shard_path.c_str());
}

void handle_connection(int client_fd) {
    std::vector<int8_t> qlut_buf;
    std::vector<float> activation_buf;
    std::vector<bitnet_float_type> out_buf;
    std::vector<uint8_t> send_buf; // reused scratch for the batched response below
    std::vector<tp_batch_item> items_buf;
    std::vector<resolved_tile_job> jobs_buf;

    for (;;) {
        tp_msg_header hdr;
        if (!tp_recv_header(client_fd, &hdr)) {
            break;
        }

        if (hdr.type == TP_MSG_PING) {
            tp_send_msg(client_fd, TP_MSG_PONG, nullptr, 0);
            continue;
        }

        if (hdr.type == TP_MSG_MATMUL_REQUEST) {
            if (hdr.payload_len < sizeof(tp_matmul_request)) {
                break;
            }
            tp_matmul_request req;
            if (!tp_recv_all(client_fd, &req, sizeof(req))) {
                break;
            }
            const uint32_t act_bytes = hdr.payload_len - sizeof(req);
            if (act_bytes != req.k * sizeof(float)) {
                break;
            }
            activation_buf.resize(req.k);
            if (!tp_recv_all(client_fd, activation_buf.data(), act_bytes)) {
                break;
            }

            auto it = g_tensors.find(tensor_key(req.layer_idx, req.tensor_kind));
            if (it == g_tensors.end()) {
                tp_error_response err{req.seq_id, 1};
                tp_send_msg(client_fd, TP_MSG_ERROR, &err, sizeof(err));
                continue;
            }
            const local_tensor & t = it->second;
            if ((int) req.m_full != t.m_full || (int) req.k != t.k) {
                tp_error_response err{req.seq_id, 2};
                tp_send_msg(client_fd, TP_MSG_ERROR, &err, sizeof(err));
                continue;
            }

            // Mirrors ggml_bitnet_mul_mat_get_wsize's sizing: 16 bytes of QLUT
            // per K element plus one lut-scale.
            qlut_buf.resize((size_t) t.k * 16);
            bitnet_float_type lut_scale;
            ggml_preprocessor(t.m_full, t.k, activation_buf.data(), &lut_scale, qlut_buf.data());

            out_buf.resize((size_t) t.bm * t.n_local_tiles);
            // Matches bitnet_tensor_extra::scales[0]'s hardcoded identity on
            // the orchestrator -- the real per-tensor dequant scale is
            // applied there, post-gather, via build_lora_mm's ggml_mul. Using
            // the same constant here keeps shard output bit-exact with the
            // single-device path.
            bitnet_float_type dummy_scale = 1.0f;

            const size_t tile_bytes = (size_t) t.k * t.bm / 4;
            jobs_buf.clear();
            jobs_buf.reserve(t.n_local_tiles);
            for (int i = 0; i < t.n_local_tiles; i++) {
                jobs_buf.push_back({t.m_full, t.k, t.blob + (size_t) i * tile_bytes,
                                     out_buf.data() + (size_t) i * t.bm});
            }
            tp_run_tile_jobs(jobs_buf, qlut_buf.data(), &lut_scale, &dummy_scale);

            tp_matmul_response resp{req.seq_id, req.layer_idx, req.tensor_kind, (uint32_t) out_buf.size()};
            const size_t result_bytes = out_buf.size() * sizeof(bitnet_float_type);
            tp_msg_header out_hdr{TP_PROTO_MAGIC, TP_PROTO_VERSION, TP_MSG_MATMUL_RESPONSE,
                                   (uint32_t) (sizeof(resp) + result_bytes)};

            // One send() for the whole response instead of three -- each send
            // is a separate syscall and, even with TCP_NODELAY, a separate TCP
            // segment; batching cuts both per reply.
            send_buf.resize(sizeof(out_hdr) + sizeof(resp) + result_bytes);
            memcpy(send_buf.data(), &out_hdr, sizeof(out_hdr));
            memcpy(send_buf.data() + sizeof(out_hdr), &resp, sizeof(resp));
            memcpy(send_buf.data() + sizeof(out_hdr) + sizeof(resp), out_buf.data(), result_bytes);
            if (!tp_send_all(client_fd, send_buf.data(), send_buf.size())) break;
            continue;
        }

        if (hdr.type == TP_MSG_BATCH_REQUEST) {
            if (hdr.payload_len < sizeof(tp_batch_request_header)) {
                break;
            }
            tp_batch_request_header bhdr;
            if (!tp_recv_all(client_fd, &bhdr, sizeof(bhdr))) {
                break;
            }
            items_buf.resize(bhdr.n_items);
            const size_t items_bytes = items_buf.size() * sizeof(tp_batch_item);
            if (!tp_recv_all(client_fd, items_buf.data(), items_bytes)) {
                break;
            }
            if (hdr.payload_len < sizeof(bhdr) + items_bytes) {
                break;
            }
            const uint32_t act_bytes = hdr.payload_len - sizeof(bhdr) - (uint32_t) items_bytes;
            if (act_bytes != bhdr.k * sizeof(float)) {
                break;
            }
            activation_buf.resize(bhdr.k);
            if (!tp_recv_all(client_fd, activation_buf.data(), act_bytes)) {
                break;
            }

            // Resolve every item up front so a bad item errors out before any
            // compute runs (mirrors the single-item path's check-then-compute
            // order).
            std::vector<const local_tensor *> resolved(items_buf.size());
            uint32_t err_code = 0;
            for (size_t i = 0; i < items_buf.size() && err_code == 0; i++) {
                auto it = g_tensors.find(tensor_key(bhdr.layer_idx, items_buf[i].tensor_kind));
                if (it == g_tensors.end()) {
                    err_code = 1;
                    break;
                }
                const local_tensor & t = it->second;
                if ((int) items_buf[i].m_full != t.m_full || (int) bhdr.k != t.k) {
                    err_code = 2;
                    break;
                }
                resolved[i] = &t;
            }
            if (err_code != 0) {
                tp_error_response err{bhdr.seq_id, err_code};
                tp_send_msg(client_fd, TP_MSG_ERROR, &err, sizeof(err));
                continue;
            }

            // Shared LUT: ggml_preprocessor's dispatch only varies by k (every
            // m sharing a k resolves to the same preprocessor_k<K> template --
            // see bitnet-lut-kernels.h), so one call serves the whole batch
            // instead of once per item. This is the actual point of batching:
            // the worker gets to split ALL items' tiles across its cores in
            // one pass below instead of one pass (and one compute-pool-mutex
            // acquisition) per item.
            qlut_buf.resize((size_t) bhdr.k * 16);
            bitnet_float_type lut_scale;
            ggml_preprocessor(resolved[0]->m_full, bhdr.k, activation_buf.data(), &lut_scale, qlut_buf.data());
            bitnet_float_type dummy_scale = 1.0f;

            size_t total_rows = 0, total_tiles = 0;
            for (auto * t : resolved) {
                total_rows += (size_t) t->bm * t->n_local_tiles;
                total_tiles += (size_t) t->n_local_tiles;
            }
            out_buf.resize(total_rows);

            jobs_buf.clear();
            jobs_buf.reserve(total_tiles);
            size_t running_offset = 0;
            for (size_t i = 0; i < resolved.size(); i++) {
                const local_tensor & t = *resolved[i];
                const size_t tile_bytes = (size_t) t.k * t.bm / 4;
                for (int ti = 0; ti < t.n_local_tiles; ti++) {
                    jobs_buf.push_back({t.m_full, t.k, t.blob + (size_t) ti * tile_bytes,
                                         out_buf.data() + running_offset + (size_t) ti * t.bm});
                }
                running_offset += (size_t) t.bm * t.n_local_tiles;
            }
            tp_run_tile_jobs(jobs_buf, qlut_buf.data(), &lut_scale, &dummy_scale);

            std::vector<tp_batch_result_item> result_items(resolved.size());
            for (size_t i = 0; i < resolved.size(); i++) {
                result_items[i] = {items_buf[i].tensor_kind, (uint32_t) (resolved[i]->bm * resolved[i]->n_local_tiles)};
            }
            const size_t result_items_bytes = result_items.size() * sizeof(tp_batch_result_item);
            const size_t result_bytes = out_buf.size() * sizeof(bitnet_float_type);
            tp_batch_response_header out_bhdr{bhdr.seq_id, bhdr.layer_idx, (uint32_t) resolved.size()};
            tp_msg_header out_hdr{TP_PROTO_MAGIC, TP_PROTO_VERSION, TP_MSG_BATCH_RESPONSE,
                                   (uint32_t) (sizeof(out_bhdr) + result_items_bytes + result_bytes)};

            send_buf.resize(sizeof(out_hdr) + sizeof(out_bhdr) + result_items_bytes + result_bytes);
            uint8_t * p = send_buf.data();
            memcpy(p, &out_hdr, sizeof(out_hdr)); p += sizeof(out_hdr);
            memcpy(p, &out_bhdr, sizeof(out_bhdr)); p += sizeof(out_bhdr);
            memcpy(p, result_items.data(), result_items_bytes); p += result_items_bytes;
            memcpy(p, out_buf.data(), result_bytes);
            if (!tp_send_all(client_fd, send_buf.data(), send_buf.size())) break;
            continue;
        }

        break; // unknown message type
    }
    close(client_fd);
}

} // namespace

int main(int argc, char ** argv) {
    std::string shard_path, manifest_path, bind_addr = "0.0.0.0";
    int rank = -1;
    int port = 7420;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (arg == "--shard") shard_path = next();
        else if (arg == "--manifest") manifest_path = next();
        else if (arg == "--rank") rank = std::stoi(next());
        else if (arg == "--bind") bind_addr = next();
        else if (arg == "--port") port = std::stoi(next());
        else if (arg == "--threads") g_num_threads = std::stoi(next());
        else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (shard_path.empty() || manifest_path.empty() || rank < 0) {
        fprintf(stderr,
                "usage: %s --shard <shard.gguf> --manifest <tp_manifest.json> --rank <N> "
                "[--bind <addr>] [--port <port>] [--threads <n>]\n",
                argv[0]);
        return 1;
    }
    if (g_num_threads < 1) {
        fprintf(stderr, "--threads must be >= 1\n");
        return 1;
    }

    load_shard(shard_path, manifest_path, rank);

    // Spawn the compute helper pool (g_num_threads - 1 threads; this
    // connection-handling thread itself is ith=0) once, before accept()
    // starts -- see tp_compute_helper's comment for why one shared pool for
    // the whole process is safe.
    for (int ith = 1; ith < g_num_threads; ith++) {
        auto h = std::make_unique<tp_compute_helper>();
        h->ith = ith;
        tp_compute_helper * h_ptr = h.get();
        h->thread = std::thread(tp_compute_helper_loop, h_ptr);
        g_helpers.push_back(std::move(h));
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "socket() failed: %s\n", strerror(errno));
        return 1;
    }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "invalid --bind address: %s\n", bind_addr.c_str());
        return 1;
    }
    if (bind(listen_fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        fprintf(stderr, "bind failed on %s:%d: %s\n", bind_addr.c_str(), port, strerror(errno));
        return 1;
    }
    if (listen(listen_fd, 16) != 0) {
        fprintf(stderr, "listen failed: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "rank %d: listening on %s:%d\n", rank, bind_addr.c_str(), port);

    for (;;) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            continue;
        }
        // Small-message request/response protocol -- without this, Nagle's
        // algorithm interacting with the orchestrator's delayed ACKs adds
        // tens of milliseconds per round trip, dominating everything else
        // given 210 of these happen per generated token.
        int one = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        // The orchestrator opens one persistent connection per worker at
        // startup, so one handler thread per accepted connection is plenty --
        // no epoll/select needed for a handful of sockets.
        std::thread(handle_connection, client_fd).detach();
    }
}
