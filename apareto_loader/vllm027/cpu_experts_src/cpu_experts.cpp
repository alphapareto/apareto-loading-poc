// Host side of CPU-served decode misses: the request/response slots in pinned mapped
// memory, the server thread with its pinned worker pool, the AVX2 NVFP4 x Q8_0 expert kernel, and the torch bindings.
//
// The expert kernel (quant_q8, values32, hsum8, row_dot) is derived from Strata (github.com/Niko1221/Strata, src/kernels/cpu/nvfp4_avx2.cpp, MIT License) and ggml's NVFP4 x
// Q8_0 dot product (MIT License). It reads the layer store's records in place: w13 = [up; gate] packed E2M1 with
// adjacent elements in one byte, block scales swizzled into 128x4 tiles, per-tensor global scales divided out.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include "cpu_experts.h"

namespace cpux {

void launch_ring(cudaStream_t, int, int, const int32_t*, const float*, const bool*, const int32_t*, int, int,
                 const int32_t*, const int64_t*,
                 const void*, int32_t*, int32_t*, float*, int32_t*, float*, Req*, unsigned long long*, uint32_t*,
                 int32_t*, const int32_t*, int64_t, unsigned long long*, int, const int32_t*, float, const int32_t*,
                 const int64_t*, const float*, int, float*);
void launch_finish(cudaStream_t, int, int, const int32_t*, const uint32_t*, const uint32_t*, int32_t*, uint64_t,
                   uint64_t, const Req*, const Resp*, Offsets, float*, void*, unsigned long long*, float*);

// Idle threads keep spinning this long before yielding in short sleeps: longer than the decode pauses of a mask
// refresh or probe (~0.2 s), so the first requests after one do not wait for a sleeping thread.
constexpr int IDLE_SPIN_MS = 2000;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + cudaGetErrorString(e_) + " at " #x); } while (0)

// ---- the expert kernel ------------------------------------------------------------------------------------------
static const int8_t kvalues_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float UE4M3[256];
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
static inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
static inline uint16_t f2h(float f) { return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0); }
static inline size_t sw_off(int r, int c, int C) {
    return ((size_t) ((r / 128) * (C / 4) + c / 4) * 32 + r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4;
}
static void init_tables() {
    for (int x = 0; x < 256; ++x) {
        if (x == 0 || (x & 0x7F) == 0x7F) { UE4M3[x] = 0; continue; }
        int e = (x >> 3) & 0xF, m = x & 7;
        float v = 0.5f * (e == 0 ? m * std::ldexp(1.f, -9) : (1.f + m / 8.f) * std::ldexp(1.f, e - 7));
        UE4M3[x] = (x & 0x80) ? -v : v;
    }
}
static void quant_q8(const float* x, int n, block_q8_0* y, float* dy) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f; y[b].d = f2h(d);
        for (int i = 0; i < 32; ++i) y[b].qs[i] = (int8_t) std::lrintf(x[b * 32 + i] * id);
        dy[b] = h2f(y[b].d);
    }
}
static inline __m256i values32(const uint8_t* qs, const __m128i kv) {
    const __m128i q = _mm_loadu_si128((const __m128i*) qs), m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
    const __m128i hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}
static inline float hsum8(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r)); r = _mm_add_ss(r, _mm_movehdup_ps(r)); return _mm_cvtss_f32(r);
}
// One weight row (C = row length / 16 scale columns, nb = row length / 64) against one Q8_0 activation vector.
static inline float row_dot(const uint8_t* row, const uint8_t* scales, int r, int C, int nb, const block_q8_0* y, const float* dy) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) kvalues_fp4); const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t* qs = row + 32 * ib; const uint8_t* d = scales + sw_off(r, 4 * ib, C);
        const __m256i v01 = values32(qs, kv), v23 = values32(qs + 16, kv);
        const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(UE4M3[d[1]]), _mm_set1_ps(UE4M3[d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(UE4M3[d[3]]), _mm_set1_ps(UE4M3[d[2]]));
        const __m256i q01 = _mm256_loadu_si256((const __m256i*) y[2 * ib].qs), q23 = _mm256_loadu_si256((const __m256i*) y[2 * ib + 1].qs);
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
        const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(u01, _mm256_set1_ps(dy[2 * ib])), _mm256_cvtepi32_ps(p1), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(u23, _mm256_set1_ps(dy[2 * ib + 1])), _mm256_cvtepi32_ps(p2), acc);
    }
    return hsum8(acc);
}

// ---- a pinned spinning worker pool --------------------------------------------------------------------------------
// The caller is worker 0. A job is a function of (begin, end) over [0, items), split statically across workers.
class Pool {
public:
    Pool(const std::vector<int>& cores) : cores_(cores) {
        for (size_t w = 1; w < cores.size(); ++w) threads_.emplace_back([this, w] { run((int) w); });
    }
    ~Pool() { stop_ = true; for (auto& t : threads_) t.join(); }
    template <class F> void parallel(int items, F&& f) {
        fn_ = [&f](int a, int b) { f(a, b); };
        items_ = items;
        done_.store(0);
        generation_.fetch_add(1, std::memory_order_release);
        slice(0);
        while (done_.load(std::memory_order_acquire) != (int) threads_.size()) _mm_pause();
    }
    static void pin(int core) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
private:
    void slice(int w) {
        const int n = (int) cores_.size(), a = (int) ((long) items_ * w / n), b = (int) ((long) items_ * (w + 1) / n);
        if (a < b) fn_(a, b);
    }
    void run(int w) {
        pin(cores_[w]);
        // Generation 0 is "no job yet": a job can be published before this thread first runs, and reading the
        // current generation here would skip it (and the caller would wait for this worker forever).
        unsigned seen = 0;
        auto idle_since = std::chrono::steady_clock::now();
        while (!stop_) {
            unsigned g = generation_.load(std::memory_order_acquire);
            if (g == seen) {
                // Spin while decode is running; after 100 ms without work, yield the core in short sleeps.
                if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(IDLE_SPIN_MS)) usleep(20);
                else _mm_pause();
                continue;
            }
            seen = g;
            slice(w);
            done_.fetch_add(1, std::memory_order_acq_rel);
            idle_since = std::chrono::steady_clock::now();
        }
    }
    std::vector<int> cores_;
    std::vector<std::thread> threads_;
    std::function<void(int, int)> fn_;
    std::atomic<unsigned> generation_{0};
    std::atomic<int> done_{0};
    int items_ = 0;
    std::atomic<bool> stop_{false};
};

// ---- the server ---------------------------------------------------------------------------------------------------
struct State {
    int layers = 0;
    Offsets off{};
    uint64_t timeout_ns = 0, per_pair_ns = 0;
    // pinned host, mapped
    Req* req = nullptr; Resp* resp = nullptr;
    volatile unsigned long long* doorbell = nullptr; volatile uint32_t* done = nullptr;
    Req* dreq = nullptr; Resp* dresp = nullptr; unsigned long long* ddoorbell = nullptr; uint32_t* ddone = nullptr;
    // device
    torch::Tensor seq, nreq, decision, ff, counters, enabled;
    torch::Tensor norms;     // [layers, TMAX * TOP] per (token row, k): the CPU-computed output norm; -1 not on the CPU, -2 timed out
    // absent allowed experts to the CPU (1) or the GPU's zero-copy path (0); the rescue rule
    // (threshold, cap per token row per layer, a device table [layers]; 0 = off in that layer) and its device
    // switch, set between steps by phase.
    int absent_cpu = 1; float rescue_x = 0.f;
    torch::Tensor rescue_live, rescue_caps;
    std::unique_ptr<Pool> pool;
    std::thread server;
    std::atomic<bool> stop{false};
    std::mutex norm_lock; float norm_sq[PMAX];
    std::atomic<unsigned long long> served_requests{0}, served_pairs{0}, served_experts{0}, stale{0}, compute_ns{0};
    // compute scratch (server thread only)
    block_q8_0 qx[TMAX][HIDDEN / 32]; float dx[TMAX][HIDDEN / 32];
    float ff_host[PMAX][TMAX][MIDDLE];
    block_q8_0 qf[PMAX][TMAX][MIDDLE / 32]; float df[PMAX][TMAX][MIDDLE / 32];
};
static std::unique_ptr<State> S;

// Compute one request: the distinct experts it names, each once for every token row that missed it. Returns false
// when the GPU moved on mid-way (a later request for this layer): the rest is not worth computing.
static bool serve(const Req& r, Resp& o, uint32_t seq) {
    State& s = *S;
    const int tokens = r.tokens, pairs = r.pairs;
    int experts[PMAX], nexp = 0, of_pair[PMAX];
    for (int p = 0; p < pairs; ++p) {
        int e = -1;
        for (int k = 0; k < nexp; ++k) if (r.record[experts[k]] == r.record[p]) { e = k; break; }
        if (e < 0) { e = nexp; experts[nexp++] = p; }
        of_pair[p] = e;
    }
    // token rows that each distinct expert serves, with the pair's routing weight
    int rows_of[PMAX][TMAX], nrows[PMAX] = {0}, pair_of[PMAX][TMAX]; float w_of[PMAX][TMAX];
    for (int p = 0; p < pairs; ++p) {
        int e = of_pair[p]; rows_of[e][nrows[e]] = r.tok[p]; pair_of[e][nrows[e]] = p; w_of[e][nrows[e]++] = r.scale[p];
    }
    for (int p = 0; p < pairs; ++p) s.norm_sq[p] = 0.f;
    for (int t = 0; t < tokens; ++t) quant_q8(r.hidden[t], HIDDEN, s.qx[t], s.dx[t]);
    const Offsets off = s.off;
    s.pool->parallel(nexp * MIDDLE, [&](int a, int b) {
        for (int i = a; i < b; ++i) {
            const int e = i / MIDDLE, row = i % MIDDLE;
            const uint8_t* R = (const uint8_t*) r.record[experts[e]];
            float g13; memcpy(&g13, R + off.g13, 4);
            for (int j = 0; j < nrows[e]; ++j) {
                const int t = rows_of[e][j];
                float g = row_dot(R + off.w13 + (size_t) (MIDDLE + row) * (HIDDEN / 2), R + off.s13, MIDDLE + row,
                                  HIDDEN / 16, HIDDEN / 64, s.qx[t], s.dx[t]) / g13;
                float u = row_dot(R + off.w13 + (size_t) row * (HIDDEN / 2), R + off.s13, row, HIDDEN / 16, HIDDEN / 64,
                                  s.qx[t], s.dx[t]) / g13;
                s.ff_host[e][j][row] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    if (((const volatile Req&) r).seq != seq) return false;
    for (int e = 0; e < nexp; ++e)
        for (int j = 0; j < nrows[e]; ++j) quant_q8(s.ff_host[e][j], MIDDLE, s.qf[e][j], s.df[e][j]);
    s.pool->parallel(HIDDEN, [&](int a, int b) {
        float sq[PMAX] = {0};
        for (int row = a; row < b; ++row) {
            float acc[TMAX] = {0};
            for (int e = 0; e < nexp; ++e) {
                const uint8_t* R = (const uint8_t*) r.record[experts[e]];
                float g2; memcpy(&g2, R + off.g2, 4);
                for (int j = 0; j < nrows[e]; ++j) {
                    const float v = row_dot(R + off.w2 + (size_t) row * (MIDDLE / 2), R + off.s2, row,
                                            MIDDLE / 16, MIDDLE / 64, s.qf[e][j], s.df[e][j]) / g2;
                    acc[rows_of[e][j]] += w_of[e][j] * v;
                    sq[pair_of[e][j]] += v * v;
                }
            }
            for (int t = 0; t < tokens; ++t) o.out[t][row] = acc[t];
        }
        std::lock_guard<std::mutex> hold(s.norm_lock);
        for (int p = 0; p < pairs; ++p) s.norm_sq[p] += sq[p];
    });
    for (int p = 0; p < pairs; ++p) o.norm[p] = std::sqrt(s.norm_sq[p]);
    s.served_experts += nexp;
    return true;
}

static void server_loop() {
    State& s = *S;
    unsigned long long seen = *s.doorbell;
    auto idle_since = std::chrono::steady_clock::now();
    while (!s.stop.load(std::memory_order_relaxed)) {
        const unsigned long long bell = *s.doorbell;
        if (bell == seen) {
            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(IDLE_SPIN_MS)) usleep(20);
            else _mm_pause();
            continue;
        }
        seen = bell;
        const int layer = (int) (bell & 0xFF);
        const uint32_t seq = (uint32_t) (bell >> 8);
        Req& r = s.req[layer];
        std::atomic_thread_fence(std::memory_order_acquire);
        if (((volatile Req&) r).seq != seq) { s.stale++; continue; }     // the GPU has moved on
        auto t0 = std::chrono::steady_clock::now();
        if (!serve(r, s.resp[layer], seq)) { s.stale++; continue; }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        s.done[layer] = seq;
        s.compute_ns += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        s.served_requests++; s.served_pairs += r.pairs;
        idle_since = std::chrono::steady_clock::now();
    }
}

// ---- bindings -----------------------------------------------------------------------------------------------------
static void start(int layers, std::vector<int64_t> offsets, std::vector<int64_t> cores, double timeout_us,
                  double per_pair_us, int64_t device, bool absent_cpu, double rescue_x, std::vector<int64_t> rescue_caps) {
    if (S) throw std::runtime_error("cpu experts already started");
    if (offsets.size() != 6 || cores.empty()) throw std::runtime_error("offsets: 6 values; cores: at least one");
    if (layers > 255) throw std::runtime_error("the doorbell carries the layer in 8 bits");
    init_tables();
    S = std::make_unique<State>();
    State& s = *S;
    s.layers = layers;
    s.off = Offsets{offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5]};
    s.timeout_ns = (uint64_t) (timeout_us * 1000.0);
    s.per_pair_ns = (uint64_t) (per_pair_us * 1000.0);
    CK(cudaHostAlloc((void**) &s.req, sizeof(Req) * layers, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &s.resp, sizeof(Resp) * layers, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &s.doorbell, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &s.done, sizeof(uint32_t) * layers, cudaHostAllocMapped));
    memset(s.req, 0, sizeof(Req) * layers); memset(s.resp, 0, sizeof(Resp) * layers);
    *(unsigned long long*) s.doorbell = 0; memset((void*) s.done, 0, sizeof(uint32_t) * layers);
    CK(cudaHostGetDevicePointer((void**) &s.dreq, s.req, 0)); CK(cudaHostGetDevicePointer((void**) &s.dresp, s.resp, 0));
    CK(cudaHostGetDevicePointer((void**) &s.ddoorbell, (void*) s.doorbell, 0));
    CK(cudaHostGetDevicePointer((void**) &s.ddone, (void*) s.done, 0));
    auto dev = torch::TensorOptions().device(torch::kCUDA, device);
    s.seq = torch::zeros({layers}, dev.dtype(torch::kInt32));
    s.nreq = torch::zeros({layers}, dev.dtype(torch::kInt32));
    s.decision = torch::zeros({layers}, dev.dtype(torch::kInt32));
    s.ff = torch::zeros({PMAX, MIDDLE}, dev.dtype(torch::kFloat32));
    s.norms = torch::full({layers, PMAX}, -1.f, dev.dtype(torch::kFloat32));
    s.counters = torch::zeros({C_COUNT}, dev.dtype(torch::kInt64));
    s.enabled = torch::ones({1}, dev.dtype(torch::kInt32));
    s.rescue_live = torch::zeros({1}, dev.dtype(torch::kInt32));
    s.absent_cpu = absent_cpu ? 1 : 0; s.rescue_x = (float) rescue_x;
    if ((int) rescue_caps.size() != layers) throw std::runtime_error("rescue caps: one per layer");
    for (int64_t c : rescue_caps)
        if (c < 0 || c > TOP) throw std::runtime_error("rescue cap: 0..8 per token row");
    s.rescue_caps = torch::tensor(std::vector<int32_t>(rescue_caps.begin(), rescue_caps.end()), torch::kInt32).to(dev.device());
    std::vector<int> c(cores.begin(), cores.end());
    // The server thread is worker 0 on the first core; the pool's other workers take the rest.
    std::thread t([c] {
        Pool::pin(c[0]);
        S->pool = std::make_unique<Pool>(c);
        server_loop();
        S->pool.reset();
    });
    s.server = std::move(t);
}

static void stop() {
    if (!S) return;
    S->stop = true;
    if (S->server.joinable()) S->server.join();
    cudaFreeHost(S->req); cudaFreeHost(S->resp); cudaFreeHost((void*) S->doorbell); cudaFreeHost((void*) S->done);
    S.reset();
}

// [layers, TMAX * TOP]: the last step's CPU output norm per (token row, k).
static torch::Tensor norms() { return S->norms; }

static void set_enabled(bool on) { S->enabled.fill_(on ? 1 : 0); }
// Stream-ordered: decode steps enqueued after this call see the new value.
static void set_rescue(bool on) { S->rescue_live.fill_(on ? 1 : 0); }

// Splits the layer's routing into GPU pairs (written to gpu_ids / gpu_scales) and CPU pairs (the request).
static void ring(int64_t layer, torch::Tensor logical, torch::Tensor weights, torch::Tensor keep, torch::Tensor table,
                 torch::Tensor host_w13, torch::Tensor hidden, torch::Tensor routed, torch::Tensor gpu_ids,
                 torch::Tensor gpu_scales, torch::Tensor cpu_ids, torch::Tensor cpu_scales, torch::Tensor rescue_w13,
                 torch::Tensor row_slot, torch::Tensor rescue_gain) {
    State& s = *S;
    TORCH_CHECK(logical.scalar_type() == torch::kInt32 && logical.is_contiguous() && logical.size(1) == TOP);
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && weights.is_contiguous());
    TORCH_CHECK(keep.scalar_type() == torch::kBool && keep.is_contiguous() && table.scalar_type() == torch::kInt32);
    // One keep row per request slot; row_slot[t] names token row t's slot.
    TORCH_CHECK(row_slot.scalar_type() == torch::kInt32 && row_slot.is_contiguous() && row_slot.numel() >= logical.size(0));
    TORCH_CHECK(keep.numel() % table.numel() == 0 && routed.scalar_type() == torch::kInt32);
    TORCH_CHECK(host_w13.scalar_type() == torch::kInt64 && host_w13.is_contiguous());
    TORCH_CHECK(rescue_w13.scalar_type() == torch::kInt64 && rescue_w13.is_contiguous() && rescue_w13.numel() == host_w13.numel());
    // The layer's row of the rescue score's gain table, one value per expert.
    // One row shared by every slot, or one row per keep slot (a per-request gain under concurrent decode).
    TORCH_CHECK(rescue_gain.scalar_type() == torch::kFloat32 && rescue_gain.is_contiguous()
                && (rescue_gain.numel() == table.numel() || rescue_gain.numel() == keep.numel()));
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && hidden.is_contiguous() && hidden.size(1) == HIDDEN);
    TORCH_CHECK(gpu_ids.is_contiguous() && gpu_scales.is_contiguous() && routed.is_contiguous());
    launch_ring(at::cuda::getCurrentCUDAStream(), (int) layer, (int) logical.size(0), logical.data_ptr<int32_t>(),
                weights.data_ptr<float>(), keep.data_ptr<bool>(), row_slot.data_ptr<int32_t>(), (int) table.numel(),
                (int) routed.numel(), table.data_ptr<int32_t>(), host_w13.data_ptr<int64_t>(),
                hidden.data_ptr(), routed.data_ptr<int32_t>(), gpu_ids.data_ptr<int32_t>(), gpu_scales.data_ptr<float>(),
                cpu_ids.data_ptr<int32_t>(), cpu_scales.data_ptr<float>(), s.dreq, s.ddoorbell, (uint32_t*) s.seq.data_ptr<int32_t>(), s.nreq.data_ptr<int32_t>(),
                s.enabled.data_ptr<int32_t>(), s.off.w13, (unsigned long long*) s.counters.data_ptr<int64_t>(),
                s.absent_cpu, s.rescue_live.data_ptr<int32_t>(), s.rescue_x, s.rescue_caps.data_ptr<int32_t>(), rescue_w13.data_ptr<int64_t>(),
                rescue_gain.data_ptr<float>(), rescue_gain.numel() == table.numel() ? 0 : (int) table.numel(),
                s.norms.data_ptr<float>());
}

// Waits for the layer's CPU pairs (or computes them after the timeout) and adds them into `fused` in place.
static void finish(int64_t layer, torch::Tensor fused) {
    State& s = *S;
    TORCH_CHECK(fused.scalar_type() == torch::kBFloat16 && fused.is_contiguous() && fused.size(1) == HIDDEN);
    launch_finish(at::cuda::getCurrentCUDAStream(), (int) layer, (int) fused.size(0), s.nreq.data_ptr<int32_t>(),
                  (const uint32_t*) s.seq.data_ptr<int32_t>(), s.ddone, s.decision.data_ptr<int32_t>(), s.timeout_ns,
                  s.per_pair_ns, s.dreq, s.dresp, s.off, s.ff.data_ptr<float>(), fused.data_ptr(),
                  (unsigned long long*) s.counters.data_ptr<int64_t>(), s.norms.data_ptr<float>());
}

static std::vector<int64_t> stats() {
    State& s = *S;
    auto c = s.counters.cpu();
    std::vector<int64_t> out(c.data_ptr<int64_t>(), c.data_ptr<int64_t>() + C_COUNT);
    for (auto v : {s.served_requests.load(), s.served_pairs.load(), s.served_experts.load(), s.stale.load(),
                   s.compute_ns.load()})
        out.push_back((int64_t) v);
    return out;
}

// Test hook: the layer's last request as written by the ring kernel (tokens, then (token, expert, scale) per pair).
static std::vector<double> request(int64_t layer) {
    const volatile Req& r = S->req[layer];
    std::vector<double> out{(double) r.seq, (double) r.tokens, (double) r.pairs};
    for (int p = 0; p < r.pairs; ++p) { out.push_back(r.tok[p]); out.push_back(r.expert[p]); out.push_back(r.scale[p]); }
    return out;
}

// Test hook: compute one expert for one token row on the CPU exactly as the server does (record = host address).
static torch::Tensor expert(int64_t record, torch::Tensor x, std::vector<int64_t> offsets, int64_t threads_unused) {
    init_tables();
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.numel() == HIDDEN);
    Offsets off{offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5]};
    const uint8_t* R = (const uint8_t*) record;
    block_q8_0 qx[HIDDEN / 32], qf[MIDDLE / 32]; float dx[HIDDEN / 32], df[MIDDLE / 32], ff[MIDDLE];
    quant_q8(x.data_ptr<float>(), HIDDEN, qx, dx);
    float g13, g2; memcpy(&g13, R + off.g13, 4); memcpy(&g2, R + off.g2, 4);
    for (int row = 0; row < MIDDLE; ++row) {
        float g = row_dot(R + off.w13 + (size_t) (MIDDLE + row) * (HIDDEN / 2), R + off.s13, MIDDLE + row, HIDDEN / 16, HIDDEN / 64, qx, dx) / g13;
        float u = row_dot(R + off.w13 + (size_t) row * (HIDDEN / 2), R + off.s13, row, HIDDEN / 16, HIDDEN / 64, qx, dx) / g13;
        ff[row] = g / (1.f + std::exp(-g)) * u;
    }
    quant_q8(ff, MIDDLE, qf, df);
    auto out = torch::empty({HIDDEN}, torch::kFloat32);
    for (int row = 0; row < HIDDEN; ++row)
        out.data_ptr<float>()[row] = row_dot(R + off.w2 + (size_t) row * (MIDDLE / 2), R + off.s2, row, MIDDLE / 16, MIDDLE / 64, qf, df) / g2;
    return out;
}

}  // namespace cpux

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("start", &cpux::start);
    m.def("stop", &cpux::stop);
    m.def("set_enabled", &cpux::set_enabled);
    m.def("set_rescue", &cpux::set_rescue);
    m.def("norms", &cpux::norms);
    m.def("ring", &cpux::ring);
    m.def("finish", &cpux::finish);
    m.def("stats", &cpux::stats);
    m.def("expert", &cpux::expert);
    m.def("request", &cpux::request);
    m.attr("TMAX") = cpux::TMAX;
}
