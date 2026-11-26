// CUDA side of CPU-served decode misses. Three kernels per MoE layer of a decode step:
//   ring    (1 block)  split the router's top-8 into GPU pairs and CPU pairs, write the CPU request, ring the doorbell;
//                      with the rescue rule on, also the heaviest dropped picks above the threshold.
//                      Each token row is tested against its own request's keep-set: row t reads keep row row_slot[t]
//                      (one request: row_slot is all zeros, keep one row)
//   wait    (1 thread) wait for the server's done flag; after the timeout decide on the fallback
//   finish  (grid)     add the server's output, or compute the timed-out pairs zero-copy from their records and add
// plus fallback_up (grid), the first half of the fallback (gate/up). Every kernel exits at once when its layer had no
// CPU pairs (or, for the fallback, when the CPU answered in time), so the graph shape is fixed.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "cpu_experts.h"

namespace cpux {

__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__constant__ float c_e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float e4m3(uint8_t x) {
    __nv_fp8_e4m3 v; v.__x = x; return (x & 0x7F) == 0x7F ? 0.f : float(v);
}
// FlashInfer CUTLASS block-scale swizzle (128x4 tiles), as in the layer store.
__device__ __forceinline__ size_t sw_off(int r, int c, int C) {
    return ((size_t) ((r / 128) * (C / 4) + c / 4) * 32 + r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4;
}

__global__ void k_ring(int layer, int tokens, const int32_t* logical, const float* weights, const bool* keep,
                       const int32_t* row_slot, int keep_stride, int routed_cols, const int32_t* table, const int64_t* host_w13, const __nv_bfloat16* hidden, int32_t* routed,
                       int32_t* gpu_ids, float* gpu_scales, int32_t* cpu_ids, float* cpu_scales, Req* req,
                       volatile unsigned long long* doorbell, uint32_t* seq, int32_t* nreq, const int32_t* enabled,
                       int64_t off_w13, unsigned long long* counters, int absent_cpu, const int32_t* rescue_live,
                       float rescue_x, const int32_t* rescue_caps, const int64_t* rescue_w13, const float* rescue_gain, int gain_stride,
                       float* norms) {
    __shared__ int n;
    Req* r = req + layer;
    if (threadIdx.x == 0) n = 0;
    for (int i = threadIdx.x; i < PMAX; i += blockDim.x) norms[layer * PMAX + i] = -1.f;   // -1 not on the CPU
    __syncthreads();
    const bool cpu = *enabled && tokens <= TMAX;
    for (int i = threadIdx.x; i < tokens * TOP; i += blockDim.x) {
        int id = logical[i];
        float w = weights[i];
        const int slot = row_slot[i / TOP];           // -1: a padding row, nothing allowed
        bool allowed = id >= 0 && slot >= 0 && keep[(size_t) slot * keep_stride + id];
        int out_id = -1; float out_w = 0.f, cpu_w = 0.f; int cpu_id = -1;
        if (allowed) {
            int64_t address = host_w13[id];
            if (cpu && absent_cpu && table[id] < 0 && address != 0) {
                int p = atomicAdd(&n, 1);
                r->tok[p] = i / TOP; r->expert[p] = id; r->slot[p] = i; r->scale[p] = w; r->record[p] = (uint64_t) (address - off_w13);
                cpu_id = id; cpu_w = w;
            } else {
                out_id = id; out_w = w;       // resident, or absent and served zero-copy as without the CPU path
                if (table[id] < 0 && address == 0) atomicAdd(counters + C_BAD, 1ull);
            }
        }
        gpu_ids[i] = out_id; gpu_scales[i] = out_w;
        cpu_ids[i] = cpu_id; cpu_scales[i] = cpu_w;     // what the CPU serves, per (token, k): the audit's view
        if (i < routed_cols) routed[i] = allowed ? id : -1;   // the between-steps admission sees every row's allowed ids
    }
    for (int i = tokens * TOP + threadIdx.x; i < routed_cols; i += blockDim.x) routed[i] = -1;   // rows not in this step
    __syncthreads();
    // Rescue: per token row, the heaviest pick the keep-set drops, if its routing weight is
    // above rescue_x, is computed with that weight, on the GPU when the bank holds it, otherwise on the CPU; up to rescue_cap per row, heaviest first (the first k
    // on a tie). The other picks are not renormalised. Nothing is rescued when the row's heaviest drop is at or
    // under the threshold. "Heaviest" is the score weight x rescue_gain[expert] (this layer's row of the gain
    // table; all ones = the plain weight rule); the rescued pick is still computed with its routing weight.
    // rescue_caps[layer]: the cap is per layer; 0 turns the rule off in this layer.
    const int rescue_cap = rescue_caps[layer];
    if (cpu && rescue_cap > 0 && *rescue_live && threadIdx.x < tokens && row_slot[threadIdx.x] >= 0) {
        const int t = threadIdx.x;
        const bool* own = keep + (size_t) row_slot[t] * keep_stride;
        const float* gain = rescue_gain + (size_t) row_slot[t] * gain_stride;   // gain_stride 0: one row for every slot
        unsigned chosen = 0;
        for (int c = 0; c < rescue_cap; ++c) {
            int best = -1; float bs = rescue_x;
            for (int k = 0; k < TOP; ++k) {
                const int id = logical[t * TOP + k];
                if (id < 0 || own[id] || ((chosen >> k) & 1u)) continue;
                const float score = __fmul_rn(weights[t * TOP + k], gain[id]);
                if (score > bs) { bs = score; best = k; }
            }
            if (best < 0) break;
            chosen |= 1u << best;
            const int i = t * TOP + best, id = logical[i];
            const float bw = weights[i];
            if (table[id] >= 0) {
                // Resident in the bank (concurrent decode: another request's mask holds it): the GPU serves it.
                gpu_ids[i] = id; gpu_scales[i] = bw;
                atomicAdd(counters + C_RESCUED_GPU, 1ull);
            } else {
                const int64_t address = rescue_w13[id];      // every record's host address (the whole store is pinned)
                if (address == 0) { atomicAdd(counters + C_RESCUE_NO_RECORD, 1ull); continue; }
                const int p = atomicAdd(&n, 1);
                r->tok[p] = t; r->expert[p] = id; r->slot[p] = i; r->scale[p] = bw; r->record[p] = (uint64_t) (address - off_w13);
                cpu_ids[i] = id; cpu_scales[i] = bw;
            }
            atomicAdd(counters + C_RESCUED, 1ull);
            atomicAdd(counters + C_RESCUE_UWEIGHT, (unsigned long long) (bw * 1e6f + .5f));
        }
    }
    __syncthreads();
    const int pairs = n;
    if (pairs > 0) {
        for (int i = threadIdx.x; i < tokens * HIDDEN; i += blockDim.x)
            r->hidden[i / HIDDEN][i % HIDDEN] = __bfloat162float(hidden[i]);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t s = seq[layer] + 1;
            seq[layer] = s;
            r->tokens = tokens; r->pairs = pairs; r->seq = s;
            __threadfence_system();
            *doorbell = ((unsigned long long) s << 8) | (unsigned long long) layer;
            __threadfence_system();
            atomicAdd(counters + C_LAYERS, 1ull);
            atomicAdd(counters + C_PAIRS, (unsigned long long) pairs);
        }
    }
    if (threadIdx.x == 0) nreq[layer] = pairs;
}

// decision[layer]: 0 no CPU pairs, 1 the CPU answered, 2 timed out (fallback).
// The timeout grows with the request: a multi-row step (decode plus probe rows) can name ~20 experts in one layer.
__global__ void k_wait(int layer, const int32_t* nreq, const uint32_t* seq, volatile const uint32_t* done,
                       int32_t* decision, uint64_t timeout_ns, uint64_t per_pair_ns, unsigned long long* counters) {
    if (threadIdx.x != 0) return;
    const int pairs = nreq[layer];
    if (pairs == 0) { decision[layer] = 0; return; }
    const uint32_t want = seq[layer];
    const uint64_t t0 = gtimer();
    timeout_ns += per_pair_ns * (uint64_t) pairs;
    int d = 1;
    while (done[layer] != want) {
        if (gtimer() - t0 > timeout_ns) {
            d = 2;
            atomicAdd(counters + C_TIMEOUTS, 1ull);
            atomicAdd(counters + C_FALLBACK_PAIRS, (unsigned long long) pairs);
            break;
        }
    }
    __threadfence_system();
    decision[layer] = d;
    atomicAdd(counters + C_WAIT_NS, (unsigned long long) (gtimer() - t0));
}

// Fallback, gate/up: one warp per (pair, row of MIDDLE); ff[p][row] = silu(gate) * up, fp32 activations.
__global__ void k_fallback_up(int layer, const int32_t* decision, const Req* req, Offsets off, float* ff) {
    if (decision[layer] != 2) return;
    const volatile Req* r = req + layer;
    const int pairs = r->pairs;
    const int warps = gridDim.x * (blockDim.x / 32), lane = threadIdx.x & 31;
    for (int item = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); item < pairs * MIDDLE; item += warps) {
        const int p = item / MIDDLE, row = item % MIDDLE, t = r->tok[p];
        const uint8_t* R = (const uint8_t*) r->record[p];
        const float g13 = *(const float*) (R + off.g13);
        float acc[2];
        for (int which = 0; which < 2; ++which) {               // w13 = [up; gate]
            const int wr = which == 0 ? MIDDLE + row : row;
            const uint8_t* q = R + off.w13 + (size_t) wr * (HIDDEN / 2);
            const uint8_t* s = R + off.s13;
            float a = 0.f;
            for (int c = lane; c < HIDDEN / 2; c += 32) {        // byte c holds elements 2c (low) and 2c+1 (high)
                const uint8_t v = q[c];
                const float sc = e4m3(s[sw_off(wr, (2 * c) / 16, HIDDEN / 16)]);
                a += sc * (c_e2m1[v & 15] * r->hidden[t][2 * c] + c_e2m1[v >> 4] * r->hidden[t][2 * c + 1]);
            }
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            acc[which] = a / g13;
        }
        if (lane == 0) { const float g = acc[0], u = acc[1]; ff[p * MIDDLE + row] = g / (1.f + expf(-g)) * u; }
    }
}

// Add the CPU's rows (decision 1) or compute the down projection of the timed-out pairs and add it (decision 2).
__global__ void k_finish(int layer, int tokens, const int32_t* decision, const Req* req, const Resp* resp,
                         Offsets off, const float* ff, __nv_bfloat16* fused, float* norms) {
    const int d = decision[layer];
    if (d == 0) return;
    if (blockIdx.x == 0) {      // each CPU pair's output norm at its (token row, k); -2 when it timed out
        const volatile Req* q = req + layer;
        const volatile Resp* o = resp + layer;
        for (int p = threadIdx.x; p < q->pairs; p += blockDim.x) norms[layer * PMAX + q->slot[p]] = d == 1 ? o->norm[p] : -2.f;
    }
    if (d == 1) {
        const volatile Resp* o = resp + layer;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tokens * HIDDEN; i += gridDim.x * blockDim.x)
            fused[i] = __float2bfloat16(__bfloat162float(fused[i]) + o->out[i / HIDDEN][i % HIDDEN]);
        return;
    }
    const volatile Req* r = req + layer;
    const int pairs = r->pairs;
    const int warps = gridDim.x * (blockDim.x / 32), lane = threadIdx.x & 31;
    for (int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); row < HIDDEN; row += warps) {
        float acc[TMAX];
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int p = 0; p < pairs; ++p) {
            const uint8_t* R = (const uint8_t*) r->record[p];
            const uint8_t* q = R + off.w2 + (size_t) row * (MIDDLE / 2);
            const uint8_t* s = R + off.s2;
            const float* f = ff + p * MIDDLE;
            float a = 0.f;
            for (int c = lane; c < MIDDLE / 2; c += 32) {
                const uint8_t v = q[c];
                const float sc = e4m3(s[sw_off(row, (2 * c) / 16, MIDDLE / 16)]);
                a += sc * (c_e2m1[v & 15] * f[2 * c] + c_e2m1[v >> 4] * f[2 * c + 1]);
            }
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            acc[r->tok[p]] += r->scale[p] * a / *(const float*) (R + off.g2);
        }
        if (lane < tokens) {
            const int i = lane * HIDDEN + row;
            fused[i] = __float2bfloat16(__bfloat162float(fused[i]) + acc[lane]);
        }
    }
}

void launch_ring(cudaStream_t st, int layer, int tokens, const int32_t* logical, const float* weights, const bool* keep,
                 const int32_t* row_slot, int keep_stride, int routed_cols, const int32_t* table, const int64_t* host_w13, const void* hidden, int32_t* routed, int32_t* gpu_ids,
                 float* gpu_scales, int32_t* cpu_ids, float* cpu_scales, Req* req, unsigned long long* doorbell,
                 uint32_t* seq, int32_t* nreq, const int32_t* enabled, int64_t off_w13, unsigned long long* counters,
                 int absent_cpu, const int32_t* rescue_live, float rescue_x, const int32_t* rescue_caps, const int64_t* rescue_w13,
                 const float* rescue_gain, int gain_stride, float* norms) {
    k_ring<<<1, 256, 0, st>>>(layer, tokens, logical, weights, keep, row_slot, keep_stride, routed_cols, table, host_w13, (const __nv_bfloat16*) hidden,
                              routed, gpu_ids, gpu_scales, cpu_ids, cpu_scales, req, doorbell, seq, nreq, enabled,
                              off_w13, counters, absent_cpu, rescue_live, rescue_x, rescue_caps, rescue_w13, rescue_gain, gain_stride, norms);
}

void launch_finish(cudaStream_t st, int layer, int tokens, const int32_t* nreq, const uint32_t* seq, const uint32_t* done,
                   int32_t* decision, uint64_t timeout_ns, uint64_t per_pair_ns, const Req* req, const Resp* resp,
                   Offsets off, float* ff, void* fused, unsigned long long* counters, float* norms) {
    k_wait<<<1, 32, 0, st>>>(layer, nreq, seq, done, decision, timeout_ns, per_pair_ns, counters);
    k_fallback_up<<<64, 256, 0, st>>>(layer, decision, req, off, ff);
    k_finish<<<64, 256, 0, st>>>(layer, tokens, decision, req, resp, off, ff, (__nv_bfloat16*) fused, norms);
}

}  // namespace cpux
