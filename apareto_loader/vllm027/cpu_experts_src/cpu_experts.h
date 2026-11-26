// CPU-served decode misses: shared layout between the host server and the CUDA kernels.
#pragma once
#include <cstdint>

namespace cpux {

constexpr int HIDDEN = 2048, MIDDLE = 512, TOP = 8;
constexpr int TMAX = 8;                 // token rows a request carries; a larger decode batch stays on the GPU path
constexpr int PMAX = TMAX * TOP;        // (token, expert) pairs a request carries

// One per layer, in pinned host memory mapped into the device. The ring kernel writes it, the server reads it.
struct Req {
    uint32_t seq;                       // the layer's sequence number of this request (device counter)
    int32_t tokens, pairs;
    int32_t tok[PMAX], expert[PMAX];
    int32_t slot[PMAX];                 // the pair's (token row, top-k) index t * TOP + k: where its output norm is filed
    float scale[PMAX];                  // routing weight of the pair
    uint64_t record[PMAX];              // host address of the expert's record (host tier)
    float hidden[TMAX][HIDDEN];         // the MoE input rows, fp32
};

// One per layer: the routing-weighted sum of the CPU-served experts per token row.
struct Resp {
    float out[TMAX][HIDDEN];
    float norm[PMAX];                   // per pair: L2 norm of the expert's own output (before the routing weight)
};

// Record field offsets in bytes (layer store layout, apareto_loader/layer_store.py).
struct Offsets {
    int64_t w13, s13, w2, s2, g13, g2;
};

// Device counters, read by stats(): index meaning.
// C_RESCUED: dropped picks served by the rescue rule; C_RESCUE_UWEIGHT: their routing weight in millionths;
// C_RESCUE_NO_RECORD: a pick the rule chose that has no host record (not served); C_RESCUED_GPU: rescues the bank
// held, served on the GPU (counted in C_RESCUED too).
enum Counter { C_LAYERS = 0, C_PAIRS, C_TIMEOUTS, C_FALLBACK_PAIRS, C_WAIT_NS, C_BAD, C_RESCUED, C_RESCUE_UWEIGHT,
               C_RESCUE_NO_RECORD, C_RESCUED_GPU, C_COUNT };

}  // namespace cpux
