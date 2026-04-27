#include <metal_stdlib>
using namespace metal;

// Wave 5b.1 iter 4 — chunk-local cumsum for [B, T, H] log-decay tensor.
//
// Spec source: FLA `chunk_local_cumsum_scalar` at
//   /opt/vllm/vllm/model_executor/layers/fla/ops/cumsum.py:160-195.
//
// For each (b, h) and each chunk i_t in 0..NT:
//   g_cumsum[b, t_start..t_start+BT, h] = inclusive_prefix_sum_along_t(
//       g_in[b, t_start..t_start+BT, h]
//   )
//
// Operates in f32 throughout (g is f32).
//
// # Layout
//
//   g_in     : [B, T, H] f32 — H innermost (time-stride = H)
//   g_cumsum : [B, T, H] f32 — same layout
//
// # Threading model
//
//   Grid: (1, H, B*NT). One threadgroup per (i_t, h, b). Each thread in the
//   threadgroup owns one timestep i ∈ [0, BT) but only thread 0 writes the
//   sequential per-chunk scan accumulator. (Iter 4.5 audit fix a62b111 reduced
//   grid_x from BT to 1; the prior dispatch had 64 redundant threadgroups
//   producing duplicate writes that the GPU coalesced — semantics correct
//   but pure dispatch overhead.)
//
//   Sequential per-chunk scan: thread 0 walks i=0..BT-1, accumulating.
//   This is acceptable because BT=64 is small and we have B*NT*H
//   threadgroups running in parallel (8 in the test fixture).
//
//   Iter 5 perf can replace with a Hillis-Steele scan if needed; the
//   8-threadgroup serial scan is well below noise on M5 Max.

constant constexpr uint MAX_BT = 64;

kernel void chunk_local_cumsum_g_f32(
    device const float *g_in     [[buffer(0)]],
    device       float *g_cumsum [[buffer(1)]],
    device const uint  *params   [[buffer(2)]],   // [B, T, H, BT, NT]
    uint3 tg_pos                  [[threadgroup_position_in_grid]],
    uint3 t_pos                   [[thread_position_in_threadgroup]]
) {
    const uint B  = params[0];
    const uint T  = params[1];
    const uint H  = params[2];
    const uint BT = params[3];
    const uint NT = params[4];

    const uint i_h = tg_pos.y;
    const uint b_nt = tg_pos.z;
    const uint b   = b_nt / NT;
    const uint i_t = b_nt % NT;
    if (b >= B || i_h >= H || i_t >= NT) {
        return;
    }

    // Only thread 0 walks the chunk serially. BT <= 64 so this is bounded;
    // parallelism comes from the (B * NT * H) threadgroups.
    if (t_pos.x != 0) {
        return;
    }

    const uint t_start = i_t * BT;
    const uint base = (b * T + t_start) * H + i_h;
    float acc = 0.0f;
    for (uint i = 0; i < BT; ++i) {
        acc += g_in[base + i * H];
        g_cumsum[base + i * H] = acc;
    }
}
