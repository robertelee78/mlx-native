/// Fused per-head RMSNorm + SiLU gate kernel for DeltaNet op 8.
///
/// Computes:
///   normed[t, vh, d] = attn_out[t, vh, d] * rsqrt(mean(attn_out[t, vh, :]^2) + eps)
///                      * weight[d]
///   output[t, vh, d] = normed[t, vh, d] * silu(z[t, vh, d])
///
/// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
///
/// This replaces the CPU bridge in apply_ssm_norm_and_gate: eliminates 2 GPU
/// downloads + CPU RMSNorm/SiLU loop + 1 upload per delta-net layer.
///
/// All buffers are f32.  Shapes:
///   attn_out / z / output : [seq * n_v_heads, d_v]  (row = one (t,vh) head-vec)
///   weight                : [d_v]
///   params                : [eps, d_v_f, unused, unused]  (float4)
///
/// Grid threadgroups : (seq * n_v_heads, 1, 1) — one threadgroup per head-vec
/// Threads/threadgroup: next_pow2(d_v) capped at 256 or 1024 per Metal limits.

#include <metal_stdlib>
using namespace metal;

kernel void ssm_norm_gate_f32(
    device const float *attn_out  [[buffer(0)]],  // [rows, d_v]
    device const float *weight    [[buffer(1)]],  // [d_v]  — ssm_norm weight
    device const float *z         [[buffer(2)]],  // [rows, d_v]
    device       float *output    [[buffer(3)]],  // [rows, d_v]
    device const float *params    [[buffer(4)]],  // [eps, d_v_f]
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const float eps  = params[0];
    const uint  d_v  = uint(params[1]);
    const uint  base = row_idx * d_v;

    // Phase 1: sum of squares over this head vector.
    float partial_sq = 0.0f;
    for (uint i = tid; i < d_v; i += tg_size) {
        const float v = attn_out[base + i];
        partial_sq += v * v;
    }
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(d_v) + eps);

    // Phase 2: compute normed * silu(z).
    for (uint i = tid; i < d_v; i += tg_size) {
        const float x     = attn_out[base + i];
        const float normed = x * rms_inv * weight[i];
        const float zv    = z[base + i];
        const float sig   = 1.0f / (1.0f + precise::exp(-zv));
        output[base + i]  = normed * (zv * sig);
    }
}
