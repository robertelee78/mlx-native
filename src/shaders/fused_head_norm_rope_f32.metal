#include <metal_stdlib>
using namespace metal;

/// Fused per-head RMS normalization + NeoX rotary position embedding (float32).
///
/// Replaces two separate kernel dispatches (rms_norm_f32 per-head + rope_neox_f32)
/// with a single kernel launch per Q or K projection.
///
/// Supports optional freq_factors for Gemma 4's global attention layers.
/// When freq_factors[pair_idx] is very large (e.g. 1e30), the effective
/// rotation angle approaches zero, producing identity (cos~1, sin~0).
///
/// Layout:
///   Input : [n_heads * head_dim] f32  — one token's Q or K after linear projection
///   Output: [n_heads * head_dim] f32  — normalized and rotated
///
/// One threadgroup per head, threads cooperate on:
///   1. Parallel sum-of-squares reduction (shared f32 scratch)
///   2. Normalize every element with optional weight scale
///   3. NeoX-convention rotation on the first 2*half_rope_dim elements
///
/// Buffer layout:
///   buffer(0): input         — float [n_heads * head_dim]
///   buffer(1): output        — float [n_heads * head_dim]
///   buffer(2): norm_weight   — float [head_dim] (ignored when has_weight=0)
///   buffer(3): params        — FusedHeadNormRopeF32Params struct
///   buffer(4): positions     — uint [seq_len]
///   buffer(5): freq_factors  — float [half_rope_dim] (ignored when has_freq_factors=0)
///
/// Threadgroup: (min(256, next_pow2(head_dim)), 1, 1)
/// Grid       : (n_heads, 1, 1)
///
/// Shared memory: head_dim * sizeof(float) at threadgroup(0) for the reduction
/// and caching normalized values.

struct FusedHeadNormRopeF32Params {
    uint  head_dim;
    uint  n_heads;
    uint  half_rope_dim;   // may be < head_dim/2 for partial rotary
    float eps;
    uint  has_weight;      // 0 = no weight scale (V-norm variant), nonzero = apply weight
    float theta;           // RoPE base frequency (e.g. 10000 or 1000000)
    uint  has_freq_factors; // nonzero if freq_factors buffer is valid
    uint  _pad;            // alignment padding
};

kernel void fused_head_norm_rope_f32(
    device const float*                      input        [[buffer(0)]],
    device float*                            output       [[buffer(1)]],
    device const float*                      norm_weight  [[buffer(2)]],
    constant FusedHeadNormRopeF32Params&     params       [[buffer(3)]],
    device const uint*                       positions    [[buffer(4)]],
    device const float*                      freq_factors [[buffer(5)]],
    uint head_id  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared             [[threadgroup(0)]]
) {
    const uint head_dim     = params.head_dim;
    const uint half_rope    = params.half_rope_dim;
    const uint half_dim     = head_dim / 2;
    const float eps         = params.eps;
    const bool has_weight   = (params.has_weight != 0u);
    const float theta       = params.theta;
    const bool has_ff       = (params.has_freq_factors != 0u);
    const uint n_heads      = params.n_heads;

    // For single-token decode, seq_idx is always 0; head_id is which head.
    // The position comes from positions[seq_idx] where seq_idx = head_id / n_heads
    // But since we grid one TG per head and seq_len=1: seq_idx = 0.
    const uint seq_idx = head_id / n_heads;
    const uint pos = positions[seq_idx];

    const uint base = head_id * head_dim;

    // -------------------------------------------------------------------------
    // Phase 1: parallel sum-of-squares reduction
    // -------------------------------------------------------------------------
    float partial_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        const float val = input[base + i];
        partial_sq += val * val;
    }

    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(head_dim) + eps);

    // -------------------------------------------------------------------------
    // Phase 2: normalize (optionally scale with weight), store in shared
    // -------------------------------------------------------------------------
    for (uint i = tid; i < head_dim; i += tg_size) {
        float val = input[base + i] * rms_inv;
        if (has_weight) {
            val *= norm_weight[i];
        }
        shared[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 3: apply NeoX rotation on the first 2*half_rope dimensions
    //
    // NeoX convention: pairs (d[pair_idx], d[pair_idx + half_dim])
    // Using head_dim as denominator to match mlx-lm's ProportionalRoPE.
    // -------------------------------------------------------------------------
    for (uint i = tid; i < half_rope; i += tg_size) {
        const float x0 = shared[i];
        const float x1 = shared[i + half_dim];

        const float dim_ratio = float(2 * i) / float(head_dim);
        float freq = float(pos) / pow(theta, dim_ratio);

        if (has_ff) {
            freq /= freq_factors[i];
        }

        const float cos_a = cos(freq);
        const float sin_a = sin(freq);

        output[base + i]            = x0 * cos_a - x1 * sin_a;
        output[base + i + half_dim] = x1 * cos_a + x0 * sin_a;
    }

    // Pass through non-rotated dimensions:
    // [half_rope..half_dim) in first half and [half_dim+half_rope..head_dim) in second half
    for (uint i = tid; i < half_dim - half_rope; i += tg_size) {
        uint src = half_rope + i;
        output[base + src]            = shared[src];
        output[base + src + half_dim] = shared[src + half_dim];
    }
}
