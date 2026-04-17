#include <metal_stdlib>
using namespace metal;

/// Fused per-head RMS normalization + NeoX rotary position embedding (bfloat16).
///
/// Replaces two separate kernel dispatches (rms_norm_bf16 per-head + rope_neox_bf16)
/// with a single pass over each head's data, eliminating the intermediate normalized
/// buffer and the second kernel launch.
///
/// Layout:
///   Input : [n_heads * head_dim] bf16  — one token's Q or K after linear projection
///   Output: [n_heads * head_dim] bf16  — normalized and rotated
///
/// One threadgroup per head, threads cooperate on:
///   1. Parallel sum-of-squares reduction (shared f32 scratch)
///   2. Normalize every element with optional weight scale
///   3. NeoX-convention rotation on the first 2*half_rope_dim elements
///
/// Buffer layout:
///   buffer(0): input         — bfloat [n_heads * head_dim]
///   buffer(1): output        — bfloat [n_heads * head_dim]
///   buffer(2): norm_weight   — bfloat [head_dim] (or nullptr when has_weight=false)
///   buffer(3): cos_cache     — float  [half_rope_dim]  precomputed cos values
///   buffer(4): sin_cache     — float  [half_rope_dim]  precomputed sin values
///   buffer(5): params        — { head_dim, n_heads, half_rope_dim, eps, has_weight }
///
/// Threadgroup: (min(256, next_pow2(head_dim)), 1, 1)
/// Grid       : (n_heads, 1, 1)
///
/// Shared memory: tg_size * sizeof(float) at threadgroup(0) for the reduction.

struct FusedHeadNormRopeParams {
    uint  head_dim;
    uint  n_heads;
    uint  half_rope_dim;   // may be < head_dim/2 for partial rotary
    float eps;
    uint  has_weight;      // 0 = no weight scale (V-norm variant), nonzero = apply weight
};

kernel void fused_head_norm_rope_bf16(
    device const bfloat*                  input       [[buffer(0)]],
    device bfloat*                        output      [[buffer(1)]],
    device const bfloat*                  norm_weight [[buffer(2)]],
    device const float*                   cos_cache   [[buffer(3)]],
    device const float*                   sin_cache   [[buffer(4)]],
    constant FusedHeadNormRopeParams&     params      [[buffer(5)]],
    uint head_id  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared             [[threadgroup(0)]]
) {
    const uint head_dim     = params.head_dim;
    const uint half_rope    = params.half_rope_dim;
    const float eps         = params.eps;
    const bool has_weight   = (params.has_weight != 0u);

    const uint base = head_id * head_dim;

    // -------------------------------------------------------------------------
    // Phase 1: parallel sum-of-squares reduction
    // -------------------------------------------------------------------------
    float partial_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
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

    // All threads must complete the broadcast-read of shared[0] for rms_inv
    // BEFORE any thread (notably tid==0) overwrites shared[0] as part of
    // Phase 2's shared[i] = val write. See the matching comment in the f32
    // sibling; same race, same fix. Ref: docs/spike-batched-prefill-race-
    // rootcause.md (hf2q, 2026-04-16).
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2: normalize (optionally scale with weight), then apply NeoX RoPE
    //
    // NeoX convention: rotate pairs (d[i], d[i + half_rope]) for i < half_rope,
    // pass through elements in [2*half_rope, head_dim) unchanged.
    //
    // Strategy: each thread computes its normalized element(s), stores them in
    // the shared buffer (reusing the float scratch since the reduction is done),
    // then performs the rotation read/write from shared to avoid double-reads.
    // -------------------------------------------------------------------------

    // Reuse shared memory for storing normalized f32 values.
    // We need head_dim floats; tg_size is a power-of-two >= head_dim, so we have
    // enough space (tg_size >= head_dim by construction of the threadgroup size).
    //
    // Each thread fills its owned slots:
    for (uint i = tid; i < head_dim; i += tg_size) {
        float val = static_cast<float>(input[base + i]) * rms_inv;
        if (has_weight) {
            val *= static_cast<float>(norm_weight[i]);
        }
        shared[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply NeoX rotation on the first 2*half_rope dimensions.
    // Thread tid handles pair index tid (and tid + half_rope is the other element).
    for (uint i = tid; i < half_rope; i += tg_size) {
        const float x0  = shared[i];
        const float x1  = shared[i + half_rope];
        const float c   = cos_cache[i];
        const float s   = sin_cache[i];

        output[base + i]            = bfloat(x0 * c - x1 * s);
        output[base + i + half_rope] = bfloat(x1 * c + x0 * s);
    }

    // Pass through elements beyond the rotated region.
    const uint rope_end = 2u * half_rope;
    for (uint i = rope_end + tid; i < head_dim; i += tg_size) {
        output[base + i] = bfloat(shared[i]);
    }
}
