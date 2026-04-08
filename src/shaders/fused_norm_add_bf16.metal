#include <metal_stdlib>
using namespace metal;

/// Fused RMS normalization + residual addition (bfloat16).
///
/// Computes Gemma4's post-attention / post-FFN ordering:
///   normed[i] = rms_norm(input[i], weight[i], eps)
///   output[i] = residual[i] + normed[i]
///
/// Fuses two separate dispatches (rms_norm_bf16 + elementwise_add_bf16) into
/// one kernel launch per transformer sub-layer.  Saves 60-120 dispatches across
/// Gemma4's 30 layers.
///
/// Buffer layout:
///   buffer(0): residual — bfloat [rows * dim]   residual stream (unmodified)
///   buffer(1): input    — bfloat [rows * dim]   sublayer output (to normalize)
///   buffer(2): weight   — bfloat [dim]          RMS norm learned scale
///   buffer(3): output   — bfloat [rows * dim]   residual + normed result
///   buffer(4): dim      — uint
///   buffer(5): rows     — uint
///   buffer(6): eps      — float
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row
/// Grid       : (rows, 1, 1)
/// Shared mem : tg_size * sizeof(float) for the sum-of-squares reduction

kernel void fused_norm_add_bf16(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* input    [[buffer(1)]],
    device const bfloat* weight   [[buffer(2)]],
    device bfloat*       output   [[buffer(3)]],
    constant uint&       dim      [[buffer(4)]],
    constant uint&       rows     [[buffer(5)]],
    constant float&      eps      [[buffer(6)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (row_id >= rows) { return; }

    const uint base = row_id * dim;

    // -------------------------------------------------------------------------
    // Phase 1: accumulate partial sum-of-squares over input (not the sum).
    //
    // We normalize `input` alone — the residual is added after.
    // -------------------------------------------------------------------------
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = static_cast<float>(input[base + i]);
        partial_sq += v * v;
    }

    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction to obtain total sum-of-squares.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rms_inv = rsqrt(mean(input^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // -------------------------------------------------------------------------
    // Phase 2: normalize input, apply weight, add residual, store output.
    //
    //   normed[i] = float(input[i]) * rms_inv * float(weight[i])
    //   output[i] = bfloat(float(residual[i]) + normed[i])
    // -------------------------------------------------------------------------
    for (uint i = tid; i < dim; i += tg_size) {
        const float normed = static_cast<float>(input[base + i])
                           * rms_inv
                           * static_cast<float>(weight[i]);
        output[base + i] = bfloat(static_cast<float>(residual[base + i]) + normed);
    }
}

/// Fused RMS normalization (no weight) + residual addition (bfloat16).
///
/// Computes:
///   normed[i] = input[i] * rsqrt(mean(input^2) + eps)   (no weight scale)
///   output[i] = residual[i] + normed[i]
///
/// Used for V-head norms in Gemma4 that have no learned scale parameter.
///
/// Buffer layout:
///   buffer(0): residual — bfloat [rows * dim]
///   buffer(1): input    — bfloat [rows * dim]
///   buffer(2): output   — bfloat [rows * dim]
///   buffer(3): dim      — uint
///   buffer(4): rows     — uint
///   buffer(5): eps      — float
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row
/// Grid       : (rows, 1, 1)
/// Shared mem : tg_size * sizeof(float) for the reduction

kernel void fused_norm_add_no_weight_bf16(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* input    [[buffer(1)]],
    device bfloat*       output   [[buffer(2)]],
    constant uint&       dim      [[buffer(3)]],
    constant uint&       rows     [[buffer(4)]],
    constant float&      eps      [[buffer(5)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (row_id >= rows) { return; }

    const uint base = row_id * dim;

    // Phase 1: accumulate partial sum-of-squares over input.
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = static_cast<float>(input[base + i]);
        partial_sq += v * v;
    }

    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize (no weight), add residual, store output.
    for (uint i = tid; i < dim; i += tg_size) {
        const float normed = static_cast<float>(input[base + i]) * rms_inv;
        output[base + i] = bfloat(static_cast<float>(residual[base + i]) + normed);
    }
}
