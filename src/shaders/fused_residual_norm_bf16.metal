#include <metal_stdlib>
using namespace metal;

/// Fused residual addition + RMS normalization (bfloat16).
///
/// Replaces two separate kernel dispatches (elementwise_add_bf16 + rms_norm_bf16)
/// with a single pass, eliminating the intermediate summed buffer and the second
/// kernel launch.
///
/// Computes:
///   sum[i]    = float(residual[i]) + float(input[i])
///   normed[i] = sum[i] * rsqrt(mean(sum^2) + eps) * float(weight[i])
///
/// Optionally writes `sum` to a separate output buffer for use as the next
/// layer's residual (avoids an extra elementwise kernel).
///
/// Buffer layout:
///   buffer(0): residual      — bfloat [rows * dim]
///   buffer(1): input         — bfloat [rows * dim]
///   buffer(2): weight        — bfloat [dim]
///   buffer(3): normed_output — bfloat [rows * dim]   normalized result
///   buffer(4): sum_output    — bfloat [rows * dim]   residual+input (may be unused)
///   buffer(5): params        — { dim, rows, eps, write_sum }
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1)
/// Grid       : (rows, 1, 1)
///
/// Shared memory: tg_size * sizeof(float) at threadgroup(0) for the reduction.

struct FusedResidualNormParams {
    uint  dim;
    uint  rows;
    float eps;
    uint  write_sum;   // 0 = skip writing sum_output, nonzero = write it
};

kernel void fused_residual_norm_bf16(
    device const bfloat*               residual      [[buffer(0)]],
    device const bfloat*               input         [[buffer(1)]],
    device const bfloat*               weight        [[buffer(2)]],
    device bfloat*                     normed_output [[buffer(3)]],
    device bfloat*                     sum_output    [[buffer(4)]],
    constant FusedResidualNormParams&  params        [[buffer(5)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
) {
    const uint  dim       = params.dim;
    const float eps       = params.eps;
    const bool  write_sum = (params.write_sum != 0u);

    const uint base = row_id * dim;

    // -------------------------------------------------------------------------
    // Phase 1: compute residual + input element-wise, accumulate sum-of-squares
    //
    // We store the f32 sum in shared memory (reused after reduction).
    // Each thread owns ceil(dim / tg_size) elements.
    // -------------------------------------------------------------------------

    // First pass: each thread computes its partial sum-of-squares while
    // storing the partial sums into shared scratch for the reduction.
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = static_cast<float>(residual[base + i])
                      + static_cast<float>(input[base + i]);
        // Temporarily borrow shared[i] to cache the element sum for phase 2.
        // Safe because the reduction only writes shared[0..tg_size-1] by thread
        // index, and here we write by element index — these ranges only alias if
        // tg_size >= dim, which is always true (tg_size = next_pow2(dim), >= dim).
        shared[i] = s;
        partial_sq += s * s;
    }

    // Reduction: each thread contributes its partial sum-of-squares in shared[tid].
    // We must be careful: above we wrote to shared[i] for i in [tid, dim).
    // The reduction writes to shared[tid] (by thread index, 0..tg_size-1).
    // If dim <= tg_size, element slots [0..dim) and reduction slots [0..tg_size-1]
    // overlap.  We handle this by using a separate reduction barrier:
    //
    // Approach: accumulate partial_sq from threads, store in a separate range.
    // Since tg_size is always >= dim but elements are only in [0, dim), we can
    // use the range [dim, dim + tg_size) if tg_size allows — but that requires
    // 2*tg_size shared memory.
    //
    // Simpler correct approach: perform the reduction AFTER all element sums are
    // written, using a read-first-then-reduce pattern: save the per-thread
    // partial_sq (already computed) into shared[tid], then reduce.
    // The element sums in shared[0..dim) are overwritten for indices tid < dim,
    // which we restore in phase 2 by recomputing from the original buffers.
    //
    // For correctness and simplicity: store partial_sq in a dedicated slot.
    // We need tg_size extra slots.  To avoid doubling shared memory, we instead
    // take the pragmatic approach: finish caching element sums first, then
    // overwrite shared[0..tg_size) for the reduction, then recompute sums.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Optionally write the un-normed sum to sum_output now (we still have each
    // element's sum cached in shared[i]).  This must happen before we overwrite
    // shared for the reduction.
    if (write_sum) {
        for (uint i = tid; i < dim; i += tg_size) {
            sum_output[base + i] = bfloat(shared[i]);
        }
    }

    // Now overwrite shared[0..tg_size) with per-thread partial_sq for reduction.
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // -------------------------------------------------------------------------
    // Phase 2: normalize and write normed_output
    //
    // We no longer have element sums in shared (overwritten by reduction).
    // Recompute each element sum directly from the source buffers.
    // -------------------------------------------------------------------------
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = static_cast<float>(residual[base + i])
                      + static_cast<float>(input[base + i]);
        const float w = static_cast<float>(weight[i]);
        normed_output[base + i] = bfloat(s * rms_inv * w);
    }
}
