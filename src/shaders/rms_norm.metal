#include <metal_stdlib>
using namespace metal;

/// RMS Normalization kernel.
///
/// Computes: output = x * rsqrt(mean(x^2) + eps) * weight
/// The mean is computed over the last dimension.
///
/// Buffer layout:
///   buffer(0): input   — float array of shape [rows, dim]
///   buffer(1): weight  — float array of shape [dim]
///   buffer(2): output  — float array of shape [rows, dim]
///   buffer(3): params  — float2: (eps, dim_f)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void rms_norm_f32(
    device const float *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device float       *output    [[buffer(2)]],
    device const float *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: compute partial sum of squares
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize and apply weight
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms_inv * weight[i];
    }
}

kernel void rms_norm_f16(
    device const half  *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device half        *output    [[buffer(2)]],
    device const float *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize, compute in f32, store as f16
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        output[base + i] = half(val * rms_inv * weight[i]);
    }
}
