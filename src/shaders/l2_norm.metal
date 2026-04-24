#include <metal_stdlib>
using namespace metal;

// L2 Normalization kernel.
//
// Computes: output = x / sqrt(sum(x^2) + eps)
// The sum is computed over the last dimension (per-row).
//
// Spec source: ADR-013 Decision 3. Formula derived from the mathematical
// definition of L2 normalization (x / ||x||_2, with epsilon for stability).
// Used by Gated DeltaNet on Q and K after conv1d state update
// (delta-net-base.cpp:320-321 references; no code copied).
//
// Buffer layout:
//   buffer(0): input   - array of shape [rows, dim]  (element dtype varies)
//   buffer(1): output  - array of shape [rows, dim]
//   buffer(2): params  - float2: (eps, dim_f)
//
// Threadgroup: (threadgroup_size, 1, 1) - one threadgroup per row
// Grid threadgroups: (rows, 1, 1)
//
// Accumulation is always performed in f32 for numerical stability, regardless
// of the input dtype (matches ADR-011 convention).

kernel void l2_norm_f32(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    // Phase 1: compute partial sum of squares in f32.
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i];
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // L2 norm uses sum-of-squares (not mean-of-squares like RMS norm).
    const float inv = rsqrt(shared[0] + eps);

    // Phase 2: write normalized output.
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * inv;
    }
}

kernel void l2_norm_f16(
    device const half  *input   [[buffer(0)]],
    device half        *output  [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = float(input[base + i]);
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shared[0] + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = half(float(input[base + i]) * inv);
    }
}

kernel void l2_norm_bf16(
    device const bfloat *input   [[buffer(0)]],
    device bfloat       *output  [[buffer(1)]],
    device const float  *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = float(input[base + i]);
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shared[0] + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = bfloat(float(input[base + i]) * inv);
    }
}
