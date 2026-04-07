#include <metal_stdlib>
using namespace metal;

/// Numerically stable softmax along the last dimension.
///
/// Algorithm:
///   1. Find max(x) for each row (subtract for stability)
///   2. Compute exp(x - max) for each element
///   3. Sum the exponentials
///   4. Divide each exp by the sum
///
/// All accumulations use f32 for numerical stability even with f16 inputs.
///
/// Buffer layout:
///   buffer(0): input  — float array of shape [rows, cols]
///   buffer(1): output — float array of shape [rows, cols]
///   buffer(2): params — float2: (cols_f, 0)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void softmax_f32(
    device const float *input     [[buffer(0)]],
    device float       *output    [[buffer(1)]],
    device const float *params    [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const uint cols = uint(params[0]);
    const uint base = row_idx * cols;

    // Phase 1: find row max
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tg_size) {
        local_max = max(local_max, input[base + i]);
    }

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: compute exp(x - max) and accumulate sum
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        const float e = exp(input[base + i] - row_max);
        output[base + i] = e;  // store intermediate exp values
        local_sum += e;
    }

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize
    const float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < cols; i += tg_size) {
        output[base + i] *= inv_sum;
    }
}

kernel void softmax_f16(
    device const half  *input     [[buffer(0)]],
    device half        *output    [[buffer(1)]],
    device const float *params    [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const uint cols = uint(params[0]);
    const uint base = row_idx * cols;

    // Phase 1: find row max (accumulate in f32)
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tg_size) {
        local_max = max(local_max, float(input[base + i]));
    }

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: exp(x - max) accumulated in f32, stored to a temporary
    // We reuse the output buffer to store the f32 exp values packed as f16,
    // but compute the sum in f32.
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        const float e = exp(float(input[base + i]) - row_max);
        output[base + i] = half(e);  // store intermediate
        local_sum += e;
    }

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize
    const float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < cols; i += tg_size) {
        // Re-read the f16 intermediate, promote to f32, normalize, store as f16
        output[base + i] = half(float(output[base + i]) * inv_sum);
    }
}
