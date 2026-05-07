#include <metal_stdlib>
using namespace metal;

/// Per-row sum reduction along the last dimension of a 2-D tensor.
///
///   `output[b] = Σ_j input[b, j]`   for b ∈ [0, rows)
///
/// One threadgroup per row; tree reduction over the columns.  Pattern
/// matches softmax.metal forward (see also softmax_backward.metal).
///
/// Buffer layout:
///   buffer(0): input  — float [rows, cols]
///   buffer(1): output — float [rows]
///   buffer(2): params — float2: (cols_f, 0)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void row_sum_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const float *params [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const uint cols = uint(params[0]);
    const uint base = row_idx * cols;

    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        local_sum += input[base + i];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row_idx] = shared[0];
    }
}

/// Backward for row_sum: dx[b, i] = d_out[b]  (broadcast-along-cols).
///
/// Buffer layout:
///   buffer(0): d_out  — float [rows]   (upstream gradient)
///   buffer(1): dx     — float [rows, cols]   (output)
///   buffer(2): params — float2: (cols_f, 0)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)
///
/// Each thread writes a strided subset of the dx[b, :] row.

kernel void row_sum_backward_f32(
    device const float *d_out  [[buffer(0)]],
    device float       *dx     [[buffer(1)]],
    device const float *params [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]]
) {
    const uint cols = uint(params[0]);
    const uint base = row_idx * cols;
    const float v = d_out[row_idx];
    for (uint i = tid; i < cols; i += tg_size) {
        dx[base + i] = v;
    }
}
