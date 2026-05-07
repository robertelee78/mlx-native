#include <metal_stdlib>
using namespace metal;

/// Backward pass for row-wise softmax: given y = softmax(x) and dy
/// (upstream gradient w.r.t. y), compute dx (gradient w.r.t. x).
///
/// Math:
///   dx[b, i] = y[b, i] · (dy[b, i] − Σ_j y[b, j] · dy[b, j])
///
/// Algorithm:
///   Phase 1: each thread accumulates a partial sum of y[i] · dy[i]
///            for its strided column subset.
///   Phase 2: tree reduction across the threadgroup → row_dot.
///   Phase 3: each thread writes dx[i] = y[i] · (dy[i] − row_dot)
///            for its strided columns.
///
/// All accumulations use f32 for numerical stability even with f16
/// inputs.  Matches the convention of softmax.metal forward.
///
/// Buffer layout:
///   buffer(0): y       — float array of shape [rows, cols]   (the softmax output)
///   buffer(1): dy      — float array of shape [rows, cols]   (upstream grad)
///   buffer(2): dx      — float array of shape [rows, cols]   (output)
///   buffer(3): params  — float2: (cols_f, 0)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void softmax_backward_f32(
    device const float *y         [[buffer(0)]],
    device const float *dy        [[buffer(1)]],
    device float       *dx        [[buffer(2)]],
    device const float *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const uint cols = uint(params[0]);
    const uint base = row_idx * cols;

    // Phase 1: accumulate y[i] * dy[i] in strided fashion.
    float local_dot = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        local_dot += y[base + i] * dy[base + i];
    }
    shared[tid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree reduction → shared[0] = row_dot.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_dot = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: write dx[i] = y[i] * (dy[i] - row_dot).
    for (uint i = tid; i < cols; i += tg_size) {
        dx[base + i] = y[base + i] * (dy[base + i] - row_dot);
    }
}
