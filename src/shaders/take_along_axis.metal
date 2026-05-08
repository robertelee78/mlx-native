#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-e1 — take_along_axis (gather) + scatter-backward
/// kernels for the GpuTape autograd pipeline.
///
/// Used by MoE router (mlx-lm/qwen3_next.py:Qwen3NextSparseMoeBlock):
///   gates  = softmax(linear(x))      // [n_tokens, n_experts]
///   inds   = argpartition(gates, k)  // [n_tokens, top_k] (u32)
///   scores = take_along_axis(gates, inds, axis=-1)   // [n_tokens, top_k]
///
/// Forward (per (r, j)):
///   y[r, j] = x[r, indices[r, j]]
///
/// Backward (zero-init dx, per (r, j) scatter):
///   dx[r, indices[r, j]] += dy[r, j]
///
/// Top-K gives distinct indices per row → no atomic collisions. We
/// rely on the caller to pre-zero dx (alloc_buffer zero-fills per
/// MlxDevice contract).
///
/// Layout:
///   x       : `[rows, cols]` row-major f32
///   indices : `[rows, k]` row-major u32
///   y       : `[rows, k]` row-major f32

kernel void take_along_axis_f32(
    device const float *x       [[buffer(0)]],
    device const uint  *indices [[buffer(1)]],
    device float       *y       [[buffer(2)]],
    device const uint  *params  [[buffer(3)]],   // [rows, cols, k]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint rows = params[0];
    const uint cols = params[1];
    const uint k    = params[2];

    const uint r = tid.x;
    const uint j = tid.y;
    if (r >= rows || j >= k) return;

    const uint idx = indices[r * k + j];
    if (idx >= cols) {
        // Out-of-bounds index — write 0 to surface the bug rather
        // than read OOB.  Caller validates indices but this is a
        // defensive guard inside the kernel.
        y[r * k + j] = 0.0f;
        return;
    }
    y[r * k + j] = x[r * cols + idx];
}

/// Scatter `dy` into `dx` at the positions specified by `indices`.
/// `dx` must be PRE-ZEROED by caller (alloc_buffer satisfies this).
///
/// One thread per (r, j); writes `dx[r, indices[r, j]] = dy[r, j]`.
/// Top-K indices are distinct within a row → no race even without
/// atomics.
kernel void take_along_axis_backward_f32(
    device const float *dy      [[buffer(0)]],
    device const uint  *indices [[buffer(1)]],
    device float       *dx      [[buffer(2)]],
    device const uint  *params  [[buffer(3)]],   // [rows, cols, k]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint rows = params[0];
    const uint cols = params[1];
    const uint k    = params[2];

    const uint r = tid.x;
    const uint j = tid.y;
    if (r >= rows || j >= k) return;

    const uint idx = indices[r * k + j];
    if (idx >= cols) return;
    dx[r * cols + idx] = dy[r * k + j];
}
