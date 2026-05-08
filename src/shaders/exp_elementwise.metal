#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-c1 — elementwise exponential.  Building block for
/// the GatedDeltaNet recurrence's `alpha = exp(-g[t])` state-decay
/// factor (mlx-lm/qwen3_5.py:GatedDeltaNet.__call__).
///
/// Forward:  y[i] = exp(x[i])
/// Backward: dx[i] = dy[i] · exp(x[i]) = dy[i] · y[i]
///
/// We expose y as the SECOND backward input (caller passes the
/// forward output back through to the backward kernel) — saves
/// recomputing exp during backward.

kernel void exp_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const uint  *params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    output[tid] = metal::exp(input[tid]);
}

/// Backward: `dx = dy * y` where y is the forward output.  Caller
/// must pass the FORWARD OUTPUT (not the input) — this is the
/// canonical autograd pattern that avoids recomputation.
kernel void exp_backward_f32(
    device const float *y      [[buffer(0)]],   // forward output
    device const float *dy     [[buffer(1)]],
    device float       *dx     [[buffer(2)]],
    device const uint  *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    dx[tid] = dy[tid] * y[tid];
}
