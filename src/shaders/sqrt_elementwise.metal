#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-misc-3 — elementwise sqrt + backward.
///
/// Forward:  y[i] = sqrt(x[i])
/// Backward: dx[i] = dy[i] / (2 · y[i])  (uses forward output;
///                                        avoids recomputing 1/(2√x))
///
/// Caller responsibility: x[i] >= 0 for all i (sqrt of negative
/// floats returns NaN per IEEE 754).  Backward produces NaN at
/// x[i] = 0 (division by zero) — caller should add eps if needed.

kernel void sqrt_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const uint  *params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    output[tid] = metal::sqrt(input[tid]);
}

/// Backward: dx[i] = dy[i] / (2 · y[i]).
kernel void sqrt_backward_f32(
    device const float *y      [[buffer(0)]],   // forward output
    device const float *dy     [[buffer(1)]],
    device float       *dx     [[buffer(2)]],
    device const uint  *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    dx[tid] = dy[tid] / (2.0f * y[tid]);
}
