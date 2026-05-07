#include <metal_stdlib>
using namespace metal;

/// Elementwise SiLU (swish) forward.
///
///   silu(x) = x · sigmoid(x) = x / (1 + exp(-x))
///
/// Buffer layout:
///   buffer(0): input  — float[n]
///   buffer(1): output — float[n]
///   buffer(2): params — uint[1]: n
///
/// Grid: 1D threads across n.
kernel void silu_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const uint  *params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float x = input[tid];
    const float s = 1.0f / (1.0f + metal::exp(-x));
    output[tid] = x * s;
}

/// Elementwise SiLU backward.
///
///   silu'(x) = sigmoid(x) + x · sigmoid(x) · (1 − sigmoid(x))
///            = sigmoid(x) · (1 + x · (1 − sigmoid(x)))
///   dx[i]    = dy[i] · silu'(x[i])
///
/// `x` is the FORWARD INPUT (not the forward output).
///
/// Buffer layout:
///   buffer(0): x      — float[n]    (forward input)
///   buffer(1): dy     — float[n]    (upstream gradient)
///   buffer(2): dx     — float[n]    (output gradient)
///   buffer(3): params — uint[1]: n
///
/// Grid: 1D threads across n.
kernel void silu_backward_f32(
    device const float *x      [[buffer(0)]],
    device const float *dy     [[buffer(1)]],
    device float       *dx     [[buffer(2)]],
    device const uint  *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float xv = x[tid];
    const float s = 1.0f / (1.0f + metal::exp(-xv));
    // silu'(x) = s · (1 + x · (1 − s))
    const float deriv = s * (1.0f + xv * (1.0f - s));
    dx[tid] = dy[tid] * deriv;
}
