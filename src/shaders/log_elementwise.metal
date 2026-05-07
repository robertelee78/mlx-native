#include <metal_stdlib>
using namespace metal;

/// Elementwise natural logarithm: `output[i] = log(input[i])`.
///
/// Used by reverse-mode autograd in downstream crates (hf2q ADR-020
/// Track 1 Linear-aware backward).  Caller must ensure input is
/// strictly positive — the kernel does not check; `log(x ≤ 0)` will
/// produce NaN or -inf per IEEE 754.
///
/// Buffer layout:
///   buffer(0): input  — float array of shape [n]
///   buffer(1): output — float array of shape [n] (same n)
///
/// Threadgroup: (threadgroup_size, 1, 1).  One thread per element.

kernel void log_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    uint                tid     [[thread_position_in_grid]]
) {
    output[tid] = log(input[tid]);
}

/// Backward pass for elementwise log.  Given the FORWARD INPUT `x`
/// (NOT the forward output — log(x) — because we need the original
/// values to divide by) and upstream gradient `dy`, computes
///
///   dx[i] = dy[i] / x[i]
///
/// matching the analytical derivative ∂log(x)/∂x = 1/x.
///
/// Buffer layout:
///   buffer(0): x  — forward input float array of shape [n]
///   buffer(1): dy — upstream gradient float array of shape [n]
///   buffer(2): dx — output float array of shape [n]
///
/// Threadgroup: (threadgroup_size, 1, 1).  One thread per element.

kernel void log_backward_f32(
    device const float *x       [[buffer(0)]],
    device const float *dy      [[buffer(1)]],
    device float       *dx      [[buffer(2)]],
    uint                tid     [[thread_position_in_grid]]
) {
    dx[tid] = dy[tid] / x[tid];
}
