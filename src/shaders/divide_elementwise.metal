#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-misc-1 — elementwise division `y = a / b`.
///
/// Forward:  y[i] = a[i] / b[i]
/// Backward: da[i] = dy[i] / b[i]
///           db[i] = -dy[i] · y[i] / b[i]   (uses forward output to
///                                             avoid recomputing a/b²)
///
/// Cleaner than the `exp(-log(x))` reciprocal trick used in
/// iter-11h-e2's renorm path; works for negative `b` too (whereas
/// `log(b)` is undefined for `b < 0`).
///
/// All buffers same length n.

kernel void divide_f32(
    device const float *a       [[buffer(0)]],
    device const float *b       [[buffer(1)]],
    device float       *y       [[buffer(2)]],
    device const uint  *params  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    y[tid] = a[tid] / b[tid];
}

/// Backward: produces both da and db in one dispatch.
///   da[i] = dy[i] / b[i]
///   db[i] = -dy[i] · y[i] / b[i]
kernel void divide_backward_f32(
    device const float *b       [[buffer(0)]],
    device const float *y       [[buffer(1)]],   // forward output
    device const float *dy      [[buffer(2)]],
    device float       *da      [[buffer(3)]],
    device float       *db      [[buffer(4)]],
    device const uint  *params  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float bi = b[tid];
    const float dyi = dy[tid];
    da[tid] = dyi / bi;
    db[tid] = -dyi * y[tid] / bi;
}
