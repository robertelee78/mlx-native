#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-c2 — vector outer product `Y = lhs ⊗ rhs`.
/// Building block for `gated_delta_update`'s state-update term
/// `state += outer(delta, k)` (mlx-lm/gated_delta.py:_gated_delta_step_ops).
///
/// Distinct from matmul: matmul kernel has a 32-element floor on each
/// dim (M, N, K all ≥ 32 for backward dW dispatch); outer products
/// have inner-dim = 1, so they fall below the floor.  This kernel
/// family handles the 1-D × 1-D = 2-D case directly.
///
/// Layout:
///   lhs : `[N]` row-major f32
///   rhs : `[M]` row-major f32
///   y   : `[N, M]` row-major f32:  y[i, j] = lhs[i] · rhs[j]
///
/// Math:
///   Forward: y[i, j] = lhs[i] · rhs[j]
///   Backward dlhs[i] = Σ_j dy[i, j] · rhs[j]   (row sum of dy * rhs[None, :])
///   Backward drhs[j] = Σ_i dy[i, j] · lhs[i]   (col sum of dy * lhs[:, None])

// ──────────────────────────────────────────────────────────────────
// Forward
// ──────────────────────────────────────────────────────────────────
kernel void outer_product_f32(
    device const float *lhs    [[buffer(0)]],
    device const float *rhs    [[buffer(1)]],
    device float       *y      [[buffer(2)]],
    device const uint  *params [[buffer(3)]],   // [N, M]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint N = params[0];
    const uint M = params[1];
    const uint i = tid.x;
    const uint j = tid.y;
    if (i >= N || j >= M) return;
    y[i * M + j] = lhs[i] * rhs[j];
}

// ──────────────────────────────────────────────────────────────────
// Backward dlhs: dlhs[i] = Σ_j dy[i, j] · rhs[j]
// ──────────────────────────────────────────────────────────────────
kernel void outer_product_backward_lhs_f32(
    device const float *dy     [[buffer(0)]],
    device const float *rhs    [[buffer(1)]],
    device float       *dlhs   [[buffer(2)]],
    device const uint  *params [[buffer(3)]],   // [N, M]
    uint tid [[thread_position_in_grid]]
) {
    const uint N = params[0];
    const uint M = params[1];
    const uint i = tid;
    if (i >= N) return;
    float acc = 0.0f;
    for (uint j = 0; j < M; ++j) {
        acc += dy[i * M + j] * rhs[j];
    }
    dlhs[i] = acc;
}

// ──────────────────────────────────────────────────────────────────
// Backward drhs: drhs[j] = Σ_i dy[i, j] · lhs[i]
// ──────────────────────────────────────────────────────────────────
kernel void outer_product_backward_rhs_f32(
    device const float *dy     [[buffer(0)]],
    device const float *lhs    [[buffer(1)]],
    device float       *drhs   [[buffer(2)]],
    device const uint  *params [[buffer(3)]],   // [N, M]
    uint tid [[thread_position_in_grid]]
) {
    const uint N = params[0];
    const uint M = params[1];
    const uint j = tid;
    if (j >= M) return;
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        acc += dy[i * M + j] * lhs[i];
    }
    drhs[j] = acc;
}
