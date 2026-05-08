#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-11h-b — Causal depthwise 1D convolution kernels for
/// the GpuTape autograd pipeline.
///
/// Distinct from `ssm_conv.metal` which (a) fuses a SiLU activation
/// and (b) takes a state buffer for autoregressive decode.  This
/// kernel family is for TRAINING-MODE forward + backward:
///   * No state (zero-pad on the past — first K-1 outputs see
///     fewer-than-K input taps).
///   * No SiLU fusion (silu is a separate `OpKind` in GpuTape; this
///     keeps backward derivation clean and aligned with the
///     "one OpKind = one math primitive" composition principle
///     established by iter-11h-a).
///
/// ## Math
///
/// Forward (per (t, c)):
///   y[t, c] = Σ_{k=0..K-1, t+k-(K-1) >= 0} kernel_w[c, k] · x[t+k-(K-1), c]
///
/// Equivalently with `i = t+k-(K-1)`:
///   y[t, c] = Σ_{i=max(0, t-K+1)..t} kernel_w[c, t-i+(K-1)] · x[i, c]
///
/// Backward dx (per (i, c)):
///   dx[i, c] = Σ_{k=0..K-1, t=i+(K-1)-k in [0, n)} kernel_w[c, k] · dy[t, c]
///
/// Backward dw (per (c, k)):
///   dw[c, k] = Σ_{t=K-1-k..n-1} x[t+k-(K-1), c] · dy[t, c]
///            = Σ_{i=0..n-1-(K-1-k)} x[i, c] · dy[i+(K-1)-k, c]
///
/// ## Layout
///
///   x : `[n_tokens, channels]` row-major: `x[t][c]` at `t*channels + c`.
///   y : `[n_tokens, channels]` same layout as x.
///   kernel_w : `[channels, K]` row-major: `w[c][k]` at `c*K + k`.
///
/// (Matches `ssm_conv.metal`'s `kernel_w` layout convention; differs
/// in the activation/state input plane shape because we drop n_seqs
/// — this kernel handles a single sequence at a time, n_seqs=1
/// implicit.  Multi-seq batching can wrap with an outer dispatch.)

// ──────────────────────────────────────────────────────────────────
// Forward: y[t, c] = Σ_k kernel_w[c, k] · x_ext[t+k-(K-1), c]
//   x_ext zero-pads for indices < 0.
// ──────────────────────────────────────────────────────────────────
kernel void conv1d_depthwise_causal_forward_f32(
    device const float *x         [[buffer(0)]],
    device const float *kernel_w  [[buffer(1)]],
    device float       *y         [[buffer(2)]],
    device const uint  *params    [[buffer(3)]],   // [n_tokens, channels, K]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint n_tokens = params[0];
    const uint channels = params[1];
    const uint K        = params[2];

    const uint t = tid.x;
    const uint c = tid.y;
    if (t >= n_tokens || c >= channels) return;

    float sum = 0.0f;
    // k_min: smallest k such that t+k-(K-1) >= 0, i.e. k >= K-1-t.
    const uint k_min = (t + 1u >= K) ? 0u : (K - 1u - t);
    for (uint k = k_min; k < K; ++k) {
        const uint i = t + k - (K - 1u);
        sum += kernel_w[c * K + k] * x[i * channels + c];
    }
    y[t * channels + c] = sum;
}

// ──────────────────────────────────────────────────────────────────
// Backward dx: dx[i, c] = Σ_{k where t=i+(K-1)-k is valid} kernel_w[c, k] · dy[t, c]
//   t = i + (K-1) - k in [0, n_tokens) ⇒
//     k in [max(0, i+(K-1) - (n_tokens-1)), min(K-1, i+(K-1))]
//                          ↑ = max(0, i-K+1+n_tokens-1)... wait i+(K-1) - (n_tokens-1) = i-n_tokens+K
//                      so k_min = max(0, i+K - n_tokens) (drop terms where t >= n_tokens)
//                      and k_max = min(K-1, i+(K-1)) = K-1 since i >= 0 ⇒ i+(K-1) >= K-1
// ──────────────────────────────────────────────────────────────────
kernel void conv1d_depthwise_causal_backward_dx_f32(
    device const float *dy        [[buffer(0)]],
    device const float *kernel_w  [[buffer(1)]],
    device float       *dx        [[buffer(2)]],
    device const uint  *params    [[buffer(3)]],   // [n_tokens, channels, K]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint n_tokens = params[0];
    const uint channels = params[1];
    const uint K        = params[2];

    const uint i = tid.x;
    const uint c = tid.y;
    if (i >= n_tokens || c >= channels) return;

    float sum = 0.0f;
    // k_min = max(0, i+K - n_tokens). Branch-free via conditional.
    const uint k_min = (i + K > n_tokens) ? (i + K - n_tokens) : 0u;
    // k_max = K-1 always when i >= 0 (always true for unsigned).
    for (uint k = k_min; k < K; ++k) {
        const uint t = i + (K - 1u) - k;
        sum += kernel_w[c * K + k] * dy[t * channels + c];
    }
    dx[i * channels + c] = sum;
}

// ──────────────────────────────────────────────────────────────────
// Backward dw: dw[c, k] = Σ_{t=K-1-k..n-1} x[t+k-(K-1), c] · dy[t, c]
//   Substituting i = t+k-(K-1):
//   dw[c, k] = Σ_{i=0..n-1-(K-1-k)} x[i, c] · dy[i+(K-1)-k, c]
//                   = Σ_{i=0..n-1-(K-1-k)} x[i, c] · dy[t, c]   where t = i+K-1-k
//
// Range guard: we need t < n_tokens.  For each (c, k), iterate i from
// 0 upward; stop when t = i+K-1-k >= n_tokens.
// ──────────────────────────────────────────────────────────────────
kernel void conv1d_depthwise_causal_backward_dw_f32(
    device const float *x         [[buffer(0)]],
    device const float *dy        [[buffer(1)]],
    device float       *dw        [[buffer(2)]],
    device const uint  *params    [[buffer(3)]],   // [n_tokens, channels, K]
    uint2 tid [[thread_position_in_grid]]
) {
    const uint n_tokens = params[0];
    const uint channels = params[1];
    const uint K        = params[2];

    const uint c = tid.x;
    const uint k = tid.y;
    if (c >= channels || k >= K) return;

    float sum = 0.0f;
    // t = i + (K-1) - k starts at i=0 → t = K-1-k.  Stop at t = n_tokens-1.
    // i_max = n_tokens - 1 - (K-1-k) = n_tokens - K + k.
    // Guard against underflow when n_tokens < K - k.
    if (n_tokens + k < K) {
        dw[c * K + k] = 0.0f;
        return;
    }
    const uint i_max = n_tokens + k - K;  // inclusive bound
    for (uint i = 0u; i <= i_max; ++i) {
        const uint t = i + (K - 1u) - k;
        sum += x[i * channels + c] * dy[t * channels + c];
    }
    dw[c * K + k] = sum;
}
