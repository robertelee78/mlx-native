#include <metal_stdlib>
using namespace metal;

// ADR-034 task #90 Step 4b (2026-05-21) — Per-position conv-state capture
// kernel for K=N speculative decoding rollback on hybrid Qwen 3.5/3.6.
//
// Companion to `ssm_conv_forward_f32` (existing kernel A in ssm_conv.metal)
// + the recurrent-state capture kernel landed at mlx-native e6fce4f.
//
// Port of MTPLX `gdn_capture.py:61-125` (`_make_linear_conv1d_kernel`).
//
// # Purpose
//
// The existing `ssm_conv_forward_f32` writes only the FINAL conv-state
// (after processing all n_tokens). On partial-reject in K=N spec-decode,
// the caller needs the state AFTER token `accepted_idx` (some intermediate
// position), not the final state. This kernel writes the per-position
// state at EVERY token slot t ∈ [0, n_tokens), so the caller can roll
// back to `conv_capture[..., accepted_idx, ...]`.
//
// # Output buffer layout
//
//   conv_capture[s, t, i, c]   shape [n_seqs, n_tokens, K-1, channels]
//   offset:  s * n_tokens*(K-1)*channels
//          + t * (K-1)*channels
//          + i * channels
//          + c
//
// Innermost `channels` matches the existing state buffer layout
// `state[s, c, i]` (channels-major with K-1 stride 1) — note the
// **per-position** capture layout has channels innermost with i as
// the next dim, mirroring MTPLX's `conv_states[(b * T + t) * Keep + k) * ConvDim + c_idx]`.
//
// # State semantics
//
// conv_capture[s, t, k, c] = the value that would be at state-slot `k`
// of `new_state[s, c, k]` if the conv had been called with n_tokens
// stopping at `t+1` instead of the full batch. Equivalently, the
// (k+1)-th entry of a sliding window of K inputs ending at position t.
//
// # Determinism contract
//
// `conv_capture[..., n_tokens-1, ...]` MUST equal `new_state[..., ...]`
// from the existing `ssm_conv_state_update_f32` kernel (the existing
// state-update kernel produces the final state; the capture kernel
// produces it AND the per-token intermediates).
//
// # Buffer bindings
//
//   buffer(0): x              f32  [c, t, s]  (input)
//   buffer(1): kernel_w       f32  [k, c]     (conv weights)
//   buffer(2): old_state      f32  [i, c, s]  (initial K-1 state)
//   buffer(3): y              f32  [c, t, s]  (output, same as ssm_conv_forward)
//   buffer(4): conv_capture   f32  [s, t, k, c]  (NEW: per-position state)
//   buffer(5): params         uint[4]  (channels, n_tokens, n_seqs, k_width)
//
// # Compile-time semantics match ssm_conv_forward_f32
//
// SiLU is applied to y (same as existing kernel). conv_capture is the
// raw (un-SiLU'd) recurrent state — that matches the existing
// `new_state` buffer contract: state is the raw input window, NOT
// SiLU-activated.

kernel void ssm_conv_capture_forward_f32(
    device const float *x              [[buffer(0)]],
    device const float *kernel_w       [[buffer(1)]],
    device const float *old_state      [[buffer(2)]],
    device       float *y              [[buffer(3)]],
    device       float *conv_capture   [[buffer(4)]],
    device const uint  *params         [[buffer(5)]],
    uint3 tid                          [[thread_position_in_grid]]
) {
    const uint channels  = params[0];
    const uint n_tokens  = params[1];
    const uint n_seqs    = params[2];
    const uint k_width   = params[3];
    const uint k_minus1  = k_width - 1u;

    const uint c = tid.x;
    const uint t = tid.y;
    const uint s = tid.z;
    if (c >= channels || t >= n_tokens || s >= n_seqs) {
        return;
    }

    const uint x_seq_stride = n_tokens * channels;
    const uint s_seq_stride = k_minus1 * channels;
    // conv_capture stride: per (s, t) block is k_minus1 * channels.
    const uint capture_per_t = k_minus1 * channels;
    const uint capture_seq_stride = n_tokens * capture_per_t;

    // ── Step 1: forward output (identical to ssm_conv_forward_f32) ──
    float sum = 0.0f;
    for (uint k = 0; k < k_width; ++k) {
        const uint t_ext = t + k;
        float val;
        if (t_ext < k_minus1) {
            val = old_state[s * s_seq_stride + c * k_minus1 + t_ext];
        } else {
            const uint t_in = t_ext - k_minus1;
            val = x[s * x_seq_stride + t_in * channels + c];
        }
        sum += kernel_w[c * k_width + k] * val;
    }
    const float out_y = sum / (1.0f + metal::exp(-sum));
    y[s * x_seq_stride + t * channels + c] = out_y;

    // ── Step 2: per-position state capture ──
    //
    // conv_capture[s, t, i, c] for i in 0..k_minus1 = the i-th entry of
    // the state that would be produced by `ssm_conv_state_update_f32`
    // if n_tokens stopped at t+1. Per the existing state-update math
    // (ssm_conv.metal:155-163):
    //   t_ext = (t+1) + i  // here we shifted n_tokens → (t+1)
    //   if t_ext < k_minus1: val = old_state[..t_ext]
    //   else:               val = x[..t_ext - k_minus1]
    //
    // Each thread writes one (i, c) at this (s, t) slot.
    for (uint i = 0; i < k_minus1; ++i) {
        const uint t_ext_state = (t + 1u) + i;
        float state_val;
        if (t_ext_state < k_minus1) {
            state_val = old_state[s * s_seq_stride + c * k_minus1 + t_ext_state];
        } else {
            const uint t_in = t_ext_state - k_minus1;
            state_val = x[s * x_seq_stride + t_in * channels + c];
        }
        // capture[s, t, i, c] = state_val
        conv_capture[s * capture_seq_stride + t * capture_per_t + i * channels + c] = state_val;
    }
}
