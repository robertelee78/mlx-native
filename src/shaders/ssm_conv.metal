#include <metal_stdlib>
using namespace metal;

// SSM depthwise causal 1D convolution + SiLU for Qwen3.5 Gated DeltaNet.
//
// Spec source: ADR-013 Decision 7. Formula derived from Mamba-family causal
// depthwise 1D convolution literature plus the spec block in ADR-013; no
// llama.cpp source referenced.
//
// Operation:
//   ssm_conv(x, kernel_w, state) -> (y, new_state)
//     x:        [channels, n_tokens, n_seqs]
//     kernel_w: [K, channels]           (K = 4 for Qwen3.5)
//     state:    [K-1, channels, n_seqs] (previous (K-1) conv inputs per seq)
//   extended(c, t_ext, s) =
//     state(t_ext, c, s)              if t_ext < K-1
//     x(c, t_ext - (K-1), s)          otherwise
//   y(c, t, s) = silu( sum_{k=0..K} kernel_w(k, c) * extended(c, t + k, s) )
//   new_state(i, c, s) = extended(c, n_tokens + i, s)  for i in 0..K-1
//
// Causality: sum index t+k, for k in [0, K-1], maps to extended indices
// [t, t+K-1]. At t=0 these are [0, K-1] which are the (K-1) state tokens
// and the first x token, so each output only depends on past/current inputs.
//
// Memory layout (column-major / innermost-first):
//   x[c, t, s]        at offset s * (n_tokens * channels) + t * channels + c
//   y[c, t, s]        same as x
//   state[i, c, s]    at offset s * ((K-1) * channels) + c * (K-1) + i
//   kernel_w[k, c]    at offset c * K + k
//
// Numerical stability: accumulation is performed in f32 regardless of input
// dtype. SiLU uses `x / (1 + exp(-x))`.

// -----------------------------------------------------------------
// Kernel A — convolution forward (one thread per (c, t, s))
// -----------------------------------------------------------------

kernel void ssm_conv_forward_f32(
    device const float *x           [[buffer(0)]],
    device const float *kernel_w    [[buffer(1)]],
    device const float *state       [[buffer(2)]],
    device float       *y           [[buffer(3)]],
    device const uint  *params      [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint channels  = params[0];
    const uint n_tokens  = params[1];
    const uint n_seqs    = params[2];
    const uint k_width   = params[3]; // K, typically 4

    const uint c = tid.x;
    const uint t = tid.y;
    const uint s = tid.z;
    if (c >= channels || t >= n_tokens || s >= n_seqs) {
        return;
    }

    const uint x_seq_stride = n_tokens * channels;
    const uint s_seq_stride = (k_width - 1u) * channels;

    float sum = 0.0f;
    for (uint k = 0; k < k_width; ++k) {
        const uint t_ext = t + k;
        float val;
        if (t_ext < k_width - 1u) {
            val = state[s * s_seq_stride + c * (k_width - 1u) + t_ext];
        } else {
            const uint t_in = t_ext - (k_width - 1u);
            val = x[s * x_seq_stride + t_in * channels + c];
        }
        sum += kernel_w[c * k_width + k] * val;
    }

    // SiLU: y = sum * sigmoid(sum) = sum / (1 + exp(-sum))
    const float out = sum / (1.0f + metal::exp(-sum));

    y[s * x_seq_stride + t * channels + c] = out;
}

kernel void ssm_conv_forward_bf16(
    device const bfloat *x          [[buffer(0)]],
    device const bfloat *kernel_w   [[buffer(1)]],
    device const bfloat *state      [[buffer(2)]],
    device bfloat       *y          [[buffer(3)]],
    device const uint   *params     [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint channels = params[0];
    const uint n_tokens = params[1];
    const uint n_seqs   = params[2];
    const uint k_width  = params[3];

    const uint c = tid.x;
    const uint t = tid.y;
    const uint s = tid.z;
    if (c >= channels || t >= n_tokens || s >= n_seqs) {
        return;
    }

    const uint x_seq_stride = n_tokens * channels;
    const uint s_seq_stride = (k_width - 1u) * channels;

    float sum = 0.0f;
    for (uint k = 0; k < k_width; ++k) {
        const uint t_ext = t + k;
        float val;
        if (t_ext < k_width - 1u) {
            val = float(state[s * s_seq_stride + c * (k_width - 1u) + t_ext]);
        } else {
            const uint t_in = t_ext - (k_width - 1u);
            val = float(x[s * x_seq_stride + t_in * channels + c]);
        }
        sum += float(kernel_w[c * k_width + k]) * val;
    }

    const float out = sum / (1.0f + metal::exp(-sum));

    y[s * x_seq_stride + t * channels + c] = bfloat(out);
}

// -----------------------------------------------------------------
// Kernel B — state update (one thread per (i, c, s))
//
// new_state(i, c, s) = extended(c, n_tokens + i, s)
//                    = state(n_tokens + i, c, s)           if n_tokens + i < K - 1
//                    = x(c, n_tokens + i - (K - 1), s)      otherwise
//
// Written in two passes because in the n_tokens + i < K - 1 branch we read
// from `state` which aliases the output buffer. Caller must provide a
// separate `new_state` buffer to avoid the aliasing race when n_tokens < K-1.
// -----------------------------------------------------------------

kernel void ssm_conv_state_update_f32(
    device const float *x          [[buffer(0)]],
    device const float *old_state  [[buffer(1)]],
    device float       *new_state  [[buffer(2)]],
    device const uint  *params     [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint channels = params[0];
    const uint n_tokens = params[1];
    const uint n_seqs   = params[2];
    const uint k_width  = params[3];
    const uint k_minus1 = k_width - 1u;

    const uint i = tid.x; // 0..k-1
    const uint c = tid.y;
    const uint s = tid.z;
    if (i >= k_minus1 || c >= channels || s >= n_seqs) {
        return;
    }

    const uint x_seq_stride = n_tokens * channels;
    const uint s_seq_stride = k_minus1 * channels;
    const uint t_ext = n_tokens + i;

    float val;
    if (t_ext < k_minus1) {
        val = old_state[s * s_seq_stride + c * k_minus1 + t_ext];
    } else {
        const uint t_in = t_ext - k_minus1;
        val = x[s * x_seq_stride + t_in * channels + c];
    }
    new_state[s * s_seq_stride + c * k_minus1 + i] = val;
}

kernel void ssm_conv_state_update_bf16(
    device const bfloat *x          [[buffer(0)]],
    device const bfloat *old_state  [[buffer(1)]],
    device bfloat       *new_state  [[buffer(2)]],
    device const uint   *params     [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint channels = params[0];
    const uint n_tokens = params[1];
    const uint n_seqs   = params[2];
    const uint k_width  = params[3];
    const uint k_minus1 = k_width - 1u;

    const uint i = tid.x;
    const uint c = tid.y;
    const uint s = tid.z;
    if (i >= k_minus1 || c >= channels || s >= n_seqs) {
        return;
    }

    const uint x_seq_stride = n_tokens * channels;
    const uint s_seq_stride = k_minus1 * channels;
    const uint t_ext = n_tokens + i;

    bfloat val;
    if (t_ext < k_minus1) {
        val = old_state[s * s_seq_stride + c * k_minus1 + t_ext];
    } else {
        const uint t_in = t_ext - k_minus1;
        val = x[s * x_seq_stride + t_in * channels + c];
    }
    new_state[s * s_seq_stride + c * k_minus1 + i] = val;
}
