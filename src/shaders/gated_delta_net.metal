#include <metal_stdlib>
using namespace metal;

// Gated DeltaNet fused kernel.
//
// Spec source: ADR-013 Decision 6. Derived from the DeltaNet recurrence
// (delta-net-base.cpp fused path) + Qwen3.5 math. No Metal or C code copied.
//
// # Mathematical recurrence (per token t within a seq)
//
//   alpha       = exp(-g[t])                                         // scalar
//   state_dec   = alpha * state                                      // [D_k, D_v]
//   delta       = v[t] - state_dec @ k[t]                           // [D_v]
//   state'      = state_dec + beta[t] * outer(delta, k[t])          // [D_k, D_v]
//   output[t]   = state' @ q[t]                                      // [D_v]
//
// IMPORTANT: alpha is applied to state BEFORE computing delta = v - state@k.
// This matches llama.cpp build_delta_net_autoregressive (line 338-360):
//   g = ggml_exp(g); s = s*g; sk = sum(s*k); d = v-sk; s = s + outer(beta*d, k)
//
// GQA broadcast: `num_v_heads` may exceed `num_k_heads`. Each v_head is
// mapped to a k_head by `k_head = v_head / group_ratio` where
// `group_ratio = num_v_heads / num_k_heads`.
//
// # Memory layouts (innermost-first / column-major)
//
//   q[d_k, k_head, t, s] and k[...]  — shape [D_k, n_k_heads, n_tokens, n_seqs]
//   v[d_v, v_head, t, s]             — shape [D_v, n_v_heads, n_tokens, n_seqs]
//   g[v_head, t, s] and beta[...]    — shape [n_v_heads, n_tokens, n_seqs]
//   state[d_k, d_v, v_head, s]       — shape [D_k, D_v, n_v_heads, n_seqs]
//                                      (d_k innermost for per-thread contiguous loads)
//   output[d_v, v_head, t, s]        — same shape as v
//
// # Threading model
//
//   One threadgroup per (v_head, seq). Threadgroup size = D_v threads.
//   Thread i (tid) owns state_row[:] = state[*, i, v_head, s] — the i-th
//   column of the state matrix, stored CONTIGUOUSLY in thread-private memory
//   across all tokens (loaded once at start, kept in registers / private,
//   written once at end). This is the key perf invariant.
//
// # Shared memory layout (per threadgroup)
//
//   sh_k[0 .. D_k]              — current token's K vector
//   sh_q[D_k .. 2*D_k]          — current token's Q vector
//   sh_v[2*D_k .. 2*D_k + D_v]  — current token's V vector
//   sh_delta[2*D_k + D_v .. 2*D_k + 2*D_v]  — delta = v - state@k
//
// Total shared bytes = (2*D_k + 2*D_v) * 4. For D=128 that's 2KB.
//
// # Buffer bindings
//
//   buffer(0): q           f32
//   buffer(1): k           f32
//   buffer(2): v           f32
//   buffer(3): g           f32 (one per v_head per token per seq)
//   buffer(4): beta        f32 (one per v_head per token per seq)
//   buffer(5): state_in    f32
//   buffer(6): output      f32
//   buffer(7): state_out   f32
//   buffer(8): params      uint[8]: (D_k, D_v, n_k_heads, n_v_heads,
//                                    n_tokens, n_seqs, 0, 0)

// Hard cap for thread-private state row. Qwen3.5 uses D_k = D_v = 128 so
// a 128-float private row is sufficient. Growing beyond this will spill to
// slower private memory; kernel still works but loses perf.
constant uint MAX_STATE_D = 128u;

kernel void gated_delta_net_f32(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    threadgroup float  *shared_mem  [[threadgroup(0)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint D_k       = params[0];
    const uint D_v       = params[1];
    const uint n_k_heads = params[2];
    const uint n_v_heads = params[3];
    const uint n_tokens  = params[4];
    const uint n_seqs    = params[5];

    const uint v_head = tgid.x;
    const uint seq    = tgid.y;

    if (v_head >= n_v_heads || seq >= n_seqs) return;
    if (tid >= D_v) return;

    // GQA broadcast: map v_head to k_head using modulo (tiled), matching
    // llama.cpp's fused Metal kernel (i01 = i21 % args.ne01) and the
    // ggml_repeat_4d tiled expansion used in the non-fused path.
    // NOT division (block-style), which would give a different ordering.
    const uint k_head = v_head % n_k_heads;

    // Strides.
    const uint kq_token_stride   = n_k_heads * D_k;
    const uint kq_seq_stride     = n_tokens * kq_token_stride;
    const uint v_token_stride    = n_v_heads * D_v;
    const uint v_seq_stride      = n_tokens * v_token_stride;
    const uint scalar_seq_stride = n_tokens * n_v_heads;
    const uint state_head_stride = D_v * D_k;
    const uint state_seq_stride  = n_v_heads * state_head_stride;

    // Private state column — state[:, tid, v_head, seq] with d_k fastest.
    thread float state_row[MAX_STATE_D];

    // Load initial state.
    const uint state_base = seq * state_seq_stride + v_head * state_head_stride + tid * D_k;
    for (uint j = 0; j < D_k; ++j) {
        state_row[j] = state_in[state_base + j];
    }

    // Shared memory split: sh_k[D_k], sh_q[D_k], sh_v[D_v], sh_delta[D_v].
    threadgroup float *sh_k     = shared_mem;
    threadgroup float *sh_q     = shared_mem + D_k;
    threadgroup float *sh_v     = shared_mem + 2u * D_k;
    threadgroup float *sh_delta = shared_mem + 2u * D_k + D_v;

    for (uint t = 0; t < n_tokens; ++t) {
        const uint kq_base = seq * kq_seq_stride + t * kq_token_stride + k_head * D_k;
        const uint v_base  = seq * v_seq_stride + t * v_token_stride + v_head * D_v;
        const uint sc_idx  = seq * scalar_seq_stride + t * n_v_heads + v_head;

        // Threads cooperatively load k and q (D_k each). Threadgroup has
        // D_v threads; loop handles D_k != D_v.
        for (uint j = tid; j < D_k; j += D_v) {
            sh_k[j] = k[kq_base + j];
            sh_q[j] = q[kq_base + j];
        }
        // v has D_v elements → one per thread.
        sh_v[tid] = v[v_base + tid];

        const float beta_val = beta[sc_idx];
        const float g_val    = g[sc_idx];
        const float alpha    = metal::exp(-g_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: decay state — apply alpha to state_row BEFORE computing sk.
        // This matches llama.cpp: s = s * exp(gate); sk = sum(s * k).
        for (uint j = 0; j < D_k; ++j) {
            state_row[j] *= alpha;
        }

        // Step 2: sk = (alpha*state) @ k for this thread's d_v row.
        float sk = 0.0f;
        for (uint j = 0; j < D_k; ++j) {
            sk += state_row[j] * sh_k[j];
        }
        // delta[tid] = v[tid] - sk  (using decayed state)
        sh_delta[tid] = sh_v[tid] - sk;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: update state_row: state[j, i] += beta * delta[i] * k[j].
        // Note: state_row is already alpha-decayed from step 1.
        const float delta_i = sh_delta[tid];
        const float beta_delta = beta_val * delta_i;
        for (uint j = 0; j < D_k; ++j) {
            state_row[j] += beta_delta * sh_k[j];
        }

        // output[i] = state' @ q = dot(state_row, q).
        float out_i = 0.0f;
        for (uint j = 0; j < D_k; ++j) {
            out_i += state_row[j] * sh_q[j];
        }

        const uint out_base = seq * v_seq_stride + t * v_token_stride + v_head * D_v + tid;
        output[out_base] = out_i;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save final state.
    const uint state_out_base = seq * state_seq_stride + v_head * state_head_stride + tid * D_k;
    for (uint j = 0; j < D_k; ++j) {
        state_out[state_out_base + j] = state_row[j];
    }
}
