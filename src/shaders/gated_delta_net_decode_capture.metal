#include <metal_stdlib>
using namespace metal;

// ADR-034 task #90 (2026-05-21) — Gated DeltaNet recurrent kernel with
// **per-position state capture** for K≥2 speculative decoding on hybrid
// Qwen 3.5/3.6 (the structural lever MTPLX has and we don't, per the
// 2026-05-21 deep-research synthesis; gdn_capture.py:128-201 reference).
//
// IDENTICAL math + threading model to `gated_delta_net_decode_f32_<NSG>`
// (`gated_delta_net_decode.metal`), with ONE additional output buffer:
//   state_capture[D_k, D_v, n_v_heads, n_tokens, n_seqs] — recurrent
//   state AFTER each token's update, written inside the token loop.
//
// On partial-reject of K drafts in spec-decode, the caller selects
// `state_capture[..., accepted_idx, ...]` as the new active state for
// the LinearAttnStateSlot — solving the GDN rollback gap documented at
// [[project_adr034_kn_chained_wip_2026_05_21]] / task #86.
//
// # Buffer bindings — extends the decode kernel with state_capture at slot 9
//
//   buffer(0): q              f32
//   buffer(1): k              f32
//   buffer(2): v              f32
//   buffer(3): g              f32
//   buffer(4): beta           f32
//   buffer(5): state_in       f32
//   buffer(6): output         f32
//   buffer(7): state_out      f32   (= state after final token, same as decode)
//   buffer(8): params         uint[9]: (D_k, D_v, n_k_heads, n_v_heads,
//                                       n_tokens, n_seqs, 0, 0, q_scale_bits)
//                                       index 8 = q_scale_bits (0 = no fold-in,
//                                       caller pre-scales q; non-zero = f32 bits
//                                       of q_scale applied at writeback)
//   buffer(9): state_capture  f32   (per-position state, ADDED in this kernel)
//
// # Memory layout for state_capture
//
// Innermost D_k (matching state buffer layout), then D_v, then n_v_heads,
// then n_tokens, then n_seqs:
//   state_capture[d_k, d_v, vh, t, s]   shape [D_k, D_v, n_v_heads, n_tokens, n_seqs]
//   offset of (s, t, vh, d_v, d_k):
//     s * state_capture_seq_stride
//     + t * state_capture_token_stride
//     + vh * D_v * D_k
//     + d_v * D_k
//     + d_k
//   where:
//     state_capture_token_stride = n_v_heads * D_v * D_k
//     state_capture_seq_stride   = n_tokens * state_capture_token_stride
//
// # Memory cost
//
// For Qwen 3.5/3.6 (D_k=128, D_v=128, n_v_heads=8, n_seqs=1, K+1=4):
//   per-call:   128 * 128 * 8 * 4 * 4 bytes = 2 MB per LA layer
//   per-forward (30+ LA layers): ~60-90 MB total
//
// Caller allocates this buffer once per spec-decode iter and reads
// states[accepted_idx] on partial-reject.

template <short NSG, bool SELECTED>
inline void gated_delta_net_decode_capture_impl(
    device const float *q,
    device const float *k,
    device const float *v,
    device const float *g,
    device const float *beta,
    device const float *state_in,
    device       float *output,
    device       float *state_out,
    device const uint  *params,
    device       float *state_capture,
    const uint capture_token,
    uint3 tpitg,
    uint3 tgpig
) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint D_k       = params[0];
    const uint D_v       = params[1];
    const uint n_k_heads = params[2];
    const uint n_v_heads = params[3];
    const uint n_tokens  = params[4];
    const uint n_seqs    = params[5];
    // ADR-033 §Pi iter 25: q_scale fold-in — mirror of the non-capture
    // gated_delta_net_decode kernel so capture stays bit-equivalent. When
    // params[8] != 0 it is the f32-bits encoding of q_scale, applied at
    // writeback; params[8] == 0 preserves the legacy contract (q pre-scaled).
    const uint q_scale_bits = params[8];
    const float q_scale = q_scale_bits == 0u ? 1.0f : as_type<float>(q_scale_bits);

    const uint v_head = tgpig.y;
    const uint seq    = tgpig.z;
    const uint i20    = tgpig.x * (uint)NSG + ty;            // d_v row index

    if (v_head >= n_v_heads || seq >= n_seqs || i20 >= D_v) return;

    const uint k_head = v_head % n_k_heads;

    // Strides — identical to non-capture variant.
    const uint kq_token_stride   = n_k_heads * D_k;
    const uint kq_seq_stride     = n_tokens * kq_token_stride;
    const uint v_token_stride    = n_v_heads * D_v;
    const uint v_seq_stride      = n_tokens * v_token_stride;
    const uint scalar_seq_stride = n_tokens * n_v_heads;
    const uint state_head_stride = D_v * D_k;
    const uint state_seq_stride  = n_v_heads * state_head_stride;

    // ADR-034 task #90 — per-position capture strides.
    const uint state_capture_token_stride = state_seq_stride;        // n_v_heads * D_v * D_k
    const uint state_capture_seq_stride   = n_tokens * state_capture_token_stride;

    // Per-thread state row slice — NSG cells covering D_k positions
    // `is = tx*NSG + j` for j in 0..NSG.
    float ls[NSG];
    const uint state_row_base =
        seq * state_seq_stride + v_head * state_head_stride + i20 * D_k;

    #pragma clang loop unroll(full)
    for (short j = 0; j < NSG; ++j) {
        const uint is = tx * (uint)NSG + (uint)j;
        ls[j] = state_in[state_row_base + is];
    }

    // Iterate tokens. AFTER each token's state update we ALSO write to
    // state_capture so the caller can roll back to any accepted prefix.
    for (uint t = 0; t < n_tokens; ++t) {
        const uint kq_base = seq * kq_seq_stride + t * kq_token_stride + k_head * D_k;
        const uint v_base  = seq * v_seq_stride + t * v_token_stride + v_head * D_v;
        const uint sc_idx  = seq * scalar_seq_stride + t * n_v_heads + v_head;

        const float beta_val = beta[sc_idx];
        const float g_val    = g[sc_idx];
        const float alpha    = metal::exp(-g_val);

        // Step 1+2: decay + partial s_k.
        float partial_sk = 0.0f;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            const uint is = tx * (uint)NSG + (uint)j;
            ls[j] *= alpha;
            partial_sk += ls[j] * k[kq_base + is];
        }
        const float sk = metal::simd_sum(partial_sk);

        const float delta = (v[v_base + i20] - sk) * beta_val;

        // Step 3+4: state update + output partial.
        float partial_y = 0.0f;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            const uint is = tx * (uint)NSG + (uint)j;
            ls[j] += delta * k[kq_base + is];
            partial_y += ls[j] * q[kq_base + is];
        }
        const float y = metal::simd_sum(partial_y);

        // Output: lane 0 writes the fully-reduced value.
        if (tx == 0) {
            output[seq * v_seq_stride + t * v_token_stride + v_head * D_v + i20] = y * q_scale;
        }

        // ADR-034 task #90 — capture per-position state AFTER this
        // token's update. Each thread writes its NSG cells.
        if (!SELECTED || t == capture_token) {
            const uint capture_row_base =
                (SELECTED
                    ? seq * state_capture_token_stride
                    : seq * state_capture_seq_stride + t * state_capture_token_stride)
                + v_head * state_head_stride
                + i20 * D_k;
            #pragma clang loop unroll(full)
            for (short j = 0; j < NSG; ++j) {
                const uint is = tx * (uint)NSG + (uint)j;
                state_capture[capture_row_base + is] = ls[j];
            }
        }
    }

    // Save final state — preserves the existing decode kernel's contract.
    // After this kernel returns, state_out == state_capture[..., n_tokens-1, ...].
    #pragma clang loop unroll(full)
    for (short j = 0; j < NSG; ++j) {
        const uint is = tx * (uint)NSG + (uint)j;
        state_out[state_row_base + is] = ls[j];
    }
}

// Concrete kernel functions — one per NSG variant.

kernel void gated_delta_net_decode_capture_f32_1(
    device const float *q             [[buffer(0)]],
    device const float *k             [[buffer(1)]],
    device const float *v             [[buffer(2)]],
    device const float *g             [[buffer(3)]],
    device const float *beta          [[buffer(4)]],
    device const float *state_in      [[buffer(5)]],
    device float       *output        [[buffer(6)]],
    device float       *state_out     [[buffer(7)]],
    device const uint  *params        [[buffer(8)]],
    device float       *state_capture [[buffer(9)]],
    uint3 tpitg                       [[thread_position_in_threadgroup]],
    uint3 tgpig                       [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_capture_impl<1, false>(
        q, k, v, g, beta, state_in, output, state_out, params, state_capture,
        0u, tpitg, tgpig);
}

kernel void gated_delta_net_decode_capture_f32_2(
    device const float *q             [[buffer(0)]],
    device const float *k             [[buffer(1)]],
    device const float *v             [[buffer(2)]],
    device const float *g             [[buffer(3)]],
    device const float *beta          [[buffer(4)]],
    device const float *state_in      [[buffer(5)]],
    device float       *output        [[buffer(6)]],
    device float       *state_out     [[buffer(7)]],
    device const uint  *params        [[buffer(8)]],
    device float       *state_capture [[buffer(9)]],
    uint3 tpitg                       [[thread_position_in_threadgroup]],
    uint3 tgpig                       [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_capture_impl<2, false>(
        q, k, v, g, beta, state_in, output, state_out, params, state_capture,
        0u, tpitg, tgpig);
}

kernel void gated_delta_net_decode_capture_f32_4(
    device const float *q             [[buffer(0)]],
    device const float *k             [[buffer(1)]],
    device const float *v             [[buffer(2)]],
    device const float *g             [[buffer(3)]],
    device const float *beta          [[buffer(4)]],
    device const float *state_in      [[buffer(5)]],
    device float       *output        [[buffer(6)]],
    device float       *state_out     [[buffer(7)]],
    device const uint  *params        [[buffer(8)]],
    device float       *state_capture [[buffer(9)]],
    uint3 tpitg                       [[thread_position_in_threadgroup]],
    uint3 tgpig                       [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_capture_impl<4, false>(
        q, k, v, g, beta, state_in, output, state_out, params, state_capture,
        0u, tpitg, tgpig);
}

#define DEFINE_SELECTED_CAPTURE(NSG) \
kernel void gated_delta_net_decode_selected_capture_f32_##NSG( \
    device const float *q [[buffer(0)]], \
    device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], \
    device const float *g [[buffer(3)]], \
    device const float *beta [[buffer(4)]], \
    device const float *state_in [[buffer(5)]], \
    device float *output [[buffer(6)]], \
    device float *state_out [[buffer(7)]], \
    device const uint *params [[buffer(8)]], \
    device float *state_capture [[buffer(9)]], \
    constant uint &capture_token [[buffer(10)]], \
    uint3 tpitg [[thread_position_in_threadgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]]) { \
    gated_delta_net_decode_capture_impl<NSG, true>( \
        q, k, v, g, beta, state_in, output, state_out, params, state_capture, \
        capture_token, tpitg, tgpig); \
}

DEFINE_SELECTED_CAPTURE(1)
DEFINE_SELECTED_CAPTURE(2)
DEFINE_SELECTED_CAPTURE(4)
