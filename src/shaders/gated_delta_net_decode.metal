#include <metal_stdlib>
using namespace metal;

// Fused Gated DeltaNet recurrent kernel — SIMD-group/`simd_sum` variant.
//
// ADR-015 iter56 — peer-pattern NSG variant using
// 32-lane warp-level reductions instead of `threadgroup_barrier` +
// shared memory. Equivalent math to `gated_delta_net_f32` (the existing
// 128-thread/threadgroup variant) but uses dramatically less synchronization
// and zero shared memory, eliminating the threadgroup_barrier stalls that
// dominate decode wall-time on Qwen3.6 35B-A3B (`d_v=128`, NSG=4 → 128 threads
// arranged as 4 SIMD groups instead of one 128-thread block of 1 simd group).
//
// # Mathematical recurrence (per token t within a seq) — IDENTICAL to
//                                                       `gated_delta_net_f32`
//
//   alpha       = exp(-g[t])                                         // scalar
//   state       = alpha * state                                      // [D_k, D_v]
//   sk[i]       = state[*, i] · k[t]                                 // [D_v]
//   delta[i]    = (v[t][i] - sk[i]) * beta[t]                        // [D_v]
//   state[*,i] += delta[i] * k[t]                                    // outer product
//   output[t,i] = state[*, i] · q[t]                                 // [D_v]
//
// NB: caller pre-scales q with 1/sqrt(D_k) (matching the existing
// `gated_delta_net_f32` kernel's contract). The peer variant folds the
// scale in the kernel; we keep the existing two-step contract to make
// iter56 a drop-in parity replacement.
//
// # Memory layouts (innermost-first / column-major)
//
//   q[d_k, k_head, t, s]     shape [D_k, n_k_heads, n_tokens, n_seqs]
//   k[d_k, k_head, t, s]     shape [D_k, n_k_heads, n_tokens, n_seqs]
//   v[d_v, v_head, t, s]     shape [D_v, n_v_heads, n_tokens, n_seqs]
//   g[v_head, t, s]          shape [n_v_heads, n_tokens, n_seqs]
//   beta[v_head, t, s]       shape [n_v_heads, n_tokens, n_seqs]
//   state[d_k, d_v, vh, s]   shape [D_k, D_v, n_v_heads, n_seqs]
//                            (D_k innermost — per-row contiguous of length D_k)
//   output[d_v, vh, t, s]    shape [D_v, n_v_heads, n_tokens, n_seqs]
//
// # Threading model
//
//   threadgroup size:   (32, NSG, 1) — 32*NSG threads, NSG SIMD groups
//   grid (in tg):       (D_v / NSG, n_v_heads, n_seqs)
//
//   A single threadgroup handles a contiguous range of `NSG` d_v rows:
//     i20 = tgpig.x * NSG + ty
//   Within that row, thread `tx` (lane within a SIMD group) handles
//   `NSG` cells of D_k:
//     is = tx * NSG + j  for j in 0..NSG
//
//   Each thread holds NSG state cells in private memory (registers) — bytes
//   per thread = 4*NSG, fitting comfortably in the register file (NSG ≤ 4 →
//   ≤ 16 bytes), eliminating the spill-to-private-memory hit suffered by the
//   128-thread variant which needs 128*4=512 bytes/thread.
//
//   Cross-lane reductions (s_k and y) use `simd_sum` — single-cycle warp
//   reduction. Output writeback is performed only by lane 0.
//
// # Buffer bindings — matches `gated_delta_net_f32` for drop-in dispatch
//
//   buffer(0): q           f32
//   buffer(1): k           f32
//   buffer(2): v           f32
//   buffer(3): g           f32
//   buffer(4): beta        f32
//   buffer(5): state_in    f32
//   buffer(6): output      f32
//   buffer(7): state_out   f32
//   buffer(8): params      uint[9]: (D_k, D_v, n_k_heads, n_v_heads,
//                                    n_tokens, n_seqs, 0, 0,
//                                    q_scale_bits)  // ADR-033 §Pi iter 25:
//                                    q_scale_bits = as_type<uint>(1.0f/sqrt(D_k))
//                                    if non-zero, kernel applies q_scale at
//                                    output writeback (peer pattern,
//                                    eliminates the per-layer scalar_mul_f32
//                                    pre-pass dispatch). If zero, kernel runs
//                                    in legacy mode (caller pre-scaled q).

// `NSG` per-thread state-cell count. Caller selects via kernel-name dispatch:
// `gated_delta_net_decode_f32_1` / `_2` / `_4`. NSG must be an integer divisor
// of D_v and (NSG * 32) must be at least D_v (otherwise some lanes are idle
// and the simd_sum on s_k accumulates wrong sub-sums). Caller enforces these.
//
// # Two addressing schemes, bit-identical results
//
// * `gated_delta_net_decode_pointer_impl` (host names
//   `gated_delta_net_decode_seq_f32_<NSG>`; dispatched for n_tokens >= 2):
//   advances per-lane pointers once per token (the reference
//   implementation's pattern) instead of rebuilding `uint` element indexes
//   every token, and takes D_k = 32 * NSG as a compile-time constant (the
//   dispatcher selects NSG = D_k / 32 and requires D_k % 32 == 0, so this is
//   exact; params[0] is not read). A wrapping `uint` index
//   (`k[kq_base + tx*NSG + j]`) blocks the compiler from folding the
//   constant `j` into the load's immediate offset; a running pointer `kp`
//   with `kp[j]` (j constant after unrolling) folds it. Measured ~27% faster
//   at n_tokens = 512, but slower at n_tokens = 1, where there is no
//   per-token index math to save.
// * `gated_delta_net_decode_index_impl` (host names
//   `gated_delta_net_decode_f32_<NSG>`, dispatched for n_tokens == 1, and
//   `gated_delta_net_decode_legacy_f32_<NSG>`, the test/bench reference):
//   rebuilds `uint` indexes every token.
//
// Every load reads the same element in both, and the floating-point
// operations per state element and per output element are the same in kind
// and order, so output and state are bit-identical
// (`tests/gated_delta_net_decode_addressing.rs`).

template <short NSG>
inline void gated_delta_net_decode_pointer_impl(
    device const float *q,
    device const float *k,
    device const float *v,
    device const float *g,
    device const float *beta,
    device const float *state_in,
    device       float *output,
    device       float *state_out,
    device const uint  *params,
    uint3 tpitg,
    uint3 tgpig
) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    constexpr uint D_k   = 32u * (uint)NSG;
    const uint D_v       = params[1];
    const uint n_k_heads = params[2];
    const uint n_v_heads = params[3];
    const uint n_tokens  = params[4];
    const uint n_seqs    = params[5];
    // q_scale fold-in: params[8] != 0 is the f32 bits of q_scale applied at
    // writeback; params[8] == 0 keeps the caller-pre-scales-q contract.
    const uint q_scale_bits = params[8];
    const float q_scale = q_scale_bits == 0u ? 1.0f : as_type<float>(q_scale_bits);

    const uint v_head = tgpig.y;
    const uint seq    = tgpig.z;
    const uint i20    = tgpig.x * (uint)NSG + ty;            // d_v row index

    if (v_head >= n_v_heads || seq >= n_seqs || i20 >= D_v) return;

    // GQA: tiled mapping.
    const uint k_head = v_head % n_k_heads;

    // Per-token pointer steps (elements).
    const uint kq_token_stride = n_k_heads * D_k;
    const uint v_token_stride  = n_v_heads * D_v;

    // Lane base pointers at token 0. Each lane owns D_k cells
    // `tx*NSG + j`, j in 0..NSG, of state row i20.
    const uint state_row_base =
        (seq * n_v_heads + v_head) * (D_v * D_k) + i20 * D_k + tx * (uint)NSG;
    device const float *kp = k + (seq * n_tokens * kq_token_stride + k_head * D_k + tx * (uint)NSG);
    device const float *qp = q + (seq * n_tokens * kq_token_stride + k_head * D_k + tx * (uint)NSG);
    device const float *vp = v + (seq * n_tokens * v_token_stride + v_head * D_v + i20);
    device       float *op = output + (seq * n_tokens * v_token_stride + v_head * D_v + i20);
    device const float *gp = g    + (seq * n_tokens * n_v_heads + v_head);
    device const float *bp = beta + (seq * n_tokens * n_v_heads + v_head);

    float ls[NSG];
    {
        device const float *sp = state_in + state_row_base;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            ls[j] = sp[j];
        }
    }

    // Iterate tokens. For pure decode this loop runs once.
    for (uint t = 0; t < n_tokens; ++t) {
        const float beta_val = bp[0];
        const float g_val    = gp[0];
        const float alpha    = metal::exp(-g_val);

        // Step 1+2: decay state in place AND compute partial s_k for this thread.
        float partial_sk = 0.0f;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            ls[j] *= alpha;
            partial_sk += ls[j] * kp[j];
        }
        // Cross-lane reduction: 32 lanes -> single broadcast value.
        const float sk = metal::simd_sum(partial_sk);

        // delta[i20] (uniform across this row's lanes).
        const float delta = (vp[0] - sk) * beta_val;

        // Step 3+4: state update + output partial.
        float partial_y = 0.0f;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            ls[j] += delta * kp[j];
            partial_y += ls[j] * qp[j];
        }
        const float y = metal::simd_sum(partial_y);

        // Output: lane 0 of each (i20) row writes the fully-reduced value.
        if (tx == 0) {
            op[0] = y * q_scale;
        }

        kp += kq_token_stride;
        qp += kq_token_stride;
        vp += v_token_stride;
        op += v_token_stride;
        gp += n_v_heads;
        bp += n_v_heads;
    }

    // Save final state — each thread writes its NSG cells.
    {
        device float *sp = state_out + state_row_base;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            sp[j] = ls[j];
        }
    }
}

// Index-rebuild addressing (the pre-rewrite kernel, unchanged).
template <short NSG>
inline void gated_delta_net_decode_index_impl(
    device const float *q,
    device const float *k,
    device const float *v,
    device const float *g,
    device const float *beta,
    device const float *state_in,
    device       float *output,
    device       float *state_out,
    device const uint  *params,
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
    // ADR-033 §Pi iter 25 (2026-05-23): q_scale fold-in (peer pattern).
    // If params[8] != 0, interpret as a f32 bits encoding of q_scale, and
    // apply at output writeback. Default (params[8]==0) preserves legacy
    // contract (caller pre-scales q).
    const uint q_scale_bits = params[8];
    const float q_scale = q_scale_bits == 0u ? 1.0f : as_type<float>(q_scale_bits);

    const uint v_head = tgpig.y;
    const uint seq    = tgpig.z;
    const uint i20    = tgpig.x * (uint)NSG + ty;            // d_v row index

    if (v_head >= n_v_heads || seq >= n_seqs || i20 >= D_v) return;

    // GQA: tiled mapping (matches the existing `gated_delta_net_f32`
    // kernel).
    const uint k_head = v_head % n_k_heads;

    // Strides (matches `gated_delta_net_f32` — D_k innermost in state).
    const uint kq_token_stride   = n_k_heads * D_k;
    const uint kq_seq_stride     = n_tokens * kq_token_stride;
    const uint v_token_stride    = n_v_heads * D_v;
    const uint v_seq_stride      = n_tokens * v_token_stride;
    const uint scalar_seq_stride = n_tokens * n_v_heads;
    const uint state_head_stride = D_v * D_k;
    const uint state_seq_stride  = n_v_heads * state_head_stride;

    // q_scale handling (ADR-033 §Pi iter 25): this kernel folds in
    // `q_scale` at writeback ONLY when params[8] (q_scale_bits) != 0 — see
    // the q_scale computation above and the `y * q_scale` store below. When
    // params[8] == 0 (legacy contract), q_scale == 1.0 and the caller is
    // responsible for pre-scaling `q_l2 -> q_scaled` via `scalar_mul_f32`
    // before dispatch, matching `gated_delta_net_f32`. This keeps the kernel
    // a drop-in for `dispatch_gated_delta_net` and the parity ground-truth.

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

    // Iterate tokens. For pure decode this loop runs once.
    for (uint t = 0; t < n_tokens; ++t) {
        const uint kq_base = seq * kq_seq_stride + t * kq_token_stride + k_head * D_k;
        const uint v_base  = seq * v_seq_stride + t * v_token_stride + v_head * D_v;
        const uint sc_idx  = seq * scalar_seq_stride + t * n_v_heads + v_head;

        const float beta_val = beta[sc_idx];
        const float g_val    = g[sc_idx];
        const float alpha    = metal::exp(-g_val);

        // Step 1+2: decay state in place AND compute partial s_k for this thread.
        float partial_sk = 0.0f;
        #pragma clang loop unroll(full)
        for (short j = 0; j < NSG; ++j) {
            const uint is = tx * (uint)NSG + (uint)j;
            ls[j] *= alpha;
            partial_sk += ls[j] * k[kq_base + is];
        }
        // Cross-lane reduction: 32 lanes -> single broadcast value.
        const float sk = metal::simd_sum(partial_sk);

        // delta[i20] (uniform across this row's lanes).
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

        // Output: lane 0 of each (i20) row writes the fully-reduced value.
        // ADR-033 §Pi iter 25: apply q_scale at writeback
        // (`y * q_scale`). When q_scale_bits == 0, q_scale == 1.0 and
        // the legacy caller-pre-scales-q contract is preserved.
        if (tx == 0) {
            output[seq * v_seq_stride + t * v_token_stride + v_head * D_v + i20] = y * q_scale;
        }
    }

    // Save final state — each thread writes its NSG cells.
    #pragma clang loop unroll(full)
    for (short j = 0; j < NSG; ++j) {
        const uint is = tx * (uint)NSG + (uint)j;
        state_out[state_row_base + is] = ls[j];
    }
}

// Concrete kernel functions — each NSG variant is a separate Metal entry
// point selected by the host-side dispatcher based on D_k.

kernel void gated_delta_net_decode_f32_1(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<1>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_f32_2(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<2>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_f32_4(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<4>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

// Pointer-addressing entry points — dispatched for n_tokens >= 2.

kernel void gated_delta_net_decode_seq_f32_1(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_pointer_impl<1>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_seq_f32_2(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_pointer_impl<2>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_seq_f32_4(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_pointer_impl<4>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

// Legacy-named index-rebuild entry points — test/bench reference only
// (same code as `gated_delta_net_decode_f32_<NSG>`).

kernel void gated_delta_net_decode_legacy_f32_1(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<1>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_legacy_f32_2(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<2>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}

kernel void gated_delta_net_decode_legacy_f32_4(
    device const float *q           [[buffer(0)]],
    device const float *k           [[buffer(1)]],
    device const float *v           [[buffer(2)]],
    device const float *g           [[buffer(3)]],
    device const float *beta        [[buffer(4)]],
    device const float *state_in    [[buffer(5)]],
    device float       *output      [[buffer(6)]],
    device float       *state_out   [[buffer(7)]],
    device const uint  *params      [[buffer(8)]],
    uint3 tpitg                     [[thread_position_in_threadgroup]],
    uint3 tgpig                     [[threadgroup_position_in_grid]]
) {
    gated_delta_net_decode_index_impl<4>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}
