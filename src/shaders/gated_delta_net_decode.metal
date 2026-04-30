#include <metal_stdlib>
using namespace metal;

// Fused Gated DeltaNet recurrent kernel — SIMD-group/`simd_sum` variant.
//
// ADR-015 iter56 — mirrors llama.cpp's `kernel_gated_delta_net_f32_<NSG>`
// (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:2532-2647`) using
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
// `gated_delta_net_f32` kernel's contract). llama.cpp's variant folds the
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
//   buffer(8): params      uint[8]: (D_k, D_v, n_k_heads, n_v_heads,
//                                    n_tokens, n_seqs, 0, 0)

// `NSG` per-thread state-cell count. Caller selects via kernel-name dispatch:
// `gated_delta_net_decode_f32_1` / `_2` / `_4`. NSG must be an integer divisor
// of D_v and (NSG * 32) must be at least D_v (otherwise some lanes are idle
// and the simd_sum on s_k accumulates wrong sub-sums). Caller enforces these.
//
// MAX_STATE_D — hard cap on D_k. Each thread holds NSG cells, threads-per-row
// is 32, so total covered D_k cells = NSG * 32. Qwen3.5/3.6 use D_k = 128, so
// NSG=4 covers the full row. Increasing this requires growing NSG.

template <short NSG>
inline void gated_delta_net_decode_impl(
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

    const uint v_head = tgpig.y;
    const uint seq    = tgpig.z;
    const uint i20    = tgpig.x * (uint)NSG + ty;            // d_v row index

    if (v_head >= n_v_heads || seq >= n_seqs || i20 >= D_v) return;

    // GQA: tiled mapping (matches llama.cpp `i01 = i21 % args.ne01` and
    // the existing `gated_delta_net_f32` kernel).
    const uint k_head = v_head % n_k_heads;

    // Strides (matches `gated_delta_net_f32` — D_k innermost in state).
    const uint kq_token_stride   = n_k_heads * D_k;
    const uint kq_seq_stride     = n_tokens * kq_token_stride;
    const uint v_token_stride    = n_v_heads * D_v;
    const uint v_seq_stride      = n_tokens * v_token_stride;
    const uint scalar_seq_stride = n_tokens * n_v_heads;
    const uint state_head_stride = D_v * D_k;
    const uint state_seq_stride  = n_v_heads * state_head_stride;

    // NOTE: this kernel does NOT fold-in `q_scale = 1/sqrt(D_k)` (whereas
    // llama.cpp's `kernel_gated_delta_net_impl` does). The hf2q caller
    // pre-scales `q_l2 -> q_scaled` via `scalar_mul_f32` before dispatching,
    // matching the existing `gated_delta_net_f32` kernel's contract. Keeping
    // this kernel a drop-in for `dispatch_gated_delta_net` is the parity
    // ground-truth for iter56.

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
        // (q_scale=1/sqrt(D_k) is applied by the caller to `q` upstream.)
        if (tx == 0) {
            output[seq * v_seq_stride + t * v_token_stride + v_head * D_v + i20] = y;
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
    gated_delta_net_decode_impl<1>(
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
    gated_delta_net_decode_impl<2>(
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
    gated_delta_net_decode_impl<4>(
        q, k, v, g, beta, state_in, output, state_out, params, tpitg, tgpig);
}
