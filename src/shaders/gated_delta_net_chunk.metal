#include <metal_stdlib>
using namespace metal;

// Wave 5b — chunk-parallel Gated DeltaNet inter-chunk state-recurrence kernel.
//
// Spec source (math): arXiv 2412.06464 §4 (Yang–Hatamizadeh; chunkwise
// parallelization of the gated delta rule).
// Spec source (algorithm structure): FLA's
// `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at
// /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-298.
//
// No FLA / Triton / CUDA / Metal code is copied. The math here is a
// re-derivation from the paper's recurrence (Eq. 5–6 + §4 chunkwise form);
// the algorithmic structure (load / dot / mask / accumulate per chunk)
// follows the FLA Triton kernel pattern but is open-coded for Metal.
//
// # Algorithm (per (batch b, head i_h, V-tile i_v))
//
//   b_h := h0[b, i_h, i_v_tile, :]              # [BV, K] f32 in shared mem
//   for i_t in 0..NT:
//       store b_h -> h_out[b, i_t, i_h, i_v_tile, :]   # bf16 store
//       b_w  := w[b, t_chunk, i_h, :]                  # [BT, K] bf16
//       b_u  := u[b, t_chunk, i_h, i_v_tile]            # [BT, BV] bf16
//       b_v  := b_u - b_w @ b_h^T                      # [BT, BV] f32
//       store bf16(b_v) -> v_new[b, t_chunk, i_h, i_v_tile]
//
//       last_t := (i_t+1)*BT - 1
//       g_last := g[b, last_t, i_h]                    # f32 scalar
//       g_blk  := g[b, t_chunk, i_h]                   # f32 [BT]
//       b_v    := b_v * exp(g_last - g_blk)[:, None]
//       b_h    := b_h * exp(g_last)
//
//       b_k := k[b, t_chunk, i_h / GROUP_RATIO, :]     # [BT, K] bf16, GQA
//       b_h += b_v.T @ b_k                              # outer accumulate
//
//   final_state[b, i_h, i_v_tile, :] := b_h            # f32 store
//
// All matmul-style dots: bf16 storage → f32 accumulator.  b_h stays in f32
// in shared memory across the T loop.  The state is V-tile-blocked
// (BV=32 V rows per threadgroup), so threadgroup memory holds a `[BV, K]`
// f32 tile = 32 * 128 * 4 = 16 KB, well under M5 Max's 32 KB limit.
//
// # Memory layouts (innermost-first)
//
//   k:    [B, T, Hg, K]      bf16  — `K` innermost
//   w:    [B, T, H,  K]      bf16  — `K` innermost
//   u:    [B, T, H,  V]      bf16  — `V` innermost
//   g:    [B, T, H]          f32   — `H` innermost
//   h0:   [B, H, V, K]       f32   — `K` innermost (matches FLA p_h0 layout)
//   h_out:[B, NT, H, V, K]   bf16  — `K` innermost
//   v_new:[B, T, H, V]       bf16  — `V` innermost (same as u)
//   final:[B, H, V, K]       f32   — `K` innermost (same as h0)
//
// # Threading
//
//   Grid: (NV_TILES = V/BV, H, B)
//   Threadgroup: 128 threads (4 simdgroups × 32 lanes), arranged as a flat 1D
//     so we can repurpose them flexibly across the per-stage operations.
//
// # Buffer bindings (matching the host)
//
//   buffer(0): k           bf16
//   buffer(1): w           bf16
//   buffer(2): u           bf16
//   buffer(3): g           f32
//   buffer(4): h0          f32
//   buffer(5): h_out       bf16
//   buffer(6): v_new       bf16
//   buffer(7): final_state f32
//   buffer(8): params      uint[8] = [B, T, Hg, H, K, V, BT, NT]

// =====================================================================
// Iter 1 (commit 2): STUB — kernel registered but body intentionally
// returns zero outputs. The test exercises the dispatch path correctly
// (so a stub-only commit is sufficient to confirm the dispatch is wired
// up); the assertion against the FLA fixture will fail until commit 3
// lands the real recurrence body.
// =====================================================================

constant uint BV = 32u;          // V-tile width
constant uint TG_THREADS = 128u; // threadgroup size (4 simdgroups)

kernel void gated_delta_net_chunk_inter_state_bf16(
    device const bfloat *k         [[buffer(0)]],
    device const bfloat *w         [[buffer(1)]],
    device const bfloat *u         [[buffer(2)]],
    device const float  *g         [[buffer(3)]],
    device const float  *h0        [[buffer(4)]],
    device bfloat       *h_out     [[buffer(5)]],
    device bfloat       *v_new     [[buffer(6)]],
    device float        *final_state [[buffer(7)]],
    device const uint   *params    [[buffer(8)]],
    threadgroup float   *shared_mem [[threadgroup(0)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Unused in stub; suppress warnings.
    (void)k; (void)w; (void)u; (void)g; (void)h0;
    (void)shared_mem;

    const uint B   = params[0];
    const uint T   = params[1];
    (void)params;
    (void)B; (void)T;
    const uint Hg  = params[2];
    const uint H   = params[3];
    const uint K   = params[4];
    const uint V   = params[5];
    const uint BT  = params[6];
    const uint NT  = params[7];
    (void)Hg; (void)BT;

    const uint i_v = tgid.x; // V-tile index
    const uint i_h = tgid.y; // head
    const uint i_b = tgid.z; // batch
    const uint tid = tid3.x;

    if (i_v * BV >= V || i_h >= H) return;

    // Stub: zero the per-(batch, head, v-tile) slice of h_out, v_new,
    // and final_state. The tests' first assertion (FLA reference parity)
    // will fail with a 5e-3 tolerance violation, demonstrating the test
    // is wired but the kernel body is not yet correct.

    // Zero h_out[b, *, i_h, i_v_tile, *]    — [NT, BV, K] bf16
    for (uint i_t = 0; i_t < NT; ++i_t) {
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            const uint vv = flat / K;        // 0..BV
            const uint kk = flat - vv * K;   // 0..K
            const uint v_idx = i_v * BV + vv;
            if (v_idx < V) {
                const uint h_off = ((((i_b * NT) + i_t) * H + i_h) * V + v_idx) * K + kk;
                h_out[h_off] = bfloat(0.0f);
            }
        }
    }

    // Zero v_new[b, *, i_h, i_v_tile]       — [T, BV] bf16
    for (uint t = 0; t < T; ++t) {
        for (uint vv = tid; vv < BV; vv += TG_THREADS) {
            const uint v_idx = i_v * BV + vv;
            if (v_idx < V) {
                const uint vn_off = (((i_b * T) + t) * H + i_h) * V + v_idx;
                v_new[vn_off] = bfloat(0.0f);
            }
        }
    }

    // Zero final_state[b, i_h, i_v_tile, *] — [BV, K] f32
    for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
        const uint vv = flat / K;
        const uint kk = flat - vv * K;
        const uint v_idx = i_v * BV + vv;
        if (v_idx < V) {
            const uint fs_off = ((i_b * H + i_h) * V + v_idx) * K + kk;
            final_state[fs_off] = 0.0f;
        }
    }
}
