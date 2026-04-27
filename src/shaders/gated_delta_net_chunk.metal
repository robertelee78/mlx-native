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
//   bh := h0[b, i_h, BV-rows-of-tile, :]        # [BV, K] f32 in shared mem
//   for i_t in 0..NT:
//       store bh -> h_out[b, i_t, i_h, BV-rows, :]    # bf16 store
//       bv  := u[b, t_chunk, i_h, BV-rows]            # [BT, BV] f32
//             - w[b, t_chunk, i_h, :] @ bh^T          # [BT, K] @ [K, BV]
//       store bf16(bv) -> v_new[b, t_chunk, i_h, BV-rows]
//
//       last_t := (i_t+1)*BT - 1
//       g_last := g[b, last_t, i_h]                   # f32 scalar
//       g_blk  := g[b, t_start..t_end, i_h]           # f32 [BT]
//       bv     := bf16-roundtrip(bv) * exp(g_last - g_blk)[:, None]
//       bh     := bh * exp(g_last)
//
//       bh += bv.T @ k[b, t_chunk, i_h/group_ratio, :]  # outer accumulate
//
//   final_state[b, i_h, BV-rows, :] := bh             # f32 store
//
// # Numerical precision
//
//   Inputs / stored intermediates: bf16. Accumulators: f32. The bh state
//   stays in f32 in shared memory across the T-loop (matches FLA's policy
//   at chunk_delta_h.py:85 — `b_h1 = tl.zeros([BV, 64], dtype=tl.float32)`).
//
// # bf16 round-trip on bv (FLA parity, post-gate placement)
//
//   FLA places the only bf16 round-trip on b_v at line 255
//   (`b_v = b_v.to(k.dtype.element_ty)`) — AFTER the gate multiply at
//   line 213, BEFORE the outer-update dot at line 261. The cast must
//   sit between gate and dot, not before either.
//
//   The kernel mirrors that ordering exactly:
//     1. local_v = u - w @ bh^T              (f32)
//     2. local_v *= exp(g_last - g_blk)      (FLA :213, gate in f32)
//     3. bh      *= exp(g_last)              (FLA :215, gate in f32)
//     4. bv_stage[*] = bf16_round(local_v)   (FLA :255, post-gate cast)
//     5. bh += bv_stage^T @ b_k              (FLA :261, outer dot reads
//                                             bf16-rounded values)
//
//   Wave5b.1 iter1.5 corrected this from an earlier ordering that
//   bf16-rounded BEFORE the gate; that diverged from FLA by ~6e-4 on
//   final_state and was caught by the FLA-line-255 oracle in
//   tests/fixtures/gated_delta_net_chunk_oracle.py.
//
// # Threadgroup memory layout
//
//   We allocate (BT*BV + BV*K) * 4 bytes = (2048 + 4096) * 4 = 24 KB at
//   threadgroup(0). The first BT*BV floats are `bv_stage` (used to
//   publish per-thread bv values to all threads for the outer-update);
//   the next BV*K floats are `bh` (the running f32 state tile).
//   24 KB < 32 KB M5 Max max threadgroup memory.
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
//   Threadgroup: TG_THREADS = 128 (4 simdgroups × 32 lanes), flat 1D.
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

constant uint BV = 32u;          // V-tile width (per-threadgroup V slice)
constant uint TG_THREADS = 128u; // threadgroup size

// bf16 round-trip — cast to bfloat then back to float to truncate
// f32 -> bf16 -> f32 (matches PyTorch `.to(bfloat16).float()`).
inline float bf16_round(float x) {
    return float(bfloat(x));
}

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
    const uint B   = params[0];
    const uint T   = params[1];
    const uint Hg  = params[2];
    const uint H   = params[3];
    const uint K   = params[4];
    const uint V   = params[5];
    const uint BT  = params[6];
    const uint NT  = params[7];

    const uint i_v = tgid.x; // V-tile index
    const uint i_h = tgid.y; // V-head
    const uint i_b = tgid.z; // batch
    const uint tid = tid3.x;

    if (i_b >= B || i_h >= H) return;
    if (i_v * BV >= V) return;

    const uint v_base       = i_v * BV;          // first V-row in tile
    const uint group_ratio  = H / Hg;
    const uint kh           = i_h / group_ratio; // GQA-mapped K-head

    // Strides (in elements).
    const uint k_t_stride   = Hg * K;
    const uint k_seq_stride = T * k_t_stride;
    const uint w_t_stride   = H * K;
    const uint w_seq_stride = T * w_t_stride;
    const uint u_t_stride   = H * V;
    const uint u_seq_stride = T * u_t_stride;
    const uint g_t_stride   = H;
    const uint g_seq_stride = T * g_t_stride;
    const uint state_head_stride = V * K;
    const uint state_seq_stride  = H * state_head_stride;
    const uint h_chunk_stride    = H * state_head_stride;
    const uint h_seq_stride      = NT * h_chunk_stride;

    // Threadgroup-memory partition (24 KB total — host allocates exactly
    // this much; see gated_delta_net_chunk.rs `dispatch_*`).
    //   shared_mem[0          .. BT*BV)        = bv_stage    [BT, BV]  f32
    //   shared_mem[BT*BV      .. BT*BV + BV*K) = bh          [BV, K]   f32
    threadgroup float *bv_stage = shared_mem;
    threadgroup float *bh       = shared_mem + (BT * BV);

    // ===================================================================
    // 1. Load h0 -> bh.
    // ===================================================================
    {
        const uint h0_base = i_b * state_seq_stride + i_h * state_head_stride;
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            const uint vv = flat / K;
            const uint kk = flat - vv * K;
            const uint vidx = v_base + vv;
            float val = 0.0f;
            if (vidx < V) {
                val = h0[h0_base + vidx * K + kk];
            }
            bh[vv * K + kk] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread accumulator for bv during phase 2b.  We don't persist
    // it across the loop — it's regenerated each chunk.
    constexpr uint CELLS_BV          = 64u * 32u;       // BT * BV
    constexpr uint CELLS_PER_THREAD  = CELLS_BV / TG_THREADS; // 16
    thread float local_v[CELLS_PER_THREAD];

    // ===================================================================
    // 2. Per-chunk loop.
    // ===================================================================
    for (uint i_t = 0; i_t < NT; ++i_t) {
        const uint t_start = i_t * BT;
        const uint last_t  = t_start + BT - 1;

        // -----------------------------------------------------------
        // 2a. Snapshot bh -> h_out.
        // -----------------------------------------------------------
        const uint h_out_base =
            i_b * h_seq_stride + i_t * h_chunk_stride + i_h * state_head_stride;
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            const uint vv = flat / K;
            const uint kk = flat - vv * K;
            const uint vidx = v_base + vv;
            if (vidx < V) {
                h_out[h_out_base + vidx * K + kk] = bfloat(bh[vv * K + kk]);
            }
        }
        // No barrier needed — bh is read-only here.

        // -----------------------------------------------------------
        // 2b. Compute bv = u_chunk - w_chunk @ bh^T.
        //     Each thread owns CELLS_PER_THREAD (bt, bv) cells, walked
        //     by `flat = c * TG_THREADS + tid`.
        // -----------------------------------------------------------
        const uint w_base = i_b * w_seq_stride + (t_start * w_t_stride) + i_h * K;
        const uint u_base = i_b * u_seq_stride + (t_start * u_t_stride) + i_h * V;

        for (uint c = 0; c < CELLS_PER_THREAD; ++c) {
            const uint flat = c * TG_THREADS + tid;
            const uint bt_idx = flat / BV;
            const uint bv_idx = flat - bt_idx * BV;
            const uint vidx   = v_base + bv_idx;

            float u_val = 0.0f;
            if (vidx < V) {
                u_val = float(u[u_base + bt_idx * u_t_stride + vidx]);
            }

            float dot = 0.0f;
            for (uint kk = 0; kk < K; ++kk) {
                const float w_val  = float(w[w_base + bt_idx * w_t_stride + kk]);
                const float bh_val = bh[bv_idx * K + kk];
                dot += w_val * bh_val;
            }
            local_v[c] = u_val - dot;
        }

        // -----------------------------------------------------------
        // 2c. Store bf16(bv) -> v_new (BEFORE the gate multiply, matching
        //     FLA's tl.store at line 199-203 — the stored v_new is
        //     u - w @ h^T without the gate scaling).
        // -----------------------------------------------------------
        const uint vn_base = i_b * u_seq_stride + (t_start * u_t_stride) + i_h * V;
        for (uint c = 0; c < CELLS_PER_THREAD; ++c) {
            const uint flat = c * TG_THREADS + tid;
            const uint bt_idx = flat / BV;
            const uint bv_idx = flat - bt_idx * BV;
            const uint vidx   = v_base + bv_idx;
            if (vidx < V) {
                v_new[vn_base + bt_idx * u_t_stride + vidx] = bfloat(local_v[c]);
            }
        }

        // -----------------------------------------------------------
        // 2d. Gate multiplies (FLA chunk_delta_h.py:213, :215).
        //     local_v[c] *= exp(g_last - g_blk[bt])   (f32, pre-bf16-round)
        //     bh[*]      *= exp(g_last)
        // -----------------------------------------------------------
        const uint g_base = i_b * g_seq_stride + i_h;
        const float g_last = g[g_base + last_t * g_t_stride];
        const float exp_g_last = metal::exp(g_last);

        for (uint c = 0; c < CELLS_PER_THREAD; ++c) {
            const uint flat = c * TG_THREADS + tid;
            const uint bt_idx = flat / BV;
            const float g_t = g[g_base + (t_start + bt_idx) * g_t_stride];
            const float scale = metal::exp(g_last - g_t);
            local_v[c] = local_v[c] * scale; // FLA :213 — gate in f32, no pre-round
        }

        // Scale bh by exp_g_last in place (cooperative across threads).
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            bh[flat] *= exp_g_last;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -----------------------------------------------------------
        // 2e. Publish bf16-rounded local_v to bv_stage (FLA chunk_delta_h.py:255).
        //     The bf16 round-trip happens HERE (post-gate, pre-outer-dot),
        //     not before the gate. The outer dot in 2f reads the
        //     bf16-rounded values from bv_stage.
        // -----------------------------------------------------------
        for (uint c = 0; c < CELLS_PER_THREAD; ++c) {
            const uint flat = c * TG_THREADS + tid;
            bv_stage[flat] = bf16_round(local_v[c]); // FLA :255 — post-gate bf16 cast
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -----------------------------------------------------------
        // 2f. Outer-update: bh[bv, kk] += sum_bt(bv_stage[bt, bv] * k[bt, kk]).
        //     Each thread owns BV*K / TG_THREADS = 32 cells.
        // -----------------------------------------------------------
        const uint k_chunk_base = i_b * k_seq_stride + t_start * k_t_stride + kh * K;
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            const uint bv_idx = flat / K;
            const uint kk     = flat - bv_idx * K;
            float acc = 0.0f;
            for (uint bt_idx = 0; bt_idx < BT; ++bt_idx) {
                const float bv_val = bv_stage[bt_idx * BV + bv_idx];
                const float bk_val = float(k[k_chunk_base + bt_idx * k_t_stride + kk]);
                acc += bv_val * bk_val;
            }
            bh[flat] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================================================================
    // 3. Store final state.
    // ===================================================================
    {
        const uint fs_base = i_b * state_seq_stride + i_h * state_head_stride;
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            const uint vv = flat / K;
            const uint kk = flat - vv * K;
            const uint vidx = v_base + vv;
            if (vidx < V) {
                final_state[fs_base + vidx * K + kk] = bh[vv * K + kk];
            }
        }
    }
}
