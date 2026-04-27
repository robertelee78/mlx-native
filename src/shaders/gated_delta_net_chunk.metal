#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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
// # Wave 5b.2 iter 1 — simdgroup_matrix MMA on BOTH matmul sections
//
//   Both section 2b (`bv = u - w @ bh^T`) and section 2f (`bh += bv^T @ k`)
//   use `simdgroup_matrix<float, 8, 8>` 8×8 hardware MMA on Apple Silicon
//   AMX/Tensor units. Per-thread MAC loops in the prior implementation
//   underused the simdgroup matrix engines.
//
//   Tile partitioning (BT=64, BV=32, K=128, TG_THREADS=128 = 4 simdgroups):
//
//   Section 2b: `bv[BT, BV] = u - w @ bh^T`
//     * Each simdgroup `s` owns BT rows [s*16, s*16+16) (2 row-tiles of 8)
//       × all 4 BV col-tiles → 8 accumulator fragments per simdgroup.
//     * K-reduction: 16 8×8 K-tiles. Per K-tile: load 2 w-frags (bf16 from
//       device, K-row-stride = w_t_stride) + 4 bh-frags (f32 from
//       threadgroup, transposed via `simdgroup_load(..., true)`); 8 MMAs.
//     * After loop, extract per-thread fragment elements and compute
//       `local_v[c] = u_val - bv_acc[c]` cell-wise (16 cells/thread).
//
//   Section 2f: `bh[BV, K] += bv_stage @ k`
//     * Each simdgroup `s` owns BV rows [s*8, s*8+8) (1 row-tile of 8)
//       × all 16 K col-tiles → 16 accumulator fragments per simdgroup.
//     * Initial accumulator load: bh frags from threadgroup memory
//       (already gated by exp_g_last in section 2d).
//     * BT-reduction: 8 8×8 BT-tiles. Per BT-tile: load 1 bv-frag (bf16
//       from threadgroup, layout `[BV, BT]`) + 16 k-frags (bf16 from
//       device, K-row-stride = k_t_stride); 16 MMAs.
//     * After loop, store 16 bh frags back to threadgroup `bh`.
//
//   Per-thread fragment-element mapping (BaseMMAFrag<T, 8, 8>::get_coord):
//     qid = lane_id / 4
//     fm  = (qid & 4) + (lane_id / 2) % 4
//     fn  = (qid & 2) * 2 + (lane_id % 2) * 2
//   Each lane owns 2 elements per 8×8 frag at row=fm, cols=fn,fn+1.
//
//   For section 2b's per-thread `local_v[c]` (c ∈ 0..16):
//     i = c / 8                  (row-tile within sg, 0..1)
//     j = (c % 8) / 2            (col-tile, 0..3)
//     e = c % 2                  (lane sub-element, 0..1)
//     bt = sgid*16 + i*8 + fm
//     bv_idx = j*8 + fn + e
//
// # Numerical precision
//
//   Inputs / stored intermediates: bf16. Accumulators: f32. The bh state
//   stays in f32 in shared memory across the T-loop (matches FLA's policy
//   at chunk_delta_h.py:85 — `b_h1 = tl.zeros([BV, 64], dtype=tl.float32)`).
//
//   The 8×8 MMA reduction is deterministic across runs; the lane-element
//   ordering within an 8×8 frag is fixed by Apple Silicon's hardware
//   tensor unit. This gives bit-equivalent or strictly better numerical
//   error vs the per-thread serial sum (FLA-line-255 oracle still passes
//   at 0.0e0).
//
// # bf16 round-trip on bv (FLA parity, post-gate placement)
//
//   FLA places the only bf16 round-trip on b_v at line 255
//   (`b_v = b_v.to(k.dtype.element_ty)`) — AFTER the gate multiply at
//   line 213, BEFORE the outer-update dot at line 261. The cast must
//   sit between gate and dot, not before either.
//
//   The kernel mirrors that ordering exactly:
//     1. local_v = u - w @ bh^T              (f32, MMA in 2b)
//     2. local_v *= exp(g_last - g_blk)      (FLA :213, gate in f32)
//     3. bh      *= exp(g_last)              (FLA :215, gate in f32)
//     4. bv_stage[*] = bf16(local_v)         (FLA :255, post-gate cast)
//     5. bh += bv_stage @ b_k                (FLA :261, outer dot reads
//                                             bf16-rounded values, MMA in 2f)
//
//   Wave5b.1 iter1.5 corrected this from an earlier ordering that
//   bf16-rounded BEFORE the gate; that diverged from FLA by ~6e-4 on
//   final_state and was caught by the FLA-line-255 oracle in
//   tests/fixtures/gated_delta_net_chunk_oracle.py.
//
// # Threadgroup memory layout (post-iter-1 simdgroup_matrix MMA)
//
//   We allocate (BV*K * 4) + (BV*BT * 2) bytes = 16 KB + 4 KB = 20 KB at
//   threadgroup(0). The first BV*K floats are `bh` (the running f32 state
//   tile); the next BV*BT bfloats are `bv_stage` (used to publish per-thread
//   bv values to all threads for the outer-update). bv_stage layout is
//   `[BV, BT]` (transposed from the prior `[BT, BV]` f32 layout — this
//   saves 4 KB threadgroup memory and lets section 2f's MMA load
//   `simdgroup_matrix<bfloat>` frags directly without transpose).
//   20 KB < 32 KB M5 Max max threadgroup memory.
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
constant uint NSG = 4u;          // simdgroups per threadgroup (= TG_THREADS / 32)

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
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort slid [[thread_index_in_simdgroup]]
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

    // Threadgroup-memory partition (20 KB total — host allocates exactly
    // this much; see gated_delta_net_chunk.rs `dispatch_*`).
    //   shared_mem[0          .. BV*K)            = bh          [BV, K]   f32 (16 KB)
    //   shared_mem[BV*K bytes .. BV*K+BV*BT bytes)= bv_stage_bf [BV, BT]  bf16 (4 KB)
    threadgroup float  *bh          = shared_mem;
    threadgroup bfloat *bv_stage_bf =
        (threadgroup bfloat *)(shared_mem + (BV * K));

    // simdgroup-frag lane coordinates (BaseMMAFrag<T, 8, 8>::get_coord).
    const ushort qid = slid / 4;
    const ushort fm  = (qid & 4) + ((slid / 2) % 4);
    const ushort fn  = (qid & 2) * 2 + (slid % 2) * 2;

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

    // Per-thread accumulator for bv during phase 2b — populated from MMA
    // fragment elements after the K-reduction MMA loop.
    //
    // CELLS_PER_THREAD = 16: 2 row-tiles × 4 col-tiles × 2 elements.
    // c-mapping: c = i*8 + j*2 + e where i = row-tile (0..1), j = col-tile
    // (0..3), e = lane sub-element (0..1).
    //   bt_local = i*8 + fm                (within sg)
    //   bv_local = j*8 + fn + e
    constexpr uint CELLS_PER_THREAD = 16u;
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
        // 2b. Compute bv = u - w @ bh^T  via simdgroup_matrix<float, 8, 8>
        //     8×8 MMA. Each simdgroup owns BT rows [sgid*16, sgid*16+16)
        //     (2 row-tiles) × all 4 BV col-tiles = 8 fragments.
        // -----------------------------------------------------------
        const uint w_base = i_b * w_seq_stride + (t_start * w_t_stride) + i_h * K;
        const uint u_base = i_b * u_seq_stride + (t_start * u_t_stride) + i_h * V;

        // 8 accumulator frags per simdgroup, indexed [i*4 + j].
        simdgroup_matrix<float, 8, 8> bv_acc[8];
        for (uint t = 0; t < 8; ++t) {
            bv_acc[t] = simdgroup_matrix<float, 8, 8>(0);
        }

        // K-reduction: 16 K-tiles of 8.
        const uint num_k_tiles = K / 8u;
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k_origin = kt * 8u;

            simdgroup_barrier(mem_flags::mem_none);

            // Load 2 w fragments — sgid's BT row-tiles (2 of them).
            //   w[B,T,H,K] row-major K-innermost; bt-row stride = w_t_stride.
            //   Frag i covers BT rows [sgid*16 + i*8, sgid*16 + i*8 + 8).
            simdgroup_matrix<bfloat, 8, 8> w_frag[2];
            const device bfloat *w_ptr_base =
                w + w_base + sgid * 16u * w_t_stride + k_origin;
            simdgroup_load(w_frag[0], w_ptr_base + 0u * 8u * w_t_stride,
                           w_t_stride, 0, false);
            simdgroup_load(w_frag[1], w_ptr_base + 1u * 8u * w_t_stride,
                           w_t_stride, 0, false);

            // Load 4 bh^T fragments — all 4 BV col-tiles.
            //   bh layout is [BV, K] row-major (K-innermost, row stride = K).
            //   bh^T frag at (k_origin..k_origin+8, bv_origin..bv_origin+8)
            //   reads bh[bv_origin+j][k_origin+i] which is bh + bv_origin*K
            //   + k_origin with row stride K and transpose=true.
            simdgroup_matrix<float, 8, 8> bh_frag_T[4];
            for (uint j = 0; j < 4; ++j) {
                const uint bv_origin = j * 8u;
                threadgroup const float *bh_ptr =
                    bh + bv_origin * K + k_origin;
                simdgroup_load(bh_frag_T[j], bh_ptr, K, 0, true);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 8 MMAs: bv_acc[i*4+j] += w_frag[i] @ bh_frag_T[j].
            for (uint i = 0; i < 2; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(
                        bv_acc[i * 4 + j], w_frag[i], bh_frag_T[j],
                        bv_acc[i * 4 + j]);
                }
            }
        }

        // Extract per-thread fragment elements into local_v and subtract u.
        // c-mapping: c = i*8 + j*2 + e (see CELLS_PER_THREAD comment).
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) {
                thread auto &elems = bv_acc[i * 4 + j].thread_elements();
                const uint bt = sgid * 16u + i * 8u + fm;
                const uint bv0 = j * 8u + fn;
                // u_val for the two elements at (bt, bv0) and (bt, bv0+1).
                float u0 = 0.0f, u1 = 0.0f;
                const uint vidx0 = v_base + bv0;
                const uint vidx1 = v_base + bv0 + 1u;
                if (vidx0 < V) {
                    u0 = float(u[u_base + bt * u_t_stride + vidx0]);
                }
                if (vidx1 < V) {
                    u1 = float(u[u_base + bt * u_t_stride + vidx1]);
                }
                const uint c0 = i * 8u + j * 2u + 0u;
                const uint c1 = i * 8u + j * 2u + 1u;
                local_v[c0] = u0 - float(elems[0]);
                local_v[c1] = u1 - float(elems[1]);
            }
        }

        // -----------------------------------------------------------
        // 2c. Store bf16(bv) -> v_new (BEFORE the gate multiply, matching
        //     FLA's tl.store at line 199-203 — the stored v_new is
        //     u - w @ h^T without the gate scaling).
        // -----------------------------------------------------------
        const uint vn_base = i_b * u_seq_stride + (t_start * u_t_stride) + i_h * V;
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) {
                const uint bt = sgid * 16u + i * 8u + fm;
                const uint bv0 = j * 8u + fn;
                const uint c0 = i * 8u + j * 2u + 0u;
                const uint c1 = i * 8u + j * 2u + 1u;
                const uint vidx0 = v_base + bv0;
                const uint vidx1 = v_base + bv0 + 1u;
                if (vidx0 < V) {
                    v_new[vn_base + bt * u_t_stride + vidx0] = bfloat(local_v[c0]);
                }
                if (vidx1 < V) {
                    v_new[vn_base + bt * u_t_stride + vidx1] = bfloat(local_v[c1]);
                }
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

        // Per-thread gate of local_v. Each thread holds 16 cells across
        // 2 distinct BT rows (i = 0, 1) — so two distinct g_t values.
        for (uint i = 0; i < 2; ++i) {
            const uint bt = sgid * 16u + i * 8u + fm;
            const float g_t = g[g_base + (t_start + bt) * g_t_stride];
            const float scale = metal::exp(g_last - g_t);
            for (uint j = 0; j < 4; ++j) {
                const uint c0 = i * 8u + j * 2u + 0u;
                const uint c1 = i * 8u + j * 2u + 1u;
                local_v[c0] *= scale;
                local_v[c1] *= scale;
            }
        }

        // Scale bh by exp_g_last in place (cooperative across threads).
        for (uint flat = tid; flat < BV * K; flat += TG_THREADS) {
            bh[flat] *= exp_g_last;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -----------------------------------------------------------
        // 2e. Publish bf16-rounded local_v to bv_stage_bf (FLA :255).
        //     Layout `[BV, BT]` (transposed): bv_stage_bf[bv * BT + bt].
        //     This lets section 2f's MMA load bf16 frags directly without
        //     a transpose flag.
        // -----------------------------------------------------------
        for (uint i = 0; i < 2; ++i) {
            for (uint j = 0; j < 4; ++j) {
                const uint bt = sgid * 16u + i * 8u + fm;
                const uint bv0 = j * 8u + fn;
                const uint c0 = i * 8u + j * 2u + 0u;
                const uint c1 = i * 8u + j * 2u + 1u;
                bv_stage_bf[bv0 * BT + bt]      = bfloat(local_v[c0]);
                bv_stage_bf[(bv0 + 1u) * BT + bt] = bfloat(local_v[c1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -----------------------------------------------------------
        // 2f. Outer-update: bh += bv_stage_bf @ k  (MMA).
        //     Each simdgroup `s` owns BV rows [s*8, s*8+8) (1 row-tile)
        //     × all 16 K col-tiles → 16 accumulator fragments per sg.
        //     Initial accumulator = (already-gated) bh frags loaded from
        //     threadgroup memory.
        // -----------------------------------------------------------
        const uint k_chunk_base =
            i_b * k_seq_stride + t_start * k_t_stride + kh * K;

        // Load 16 bh accumulator frags (BV row-tile = sgid, all K col-tiles).
        //   Tile count is hard-coded for K=128 (= 16 8-element K-tiles). The
        //   compile-time loop bound is LOAD-BEARING — Metal's MMA scheduler
        //   needs unrollable loops for the simdgroup_matrix tile sequence;
        //   making the bound runtime via `K/8` collapses inter_state from
        //   1.08 ms to 3.40 ms (3.15× regression, measured 2026-04-27).
        //   So MAX_K is narrowed to 128 in src/ops/gated_delta_net_chunk.rs;
        //   K=192 dispatch is rejected by validate(). To support K=192/256
        //   later, port FLA's b_h1..b_h4 bank-split (chunk_delta_h.py:215-221)
        //   which keeps each kernel's K-tile count compile-time-known per bank.
        simdgroup_matrix<float, 8, 8> bh_acc[16];
        {
            threadgroup const float *bh_row_ptr = bh + sgid * 8u * K;
            for (uint kj = 0; kj < 16; ++kj) {
                simdgroup_load(bh_acc[kj], bh_row_ptr + kj * 8u, K, 0, false);
            }
        }

        // BT-reduction: 8 BT-tiles of 8.
        const uint num_bt_tiles = BT / 8u;
        for (uint bj = 0; bj < num_bt_tiles; ++bj) {
            const uint bt_origin = bj * 8u;

            simdgroup_barrier(mem_flags::mem_none);

            // Load 1 bv frag — sgid's BV row-tile, 8 BT cols.
            //   bv_stage_bf layout `[BV, BT]` row-major (BT innermost,
            //   row stride = BT). Frag covers BV rows [sgid*8, sgid*8+8)
            //   × BT cols [bt_origin, bt_origin+8).
            simdgroup_matrix<bfloat, 8, 8> bv_frag;
            simdgroup_load(bv_frag,
                           bv_stage_bf + sgid * 8u * BT + bt_origin,
                           BT, 0, false);

            // Load 16 k frags — 8 BT rows × 8 K cols each.
            //   k[B,T,Hg,K] row-major K-innermost; bt-row stride = k_t_stride.
            simdgroup_matrix<bfloat, 8, 8> k_frag[16];
            const device bfloat *k_row_base =
                k + k_chunk_base + bt_origin * k_t_stride;
            for (uint kj = 0; kj < 16; ++kj) {
                simdgroup_load(k_frag[kj], k_row_base + kj * 8u,
                               k_t_stride, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 16 MMAs: bh_acc[kj] += bv_frag @ k_frag[kj].
            for (uint kj = 0; kj < 16; ++kj) {
                simdgroup_multiply_accumulate(
                    bh_acc[kj], bv_frag, k_frag[kj], bh_acc[kj]);
            }
        }

        // Store bh frags back to threadgroup memory.
        {
            threadgroup float *bh_row_ptr = bh + sgid * 8u * K;
            for (uint kj = 0; kj < 16; ++kj) {
                simdgroup_store(bh_acc[kj], bh_row_ptr + kj * 8u, K, 0, false);
            }
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
