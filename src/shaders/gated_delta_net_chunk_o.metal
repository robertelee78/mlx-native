#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Wave 5b.1 iter 3 — chunk_fwd_o Metal kernel.
// Wave 5b.2 iter 2 — simdgroup_matrix<float, 8, 8> MMA on three matmul
// sections (q · h^T, q · k^T, b_A · v_new), mirroring the inter_state
// iter 1 recipe.
//
// Spec source:
// - FLA reference: `chunk_fwd_kernel_o` at
//   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py:42-138
//
// No FLA / Triton / CUDA / Metal code is copied. The math here is a
// re-derivation from the FLA spec; the algorithmic structure (load q/k/h,
// dot-and-accumulate across K-tiles, gate, mask, bf16 cast, dot with v)
// follows the FLA Triton kernel pattern but is open-coded for Metal.
//
// # Algorithm (per (batch b, V-head i_h, chunk i_t, V-tile i_v))
//
//   kh         = i_h / (H / Hg)                           # GQA-mapped K-head
//   bo_acc     = zeros([BT, BV])                          # f32 in MMA frags
//   bA_acc     = zeros([BT, BT])                          # f32 in MMA frags
//
//   # K-reduction (FLA :93-113), FUSED — both sections share the same q-load.
//   for kt in 0..(K // 8):
//       bo_acc[BT, BV] += q[BT, kt..] @ trans(h[BV, kt..])      # FLA :111
//       bA_acc[BT, BT] += q[BT, kt..] @ trans(k[BT, kt..])      # FLA :113
//
//   # Gate (FLA :115-120; USE_G == True for GDN):
//   bo_acc[i, j] *= exp(g[t_start + i])
//   bA_acc[i, j] *= exp(g[t_start + i] - g[t_start + j])
//
//   # Causal+diag mask (FLA :122-125 — INCLUSIVE of diagonal, `>=`):
//   bA_acc[i, j] = (i >= j) ? bA_acc[i, j] : 0
//
//   # bf16 round-trip on bA_acc (FLA :137 — AFTER mask, BEFORE dot with v):
//   bA_stage[i, j] = bfloat(bA_acc[i, j])  # publish to threadgroup mem
//
//   # Closing dot — bf16 b_A × bf16 v_new → f32 acc (MMA), then scale + store:
//   for bt_tile in 0..(BT // 8):
//       bo_acc[BT, BV] += bA_stage[BT, bt_tile..] @ v_new[bt_tile.., BV]
//   o[BT, BV] = bfloat(bo_acc * scale)                    # FLA :137
//
// # Wave 5b.2 iter 2 — simdgroup_matrix MMA on THREE matmul sections
//
//   All three matmul sections (2A: q·h^T over K, 2B: q·k^T over K,
//   2C: bA_bf16·v_new over BT) use `simdgroup_matrix<float, 8, 8>` 8×8
//   hardware MMA on Apple Silicon AMX/Tensor units. Per-thread MAC loops
//   in iter 3 underused the simdgroup matrix engines; iter 1's lesson on
//   inter_state (17.4× speedup) transfers directly here, with the bonus
//   that 2A and 2B share the same q-frag per K-tile (single q load amortized
//   across 4 BV-col MMAs + 8 BT-col MMAs = 12 MMAs/load).
//
//   Tile partitioning (BT=64, BV=32, K=128, TG_THREADS=256 = 8 simdgroups):
//
//   Section 2A: `bo_acc[BT, BV] += q · h^T`
//     * Each simdgroup `s` owns BT row [s*8, s*8+8) (1 row-tile)
//       × all 4 BV col-tiles → 4 accumulator fragments per simdgroup.
//     * K-reduction: 16 8×8 K-tiles. Per K-tile: load 1 q-frag (bf16,
//       sgid's BT row, 8 K cols, K-row-stride = qk_t_stride) + 4 h-frags
//       (bf16, h is [V, K] row-major K-innermost; transpose-load via
//       `simdgroup_load(..., true)` produces h^T tile [K, BV])
//       — wait: simdgroup_load reads as [rows, cols] in source layout.
//       For h^T frag covering (k_origin..+8, bv_origin..+8) we read
//       h[bv_origin+j][k_origin+i] which is base + bv_origin*K + k_origin
//       with row stride K and `transpose=true`. 4 MMAs.
//
//   Section 2B: `bA_acc[BT, BT] += q · k^T` (FUSED into 2A's K-loop)
//     * Each simdgroup `s` owns BT row [s*8, s*8+8) (same as 2A)
//       × all 8 BT col-tiles → 8 accumulator fragments per simdgroup.
//     * Per K-tile (already inside 2A's loop, q-frag reused):
//       load 8 k^T-frags (bf16, k is [BT, K] row-major K-innermost,
//       transpose-load via `simdgroup_load(..., true)`); 8 MMAs.
//
//   Section 2C: `bo_acc += bA_bf16 · v_new`
//     * Each simdgroup `s` owns same BT row-tile [s*8, s*8+8)
//       × all 4 BV col-tiles → 4 accumulator fragments per simdgroup
//       (bo_acc reused from 2A — already gated, scale applied at store).
//     * BT-reduction: 8 8×8 BT-tiles. Per BT-tile: load 1 bA_bf16 frag
//       (bf16, bA_stage layout [BT, BT] row-major, BT cols innermost,
//       row stride = BT) + 4 v-frags (bf16, v_new is [BT, BV] within
//       a chunk-V-tile, row stride = v_t_stride); 4 MMAs.
//
//   Per-thread fragment-element mapping (BaseMMAFrag<T, 8, 8>::get_coord):
//     qid = lane_id / 4
//     fm  = (qid & 4) + (lane_id / 2) % 4
//     fn  = (qid & 2) * 2 + (lane_id % 2) * 2
//   Each lane owns 2 elements per 8×8 frag at row=fm, cols=fn,fn+1.
//
// # Numerical precision
//
//   Inputs / stored intermediates: bf16. Accumulators: f32. The 8×8 MMA
//   reduction is deterministic across runs; the lane-element ordering
//   within an 8×8 frag is fixed by Apple Silicon's hardware tensor unit.
//   This gives bit-equivalent or strictly better numerical error vs the
//   per-thread serial sum (FLA-line-137 oracle still passes at < 1e-6).
//
// # CRITICAL ORDERING DETAILS (per iter-1.5 lesson)
//
// 1. The bf16 cast of bA_acc happens AFTER the mask, BEFORE the dot with
//    v (FLA line 137). Moving or skipping it changes numerics. The MMA
//    reformulation preserves this: bA_acc lives in f32 frags during 2B,
//    is gated/masked in f32 (per-thread element extract), then bf16-cast
//    and published to threadgroup `bA_stage` BEFORE 2C reads it.
// 2. The mask is `>=` (causal+diag) — NOT strict `>` like kkt. The
//    diagonal element corresponds to attention to the same token and is
//    included.
// 3. Two `* scale` multiplications on FLA line 137: BOTH `bo_acc * scale`
//    AND `dot_result * scale`. Mathematically equivalent to one global
//    scale on the sum; we apply scale once at the final bf16 store.
// 4. h dimension order is [V, K] per (chunk, head) — `b_o += dot(b_q,
//    trans(b_h))` per FLA :111.
//
// # Threadgroup memory budget (post-iter-2 MMA, BT=64)
//
//   bA_stage : BT * BT * 2 bytes (bf16) = 64 * 64 * 2 = 8 KB
//   Total: 8 KB (down from 28 KB iter 3 — bo_acc and ba_acc now live in
//   simdgroup-matrix accumulator registers, not threadgroup memory).
//
// # K-tile count is hard-coded — DO NOT generalize
//
//   The K-reduction loop runs `num_k_tiles = 16` (= K/8 at K=128). The
//   compile-time loop bound is LOAD-BEARING — Metal's MMA scheduler needs
//   unrollable loops for the simdgroup_matrix tile sequence; making the
//   bound runtime via `K/8u` collapsed inter_state's section 2f from
//   1.08 ms to 3.40 ms (3.15× regression, measured 2026-04-27, iter 1.5).
//   So MAX_K is narrowed to 128 in src/ops/gated_delta_net_chunk_o.rs;
//   K=192/256 dispatch is rejected by validate(). To support K=192/256
//   later, port FLA's bank-split which keeps each kernel's K-tile count
//   compile-time-known per bank.
//
// # Memory layouts (innermost-first)
//
//   q:     [B, T, Hg, K]    bf16  — K innermost
//   k:     [B, T, Hg, K]    bf16  — K innermost
//   v_new: [B, T, H,  V]    bf16  — V innermost
//   h:     [B, NT, H, V, K] bf16  — K innermost (per chunk-head)
//   g:     [B, T, H]        f32   — H innermost (chunk-cumsumed)
//   o:     [B, T, H, V]     bf16  — V innermost (output)
//
// # Threading
//
//   Grid: (NV, NT, B*H) where NV = V/BV
//        program_id(0) -> i_v (V-tile index)
//        program_id(1) -> i_t (chunk index)
//        program_id(2) -> i_bh (flattened batch * head)
//          -> i_b = i_bh / H,  i_h = i_bh % H
//
//   Threadgroup: TG_THREADS = 256 threads (8 simdgroups × 32 lanes).
//
// # Buffer bindings
//
//   buffer(0): q        bf16
//   buffer(1): k        bf16
//   buffer(2): v        bf16  (= v_new from iter 1)
//   buffer(3): h        bf16
//   buffer(4): g        f32
//   buffer(5): o        bf16  (output)
//   buffer(6): params   uint[11] = [B, T, Hg, H, K, V, BT, NT, BK, BV]
//                                   plus float[1] scale at byte offset 40.

constant uint TG_THREADS = 256u;
constant uint NSG = 8u;        // simdgroups per threadgroup (= TG_THREADS / 32)

kernel void gated_delta_net_chunk_o_bf16(
    device const bfloat *q          [[buffer(0)]],
    device const bfloat *k          [[buffer(1)]],
    device const bfloat *v          [[buffer(2)]],
    device const bfloat *h          [[buffer(3)]],
    device const float  *g          [[buffer(4)]],
    device bfloat       *o          [[buffer(5)]],
    device const uint   *params     [[buffer(6)]],
    threadgroup bfloat  *bA_stage   [[threadgroup(0)]],   // [BT, BT] bf16, 8 KB
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
    // params[8] = BK, params[9] = BV — retained for host/test compatibility,
    // not used by this MMA-rewritten kernel (BV = 32 is hard-coded via the
    // 4 BV col-tile count; BK is irrelevant since the K-loop is per 8-tile).

    // scale is packed as float in the same buffer at u32 index 10
    // (host writes scale.to_bits() at u32[10]).
    const float scale = as_type<float>(params[10]);

    const uint i_v_tile = tgid.x;
    const uint i_t      = tgid.y;
    const uint i_bh     = tgid.z;
    const uint tid      = tid3.x;

    const uint i_b = i_bh / H;
    const uint i_h = i_bh - i_b * H;

    if (i_b >= B || i_h >= H || i_t >= NT) return;

    const uint nv  = V / 32u;
    if (i_v_tile >= nv) return;

    const uint group_ratio = H / Hg;
    const uint kh          = i_h / group_ratio;
    const uint t_start     = i_t * BT;
    const uint v_off       = i_v_tile * 32u;

    // ---- Strides (in elements) ----
    // q, k: [B, T, Hg, K]
    const uint qk_t_stride   = Hg * K;
    const uint qk_seq_stride = T * qk_t_stride;
    // v, o: [B, T, H, V]
    const uint v_t_stride    = H * V;
    const uint v_seq_stride  = T * v_t_stride;
    // h: [B, NT, H, V, K]
    const uint h_v_stride    = K;
    const uint h_h_stride    = V * h_v_stride;
    const uint h_nt_stride   = H * h_h_stride;
    const uint h_seq_stride  = NT * h_nt_stride;
    // g: [B, T, H]
    const uint g_t_stride    = H;
    const uint g_seq_stride  = T * g_t_stride;

    // Per-(b, h, i_t, i_v) bases.
    const uint q_chunk_base = i_b * qk_seq_stride
                            + t_start * qk_t_stride
                            + kh * K;
    const uint k_chunk_base = i_b * qk_seq_stride
                            + t_start * qk_t_stride
                            + kh * K;
    const uint v_chunk_base = i_b * v_seq_stride
                            + t_start * v_t_stride
                            + i_h * V;
    const uint h_chunk_base = i_b * h_seq_stride
                            + i_t * h_nt_stride
                            + i_h * h_h_stride;
    const uint o_chunk_base = i_b * v_seq_stride
                            + t_start * v_t_stride
                            + i_h * V;
    const uint g_base       = i_b * g_seq_stride + i_h;

    // simdgroup-frag lane coordinates (BaseMMAFrag<T, 8, 8>::get_coord).
    const ushort qid = slid / 4;
    const ushort fm  = (qid & 4) + ((slid / 2) % 4);
    const ushort fn  = (qid & 2) * 2 + (slid % 2) * 2;

    // ===================================================================
    // 1. Initialise MMA accumulator frags.
    //    bo_acc: 4 frags per sg (1 BT row-tile × 4 BV col-tiles).
    //    bA_acc: 8 frags per sg (1 BT row-tile × 8 BT col-tiles).
    // ===================================================================
    simdgroup_matrix<float, 8, 8> bo_acc[4];
    simdgroup_matrix<float, 8, 8> bA_acc[8];
    for (uint c = 0; c < 4; ++c) {
        bo_acc[c] = simdgroup_matrix<float, 8, 8>(0);
    }
    for (uint c = 0; c < 8; ++c) {
        bA_acc[c] = simdgroup_matrix<float, 8, 8>(0);
    }

    // ===================================================================
    // 2. K-reduction loop (16 K-tiles of 8). Sections 2A and 2B fused —
    //    they share the same q-frag (sgid's BT row-tile, current K-tile).
    //    Per FLA :93-113.
    //
    //    HARD-CODED at 16 (= K=128 / 8). MAX_K=128 in validate(); see
    //    iter 1.5 lesson on compile-time loop bounds.
    // ===================================================================
    {
        const uint sg_bt_origin = sgid * 8u;     // BT row this sg owns
        const device bfloat *q_row_base =
            q + q_chunk_base + sg_bt_origin * qk_t_stride;
        const device bfloat *h_base = h + h_chunk_base + v_off * h_v_stride;
        const device bfloat *k_row_base = k + k_chunk_base;

        const uint num_k_tiles = 16u;
        for (uint kt = 0; kt < num_k_tiles; ++kt) {
            const uint k_origin = kt * 8u;

            simdgroup_barrier(mem_flags::mem_none);

            // Load 1 q-frag : BT rows [sg_bt_origin, +8), K cols [k_origin, +8).
            simdgroup_matrix<bfloat, 8, 8> q_frag;
            simdgroup_load(q_frag, q_row_base + k_origin,
                           qk_t_stride, 0, false);

            // 2A: load 4 h^T-frags — all 4 BV col-tiles.
            //   h layout is [BV-rows, K] row-major (K-innermost, row stride = K).
            //   h^T frag at (k_origin..+8, bv_origin..+8) reads
            //   h[bv_origin+j][k_origin+i] — base = h_base + bv_origin*K + k_origin
            //   with row stride K and transpose=true.
            simdgroup_matrix<bfloat, 8, 8> h_frag_T[4];
            for (uint j = 0; j < 4; ++j) {
                const uint bv_origin = j * 8u;
                const device bfloat *h_ptr =
                    h_base + bv_origin * h_v_stride + k_origin;
                simdgroup_load(h_frag_T[j], h_ptr, h_v_stride, 0, true);
            }

            // 2B: load 8 k^T-frags — all 8 BT col-tiles.
            //   k layout is [BT, K] row-major (K-innermost, row stride = qk_t_stride).
            //   k^T frag at (k_origin..+8, bt_col..+8) reads
            //   k[bt_col+j][k_origin+i] — base = k_row_base + bt_col*qk_t_stride
            //   + k_origin with row stride qk_t_stride and transpose=true.
            simdgroup_matrix<bfloat, 8, 8> k_frag_T[8];
            for (uint j = 0; j < 8; ++j) {
                const uint bt_col = j * 8u;
                const device bfloat *k_ptr =
                    k_row_base + bt_col * qk_t_stride + k_origin;
                simdgroup_load(k_frag_T[j], k_ptr, qk_t_stride, 0, true);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 2A: 4 MMAs — bo_acc[j] += q_frag @ h_frag_T[j].
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(
                    bo_acc[j], q_frag, h_frag_T[j], bo_acc[j]);
            }
            // 2B: 8 MMAs — bA_acc[j] += q_frag @ k_frag_T[j].
            for (uint j = 0; j < 8; ++j) {
                simdgroup_multiply_accumulate(
                    bA_acc[j], q_frag, k_frag_T[j], bA_acc[j]);
            }
        }
    }

    // ===================================================================
    // 3. Apply gate to bo_acc + gate/mask/bf16-stage bA_acc.
    //    Per-thread element extract from MMA frags.
    //
    //    bo_acc[i, j] *= exp(g[t_start + i])
    //    bA_acc[i, j] *= exp(g[t_start + i] - g[t_start + j])
    //    bA_acc[i, j] = (i >= j) ? bA_acc[i, j] : 0    (FLA :122-125 `>=`)
    //    bA_stage[i, j] = bfloat(bA_acc[i, j])         (FLA :137)
    // ===================================================================
    const uint sg_bt_origin = sgid * 8u;
    const uint bt_local = sg_bt_origin + fm;       // BT row index this lane owns
    const float g_i = g[g_base + (t_start + bt_local) * g_t_stride];
    const float exp_g_i = metal::exp(g_i);

    // 3A: gate bo_acc in registers.
    for (uint j = 0; j < 4; ++j) {
        thread auto &elems = bo_acc[j].thread_elements();
        elems[0] = float(elems[0]) * exp_g_i;
        elems[1] = float(elems[1]) * exp_g_i;
    }

    // 3B: gate + mask + bf16-stage bA_acc.
    //   Each lane owns 2 elements per frag at (fm, fn, fn+1).
    //   Across 8 col-tiles: bt_col = j*8 + fn, j*8 + fn + 1 (16 total cells/lane).
    for (uint j = 0; j < 8; ++j) {
        thread auto &elems = bA_acc[j].thread_elements();
        const uint bt_col0 = j * 8u + fn;
        const uint bt_col1 = bt_col0 + 1u;
        const float g_j0 = g[g_base + (t_start + bt_col0) * g_t_stride];
        const float g_j1 = g[g_base + (t_start + bt_col1) * g_t_stride];
        float v0 = float(elems[0]) * metal::exp(g_i - g_j0);
        float v1 = float(elems[1]) * metal::exp(g_i - g_j1);
        // FLA :124 — INCLUSIVE causal mask (`>=`, NOT strict).
        v0 = (bt_local >= bt_col0) ? v0 : 0.0f;
        v1 = (bt_local >= bt_col1) ? v1 : 0.0f;
        // bf16 round-trip + publish to threadgroup mem.
        // bA_stage layout: [BT, BT] row-major (BT cols innermost, row stride = BT).
        bA_stage[bt_local * BT + bt_col0] = bfloat(v0);
        bA_stage[bt_local * BT + bt_col1] = bfloat(v1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===================================================================
    // 4. Section 2C: bo_acc += bA_stage @ v_new  (8 BT-tiles of 8).
    //    Each sg owns same BT row-tile [sg_bt_origin, +8) × 4 BV col-tiles.
    //    Per BT-tile: 1 bA_stage frag (BT-row sgid, 8 BT cols) + 4 v-frags
    //    (8 BT rows × 8 BV cols each, 4 col-tiles total) → 4 MMAs.
    //
    //    HARD-CODED at 8 (= BT=64 / 8). BT=64 enforced by validate().
    // ===================================================================
    {
        // bA_stage frag origin row = sg_bt_origin (same as q's BT row-tile).
        const threadgroup bfloat *bA_row_base = bA_stage + sg_bt_origin * BT;
        const device bfloat *v_base = v + v_chunk_base + v_off;

        const uint num_bt_tiles = 8u;
        for (uint bj = 0; bj < num_bt_tiles; ++bj) {
            const uint bt_origin = bj * 8u;

            simdgroup_barrier(mem_flags::mem_none);

            // Load 1 bA-frag : BT rows [sg_bt_origin, +8), BT cols
            // [bt_origin, +8). Row stride = BT.
            simdgroup_matrix<bfloat, 8, 8> bA_frag;
            simdgroup_load(bA_frag, bA_row_base + bt_origin, BT, 0, false);

            // Load 4 v-frags — 8 BT rows [bt_origin, +8) × 8 BV cols each.
            //   v_new[B, T, H, V] row-major V-innermost; row stride = v_t_stride.
            simdgroup_matrix<bfloat, 8, 8> v_frag[4];
            const device bfloat *v_row_base =
                v_base + bt_origin * v_t_stride;
            for (uint j = 0; j < 4; ++j) {
                const uint bv_origin = j * 8u;
                simdgroup_load(v_frag[j], v_row_base + bv_origin,
                               v_t_stride, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // 4 MMAs: bo_acc[j] += bA_frag @ v_frag[j].
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(
                    bo_acc[j], bA_frag, v_frag[j], bo_acc[j]);
            }
        }
    }

    // ===================================================================
    // 5. Final scale + bf16 store of bo_acc to o.
    //    Per FLA :137: o = (bo_acc_pre_2C + dot_2C) * scale, applied as
    //    a single multiply on the post-2C accumulator (mathematically
    //    equivalent to the spec's two `* scale` terms — distributive law
    //    over the sum).
    //
    //    Each lane owns 2 elements per frag at (fm, fn, fn+1) across 4
    //    BV col-tiles = 8 cells.
    // ===================================================================
    {
        const uint o_row_base = o_chunk_base + bt_local * v_t_stride + v_off;
        for (uint j = 0; j < 4; ++j) {
            thread auto &elems = bo_acc[j].thread_elements();
            const uint bv0 = j * 8u + fn;
            const uint bv1 = bv0 + 1u;
            const float val0 = float(elems[0]) * scale;
            const float val1 = float(elems[1]) * scale;
            o[o_row_base + bv0] = bfloat(val0);
            o[o_row_base + bv1] = bfloat(val1);
        }
    }
}
