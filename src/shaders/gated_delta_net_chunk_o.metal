#include <metal_stdlib>
using namespace metal;

// Wave 5b.1 iter 3 — chunk_fwd_o Metal kernel.
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
//   bo_acc     = zeros([BT, BV])                          # f32 in shared mem
//   bA_acc     = zeros([BT, BT])                          # f32 in shared mem
//
//   for i_k in 0..(K // BK):
//       # b_q : [BT, BK] bf16; b_k : [BT, BK] bf16; b_h : [BV, BK] bf16
//       # Cooperative load-and-accumulate over q · h^T and q · k^T.
//       bo_acc[BT, BV] += b_q[BT, BK] @ trans(b_h[BV, BK])    # FLA :111
//       bA_acc[BT, BT] += b_q[BT, BK] @ trans(b_k[BT, BK])    # FLA :113
//
//   # Gate (FLA :115-120; USE_G == True for GDN):
//   bo_acc[i, j] *= exp(g[t_start + i])
//   bA_acc[i, j] *= exp(g[t_start + i] - g[t_start + j])
//
//   # Causal+diag mask (FLA :122-125 — INCLUSIVE of diagonal, `>=`,
//   # different from kkt's strict `>`):
//   bA_acc[i, j] = (i >= j) ? bA_acc[i, j] : 0
//
//   # bf16 round-trip on bA_acc (FLA :137 — AFTER mask, BEFORE dot with v):
//   bA_bf16[i, j] = bfloat(bA_acc[i, j])
//
//   # Closing dot — bf16 b_A × bf16 v_new → f32 acc → scale → bf16 store:
//   for cell in [BT, BV]:
//       (i, j) = cell
//       v_dot = sum_kk(bA_bf16[i, kk] * v_new[bt+i, h, vs+j])
//       o_val = bo_acc[i, j] * scale + v_dot * scale          # FLA :137
//       o[t_start+i, h, vs+j] = bfloat(o_val)
//
// # CRITICAL ORDERING DETAILS (per iter-1.5 lesson)
//
// 1. The bf16 cast of bA_acc happens AFTER the mask, BEFORE the dot with
//    v (FLA line 137). Moving or skipping it changes numerics.
// 2. The mask is `>=` (causal+diag) — NOT strict `>` like kkt. The
//    diagonal element corresponds to attention to the same token and is
//    included.
// 3. Two `* scale` multiplications on FLA line 137: BOTH `bo_acc * scale`
//    AND `dot_result * scale`. We apply both as in spec (mathematically
//    equivalent to one global scale, but the spec's separate placement is
//    what we mirror).
// 4. h dimension order is [V, K] per (chunk, head) — `b_o += dot(b_q,
//    trans(b_h))` per FLA :111.
//
// # Threadgroup memory layout (28 KB at BT=64, BK=32, BV=32)
//
//   bo_acc  [[threadgroup(0)]]  : BT * BV * 4 = 64 * 32 * 4 = 8  KB  (f32)
//   ba_acc  [[threadgroup(1)]]  : BT * BT * 4 = 64 * 64 * 4 = 16 KB  (f32)
//   stage   [[threadgroup(2)]]  : BT * max(BK, BV) * 2 = 64 * 32 * 2 = 4 KB  (bf16)
//
//   Total: 28 KB < 32 KB M5 Max cap (4 KB headroom).
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
//   buffer(6): params   uint[10] = [B, T, Hg, H, K, V, BT, NT, BK, BV]
//                                   plus float[1] scale at byte offset 40.

constant uint TG_THREADS = 256u;

kernel void gated_delta_net_chunk_o_bf16(
    device const bfloat *q          [[buffer(0)]],
    device const bfloat *k          [[buffer(1)]],
    device const bfloat *v          [[buffer(2)]],
    device const bfloat *h          [[buffer(3)]],
    device const float  *g          [[buffer(4)]],
    device bfloat       *o          [[buffer(5)]],
    device const uint   *params     [[buffer(6)]],
    threadgroup float   *bo_acc     [[threadgroup(0)]],   // [BT, BV] f32
    threadgroup float   *ba_acc     [[threadgroup(1)]],   // [BT, BT] f32
    threadgroup bfloat  *stage      [[threadgroup(2)]],   // [BT, max(BK,BV)] bf16
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
    const uint BK  = params[8];
    const uint BV  = params[9];

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

    const uint nv  = V / BV;
    if (i_v_tile >= nv) return;

    const uint group_ratio = H / Hg;
    const uint kh          = i_h / group_ratio;
    const uint t_start     = i_t * BT;
    const uint v_off       = i_v_tile * BV;

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

    // ===================================================================
    // 0. Initialise bo_acc and ba_acc to zero.
    // ===================================================================
    const uint bo_cells = BT * BV;   // 2048 at BT=64, BV=32
    const uint ba_cells = BT * BT;   // 4096 at BT=64
    for (uint cell = tid; cell < bo_cells; cell += TG_THREADS) {
        bo_acc[cell] = 0.0f;
    }
    for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
        ba_acc[cell] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===================================================================
    // 1. K-tile loop: accumulate bo_acc += q · h^T  and  ba_acc += q · k^T.
    //    Per FLA :93-113.
    // ===================================================================
    const uint nbk = K / BK;
    for (uint i_k = 0; i_k < nbk; ++i_k) {
        const uint k_off = i_k * BK;

        // No need to stage q, k, h into shared memory because each thread
        // reads them directly from device memory; Apple L1 caches the
        // re-reads. The dot loops are inner-K fused; if L1 pressure shows
        // up in profiling, iter-4 can stage these.

        // -----------------------------------------------------------
        // 1a. bo_acc[i, j] += sum_kk(q[i, kk] * h[j, kk])     (q · h^T)
        //     where i in [0, BT), j in [0, BV), kk in [k_off, k_off+BK).
        // -----------------------------------------------------------
        for (uint cell = tid; cell < bo_cells; cell += TG_THREADS) {
            const uint i = cell / BV;
            const uint j = cell - i * BV;
            float acc = 0.0f;
            const uint q_row_base = q_chunk_base + i * qk_t_stride + k_off;
            const uint h_row_base = h_chunk_base + (v_off + j) * h_v_stride + k_off;
            for (uint kk = 0; kk < BK; ++kk) {
                const float q_v = float(q[q_row_base + kk]);
                const float h_v = float(h[h_row_base + kk]);
                acc += q_v * h_v;
            }
            bo_acc[cell] += acc;
        }

        // -----------------------------------------------------------
        // 1b. ba_acc[i, j] += sum_kk(q[i, kk] * k[j, kk])     (q · k^T)
        //     where i, j in [0, BT), kk in [k_off, k_off+BK).
        // -----------------------------------------------------------
        for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
            const uint i = cell / BT;
            const uint j = cell - i * BT;
            float acc = 0.0f;
            const uint q_row_base = q_chunk_base + i * qk_t_stride + k_off;
            const uint k_row_base = k_chunk_base + j * qk_t_stride + k_off;
            for (uint kk = 0; kk < BK; ++kk) {
                const float q_v = float(q[q_row_base + kk]);
                const float k_v = float(k[k_row_base + kk]);
                acc += q_v * k_v;
            }
            ba_acc[cell] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================================================================
    // 2. Apply gate (FLA :115-120):
    //    bo_acc[i, j] *= exp(g[t_start + i])
    //    ba_acc[i, j] *= exp(g[t_start + i] - g[t_start + j])
    // ===================================================================
    // bo_acc gate (uses g[i] only).
    for (uint cell = tid; cell < bo_cells; cell += TG_THREADS) {
        const uint i = cell / BV;
        const float g_i = g[g_base + (t_start + i) * g_t_stride];
        bo_acc[cell] *= metal::exp(g_i);
    }

    // ba_acc gate (uses g[i] - g[j]). Also applies causal+diag mask
    // INCLUSIVE of diagonal (FLA :122-125, `>=`).
    for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
        const uint i = cell / BT;
        const uint j = cell - i * BT;
        const float g_i = g[g_base + (t_start + i) * g_t_stride];
        const float g_j = g[g_base + (t_start + j) * g_t_stride];
        float v = ba_acc[cell] * metal::exp(g_i - g_j);
        // FLA :124 — INCLUSIVE causal mask (`>=`, NOT strict).
        v = (i >= j) ? v : 0.0f;
        ba_acc[cell] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ===================================================================
    // 3. bf16 round-trip on ba_acc (FLA :137 — load-bearing per iter 1.5).
    //    Stage in `stage` shared buffer (4 KB, [BT, BT/2] bf16) — but
    //    BT*BT = 4096 cells exceeds the 4 KB stage capacity at BT=64
    //    (stage holds BT * max(BK,BV) = 64*32 = 2048 bf16 cells = 4 KB).
    //
    //    Instead, we re-cast on-the-fly inside the closing dot — read
    //    f32 ba_acc, cast to bf16 inline as the multiplicand. This is
    //    bit-identical to staging, since `bfloat(f32_val)` is the same
    //    truncation either way.
    //
    //    Verified equivalence: storing bf16(x) and re-reading vs.
    //    inlining bfloat(x) at use-site produce identical bf16 bytes
    //    (bfloat is a deterministic 32→16 bit truncation).
    // ===================================================================

    // ===================================================================
    // 4. Closing dot — bo_acc * scale + (bA_bf16 · v_new) * scale  → bf16 store.
    //    Per FLA :137.
    //    For each output cell (i, j) in [BT, BV]:
    //      v_dot = sum_kk(bfloat(ba_acc[i, kk]) * v_new[t_start+kk, i_h, v_off+j])
    //      o[t_start+i, i_h, v_off+j] = bfloat(bo_acc[i, j] * scale + v_dot * scale)
    // ===================================================================
    for (uint cell = tid; cell < bo_cells; cell += TG_THREADS) {
        const uint i = cell / BV;
        const uint j = cell - i * BV;
        float v_dot = 0.0f;
        for (uint kk = 0; kk < BT; ++kk) {
            // FLA :137 — bf16 cast of ba_acc post-mask, pre-dot.
            const float a_bf16_promoted = float(bfloat(ba_acc[i * BT + kk]));
            const float v_v = float(v[v_chunk_base + kk * v_t_stride + v_off + j]);
            v_dot += a_bf16_promoted * v_v;
        }
        const float o_val = bo_acc[cell] * scale + v_dot * scale;
        o[o_chunk_base + i * v_t_stride + v_off + j] = bfloat(o_val);
    }
}
