#include <metal_stdlib>
using namespace metal;

// Wave 5b.1 iter 2 — recompute_w_u_fwd Metal kernel.
//
// Spec source:
// - FLA reference: `recompute_w_u_fwd_kernel` at
//   /opt/vllm/vllm/model_executor/layers/fla/ops/wy_fast.py:29-117
//
// No FLA / Triton / CUDA / Metal code is copied. The math here is a
// re-derivation from the FLA spec; the algorithmic structure (load A,
// stage scaled b_vb / b_kb in bf16, dot, bf16 store) follows the FLA
// Triton kernel pattern but is open-coded for Metal.
//
// # Algorithm (per (batch b, V-head i_h, chunk i_t))
//
//   kh        = i_h / (H / Hg)
//   b_beta    = beta[b, t_chunk, i_h]                   # [BT] f32
//   b_A       = A[b, t_chunk, i_h, :]                   # [BT, BT] f32 (post-solve)
//   b_g       = exp(g[b, t_chunk, i_h])                 # [BT] f32  (FLA :72)
//
//   ba_tile  ← b_A                                       # f32 in shared mem
//
//   # u-loop (FLA wy_fast.py:74-94):
//   for i_v in 0..(V // BV):
//       # Cooperative load of stage[BT, BV] = bf16(v_tile.float() * beta[:, None]):
//       stage[bt, bv] = bfloat(v_tile[bt, bv].float() * b_beta[bt])    # FLA :92
//       barrier
//       # Per-thread compute u tile cell:
//       u[bt, bv] = sum_kk(ba_tile[bt, kk] * stage[kk, bv].float())     # FLA :93
//       store bf16(u) to u[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV]
//
//   # w-loop (FLA wy_fast.py:96-116):
//   for i_k in 0..(K // BK):
//       # Cooperative load of stage[BT, BK] = bf16(k_tile.float() * beta * exp_g):
//       stage[bt, bk] = bfloat(k_tile[bt, bk].float() * b_beta[bt] * b_g[bt])  # FLA :114
//       barrier
//       w[bt, bk] = sum_kk(ba_tile[bt, kk] * stage[kk, bk].float())     # FLA :115
//       store bf16(w) to w[b, t_chunk, i_h, i_k*BK:(i_k+1)*BK]
//
// # Numerical precision
//
//   Inputs k bf16, v bf16, beta f32, g f32, A f32. Intermediate dot accums in f32.
//   The bf16 cast on the scaled operand (b_vb at FLA :92, b_kb at FLA :114)
//   happens AFTER the scale, BEFORE the dot. The dot accumulator stays in f32.
//
// # bf16 round-trip placement (mirror FLA spec)
//
//   FLA :92 (u-loop):  b_vb = (b_v.float() * b_beta[:, None]).to(bf16)
//   FLA :93:           b_u  = dot(b_A, b_vb)            # f32 × bf16 → f32 acc
//   FLA :114 (w-loop): b_kb = (b_k.float() * b_beta[:, None] * b_g[:, None]).to(bf16)
//   FLA :115:          b_w  = dot(b_A, b_kb)            # f32 × bf16 → f32 acc
//
// # Threadgroup memory layout (24 KB at BT=64, BK=BV=64)
//
//   ba_tile  [[threadgroup(0)]]  : BT * BT * 4 = 16 KB  (f32)
//   stage    [[threadgroup(1)]]  : BT * max(BV, BK) * 2 = 8 KB  (bf16)
//
// # Memory layouts (innermost-first)
//
//   k:    [B, T, Hg, K]    bf16  — K innermost
//   v:    [B, T, H,  V]    bf16  — V innermost
//   beta: [B, T, H]        f32
//   g:    [B, T, H]        f32   — chunk-cumsumed
//   A:    [B, T, H, BT]    f32   — BT innermost (post-solve_tril)
//   w:    [B, T, H, K]     bf16  — K innermost (output)
//   u:    [B, T, H, V]     bf16  — V innermost (output)
//
// # Threading
//
//   Grid: (NT, H, B)
//   Threadgroup: TG_THREADS = 256
//
// # Buffer bindings
//
//   buffer(0): k        bf16
//   buffer(1): v        bf16
//   buffer(2): beta     f32
//   buffer(3): g        f32
//   buffer(4): A        f32
//   buffer(5): w        bf16  (output)
//   buffer(6): u        bf16  (output)
//   buffer(7): params   uint[8] = [B, T, Hg, H, K, V, BT, NT]

constant uint TG_THREADS = 256u;

kernel void gated_delta_net_recompute_wu_bf16(
    device const bfloat *k          [[buffer(0)]],
    device const bfloat *v          [[buffer(1)]],
    device const float  *beta       [[buffer(2)]],
    device const float  *g          [[buffer(3)]],
    device const float  *A          [[buffer(4)]],
    device bfloat       *w          [[buffer(5)]],
    device bfloat       *u          [[buffer(6)]],
    device const uint   *params     [[buffer(7)]],
    threadgroup float   *ba_tile    [[threadgroup(0)]],   // [BT, BT] f32
    threadgroup bfloat  *stage      [[threadgroup(1)]],   // [BT, max(BV,BK)] bf16
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

    // BV / BK fixed at 64 in iter 2 (matches host validation).
    const uint BV = 64u;
    const uint BK = 64u;

    const uint i_t = tgid.x;
    const uint i_h = tgid.y;
    const uint i_b = tgid.z;
    const uint tid = tid3.x;

    if (i_b >= B || i_h >= H || i_t >= NT) return;

    const uint group_ratio = H / Hg;
    const uint kh          = i_h / group_ratio;
    const uint t_start     = i_t * BT;

    // Strides (in elements).
    const uint k_t_stride   = Hg * K;
    const uint k_seq_stride = T * k_t_stride;
    const uint v_t_stride   = H * V;
    const uint v_seq_stride = T * v_t_stride;
    const uint g_t_stride   = H;
    const uint g_seq_stride = T * g_t_stride;
    const uint a_t_stride   = H * BT;
    const uint a_seq_stride = T * a_t_stride;
    const uint w_t_stride   = H * K;
    const uint w_seq_stride = T * w_t_stride;
    const uint u_t_stride   = H * V;
    const uint u_seq_stride = T * u_t_stride;

    // ===================================================================
    // 0. Load b_A → ba_tile [BT, BT] f32.
    // ===================================================================
    const uint a_chunk_base = i_b * a_seq_stride
                            + t_start * a_t_stride
                            + i_h * BT;
    const uint ba_cells = BT * BT;   // = 4096
    for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
        const uint i = cell / BT;
        const uint j = cell - i * BT;
        // A[b, t_start+i, i_h, j]
        ba_tile[cell] = A[a_chunk_base + i * a_t_stride + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint beta_base = i_b * g_seq_stride + i_h;   // beta same layout as g
    const uint g_base    = i_b * g_seq_stride + i_h;

    // ===================================================================
    // 1. u-loop: for each i_v, stage b_vb in bf16, then compute u tile.
    // ===================================================================
    {
        const uint v_chunk_base = i_b * v_seq_stride
                                + t_start * v_t_stride
                                + i_h * V;
        const uint u_chunk_base = i_b * u_seq_stride
                                + t_start * u_t_stride
                                + i_h * V;
        const uint nbv = V / BV;

        for (uint i_v = 0; i_v < nbv; ++i_v) {
            const uint v_off = i_v * BV;

            // 1a. Load stage[BT, BV] = bf16(v_tile.float() * beta[:, None]).
            const uint stage_cells = BT * BV;   // = 4096
            for (uint cell = tid; cell < stage_cells; cell += TG_THREADS) {
                const uint bt_idx = cell / BV;
                const uint bv_idx = cell - bt_idx * BV;
                const float v_val = float(v[v_chunk_base + bt_idx * v_t_stride + v_off + bv_idx]);
                const float beta_v = beta[beta_base + (t_start + bt_idx) * g_t_stride];
                // FLA :92: scale in f32, cast to bf16.
                stage[bt_idx * BV + bv_idx] = bfloat(v_val * beta_v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 1b. Compute u tile: u[i, j] = sum_kk(ba_tile[i, kk] * stage[kk, j].float()).
            //     Each thread owns BT*BV/TG_THREADS = 16 cells.
            for (uint cell = tid; cell < stage_cells; cell += TG_THREADS) {
                const uint i = cell / BV;
                const uint j = cell - i * BV;
                float acc = 0.0f;
                for (uint kk = 0; kk < BT; ++kk) {
                    acc += ba_tile[i * BT + kk] * float(stage[kk * BV + j]);
                }
                u[u_chunk_base + i * u_t_stride + v_off + j] = bfloat(acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ===================================================================
    // 2. w-loop: for each i_k, stage b_kb in bf16, then compute w tile.
    // ===================================================================
    {
        const uint k_chunk_base = i_b * k_seq_stride
                                + t_start * k_t_stride
                                + kh * K;
        const uint w_chunk_base = i_b * w_seq_stride
                                + t_start * w_t_stride
                                + i_h * K;
        const uint nbk = K / BK;

        for (uint i_k = 0; i_k < nbk; ++i_k) {
            const uint k_off = i_k * BK;

            // 2a. Load stage[BT, BK] = bf16(k_tile.float() * beta * exp(g)).
            const uint stage_cells = BT * BK;   // = 4096
            for (uint cell = tid; cell < stage_cells; cell += TG_THREADS) {
                const uint bt_idx = cell / BK;
                const uint bk_idx = cell - bt_idx * BK;
                const float k_val   = float(k[k_chunk_base + bt_idx * k_t_stride + k_off + bk_idx]);
                const float beta_v  = beta[beta_base + (t_start + bt_idx) * g_t_stride];
                const float g_v     = g[g_base + (t_start + bt_idx) * g_t_stride];
                const float exp_g_v = metal::exp(g_v);
                // FLA :114: scale in f32 (b_k * beta * exp(g)), cast to bf16.
                stage[bt_idx * BK + bk_idx] = bfloat(k_val * beta_v * exp_g_v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 2b. Compute w tile: w[i, j] = sum_kk(ba_tile[i, kk] * stage[kk, j].float()).
            for (uint cell = tid; cell < stage_cells; cell += TG_THREADS) {
                const uint i = cell / BK;
                const uint j = cell - i * BK;
                float acc = 0.0f;
                for (uint kk = 0; kk < BT; ++kk) {
                    acc += ba_tile[i * BT + kk] * float(stage[kk * BK + j]);
                }
                w[w_chunk_base + i * w_t_stride + k_off + j] = bfloat(acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
