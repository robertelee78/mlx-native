#include <metal_stdlib>
using namespace metal;

// Wave 5b.1 iter 2 — chunk_scaled_dot_kkt Metal kernel.
//
// Spec source:
// - FLA reference: `chunk_scaled_dot_kkt_fwd_kernel` at
//   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-99
//
// No FLA / Triton / CUDA / Metal code is copied. The math here is a
// re-derivation from the FLA spec; the algorithmic structure (load /
// scale / bf16-cast / dot / gate / mask) follows the FLA Triton kernel
// pattern but is open-coded for Metal.
//
// # Algorithm (per (batch b, V-head i_h, chunk i_t))
//
//   kh        = i_h / (H / Hg)                        # GQA-mapped K-head
//   b_beta    = beta[b, t_chunk, i_h]                 # [BT] f32
//   b_g       = g[b, t_chunk, i_h]                    # [BT] f32 (cumsumed)
//   ba_acc    = zeros([BT, BT])                       # f32 in shared mem
//   for i_k in 0..(K // BK):
//       # Cooperative load of bk_stage[BT, BK] in bf16 (post-scale-cast):
//       for cells in (BT*BK) / TG_THREADS:
//           bk_stage[bt, bk] = bf16(b_k[bt, bk].float() * b_beta[bt])  # FLA :86
//       barrier
//       # Per-thread accumulate into ba_acc[BT, BT]:
//       for cells in (BT*BT) / TG_THREADS:
//           (i, j) = unflatten(cell)
//           dot = 0.0f
//           for kk in 0..BK:
//               dot += bk_stage[i, kk].float() * b_k_orig[j, kk].float()
//           ba_acc[i, j] += dot
//       barrier
//   # Apply gate exp(b_g[:, None] - b_g[None, :]):
//   for cells in (BT*BT) / TG_THREADS:
//       (i, j) = unflatten(cell)
//       ba_acc[i, j] *= exp(b_g[i] - b_g[j])
//   # Apply strict-lower mask (row > col only) and store to A:
//   for cells in (BT*BT) / TG_THREADS:
//       (i, j) = unflatten(cell)
//       v = (i > j) ? ba_acc[i, j] : 0.0f
//       A[b, t_chunk[i], i_h, j] = v
//
// # Numerical precision
//
//   Inputs k bf16, beta f32, g f32. Intermediate ba_acc in f32. The bf16
//   cast on b_kb (post-scale, pre-dot) follows FLA wy_fast.py:86; the dot
//   accumulator stays in f32 (bf16 × bf16 → f32 via PyTorch promote-to-f32
//   when reading the bf16 bytes).
//
// # bf16 round-trip placement
//
//   FLA line 85: b_kb = b_k * b_beta[:, None]    # bf16 * f32 → f32
//   FLA line 86: b_A += dot(b_kb.to(b_k.dtype), trans(b_k))
//                              ^^^^^^^^^^^^^^^^^
//                              cast to bf16 BEFORE dot, AFTER scale
//
//   We mirror that ordering exactly:
//     1. local_kb_f32 = b_k.float() * b_beta[bt]      (FLA :85, scale in f32)
//     2. bk_stage[bt, bk] = bfloat(local_kb_f32)      (FLA :86, post-scale cast)
//     3. ba_acc[i, j] += bk_stage[i, kk].float() * b_k[j, kk].float()  (f32 dot)
//
// # Threadgroup memory layout (24 KB total at BT=BK=64)
//
//   bk_stage  [[threadgroup(0)]]  : BT * BK * 2 bytes = 8  KB  (bf16)
//   ba_acc    [[threadgroup(1)]]  : BT * BT * 4 bytes = 16 KB  (f32)
//
//   Total: 24 KB < 32 KB M5 Max cap.
//
// # Memory layouts (innermost-first)
//
//   k:    [B, T, Hg, K]   bf16  — K innermost
//   beta: [B, T, H]       f32   — H innermost
//   g:    [B, T, H]       f32   — H innermost
//   A:    [B, T, H, BT]   f32   — BT innermost (row-major within each chunk's
//                                  [BT, BT] block, which is stored at rows
//                                  [bos+i_t*BT : bos+(i_t+1)*BT] of A)
//
// # Threading
//
//   Grid: (NT, H, B)
//   Threadgroup: TG_THREADS = 256 (8 simdgroups × 32 lanes), flat 1D.
//
// # Buffer bindings
//
//   buffer(0): k        bf16
//   buffer(1): beta     f32
//   buffer(2): g        f32
//   buffer(3): A        f32  (output)
//   buffer(4): params   uint[8] = [B, T, Hg, H, K, BT, NT, BK]

constant uint TG_THREADS = 256u;
constant uint BT_FIXED   = 64u;   // chunk size (iter-2 fixed)
constant uint BK_FIXED   = 64u;   // K-tile width (iter-2 fixed)

kernel void gated_delta_net_kkt_bf16(
    device const bfloat *k          [[buffer(0)]],
    device const float  *beta       [[buffer(1)]],
    device const float  *g          [[buffer(2)]],
    device float        *A          [[buffer(3)]],
    device const uint   *params     [[buffer(4)]],
    threadgroup bfloat  *bk_stage   [[threadgroup(0)]],   // [BT, BK] bf16
    threadgroup float   *ba_acc     [[threadgroup(1)]],   // [BT, BT] f32
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint B   = params[0];
    const uint T   = params[1];
    const uint Hg  = params[2];
    const uint H   = params[3];
    const uint K   = params[4];
    const uint BT  = params[5];   // = 64 in iter-2
    const uint NT  = params[6];
    const uint BK  = params[7];   // = 64 in iter-2

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
    const uint g_t_stride   = H;
    const uint g_seq_stride = T * g_t_stride;
    // A: [B, T, H, BT] — stride for (b, t, h, bt) is (T*H*BT, H*BT, BT, 1).
    const uint a_t_stride   = H * BT;
    const uint a_seq_stride = T * a_t_stride;

    // ===================================================================
    // 0. Initialize ba_acc to zero.
    // ===================================================================
    const uint ba_cells = BT * BT;   // = 4096 at BT=64
    for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
        ba_acc[cell] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load b_beta into a local register slab (one f32 per row).
    // Each thread computes its own b_beta[bt] on demand to avoid a
    // dedicated shared buffer; with BT=64 the load is cheap and cached.
    // (We could place beta in shared mem, but the redundant per-thread
    // re-reads are coalesced by Apple's L1 — measurement first; optimize
    // later if profiling shows pressure.)

    const uint beta_base = i_b * g_seq_stride + i_h;   // beta same layout as g
    const uint g_base    = i_b * g_seq_stride + i_h;

    // ===================================================================
    // 1. K-tile loop: for each i_k in 0..K/BK, load bk_stage and accumulate.
    // ===================================================================
    const uint nbk = K / BK;          // = 2 at K=128, BK=64
    const uint bk_cells = BT * BK;    // = 4096

    for (uint i_k = 0; i_k < nbk; ++i_k) {
        const uint k_off = i_k * BK;

        // -----------------------------------------------------------
        // 1a. Cooperative load of bk_stage[bt, bk] = bf16(k[bt, bk] * beta[bt]).
        //     Each thread owns bk_cells/TG_THREADS = 16 cells.
        // -----------------------------------------------------------
        const uint k_chunk_base = i_b * k_seq_stride + t_start * k_t_stride + kh * K;
        for (uint cell = tid; cell < bk_cells; cell += TG_THREADS) {
            const uint bt_idx = cell / BK;
            const uint bk_idx = cell - bt_idx * BK;
            const float k_val   = float(k[k_chunk_base + bt_idx * k_t_stride + k_off + bk_idx]);
            const float beta_v  = beta[beta_base + (t_start + bt_idx) * g_t_stride];
            // FLA :85 scale in f32; FLA :86 cast to bf16.
            bk_stage[bt_idx * BK + bk_idx] = bfloat(k_val * beta_v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -----------------------------------------------------------
        // 1b. Per-thread accumulate into ba_acc[BT, BT].
        //     ba_acc[i, j] += sum_kk(bk_stage[i, kk] * k[j, kk])
        //     Each thread owns ba_cells/TG_THREADS = 16 cells.
        // -----------------------------------------------------------
        for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
            const uint i = cell / BT;
            const uint j = cell - i * BT;
            // bk_stage row i (bf16, post-scale), original k row j (bf16).
            float dot_val = 0.0f;
            for (uint kk = 0; kk < BK; ++kk) {
                const float a = float(bk_stage[i * BK + kk]);
                const float b = float(k[k_chunk_base + j * k_t_stride + k_off + kk]);
                dot_val += a * b;
            }
            ba_acc[cell] += dot_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===================================================================
    // 2. Apply gate: ba_acc[i, j] *= exp(g[i] - g[j]).
    //    Then mask: keep only strict-lower (i > j); zero the rest.
    //    Then store to A.
    // ===================================================================
    for (uint cell = tid; cell < ba_cells; cell += TG_THREADS) {
        const uint i = cell / BT;
        const uint j = cell - i * BT;

        const float g_i = g[g_base + (t_start + i) * g_t_stride];
        const float g_j = g[g_base + (t_start + j) * g_t_stride];

        float v = ba_acc[cell] * metal::exp(g_i - g_j);
        // FLA :94-95: strict-lower mask (i > j ONLY).
        v = (i > j) ? v : 0.0f;

        // Store to A[b, t_start+i, h, j].
        const uint a_off = i_b * a_seq_stride
                         + (t_start + i) * a_t_stride
                         + i_h * BT
                         + j;
        A[a_off] = v;
    }
}
