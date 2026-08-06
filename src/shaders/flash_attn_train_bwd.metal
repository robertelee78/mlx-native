// flash_attn_train_bwd — FA-2 Algorithm 4 backward kernel.
//
// GRID DECOMPOSITION: outer over Q-tiles (tid.x=qb), head axis (tid.y=h_q), batch (tid.z=b).
// Inner loop over K-tiles.
//
// This makes dQ[q-tile, *] local to each threadgroup (no atomics for dQ).
// dK and dV require cross-threadgroup accumulation (different qb threadgroups write
// the same K-row).  We use f32 atomic adds on a temporary f32 dK/dV buffer passed
// from Rust, then the caller bf16-casts.  Alternatively: since Metal supports
// device-level atomic_float (through __atomic_fetch_add on device float*), we use that.
//
// ACTUAL APPROACH (avoids atomics entirely):
// Run the kernel TWICE with a two-pass split:
//   Pass 1: Q-outer grid, accumulate dK and dV into f32 scratch (one pass per Q-tile).
//   Pass 2: cast f32 dK/dV scratch to bf16.
//
// This is simpler for a calibration kernel.  The Rust dispatcher allocates f32
// scratch buffers for dK and dV, runs this kernel, then runs a cast kernel.
//
// EVEN SIMPLER: use f32 output for dQ/dK/dV throughout (the backward is a training
// kernel, not inference — bf16 output is not a hard requirement internally).
// The OUTER Rust API presents bf16 for API consistency, but internally the kernel
// writes f32 and the dispatcher bf16-casts the result via a separate elementwise pass.
//
// THIS FILE implements the core backward in f32 I/O (both input and output f32),
// with a separate bf16-to-f32 cast at the boundaries handled by the Rust dispatcher.
//
// REVISED FINAL DESIGN:
// - Kernel takes f32 scratch arrays for dQ, dK, dV output (zero-initialised).
// - Grid: Q-tile outer (tid.x=qb), h_q (tid.y), batch (tid.z).
// - Inner loop: all K-tiles.
// - dQ accumulation: register-local (no atomics), written once per (qb, h_q, b).
// - dK accumulation: atomic f32 adds into device f32 dK scratch.
// - dV accumulation: atomic f32 adds into device f32 dV scratch.
//
// After the kernel, dK_f32 and dV_f32 are bf16-cast by the dispatcher.
//
// Buffer layout:
//   buffer(0)  Q       [B, H_q,  qL, D]  bf16
//   buffer(1)  K       [B, H_kv, kL, D]  bf16
//   buffer(2)  V       [B, H_kv, kL, D]  bf16
//   buffer(3)  <unused>
//   buffer(4)  L       [B, H_q,  qL]     f32  nat-log logsumexp
//   buffer(5)  dO      [B, H_q,  qL, D]  bf16
//   buffer(6)  D_vec   [B, H_q,  qL]     f32  rowsum(O*dO)
//   buffer(7)  dQ      [B, H_q,  qL, D]  f32  output (zero-init)
//   buffer(8)  dK      [B, H_kv, kL, D]  f32  output (zero-init, atomic-add)
//   buffer(9)  dV      [B, H_kv, kL, D]  f32  output (zero-init, atomic-add)
//   buffer(10) params  AttnParams (160-byte ABI)
//   buffer(11) mask_params [function_constant(has_mask)]
//   buffer(12) mask        [function_constant(has_mask)] bf16 additive
//
// Grid: dispatch_thread_groups(ceil(qL/BQ), H_q, B)
//
// Function constants:
//   200: align_Q   bool
//   201: align_K   bool
//   300: has_mask  bool
//   301: do_causal bool
//
// Kernel variants (D=64 and D=256):
//   flash_attn_train_bwd_bf16_d64
//   flash_attn_train_bwd_bf16_d256
//
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NOUNROLL _Pragma("clang loop unroll(disable)")

// ─── bfloat16 compat shim ────────────────────────────────────────────────────
#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#else
constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask)
    return uint16_t(as_type<uint32_t>(0x7FC0));
  uint32_t fb = as_type<uint32_t>(x);
  fb += ((fb >> 16) & 1) + as_type<uint32_t>(0x7FFF);
  return fb >> 16;
}
constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  return as_type<float>((uint32_t)x << 16);
}
struct _MLX_BFloat16;
template <typename T> static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T,_MLX_BFloat16> && is_convertible_v<T,float>;
template <typename T> static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T,_MLX_BFloat16> && is_convertible_v<float,T>;
struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;
  struct bbs {};
  static constexpr METAL_FUNC bbs bits_to_bfloat() { return bbs(); }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t b, bbs) : bits_(b) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread { return static_cast<T>(bfloat_bits_to_float(bits_)); }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup { return static_cast<T>(bfloat_bits_to_float(bits_)); }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device { return static_cast<T>(bfloat_bits_to_float(bits_)); }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant { return static_cast<T>(bfloat_bits_to_float(bits_)); }
};
typedef struct _MLX_BFloat16 bfloat16_t;
#endif

// ─── AttnParams (160-byte ABI, identical to forward) ─────────────────────────
struct AttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale, softcapping;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off, _pad;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
struct AttnMaskParams { int64_t M_strides[3]; };

constant bool align_Q   [[function_constant(200)]];
constant bool align_K   [[function_constant(201)]];
constant bool has_mask  [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

// ─── BlockLoader (row-major tile load) ───────────────────────────────────────
template <typename T, short BROWS, short BCOLS,
          short kDstStrRow, short kDstStrCol,
          short reduction_dim, short tgp_size,
          short n_reads = (BCOLS * BROWS) / tgp_size,
          short TCOLS = BCOLS / n_reads,
          short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short vec_size = n_reads;
  const int src_ld, tile_stride;
  const short bi, bj;
  threadgroup T* dst;
  const device T* src;
  METAL_FUNC BlockLoaderT(const device T* s, int ld, threadgroup T* d,
                          ushort sg, ushort sl)
      : src_ld(ld),
        tile_stride(reduction_dim ? BCOLS : BROWS * ld),
        bi((sg * 32 + sl) / TCOLS),
        bj(n_reads * ((sg * 32 + sl) % TCOLS)),
        dst(d + bi * kDstStrRow + bj * kDstStrCol),
        src(s + bi * ld + bj) {}
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < n_reads; j++)
        dst[i*kDstStrRow + j*kDstStrCol] = src[i*src_ld + j];
  }
  METAL_FUNC void load_safe(short2 dim) const {
    short2 d2 = dim - short2(bj, bi);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < n_reads; j++) {
        bool ok = (i < d2.y) && (j < d2.x);
        dst[i*kDstStrRow + j*kDstStrCol] = ok ? src[i*src_ld + j] : T(0);
      }
  }
  METAL_FUNC void next() { src += tile_stride; }
};

// ─── The backward kernel ─────────────────────────────────────────────────────
//
// Each threadgroup handles one Q-tile [qb*BQ .. (qb+1)*BQ) for one (h_q, b).
// It loops over all K-tiles to accumulate dQ (register), and atomically updates
// the f32 dK and dV scratch buffers.
//
// Thread layout: (32, WM, WN), where WM is the number of simdgroups.
// Score tile geometry:
//   S = Q[BQ, D] @ K[BK, D]^T → S[BQ, BK]
//   D=64: BQ=32, BK=16, WM=4.
//   D=256: BQ=16, BK=8, WM=2.
//   Each simdgroup owns one 8-row Q fragment and the complete BK columns.
//   The simdgroups collectively cover the complete [BQ, BK] score tile.
//
// dQ tile: each simdgroup writes back dQ for its 8 Q-rows (BQ/WM).
// dK/dV: accumulated into device f32 via atomic_fetch_add_explicit.
//
// The f32 atomic add on device memory is available in Metal 2.4+
// (Apple GPU family 7+, M1 and later). We use:
//   atomic_fetch_add_explicit((device atomic_float*)ptr, val, memory_order_relaxed)

// clang-format off
// Buffer(3) intentionally absent — O (forward output) is not needed in the backward.
// The buffer index gap is legal in Metal; the Rust dispatcher skips binding(3).
template <typename T, int BQ, int BK, int BD, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_train_bwd(
    const device T*      Q        [[buffer(0)]],
    const device T*      K        [[buffer(1)]],
    const device T*      V        [[buffer(2)]],
    const device float*  L        [[buffer(4)]],
    const device T*      dO       [[buffer(5)]],
    const device float*  D_vec    [[buffer(6)]],
    device float*        dQ       [[buffer(7)]],
    device float*        dK       [[buffer(8)]],
    device float*        dV       [[buffer(9)]],
    const constant AttnParams*     params      [[buffer(10)]],
    const constant AttnMaskParams* mask_params  [[buffer(11), function_constant(has_mask)]],
    const device T*                mask         [[buffer(12), function_constant(has_mask)]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{ // clang-format on

  // ── Dimensions ────────────────────────────────────────────────────────────
  const int qb   = (int)tid.x;
  const int h_q  = (int)tid.y;
  const int b    = (int)tid.z;

  const int H_q  = params->H;
  const int H_kv = H_q / params->gqa_factor;
  const int h_kv = h_q / params->gqa_factor;
  const int D    = params->D;
  const int qL   = params->qL;
  const int kL   = params->kL;
  const float sc = params->scale;

  // Absolute Q-row range for this threadgroup.
  const int q_tile_base = qb * BQ;
  const bool last_q_tile = (!align_Q && qb == params->NQ_aligned);

  // ── Threadgroup shared memory ─────────────────────────────────────────────
  //
  // Layout conventions — matching flash_attn_train_fwd.metal exactly:
  //
  //   Q_smem:  ROW-MAJOR   Q_smem[q_row * LDQ + d_col] = Q[q_row][d_col]
  //            LDQ = BD + padQ.  Total: BQ * (BD + padQ) elements.
  //
  //   K_smem:  COLUMN-MAJOR K_smem[k_row + d_col * LDK] = K[k_row][d_col]
  //            LDK = BK + padK.  Total: (BK + padK) * BD elements.
  //            Loaded by BlockLoaderT<BK, BD, kDstStrRow=1, kDstStrCol=LDK, ...>.
  //
  //   V_smem:  ROW-MAJOR   V_smem[v_row * LDV + d_col] = V[v_row][d_col]
  //            LDV = BD + padV.  Total: BK * (BD + padV) elements.
  //            Loaded by BlockLoaderT<BK, BD, kDstStrRow=LDV, kDstStrCol=1, ...>.
  //
  // dO is read directly from device memory, and all three staged tensors use
  // row-major zero-padding layouts. For the D=256 specialization (BQ=16,
  // BK=8), the static budget is:
  //   Q_smem: 16 * 256 * 2 = 8192 bytes
  //   K_smem:  8 * 256 * 2 = 4096 bytes
  //   V_smem:  8 * 256 * 2 = 4096 bytes
  //   Total:                  16384 bytes
  // BK=8 also halves the score, probability, and gradient vectors compared
  // with the M1-incompatible BK=16 specialization.
  //
  // Use row-major K and V with zero padding:
  //   K_smem[k_row * BD + d_col] = K[k_row][d_col]  (row-major, LDK=BD)
  //   V_smem[v_row * BD + d_col] = V[v_row][d_col]  (row-major, LDV=BD)
  //
  // BlockLoaderT for row-major K: kDstStrRow=BD, kDstStrCol=1, reduction_dim=0.
  // This is identical to the VBlockLoader in the forward.
  //
  // Access in scalar loops: K[k][d] = K_smem[k*BD + d].
  //                          V[k][d] = V_smem[k*BD + d].
  //
  // Q_smem uses the same no-padding row-major layout.

  constexpr short LDQ = BD;          // Q_smem: row-major, no padding
  constexpr short LDK = BD;          // K_smem: row-major, no padding
  constexpr short LDV = BD;          // V_smem: row-major, no padding

  threadgroup T Q_smem[BQ * BD];    // [BQ, BD] row-major
  threadgroup T K_smem[BK * BD];    // [BK, BD] row-major
  threadgroup T V_smem[BK * BD];    // [BK, BD] row-major

  // ── Load Q tile (fixed for this threadgroup) ─────────────────────────────
  // QLoader: row-major, kDstStrRow=LDQ, kDstStrCol=1, reduction_dim=1.
  using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, WM*WN*32>;

  const device T* Q_head  = Q   + (long)b * params->Q_strides[0] + (long)h_q * params->Q_strides[1];
  const device T* dO_head = dO  + (long)b * params->O_strides[0] + (long)h_q * params->O_strides[1];

  {
    QLoader lq(Q_head + (long)qb * BQ * params->Q_strides[2], params->Q_strides[2],
               Q_smem, simd_group_id, simd_lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_q_tile) {
      lq.load_safe(short2(BD, params->qL_rem));
    } else {
      lq.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ── Per-thread data from L and D ─────────────────────────────────────────
  // Each simdgroup owns tq_base = simd_group_id * 8 Q-rows.
  // Thread sm = row within 8-row frag → absolute Q-row = q_tile_base + tq_base + sm.
  constexpr short kFrag   = 8;
  constexpr int   kNWarps = WM * WN;

  // Number of simdgroup-matrix D-fragments per thread.
  constexpr int TQ_s = BQ / (kNWarps * kFrag);  // must be 1 for this tile geometry
  constexpr int TD   = BD / kFrag;               // D-cols per thread (8 for D=64, 32 for D=256)

  static_assert(TQ_s == 1, "TQ must be 1 for this tile geometry");

  // simd lane coordinates in the 8×8 frag.
  // get_coord formula (from BaseMMAFrag): qid=lane/4, row=(qid&4)+((lane/2)%4), col=(qid&2)*2+(lane%2)*2
  const ushort qid = simd_lane_id / 4;
  const short  sm  = (short)((qid & 4) + ((simd_lane_id / 2) % 4));  // row in frag [0..7]
  const short  sn  = (short)((qid & 2) * 2 + (simd_lane_id % 2) * 2); // col in frag [0,2,4,6]

  const short tq_base    = (short)simd_group_id * kFrag;
  const int   q_abs_this = q_tile_base + (int)tq_base + (int)sm;
  const bool  row_valid  = (q_abs_this < qL);

  const device float* L_head   = L     + (long)(b * H_q + h_q) * qL;
  const device float* D_head   = D_vec + (long)(b * H_q + h_q) * qL;

  const float L_i = row_valid ? L_head[q_abs_this] : 0.0f;
  const float D_i = row_valid ? D_head[q_abs_this] : 0.0f;

  // ── dQ accumulator (f32, register-local) ─────────────────────────────────
  // Shape: [TQ_s=1, TD] per simdgroup.
  // This thread owns dQ[q_abs_this, d_col] for d_col ∈ {sn + id*kFrag, sn+1 + id*kFrag}.
  float dQ_acc[TD][2];
  // TD reaches 32 at D=256. Keeping this loop rolled avoids materializing the
  // complete accumulator initialization in the M1 pipeline compiler.
  STEEL_PRAGMA_NOUNROLL
  for (int id = 0; id < TD; id++) { dQ_acc[id][0] = 0.f; dQ_acc[id][1] = 0.f; }

  // Shmem offsets are computed inline in the scalar loops below; no pre-computed
  // offset variables are needed (avoids unused-variable Metal warnings).

  // ── K-tile loop ───────────────────────────────────────────────────────────
  for (int kb = 0; kb < params->NK; kb++) {

    // Causal short-circuit: if all K-positions in this tile are future for all Q-rows,
    // skip. K-col range: [kb*BK .. kb*BK+BK-1]. Q-tile min row: q_tile_base.
    // Causally masked: k > q. Skip if kb*BK > max(q_tile_base + BQ - 1), i.e.,
    // ALL Q-rows have k > q for this K-tile.
    if (do_causal) {
      const int q_tile_max = q_tile_base + BQ - 1;
      if (kb * BK > q_tile_max) break;  // K-tiles are ordered, so we can break
    }

    const bool last_k_tile = (!align_K && kb == params->NK_aligned);
    const int  k_abs_base  = kb * BK;

    // ── Load K and V for this K-tile ────────────────────────────────────
    {
      // Row-major layout for both K and V:
      //   K_smem[k_row * LDK + d_col] = K[k_row][d_col]  (LDK = BD)
      //   V_smem[v_row * LDV + d_col] = V[v_row][d_col]  (LDV = BD)
      // BlockLoaderT: kDstStrRow=LDK, kDstStrCol=1, reduction_dim=0.
      using KLoader = BlockLoaderT<T, BK, BD, LDK, 1, 0, WM*WN*32>;
      using VLoader = BlockLoaderT<T, BK, BD, LDV, 1, 0, WM*WN*32>;

      const device T* K_tile = K
          + (long)b    * params->K_strides[0]
          + (long)h_kv * params->K_strides[1]
          + (long)kb   * BK * params->K_strides[2];
      const device T* V_tile = V
          + (long)b    * params->V_strides[0]
          + (long)h_kv * params->V_strides[1]
          + (long)kb   * BK * params->V_strides[2];

      KLoader lk(K_tile, params->K_strides[2], K_smem, simd_group_id, simd_lane_id);
      VLoader lv(V_tile, params->V_strides[2], V_smem, simd_group_id, simd_lane_id);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (last_k_tile) {
        lk.load_safe(short2(BD, params->kL_rem));
        lv.load_safe(short2(BD, params->kL_rem));
      } else {
        lk.load_unsafe();
        lv.load_unsafe();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Scalar FA-2 backward equations for this (Q-tile, K-tile) pair ────
    //
    // Row-major shmem layout (LDQ=LDK=LDV=BD, no padding):
    //   Q_smem[q_row * BD + d] = Q[q_row][d]
    //   K_smem[k_row * BD + d] = K[k_row][d]
    //   V_smem[v_row * BD + d] = V[v_row][d]
    // dO is read directly from device memory (no shmem copy, saves 16 KB at D=256).
    //
    // This thread owns Q row q_abs_this.  It skips OOB rows.

    if (!row_valid) {
      // Skip OOB Q-rows — but still need the barrier to stay in sync with
      // valid threads that will call threadgroup_barrier below.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      continue;
    }

    // Base pointer to this thread's dO row in device memory.
    const device T* do_row_ptr = dO_head + (long)q_abs_this * params->O_strides[2];

    // S[k] = scale * sum_d Q[q_abs_this][d] * K[k][d]
    float S_vec[BK];
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < BK; k++) {
      float acc = 0.f;
      // Full-unrolling 256 FMAs for every BK lane expands the D=256 pipeline
      // aggressively; retain the constant-bound runtime loop on all devices.
      STEEL_PRAGMA_NOUNROLL
      for (short d = 0; d < BD; d++) {
        acc += float(Q_smem[(tq_base+sm)*LDQ + d]) * float(K_smem[k*LDK + d]);
      }
      S_vec[k] = sc * acc;
    }

    // Apply masks to S_vec.
    // Out-of-bounds K cols.
    if (last_k_tile) {
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < BK; k++) {
        if (k >= params->kL_rem) S_vec[k] = -metal::numeric_limits<float>::infinity();
      }
    }
    // Causal: S[k] = -inf when k_abs_base+k > q_abs_this.
    if (do_causal) {
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < BK; k++) {
        if (k_abs_base + k > q_abs_this) S_vec[k] = -metal::numeric_limits<float>::infinity();
      }
    }
    // Additive mask.
    if (has_mask) {
      const device T* mask_row = mask
          + (long)b   * mask_params->M_strides[0]
          + (long)h_q * mask_params->M_strides[1]
          + (long)q_abs_this * mask_params->M_strides[2];
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < BK; k++) {
        const int k_abs = k_abs_base + k;
        float mv = (k_abs < kL) ? float(mask_row[k_abs]) : -metal::numeric_limits<float>::infinity();
        S_vec[k] = (mv == -metal::numeric_limits<float>::infinity()) ? mv : S_vec[k] + mv;
      }
    }

    // P[k] = exp(S[k] - L[q_abs_this]).
    float P_vec[BK];
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < BK; k++) {
      P_vec[k] = metal::exp(S_vec[k] - L_i);
    }

    // dP[k] = sum_d dO[q_abs_this][d] * V[k][d].
    float dP_vec[BK];
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < BK; k++) {
      float acc = 0.f;
      STEEL_PRAGMA_NOUNROLL
      for (short d = 0; d < BD; d++) {
        acc += float(do_row_ptr[d]) * float(V_smem[k*LDV + d]);
      }
      dP_vec[k] = acc;
    }

    // dS[k] = P[k] * (dP[k] - D[q_abs_this]).
    float dS_vec[BK];
    STEEL_PRAGMA_UNROLL
    for (short k = 0; k < BK; k++) {
      dS_vec[k] = P_vec[k] * (dP_vec[k] - D_i);
    }

    // dQ[q_abs_this][d] += scale * sum_k dS[k] * K[k][d].
    // dQ_acc[id][0/1] accumulates for d = id*kFrag+sn and id*kFrag+sn+1.
    STEEL_PRAGMA_NOUNROLL
    for (short id = 0; id < TD; id++) {
      const short d0 = id * kFrag + sn;
      const short d1 = d0 + 1;
      float acc0 = 0.f, acc1 = 0.f;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < BK; k++) {
        float ds = dS_vec[k];
        if (d0 < BD) acc0 += ds * float(K_smem[k*LDK + d0]);
        if (d1 < BD) acc1 += ds * float(K_smem[k*LDK + d1]);
      }
      dQ_acc[id][0] += sc * acc0;
      dQ_acc[id][1] += sc * acc1;
    }

    // dK[k][d] += scale * dS[k] * Q[q_abs_this][d].  (atomic f32)
    // dV[k][d] += P[k] * dO[q_abs_this][d].           (atomic f32)
    //
    // dK and dV device buffers: [B, H_kv, kL, D] f32, row-major.
    //
    // WHY sn == 0 gate: within each simdgroup of 32 lanes the formula
    //   sm = (qid & 4) + ((lane/2) % 4),  sn = (qid & 2)*2 + (lane%2)*2
    // assigns each sm value (Q-row) to exactly 4 lanes (sn ∈ {0,2,4,6}).
    // All 4 lanes compute identical S_vec / P_vec / dS_vec (same Q-row,
    // same K/V tile), so without a gate they each atomic-add the same
    // value → 4× overcounting in dK and dV.
    // Gating on sn == 0 lets exactly one lane per (simdgroup, Q-row)
    // perform the full BD-wide accumulation.
    if (sn == 0) {
      const long kv_row_base_elems =
          (long)b    * (long)(H_kv * kL * D)
        + (long)h_kv * (long)(kL * D);

      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < BK; k++) {
        const int k_abs = k_abs_base + k;
        if (k_abs >= kL) continue;

        const float p_k  = P_vec[k];
        const float ds_k = dS_vec[k];
        const long row_elem = kv_row_base_elems + (long)k_abs * D;

        STEEL_PRAGMA_NOUNROLL
        for (short d = 0; d < BD; d++) {
          const float q_val  = float(Q_smem[(tq_base+sm)*LDQ + d]);
          const float do_val = float(do_row_ptr[d]);
          atomic_fetch_add_explicit(
              (device atomic_float*)&dK[row_elem + d],
              sc * ds_k * q_val, memory_order_relaxed);
          atomic_fetch_add_explicit(
              (device atomic_float*)&dV[row_elem + d],
              p_k * do_val, memory_order_relaxed);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  } // end K-tile loop

  // ── Write dQ for this Q-tile ───────────────────────────────────────────────
  if (!row_valid) return;

  device float* dQ_head = dQ
      + (long)b   * (long)(H_q * qL * D)
      + (long)h_q * (long)(qL * D)
      + (long)q_abs_this * D;

  STEEL_PRAGMA_NOUNROLL
  for (short id = 0; id < TD; id++) {
    const short d0 = id * kFrag + sn;
    const short d1 = d0 + 1;
    if (d0 < D) dQ_head[d0] = dQ_acc[id][0];
    if (d1 < D) dQ_head[d1] = dQ_acc[id][1];
  }
}
// clang-format on

// ─── bf16 cast kernel (f32 → bf16, elementwise) ──────────────────────────────
//
// Grid: dispatch_thread_groups(ceil(n/tg_x), 1, 1), tg_size=(tg_x,1,1).
// buffer(2) carries n_elems as a uint to guard the last partial tile.
[[kernel]] void f32_to_bf16_cast(
    const device float*    src     [[buffer(0)]],
    device bfloat16_t*     dst     [[buffer(1)]],
    const constant uint*   n_elems [[buffer(2)]],
    uint                   tid     [[thread_position_in_grid]])
{
  if (tid < *n_elems) dst[tid] = bfloat16_t(src[tid]);
}

// ─── Instantiations ───────────────────────────────────────────────────────────

#define instantiate_bwd(name, io_t, bq, bk, bd, wm, wn) \
  template [[host_name(name)]] [[kernel]] \
  decltype(attention_train_bwd<io_t, bq, bk, bd, wm, wn>) \
  attention_train_bwd<io_t, bq, bk, bd, wm, wn>;

// D=64 retains the 32-row/16-key/four-simdgroup tile. D=256 uses a
// 16-row/eight-key/two-simdgroup tile, reducing static threadgroup memory from
// exactly 32 KiB to 16 KiB and halving its per-thread BK working set.
instantiate_bwd("flash_attn_train_bwd_bf16_d64",  bfloat16_t, 32, 16,  64, 4, 1)
instantiate_bwd("flash_attn_train_bwd_bf16_d256", bfloat16_t, 16,  8, 256, 2, 1)
