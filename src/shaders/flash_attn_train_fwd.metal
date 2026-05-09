// flash_attn_train_fwd — FA-2 forward with logsumexp output.
//
// Fork of flash_attn_prefill.metal.  Algorithm is IDENTICAL; the ONLY delta
// is an additional device float* L_out [[buffer(8)]] that receives the
// per-row natural-log logsumexp after the K-tile sweep.
//
// Logsumexp formula (FA-2 Algorithm 1 convention, nat-log domain):
//
//   Q is pre-scaled by scale * log2(e), so:
//     max_score[r]  — running row-max in base-2 space
//     sum_score[r]  — sum_j exp2(s_ij - max_score[r])
//
//   Convert to nat-log:
//     L_i = max_score[r] * ln(2)  +  ln(sum_score[r])
//         = max_score[r] * M_LN2_F + log(sum_score[r])
//
//   Matches FA-2 paper eq. (5): L_i = m_i + log(sum_j exp(s_ij - m_i)).
//   Backward uses exp(S_ij - L_i) to recompute softmax weights.
//
//   Fully-masked row: sum_score[r] == 0.0 → log(0.0f) = -inf (IEEE-754).
//   Correct: O_i = 0 for fully-masked rows, so dO_i = 0 in the backward.
//
// Buffer layout:
//   buffer(0)  Q      [B, H,    qL, D]  bf16
//   buffer(1)  K      [B, H_kv, kL, D]  bf16
//   buffer(2)  V      [B, H_kv, kL, D]  bf16
//   buffer(3)  O      [B, H,    qL, D]  bf16   (output)
//   buffer(4)  AttnParams constant block (160 bytes, same ABI as prefill)
//   buffer(5)  AttnMaskParams  [function_constant(has_mask)]
//   buffer(6)  mask            [function_constant(has_mask)]
//   buffer(8)  L_out  [B, H,    qL]    f32    (nat-log logsumexp output)
//
// Function constants (same indices as flash_attn_prefill.metal):
//   200: align_Q   bool — qL % BQ == 0
//   201: align_K   bool — kL % BK == 0
//   300: has_mask  bool — additive/bool mask buffer is bound
//   301: do_causal bool — in-kernel causal masking
//
// Kernel variants:
//   flash_attn_train_fwd_bf16_d64          BQ=32, BK=16, BD=64,  WM=4, WN=1
//   flash_attn_train_fwd_bf16_d64_boolmask (bool mask variant)
//   flash_attn_train_fwd_bf16_d256         BQ=32, BK=16, BD=256, WM=4, WN=1
//   flash_attn_train_fwd_bf16_d256_boolmask
//
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// ─── bfloat16 compat shim (matches flash_attn_prefill.metal verbatim) ────────

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;
typedef half float16_t;

#else

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  uint32_t float_bits = as_type<uint32_t>(x);
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;
template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}
  template <typename T, typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
  template <typename T, typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }
#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);
bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);
bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);
#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }
#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);
#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);
bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);
#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }
#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);
bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);
#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

// ─── BlockLoaderT ─────────────────────────────────────────────────────────────

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;
  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;

  METAL_FUNC BlockLoaderT(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] =
            op.apply(dst[i * kDstStrRow + j * kDstStrCol]);
      }
    }
  }

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

// ─── Integral constant ────────────────────────────────────────────────────────

template <int val>
using Int = integral_constant<int, val>;

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }
integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);
integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);
integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);
#undef integral_const_binop

// ─── MMA frag + tile ──────────────────────────────────────────────────────────

template <typename T>
struct pointer_element {};
template <typename T>
struct pointer_element<thread T*> { using type = remove_cv_t<T>; };
template <typename T>
struct pointer_element<device T*> { using type = remove_cv_t<T>; };
template <typename T>
struct pointer_element<constant T*> { using type = remove_cv_t<T>; };
template <typename T>
struct pointer_element<threadgroup T*> { using type = remove_cv_t<T>; };
template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(kFragRows_ == 8, "Only 8x8 MMA frags supported");
  static_assert(kFragCols_ == 8, "Only 8x8 MMA frags supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32; // = 2
  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(kElemRows * kElemCols == kElemsPerFrag, "Shape inconsistency");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  template <typename U>
  using dtype_mat_t = metal::simdgroup_matrix<U, kFragRows, kFragCols>;
  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY, typename OffX, typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y,
      OffX off_x = Int<0>{}, OffY off_y = Int<0>{}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY, typename OffX, typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y,
      OffX off_x = Int<0>{}, OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;
    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;
    mma(D_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread dtype_mat_t<Atype>& A,
      thread dtype_mat_t<Btype>& B,
      thread dtype_mat_t<Ctype>& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);
    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <typename T, int kTileRows_, int kTileCols_,
          class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows    = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols    = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;
  STEEL_CONST int kTileRows    = kTileRows_;
  STEEL_CONST int kTileCols    = kTileCols_;
  STEEL_CONST int kRows        = kTileRows * kFragRows;
  STEEL_CONST int kCols        = kTileCols * kFragCols;
  STEEL_CONST int kNumFrags    = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;
  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) val_frags[i] = frag_type(0);
  }

  METAL_FUNC constexpr thread frag_type& frag_at(short i, short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(short i, short j) const {
    return val_frags[i * kTileCols + j];
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j)
        MMAFrag_t::template row_reduce<Op>(frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j)
        MMAFrag_t::template row_bin_op<Op>(frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
  }

  // Load from threadgroup (str_x, str_y as compile-time constants)
  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j)
        MMAFrag_t::load(frag_at(i, j),
            &(src[(i * kFragRows) * w_x * str_x + (j * kFragCols) * w_y * str_y]),
            Int<str_x>{}, Int<str_y>{});
  }

  // Store to device (w_x, w_y direction, ld runtime stride)
  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i)
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j)
        MMAFrag_t::store(frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld, Int<1>{});
  }

  // Store to device with bounds check
  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i)
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j)
        MMAFrag_t::store_safe(frag_at(i, j), dst, ld, Int<1>{},
            dst_tile_dims.y, dst_tile_dims.x,
            (i * kFragRows) * w_x, (j * kFragCols) * w_y);
  }
};

// ─── tile_matmad ─────────────────────────────────────────────────────────────
// Must be defined BEFORE the kernel template that calls it.

template <typename Dtype, typename Atype, typename Btype, typename Ctype,
          int M, int N, int K,
          class MMAFragD, class MMAFragA, class MMAFragB, class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}

// ─── AttnParams ABI (160-byte, identical to flash_attn_prefill.metal) ─────────

struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  float softcapping;
  int NQ;
  int NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
  // 4 bytes implicit pad before the first int64_t (compiler-inserted)
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

struct AttnMaskParams {
  int64_t M_strides[3];
};

// ─── Softmax ops (identical to flash_attn_prefill.metal) ─────────────────────

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  // Unguarded: M is finite so exp2(x-M) = 0 when x = -inf (IEEE-754 exact).
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct DivOp {
  // Single surviving guard: fully-masked row → sum_score == 0 → output 0.
  // Matches llama.cpp ggml-metal.metal:6358.
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == T(0)) ? T(0) : x / y;
  }
};
template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T s) : scale(s) {}
  METAL_FUNC T apply(T x) const { return scale * x; }
};

// ─── Function constants ───────────────────────────────────────────────────────

constant bool align_Q   [[function_constant(200)]];
constant bool align_K   [[function_constant(201)]];
constant bool has_mask  [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

// ─── attention_train_fwd kernel ───────────────────────────────────────────────

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_train_fwd(
    const device T*  Q         [[buffer(0)]],
    const device T*  K         [[buffer(1)]],
    const device T*  V         [[buffer(2)]],
    device T*        O         [[buffer(3)]],
    const constant AttnParams* params       [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask               [[buffer(6), function_constant(has_mask)]],
    device float*    L_out     [[buffer(8)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  // ── Move pointers to this threadgroup's tile ──────────────────────────────

  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] +
       tidl.y * params->Q_strides[1] +
       tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] +
       kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] +
       kv_head_idx * params->V_strides[1];

  O += tidl.z * params->O_strides[0] +
       tidl.y * params->O_strides[1] +
       tidl.x * BQ * params->O_strides[2];

  // L_out layout: [B, H, qL] row-major.
  // Advance to (b=tid.z, h=tid.y) plane; per-row write uses abs_row below.
  L_out += tidl.z * (ulong)(params->H * params->qL) +
           tidl.y * (ulong)(params->qL);

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] +
            tidl.y * mask_params->M_strides[1];
  }

  // ── Threadgroup shared memory ──────────────────────────────────────────────

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * BD;
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];
  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // ── Block loaders ──────────────────────────────────────────────────────────

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

  QBlockLoader loader_q(Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  // Pre-scale Q by scale * log2(e) so inner products are in base-2 space.
  TransformScale<T> ts(static_cast<T>(params->scale * 1.44269504089));

  // ── MMA tile setup ────────────────────────────────────────────────────────

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kFragSize); // = 1
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "TQ must be 1 for this kernel");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;  // row within 8x8 frag
  const short sn = simd_coord.x;  // col offset within 8x8 frag
  const short tm = kFragSize * TQ * simd_group_id; // simdgroup row base

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  // ── Load Q + apply scale ──────────────────────────────────────────────────

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // ── Init per-row softmax state ────────────────────────────────────────────

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread; // = 1

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Finite-M sentinel: -FLT_MAX/2 so exp2(masked_score - M) = 0, not NaN.
  // Matches llama.cpp ggml-metal.metal:5891; see flash_attn_prefill.metal preamble.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -FLT_MAX / AccumType(2);
  }

  // ── Causal kb_lim (same logic as flash_attn_prefill.metal:1325-1348) ──────

  int kb_lim = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    int causal_kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(kb_lim, causal_kb_lim);
  }

  // ── K-tile sweep ──────────────────────────────────────────────────────────

  for (int kb = 0; kb < kb_lim; kb++) {

    // Load K tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // S = Q @ K^T  (MMA in base-2 scale)
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Mask out K positions beyond kL_rem in the last partial K tile
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Causal mask: positions where row_abs < col_abs get score = -inf
    if (do_causal && kb >= (kb_lim - (BQ + BK - 1) / BK - int(!align_K))) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Additive/bool mask
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;
      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          frag_t mfrag;
          MMAFrag_mask_t::load_safe(
              mfrag, mask,
              mask_params->M_strides[2], Int<1>{},
              params->qL, params->kL,
              row_pos, col_pos);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              // Additive mask in natural-log space → multiply by log2(e) for base-2 space
              Stile.frag_at(i, j)[jj] += 1.44269504089 * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    // Load V tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Online softmax update
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

    Stile.template row_reduce<MaxOp>(new_max);          // new_max = row max of S
    Stile.template row_bin_op<ExpSubOp>(new_max);        // S = exp2(S - new_max)

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]); // rescale = exp2(old_max - new_max)
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);     // sum of exp2 values

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);            // rescale O

    // Accumulate O += softmax(S) @ V
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + kk * LDV_tgp + dd]);
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    loader_k.next();
    loader_v.next();
  } // end K-tile sweep

  // ── Normalise O ───────────────────────────────────────────────────────────

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // ── Emit L_out (logsumexp in natural-log domain) ──────────────────────────
  //
  // FA-2 Algorithm 1 eq.(5):  L_i = m_i + log( sum_j exp(s_ij - m_i) )
  //
  // Kernel accumulates in base-2:
  //   max_score[0]  = row-max in base-2 = m_i * log2(e)
  //   sum_score[0]  = sum_j exp2(s_ij - max_score[0])
  //                 = sum_j exp( (s_ij - m_i*log2(e)) * ln(2) )
  //
  // Converting:
  //   m_i_nat  = max_score[0] * ln(2)          [base-2 → nat-log domain]
  //   log(sum_score[0])                         [already in nat-log]
  //   L_i = m_i_nat + log(sum_score[0])
  //       = max_score[0] * M_LN2_F + log(sum_score[0])
  //
  // Guard: sn == 0 selects one thread per row (all threads sharing a row
  // have the same max_score[0] / sum_score[0] after simd-shuffle row_reduce).
  // Bound check: abs_row < qL guards the last partial Q tile.
  //
  // Fully-masked row: sum_score[0] == 0.0 → log(0.0f) = -inf (correct).
  {
    // M_LN2_F is already a Metal SDK macro — use a local name to avoid collision.
    constexpr float LN2_F = 0.693147180559945f; // = log(2.0f)

    const int abs_row = int(tid.x) * BQ + tm + sm;

    if (sn == 0 && abs_row < params->qL) {
      L_out[abs_row] = max_score[0] * LN2_F + log(sum_score[0]);
    }
  }

  // ── Store O ───────────────────────────────────────────────────────────────

  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

// clang-format off

// ──────────────────────────────────────────────────────────────────────────
// Kernel instantiations
// ──────────────────────────────────────────────────────────────────────────

#define instantiate_train_fwd(name, io_dtype, bq, bk, bd, wm, wn, mask_dtype) \
  template [[host_name(name)]] [[kernel]]                                       \
  decltype(attention_train_fwd<io_dtype, bq, bk, bd, wm, wn, mask_dtype, float>) \
  attention_train_fwd<io_dtype, bq, bk, bd, wm, wn, mask_dtype, float>;

// D=64, bf16 I/O.  BQ=32, BK=16, WM=4, WN=1 → 128 threads/threadgroup.
// Threadgroup memory: ~5 KB at bf16 — well under 32 KB Apple Silicon limit.
// TQ=1 ✓: BQ/(WM*WN*kFragSize) = 32/(4*8) = 1.
instantiate_train_fwd("flash_attn_train_fwd_bf16_d64",          bfloat16_t, 32, 16,  64, 4, 1, bfloat16_t)
instantiate_train_fwd("flash_attn_train_fwd_bf16_d64_boolmask", bfloat16_t, 32, 16,  64, 4, 1, bool)

// D=256, bf16 I/O.  Same tile geometry.  Threadgroup memory: ~29 KB — fits.
// Production Qwen3.6-35B-A3B shape (D=256, n_heads=16, n_kv_heads=2).
instantiate_train_fwd("flash_attn_train_fwd_bf16_d256",          bfloat16_t, 32, 16, 256, 4, 1, bfloat16_t)
instantiate_train_fwd("flash_attn_train_fwd_bf16_d256_boolmask", bfloat16_t, 32, 16, 256, 4, 1, bool)

// clang-format on
