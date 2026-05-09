// flash_attn_train_bwd_compute_d — FA-2 backward pre-pass: compute D vector.
//
// D[b, h, i] = rowsum_d( O[b, h, i, d] * dO[b, h, i, d] )
//
// This is equation (5) in Dao 2023 FA-2 Algorithm 4.  One threadgroup per
// (b, h, i) row.  Threads in the group stride over head_dim with a simd
// tree-reduction.
//
// Buffer layout:
//   buffer(0)  O      [B, H_q, qL, D]  bf16  (forward output)
//   buffer(1)  dO     [B, H_q, qL, D]  bf16  (upstream gradient)
//   buffer(2)  D_out  [B, H_q, qL]     f32   (output)
//   buffer(3)  params [4 * uint32]     —     {B, H_q, qL, D}
//
// Grid geometry:
//   dispatch_thread_groups(threadgroups=qL, 1, B*H_q)
//   threadgroup_size      = min(256, next_pow2(D))
//
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ── bfloat16 compat shim ───────────────────────────────────────────────────
// Copied verbatim from flash_attn_train_fwd.metal so this file is
// self-contained and compiles independently.

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

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

typedef struct _MLX_BFloat16 bfloat16_t;

#endif // __HAVE_BFLOAT__

// ── Param struct ────────────────────────────────────────────────────────────

struct ComputeDParams {
  uint batch;
  uint n_q_heads;
  uint q_seq_len;
  uint head_dim;
};

// ── Kernel ──────────────────────────────────────────────────────────────────
//
// Grid: dispatch_thread_groups(qL, 1, B * H_q)
//   tid.z = flat (batch * n_q_heads) index = b * H_q + h
//   tid.x = query sequence position i
// Threadgroup: (tg_size, 1, 1) where tg_size = min(256, next_pow2(D))

[[kernel]] void flash_attn_train_bwd_compute_d_bf16(
    const device bfloat16_t*       O      [[buffer(0)]],
    const device bfloat16_t*       dO     [[buffer(1)]],
    device float*                  D_out  [[buffer(2)]],
    const constant ComputeDParams* params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tgs [[threads_per_threadgroup]])
{
  // Unpack: x=tid-in-tg, gid.x=row(i), gid.z=flat(b,h).
  const uint tid_x   = tid.x;
  const uint tg_size = tgs.x;
  const uint i       = gid.x;          // query row
  const uint bh      = gid.z;          // flat (batch * n_q_heads) index
  const uint D       = params->head_dim;
  const uint qL      = params->q_seq_len;

  if (i >= qL) return;              // out-of-bounds guard

  // Base offset into [B, H, qL, D] bf16 layout.
  const uint row_base = bh * qL * D + i * D;

  // Each thread accumulates a partial sum over its stride of D.
  float partial = 0.0f;
  for (uint d = tid_x; d < D; d += tg_size) {
    float o_val  = float(O[row_base + d]);
    float do_val = float(dO[row_base + d]);
    partial += o_val * do_val;
  }

  // Simd-group tree reduction within each warp.
  partial = simd_sum(partial);

  // Threadgroup reduction across warps via shared memory (up to 8 warps = 256 threads).
  threadgroup float smem[8];
  const uint simd_lane  = tid_x % 32;
  const uint simd_group = tid_x / 32;

  if (simd_lane == 0) {
    smem[simd_group] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final cross-warp reduction done by the first warp.
  if (simd_group == 0) {
    const uint n_warps = (tg_size + 31) / 32;
    float val = (simd_lane < n_warps) ? smem[simd_lane] : 0.0f;
    val = simd_sum(val);
    if (simd_lane == 0) {
      // D_out layout: [B, H_q, qL] row-major.
      D_out[bh * qL + i] = val;
    }
  }
}
