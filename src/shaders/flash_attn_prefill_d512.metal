// flash_attn_prefill_d512 — mlx-native NSG=8 flash-attention prefill kernel
// ported from llama.cpp's `kernel_flash_attn_ext_impl` template.
//
// ## What this file is
//
// A SECOND, independent Metal kernel implementing scaled-dot-product attention
// prefill for head_dim=512 (DK=DV=512) using llama.cpp's per-simdgroup-Q-
// distributed decomposition.  It is the companion to, NOT a replacement for,
// `flash_attn_prefill.metal` — the candle-derived D=256 kernel there is
// retained as-is (correct, fast, well-tested).
//
// ## Why a separate file (vs extending `flash_attn_prefill.metal`)
//
// The candle-derived template in `flash_attn_prefill.metal` uses a
// per-warp-Q-stacking layout where each simdgroup owns `BQ / kNWarps`
// Q rows of work.  The `BQ >= kNWarps * kFragSize` static_assert
// (`flash_attn_prefill.metal:1250-1251`) forces a minimum BQ of 64 at NSG=8
// (`kNWarps=WM*WN=8`), which at BD=512 bf16 needs a Q threadgroup tile of
// `64 * 520 * 2 = 66,560 B` — over 2× the 32 KiB Apple Silicon threadgroup-
// memory budget.  Cannot host NSG=8 at D=512.
//
// llama.cpp gets NSG=8 at D=512 within 28 KiB by using a FUNDAMENTALLY
// DIFFERENT per-simdgroup decomposition:
//
//   - Q rows are DISTRIBUTED across simdgroups (NQ = Q/NSG rows per simdgroup),
//     not stacked.  The Q tile lives ONCE in threadgroup memory (`sq[Q × DK]`)
//     and each simdgroup walks its owned Q rows by indexing `j = jj*NSG + sgitg`.
//   - The output `O` accumulator lives in THREADGROUP MEMORY (`so[Q × PV]` in
//     `half`), not in registers.  Each KV chunk pays `load + store` of the
//     per-simdgroup Otile frags (NO = PV8/NSG = 8 frags at NSG=8) to trade
//     shmem traffic for far lower register pressure (16 vs 128 per-thread
//     registers), which drives occupancy up by 8× on M5 Max.
//   - K/V are loaded directly from device memory via `simdgroup_load(…,
//     NS10, 0, true)` for K and `simdgroup_load(…, NS20, 0, false)` for V,
//     with each simdgroup owning disjoint 8-column slices of the K×V tile.
//
// These three differences are architectural — they cannot be reached from
// the candle template by parameter tuning.  The new kernel is a direct
// port of `kernel_flash_attn_ext_impl` from llama.cpp's
// `ggml-metal.metal:5736-6375`, specialised for:
//
//   - DK = DV = 512 only
//   - bf16/f16 I/O (f32 excluded as in flash_attn_prefill.metal — 32 KiB TG
//     budget is the hard limit)
//   - Unquantized K/V cache (is_q=0 branch only — llama.cpp's quantized
//     branch at `:6066-6127, :6257-6322` is dropped)
//   - NQPSG=8, NCPSG=64 hard-coded (as llama.cpp does at the host thunk
//     defaults `ggml-metal.metal:6403-6404`)
//   - NSG selectable at pipeline-creation time via the int function constant
//     `nsg` (index 322, mirrors llama.cpp's `FC_flash_attn_ext_nsg` at
//     `ggml-metal.metal:5735` = `FC_FLASH_ATTN_EXT + 22`).  Supported
//     values: 4 and 8.  NSG=8 is the required configuration for the 32 KiB
//     budget headroom win (§2.3 in ADR-011-phase2-port-d512.md); NSG=4
//     is provided to mirror llama.cpp's dispatch flexibility.
//   - In-kernel causal masking + additive (bf16/f16) OR bool mask, matching
//     the D=256 kernel's API surface (same function-constant indices for
//     `align_Q, align_K, has_mask, do_causal`).
//
// ## Features excluded vs llama.cpp's kernel (intentional scope reduction)
//
// These branches exist in llama.cpp's kernel and are dropped here.  They are
// all zero-cost-when-not-used (dead-code eliminated via function constants
// at pipeline creation), so retaining them would be free at runtime but
// would add ~200 LOC of complexity.  If future Gemma / DeepSeek-style
// features land, they can be re-added in a follow-up ADR:
//
//   - Quantized K/V cache (is_q=1 path): llama.cpp's Gemma 4 runs bf16 K/V,
//     matching our pipeline.  Port `:6066-6127, :6257-6322` when needed.
//   - Sinks (`FC_flash_attn_ext_has_sinks`): attention-sinks for StreamingLLM.
//     llama.cpp `:5722, :6328-6346`.
//   - ALiBi-style bias (`FC_flash_attn_ext_has_bias`): not used by Gemma 4.
//     llama.cpp `:5723, :5896-5903, :6146-6150`.
//   - Softcap (`FC_flash_attn_ext_has_scap`): Gemma 2 used it; Gemma 4
//     doesn't.  llama.cpp `:5724, :6140-6142`.
//   - KV-pad tail handling (`FC_flash_attn_ext_has_kvpad`): zero-copy padding
//     of the trailing partial KV tile.  We handle kL % NCPSG via per-position
//     -inf mask-out in the last KV chunk instead (matches our D=256 kernel's
//     approach via `align_K=false` + `kL_rem`).  llama.cpp `:5725, :5914-5949`.
//   - Broadcast mask (`FC_flash_attn_ext_bc_mask`): mask broadcast across
//     Q-rows.  Not used by our dispatcher.  llama.cpp `:5727, :5969-5970`.
//   - Per-tile pre-pass skip (`blk`): llama.cpp's `flash_attn_ext_blk`
//     writes a `{0,1,2}` bitmap per KV chunk letting the prefill skip
//     all-masked or pass-through chunks.  Deferred to Phase 4.  We treat
//     every chunk as `blk_cur = 1` (full mask).  llama.cpp `:5775, :5951-6005`.
//
// ## Numerical regime — identical to the D=256 kernel
//
// The row-max `M` is initialised to the FINITE sentinel `-FLT_MAX/2`
// (llama.cpp convention, `ggml-metal.metal:5891`).  Masked scores arrive
// as `-inf` from the additive mask buffer; `simd_max(-FLT_MAX/2, -inf)`
// floors at `-FLT_MAX/2` so `M` stays finite.  Every `exp(score - M)`
// evaluates as `exp(-inf) = +0.0` (IEEE-754 exact), never NaN.  The ONE
// output-side guard is at the final `output / sum_score` store:
//
//     S == 0 ? 0 : 1/S
//
// mirroring `ggml-metal.metal:6358`.  Same regime as `flash_attn_prefill.metal`;
// see that file's preamble for the full rationale and
// ADR-011-phase2-port-sentinel.md §1-3 for the line-by-line trace.
//
// ## Exponential base
//
// llama.cpp uses natural-base `exp` throughout.  Our candle-derived D=256
// kernel uses `fast::exp2` with Q pre-scaled by `scale * log2(e)`.  This
// file MIRRORS the D=256 choice (exp2 + pre-scale) so the host-side
// `TransformScale` contract and `AttnParams::scale` semantics are identical
// across both kernels.  The mathematical result is unchanged.
//
// ## Shared-memory layout (dynamic, sized by host)
//
// The dispatcher passes `threadgroup half*` with size `smem_bytes` as
// set via `setThreadgroupMemoryLength` on the encoder:
//
//   Offset 0         : sq[Q × DK]   — query tile, as I/O-dtype T (bf16/f16)
//   Offset Q×DK×2    : so[Q × PV]   — output accumulator in half
//   Offset Q×(DK+2PV): ss[Q × SH]   — softmax/mask scratch (SH = 2*C)
//   Offset Q×T +     : sk/sv ... per-simdgroup dequant scratch (is_q=1 only;
//   sgitg*(4*16*KV)    dropped in this port)
//   Total: Q × (DK + 2*PV) × sizeof(half) = 8 × (512 + 1024) × 2 = 24,576 B
//   Padded to 16 B alignment: 24,576 B.  We set 28,672 B to match llama.cpp
//   exactly (§2.3 in ADR-011-phase2-port-d512.md).
//
// References (kernel body is a direct port; see inline citations):
//   - /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375
//     (kernel_flash_attn_ext_impl — single template, all DKs, all NSGs)
//   - /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6421-6427
//     (dispatch thunk; cases 4/8 → kernel_flash_attn_ext_impl<…, NSG>)
//   - /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:77, 93-94
//     (FC offsets, NQPSG=8, NCPSG=64 defines)
//   - /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2750-2900
//     (host dispatch: ne00>=512 → nsg=8; FATTN_SMEM size formula; grid)
//   - ADR-011-phase2-port-d512.md  (port spec; this file follows §7 checklist)
//
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// `FOR_UNROLL` mirrors llama.cpp's macro at ggml-metal.metal:26.
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ──────────────────────────────────────────────────────────────────────────
// Common structs (same ABI as flash_attn_prefill.metal)
// ──────────────────────────────────────────────────────────────────────────
//
// Host mirror: /opt/mlx-native/src/ops/flash_attn_prefill.rs AttnParamsGpu.
// Field order / types / total 160 B byte-for-byte identical to the
// definition in flash_attn_prefill.metal:1025-1051.
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

  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

struct AttnMaskParams {
  int64_t M_strides[3];
};

// ──────────────────────────────────────────────────────────────────────────
// Function constants
// ──────────────────────────────────────────────────────────────────────────
//
// Indices 200/201/300/301 MATCH flash_attn_prefill.metal so dispatcher
// helpers are uniform across D=256 and D=512.
//
// Index 322 is the int function constant controlling NSG — mirrors
// llama.cpp's `FC_flash_attn_ext_nsg` (ggml-metal.metal:5735; value is
// `FC_FLASH_ATTN_EXT + 22` = 300 + 22 = 322 in their numbering scheme,
// adopted here to keep semantic parity).  Supported values: 4, 8.

constant bool align_Q  [[function_constant(200)]];
constant bool align_K  [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

// Wave 2E tile-skip pre-pass.  See flash_attn_prefill.metal's `has_blk`
// declaration for the full contract; same index 303 across both D=256
// and D=512 so the dispatcher can set one bool and feed either kernel.
constant bool has_blk  [[function_constant(303)]];

constant int  fc_nsg   [[function_constant(322)]];

// Provide sensible defaults when a function constant isn't set (avoids
// undefined-behaviour during preview compiles).  Actual values are always
// supplied by the dispatcher at pipeline-creation time.
constant bool align_Q_def  = is_function_constant_defined(align_Q)  ? align_Q  : true;
constant bool align_K_def  = is_function_constant_defined(align_K)  ? align_K  : true;
constant bool has_mask_def = is_function_constant_defined(has_mask) ? has_mask : false;
constant bool do_causal_def = is_function_constant_defined(do_causal) ? do_causal : false;
constant bool has_blk_def  = is_function_constant_defined(has_blk)  ? has_blk  : false;
constant int  nsg_def      = is_function_constant_defined(fc_nsg)   ? fc_nsg   : 8;

// ──────────────────────────────────────────────────────────────────────────
// Kernel template
// ──────────────────────────────────────────────────────────────────────────
//
// Faithful port of `kernel_flash_attn_ext_impl` from
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375 for the
// (DK=DV=512, bf16/f16 K/V, is_q=0) slice, with NSG as a template parameter
// (compile-time, specialised from the int function constant via the outer
// wrapper below).
//
// T    — I/O scalar type: bfloat or half.
// MaskT — Mask scalar type: same as T (additive) or `bool` (is_attended).
// NSG  — Simdgroups per threadgroup: 4 or 8.
//
// See llama.cpp kernel_flash_attn_ext_impl template parameters at
// ggml-metal.metal:5738-5766 for the full type-plumbing model; we collapse
// the 24 type/arity parameters there to {T, MaskT, NSG} because we fix
// DK=DV=512, q_t=k_t=v_t=kd_t=vd_t=T, q8x8_t=k8x8_t=v8x8_t=simdgroup_T8x8,
// qk_t=s_t=float, o_t=T (to match our D=256 kernel's output precision
// contract — bf16/f16 final store, f32 internal accum).
// NOTE: `flash_attn_prefill_d512_impl` is a HELPER function, NOT a Metal
// kernel.  It does not carry `[[kernel]]` — a kernel cannot be called from
// another function in MSL, but a plain function with threadgroup/grid/
// lane-id arguments forwarded from the caller is legal.  This matches
// llama.cpp's `kernel_flash_attn_ext_impl` at ggml-metal.metal:5767
// (declared without `[[kernel]]`; called from the outer `kernel void
// kernel_flash_attn_ext` thunk at :6405).
template <typename T, typename MaskT, short NSG>
void flash_attn_prefill_d512_impl(
    const device char* q_base,
    const device char* k_base,
    const device char* v_base,
    device       char* o_base,
    const constant AttnParams& args,
    const constant AttnMaskParams& mask_args,
    const device MaskT* mask_base,
    const device char*  blk_base,
    threadgroup  half* shmem_f16,
    uint3  tgpig,
    ushort tiisg,
    ushort sgitg) {

  // ── Tile/workload shape constants ──────────────────────────────────────
  //
  // Mirrors ggml-metal.metal:5795-5814.
  //   Q   — queries per threadgroup (NQPSG),     = 8   (llama.cpp-impl.h:93)
  //   C   — cache items per threadgroup (NCPSG), = 64  (llama.cpp-impl.h:94)
  //   DK  — K head size,                         = 512 (fixed in this kernel)
  //   DV  — V head size,                         = 512 (fixed in this kernel)
  //   NQ  — Q rows per simdgroup,                = Q / NSG
  //   NC  — Q·K^T 8×8 frags per simdgroup,       = (C/8) / NSG
  //   NO  — output 8×8 frags per simdgroup,      = (DV/8) / NSG
  //   SH  — softmax scratch halves per Q row,    = 2 × C
  //   NW  — simd width,                          = 32
  constexpr short Q  = 8;
  constexpr short C  = 64;
  constexpr short DK = 512;
  constexpr short DV = 512;

  constexpr short DK4  = DK / 4;
  constexpr short DK8  = DK / 8;
  // PV matches llama.cpp's PAD2(DV, 64) (ggml-metal.metal:5804); for DV=512
  // PV=512 so there is no actual padding.
  constexpr short PV   = DV;
  constexpr short PV4  = PV / 4;
  constexpr short PV8  = PV / 8;

  constexpr short NW  = 32;
  constexpr short NQ  = Q / NSG;                 // Q rows per simdgroup
  constexpr short SH  = 2 * C;                   // softmax+mask scratch
  // `T_STRIDE` is the llama.cpp name `T` (ggml-metal.metal:5814).  We rename
  // to avoid shadowing the template type parameter `T` below.  Halves per
  // Q-major region: `DK + 2*PV` (sq takes DK halves, so takes 2*PV halves,
  // ss takes 2*SH halves = 2*(2*C) = 2*SH < 2*PV for our geometry).
  constexpr short T_STRIDE = DK + 2 * PV;

  // Layout assertions (copied from llama.cpp ggml-metal.metal:6018, :6182).
  static_assert((C / 8) % NSG == 0, "NSG must divide C/8");
  static_assert(PV8     % NSG == 0, "NSG must divide PV8");
  static_assert(Q       % NSG == 0, "NSG must divide Q");

  // ── Threadgroup memory regions ─────────────────────────────────────────
  //
  // Layout mirrors ggml-metal.metal:5816-5828.  For (DK=DV=512, Q=8, C=64,
  // NSG=8, bf16):
  //   sq offset  0 bytes ,  size 8×512×2 =  8192 bytes
  //   so offset 8192       size 8×512×2 =  8192 bytes  (always half)
  //   ss offset 16384      size 8×128×2 =  2048 bytes
  //   Total:    26624 B
  //   Dispatcher requests 28672 B (matches llama.cpp FATTN_SMEM padding).
  threadgroup T*    sq  = (threadgroup T*)    (shmem_f16 + 0 * T_STRIDE);
  threadgroup half* so  = (threadgroup half*) (shmem_f16 + 0 * T_STRIDE + Q * DK);
  // Note: ss is float-typed (s_t == float in llama.cpp FA_TYPES_BF).  Row j
  // starts at `Q*T_STRIDE halves + 2*j*SH halves` = `Q*T_STRIDE halves +
  // j*(2*SH) halves`; in float units (ss's native) row j starts at
  // `j*SH floats`.  Row stride in halves = 2*SH = 256 (for SH=128).
  //
  // The ss region is laid out as Q contiguous rows of SH floats each
  // (= 2*SH halves each).  llama.cpp's `sm2` is a half2 alias over the
  // SECOND HALF of each ss row (scores live in cols [0, C) floats; mask
  // lives in cols [C, 2*C) floats of the same row; C floats = 2*C halves
  // = C half2s).  Rather than cast `sm2` with a `+2*C halves` base offset
  // (which would require its own cast-then-index), we hold `sm2` at ss-base
  // and add `C` to the index inside the row — arithmetically identical to
  // llama.cpp ggml-metal.metal:5830 `sm2 = shmem_f16 + Q*T + 2*C` with
  // `sm2[j*SH + tiisg]` re-expressed as `sm2_at_ss_base[j*SH + C + tiisg]`.
  threadgroup float*  ss  = (threadgroup float*) (shmem_f16 + Q * T_STRIDE);
  threadgroup float2* ss2 = (threadgroup float2*)(shmem_f16 + Q * T_STRIDE);
  threadgroup half2*  sm2 = (threadgroup half2*) (shmem_f16 + Q * T_STRIDE);

  // ── Per-threadgroup (batch, head, qL-tile) indices ────────────────────
  //
  // llama.cpp uses tgpig.x for qL-tile, .y for head, .z for batch
  // (ggml-metal.metal:5781-5783).  Our host grid has the same order:
  // grid = (NQ_tiles, H, B).
  const ushort iq3 = tgpig.z;                    // batch
  const ushort iq2 = tgpig.y;                    // head
  const ushort iq1 = tgpig.x * Q;                // first Q row index (this tile)

  // ── Base pointer offsetting (applies {batch, head, qL-tile} stride) ──
  //
  // Our Rust dispatcher lays out Q / K / V / O as [B, H, L, D] contiguous
  // with element strides (B, H, L, D=1).  llama.cpp uses byte strides
  // (args.nb01, .nb02, .nb03 etc.); we substitute our i64 element strides.
  // See ggml-metal.metal:5849-5856.

  const device T* q_typed = (const device T*)(q_base);
  const device T* k_typed = (const device T*)(k_base);
  const device T* v_typed = (const device T*)(v_base);
  device       T* o_typed = (device       T*)(o_base);

  // Q / O are per-query-head (H heads).  K / V are per-KV-head (H / gqa).
  const ulong kv_head = iq2 / args.gqa_factor;

  const device T* q_head = q_typed + (ulong)iq3 * (ulong)args.Q_strides[0]
                                    + (ulong)iq2 * (ulong)args.Q_strides[1];
  const device T* k_head = k_typed + (ulong)iq3 * (ulong)args.K_strides[0]
                                    +       kv_head * (ulong)args.K_strides[1];
  const device T* v_head = v_typed + (ulong)iq3 * (ulong)args.V_strides[0]
                                    +       kv_head * (ulong)args.V_strides[1];
  device       T* o_head = o_typed + (ulong)iq3 * (ulong)args.O_strides[0]
                                    + (ulong)iq2 * (ulong)args.O_strides[1];

  // K / V element-stride between consecutive KV items.  Layout is contiguous
  // in D, stride-D between items, stride-(kL*D) between heads — matching
  // llama.cpp's `args.nb11 / sizeof(k_t)` == NS10 at runtime (we pass this
  // via args.K_strides[2]).  Similarly for V (args.V_strides[2] = NS20).
  const int NS10 = int(args.K_strides[2]);
  const int NS20 = int(args.V_strides[2]);

  // ── Per-query mask pointers ────────────────────────────────────────────
  //
  // llama.cpp loads the additive mask for each Q row once per KV chunk
  // (ggml-metal.metal:5833-5839).  Our mask layout is [B, H, qL, kL] and
  // mask_args.M_strides = (batch, head, qL-row) with inner dim = 1
  // (kL stride is 1 — element-contiguous in kL).
  //
  // We hold one device-pointer per jj index into the [NQ] loop.  For our
  // additive/bool mask variants, MaskT differs (bf16/f16 vs bool); both
  // read at (iq1 + j)*kL + ic + tiisg — llama.cpp's `pm2[jj] += NW` pattern
  // is equivalent to advancing by 32 halves = 64 bytes per step, which is
  // exactly the simdgroup-wide mask-read slab used below.
  //
  // Unused when has_mask=false — dead-code-eliminated.
  device const MaskT* pm[NQ];
  if (has_mask) {
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
      const short j = jj * NSG + sgitg;
      // mask_args.M_strides: (batch, head, qL-row).  Additive mask uses
      // element indices; bool mask is byte-sized but kL-contiguous so the
      // same offset formula works.
      pm[jj] = mask_base + (ulong)iq3 * (ulong)mask_args.M_strides[0]
                         + (ulong)iq2 * (ulong)mask_args.M_strides[1]
                         + (ulong)(iq1 + j) * (ulong)mask_args.M_strides[2];
    }
  }

  // ── Load Q tile and zero O / SS ────────────────────────────────────────
  //
  // Direct port of ggml-metal.metal:5858-5884.  Each simdgroup loads its
  // NQ owned rows of the Q tile from device memory into `sq`, then zeros
  // its owned slots in `so` and `ss`.
  //
  // Scale Q here, not per-KV-iteration, so `scale * log2(e)` is paid once
  // per token rather than once per chunk.  This matches our D=256 kernel's
  // `TransformScale` model (flash_attn_prefill.metal:1242, 1291) and makes
  // the per-row `exp2(score - max)` idiom unchanged from D=256.
  //
  // sq uses element layout [Q][DK]; each row is DK elements of dtype T.
  // We write in 4-wide groups using float4 reads from device memory
  // (matches llama.cpp's q4 device-load at :5862).
  const T q_scale = (T)(args.scale * 1.44269504089f);  // scale * log2(e)

  FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;
    const bool j_in_range = (int)(iq1 + j) < args.qL;

    const device T* q_row =
        q_head + (ulong)(iq1 + j) * (ulong)args.Q_strides[2];

    // Each lane loads ceil(DK / NW) elements; DK=512 / 32 = 16 per lane.
    // Write into `sq[j*DK + i]` where i ∈ [0, DK).  Pre-scale on load so
    // the subsequent Q·K^T matmul is already in base-2 score space.
    for (short i = tiisg; i < DK; i += NW) {
      T val = j_in_range ? q_row[i] : T(0);
      sq[j * DK + i] = val * q_scale;
    }
  }

  FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;

    // Zero per-row O (in half, regardless of T — matches ggml-metal.metal:5818).
    for (short i = tiisg; i < DV; i += NW) {
      so[j * PV + i] = (half)0.0h;
    }

    // Zero per-row SS (softmax / mask scratch).
    for (short i = tiisg; i < SH; i += NW) {
      ss[j * SH + i] = 0.0f;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Per-simdgroup softmax state ────────────────────────────────────────
  //
  // Each simdgroup owns NQ scalar (M, S) pairs, one per Q row assigned to
  // it.  M initialised to the llama.cpp finite sentinel -FLT_MAX/2
  // (ggml-metal.metal:5891); S initialised to 0 (:5888).
  float S[NQ];
  float M[NQ];
  FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    S[jj] = 0.0f;
    M[jj] = -FLT_MAX / 2.0f;
  }

  // ── Resolve KV-chunk upper bound (causal truncation) ───────────────────
  //
  // llama.cpp's in-kernel causal mask runs on a per-element basis inside
  // the softmax step (:6132-6151 via the mask logic).  Our candle-derived
  // D=256 kernel hoists the causal cut-off to the outer loop via `kb_lim`
  // (flash_attn_prefill.metal:1313-1316) to skip whole tiles after the
  // diagonal.  We do the same here for consistency with our D=256 behaviour
  // and dispatcher contract (AttnParams::qL_off).
  int kL_chunks = (args.kL + C - 1) / C;   // total KV chunks (ceil)
  if (do_causal_def) {
    // Each Q row `iq1 + j` (j in [0, Q)) can attend to K positions up to
    // `args.qL_off + iq1 + j` inclusive.  For the threadgroup, the max
    // Q row is `iq1 + Q - 1`, so kL-limit in elements is
    // `args.qL_off + iq1 + Q`.  Divide ceil by C for chunks.
    int q_max = (int)iq1 + Q + args.qL_off;
    kL_chunks = (q_max + C - 1) / C;
    if (kL_chunks < 0) kL_chunks = 0;
  }

  // Clamp to valid range.
  if (kL_chunks * C > args.kL + C) {
    kL_chunks = (args.kL + C - 1) / C;
  }

  // ── Wave 2E tile-skip pre-pass row base (D=512) ────────────────────────
  //
  // The blk buffer shape is [NQ, NK] where NQ = ceil(qL/Q), NK = ceil(kL/C).
  // Layout matches the D=256 kernel: one 2D plane broadcast across batch
  // and heads.  Each threadgroup owns one Q-tile (tgpig.x); the row base
  // is `blk + tgpig.x * NK`.
  //
  // Port of llama.cpp ggml-metal.metal:5841-5846, adapted to our 2D mask
  // + single-plane blk layout.  See /opt/hf2q/docs/ADR-011-phase2-port-tile-skip.md §6.
  const device char* blk_row = nullptr;
  if (has_blk_def) {
    const int NK_blk = (args.kL + C - 1) / C;
    blk_row = blk_base + int(tgpig.x) * NK_blk;
  }

  // ── KV-cache sweep ─────────────────────────────────────────────────────
  //
  // Direct port of the ic0 loop at ggml-metal.metal:5907-6326.  Per chunk:
  //   1. Optionally load the additive mask slab (`sm2`).
  //   2. Compute Q·K^T into `ss` via NSG simdgroups × NC 8×8 frags each.
  //   3. Online softmax: update M, S per Q row; rescale existing `so`.
  //   4. Accumulate O += P·V into the threadgroup-half `so` via NSG
  //      simdgroups × NO output frags each.
  //
  // Simplifications vs llama.cpp (see file preamble):
  //   - is_q=0 only (direct simdgroup_load from K/V device memory).
  //   - No sinks / bias / softcap / kvpad / bc_mask.
  //   - blk_cur handled via Wave 2E has_blk pre-pass when enabled.
  for (int ic0 = 0; ic0 < kL_chunks; ++ic0) {
    const int ic = ic0 * C;

    // ── Wave 2E tile-skip branch (D=512) ────────────────────────────────
    //
    // Same three-way classification as the D=256 kernel; see that file's
    // comment for the numerics argument.  When has_blk is false blk_cur is
    // forced to 1 and the entire branch is dead-coded by the compiler.
    //
    // The `continue` here skips all simdgroup work: mask load, Q·K^T,
    // softmax update, and the P·V accumulate.  Because every simdgroup
    // in the threadgroup takes the same branch (blk_cur is a single byte
    // read, identical across lanes and simdgroups), the cross-simdgroup
    // threadgroup_barrier at the end of the previous iteration has
    // already synchronised state; no barrier is needed before `continue`.
    char blk_cur = 1;
    if (has_blk_def) {
      blk_cur = blk_row[ic0];
      if (blk_cur == 0) {
        continue;
      }
    }

    // ── Mask slab load (ggml-metal.metal:5954-5981, blk_cur==1 case) ─
    //
    // Each simdgroup loads its NQ owned rows × C mask columns into `sm2`.
    // For our dispatcher: MaskT ∈ {T, bool}.  The additive path reads T
    // (bf16/f16) and writes half-promoted into sm2.  The bool path reads
    // 1 byte per element and converts to 0.0 / -FLT_MAX/2 (the finite
    // sentinel; see the apply step below).
    //
    // Wave 2E: skip the mask load when blk_cur == 2 (all-attended tile —
    // the mask-add is a no-op and the sm2 region is not consumed below
    // because the `has_mask_def && blk_cur != 2` gate in the softmax step
    // matches this elision).  Port of llama.cpp ggml-metal.metal:6145.
    if (has_mask_def && blk_cur != 2) {
      constexpr bool is_bool_mask = is_same_v<MaskT, bool>;

      FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        // llama.cpp uses 2 halves per lane (half2) to cover 64 elements per
        // simdgroup (NW=32 × 2 = 64).  For C=64 that's exactly the mask
        // slab width.  We match that pattern: each lane writes 2 halves
        // into sm2[j*SH + lane].
        if (!is_bool_mask) {
          // Additive mask in I/O dtype — cast to half2 via two reads.
          // Mask stride for kL dim is 1 (contiguous).
          const int col0 = ic + 2 * (int)tiisg;
          const int col1 = col0 + 1;

          half v0 = (col0 < args.kL) ? (half)(float)(pm[jj][col0])
                                     : (half)(-FLT_MAX / 2.0f);
          half v1 = (col1 < args.kL) ? (half)(float)(pm[jj][col1])
                                     : (half)(-FLT_MAX / 2.0f);
          // Write into row j's mask sub-region: base ss + row-j-offset
          // + (C float2-slots past the score region) + tiisg.
          // See sm2 layout note at declaration.
          sm2[j * SH + C + tiisg] = half2(v0, v1);
        } else {
          // Boolean mask — false → -FLT_MAX/2 (finite sentinel),
          //                true  → 0.0 (additive identity).
          const int col0 = ic + 2 * (int)tiisg;
          const int col1 = col0 + 1;

          bool b0 = (col0 < args.kL) ? bool(pm[jj][col0]) : false;
          bool b1 = (col1 < args.kL) ? bool(pm[jj][col1]) : false;

          half v0 = b0 ? (half)0.0h : (half)(-FLT_MAX / 2.0f);
          half v1 = b1 ? (half)0.0h : (half)(-FLT_MAX / 2.0f);
          sm2[j * SH + C + tiisg] = half2(v0, v1);
        }
      }
    }

    // ── Q·K^T matmul (ggml-metal.metal:6009-6065, is_q=0 branch) ────────
    //
    // Each simdgroup owns NC = (C/8)/NSG of the 8×8 score frags along the
    // KV-column dimension.  For NSG=8, NC=1 (one 8×8 frag per simdgroup).
    // For NSG=4, NC=2.
    //
    // K is read directly from device memory via simdgroup_load with
    // transpose=true (K columns are contiguous, scores need K^T).  NS10
    // is the stride between consecutive KV items in elements (= D for our
    // contiguous layout).
    //
    // We write the accumulated frag into `ss + 8*cc + 8*NSG*sgitg` — each
    // simdgroup owns a disjoint 8-wide column of `ss` (ss layout:
    // [Q rows × SH columns], each row is SH halves wide = 128 halves for
    // Q=8, C=64).
    {
      constexpr short NC = (C / 8) / NSG;

      // Device pointer to the first K element for this chunk's K start,
      // offset to this simdgroup's starting KV-column group.
      const device T* pk = k_head + (ulong)ic * (ulong)NS10
                                  + (ulong)sgitg * (ulong)(8 * NS10);

      // Pointer into ss at this simdgroup's column origin (in halves/floats).
      threadgroup float* ps = ss + 8 * sgitg;

      FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
        // mqk accumulator — initialised to 0, unique per simdgroup.
        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

        // DK=512 → DK8=64, DK8/2=32 matmul iterations.  Mirrors
        // ggml-metal.metal:6040-6058 (DK%16==0 path).
        #pragma unroll(4)
        for (short i = 0; i < DK8 / 2; ++i) {
          simdgroup_barrier(mem_flags::mem_none);

          simdgroup_matrix<T, 8, 8> mq[2];
          simdgroup_matrix<T, 8, 8> mk[2];

          // Load 2 × (8×8) Q frags from sq at this simdgroup's Q-row
          // owned range.  llama.cpp:6048-6049 — `pq = sq` is the shared
          // Q tile; simdgroup_load at offset `8*i` along the DK axis.
          //
          // Because sq is shared across simdgroups (all simdgroups see the
          // same Q rows), we read from the full sq buffer.  Each 8×8 Q
          // frag spans 8 Q rows × 8 DK columns, and we load ALL 8 Q rows
          // per frag (NOT just NQ) because the output score frag mqk
          // occupies the same 8 Q rows × 8 KV-col slot in ss.  This means
          // each simdgroup redundantly computes scores for all 8 Q rows
          // but owns only its NC columns — matching llama.cpp's exact
          // pattern (single `pq` pointer, not per-simdgroup).
          simdgroup_load(mq[0], sq + 0 * 8 + 16 * i, DK);
          simdgroup_load(mq[1], sq + 1 * 8 + 16 * i, DK);

          simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, NS10, 0, true);
          simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, NS10, 0, true);

          simdgroup_barrier(mem_flags::mem_none);

          simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
          simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
        }

        // Store the 8×8 score frag into ss at this simdgroup's column
        // origin.  SH = 128 is the row stride of ss in floats.
        simdgroup_store(mqk, ps, SH, 0, false);

        // Advance pk and ps to the next simdgroup's KV-column group.
        // llama.cpp:6063-6064.
        pk += 8 * (NSG * NS10);
        ps += 8 * NSG;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Online softmax + rescale O (ggml-metal.metal:6131-6174) ──────────
    //
    // Per simdgroup, per owned Q row jj:
    //   1. Read score vector ss2[j*SH/2 + tiisg] (2 floats per lane,
    //      covering 64 columns = full C).  Score already includes Q's
    //      log2(e) pre-scale; apply the additive mask if present.
    //   2. Update M, S per Flash-Attention online softmax formula.
    //   3. Store P = exp(score - M) back into ss2 (replaces score).
    //   4. Rescale the existing `so` accumulator by `ms = exp(M_old - M_new)`.
    //
    // mqk is already ss; simdgroup_store wrote 8 rows × 8 cols per frag.
    // Each simdgroup participated in all Q rows (see note in QK matmul),
    // but each simdgroup owns a different NC slice of the KV columns.
    // At this barrier, ss holds scores for all 8 Q rows × full C columns.
    //
    // Our softmax uses exp2 + base-2 scores (matches D=256 kernel;
    // `flash_attn_prefill.metal:1458`); llama.cpp uses natural exp
    // (ggml-metal.metal:6155-6156).  Because Q was pre-scaled by
    // `scale * log2(e)`, the score space is base-2, and exp2 is the
    // correct inverse.
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
      const short j = jj * NSG + sgitg;

      const float m_old = M[jj];

      // Read 2 floats (one float2) from ss2 covering columns
      // [2*tiisg, 2*tiisg+1].  Full row read = NW lanes × float2 = 64 cols.
      float2 s2 = ss2[j * (SH / 2) + tiisg];

      // Apply additive mask — the mask in sm2 is already in base-2 scale
      // when it came in as additive T, because our dispatcher writes bf16/f16
      // masks in natural-log scale; we multiply by log2(e) here to bring
      // it into the base-2 score space.  This mirrors our D=256 kernel's
      // `Stile[jj] += 1.44269504089 * mfrag[jj]` at
      // flash_attn_prefill.metal:1428.
      //
      // Under the finite-sentinel regime, masked positions come in as
      // bf16::NEG_INFINITY, which in float promotion is -inf.  `-inf *
      // log2(e) = -inf` (IEEE-754), so s2 component becomes -inf — the
      // finite M absorbs it via simd_max without producing NaN.
      if (has_mask_def && blk_cur != 2) {
        half2 m2 = sm2[j * SH + C + tiisg];
        s2 = s2 + float2((float)m2.x, (float)m2.y) * 1.44269504089f;
      }

      // Causal masking — the qL-relative absolute position of row
      // (iq1 + j) inside the full qL is `args.qL_off + iq1 + j`, and the
      // KV columns at this iteration are ic + {2*tiisg, 2*tiisg+1}.
      // Positions where col > q_abs are masked out.
      if (do_causal_def) {
        int q_abs = args.qL_off + (int)iq1 + (int)j;
        int c0 = ic + 2 * (int)tiisg;
        int c1 = c0 + 1;
        if (c0 > q_abs) s2.x = -INFINITY;
        if (c1 > q_abs) s2.y = -INFINITY;
      }

      // Align_K=false trailing-chunk guard — mask out columns beyond kL.
      // Matches our D=256 kernel's `align_K && kb == NK_aligned` branch.
      if (!align_K_def && ic + C > args.kL) {
        int c0 = ic + 2 * (int)tiisg;
        int c1 = c0 + 1;
        if (c0 >= args.kL) s2.x = -INFINITY;
        if (c1 >= args.kL) s2.y = -INFINITY;
      }

      // Row max over this simdgroup's lanes covering all C columns.
      float my_max = max(s2.x, s2.y);
      float new_max = simd_max(max(m_old, my_max));

      M[jj] = new_max;

      // exp2(score - new_max) — base-2 because Q was pre-scaled by log2(e),
      // so `s2` and `new_max` are ALREADY in base-2 score space.  There is
      // NO additional log2(e) factor: multiplying by log2(e) here would
      // convert `exp2(x*log2(e))` = `e^x`, which is natural-exp over a
      // base-2 argument and effectively re-scales scores by log2(e)≈1.4427,
      // producing a sharper-than-correct softmax.  Matches the D=256
      // kernel's `ExpSubOp::apply` at
      // /opt/mlx-native/src/shaders/flash_attn_prefill.metal:1123
      //   (`fast::exp2(x - y)`, no log2(e) factor) and llama.cpp's
      // `const float2 vs2 = exp(s2 - M[jj])` at
      // /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6156
      // (both sides in natural-log space there; we're in base-2 with the
      // same net result because `exp2(base2) = exp(natural)` when the
      // pre-scale moves the Q·K^T into base-2 space).
      //
      // Unguarded: new_max is always finite under the sentinel regime
      // (simd_max of -FLT_MAX/2 and real scores), so exp2(-inf - finite)
      // = exp2(-inf) = +0.0 IEEE-754.
      float2 vs2;
      vs2.x = fast::exp2(s2.x - new_max);
      vs2.y = fast::exp2(s2.y - new_max);

      // Rescale factor exp2(M_old - M_new) — both in base-2 score space,
      // no additional log2(e) factor.  Matches the D=256 kernel's
      // `factor[i] = fast::exp2(max_score[i] - new_max[i])` at
      // /opt/mlx-native/src/shaders/flash_attn_prefill.metal:1546 and
      // llama.cpp's `const float ms = exp(m - M[jj])` at
      // /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6155.
      // Finite by construction.
      const float ms = fast::exp2(m_old - new_max);

      // Update S.
      S[jj] = S[jj] * ms + simd_sum(vs2.x + vs2.y);

      // Store P back into ss2 for consumption by the P·V matmul below.
      ss2[j * (SH / 2) + tiisg] = vs2;

      // Rescale O for this row by `ms`.  Only this simdgroup touches its
      // owned Q row, so no cross-simdgroup synchronisation needed here
      // (barrier at end of outer loop covers the so write-back).
      // DV4 = 128; 128 / NW = 4 iterations per lane.
      FOR_UNROLL (short ii = 0; ii < DV / NW; ++ii) {
        const short i = ii * NW + tiisg;
        so[j * PV + i] = (half)((float)so[j * PV + i] * ms);
      }
    }

    // NOTE: we intentionally do NOT correct s2.x / s2.y above for the
    // "score was reverted to base-2 by the pre-scaled Q" vs "log2(e) factor
    // on mask" mismatch — they are already consistent.  The score `s2` is
    // already in base-2 (Q was pre-scaled), and the mask-addition uses
    // log2(e) to promote the natural-log mask into the base-2 space.  This
    // matches our D=256 kernel's contract bit-for-bit.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── O += P · V (ggml-metal.metal:6179-6256, is_q=0 branch) ───────────
    //
    // Each simdgroup owns NO = PV8/NSG output frags along the DV axis.
    // Load per-simdgroup output frags from `so` into registers `lo[NO]`,
    // accumulate against (P × V) across all C/8 = 8 KV-cache frags, then
    // store back to `so`.
    //
    // For DV=512 > 64 we take llama.cpp's "wide-DV" branch at :6220-6245
    // which uses NO/2 inner iterations processing 2 × 8 = 16 DV-frags per
    // s-frag.  NO=8 at NSG=8, so NO/2=4 inner iterations.
    {
      constexpr short NO = PV8 / NSG;

      // Per-llama.cpp FA_TYPES_BF (ggml-metal.metal:6444-6451):
      //   o_t        = half              (storage)
      //   o8x8_t     = simdgroup_half8x8 (MMA accumulator type)
      //   s_t        = float, s8x8_t = simdgroup_float8x8
      //   v_t        = bfloat (== T in our bf16 path)
      //
      // Metal 3's `simdgroup_multiply_accumulate` supports mixed types for
      // A, B, C, D where the compiler chooses the widest accumulation path.
      // Using `simdgroup_half8x8` for lo mirrors llama.cpp's o_t choice.
      simdgroup_matrix<half, 8, 8> lo[NO];

      // Load this simdgroup's NO output frags from `so`.
      {
        threadgroup half* sot = so + 8 * sgitg;
        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
          simdgroup_load(lo[ii], sot, PV, 0, false);
          sot += 8 * NSG;
        }
      }

      // Walk all C/8 = 8 KV-col frag positions.  V is read directly from
      // device memory with per-simdgroup offset along the DV axis.
      //
      // For DV > 64 (our case): load 2 score frags per cc iteration
      // (vs[2]), run 4 8×8 matmuls per ii (2 V-frag loads × 2 score-frag
      // multiplies).  This is the :6220-6245 branch in llama.cpp.
      {
        const device T* pv = v_head + (ulong)ic * (ulong)NS20
                                    + (ulong)sgitg * 8;  // per-simdgroup DV offset

        constexpr short NC = (C / 8) / 2;  // 4

        FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
          simdgroup_matrix<float, 8, 8> vs[2];
          simdgroup_load(vs[0], ss + 16 * cc + 0, SH, 0, false);
          simdgroup_load(vs[1], ss + 16 * cc + 8, SH, 0, false);

          FOR_UNROLL (short ii = 0; ii < NO / 2; ++ii) {
            simdgroup_matrix<T, 8, 8> mv[4];

            simdgroup_load(mv[0], pv + 0 * NSG + 16 * ii * NSG + 0 * 8 * NS20, NS20, 0, false);
            simdgroup_load(mv[1], pv + 8 * NSG + 16 * ii * NSG + 0 * 8 * NS20, NS20, 0, false);
            simdgroup_load(mv[2], pv + 0 * NSG + 16 * ii * NSG + 1 * 8 * NS20, NS20, 0, false);
            simdgroup_load(mv[3], pv + 8 * NSG + 16 * ii * NSG + 1 * 8 * NS20, NS20, 0, false);

            simdgroup_multiply_accumulate(lo[2 * ii + 0], vs[0], mv[0], lo[2 * ii + 0]);
            simdgroup_multiply_accumulate(lo[2 * ii + 1], vs[0], mv[1], lo[2 * ii + 1]);
            simdgroup_multiply_accumulate(lo[2 * ii + 0], vs[1], mv[2], lo[2 * ii + 0]);
            simdgroup_multiply_accumulate(lo[2 * ii + 1], vs[1], mv[3], lo[2 * ii + 1]);
          }

          pv += 2 * 8 * NS20;
        }
      }

      // Store this simdgroup's output frags back to `so`.
      {
        threadgroup half* sot = so + 8 * sgitg;
        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
          simdgroup_store(lo[ii], sot, PV, 0, false);
          sot += 8 * NSG;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ── Final output write (ggml-metal.metal:6349-6371) ────────────────────
  //
  // Each simdgroup writes its NQ owned Q rows to device O, dividing by
  // S[jj] with the finite-sentinel guard: if S==0 (all masked), scale=0
  // so the output row is all-zero — matches llama.cpp:6358 and our D=256
  // DivOp.  Otherwise, scale = 1/S[jj].
  FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    const short j = jj * NSG + sgitg;

    if ((int)(iq1 + j) >= args.qL) {
      continue;  // past qL tail (align_Q=false case)
    }

    device T* o_row = o_head + (ulong)(iq1 + j) * (ulong)args.O_strides[2];

    const float scale = (S[jj] == 0.0f) ? 0.0f : 1.0f / S[jj];

    // DV4 = 128.  Each lane writes DV4 / NW = 4 elements.
    FOR_UNROLL (short ii = 0; ii < DV / NW; ++ii) {
      const short i = ii * NW + tiisg;
      o_row[i] = (T)((float)so[j * PV + i] * scale);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Outer kernel wrapper — dispatches to the NSG-specialised impl via the int
// function constant `fc_nsg`.  Mirrors llama.cpp's `kernel_flash_attn_ext`
// at ggml-metal.metal:6405-6430 (switch on FC_flash_attn_ext_nsg).
// ──────────────────────────────────────────────────────────────────────────

template <typename T, typename MaskT>
[[kernel, max_total_threads_per_threadgroup(32 * 8)]]
void flash_attn_prefill_d512(
    const device char* q_base    [[buffer(0)]],
    const device char* k_base    [[buffer(1)]],
    const device char* v_base    [[buffer(2)]],
    device       char* o_base    [[buffer(3)]],
    const constant AttnParams& args [[buffer(4)]],
    const constant AttnMaskParams& mask_args [[buffer(5), function_constant(has_mask)]],
    const device MaskT* mask_base [[buffer(6), function_constant(has_mask)]],
    const device char*  blk_base  [[buffer(7), function_constant(has_blk)]],
    threadgroup  half* shmem_f16 [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

  // ── NSG dispatch ──────────────────────────────────────────────────────
  //
  // The switch is resolved at pipeline-creation time via function-constant
  // specialisation, so only the selected branch is emitted in the compiled
  // pipeline.  Cases 1 and 2 are omitted (unused by our dispatcher; matches
  // llama.cpp's commented-out cases at :6423-6424).
  switch (nsg_def) {
    case 4:
      flash_attn_prefill_d512_impl<T, MaskT, 4>(
          q_base, k_base, v_base, o_base, args,
          mask_args, mask_base, blk_base, shmem_f16, tgpig, tiisg, sgitg);
      break;
    case 8:
      flash_attn_prefill_d512_impl<T, MaskT, 8>(
          q_base, k_base, v_base, o_base, args,
          mask_args, mask_base, blk_base, shmem_f16, tgpig, tiisg, sgitg);
      break;
    default:
      // No-op on unsupported NSG.  The dispatcher only ever sets 4 or 8.
      break;
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Host-visible instantiations
// ──────────────────────────────────────────────────────────────────────────
//
// Four entry points — (bf16/f16) × (additive-T mask / bool mask).  NSG is
// NOT in the entry-point name; it is specialised via the int function
// constant at pipeline-creation time.  Matches llama.cpp's design
// (ggml-metal.metal:6510-6511 — one entry per (dtype, DK, DV), NSG chosen
// at FC-time).

#define instantiate_d512(name, T, MaskT) \
  template [[host_name(name)]] [[kernel]] \
  decltype(flash_attn_prefill_d512<T, MaskT>) \
  flash_attn_prefill_d512<T, MaskT>;

instantiate_d512("flash_attn_prefill_llamacpp_bf16_d512",          bfloat, bfloat)
instantiate_d512("flash_attn_prefill_llamacpp_bf16_d512_boolmask", bfloat, bool)
instantiate_d512("flash_attn_prefill_llamacpp_f16_d512",           half,   half)
instantiate_d512("flash_attn_prefill_llamacpp_f16_d512_boolmask",  half,   bool)
