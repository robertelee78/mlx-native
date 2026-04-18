//! Flash-attention-style tiled prefill kernel — NSG=8 D=512 host dispatch.
//!
//! Companion to [`crate::ops::flash_attn_prefill`] (candle-derived D=256
//! kernel).  This module dispatches the **llama.cpp-derived D=512 kernel**
//! in `shaders/flash_attn_prefill_d512.metal` — a faithful port of
//! `kernel_flash_attn_ext_impl` from `llama.cpp/ggml/src/ggml-metal/
//! ggml-metal.metal:5736-6375` specialised for DK=DV=512 with NSG (simdgroups
//! per threadgroup) selectable as an int function constant.
//!
//! # Why this exists (not just an extension of `flash_attn_prefill.rs`)
//!
//! The D=256 kernel in `flash_attn_prefill.metal` uses a per-warp-Q-stacking
//! template that cannot host NSG=8 at D=512 within Apple Silicon's 32 KiB
//! threadgroup-memory budget (proof in `ADR-011-phase2-port-d512.md` §3.3):
//! the `BQ >= kNWarps × kFragSize` static_assert forces BQ≥64 at NSG=8, which
//! at BD=512 bf16 requires Q_smem of 66,560 bytes — over 2× the limit.
//!
//! The new kernel is architecturally different: Q rows are DISTRIBUTED
//! across simdgroups (NQ = Q/NSG rows per simdgroup), the output accumulator
//! `O` lives in threadgroup memory (not registers), and NSG=8 fits in
//! 28,672 bytes of threadgroup memory.  This is the same geometry llama.cpp
//! uses for DK=512 (`ggml-metal-ops.cpp:2807` — `int32_t nsg = ne00 >= 512
//! ? 8 : 4;`).
//!
//! # Kernel variants registered (four entry points)
//!
//! | Kernel name                                       | I/O dtype | Mask kind |
//! |---------------------------------------------------|-----------|-----------|
//! | `flash_attn_prefill_llamacpp_bf16_d512`           | bf16      | bf16 additive |
//! | `flash_attn_prefill_llamacpp_bf16_d512_boolmask`  | bf16      | bool (`is_attended`) |
//! | `flash_attn_prefill_llamacpp_f16_d512`            | f16       | f16 additive |
//! | `flash_attn_prefill_llamacpp_f16_d512_boolmask`   | f16       | bool |
//!
//! NSG is NOT in the entry-point name; it is specialised at pipeline-creation
//! time via the int function constant at index 322 (see below).
//!
//! # Function constants
//!
//! | Index | Name       | Kind | Purpose |
//! |-------|------------|------|---------|
//! | 200   | `align_Q`  | bool | true when qL % NQPSG == 0 |
//! | 201   | `align_K`  | bool | true when kL % NCPSG == 0 |
//! | 300   | `has_mask` | bool | true when a mask buffer is bound |
//! | 301   | `do_causal`| bool | true for in-kernel causal masking |
//! | **322** | **`fc_nsg`** | **int** | **NSG (simdgroups per threadgroup): 4 or 8** |
//!
//! Index 322 is new to this module and is plumbed via
//! [`KernelRegistry::get_pipeline_with_constants`] (the bool+int variant).
//! Matches llama.cpp's `FC_flash_attn_ext_nsg` at
//! `ggml-metal.metal:5735` = `FC_FLASH_ATTN_EXT + 22` = `300 + 22 = 322`.
//!
//! # Buffer layout (indices match the MSL kernel)
//!
//! Same as the D=256 dispatcher:
//! - `buffer(0)` — Q `[B, H,    qL, D]`  device, contiguous inner dim
//! - `buffer(1)` — K `[B, H_kv, kL, D]`  device, contiguous inner dim
//! - `buffer(2)` — V `[B, H_kv, kL, D]`  device, contiguous inner dim
//! - `buffer(3)` — O `[B, H,    qL, D]`  device, written by kernel
//! - `buffer(4)` — [`AttnParamsGpu`] constant buffer
//! - `buffer(5)` — [`AttnMaskParamsGpu`] constant buffer (only when `has_mask=true`)
//! - `buffer(6)` — mask data buffer (only when `has_mask=true`)
//!
//! # Grid geometry
//!
//! - Threadgroups: `(ceil(qL / NQPSG), H, B)` where `NQPSG = 8`
//! - Threads per threadgroup: `(32, NSG, 1)` where `NSG = 8` (or 4)
//! - NSG=8: 256 threads (8 simdgroups × 32 lanes).
//! - NSG=4: 128 threads (4 simdgroups × 32 lanes).
//!
//! Matches llama.cpp's grid at `ggml-metal-ops.cpp:2861` —
//! `((ne01+nqptg-1)/nqptg, ne02, ne03, 32, nsg, 1)`.
//!
//! # Threadgroup memory
//!
//! 28,672 bytes requested via `setThreadgroupMemoryLength` at encode time.
//! Matches llama.cpp's `FATTN_SMEM(nsg=8)` exactly for (DK=DV=512, NQPSG=8,
//! NCPSG=64, is_q=0, bf16).  Derivation: §2.3 of ADR-011-phase2-port-d512.md.
//! Our actual layout (sq + so + ss) uses 20,480 bytes; the extra 8,192 bytes
//! reserve the per-simdgroup K/V dequant scratch that llama.cpp uses for the
//! `is_q=1` path (we drop that branch for Gemma 4's bf16 K/V, but keep the
//! identical shmem footprint for like-for-like behaviour).
//!
//! # Scale convention
//!
//! Same as the D=256 kernel: pass `scale = 1.0 / sqrt(head_dim)`; the kernel
//! internally multiplies by `log2(e) ≈ 1.44269504` and uses `fast::exp2`
//! throughout.  Host MUST NOT pre-multiply by `log2(e)`.
//!
//! # Mask-sentinel contract
//!
//! Identical to D=256: additive masks use **masked positions = `-INFINITY`**
//! (llama.cpp CPU-side convention).  See the D=256 dispatcher module doc
//! for the full rationale; the kernel's finite-M-sentinel regime handles
//! `-inf` mask values without NaN propagation.
//!
//! # See also
//!
//! - Kernel: `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal`
//! - Port spec: `/opt/hf2q/docs/ADR-011-phase2-port-d512.md`
//! - llama.cpp reference: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6430`

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CapturedOpKind, CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

// Re-export the shared struct definitions so callers don't need to import
// both modules.  These are the SAME types used by the D=256 dispatcher.
pub use super::flash_attn_prefill::{
    AttnMaskParamsGpu, AttnParamsGpu, FlashAttnPrefillParams,
};

// ─── Shader source ───────────────────────────────────────────────────────────

/// MSL source for the NSG=8 D=512 llama.cpp-derived kernel.
pub static FLASH_ATTN_PREFILL_D512_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_prefill_d512.metal");

// ─── Kernel entry-point names ────────────────────────────────────────────────

/// D=512 NSG-specialised, bf16 I/O, bf16 additive mask.
pub const K_LLAMACPP_BF16_D512: &str = "flash_attn_prefill_llamacpp_bf16_d512";
/// D=512 NSG-specialised, bf16 I/O, bool (`is_attended`) mask.
pub const K_LLAMACPP_BF16_D512_BOOLMASK: &str =
    "flash_attn_prefill_llamacpp_bf16_d512_boolmask";
/// D=512 NSG-specialised, f16 I/O, f16 additive mask.
pub const K_LLAMACPP_F16_D512: &str = "flash_attn_prefill_llamacpp_f16_d512";
/// D=512 NSG-specialised, f16 I/O, bool mask.
pub const K_LLAMACPP_F16_D512_BOOLMASK: &str =
    "flash_attn_prefill_llamacpp_f16_d512_boolmask";

/// All four kernel entry-point names registered by this module.
pub const ALL_KERNEL_NAMES: &[&str] = &[
    K_LLAMACPP_BF16_D512,
    K_LLAMACPP_BF16_D512_BOOLMASK,
    K_LLAMACPP_F16_D512,
    K_LLAMACPP_F16_D512_BOOLMASK,
];

// ─── Registration ─────────────────────────────────────────────────────────────

/// Register all four D=512 kernel entry points against the new shader source.
///
/// Must be called before any dispatch to these kernels.  Safe to call alongside
/// [`crate::ops::flash_attn_prefill::register`] — the D=256 and D=512 kernels
/// live in separate `.metal` files and compile independently.
pub fn register(registry: &mut KernelRegistry) {
    for &name in ALL_KERNEL_NAMES {
        registry.register_source(name, FLASH_ATTN_PREFILL_D512_SHADER_SOURCE);
    }
}

// ─── Tile geometry constants (D=512 llama.cpp geometry) ─────────────────────

/// Queries per threadgroup.  Matches `OP_FLASH_ATTN_EXT_NQPSG`
/// (llama.cpp-impl.h:93).
pub const NQPSG_D512: u32 = 8;

/// Cache items per threadgroup.  Matches `OP_FLASH_ATTN_EXT_NCPSG`
/// (llama.cpp-impl.h:94).
pub const NCPSG_D512: u32 = 64;

/// Simdgroups per threadgroup for head_dim=512.  Matches
/// `ggml-metal-ops.cpp:2807` — `ne00 >= 512 ? 8 : 4`.
///
/// Exposed as a constant (not a parameter) because:
/// - NSG=8 is the only configuration that justifies this kernel's existence
///   (NSG=4 at D=512 would work but gives up the 2× parallelism win that
///   motivates porting).
/// - NSG=4 remains instantiable via [`dispatch_flash_attn_prefill_bf16_d512_with_nsg`]
///   below, for A/B benchmarking against NSG=8.
pub const NSG_D512: u32 = 8;

/// Int function-constant index for NSG.  Mirrors llama.cpp's
/// `FC_flash_attn_ext_nsg` at `ggml-metal.metal:5735` =
/// `FC_FLASH_ATTN_EXT + 22` = `300 + 22 = 322`.
pub const FC_IDX_NSG: usize = 322;

/// Threadgroup memory footprint at NSG=8, DK=DV=512, bf16, is_q=0.
/// See module doc + ADR-011-phase2-port-d512.md §2.3.
pub const TGMEM_BYTES_D512: u32 = 28_672;

// ─── Internal dispatcher body (shared by bf16 + f16 entry points) ────────────

fn validate_params_d512(params: &FlashAttnPrefillParams) -> Result<()> {
    if params.n_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_prefill_d512: n_heads must be > 0".into(),
        ));
    }
    if params.n_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_prefill_d512: n_kv_heads must be > 0".into(),
        ));
    }
    if params.n_heads % params.n_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_prefill_d512: n_heads ({}) must be divisible by n_kv_heads ({})",
            params.n_heads, params.n_kv_heads
        )));
    }
    if params.seq_len_q == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_prefill_d512: seq_len_q must be > 0".into(),
        ));
    }
    if params.seq_len_k == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_prefill_d512: seq_len_k must be > 0".into(),
        ));
    }
    if params.batch == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_prefill_d512: batch must be > 0".into(),
        ));
    }
    Ok(())
}

fn validate_buffer_size(buf: &MlxBuffer, name: &str, expected_elements: usize) -> Result<()> {
    let expected_bytes = expected_elements * buf.dtype().size_of();
    if buf.byte_len() < expected_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_prefill_d512: {name} buffer too small: expected at least \
             {expected_bytes} bytes, got {}",
            buf.byte_len()
        )));
    }
    Ok(())
}

// ─── bf16 D=512 dispatcher (NSG=8 default) ───────────────────────────────────

/// Dispatch llama.cpp-derived NSG=8 flash-attention prefill for bf16 Q/K/V/O,
/// head_dim=512.
///
/// Encodes a compute command into `encoder` without committing.  The caller
/// controls when to call `encoder.commit_and_wait()`.
///
/// # Buffer layouts
///
/// All buffers must be contiguous (stride-1 along the innermost / head_dim
/// dimension):
///
/// - `q`    — `[batch, n_heads,    seq_len_q, 512]`, dtype BF16
/// - `k`    — `[batch, n_kv_heads, seq_len_k, 512]`, dtype BF16
/// - `v`    — `[batch, n_kv_heads, seq_len_k, 512]`, dtype BF16
/// - `mask` — `[batch, n_heads, seq_len_q, seq_len_k]`, dtype BF16
///   (additive, log-scale: 0.0 = attend, -inf = mask out), or `None`
/// - `out`  — `[batch, n_heads,    seq_len_q, 512]`, dtype BF16
///
/// # Function constants
///
/// `align_Q`, `align_K` computed from sequence lengths and tile sizes.
/// `has_mask` reflects whether `mask` is `Some(_)`.
/// `do_causal` taken from `params.do_causal`.
/// `fc_nsg` = 8 (this dispatcher).
///
/// A distinct Metal pipeline is compiled per unique combination of
/// these five constants and cached in `registry`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for:
/// - `head_dim != 512`
/// - Zero or inconsistent shape fields
/// - Buffer too small for the declared shape
/// - `n_heads` not divisible by `n_kv_heads`
/// - Any buffer dtype != BF16
///
/// Returns `MlxError::ShaderCompilationError` if Metal pipeline compilation
/// fails — typically means the kernel source has a bug or the M-series
/// GPU lacks required capabilities (bfloat / simdgroup MMA).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_bf16_d512(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
) -> Result<()> {
    // Delegate to the blk-aware dispatcher with blk=None.  Because
    // `has_blk` is a function constant (index 303), the compiled pipeline
    // with has_blk=false dead-codes every blk reference — this call has
    // zero runtime overhead vs the pre-Wave-2E code path.
    dispatch_flash_attn_prefill_bf16_d512_with_blk(
        encoder, device, registry, q, k, v, mask, None, out, params,
    )
}

/// Dispatch the NSG=8 D=512 flash-attention prefill with an optional
/// Wave 2E tile-skip byte buffer.
///
/// See
/// [`crate::ops::flash_attn_prefill::dispatch_flash_attn_prefill_bf16_d256_with_blk`]
/// for the full contract; this function is its D=512 sibling.  The `blk`
/// buffer must be built with `(BQ=8, BK=8)` to match this kernel's tile
/// geometry.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_bf16_d512_with_blk(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    blk: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
) -> Result<()> {
    dispatch_flash_attn_prefill_bf16_d512_with_nsg_and_blk(
        encoder, device, registry, q, k, v, mask, blk, out, params, NSG_D512,
    )
}

/// Like [`dispatch_flash_attn_prefill_bf16_d512`] but with an explicit NSG
/// choice.  Must be 4 or 8.  Exposed for benchmarking NSG=4 vs NSG=8.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_bf16_d512_with_nsg(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
    nsg: u32,
) -> Result<()> {
    dispatch_flash_attn_prefill_bf16_d512_with_nsg_and_blk(
        encoder, device, registry, q, k, v, mask, None, out, params, nsg,
    )
}

/// Full D=512 dispatcher with explicit NSG and optional blk — the other
/// four D=512 dispatchers all delegate here with default arguments.  Exposed
/// so the Wave 2E integration tests can pin both NSG and blk.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_bf16_d512_with_nsg_and_blk(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    blk: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
    nsg: u32,
) -> Result<()>
{
    // ── Validate ──────────────────────────────────────────────────────────
    if params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_bf16_d512: head_dim must be 512, got {}",
            params.head_dim
        )));
    }
    if nsg != 4 && nsg != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_bf16_d512: nsg must be 4 or 8, got {nsg}"
        )));
    }
    if blk.is_some() && mask.is_none() {
        return Err(MlxError::InvalidArgument(
            "dispatch_flash_attn_prefill_bf16_d512: \
             blk requires mask (a blk without a mask is meaningless)"
                .into(),
        ));
    }
    validate_params_d512(params)?;

    // All buffers must be BF16 for this dispatcher.
    for (buf, name) in &[(q, "Q"), (k, "K"), (v, "V"), (out as &MlxBuffer, "out")] {
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_flash_attn_prefill_bf16_d512: {name} buffer must be BF16, \
                 got {:?}",
                buf.dtype()
            )));
        }
    }
    if let Some(m) = mask {
        if m.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_flash_attn_prefill_bf16_d512: mask buffer must be BF16, \
                 got {:?}",
                m.dtype()
            )));
        }
    }

    let batch = params.batch as usize;
    let h = params.n_heads as usize;
    let h_kv = params.n_kv_heads as usize;
    let ql = params.seq_len_q as usize;
    let kl = params.seq_len_k as usize;
    let d = params.head_dim as usize; // = 512

    // Validate buffer element counts.
    validate_buffer_size(q, "Q", batch * h * ql * d)?;
    validate_buffer_size(k, "K", batch * h_kv * kl * d)?;
    validate_buffer_size(v, "V", batch * h_kv * kl * d)?;
    validate_buffer_size(out, "out", batch * h * ql * d)?;
    // A rank-2 mask `[qL, kL]` is the Wave 2D broadcast layout: one plane is
    // shared across all batches and heads (stride-0 in batch and head dims).
    // A rank-4 mask `[B, H, qL, kL]` is the per-head layout (back-compat).
    let mask_is_rank2_broadcast = mask.is_some_and(|m| m.shape().len() == 2);
    if let Some(m) = mask {
        if mask_is_rank2_broadcast {
            validate_buffer_size(m, "mask", ql * kl)?;
        } else {
            validate_buffer_size(m, "mask", batch * h * ql * kl)?;
        }
    }

    // ── Tile geometry ─────────────────────────────────────────────────────
    let nqpsg = NQPSG_D512;
    let ncpsg = NCPSG_D512;

    let nq = params.seq_len_q.div_ceil(nqpsg);
    let nk = params.seq_len_k.div_ceil(ncpsg);
    let nq_aligned = params.seq_len_q / nqpsg;
    let nk_aligned = params.seq_len_k / ncpsg;
    let ql_rem = params.seq_len_q % nqpsg;
    let kl_rem = params.seq_len_k % ncpsg;

    // Function constants (specialised at pipeline creation time, not dispatch).
    let align_q = ql_rem == 0;
    let align_k = kl_rem == 0;
    let has_mask = mask.is_some();
    let has_blk = blk.is_some();
    let do_causal = params.do_causal;

    // Validate blk buffer size when present.  Tile shape is fixed for D=512:
    // BQ=NQPSG=8, BK=NCPSG=64 — NO: for the Wave 2E pre-pass paired with
    // D=512, we use (BQ=8, BK=8) NOT (8, 64).  Rationale: llama.cpp's
    // pre-pass ties its tile to the main kernel's block shape; our D=512
    // main kernel's mask-load slab is C=64 wide but it's consumed AT
    // (Q-row, KV-chunk) granularity — and the blk byte indexes
    // `blk[qt][kt]` where (qt, kt) correspond to the OUTER KV-chunk loop
    // (`ic0`) of size C=64.  However, the ADR specifies (8, 8) for D=512
    // because that's the granularity at which skip decisions are made at
    // the rows-per-simdgroup level.  For consistency with ADR §5.1 and
    // the port spec, we accept BK=8 paired blk buffers.  Actually wait —
    // re-reading, the ADR §5.1 specifies main-kernel BQ×BK = (BQ=32,
    // BK=16) for D=256.  For D=512 the main kernel's ic0 loop step is C
    // (=64), not BK=8.  So for D=512 the correct blk tile shape is
    // (BQ=8, BK=64).  Let's use that.
    let bq_main = 8_u32;
    let bk_main = 64_u32;  // == NCPSG_D512: the main-kernel ic0 loop steps by C=64.

    if let Some(b) = blk {
        let nq_tiles = ql.div_ceil(bq_main as usize);
        let nk_tiles = kl.div_ceil(bk_main as usize);
        let expected = nq_tiles * nk_tiles;
        if b.byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_flash_attn_prefill_bf16_d512: blk buffer too small: \
                 expected at least {expected} bytes (NQ={nq_tiles}, \
                 NK={nk_tiles}), got {}",
                b.byte_len()
            )));
        }
    }

    // ── Kernel name ───────────────────────────────────────────────────────
    // bf16 I/O; has_mask toggles between additive-bf16-mask and bool-mask
    // pipelines (different MaskT instantiations — must pick correct entry
    // point, not just a function constant).  When has_mask=false, the
    // additive-mask pipeline is compiled with has_mask=false, dead-code-
    // eliminating all mask accesses, and never binds buffers 5/6.
    let kernel_name = K_LLAMACPP_BF16_D512;  // additive bf16 (or disabled) mask path

    // ── Pipeline lookup (mixed bool+int function constants) ──────────────
    //
    // Wave 2E adds function constant 303 (has_blk).  When has_blk=false
    // every blk reference in the shader is dead-coded.
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[
            (200, align_q),
            (201, align_k),
            (300, has_mask),
            (301, do_causal),
            (303, has_blk),
        ],
        &[
            (FC_IDX_NSG, nsg as i32),
        ],
    )?;

    // ── Build AttnParams GPU struct ───────────────────────────────────────
    //
    // Strides for layout [B, H, L, D] where the innermost (D) stride is 1:
    //   seq stride   = D
    //   head stride  = L * D
    //   batch stride = H * L * D
    //
    // Q/O use (H, qL); K/V use (H_kv, kL).

    let q_seq_stride = d as i64;
    let q_head_stride = (ql * d) as i64;
    let q_batch_stride = (h * ql * d) as i64;

    let kv_seq_stride = d as i64;
    let kv_head_stride = (kl * d) as i64;
    let kv_batch_stride = (h_kv * kl * d) as i64;

    let gqa_factor = (params.n_heads / params.n_kv_heads) as i32;

    let attn_params = AttnParamsGpu {
        b: params.batch as i32,
        h: params.n_heads as i32,
        d: params.head_dim as i32,
        ql: params.seq_len_q as i32,
        kl: params.seq_len_k as i32,
        gqa_factor,
        scale: params.scale,
        softcapping: 1.0_f32, // always disabled; see AttnParamsGpu doc
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql_rem as i32,
        kl_rem: kl_rem as i32,
        ql_off: 0, // standard prefill starts at offset 0
        _pad: 0,
        q_strides: [q_batch_stride, q_head_stride, q_seq_stride],
        k_strides: [kv_batch_stride, kv_head_stride, kv_seq_stride],
        v_strides: [kv_batch_stride, kv_head_stride, kv_seq_stride],
        o_strides: [q_batch_stride, q_head_stride, q_seq_stride],
    };

    // ── Grid geometry ──────────────────────────────────────────────────────
    //
    // Matches llama.cpp ggml-metal-ops.cpp:2861:
    //   grid = ((ne01 + nqptg - 1)/nqptg, ne02, ne03)
    //   threads / TG = (32, nsg, 1)
    //
    // ne01 = qL, ne02 = H (query heads), ne03 = B.
    let grid = MTLSize::new(
        nq as u64,
        params.n_heads as u64,
        params.batch as u64,
    );
    let tg_size = MTLSize::new(32, nsg as u64, 1);

    // ── Encode ─────────────────────────────────────────────────────────────
    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Threadgroup memory — SAME footprint for NSG=4 and NSG=8 at is_q=0
    // (per ADR §2.3; the NSG term only shows up when is_q=1 allocates
    // per-simdgroup dequant scratch, which we've dropped).  Using the
    // llama.cpp FATTN_SMEM(nsg=8) value for like-for-like memory behaviour.
    let tgmem = TGMEM_BYTES_D512 as u64;

    if has_mask {
        let mask_buf = mask.ok_or_else(|| {
            MlxError::InvalidArgument(
                "flash_attn_prefill_d512: internal error — has_mask=true but mask is None".into(),
            )
        })?;

        // Strides depend on mask rank:
        //   rank-2 `[qL, kL]` — broadcast across batch + heads: set batch_stride
        //   and head_stride to 0 so the shader re-reads the same plane for every
        //   (batch, head) pair.  The Metal shader already handles stride-0 correctly.
        //   rank-4 `[B, H, qL, kL]` — per-head layout (back-compat path).
        let (m_batch_stride, m_head_stride, m_ql_stride) = if mask_is_rank2_broadcast {
            (0_i64, 0_i64, kl as i64)
        } else {
            ((h * ql * kl) as i64, (ql * kl) as i64, kl as i64)
        };

        let mask_params = AttnMaskParamsGpu {
            m_strides: [m_batch_stride, m_head_stride, m_ql_stride],
        };

        if has_blk {
            let blk_buf = blk.ok_or_else(|| {
                MlxError::InvalidArgument(
                    "flash_attn_prefill_d512: internal error — has_blk=true but blk is None".into(),
                )
            })?;

            encoder.encode_threadgroups_with_args_and_shared(
                pipeline,
                &[
                    (0, KernelArg::Buffer(q)),
                    (1, KernelArg::Buffer(k)),
                    (2, KernelArg::Buffer(v)),
                    (3, KernelArg::Buffer(out)),
                    (4, KernelArg::Bytes(as_bytes(&attn_params))),
                    (5, KernelArg::Bytes(as_bytes(&mask_params))),
                    (6, KernelArg::Buffer(mask_buf)),
                    (7, KernelArg::Buffer(blk_buf)),
                ],
                &[(0, tgmem)],
                grid,
                tg_size,
            );
        } else {
            encoder.encode_threadgroups_with_args_and_shared(
                pipeline,
                &[
                    (0, KernelArg::Buffer(q)),
                    (1, KernelArg::Buffer(k)),
                    (2, KernelArg::Buffer(v)),
                    (3, KernelArg::Buffer(out)),
                    (4, KernelArg::Bytes(as_bytes(&attn_params))),
                    (5, KernelArg::Bytes(as_bytes(&mask_params))),
                    (6, KernelArg::Buffer(mask_buf)),
                    // buffer 7 absent — has_blk=false dead-codes blk refs.
                ],
                &[(0, tgmem)],
                grid,
                tg_size,
            );
        }
    } else {
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Buffer(q)),
                (1, KernelArg::Buffer(k)),
                (2, KernelArg::Buffer(v)),
                (3, KernelArg::Buffer(out)),
                (4, KernelArg::Bytes(as_bytes(&attn_params))),
                // buffers 5, 6, 7 intentionally absent — has_mask=false +
                // has_blk=false dead-code-eliminates mask + blk loads.
            ],
            &[(0, tgmem)],
            grid,
            tg_size,
        );
    }

    Ok(())
}

// ─── Tests (structural only; GPU tests live in tests/test_flash_attn_prefill.rs) ─

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_geometry_d512() {
        assert_eq!(NQPSG_D512, 8, "NQPSG=8 for D=512 (llama.cpp-impl.h:93)");
        assert_eq!(NCPSG_D512, 64, "NCPSG=64 for D=512 (llama.cpp-impl.h:94)");
        assert_eq!(NSG_D512, 8, "NSG=8 for D=512 (llama.cpp-ops.cpp:2807)");
        // Threadgroup size: 32 × NSG = 256 threads at NSG=8.
        assert_eq!(32 * NSG_D512, 256);
    }

    #[test]
    fn test_threadgroup_memory_matches_llamacpp() {
        // FATTN_SMEM(nsg=8, DK=DV=512, bf16, is_q=0):
        //   inner = 8 * (512 + 2 * PAD(512, 64) + 2 * (2 * 64))
        //         = 8 * (512 + 1024 + 256)
        //         = 14_336 halves
        //   bytes = 14_336 * 2 = 28_672
        assert_eq!(TGMEM_BYTES_D512, 28_672);
    }

    #[test]
    fn test_fc_idx_nsg_matches_llamacpp() {
        // FC_FLASH_ATTN_EXT = 300, offset 22 → 322.
        assert_eq!(FC_IDX_NSG, 322);
    }

    #[test]
    fn test_four_kernel_names_registered() {
        assert_eq!(ALL_KERNEL_NAMES.len(), 4);
        let mut seen = std::collections::HashSet::new();
        for &name in ALL_KERNEL_NAMES {
            assert!(!name.is_empty());
            assert!(seen.insert(name), "duplicate name: {name}");
            assert!(
                name.starts_with("flash_attn_prefill_llamacpp_"),
                "name must be prefixed with llamacpp marker: {name}"
            );
            assert!(name.contains("d512"), "all D=512 names must contain d512: {name}");
        }
    }

    #[test]
    fn test_validate_params_d512_wrong_head_dim() {
        let p = FlashAttnPrefillParams {
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 256, // wrong — this dispatcher is D=512 only
            seq_len_q: 8,
            seq_len_k: 8,
            batch: 1,
            scale: 1.0,
            do_causal: false,
        };
        // Pre-GPU sanity: validate_params_d512 doesn't check head_dim (that
        // check lives in the dispatcher proper).  But it validates shape
        // fields.  head_dim=256 passes validate_params_d512 — the dispatcher
        // rejects it at its own guard.
        assert!(validate_params_d512(&p).is_ok());
    }

    #[test]
    fn test_validate_params_d512_ok() {
        let p = FlashAttnPrefillParams {
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 512,
            seq_len_q: 128,
            seq_len_k: 128,
            batch: 1,
            scale: 1.0 / 512.0_f32.sqrt(),
            do_causal: true,
        };
        assert!(validate_params_d512(&p).is_ok());
    }

    #[test]
    fn test_validate_params_d512_zero_heads() {
        let p = FlashAttnPrefillParams {
            n_heads: 0,
            n_kv_heads: 2,
            head_dim: 512,
            seq_len_q: 8,
            seq_len_k: 8,
            batch: 1,
            scale: 1.0,
            do_causal: false,
        };
        assert!(matches!(
            validate_params_d512(&p),
            Err(MlxError::InvalidArgument(_))
        ));
    }
}
