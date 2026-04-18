//! Flash-attention tile-skip pre-pass — host dispatch.
//!
//! Wave 2E of the ADR-011 Phase 2 port.  Pairs with Wave 2D
//! (`flash_attn_prefill_mask`) and Waves 2A/2C (main kernels at D=256 and
//! D=512 respectively).  Ported from llama.cpp's `kernel_flash_attn_ext_blk`
//! at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719` with
//! host dispatch at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2750-2820`.
//!
//! ## What this module does
//!
//! Given an additive bf16 attention mask (shape `[qL, kL]`, produced by
//! [`crate::ops::flash_attn_prefill_mask`]), dispatches a Metal kernel that
//! classifies each `(BQ, BK)` tile of the mask into one of three states:
//!
//! | byte | Meaning | Main-kernel action |
//! |------|---------|--------------------|
//! | 0    | All mask values ≤ -1e30 (fully masked)  | `continue` — skip the whole KV tile |
//! | 1    | Mixed — at least one finite value AND  | Normal mask-load + mask-add |
//! |      | at least one attended cell              |  |
//! | 2    | All cells == 0.0 (fully attended)       | Skip mask-add; compute Q·K^T + softmax normally |
//!
//! The output is a byte buffer of shape `[ceil(qL/BQ), ceil(kL/BK)]` — one
//! byte per tile — consumed by the main kernel via the `has_blk` function
//! constant and buffer index 7.  Exposes:
//!
//! - [`dispatch_flash_attn_prefill_blk`] — encode the pre-pass into an
//!   existing `CommandEncoder`.
//! - [`alloc_blk_buffer`] — allocate an appropriately sized byte buffer for
//!   the caller (scratch; caller holds it alive across the pre-pass → main
//!   kernel sequence).
//! - [`BlkParams`] — host-side parameter struct.
//!
//! ## Why two kernel instantiations (D=256 and D=512)
//!
//! The pre-pass tile geometry MUST match the main kernel's KV-tile loop
//! geometry, otherwise the `blk[qt][kt]` index does not correspond to the
//! correct tile.  Our two main kernels use different `(BQ, BK)` values:
//!
//! - D=256 main kernel: `(BQ=32, BK=16)` — 32 Q rows × 16 K cols per tile.
//! - D=512 main kernel: `(BQ=8,  BK=8)`  —  8 Q rows ×  8 K cols per tile.
//!
//! Rather than compile-time branch on the tile shape inside the shader,
//! the shader takes `(BQ, BK)` as function constants (indices 400 / 401)
//! and this module compiles two distinct pipelines — one per geometry —
//! cached by the kernel registry.
//!
//! See `ADR-011-phase2-port-tile-skip.md §5.1` for the full analysis of
//! the tile-shape choice.
//!
//! ## Sentinel convention (differs from llama.cpp)
//!
//! llama.cpp uses `-MAXHALF` (f16 ≈ -65504) as the "fully masked" threshold
//! at `ggml-metal.metal:5704`.  Our Wave 2D mask builder writes bit-exact
//! `-INFINITY` for blocked cells (`bf16 0xFF80`) and bit-exact `0.0` for
//! attended cells (`bf16 0x0000`).  The pre-pass classifier uses a
//! conservative `mmax <= -1e30f` threshold that catches both true `-inf`
//! and any finite "very negative" sentinel a future caller might pass —
//! see `ADR-011-phase2-port-tile-skip.md §5.2 Note 1`.
//!
//! ## Bit-exact correctness gate
//!
//! Running the main kernel with `blk = None` (pre-pass disabled) vs
//! `blk = Some(built_blk)` MUST produce byte-identical output — the blk
//! path is a skip optimisation, NEVER a correctness change.  This is
//! enforced by `test_gpu_bf16_d256_with_blk_matches_no_blk` and
//! `test_gpu_bf16_d512_with_blk_matches_no_blk` in
//! `tests/test_flash_attn_prefill.rs`.
//!
//! ## See also
//!
//! - Kernel: `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal`
//! - Port spec: `/opt/hf2q/docs/ADR-011-phase2-port-tile-skip.md`
//! - llama.cpp ref: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719`

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

// ─── Shader source ───────────────────────────────────────────────────────────

/// MSL source for the tile-skip pre-pass classifier (embedded at compile time).
pub static FLASH_ATTN_PREFILL_BLK_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_prefill_blk.metal");

/// Kernel entry point for the bf16 pre-pass classifier.
///
/// Tile geometry `(BQ, BK)` is specialised at pipeline-creation time via
/// function constants 400 / 401 — the same entry point backs both D=256
/// (BQ=32, BK=16) and D=512 (BQ=8, BK=8) pipelines.  The kernel registry
/// caches compiled pipelines keyed by the `(name, bq, bk)` combination.
pub const K_BLK_BF16: &str = "flash_attn_prefill_blk_bf16";

// ─── Function-constant indices ───────────────────────────────────────────────

/// Function-constant index for the Q-rows-per-tile constant in the pre-pass
/// kernel.  See `flash_attn_prefill_blk.metal:96`.
pub const FC_IDX_BQ: usize = 400;

/// Function-constant index for the K-cols-per-tile constant in the pre-pass
/// kernel.  See `flash_attn_prefill_blk.metal:97`.
pub const FC_IDX_BK: usize = 401;

/// Function-constant index for `has_blk` in the D=256 and D=512 main
/// kernels.  Re-exported here so main-kernel dispatchers don't need to
/// reach into this module's kernel source to know the index.
pub const FC_IDX_HAS_BLK: usize = 303;

// ─── Registration ─────────────────────────────────────────────────────────────

/// Register the pre-pass kernel entry point against the shader source.
///
/// Must be called before any dispatch to the pre-pass kernel.  Safe to
/// call alongside [`crate::ops::flash_attn_prefill::register`] and
/// [`crate::ops::flash_attn_prefill_mask::register`] — the three kernels
/// live in separate `.metal` files and compile independently.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(K_BLK_BF16, FLASH_ATTN_PREFILL_BLK_SHADER_SOURCE);
}

// ─── MSL struct mirrors ──────────────────────────────────────────────────────

/// Rust mirror of the MSL `FlashAttnPrefillBlkParams` struct.
///
/// MSL source:
/// `flash_attn_prefill_blk.metal:95` — see the `FlashAttnPrefillBlkParams`
/// struct definition for the field-by-field reference.
///
/// Total size: 16 bytes (4 × i32).  No padding needed — all fields are
/// 4-byte aligned and the struct ends on a 4-byte boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlkParamsGpu {
    /// Query sequence length (mask rows).
    seq_len_q: i32,
    /// Key sequence length (mask cols).
    seq_len_k: i32,
    /// Stride between consecutive mask rows, in ELEMENTS (bf16 units).  For
    /// the Wave 2D mask builder's contiguous `[qL, kL]` layout this equals
    /// `seq_len_k`.
    mask_row_stride: i32,
    /// Explicit padding.  Kept as a named field so `bytemuck::Pod` can be
    /// derived without complaint about uninitialised bytes.
    _pad: i32,
}

// ─── Public Rust-side parameter struct ───────────────────────────────────────

/// Host-side parameters for the tile-skip pre-pass dispatcher.
///
/// # Invariants (enforced by the dispatcher)
///
/// - `seq_len_q > 0`, `seq_len_k > 0`.
/// - `bq > 0`, `bk > 0`.  For D=256 use `(32, 16)`; for D=512 use `(8, 8)`.
/// - Mask buffer must be bf16 with at least `seq_len_q * seq_len_k` elements.
/// - `blk_out` buffer must be at least `ceil(qL/BQ) * ceil(kL/BK)` bytes.
#[derive(Debug, Clone, Copy)]
pub struct BlkParams {
    /// Query sequence length (rows of the mask).
    pub seq_len_q: u32,
    /// Key sequence length (cols of the mask).
    pub seq_len_k: u32,
    /// Q-rows per tile.  Must match the main kernel's BQ for the pipeline
    /// the caller intends to feed: 32 for D=256, 8 for D=512.
    pub bq: u32,
    /// K-cols per tile.  Must match the main kernel's BK: 16 for D=256,
    /// 8 for D=512.
    pub bk: u32,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Number of bytes required for the `blk` output buffer.
///
/// `ceil(qL / BQ) * ceil(kL / BK)` — one byte per tile.  Pad to 32 bytes
/// (Metal's minimum buffer alignment), matching llama.cpp's
/// `GGML_PAD(…, 32)` at `ggml-metal-ops.cpp:2591`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any input is zero or the
/// multiplication overflows `usize`.
pub fn blk_buffer_byte_len(params: &BlkParams) -> Result<usize> {
    if params.seq_len_q == 0 || params.seq_len_k == 0 {
        return Err(MlxError::InvalidArgument(
            "blk_buffer_byte_len: seq lengths must be > 0".into(),
        ));
    }
    if params.bq == 0 || params.bk == 0 {
        return Err(MlxError::InvalidArgument(
            "blk_buffer_byte_len: tile dimensions (bq, bk) must be > 0".into(),
        ));
    }
    let nq = params.seq_len_q.div_ceil(params.bq) as usize;
    let nk = params.seq_len_k.div_ceil(params.bk) as usize;
    let raw = nq.checked_mul(nk).ok_or_else(|| {
        MlxError::InvalidArgument(format!(
            "blk_buffer_byte_len: nq ({}) * nk ({}) overflows usize",
            nq, nk,
        ))
    })?;
    // 32-byte alignment mirrors llama.cpp's GGML_PAD(…, 32) and keeps the
    // Metal buffer on a friendly boundary for byte-granular writes.  At
    // minimum 32 bytes so even tiny (NQ=1, NK=1) masks get a full-aligned
    // buffer.
    let aligned = (raw + 31) & !31_usize;
    Ok(aligned.max(32))
}

/// Allocate a scratch byte buffer sized for the classification output.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for invalid parameters and
/// `MlxError::BufferAllocationError` if Metal cannot allocate.
pub fn alloc_blk_buffer(device: &MlxDevice, params: &BlkParams) -> Result<MlxBuffer> {
    let byte_len = blk_buffer_byte_len(params)?;
    // Declared dtype: U8 — a single byte per element.  shape: [NQ, NK].
    let nq = params.seq_len_q.div_ceil(params.bq) as usize;
    let nk = params.seq_len_k.div_ceil(params.bk) as usize;
    device.alloc_buffer(byte_len, DType::U8, vec![nq, nk])
}

// ─── Validation ───────────────────────────────────────────────────────────────

fn validate_params(params: &BlkParams) -> Result<()> {
    if params.seq_len_q == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_flash_attn_prefill_blk: seq_len_q must be > 0".into(),
        ));
    }
    if params.seq_len_k == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_flash_attn_prefill_blk: seq_len_k must be > 0".into(),
        ));
    }
    if params.bq == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_flash_attn_prefill_blk: bq must be > 0".into(),
        ));
    }
    if params.bk == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_flash_attn_prefill_blk: bk must be > 0".into(),
        ));
    }
    Ok(())
}

// ─── Dispatcher ──────────────────────────────────────────────────────────────

/// Dispatch the tile-skip pre-pass classifier.
///
/// Encodes a compute command into `encoder` without committing.  The caller
/// must sequence this dispatch BEFORE any main-kernel dispatch that consumes
/// the `blk_out` buffer, and MUST NOT mutate `mask` between this dispatch
/// and the subsequent main-kernel read.
///
/// # Buffer layout
///
/// - `mask`    — `[seq_len_q, seq_len_k]`, dtype BF16.  Produced by
///   [`crate::ops::flash_attn_prefill_mask::build_sdpa_mask_bf16`].  Must
///   be contiguous on the kL (innermost) dimension.
/// - `blk_out` — byte buffer, shape `[ceil(qL/BQ), ceil(kL/BK)]`.  Must be
///   at least [`blk_buffer_byte_len`] bytes.
///
/// # Grid geometry
///
/// - Threadgroups: `(NK, NQ, 1)` where `NQ = ceil(qL/BQ)`, `NK = ceil(kL/BK)`.
///   One threadgroup per (Q-tile, K-tile) pair.  Matches llama.cpp at
///   `ggml-metal-ops.cpp:2770`.
/// - Threads per threadgroup: `(32, 1, 1)` — one simdgroup.  Matches
///   llama.cpp's tile-classifier dispatch.
///
/// # Function constants
///
/// - `BQ_blk` (index 400) = `params.bq`
/// - `BK_blk` (index 401) = `params.bk`
///
/// A distinct pipeline is compiled per unique `(bq, bk)` combination and
/// cached in `registry`.  For Gemma 4 the two combinations used are
/// `(32, 16)` (D=256 sliding layers) and `(8, 8)` (D=512 global layers).
///
/// # Errors
///
/// - [`MlxError::InvalidArgument`] for invalid params (zero lengths or
///   tile dims, undersized buffers, wrong mask dtype).
/// - [`MlxError::ShaderCompilationError`] if pipeline compilation fails.
pub fn dispatch_flash_attn_prefill_blk(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    mask: &MlxBuffer,
    blk_out: &MlxBuffer,
    params: &BlkParams,
) -> Result<()> {
    validate_params(params)?;

    if mask.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_blk: mask buffer must be BF16, got {:?}",
            mask.dtype()
        )));
    }

    // Mask must be at least qL * kL elements (2 bytes each).
    let ql = params.seq_len_q as usize;
    let kl = params.seq_len_k as usize;
    let mask_bytes_needed = ql
        .checked_mul(kl)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "dispatch_flash_attn_prefill_blk: qL ({}) * kL ({}) * 2 overflows usize",
                ql, kl,
            ))
        })?;
    if mask.byte_len() < mask_bytes_needed {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_blk: mask buffer too small: \
             expected at least {mask_bytes_needed} bytes, got {}",
            mask.byte_len()
        )));
    }

    // blk_out must be at least nq * nk bytes.
    let nq = params.seq_len_q.div_ceil(params.bq) as usize;
    let nk = params.seq_len_k.div_ceil(params.bk) as usize;
    let blk_bytes_needed = nq.checked_mul(nk).ok_or_else(|| {
        MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_blk: nq ({}) * nk ({}) overflows usize",
            nq, nk,
        ))
    })?;
    if blk_out.byte_len() < blk_bytes_needed {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_blk: blk_out buffer too small: \
             expected at least {blk_bytes_needed} bytes, got {}",
            blk_out.byte_len()
        )));
    }

    // ── Build shader params ───────────────────────────────────────────────
    let gpu_params = BlkParamsGpu {
        seq_len_q: params.seq_len_q as i32,
        seq_len_k: params.seq_len_k as i32,
        mask_row_stride: params.seq_len_k as i32,
        _pad: 0,
    };

    // ── Pipeline lookup (specialised by (bq, bk)) ─────────────────────────
    //
    // Two-int specialisation — there are no bool function constants in the
    // pre-pass kernel, so the bool constants slice is empty and only the
    // two int constants (400 → bq, 401 → bk) drive the cache key.
    let pipeline = registry.get_pipeline_with_constants(
        K_BLK_BF16,
        device.metal_device(),
        &[],
        &[
            (FC_IDX_BQ, params.bq as i32),
            (FC_IDX_BK, params.bk as i32),
        ],
    )?;

    // ── Grid geometry ─────────────────────────────────────────────────────
    //   threadgroups: (NK, NQ, 1)   — one TG per (qtile, ktile) pair
    //   threads / TG: (32, 1, 1)    — one simdgroup
    //
    // Matches llama.cpp's grid at ggml-metal-ops.cpp:2770-2773 (adapted to
    // our single-plane mask — no batch / kv-group broadcast dim).
    let grid = MTLSize::new(nk as u64, nq as u64, 1);
    let tg_size = MTLSize::new(32, 1, 1);

    // ── Encode ────────────────────────────────────────────────────────────
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(mask)),
            (1, KernelArg::Buffer(blk_out)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg_size,
    );

    Ok(())
}

// ─── Tests (structural; GPU tests live in tests/test_flash_attn_prefill.rs) ──

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_blk_params_gpu_size() {
        // Four 4-byte fields, no padding needed → exactly 16 bytes.
        assert_eq!(std::mem::size_of::<BlkParamsGpu>(), 16);
    }

    #[test]
    fn test_fc_indices_match_shader() {
        // Indices 400 / 401 are declared in flash_attn_prefill_blk.metal;
        // 303 is declared in both flash_attn_prefill.metal (D=256) and
        // flash_attn_prefill_d512.metal (D=512).  Changing any of these
        // requires updating both the shader and this module in lockstep.
        assert_eq!(FC_IDX_BQ, 400);
        assert_eq!(FC_IDX_BK, 401);
        assert_eq!(FC_IDX_HAS_BLK, 303);
    }

    #[test]
    fn test_blk_buffer_byte_len_d256_gemma4() {
        // qL=kL=2455, BQ=32, BK=16 → NQ=77, NK=154, raw=77*154=11858, pad→11872.
        let p = BlkParams {
            seq_len_q: 2455,
            seq_len_k: 2455,
            bq: 32,
            bk: 16,
        };
        let bytes = blk_buffer_byte_len(&p).unwrap();
        assert!(bytes >= 11858, "must cover all 11858 tiles, got {bytes}");
        assert_eq!(bytes % 32, 0, "must be 32-byte aligned");
    }

    #[test]
    fn test_blk_buffer_byte_len_d512_gemma4() {
        // qL=kL=2455, BQ=8, BK=8 → NQ=NK=307, raw=307*307=94249, pad→94272.
        let p = BlkParams {
            seq_len_q: 2455,
            seq_len_k: 2455,
            bq: 8,
            bk: 8,
        };
        let bytes = blk_buffer_byte_len(&p).unwrap();
        assert!(bytes >= 94249, "must cover all 94249 tiles, got {bytes}");
        assert_eq!(bytes % 32, 0, "must be 32-byte aligned");
    }

    #[test]
    fn test_blk_buffer_byte_len_minimum() {
        // Tiny case: NQ=NK=1 → 1 byte raw → 32 bytes after alignment.
        let p = BlkParams {
            seq_len_q: 1,
            seq_len_k: 1,
            bq: 32,
            bk: 16,
        };
        assert_eq!(blk_buffer_byte_len(&p).unwrap(), 32);
    }

    #[test]
    fn test_blk_buffer_byte_len_zero_rejected() {
        assert!(blk_buffer_byte_len(&BlkParams {
            seq_len_q: 0,
            seq_len_k: 8,
            bq: 32,
            bk: 16,
        }).is_err());
        assert!(blk_buffer_byte_len(&BlkParams {
            seq_len_q: 8,
            seq_len_k: 0,
            bq: 32,
            bk: 16,
        }).is_err());
        assert!(blk_buffer_byte_len(&BlkParams {
            seq_len_q: 8,
            seq_len_k: 8,
            bq: 0,
            bk: 16,
        }).is_err());
    }

    #[test]
    fn test_validate_params_zero_rejected() {
        assert!(validate_params(&BlkParams {
            seq_len_q: 0,
            seq_len_k: 8,
            bq: 32,
            bk: 16,
        }).is_err());
    }

    #[test]
    fn test_kernel_name_stable() {
        // Regression gate: downstream modules import K_BLK_BF16; don't
        // change it without coordinating.
        assert_eq!(K_BLK_BF16, "flash_attn_prefill_blk_bf16");
    }

    #[test]
    fn test_register_does_not_panic() {
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        // Can't directly inspect the internal map; rely on the "no panic"
        // behaviour + the stable-name assert above.  GPU-side pipeline
        // compilation is exercised by tests/test_flash_attn_prefill.rs.
    }
}
