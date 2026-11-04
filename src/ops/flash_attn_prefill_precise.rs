//! Precise tiled warm attention (hf2q ADR-063 Q4-GEMMA-KERNEL).
//!
//! F16 Q and F16 K; the scores, softmax, P, the V operand, the O
//! accumulator and the stored O are all F32, and the attention scale is
//! applied to the F32 scores (never to an F16 Q tile).  Q is taken exactly
//! as given: the caller's single F32 → F16 rounding of the unscaled Q.
//!
//! Two dispatchers, one per head-dim family:
//!
//! | Dispatcher | Entry points | Use |
//! |---|---|---|
//! | [`dispatch_flash_attn_prefill_precise_d256`] | `flash_attn_prefill_precise_d256_{v32,v16,vcodes}` | sliding layers: external mask (+ optional `blk`), in-kernel window, or causal only |
//! | [`dispatch_flash_attn_prefill_precise_d512_resume`] | `flash_attn_prefill_precise_d512_{vcodes,v16,v32}` | global layers: causal resume reading K and V in place with a `kv_capacity` head stride |
//!
//! The V input form selects the entry point ([`PreciseV`]):
//! - `F32`: dense F32 V (`_v32`).  On d512 it is parity-only (the P9
//!   reference for `_vcodes`); serving never dispatches it.
//! - `F16`: an F16 V cache, widened exactly (`_v16`).
//! - `Codes`: TQ-HB V codes plus per-position F32 norms, dequantized in the
//!   kernel to values bitwise identical to
//!   [`crate::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq`].
//!
//! Exponentials: d256 multiplies the scores by `scale · log2(e)` (factor
//! formed in F32) and uses `exp2`; d512 multiplies by `scale` and uses `exp`.
//!
//! The existing F16/BF16 prefill entry points are unchanged.
//!
//! # Tile geometry and `blk`
//!
//! d256 uses BQ = 32, BK = 16, WM = 4 for every V form (`_v32` loads F32 V
//! fragments from device memory; `_vcodes` stages the codes in threadgroup
//! memory), so the `blk` classification is the (32, 16) one:
//! [`precise_d256_blk_params`] gives the classifier parameters, and the
//! dispatcher rejects a `blk` buffer whose shape is not
//! `[ceil(qL/32), ceil(kL/16)]` or whose length is below
//! [`blk_buffer_byte_len`].  d512 uses Q 8 × C 64 with NSG 8.
//!
//! # Registration
//!
//! The six entry points are in `KernelRegistry::new()`'s source map under
//! the shader file that defines them, so they resolve from the precompiled
//! metallib by default; no `register` call is needed.
//!
//! # Ordering
//!
//! The dispatchers encode one compute dispatch and declare nothing to a
//! `GraphSession`.  Both read K and an in-cache V (codes and norms, or F16)
//! in place with a `kv_capacity` head stride, so the caller must order the
//! dispatch after the writes of those rows (e.g.
//! `barrier_between(&[k, v_codes, v_norms, q], &[out])`).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::flash_attn_prefill::{
    AttnMaskParamsGpu, AttnParamsGpu, FlashAttnPrefillResumeParams,
};
use crate::ops::flash_attn_prefill_blk::{blk_buffer_byte_len, BlkParams};
use crate::DType;

/// D=256 precise entry point, dense F32 V.
pub const K_PRECISE_D256_V32: &str = "flash_attn_prefill_precise_d256_v32";
/// D=256 precise entry point, F16 V.
pub const K_PRECISE_D256_V16: &str = "flash_attn_prefill_precise_d256_v16";
/// D=256 precise entry point, TQ-HB V codes + norms.
pub const K_PRECISE_D256_VCODES: &str = "flash_attn_prefill_precise_d256_vcodes";
/// D=512 precise causal-resume entry point, TQ-HB V codes + norms in place.
pub const K_PRECISE_D512_VCODES: &str = "flash_attn_prefill_precise_d512_vcodes";
/// D=512 precise causal-resume entry point, F16 V cache in place.
pub const K_PRECISE_D512_V16: &str = "flash_attn_prefill_precise_d512_v16";
/// D=512 precise causal-resume entry point, dense F32 V (parity only).
pub const K_PRECISE_D512_V32: &str = "flash_attn_prefill_precise_d512_v32";

/// D=256 entry points (defined in `flash_attn_prefill.metal`).
pub const PRECISE_D256_KERNEL_NAMES: &[&str] = &[
    K_PRECISE_D256_V32,
    K_PRECISE_D256_V16,
    K_PRECISE_D256_VCODES,
];
/// D=512 entry points (defined in `flash_attn_prefill_d512.metal`).
pub const PRECISE_D512_KERNEL_NAMES: &[&str] = &[
    K_PRECISE_D512_VCODES,
    K_PRECISE_D512_V16,
    K_PRECISE_D512_V32,
];

/// D=256 query tile (rows per threadgroup) of every precise entry point.
pub const PRECISE_D256_BQ: u32 = 32;
/// D=256 key tile of every precise entry point.
pub const PRECISE_D256_BK: u32 = 16;
/// D=512 query rows per threadgroup.
const PRECISE_D512_Q: u32 = 8;
/// D=512 keys per chunk.
const PRECISE_D512_C: u32 = 64;
/// D=512 simdgroups per threadgroup.
const PRECISE_D512_NSG: u64 = 8;
/// D=512 threadgroup memory of the `_v16` / `_v32` forms.
const PRECISE_D512_TGMEM: u64 = 28_672;
/// Extra threadgroup memory of D=512 `_vcodes`: centroid table and one
/// chunk's scale_norms (`D512_PRECISE_CODES_EXTRA_BYTES` in the shader).
const PRECISE_D512_CODES_EXTRA_TGMEM: u64 = 256 * 4 + 64 * 2 * 4;

/// Function-constant index of the in-kernel window predicate.
const FC_HAS_WINDOW: usize = 310;

/// Rust mirror of the MSL `AttnPreciseParams` struct (buffer 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnPreciseParamsGpu {
    /// In-kernel sliding window W (only read with the window predicate).
    pub window: i32,
    /// TQ-HB V codebook width (5, 6 or 8; `_vcodes` only).
    pub codebook_bits: i32,
    /// D=512 TQ-HB norm divisor (`_vcodes` d512 only).
    pub scale_factor_d512: f32,
    /// TQ-HB norms per position (1 for D=256, 2 for D=512).
    pub norms_per_pos: i32,
    /// V norm strides in floats: (batch, kv head, position).
    pub n_strides: [i64; 3],
}

/// V input of a precise dispatch.
#[derive(Debug, Clone, Copy)]
pub enum PreciseV<'a> {
    /// Dense F32 V (`_v32`).
    F32(&'a MlxBuffer),
    /// F16 V (`_v16`).
    F16(&'a MlxBuffer),
    /// TQ-HB V (`_vcodes`): U8 codes laid out like K, F32 norms with
    /// `head_dim / 256` norms per position.
    Codes {
        codes: &'a MlxBuffer,
        norms: &'a MlxBuffer,
        codebook_bits: u32,
        scale_factor_d512: f32,
    },
}

/// The V input form, without buffers (pipeline selection and prewarm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreciseVForm {
    F32,
    F16,
    Codes,
}

impl PreciseV<'_> {
    /// The input form of this V.
    pub fn form(&self) -> PreciseVForm {
        match self {
            Self::F32(_) => PreciseVForm::F32,
            Self::F16(_) => PreciseVForm::F16,
            Self::Codes { .. } => PreciseVForm::Codes,
        }
    }
}

/// Key visibility of a d256 precise dispatch.
#[derive(Debug, Clone, Copy)]
pub enum PreciseD256Mask<'a> {
    /// External additive F16 mask (exactly 0 or −∞), `[qL, kL]` broadcast
    /// or `[B, H, qL, kL]`, with an optional `blk` built from the fully
    /// written mask with [`precise_d256_blk_params`].
    External {
        mask: &'a MlxBuffer,
        blk: Option<&'a MlxBuffer>,
    },
    /// In-kernel predicate: key `k` is visible to query row `i` iff
    /// `k <= q_offset + i` and `q_offset + i - k < window`.
    Window { window: u32 },
    /// In-kernel causal mask only: `k <= q_offset + i`.
    Causal,
}

/// d256 visibility without buffers (pipeline selection and prewarm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreciseD256MaskKind {
    External { blk: bool },
    Window,
    Causal,
}

impl PreciseD256Mask<'_> {
    /// The visibility kind of this mask.
    pub fn kind(&self) -> PreciseD256MaskKind {
        match self {
            Self::External { blk, .. } => PreciseD256MaskKind::External { blk: blk.is_some() },
            Self::Window { .. } => PreciseD256MaskKind::Window,
            Self::Causal => PreciseD256MaskKind::Causal,
        }
    }
}

/// Shape and scale of a d256 precise dispatch.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnPrecisePrefillParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// Must be 256.
    pub head_dim: u32,
    pub seq_len_q: u32,
    pub seq_len_k: u32,
    /// Rows per KV head of K and of an in-cache V (`F16`, `Codes`): the
    /// head stride, `>= seq_len_k`.  Keys `[0, seq_len_k)` are read in
    /// place; rows past `seq_len_k` are never read.  A dense F32 V always
    /// has `seq_len_k` rows per head.
    pub kv_capacity: u32,
    pub batch: u32,
    /// Finite and > 0; applied to the F32 scores.
    pub scale: f32,
    /// Absolute key position of query row 0 (causal and window masks).
    /// Ignored by [`PreciseD256Mask::External`].
    pub q_offset: u32,
}

/// `blk` classifier parameters matching the precise d256 tile.
pub fn precise_d256_blk_params(seq_len_q: u32, seq_len_k: u32) -> BlkParams {
    BlkParams {
        seq_len_q,
        seq_len_k,
        bq: PRECISE_D256_BQ,
        bk: PRECISE_D256_BK,
    }
}

fn invalid(what: &str, detail: String) -> MlxError {
    MlxError::InvalidArgument(format!("{what}: {detail}"))
}

fn check_dtype(what: &str, buf: &MlxBuffer, name: &str, dtype: DType) -> Result<()> {
    if buf.dtype() != dtype {
        return Err(invalid(
            what,
            format!("{name} must be {dtype}, got {}", buf.dtype()),
        ));
    }
    Ok(())
}

fn check_len(what: &str, buf: &MlxBuffer, name: &str, elements: usize) -> Result<()> {
    let bytes = elements * buf.dtype().size_of();
    if buf.byte_len() < bytes {
        return Err(invalid(
            what,
            format!(
                "{name} needs at least {bytes} bytes, got {}",
                buf.byte_len()
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_common(
    what: &str,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    expected_head_dim: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    batch: u32,
    scale: f32,
) -> Result<()> {
    if head_dim != expected_head_dim {
        return Err(invalid(
            what,
            format!("head_dim must be {expected_head_dim}, got {head_dim}"),
        ));
    }
    if n_heads == 0 || n_kv_heads == 0 || seq_len_q == 0 || seq_len_k == 0 || batch == 0 {
        return Err(invalid(
            what,
            "n_heads, n_kv_heads, seq_len_q, seq_len_k and batch must be >= 1".into(),
        ));
    }
    if n_heads % n_kv_heads != 0 {
        return Err(invalid(
            what,
            format!("n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"),
        ));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(invalid(
            what,
            format!("scale must be finite and > 0, got {scale}"),
        ));
    }
    Ok(())
}

/// Validate the V input and build the precise parameter block (norm
/// strides) for it.  K and an in-cache V (`F16`, `Codes`) have `cap` rows
/// per KV head; a dense F32 V has `kl`.  Returns the block and the V row
/// count per head.
fn check_v(
    what: &str,
    v: &PreciseV<'_>,
    batch: usize,
    h_kv: usize,
    (cap, kl): (usize, usize),
    d: usize,
) -> Result<(AttnPreciseParamsGpu, usize)> {
    let rows = if matches!(v, PreciseV::F32(_)) {
        kl
    } else {
        cap
    };
    let elements = batch * h_kv * rows * d;
    let npp = d / 256;
    let mut pp = AttnPreciseParamsGpu {
        window: 0,
        codebook_bits: 0,
        scale_factor_d512: 1.0,
        norms_per_pos: npp as i32,
        n_strides: [(h_kv * rows * npp) as i64, (rows * npp) as i64, npp as i64],
    };
    match *v {
        PreciseV::F32(buf) => {
            check_dtype(what, buf, "V", DType::F32)?;
            check_len(what, buf, "V", elements)?;
        }
        PreciseV::F16(buf) => {
            check_dtype(what, buf, "V", DType::F16)?;
            check_len(what, buf, "V", elements)?;
        }
        PreciseV::Codes {
            codes,
            norms,
            codebook_bits,
            scale_factor_d512,
        } => {
            check_dtype(what, codes, "V codes", DType::U8)?;
            check_dtype(what, norms, "V norms", DType::F32)?;
            check_len(what, codes, "V codes", elements)?;
            check_len(what, norms, "V norms", batch * h_kv * rows * npp)?;
            if !matches!(codebook_bits, 5 | 6 | 8) {
                return Err(invalid(
                    what,
                    format!("codebook_bits must be 5, 6 or 8, got {codebook_bits}"),
                ));
            }
            if d == 512 && !(scale_factor_d512.is_finite() && scale_factor_d512 > 0.0) {
                return Err(invalid(
                    what,
                    format!("scale_factor_d512 must be finite and > 0, got {scale_factor_d512}"),
                ));
            }
            pp.codebook_bits = codebook_bits as i32;
            pp.scale_factor_d512 = scale_factor_d512;
        }
    }
    Ok((pp, rows))
}

fn v_bindings<'a>(v: &PreciseV<'a>) -> (KernelArg<'a>, Option<KernelArg<'a>>) {
    match *v {
        PreciseV::F32(buf) | PreciseV::F16(buf) => (KernelArg::Buffer(buf), None),
        PreciseV::Codes { codes, norms, .. } => {
            (KernelArg::Buffer(codes), Some(KernelArg::Buffer(norms)))
        }
    }
}

fn d256_kernel(form: PreciseVForm) -> &'static str {
    match form {
        PreciseVForm::F32 => K_PRECISE_D256_V32,
        PreciseVForm::F16 => K_PRECISE_D256_V16,
        PreciseVForm::Codes => K_PRECISE_D256_VCODES,
    }
}

fn d512_kernel(form: PreciseVForm) -> &'static str {
    match form {
        PreciseVForm::F32 => K_PRECISE_D512_V32,
        PreciseVForm::F16 => K_PRECISE_D512_V16,
        PreciseVForm::Codes => K_PRECISE_D512_VCODES,
    }
}

/// Function constants of a d256 precise specialization:
/// align_Q, align_K, has_mask, do_causal, has_blk, has_window.
fn d256_constants(align_q: bool, align_k: bool, kind: PreciseD256MaskKind) -> [(usize, bool); 6] {
    let (has_mask, has_blk, do_causal, has_window) = match kind {
        PreciseD256MaskKind::External { blk } => (true, blk, false, false),
        PreciseD256MaskKind::Window => (false, false, true, true),
        PreciseD256MaskKind::Causal => (false, false, true, false),
    };
    [
        (200, align_q),
        (201, align_k),
        (300, has_mask),
        (301, do_causal),
        (303, has_blk),
        (FC_HAS_WINDOW, has_window),
    ]
}

/// Function constants of a d512 precise specialization.
fn d512_constants(align_q: bool, align_k: bool, do_causal: bool) -> [(usize, bool); 5] {
    [
        (200, align_q),
        (201, align_k),
        (300, false),
        (301, do_causal),
        (303, false),
    ]
}

/// Pipeline label (`KernelRegistry` cache key) of a specialization.
fn pipeline_label(name: &str, constants: &[(usize, bool)]) -> String {
    let mut label = name.to_string();
    for &(index, value) in constants {
        label.push_str(&format!("|{index}:{}", if value { "b1" } else { "b0" }));
    }
    label
}

/// Dispatch precise tiled attention for D=256.
///
/// Layouts (contiguous, head-major): Q F16 `[B, H, qL, 256]`; K F16
/// `[B, H_kv, kv_capacity, 256]` with keys `[0, kL)` linear, read in place;
/// V per [`PreciseV`]: an in-cache V (`F16`, `Codes` with norms
/// `[B, H_kv, kv_capacity]`) like K, a dense F32 V `[B, H_kv, kL, 256]`;
/// O F32 `[B, H, qL, 256]`.  Rows `[kL, kv_capacity)` are never read.
/// Every O element of rows `< qL` is written and nothing beyond.
///
/// # Errors
///
/// `InvalidArgument` for a head_dim other than 256, empty shapes, a GQA
/// mismatch, a non-finite or non-positive scale, wrong dtypes, undersized
/// buffers, `kL > kv_capacity`, `q_offset + qL > kL` with an in-kernel
/// mask, a zero window, codebook bits outside {5, 6, 8}, or a `blk` that
/// does not match the (32, 16) tile.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_precise_d256(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: PreciseV<'_>,
    mask: PreciseD256Mask<'_>,
    out: &MlxBuffer,
    params: &FlashAttnPrecisePrefillParams,
) -> Result<()> {
    const WHAT: &str = "dispatch_flash_attn_prefill_precise_d256";
    let p = params;
    check_common(
        WHAT,
        p.n_heads,
        p.n_kv_heads,
        p.head_dim,
        256,
        p.seq_len_q,
        p.seq_len_k,
        p.batch,
        p.scale,
    )?;
    let (b, h, h_kv) = (p.batch as usize, p.n_heads as usize, p.n_kv_heads as usize);
    let (ql, kl, d) = (p.seq_len_q as usize, p.seq_len_k as usize, 256_usize);
    let cap = p.kv_capacity as usize;
    if cap < kl {
        return Err(invalid(
            WHAT,
            format!("seq_len_k ({kl}) > kv_capacity ({cap})"),
        ));
    }
    check_dtype(WHAT, q, "Q", DType::F16)?;
    check_dtype(WHAT, k, "K", DType::F16)?;
    check_dtype(WHAT, out, "out", DType::F32)?;
    check_len(WHAT, q, "Q", b * h * ql * d)?;
    check_len(WHAT, k, "K", b * h_kv * cap * d)?;
    check_len(WHAT, out, "out", b * h * ql * d)?;
    let (mut pp, v_rows) = check_v(WHAT, &v, b, h_kv, (cap, kl), d)?;

    let mut mask_strides = None;
    match mask {
        PreciseD256Mask::External { mask, blk } => {
            check_dtype(WHAT, mask, "mask", DType::F16)?;
            let rank2 = mask.shape().len() == 2;
            check_len(
                WHAT,
                mask,
                "mask",
                if rank2 { ql * kl } else { b * h * ql * kl },
            )?;
            mask_strides = Some(if rank2 {
                [0, 0, kl as i64]
            } else {
                [(h * ql * kl) as i64, (ql * kl) as i64, kl as i64]
            });
            if let Some(blk) = blk {
                let blk_params = precise_d256_blk_params(p.seq_len_q, p.seq_len_k);
                let tiles = vec![ql.div_ceil(32), kl.div_ceil(16)];
                if blk.shape() != tiles.as_slice() {
                    return Err(invalid(
                        WHAT,
                        format!(
                            "blk shape {:?} does not match the precise d256 tile grid {tiles:?} \
                             (classifier bq/bk must be {PRECISE_D256_BQ}/{PRECISE_D256_BK})",
                            blk.shape()
                        ),
                    ));
                }
                let needed = blk_buffer_byte_len(&blk_params)?;
                if blk.byte_len() < needed {
                    return Err(invalid(
                        WHAT,
                        format!("blk needs {needed} bytes, got {}", blk.byte_len()),
                    ));
                }
            }
        }
        PreciseD256Mask::Window { window } => {
            if window == 0 {
                return Err(invalid(WHAT, "window must be >= 1".into()));
            }
            pp.window = window as i32;
        }
        PreciseD256Mask::Causal => {}
    }
    if !matches!(mask, PreciseD256Mask::External { .. })
        && p.q_offset as u64 + ql as u64 > kl as u64
    {
        return Err(invalid(
            WHAT,
            format!(
                "q_offset ({}) + seq_len_q ({ql}) > seq_len_k ({kl})",
                p.q_offset
            ),
        ));
    }

    let (bq, bk) = (PRECISE_D256_BQ, PRECISE_D256_BK);
    let constants = d256_constants(p.seq_len_q % bq == 0, p.seq_len_k % bk == 0, mask.kind());
    let pipeline = registry.get_pipeline_with_bool_constants(
        d256_kernel(v.form()),
        device.metal_device(),
        &constants,
    )?;

    let q_strides = [(h * ql * d) as i64, (ql * d) as i64, d as i64];
    let k_strides = [(h_kv * cap * d) as i64, (cap * d) as i64, d as i64];
    let v_strides = [(h_kv * v_rows * d) as i64, (v_rows * d) as i64, d as i64];
    let attn = AttnParamsGpu {
        b: p.batch as i32,
        h: p.n_heads as i32,
        d: 256,
        ql: p.seq_len_q as i32,
        kl: p.seq_len_k as i32,
        gqa_factor: (p.n_heads / p.n_kv_heads) as i32,
        scale: p.scale,
        softcapping: 1.0,
        nq: p.seq_len_q.div_ceil(bq) as i32,
        nk: p.seq_len_k.div_ceil(bk) as i32,
        nq_aligned: (p.seq_len_q / bq) as i32,
        nk_aligned: (p.seq_len_k / bk) as i32,
        ql_rem: (p.seq_len_q % bq) as i32,
        kl_rem: (p.seq_len_k % bk) as i32,
        ql_off: if mask_strides.is_some() {
            0
        } else {
            p.q_offset as i32
        },
        _pad: 0,
        q_strides,
        k_strides,
        v_strides,
        o_strides: q_strides,
    };
    let mask_params = mask_strides.map(|m_strides| AttnMaskParamsGpu { m_strides });
    let (v_arg, norms_arg) = v_bindings(&v);

    let mut args: Vec<(u64, KernelArg<'_>)> = vec![
        (0, KernelArg::Buffer(q)),
        (1, KernelArg::Buffer(k)),
        (2, v_arg),
        (3, KernelArg::Buffer(out)),
        (4, KernelArg::Bytes(as_bytes(&attn))),
        (9, KernelArg::Bytes(as_bytes(&pp))),
    ];
    if let (PreciseD256Mask::External { mask, blk }, Some(mp)) = (mask, mask_params.as_ref()) {
        args.push((5, KernelArg::Bytes(as_bytes(mp))));
        args.push((6, KernelArg::Buffer(mask)));
        if let Some(blk) = blk {
            args.push((7, KernelArg::Buffer(blk)));
        }
    }
    if let Some(norms) = norms_arg {
        args.push((10, norms));
    }

    let grid = MTLSize::new(
        p.seq_len_q.div_ceil(bq) as u64,
        p.n_heads as u64,
        p.batch as u64,
    );
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(pipeline, &args, grid, MTLSize::new(32, 4, 1));
    Ok(())
}

/// Dispatch the precise D=512 causal-resume attention, reading K and V in
/// place from the cache.
///
/// Layouts: Q F16 `[B, H, qL, 512]`; K F16 `[B, H_kv, kv_capacity, 512]`
/// with keys `[0, kL)` linear; an in-cache V (`F16`, `Codes` with norms
/// `[B, H_kv, kv_capacity, 2]`) like K; a dense F32 V `[B, H_kv, kL, 512]`
/// (parity-only); O F32 `[B, H, qL, 512]`.  Rows `[kL, kv_capacity)` are
/// never read.  No K or V scratch is allocated.
///
/// # Errors
///
/// `InvalidArgument` for a head_dim other than 512, empty shapes, a GQA
/// mismatch, a non-finite or non-positive scale, wrong dtypes, undersized
/// buffers, `q_offset_in_k + qL > kL`, `kL > kv_capacity`, codebook bits
/// outside {5, 6, 8} or a non-positive `scale_factor_d512`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_precise_d512_resume(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: PreciseV<'_>,
    out: &MlxBuffer,
    params: &FlashAttnPrefillResumeParams,
) -> Result<()> {
    const WHAT: &str = "dispatch_flash_attn_prefill_precise_d512_resume";
    let p = params;
    check_common(
        WHAT,
        p.n_heads,
        p.n_kv_heads,
        p.head_dim,
        512,
        p.seq_len_q,
        p.seq_len_k,
        p.batch,
        p.scale,
    )?;
    if p.q_offset_in_k as u64 + p.seq_len_q as u64 > p.seq_len_k as u64 {
        return Err(invalid(
            WHAT,
            format!(
                "q_offset_in_k ({}) + seq_len_q ({}) > seq_len_k ({})",
                p.q_offset_in_k, p.seq_len_q, p.seq_len_k
            ),
        ));
    }
    if p.seq_len_k > p.kv_capacity {
        return Err(invalid(
            WHAT,
            format!(
                "seq_len_k ({}) > kv_capacity ({})",
                p.seq_len_k, p.kv_capacity
            ),
        ));
    }
    let (b, h, h_kv) = (p.batch as usize, p.n_heads as usize, p.n_kv_heads as usize);
    let (ql, kl, cap, d) = (
        p.seq_len_q as usize,
        p.seq_len_k as usize,
        p.kv_capacity as usize,
        512_usize,
    );
    check_dtype(WHAT, q, "Q", DType::F16)?;
    check_dtype(WHAT, k, "K", DType::F16)?;
    check_dtype(WHAT, out, "out", DType::F32)?;
    check_len(WHAT, q, "Q", b * h * ql * d)?;
    check_len(WHAT, k, "K", b * h_kv * cap * d)?;
    check_len(WHAT, out, "out", b * h * ql * d)?;
    let (pp, v_rows) = check_v(WHAT, &v, b, h_kv, (cap, kl), d)?;

    let (nq, nc) = (PRECISE_D512_Q, PRECISE_D512_C);
    let constants = d512_constants(p.seq_len_q % nq == 0, p.seq_len_k % nc == 0, p.do_causal);
    let pipeline = registry.get_pipeline_with_bool_constants(
        d512_kernel(v.form()),
        device.metal_device(),
        &constants,
    )?;

    let q_strides = [(h * ql * d) as i64, (ql * d) as i64, d as i64];
    let k_strides = [(h_kv * cap * d) as i64, (cap * d) as i64, d as i64];
    let v_strides = [(h_kv * v_rows * d) as i64, (v_rows * d) as i64, d as i64];
    let attn = AttnParamsGpu {
        b: p.batch as i32,
        h: p.n_heads as i32,
        d: 512,
        ql: p.seq_len_q as i32,
        kl: p.seq_len_k as i32,
        gqa_factor: (p.n_heads / p.n_kv_heads) as i32,
        scale: p.scale,
        softcapping: 1.0,
        nq: p.seq_len_q.div_ceil(nq) as i32,
        nk: p.seq_len_k.div_ceil(nc) as i32,
        nq_aligned: (p.seq_len_q / nq) as i32,
        nk_aligned: (p.seq_len_k / nc) as i32,
        ql_rem: (p.seq_len_q % nq) as i32,
        kl_rem: (p.seq_len_k % nc) as i32,
        ql_off: p.q_offset_in_k as i32,
        _pad: 0,
        q_strides,
        k_strides,
        v_strides,
        o_strides: q_strides,
    };
    let (v_arg, norms_arg) = v_bindings(&v);
    let mut args: Vec<(u64, KernelArg<'_>)> = vec![
        (0, KernelArg::Buffer(q)),
        (1, KernelArg::Buffer(k)),
        (2, v_arg),
        (3, KernelArg::Buffer(out)),
        (4, KernelArg::Bytes(as_bytes(&attn))),
        (9, KernelArg::Bytes(as_bytes(&pp))),
    ];
    let mut tgmem = PRECISE_D512_TGMEM;
    if let Some(norms) = norms_arg {
        args.push((10, norms));
        tgmem += PRECISE_D512_CODES_EXTRA_TGMEM;
    }

    let grid = MTLSize::new(
        p.seq_len_q.div_ceil(nq) as u64,
        p.n_heads as u64,
        p.batch as u64,
    );
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &args,
        &[(0, tgmem)],
        grid,
        MTLSize::new(32, PRECISE_D512_NSG, 1),
    );
    Ok(())
}

/// Build every pipeline specialization a route can issue and return their
/// labels (for `KernelRegistry::pipeline_identity`).
///
/// `d256` names the V form and visibility of the sliding dispatches,
/// `d512` the V form of the causal resumes; both alignments of Q and K are
/// built for each.  Fails with the first pipeline error, so a caller that
/// prewarms at model load fails closed.
pub fn prewarm_flash_attn_prefill_precise(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    d256: Option<(PreciseVForm, PreciseD256MaskKind)>,
    d512: Option<PreciseVForm>,
) -> Result<Vec<String>> {
    let mut labels = Vec::new();
    for (align_q, align_k) in [(false, false), (false, true), (true, false), (true, true)] {
        if let Some((form, kind)) = d256 {
            let constants = d256_constants(align_q, align_k, kind);
            let name = d256_kernel(form);
            registry.get_pipeline_with_bool_constants(name, device.metal_device(), &constants)?;
            labels.push(pipeline_label(name, &constants));
        }
        if let Some(form) = d512 {
            let constants = d512_constants(align_q, align_k, true);
            let name = d512_kernel(form);
            registry.get_pipeline_with_bool_constants(name, device.metal_device(), &constants)?;
            labels.push(pipeline_label(name, &constants));
        }
    }
    Ok(labels)
}

#[cfg(test)]
#[path = "flash_attn_prefill_precise_tests.rs"]
mod tests;
