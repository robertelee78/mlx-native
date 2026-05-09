//! Differentiable Rotary Position Embedding — forward + backward.
//!
//! Used by hf2q's DWQ training tape (ADR-022 / flash_attn_train Phase 1a).
//! This module provides a standalone RoPE op that is its own backward:
//!
//!   Forward:  Q' = RoPE(Q, pos)
//!   Backward: dQ = RoPE(dQ', -pos)  (rotation matrix is orthogonal; R(θ)^T = R(-θ))
//!
//! # Implementation
//!
//! Both the forward and the backward dispatch the SAME Metal kernel —
//! `rope_multi_bf16` / `rope_multi_f32` from [`super::rope_multi`].  The
//! backward simply passes a negated copy of the positions buffer.
//! No new Metal shader is needed.
//!
//! # IMROPE convention (Qwen3.5 / Qwen3.6)
//!
//! `sections = [11, 11, 10, 0]` with `freq_base = 1e7` and mode = IMROPE (40).
//! Positions layout: `int32[4 * seq_len]` — first `seq_len` entries are the
//! time-axis positions, next `seq_len` the height-axis, then width, then extra.
//! For text-only inputs all four axes equal the token's 1-D position.
//!
//! Pair indexing is NeoX-style: each thread rotates
//! `(x[p], x[p + head_dim/2])` for pair `p ∈ [0, rope_dim/2)`.
//! Pairs `p ≥ rope_dim/2` pass through unchanged (partial-rotary tail).
//!
//! # References
//!
//! - `src/ops/rope_multi.rs` — the underlying dispatch + buffer construction
//! - `src/shaders/rope_multi.metal` — IMROPE / MROPE / VISION Metal kernel
//! - `tests/test_rope_multi.rs` — parity oracle (cpu_rope_multi)
//! - `/opt/hf2q/src/inference/models/qwen35/full_attn.rs:18-19` — production call-sites
//! - `/opt/hf2q/src/inference/models/qwen35/mod.rs:235` — mrope_section=[11,11,10,0]

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::rope_multi::{
    build_rope_multi_buffers, dispatch_rope_multi, RopeMultiMode, RopeMultiParams,
};

// ---------------------------------------------------------------------------
// Public parameter struct
// ---------------------------------------------------------------------------

/// Shape + frequency parameters for a differentiable RoPE dispatch.
///
/// Non-IMROPE (plain NeoX RoPE) is expressed as `sections = [head_dim/2, 0, 0, 0]`
/// with `mode = Imrope` — all pairs fall into axis 0 (text time-axis) which is
/// the only axis used.  Alternatively, callers can use `rope_multi` directly
/// with `mode = Mrope` and `sections = [rope_dim/2, 0, 0, 0]`.
#[derive(Debug, Clone, Copy)]
pub struct RopeTrainParams {
    pub batch: u32,
    /// Number of query/key heads.
    pub n_heads: u32,
    pub seq_len: u32,
    /// Full head dimension (must be even).
    pub head_dim: u32,
    /// Number of dimensions that participate in rotation (≤ head_dim, even).
    /// Pairs `[rope_dim/2, head_dim/2)` pass through unchanged.
    pub rope_dim: u32,
    /// Base frequency (theta).  Qwen3.5/3.6: `1_000_000.0 = 1e6`.
    ///
    /// Note: the metal shader comment in `test_rope_multi.rs` line 347 uses
    /// `1e7`; the Qwen3.5 model config uses `rope_theta = 1_000_000` = 1e6.
    /// The caller MUST pass the value that matches the model's GGUF
    /// `<prefix>.rope.freq_base` key.
    pub theta_base: f32,
    /// Section counts `[s0, s1, s2, s3]` for IMROPE / MROPE.
    ///
    /// Qwen3.5 / Qwen3.6: `[11, 11, 10, 0]` (IMROPE, matches
    /// `/opt/hf2q/src/inference/models/qwen35/mod.rs:235`).
    ///
    /// Sum `s0+s1+s2+s3` should equal `rope_dim / 2` for full rotary-section
    /// coverage.  The kernel tolerates sums smaller than `rope_dim/2`
    /// (sectors wrap modulo the sum), but callers should pass the canonical
    /// value from the model config.
    ///
    /// For non-IMROPE plain NeoX: `[rope_dim/2, 0, 0, 0]` with MROPE mode
    /// puts every pair in axis-0 (time).
    pub sections: [u32; 4],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Translate `RopeTrainParams` into the `RopeMultiParams` needed by
/// [`dispatch_rope_multi`], using IMROPE mode.
fn to_rope_multi_params(p: &RopeTrainParams) -> RopeMultiParams {
    RopeMultiParams {
        head_dim: p.head_dim,
        rope_dim: p.rope_dim,
        // rope_multi rows = seq_len * n_heads (batch dim is NOT part of
        // rope_multi; callers must slice or iterate per batch element).
        // Here we fold batch into n_heads for a single dispatch covering all
        // batch × n_heads rows.
        n_heads: p.n_heads,
        seq_len: p.seq_len * p.batch,  // fold batch: n_rows = batch*seq_len*n_heads
        freq_base: p.theta_base,
        mode: RopeMultiMode::Imrope,
        sections: p.sections,
    }
}

/// Validate `RopeTrainParams`.
fn validate_params(p: &RopeTrainParams) -> Result<()> {
    if p.batch == 0 || p.n_heads == 0 || p.seq_len == 0 || p.head_dim == 0 || p.rope_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "rope_train: batch, n_heads, seq_len, head_dim, rope_dim must all be > 0".into(),
        ));
    }
    if p.head_dim % 2 != 0 || p.rope_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(
            "rope_train: head_dim and rope_dim must be even".into(),
        ));
    }
    if p.rope_dim > p.head_dim {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train: rope_dim ({}) must be <= head_dim ({})",
            p.rope_dim, p.head_dim
        )));
    }
    if !p.theta_base.is_finite() || p.theta_base <= 0.0 {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train: theta_base must be finite and positive, got {}",
            p.theta_base
        )));
    }
    Ok(())
}

/// Check that `buf` has the expected element count and dtype.
fn validate_io(label: &str, buf: &MlxBuffer, expected_elems: usize, expected_dtype: DType) -> Result<()> {
    if buf.element_count() != expected_elems {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train: {label} element count {} != expected {}",
            buf.element_count(),
            expected_elems
        )));
    }
    if buf.dtype() != expected_dtype {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train: {label} dtype {} != expected {}",
            buf.dtype(),
            expected_dtype
        )));
    }
    Ok(())
}

/// Expected element count for `in_buf` / `out_buf`.
fn tensor_elems(p: &RopeTrainParams) -> usize {
    p.batch as usize * p.n_heads as usize * p.seq_len as usize * p.head_dim as usize
}

/// Expected element count for `pos_buf`.
fn pos_elems(p: &RopeTrainParams) -> usize {
    4 * p.seq_len as usize * p.batch as usize
}

// ---------------------------------------------------------------------------
// Forward — bf16
// ---------------------------------------------------------------------------

/// Apply RoPE (IMROPE mode) to `in_buf` and write result to `out_buf`.
///
/// # Buffers
///
/// | Buffer       | Shape                                   | DType |
/// |:------------ |:--------------------------------------- |:----- |
/// | `in_buf`     | `[batch, n_heads, seq_len, head_dim]`   | bf16  |
/// | `pos_buf`    | `[4 * batch * seq_len]` i32             | i32   |
/// | `out_buf`    | same as `in_buf`                        | bf16  |
///
/// The buffer layout for `pos_buf` folds batch into the positions array:
/// `[t_positions_batch0..batchN, h_positions_..., w_positions_..., e_positions_...]`
/// where each axis block has length `batch * seq_len`.
///
/// # Grid mapping
///
/// The underlying `rope_multi` kernel maps rows as `seq_len * n_heads` where
/// `seq_len` here is `batch * seq_len` (batch is folded in) and `n_heads` is
/// as given in `params`.  Thread `(pair_idx, row_idx)` handles one NeoX pair
/// for one (head, token) row.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_forward_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    mlx_device: &MlxDevice,
    in_buf: &MlxBuffer,
    pos_buf: &MlxBuffer,
    out_buf: &MlxBuffer,
    params: &RopeTrainParams,
) -> Result<()> {
    validate_params(params)?;
    let n_elems = tensor_elems(params);
    validate_io("in_buf", in_buf, n_elems, DType::BF16)?;
    validate_io("out_buf", out_buf, n_elems, DType::BF16)?;
    // pos_buf must be i32 (signed — backward uses negative positions).
    if pos_buf.element_count() != pos_elems(params) {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train forward: pos_buf element count {} != 4 * batch({}) * seq_len({}) = {}",
            pos_buf.element_count(),
            params.batch,
            params.seq_len,
            pos_elems(params)
        )));
    }
    match pos_buf.dtype() {
        DType::I32 | DType::U32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_train forward: pos_buf dtype {other} must be i32 or u32"
            )));
        }
    }

    let mp = to_rope_multi_params(params);
    let (params_buf, rope_params_buf, sections_buf) = build_rope_multi_buffers(mlx_device, mp)?;
    dispatch_rope_multi(
        encoder,
        registry,
        device,
        in_buf,
        out_buf,
        pos_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        mp,
    )
}

// ---------------------------------------------------------------------------
// Backward — bf16
// ---------------------------------------------------------------------------

/// Apply the RoPE backward pass: `dQ = RoPE(dQ', -pos)`.
///
/// Mathematically, the rotation matrix is orthogonal: `R(θ)^T = R(-θ)`.
/// Therefore `∂Q'/∂Q = R(pos)` and the VJP is `dQ = R(pos)^T · dQ' = R(-pos) · dQ'`.
///
/// This function negates every entry in `pos_buf` (on the CPU before upload
/// to the kernel) and dispatches `dispatch_rope_forward_bf16` with the negated
/// positions.
///
/// # Buffers
///
/// | Buffer         | Shape                                 | DType |
/// |:-------------- |:------------------------------------- |:----- |
/// | `grad_out_buf` | `[batch, n_heads, seq_len, head_dim]` | bf16  |
/// | `pos_buf`      | `[4 * batch * seq_len]` i32 (forward positions; NOT negated — this function negates internally) | i32 |
/// | `grad_in_buf`  | same as `grad_out_buf`               | bf16  |
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_backward_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    mlx_device: &MlxDevice,
    grad_out_buf: &MlxBuffer,
    pos_buf: &MlxBuffer,
    grad_in_buf: &MlxBuffer,
    params: &RopeTrainParams,
) -> Result<()> {
    validate_params(params)?;
    let n_elems = tensor_elems(params);
    validate_io("grad_out_buf", grad_out_buf, n_elems, DType::BF16)?;
    validate_io("grad_in_buf", grad_in_buf, n_elems, DType::BF16)?;
    if pos_buf.element_count() != pos_elems(params) {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train backward: pos_buf element count {} != 4 * batch({}) * seq_len({}) = {}",
            pos_buf.element_count(),
            params.batch,
            params.seq_len,
            pos_elems(params)
        )));
    }
    match pos_buf.dtype() {
        DType::I32 | DType::U32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_train backward: pos_buf dtype {other} must be i32 or u32"
            )));
        }
    }

    // Build a negated positions buffer on the host.  The pos values are i32
    // (signed); negation is defined.  U32-typed buffers are reinterpreted as
    // i32 for negation (the bit pattern is the same as a two's-complement
    // signed negate for values > 0; for pos=0, -0 = 0).
    let neg_pos_buf = negate_pos_buf_i32(mlx_device, pos_buf)?;

    let mp = to_rope_multi_params(params);
    let (params_buf, rope_params_buf, sections_buf) = build_rope_multi_buffers(mlx_device, mp)?;
    dispatch_rope_multi(
        encoder,
        registry,
        device,
        grad_out_buf,
        grad_in_buf,
        &neg_pos_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        mp,
    )
}

/// Build a new i32 buffer whose values are the negation of `pos_buf`'s values.
///
/// Handles both `DType::I32` and `DType::U32` source buffers (the rope_multi
/// kernel accepts both; we produce `DType::I32` for the negated output).
fn negate_pos_buf_i32(device: &MlxDevice, pos_buf: &MlxBuffer) -> Result<MlxBuffer> {
    let n = pos_buf.element_count();
    let src_bytes: Vec<i32> = match pos_buf.dtype() {
        DType::I32 => pos_buf.as_slice::<i32>()?.to_vec(),
        DType::U32 => pos_buf
            .as_slice::<u32>()?
            .iter()
            .map(|&v| v as i32)
            .collect(),
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "negate_pos_buf: unsupported dtype {other}"
            )))
        }
    };

    let negated: Vec<i32> = src_bytes.iter().map(|&v| v.wrapping_neg()).collect();
    let mut buf = device.alloc_buffer(n * 4, DType::I32, vec![n])?;
    buf.as_mut_slice::<i32>()?.copy_from_slice(&negated);
    Ok(buf)
}

// ---------------------------------------------------------------------------
// f32 variants
// ---------------------------------------------------------------------------

/// f32 forward variant.  Same contract as the bf16 version; operates on f32
/// `in_buf` / `out_buf`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_forward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    mlx_device: &MlxDevice,
    in_buf: &MlxBuffer,
    pos_buf: &MlxBuffer,
    out_buf: &MlxBuffer,
    params: &RopeTrainParams,
) -> Result<()> {
    validate_params(params)?;
    let n_elems = tensor_elems(params);
    validate_io("in_buf", in_buf, n_elems, DType::F32)?;
    validate_io("out_buf", out_buf, n_elems, DType::F32)?;
    if pos_buf.element_count() != pos_elems(params) {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train f32 forward: pos_buf element count {} != {}",
            pos_buf.element_count(),
            pos_elems(params)
        )));
    }
    match pos_buf.dtype() {
        DType::I32 | DType::U32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_train f32 forward: pos_buf dtype {other} must be i32 or u32"
            )));
        }
    }

    let mp = to_rope_multi_params(params);
    let (params_buf, rope_params_buf, sections_buf) = build_rope_multi_buffers(mlx_device, mp)?;
    dispatch_rope_multi(
        encoder,
        registry,
        device,
        in_buf,
        out_buf,
        pos_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        mp,
    )
}

/// f32 backward variant.  Same contract as the bf16 backward.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    mlx_device: &MlxDevice,
    grad_out_buf: &MlxBuffer,
    pos_buf: &MlxBuffer,
    grad_in_buf: &MlxBuffer,
    params: &RopeTrainParams,
) -> Result<()> {
    validate_params(params)?;
    let n_elems = tensor_elems(params);
    validate_io("grad_out_buf", grad_out_buf, n_elems, DType::F32)?;
    validate_io("grad_in_buf", grad_in_buf, n_elems, DType::F32)?;
    if pos_buf.element_count() != pos_elems(params) {
        return Err(MlxError::InvalidArgument(format!(
            "rope_train f32 backward: pos_buf element count {} != {}",
            pos_buf.element_count(),
            pos_elems(params)
        )));
    }
    match pos_buf.dtype() {
        DType::I32 | DType::U32 => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "rope_train f32 backward: pos_buf dtype {other} must be i32 or u32"
            )));
        }
    }

    let neg_pos_buf = negate_pos_buf_i32(mlx_device, pos_buf)?;
    let mp = to_rope_multi_params(params);
    let (params_buf, rope_params_buf, sections_buf) = build_rope_multi_buffers(mlx_device, mp)?;
    dispatch_rope_multi(
        encoder,
        registry,
        device,
        grad_out_buf,
        grad_in_buf,
        &neg_pos_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        mp,
    )
}
