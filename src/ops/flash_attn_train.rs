//! Flash-attention training forward kernel — host dispatch.
//!
//! FA-2 forward pass that emits BOTH the attention output `O` AND the
//! per-row natural-log logsumexp `L` required by the Phase 2 backward.
//!
//! ## Algorithm
//!
//! Identical to [`super::flash_attn_prefill`] (online softmax, simdgroup MMA,
//! same tile geometry, same causal / additive-mask handling, same GQA).
//!
//! The only addition is the `L_out [B, H_q, qL]` f32 buffer at `buffer(8)`.
//! After the K-tile sweep each thread with `sn == 0` writes one f32:
//!
//! ```text
//! L[b, h, i] = max_score_b2 * ln(2) + ln(sum_score_b2)
//! ```
//!
//! where `max_score_b2` and `sum_score_b2` are the per-row base-2
//! running max / unnormalized exp2 sum from the K-sweep (Q is pre-scaled
//! by `scale * log2(e)` so all accumulators live in base-2 space).
//!
//! This equals the FA-2 paper Algorithm 1 logsumexp:
//! `L_i = m_i + log( sum_j exp(s_ij - m_i) )` in natural-log units.
//!
//! ## Buffer layout
//!
//! | Index | Name     | Shape               | DType |
//! |-------|----------|---------------------|-------|
//! | 0     | Q        | `[B, H_q, qL, D]`   | BF16  |
//! | 1     | K        | `[B, H_kv, kL, D]`  | BF16  |
//! | 2     | V        | `[B, H_kv, kL, D]`  | BF16  |
//! | 3     | O (out)  | `[B, H_q, qL, D]`   | BF16  |
//! | 4     | params   | 160-byte ABI struct  | —     |
//! | 5     | mask_params | 24-byte struct    | — (when has_mask) |
//! | 6     | mask     | `[B, H_q, qL, kL]`  | BF16 or bool (when has_mask) |
//! | 8     | L_out    | `[B, H_q, qL]`      | F32   |
//!
//! ## Function constants
//!
//! Same 4 constants as `flash_attn_prefill.metal`:
//!
//! | Index | Name      | Semantics |
//! |-------|-----------|-----------|
//! | 200   | align_Q   | `qL % BQ == 0` |
//! | 201   | align_K   | `kL % BK == 0` |
//! | 300   | has_mask  | additive/bool mask buffer bound |
//! | 301   | do_causal | in-kernel causal masking |
//!
//! ## Kernel variants
//!
//! | Name | D | I/O dtype | Mask kind |
//! |------|---|-----------|-----------|
//! | `flash_attn_train_fwd_bf16_d64`          | 64  | bf16 | bf16 additive |
//! | `flash_attn_train_fwd_bf16_d64_boolmask` | 64  | bf16 | bool |
//! | `flash_attn_train_fwd_bf16_d256`          | 256 | bf16 | bf16 additive |
//! | `flash_attn_train_fwd_bf16_d256_boolmask` | 256 | bf16 | bool |
//!
//! ## Scale convention
//!
//! Pass `scale = 1.0 / sqrt(head_dim)`.  The kernel multiplies internally by
//! `log2(e)`.  Do NOT pre-multiply by `log2(e)` on the host.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CapturedOpKind, CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::flash_attn_prefill::{AttnMaskParamsGpu, AttnParamsGpu};

// ─── Shader source ───────────────────────────────────────────────────────────

/// MSL source (embedded at compile time).
pub static FLASH_ATTN_TRAIN_FWD_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_train_fwd.metal");

// ─── Kernel names ────────────────────────────────────────────────────────────

const K_BF16_D64: &str = "flash_attn_train_fwd_bf16_d64";
const K_BF16_D64_BOOLMASK: &str = "flash_attn_train_fwd_bf16_d64_boolmask";
const K_BF16_D256: &str = "flash_attn_train_fwd_bf16_d256";
const K_BF16_D256_BOOLMASK: &str = "flash_attn_train_fwd_bf16_d256_boolmask";

const ALL_KERNEL_NAMES: &[&str] = &[
    K_BF16_D64,
    K_BF16_D64_BOOLMASK,
    K_BF16_D256,
    K_BF16_D256_BOOLMASK,
];

// ─── Registration ─────────────────────────────────────────────────────────────

/// Register all 4 training-forward kernel entry points with the registry.
///
/// Must be called before any `dispatch_flash_attn_train_fwd_*` call.
pub fn register(registry: &mut KernelRegistry) {
    for &name in ALL_KERNEL_NAMES {
        registry.register_source(name, FLASH_ATTN_TRAIN_FWD_SHADER_SOURCE);
    }
}

// ─── Tile geometry ────────────────────────────────────────────────────────────

// D=64 and D=256 share the same tile geometry.
const BQ: u32 = 32;
const BK: u32 = 16;
const WM: u32 = 4;
const WN: u32 = 1;

// ─── Public parameter struct ──────────────────────────────────────────────────

/// Host-side parameters for the flash-attention training forward dispatcher.
///
/// Mirrors [`crate::ops::flash_attn_prefill::FlashAttnPrefillParams`] but is
/// kept separate to decouple the training API from the inference API.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnTrainParams {
    /// Batch size.
    pub batch: u32,
    /// Number of query attention heads.
    pub n_q_heads: u32,
    /// Number of key/value attention heads.  Must divide `n_q_heads` evenly.
    pub n_kv_heads: u32,
    /// Head dimension.  Must be 64 (D=64 dispatcher) or 256 (D=256 dispatcher).
    pub head_dim: u32,
    /// Query sequence length.
    pub q_seq_len: u32,
    /// Key/value sequence length.
    pub k_seq_len: u32,
    /// Attention scale.  Typically `1.0 / sqrt(head_dim)`.
    ///
    /// The kernel multiplies by `log2(e) ≈ 1.44269504` internally.
    /// Do NOT pre-multiply by `log2(e)` here.
    pub scale: f32,
    /// Apply causal masking in-kernel.
    pub causal: bool,
}

// ─── Input validation ─────────────────────────────────────────────────────────

fn validate_params(p: &FlashAttnTrainParams) -> Result<()> {
    if p.n_q_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_train: n_q_heads must be > 0".into(),
        ));
    }
    if p.n_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_train: n_kv_heads must be > 0".into(),
        ));
    }
    if p.n_q_heads % p.n_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_train: n_q_heads ({}) must be divisible by n_kv_heads ({})",
            p.n_q_heads, p.n_kv_heads
        )));
    }
    if p.q_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_train: q_seq_len must be > 0".into(),
        ));
    }
    if p.k_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_train: k_seq_len must be > 0".into(),
        ));
    }
    if p.batch == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_train: batch must be > 0".into(),
        ));
    }
    Ok(())
}

fn validate_buffer_size(buf: &MlxBuffer, name: &str, expected_elements: usize) -> Result<()> {
    let expected_bytes = expected_elements * buf.dtype().size_of();
    if buf.byte_len() < expected_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_train: {name} buffer too small: expected at least \
             {expected_bytes} bytes, got {}",
            buf.byte_len()
        )));
    }
    Ok(())
}

// ─── Shared dispatch core ─────────────────────────────────────────────────────

/// Inner dispatch used by both the D=64 and D=256 public dispatchers.
///
/// `kernel_name` must be one of the 4 registered names.
/// `head_dim_expected` is checked against `params.head_dim` before dispatch.
#[allow(clippy::too_many_arguments)]
fn dispatch_inner(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    o_buf: &mut MlxBuffer,
    l_buf: &mut MlxBuffer,
    params: &FlashAttnTrainParams,
    kernel_name: &str,
    head_dim_expected: u32,
) -> Result<()> {
    // ── Validate head_dim ──────────────────────────────────────────────────
    if params.head_dim != head_dim_expected {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_train ({}): head_dim must be {head_dim_expected}, got {}",
            kernel_name, params.head_dim
        )));
    }

    validate_params(params)?;

    // ── Dtype checks ───────────────────────────────────────────────────────
    for (buf, name) in &[(q_buf, "Q"), (k_buf, "K"), (v_buf, "V"), (o_buf as &MlxBuffer, "O")] {
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train ({kernel_name}): {name} buffer must be BF16, got {:?}",
                buf.dtype()
            )));
        }
    }
    if l_buf.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_train ({kernel_name}): L_out buffer must be F32, got {:?}",
            l_buf.dtype()
        )));
    }
    if let Some(m) = mask {
        if m.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train ({kernel_name}): mask buffer must be BF16, got {:?}",
                m.dtype()
            )));
        }
    }

    // ── Shape arithmetic ───────────────────────────────────────────────────
    let batch = params.batch as usize;
    let h = params.n_q_heads as usize;
    let h_kv = params.n_kv_heads as usize;
    let ql = params.q_seq_len as usize;
    let kl = params.k_seq_len as usize;
    let d = params.head_dim as usize;

    validate_buffer_size(q_buf, "Q", batch * h * ql * d)?;
    validate_buffer_size(k_buf, "K", batch * h_kv * kl * d)?;
    validate_buffer_size(v_buf, "V", batch * h_kv * kl * d)?;
    validate_buffer_size(o_buf, "O", batch * h * ql * d)?;
    validate_buffer_size(l_buf, "L_out", batch * h * ql)?;
    if let Some(m) = mask {
        validate_buffer_size(m, "mask", batch * h * ql * kl)?;
    }

    // ── Tile geometry ──────────────────────────────────────────────────────
    let nq = params.q_seq_len.div_ceil(BQ);
    let nk = params.k_seq_len.div_ceil(BK);
    let nq_aligned = params.q_seq_len / BQ;
    let nk_aligned = params.k_seq_len / BK;
    let ql_rem = params.q_seq_len % BQ;
    let kl_rem = params.k_seq_len % BK;

    let align_q = ql_rem == 0;
    let align_k = kl_rem == 0;
    let has_mask = mask.is_some();
    let do_causal = params.causal;

    // ── Pipeline ───────────────────────────────────────────────────────────
    let pipeline = registry.get_pipeline_with_bool_constants(
        kernel_name,
        device.metal_device(),
        &[
            (200, align_q),
            (201, align_k),
            (300, has_mask),
            (301, do_causal),
        ],
    )?;

    // ── AttnParamsGpu ──────────────────────────────────────────────────────
    let q_seq_stride = d as i64;
    let q_head_stride = (ql * d) as i64;
    let q_batch_stride = (h * ql * d) as i64;

    let kv_seq_stride = d as i64;
    let kv_head_stride = (kl * d) as i64;
    let kv_batch_stride = (h_kv * kl * d) as i64;

    let gqa_factor = (params.n_q_heads / params.n_kv_heads) as i32;

    let attn_params = AttnParamsGpu {
        b: params.batch as i32,
        h: params.n_q_heads as i32,
        d: params.head_dim as i32,
        ql: params.q_seq_len as i32,
        kl: params.k_seq_len as i32,
        gqa_factor,
        scale: params.scale,
        softcapping: 1.0_f32,
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql_rem as i32,
        kl_rem: kl_rem as i32,
        ql_off: 0,
        _pad: 0,
        q_strides: [q_batch_stride, q_head_stride, q_seq_stride],
        k_strides: [kv_batch_stride, kv_head_stride, kv_seq_stride],
        v_strides: [kv_batch_stride, kv_head_stride, kv_seq_stride],
        o_strides: [q_batch_stride, q_head_stride, q_seq_stride],
    };

    // ── Grid ───────────────────────────────────────────────────────────────
    //   grid = (ceil(qL / BQ), H_q, B)
    //   tg   = (32, WM, WN)
    let grid = MTLSize::new(nq as u64, params.n_q_heads as u64, params.batch as u64);
    let tg_size = MTLSize::new(32, WM as u64, WN as u64);

    // ── Encode ────────────────────────────────────────────────────────────
    encoder.set_op_kind(CapturedOpKind::Sdpa);

    if let Some(mask_buf) = mask {
        // Rank-4 mask [B, H, qL, kL] — per-head layout.
        let m_batch_stride = (h * ql * kl) as i64;
        let m_head_stride = (ql * kl) as i64;
        let m_ql_stride = kl as i64;

        let mask_params = AttnMaskParamsGpu {
            m_strides: [m_batch_stride, m_head_stride, m_ql_stride],
        };

        encoder.encode_threadgroups_with_args(
            pipeline,
            &[
                (0, KernelArg::Buffer(q_buf)),
                (1, KernelArg::Buffer(k_buf)),
                (2, KernelArg::Buffer(v_buf)),
                (3, KernelArg::Buffer(o_buf)),
                (4, KernelArg::Bytes(as_bytes(&attn_params))),
                (5, KernelArg::Bytes(as_bytes(&mask_params))),
                (6, KernelArg::Buffer(mask_buf)),
                // buffer(7) intentionally absent (blk not used in training fwd)
                (8, KernelArg::Buffer(l_buf)),
            ],
            grid,
            tg_size,
        );
    } else {
        encoder.encode_threadgroups_with_args(
            pipeline,
            &[
                (0, KernelArg::Buffer(q_buf)),
                (1, KernelArg::Buffer(k_buf)),
                (2, KernelArg::Buffer(v_buf)),
                (3, KernelArg::Buffer(o_buf)),
                (4, KernelArg::Bytes(as_bytes(&attn_params))),
                // buffers 5, 6 absent — has_mask=false constant dead-codes them
                (8, KernelArg::Buffer(l_buf)),
            ],
            grid,
            tg_size,
        );
    }

    Ok(())
}

// ─── Public dispatchers ───────────────────────────────────────────────────────

/// Dispatch the FA-2 forward pass for bf16 Q/K/V/O, head_dim=64.
///
/// Encodes a compute command into `encoder` without committing.
///
/// # Buffer shapes
///
/// - `q_buf`  — `[batch, n_q_heads, q_seq_len, 64]`  BF16
/// - `k_buf`  — `[batch, n_kv_heads, k_seq_len, 64]` BF16
/// - `v_buf`  — `[batch, n_kv_heads, k_seq_len, 64]` BF16
/// - `mask`   — `[batch, n_q_heads, q_seq_len, k_seq_len]` BF16, or `None`
/// - `o_buf`  — `[batch, n_q_heads, q_seq_len, 64]`  BF16 (output)
/// - `l_buf`  — `[batch, n_q_heads, q_seq_len]`      F32  (logsumexp output)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for wrong head_dim, wrong dtype,
/// bad GQA ratio, or undersized buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_train_fwd_bf16_d64(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    o_buf: &mut MlxBuffer,
    l_buf: &mut MlxBuffer,
    params: &FlashAttnTrainParams,
) -> Result<()> {
    dispatch_inner(
        encoder, device, registry,
        q_buf, k_buf, v_buf, mask, o_buf, l_buf,
        params, K_BF16_D64, 64,
    )
}

/// Dispatch the FA-2 forward pass for bf16 Q/K/V/O, head_dim=256.
///
/// Same semantics as [`dispatch_flash_attn_train_fwd_bf16_d64`] but for
/// the production Qwen3.6-35B-A3B head dimension (D=256).
///
/// # Errors
///
/// Same as `dispatch_flash_attn_train_fwd_bf16_d64`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_train_fwd_bf16_d256(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    o_buf: &mut MlxBuffer,
    l_buf: &mut MlxBuffer,
    params: &FlashAttnTrainParams,
) -> Result<()> {
    dispatch_inner(
        encoder, device, registry,
        q_buf, k_buf, v_buf, mask, o_buf, l_buf,
        params, K_BF16_D256, 256,
    )
}

// ─── Kernel-name coverage test (compile-time) ─────────────────────────────────

/// Returns all 4 registered kernel names.
///
/// Exposed for integration tests (`tests/test_flash_attn_train.rs`).
/// `#[cfg(test)]` cannot be used here because integration tests are a
/// separate crate and `#[cfg(test)]` is not set for them.
#[doc(hidden)]
pub fn all_kernel_names_for_test() -> &'static [&'static str] {
    ALL_KERNEL_NAMES
}
