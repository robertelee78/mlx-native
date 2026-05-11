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
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
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
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
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
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2 — FA-2 backward kernel (dQ, dK, dV)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Three-kernel chain per call:
//   1. `flash_attn_train_bwd_compute_d_bf16` — computes D[b,h,i] = rowsum(O·dO)
//   2. `flash_attn_train_bwd_bf16_d{64,256}` — writes f32 dQ, f32 dK_scratch,
//      f32 dV_scratch via Q-tile-outer grid + atomic f32 adds for dK/dV.
//   3. `f32_to_bf16_cast` — casts f32 dK_scratch and dV_scratch to bf16 output.
//
// The caller passes pre-allocated BF16 dQ/dK/dV buffers.  The dispatcher
// allocates two intermediate f32 scratch buffers (dK_f32, dV_f32) internally;
// dQ is written directly as f32 into a temporary and then cast.
//
// Buffer sizing:
//   dK_f32  [B, H_kv, kL, D] f32
//   dV_f32  [B, H_kv, kL, D] f32
//   dQ_f32  [B, H_q,  qL, D] f32
//   D_vec   [B, H_q,  qL]    f32
//
// After the backward kernel, each f32 result is cast to the caller-supplied
// bf16 output buffer via `f32_to_bf16_cast`.

/// MSL source for the backward kernels (embedded at compile time).
pub static FLASH_ATTN_TRAIN_BWD_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_train_bwd.metal");

/// MSL source for the compute-D pre-pass kernel.
pub static FLASH_ATTN_TRAIN_BWD_COMPUTE_D_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_train_bwd_compute_d.metal");

// Backward kernel names.
const K_BWD_COMPUTE_D: &str = "flash_attn_train_bwd_compute_d_bf16";
const K_BWD_D64: &str = "flash_attn_train_bwd_bf16_d64";
const K_BWD_D256: &str = "flash_attn_train_bwd_bf16_d256";
const K_F32_TO_BF16: &str = "f32_to_bf16_cast";

const ALL_BWD_KERNEL_NAMES: &[&str] = &[
    K_BWD_COMPUTE_D,
    K_BWD_D64,
    K_BWD_D256,
    K_F32_TO_BF16,
];

/// Register all backward kernel entry points with the registry.
///
/// Must be called before any `dispatch_flash_attn_train_bwd_*` call.
/// Safe to call alongside [`register`] (forward registration).
pub fn register_bwd(registry: &mut KernelRegistry) {
    registry.register_source(K_BWD_COMPUTE_D, FLASH_ATTN_TRAIN_BWD_COMPUTE_D_SHADER_SOURCE);
    for &name in &[K_BWD_D64, K_BWD_D256, K_F32_TO_BF16] {
        registry.register_source(name, FLASH_ATTN_TRAIN_BWD_SHADER_SOURCE);
    }
}

/// Returns all 4 backward kernel names.  Exposed for integration tests.
#[doc(hidden)]
pub fn all_bwd_kernel_names_for_test() -> &'static [&'static str] {
    ALL_BWD_KERNEL_NAMES
}

// ── compute-D pre-pass ────────────────────────────────────────────────────────

/// Struct for the compute-D Metal kernel params (4 × u32 = 16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeDParams {
    batch: u32,
    n_q_heads: u32,
    q_seq_len: u32,
    head_dim: u32,
}

/// Encode the compute-D pre-pass: `D[b,h,i] = rowsum(O[b,h,i,:] * dO[b,h,i,:])`.
///
/// `d_out_buf` must be `[B, H_q, qL]` f32, allocated and zero-initialized by the
/// caller (the dispatcher allocates it internally).
///
/// Grid: `(qL, 1, B*H_q)`, tg_size = `(min(256, next_pow2(D)), 1, 1)`.
fn dispatch_compute_d(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    o_buf: &MlxBuffer,
    do_buf: &MlxBuffer,
    d_out_buf: &MlxBuffer,
    params: &FlashAttnTrainParams,
) -> Result<()> {
    let p = ComputeDParams {
        batch: params.batch,
        n_q_heads: params.n_q_heads,
        q_seq_len: params.q_seq_len,
        head_dim: params.head_dim,
    };

    let pipeline = registry.get_pipeline(K_BWD_COMPUTE_D, device)?;

    let tg_x = std::cmp::min(256, params.head_dim.next_power_of_two()) as u64;
    let grid = MTLSize::new(
        params.q_seq_len as u64,
        1,
        (params.batch * params.n_q_heads) as u64,
    );
    let tg_size = MTLSize::new(tg_x, 1, 1);

    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(o_buf)),
            (1, KernelArg::Buffer(do_buf)),
            (2, KernelArg::Buffer(d_out_buf)),
            (3, KernelArg::Bytes(as_bytes(&p))),
        ],
        grid,
        tg_size,
    );

    Ok(())
}

// ── elementwise f32→bf16 cast ─────────────────────────────────────────────────

/// Encode an elementwise f32→bf16 cast of `n_elems` elements.
///
/// `src` must be f32; `dst` must be bf16 with the same element count.
/// The cast kernel's buffer(2) receives `n_elems` as a u32 OOB guard.
fn dispatch_f32_to_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    n_elems: usize,
) -> Result<()> {
    let pipeline = registry.get_pipeline(K_F32_TO_BF16, device)?;
    let tg_x = std::cmp::min(256u64, n_elems as u64);
    let n_groups = (n_elems as u64).div_ceil(tg_x);
    let n_u32 = n_elems as u32;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&n_u32))),
        ],
        MTLSize::new(n_groups, 1, 1),
        MTLSize::new(tg_x, 1, 1),
    );
    Ok(())
}

// ── backward inner dispatch ───────────────────────────────────────────────────

/// Inner backward dispatch shared by the D=64 and D=256 public functions.
///
/// Runs the three-kernel chain:
/// 1. compute_D pre-pass
/// 2. FA-2 Algorithm 4 backward (Q-tile-outer, writes f32 dQ/dK/dV scratch)
/// 3. f32 → bf16 cast for dQ, dK, dV
#[allow(clippy::too_many_arguments)]
fn dispatch_bwd_inner(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
    do_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    dq_buf: &MlxBuffer,
    dk_buf: &MlxBuffer,
    dv_buf: &MlxBuffer,
    params: &FlashAttnTrainParams,
    bwd_kernel_name: &str,
    head_dim_expected: u32,
) -> Result<()> {
    // ── head_dim check ────────────────────────────────────────────────────────
    if params.head_dim != head_dim_expected {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_train_bwd ({bwd_kernel_name}): head_dim must be \
             {head_dim_expected}, got {}",
            params.head_dim
        )));
    }

    validate_params(params)?;

    // ── dtype checks ──────────────────────────────────────────────────────────
    for (buf, name) in &[
        (q_buf, "Q"),
        (k_buf, "K"),
        (v_buf, "V"),
        (o_buf, "O"),
        (do_buf, "dO"),
    ] {
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train_bwd ({bwd_kernel_name}): {name} buffer must be BF16, \
                 got {:?}",
                buf.dtype()
            )));
        }
    }
    for (buf, name) in &[(l_buf, "L")] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train_bwd ({bwd_kernel_name}): {name} buffer must be F32, \
                 got {:?}",
                buf.dtype()
            )));
        }
    }
    for (buf, name) in &[
        (dq_buf as &MlxBuffer, "dQ"),
        (dk_buf as &MlxBuffer, "dK"),
        (dv_buf as &MlxBuffer, "dV"),
    ] {
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train_bwd ({bwd_kernel_name}): {name} output buffer must be \
                 BF16, got {:?}",
                buf.dtype()
            )));
        }
    }
    if let Some(m) = mask {
        if m.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_train_bwd ({bwd_kernel_name}): mask buffer must be BF16, \
                 got {:?}",
                m.dtype()
            )));
        }
    }

    // ── shape arithmetic ──────────────────────────────────────────────────────
    let batch = params.batch as usize;
    let h_q = params.n_q_heads as usize;
    let h_kv = params.n_kv_heads as usize;
    let ql = params.q_seq_len as usize;
    let kl = params.k_seq_len as usize;
    let d = params.head_dim as usize;

    let q_elems = batch * h_q * ql * d;
    let kv_elems = batch * h_kv * kl * d;
    let l_elems = batch * h_q * ql;

    validate_buffer_size(q_buf, "Q", q_elems)?;
    validate_buffer_size(k_buf, "K", kv_elems)?;
    validate_buffer_size(v_buf, "V", kv_elems)?;
    validate_buffer_size(o_buf, "O", q_elems)?;
    validate_buffer_size(l_buf, "L", l_elems)?;
    validate_buffer_size(do_buf, "dO", q_elems)?;
    validate_buffer_size(dq_buf, "dQ", q_elems)?;
    validate_buffer_size(dk_buf, "dK", kv_elems)?;
    validate_buffer_size(dv_buf, "dV", kv_elems)?;
    if let Some(m) = mask {
        validate_buffer_size(m, "mask", batch * h_q * ql * kl)?;
    }

    // ── allocate internal f32 scratch buffers ─────────────────────────────────
    // alloc_buffer zero-initialises all bytes (ADR-015 iter61a fix in device.rs).
    // dK_f32 and dV_f32 accumulate via f32 atomic adds in the backward kernel;
    // dQ_f32 is written (not accumulated) by the backward kernel.
    let d_vec_buf = device
        .alloc_buffer(l_elems * 4, DType::F32, vec![l_elems])
        .map_err(|e| MlxError::InvalidArgument(format!("flash_attn_train_bwd: alloc D_vec: {e}")))?;
    let dq_f32_buf = device
        .alloc_buffer(q_elems * 4, DType::F32, vec![q_elems])
        .map_err(|e| MlxError::InvalidArgument(format!("flash_attn_train_bwd: alloc dQ_f32: {e}")))?;
    let dk_f32_buf = device
        .alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems])
        .map_err(|e| MlxError::InvalidArgument(format!("flash_attn_train_bwd: alloc dK_f32: {e}")))?;
    let dv_f32_buf = device
        .alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems])
        .map_err(|e| MlxError::InvalidArgument(format!("flash_attn_train_bwd: alloc dV_f32: {e}")))?;

    // ── tile geometry (same as forward) ───────────────────────────────────────
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

    // ── AttnParamsGpu ─────────────────────────────────────────────────────────
    let q_seq_stride = d as i64;
    let q_head_stride = (ql * d) as i64;
    let q_batch_stride = (h_q * ql * d) as i64;
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

    // ── Kernel 1: compute D ───────────────────────────────────────────────────
    dispatch_compute_d(
        encoder, registry, device.metal_device(),
        o_buf, do_buf, &d_vec_buf, params,
    )?;
    encoder.memory_barrier();

    // ── Kernel 2: FA-2 backward ───────────────────────────────────────────────
    let bwd_pipeline = registry.get_pipeline_with_bool_constants(
        bwd_kernel_name,
        device.metal_device(),
        &[
            (200, align_q),
            (201, align_k),
            (300, has_mask),
            (301, do_causal),
        ],
    )?;

    // Grid: (ceil(qL/BQ), H_q, B), tg_size: (32, WM, WN)
    let grid = MTLSize::new(nq as u64, params.n_q_heads as u64, params.batch as u64);
    let tg_size = MTLSize::new(32, WM as u64, WN as u64);

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    if let Some(mask_buf) = mask {
        let m_batch_stride = (h_q * ql * kl) as i64;
        let m_head_stride = (ql * kl) as i64;
        let m_ql_stride = kl as i64;
        let mask_params = AttnMaskParamsGpu {
            m_strides: [m_batch_stride, m_head_stride, m_ql_stride],
        };
        encoder.encode_threadgroups_with_args(
            bwd_pipeline,
            &[
                (0, KernelArg::Buffer(q_buf)),
                (1, KernelArg::Buffer(k_buf)),
                (2, KernelArg::Buffer(v_buf)),
                // buffer(3) unused (O is not needed in backward computation)
                (4, KernelArg::Buffer(l_buf)),
                (5, KernelArg::Buffer(do_buf)),
                (6, KernelArg::Buffer(&d_vec_buf)),
                (7, KernelArg::Buffer(&dq_f32_buf)),
                (8, KernelArg::Buffer(&dk_f32_buf)),
                (9, KernelArg::Buffer(&dv_f32_buf)),
                (10, KernelArg::Bytes(as_bytes(&attn_params))),
                (11, KernelArg::Bytes(as_bytes(&mask_params))),
                (12, KernelArg::Buffer(mask_buf)),
            ],
            grid,
            tg_size,
        );
    } else {
        encoder.encode_threadgroups_with_args(
            bwd_pipeline,
            &[
                (0, KernelArg::Buffer(q_buf)),
                (1, KernelArg::Buffer(k_buf)),
                (2, KernelArg::Buffer(v_buf)),
                // buffer(3) unused
                (4, KernelArg::Buffer(l_buf)),
                (5, KernelArg::Buffer(do_buf)),
                (6, KernelArg::Buffer(&d_vec_buf)),
                (7, KernelArg::Buffer(&dq_f32_buf)),
                (8, KernelArg::Buffer(&dk_f32_buf)),
                (9, KernelArg::Buffer(&dv_f32_buf)),
                (10, KernelArg::Bytes(as_bytes(&attn_params))),
            ],
            grid,
            tg_size,
        );
    }
    encoder.memory_barrier();

    // ── Kernel 3: f32 → bf16 cast for dQ, dK, dV ─────────────────────────────
    dispatch_f32_to_bf16(encoder, registry, device.metal_device(), &dq_f32_buf, dq_buf, q_elems)?;
    encoder.memory_barrier();
    dispatch_f32_to_bf16(encoder, registry, device.metal_device(), &dk_f32_buf, dk_buf, kv_elems)?;
    encoder.memory_barrier();
    dispatch_f32_to_bf16(encoder, registry, device.metal_device(), &dv_f32_buf, dv_buf, kv_elems)?;

    Ok(())
}

// ── Public backward dispatchers ───────────────────────────────────────────────

/// Dispatch the FA-2 backward pass for bf16 I/O, head_dim=64.
///
/// Encodes a three-kernel sequence into `encoder`:
/// 1. Compute D pre-pass (`D[b,h,i] = rowsum(O·dO)`).
/// 2. FA-2 Algorithm 4 backward: writes f32 dQ/dK/dV.
/// 3. Cast f32 dQ/dK/dV → bf16 output buffers.
///
/// # Buffer shapes
///
/// - `q_buf`  — `[batch, n_q_heads, q_seq_len, 64]`  BF16
/// - `k_buf`  — `[batch, n_kv_heads, k_seq_len, 64]` BF16
/// - `v_buf`  — `[batch, n_kv_heads, k_seq_len, 64]` BF16
/// - `o_buf`  — `[batch, n_q_heads, q_seq_len, 64]`  BF16 (forward output)
/// - `l_buf`  — `[batch, n_q_heads, q_seq_len]`      F32  (forward logsumexp)
/// - `do_buf` — `[batch, n_q_heads, q_seq_len, 64]`  BF16 (upstream gradient)
/// - `mask`   — `[batch, n_q_heads, q_seq_len, k_seq_len]` BF16 additive, or `None`
/// - `dq_buf` — `[batch, n_q_heads, q_seq_len, 64]`  BF16 (output, zero-init by caller)
/// - `dk_buf` — `[batch, n_kv_heads, k_seq_len, 64]` BF16 (output, zero-init by caller)
/// - `dv_buf` — `[batch, n_kv_heads, k_seq_len, 64]` BF16 (output, zero-init by caller)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for wrong head_dim, dtype mismatch,
/// GQA ratio error, or undersized buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_train_bwd_bf16_d64(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
    do_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    dq_buf: &MlxBuffer,
    dk_buf: &MlxBuffer,
    dv_buf: &MlxBuffer,
    params: &FlashAttnTrainParams,
) -> Result<()> {
    dispatch_bwd_inner(
        encoder, device, registry,
        q_buf, k_buf, v_buf, o_buf, l_buf, do_buf, mask,
        dq_buf, dk_buf, dv_buf,
        params, K_BWD_D64, 64,
    )
}

/// Dispatch the FA-2 backward pass for bf16 I/O, head_dim=256.
///
/// Same semantics as [`dispatch_flash_attn_train_bwd_bf16_d64`] but for
/// Qwen3.6-35B-A3B head dimension (D=256).
///
/// # Errors
///
/// Same as `dispatch_flash_attn_train_bwd_bf16_d64`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_train_bwd_bf16_d256(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_buf: &MlxBuffer,
    k_buf: &MlxBuffer,
    v_buf: &MlxBuffer,
    o_buf: &MlxBuffer,
    l_buf: &MlxBuffer,
    do_buf: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    dq_buf: &MlxBuffer,
    dk_buf: &MlxBuffer,
    dv_buf: &MlxBuffer,
    params: &FlashAttnTrainParams,
) -> Result<()> {
    dispatch_bwd_inner(
        encoder, device, registry,
        q_buf, k_buf, v_buf, o_buf, l_buf, do_buf, mask,
        dq_buf, dk_buf, dv_buf,
        params, K_BWD_D256, 256,
    )
}
