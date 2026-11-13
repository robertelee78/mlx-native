//! Tensor-API flash attention for prefill (hf2q ADR-063 gap program F1).
//!
//! A port of the peer's `kernel_flash_attn_ext_tensor` for F16 K/V: Metal 4
//! `mpp::tensor_ops::matmul2d` for Q·Kᵀ and P·V, 32 queries per threadgroup
//! at head dim 256 and 16 at 512, 64 cache rows per step, 8 simdgroups,
//! online softmax with lazy rescaling and half P. See
//! `shaders/flash_attn_prefill_tensor.metal`.
//!
//! Inputs and outputs use hf2q's layer layouts directly:
//! - Q F32 rows × heads × head_dim with explicit row/head strides,
//! - K/V F16 with explicit row/head strides (one KV head per
//!   `n_heads / n_kv_heads` query heads),
//! - an additive F16 mask `[rows, kv_pad]` shared by every head, and its
//!   `blk` classification from [`crate::ops::flash_attn_prefill_blk`] with
//!   `(bq, bk) = (`[`queries_per_threadgroup`]`, `[`CACHE_ROWS_PER_STEP`]`)`,
//! - O F32 `[rows, n_heads, head_dim]` contiguous.
//!
//! Host contract: `kv_pad` is a multiple of [`CACHE_ROWS_PER_STEP`], and every
//! K/V row below `kv_pad` is finite (padding rows zero, masked `-inf`).
//!
//! The pipelines exist only on devices with the Metal tensor API; use
//! [`flash_attn_prefill_tensor_available`] before offering the route.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::tensor_kernels::typed_unavailable;
use crate::DType;

/// MSL source.
pub static FLASH_ATTN_PREFILL_TENSOR_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_prefill_tensor.metal");

/// Kernel entry point at head dim 256.
pub const K_TENSOR_F16_D256: &str = "flash_attn_prefill_tensor_f16_d256";
/// Kernel entry point at head dim 512.
pub const K_TENSOR_F16_D512: &str = "flash_attn_prefill_tensor_f16_d512";

/// Cache rows per step (the kernel's `C`, the blk `bk`).
pub const CACHE_ROWS_PER_STEP: u32 = 64;
const SIMDGROUPS: u64 = 8;
/// The route as [`MlxError::Unavailable`] names it.
const FEATURE: &str = "the tensor flash attention (flash_attn_prefill_tensor)";

/// Register both entry points against the shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(K_TENSOR_F16_D256, FLASH_ATTN_PREFILL_TENSOR_SHADER_SOURCE);
    registry.register_source(K_TENSOR_F16_D512, FLASH_ATTN_PREFILL_TENSOR_SHADER_SOURCE);
}

/// Queries per threadgroup (the kernel's `Q`, the blk `bq`) for a head dim.
pub fn queries_per_threadgroup(head_dim: u32) -> Result<u32> {
    match head_dim {
        256 => Ok(32),
        512 => Ok(16),
        _ => Err(MlxError::InvalidArgument(format!(
            "flash_attn_prefill_tensor: unsupported head_dim {head_dim} (256 or 512)"
        ))),
    }
}

/// `kv_len` rounded up to a multiple of [`CACHE_ROWS_PER_STEP`].
pub fn kv_pad(kv_len: u32) -> u32 {
    kv_len.div_ceil(CACHE_ROWS_PER_STEP) * CACHE_ROWS_PER_STEP
}

fn kernel_name(head_dim: u32) -> Result<&'static str> {
    match head_dim {
        256 => Ok(K_TENSOR_F16_D256),
        512 => Ok(K_TENSOR_F16_D512),
        _ => Err(MlxError::InvalidArgument(format!(
            "flash_attn_prefill_tensor: unsupported head_dim {head_dim} (256 or 512)"
        ))),
    }
}

/// Whether the tensor pipeline for `head_dim` builds on this device. `Ok(false)`
/// only when the toolchain or device lacks the Metal tensor API; any other
/// compile failure is an error.
pub fn flash_attn_prefill_tensor_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    head_dim: u32,
) -> Result<bool> {
    let name = kernel_name(head_dim)?;
    Ok(registry
        .probe_optional_pipeline(
            name,
            device.metal_device(),
            device.registry_id(),
            |error| crate::ops::dense_mm_capability::is_unavailable_tensor_api(device, error),
        )?
        .available)
}

/// Shapes and strides of one dispatch. Strides are in elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlashAttnPrefillTensorParams {
    pub rows: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Cache rows the kernel walks; a multiple of [`CACHE_ROWS_PER_STEP`].
    pub kv_pad: u32,
    pub q_row_stride: u32,
    pub q_head_stride: u32,
    pub k_row_stride: u32,
    pub k_head_stride: u32,
    pub v_row_stride: u32,
    pub v_head_stride: u32,
    /// Pre-softmax scale folded into Q (1.0 for Gemma 4).
    pub scale: f32,
}

/// Rust mirror of the MSL `FlashAttnPrefillTensorParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnPrefillTensorParamsGpu {
    rows: i32,
    n_heads: i32,
    n_kv_heads: i32,
    kv_pad: i32,
    q_row_stride: i32,
    q_head_stride: i32,
    k_row_stride: i32,
    k_head_stride: i32,
    v_row_stride: i32,
    v_head_stride: i32,
    mask_row_stride: i32,
    scale: f32,
}

/// Threadgroup memory of one dispatch: queries (half), scores (float),
/// probabilities (half), row scale (float), rescale flag (int), padded to 16.
pub fn threadgroup_memory_bytes(head_dim: u32) -> Result<u64> {
    let q = queries_per_threadgroup(head_dim)? as u64;
    let c = CACHE_ROWS_PER_STEP as u64;
    let raw = q * head_dim as u64 * 2 + q * c * (4 + 2) + q * 4 + 4;
    Ok(raw.div_ceil(16) * 16)
}

fn invalid(msg: String) -> MlxError {
    MlxError::InvalidArgument(format!("flash_attn_prefill_tensor: {msg}"))
}

fn need_elems(
    name: &str,
    buf: &MlxBuffer,
    dtype: DType,
    last_row: u64,
    row: u64,
    last_head: u64,
    head: u64,
    width: u64,
) -> Result<()> {
    if buf.dtype() != dtype {
        return Err(invalid(format!(
            "{name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    let elems = last_row * row + last_head * head + width;
    let bytes = elems * dtype.size_of() as u64;
    if (buf.data_byte_len() as u64) < bytes {
        return Err(invalid(format!(
            "{name} holds {} bytes, the strides need {bytes}",
            buf.data_byte_len()
        )));
    }
    Ok(())
}

/// Encode one tensor flash-attention dispatch.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_attn_prefill_tensor_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: &MlxBuffer,
    blk: &MlxBuffer,
    out: &MlxBuffer,
    p: &FlashAttnPrefillTensorParams,
) -> Result<()> {
    let name = kernel_name(p.head_dim)?;
    let bq = queries_per_threadgroup(p.head_dim)?;
    if p.rows == 0 || p.n_heads == 0 || p.n_kv_heads == 0 || p.kv_pad == 0 {
        return Err(invalid(format!("empty shape {p:?}")));
    }
    if p.n_heads % p.n_kv_heads != 0 {
        return Err(invalid(format!(
            "n_heads {} is not a multiple of n_kv_heads {}",
            p.n_heads, p.n_kv_heads
        )));
    }
    if p.kv_pad % CACHE_ROWS_PER_STEP != 0 {
        return Err(invalid(format!(
            "kv_pad {} is not a multiple of {CACHE_ROWS_PER_STEP}",
            p.kv_pad
        )));
    }
    let i32_ok = [
        p.rows,
        p.n_heads,
        p.n_kv_heads,
        p.kv_pad,
        p.q_row_stride,
        p.q_head_stride,
        p.k_row_stride,
        p.k_head_stride,
        p.v_row_stride,
        p.v_head_stride,
    ]
    .iter()
    .all(|&x| x <= i32::MAX as u32);
    if !i32_ok {
        return Err(invalid(format!("a dimension exceeds i32: {p:?}")));
    }
    let (rows, hd, kvp) = (p.rows as u64, p.head_dim as u64, p.kv_pad as u64);
    let (nh, nkv) = (p.n_heads as u64, p.n_kv_heads as u64);
    if p.q_row_stride % 4 != 0 || p.q_head_stride % 4 != 0 {
        return Err(invalid(
            "Q strides must be multiples of 4 (float4 loads)".into(),
        ));
    }
    need_elems(
        "q",
        q,
        DType::F32,
        rows - 1,
        p.q_row_stride as u64,
        nh - 1,
        p.q_head_stride as u64,
        hd,
    )?;
    need_elems(
        "k",
        k,
        DType::F16,
        kvp - 1,
        p.k_row_stride as u64,
        nkv - 1,
        p.k_head_stride as u64,
        hd,
    )?;
    need_elems(
        "v",
        v,
        DType::F16,
        kvp - 1,
        p.v_row_stride as u64,
        nkv - 1,
        p.v_head_stride as u64,
        hd,
    )?;
    need_elems("mask", mask, DType::F16, rows - 1, kvp, 0, 0, kvp)?;
    need_elems("out", out, DType::F32, rows - 1, nh * hd, nh - 1, hd, hd)?;
    let blk_bytes = (rows.div_ceil(bq as u64)) * (kvp / CACHE_ROWS_PER_STEP as u64);
    if blk.dtype() != DType::U8 || (blk.data_byte_len() as u64) < blk_bytes {
        return Err(invalid(format!(
            "blk must be U8 with at least {blk_bytes} bytes (bq {bq}, bk {CACHE_ROWS_PER_STEP}), got {} {} bytes",
            blk.dtype(),
            blk.data_byte_len()
        )));
    }

    let gpu = FlashAttnPrefillTensorParamsGpu {
        rows: p.rows as i32,
        n_heads: p.n_heads as i32,
        n_kv_heads: p.n_kv_heads as i32,
        kv_pad: p.kv_pad as i32,
        q_row_stride: p.q_row_stride as i32,
        q_head_stride: p.q_head_stride as i32,
        k_row_stride: p.k_row_stride as i32,
        k_head_stride: p.k_head_stride as i32,
        v_row_stride: p.v_row_stride as i32,
        v_head_stride: p.v_head_stride as i32,
        mask_row_stride: p.kv_pad as i32,
        scale: p.scale,
    };
    let pipeline = registry
        .get_pipeline(name, device.metal_device())
        .map_err(|error| typed_unavailable(device, FEATURE, error))?;
    if pipeline.max_total_threads_per_threadgroup() < 32 * SIMDGROUPS {
        return Err(invalid(format!(
            "{name}: pipeline allows {} threads per threadgroup, needs {}",
            pipeline.max_total_threads_per_threadgroup(),
            32 * SIMDGROUPS
        )));
    }
    let smem = threadgroup_memory_bytes(p.head_dim)?;
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k)),
            (3, KernelArg::Buffer(v)),
            (4, KernelArg::Buffer(mask)),
            (5, KernelArg::Buffer(blk)),
            (6, KernelArg::Buffer(out)),
        ],
        &[(0, smem)],
        MTLSize::new(rows.div_ceil(bq as u64), nh, 1),
        MTLSize::new(32, SIMDGROUPS, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_mirror_is_twelve_words() {
        assert_eq!(std::mem::size_of::<FlashAttnPrefillTensorParamsGpu>(), 48);
    }

    #[test]
    fn threadgroup_memory_matches_the_peer_formula_and_fits_32k() {
        // GGML_PAD(Q*DK*2 + Q*C*(4+2) + Q*4 + 4, 16)
        assert_eq!(threadgroup_memory_bytes(256).unwrap(), 28816);
        assert_eq!(threadgroup_memory_bytes(512).unwrap(), 22608);
        assert!(threadgroup_memory_bytes(256).unwrap() <= 32 * 1024);
        assert!(queries_per_threadgroup(128).is_err());
    }

    #[test]
    fn kv_pad_rounds_up_to_the_step() {
        assert_eq!(kv_pad(1), 64);
        assert_eq!(kv_pad(64), 64);
        assert_eq!(kv_pad(1281), 1344);
    }
}
