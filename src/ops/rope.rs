//! Rotary Position Embedding (RoPE) GPU dispatch.
//!
//! Applies rotation in pairs of elements using cos/sin of
//! `position * theta^(-2i/d)`.  Gemma 4 uses theta=10000 for sliding
//! attention layers and theta=1000000 for global attention layers.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the RoPE kernels (embedded at compile time).
pub static ROPE_SHADER_SOURCE: &str = include_str!("../shaders/rope.metal");

/// Register RoPE shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("rope_f32", ROPE_SHADER_SOURCE);
    registry.register_source("rope_f16", ROPE_SHADER_SOURCE);
    registry.register_source("rope_bf16", ROPE_SHADER_SOURCE);
    registry.register_source("rope_neox_bf16", ROPE_SHADER_SOURCE);
    registry.register_source("rope_neox_f32", ROPE_SHADER_SOURCE);
}

/// Dispatch a RoPE operation on the GPU.
///
/// # Arguments
///
/// * `encoder`      - Command encoder to record the dispatch into.
/// * `registry`     - Kernel registry (must have RoPE sources registered).
/// * `device`       - Metal device for pipeline compilation.
/// * `input`        - Input buffer of shape `[seq_len, head_dim]` (f32 or f16).
/// * `output`       - Output buffer (same dtype and shape as input).
/// * `params_buf`   - Params buffer containing `[theta, head_dim, 0, 0]` as f32.
/// * `positions_buf` - Positions buffer containing `[pos_0, pos_1, ...]` as u32.
/// * `seq_len`      - Number of sequence positions.
/// * `head_dim`     - Dimension of each head (must be even).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32 or f16.
/// - head_dim is not even.
/// - Input and output element counts do not match.
pub fn dispatch_rope(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    positions_buf: &MlxBuffer,
    seq_len: u32,
    head_dim: u32,
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE head_dim must be even, got {}",
            head_dim
        )));
    }
    if head_dim == 0 || seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "RoPE head_dim and seq_len must be > 0".into(),
        ));
    }

    let expected_elements = (seq_len as usize) * (head_dim as usize);
    if input.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE input element count {} != seq_len({}) * head_dim({})",
            input.element_count(),
            seq_len,
            head_dim
        )));
    }
    if output.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE output element count {} != seq_len({}) * head_dim({})",
            output.element_count(),
            seq_len,
            head_dim
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "rope_f32",
        DType::F16 => "rope_f16",
        DType::BF16 => "rope_bf16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "RoPE unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let half_dim = head_dim / 2;

    // Grid: (half_dim, seq_len) — one thread per pair per position
    // Threadgroup: use a reasonable size for the pair dimension
    let tg_x = std::cmp::min(64, half_dim as u64);
    let tg_y = std::cmp::min(4, seq_len as u64);

    encoder.encode(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
            (3, positions_buf),
        ],
        MTLSize::new(half_dim as u64, seq_len as u64, 1),
        MTLSize::new(tg_x, tg_y, 1),
    );

    Ok(())
}

/// GPU params for the neox RoPE kernel's auxiliary params buffer.
///
/// Must match the uint array in `rope_neox_bf16` buffer(4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRopeNeoxParams {
    n_heads: u32,
    _pad: u32,
}

/// Dispatch a Neox/split-convention RoPE operation on the GPU (bf16 only).
///
/// The Neox convention pairs `(d[i], d[i + half_rope_dim])` instead of
/// `(d[2i], d[2i+1])`.  Supports partial rotary where only the first
/// `rope_dim` dimensions are rotated.
///
/// # Arguments
///
/// * `encoder`       - Command encoder to record the dispatch into.
/// * `registry`      - Kernel registry (must have rope_neox_bf16 registered).
/// * `device`        - Metal device for pipeline compilation.
/// * `input`         - Input buffer of shape `[seq_len * n_heads, head_dim]` (bf16).
/// * `output`        - Output buffer (same shape and dtype as input).
/// * `params_buf`    - Params buffer containing `[theta, head_dim, rope_dim, 0]` as f32.
/// * `positions_buf` - Positions buffer containing `[pos_0, pos_1, ...]` as u32 (length = seq_len).
/// * `seq_len`       - Number of sequence positions.
/// * `n_heads`       - Number of attention heads.
/// * `head_dim`      - Dimension of each head.
/// * `rope_dim`      - Number of dimensions to rotate (must be even, <= head_dim).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_neox_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    positions_buf: &MlxBuffer,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
) -> Result<()> {
    use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

    if rope_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox rope_dim must be even, got {}",
            rope_dim
        )));
    }
    if rope_dim > head_dim {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox rope_dim ({}) must be <= head_dim ({})",
            rope_dim, head_dim
        )));
    }
    if head_dim == 0 || seq_len == 0 || n_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "RoPE neox head_dim, seq_len, and n_heads must be > 0".into(),
        ));
    }

    let n_rows = (seq_len as usize) * (n_heads as usize);
    let expected_elements = n_rows * (head_dim as usize);
    if input.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox input element count {} != seq_len({}) * n_heads({}) * head_dim({})",
            input.element_count(),
            seq_len,
            n_heads,
            head_dim
        )));
    }
    if output.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox output element count {} != seq_len({}) * n_heads({}) * head_dim({})",
            output.element_count(),
            seq_len,
            n_heads,
            head_dim
        )));
    }

    let pipeline = registry.get_pipeline("rope_neox_bf16", device)?;
    let half_rope = rope_dim / 2;

    let gpu_rope_params = GpuRopeNeoxParams {
        n_heads,
        _pad: 0,
    };

    // Grid: (half_rope, n_rows) — one thread per pair per row
    let tg_x = std::cmp::min(64, half_rope as u64);
    let tg_y = std::cmp::min(4, n_rows as u64);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(params_buf)),
            (3, KernelArg::Buffer(positions_buf)),
            (4, KernelArg::Bytes(as_bytes(&gpu_rope_params))),
        ],
        MTLSize::new(half_rope as u64, n_rows as u64, 1),
        MTLSize::new(tg_x, tg_y, 1),
    );

    Ok(())
}

/// GPU params for the neox f32 RoPE kernel with freq_factors support.
///
/// Must match the uint array in `rope_neox_f32` buffer(4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRopeNeoxF32Params {
    n_heads: u32,
    has_freq_factors: u32,
}

/// Dispatch a Neox/split-convention RoPE operation on the GPU (f32) with
/// optional freq_factors support.
///
/// The Neox convention pairs `(d[i], d[i + half_rope_dim])` instead of
/// `(d[2i], d[2i+1])`.  Supports partial rotary where only the first
/// `rope_dim` dimensions are rotated.
///
/// When `freq_factors` is `Some`, each pair's base frequency is divided by
/// `freq_factors[pair_idx]`.  Gemma 4's global attention layers use this to
/// mask out rotation for certain dimensions (freq_factor=1e30 -> identity).
///
/// # Arguments
///
/// * `encoder`       - Command encoder to record the dispatch into.
/// * `registry`      - Kernel registry (must have rope_neox_f32 registered).
/// * `device`        - Metal device for pipeline compilation.
/// * `input`         - Input buffer of shape `[seq_len * n_heads, head_dim]` (f32).
/// * `output`        - Output buffer (same shape and dtype as input).
/// * `params_buf`    - Params buffer containing `[theta, head_dim, rope_dim, 0]` as f32.
/// * `positions_buf` - Positions buffer containing `[pos_0, pos_1, ...]` as u32 (length = seq_len).
/// * `freq_factors`  - Optional freq_factors buffer of shape `[rope_dim/2]` (f32).
///                     Pass `None` for standard RoPE (equivalent to all-ones).
/// * `seq_len`       - Number of sequence positions.
/// * `n_heads`       - Number of attention heads.
/// * `head_dim`      - Dimension of each head.
/// * `rope_dim`      - Number of dimensions to rotate (must be even, <= head_dim).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rope_neox_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
) -> Result<()> {
    use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

    if rope_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox f32 rope_dim must be even, got {}",
            rope_dim
        )));
    }
    if rope_dim > head_dim {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox f32 rope_dim ({}) must be <= head_dim ({})",
            rope_dim, head_dim
        )));
    }
    if head_dim == 0 || seq_len == 0 || n_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "RoPE neox f32 head_dim, seq_len, and n_heads must be > 0".into(),
        ));
    }

    let n_rows = (seq_len as usize) * (n_heads as usize);
    let expected_elements = n_rows * (head_dim as usize);
    if input.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox f32 input element count {} != seq_len({}) * n_heads({}) * head_dim({})",
            input.element_count(),
            seq_len,
            n_heads,
            head_dim
        )));
    }
    if output.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE neox f32 output element count {} != seq_len({}) * n_heads({}) * head_dim({})",
            output.element_count(),
            seq_len,
            n_heads,
            head_dim
        )));
    }

    let pipeline = registry.get_pipeline("rope_neox_f32", device)?;
    let half_rope = rope_dim / 2;

    let has_ff = freq_factors.is_some();
    let gpu_rope_params = GpuRopeNeoxF32Params {
        n_heads,
        has_freq_factors: u32::from(has_ff),
    };

    // When no freq_factors buffer is provided, bind the input buffer as a
    // harmless dummy at buffer(5) — Metal requires all declared buffers to
    // be bound. The shader checks has_freq_factors before reading.
    let ff_buf = freq_factors.unwrap_or(input);

    // Grid: (half_rope, n_rows) — one thread per pair per row
    let tg_x = std::cmp::min(64, half_rope as u64);
    let tg_y = std::cmp::min(4, n_rows as u64);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(params_buf)),
            (3, KernelArg::Buffer(positions_buf)),
            (4, KernelArg::Bytes(as_bytes(&gpu_rope_params))),
            (5, KernelArg::Buffer(ff_buf)),
        ],
        MTLSize::new(half_rope as u64, n_rows as u64, 1),
        MTLSize::new(tg_x, tg_y, 1),
    );

    Ok(())
}
