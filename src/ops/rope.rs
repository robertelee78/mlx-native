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
