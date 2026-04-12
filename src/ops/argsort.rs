//! GPU-accelerated argsort (descending) for MoE top-K routing.
//!
//! Sorts indices by value in descending order using a bitonic sort kernel.
//! For MoE with N <= 128 experts per row, this fits in a single threadgroup.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_threadgroups_with_args, KernelArg};

/// MSL source for the argsort kernel (embedded at compile time).
pub static ARGSORT_SHADER_SOURCE: &str = include_str!("../shaders/argsort.metal");

/// Register argsort shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("argsort_desc_f32", ARGSORT_SHADER_SOURCE);
}

/// MSL-compatible params struct for argsort.
///
/// Must match `ArgsortParams` in `argsort.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuArgsortParams {
    row_len: u32,
    batch_size: u32,
}

/// Dispatch an argsort (descending) operation on the GPU.
///
/// For each row of `input`, produces a permutation of indices `[0..row_len)`
/// such that `input[row][output[row][0]] >= input[row][output[row][1]] >= ...`.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have `argsort_desc_f32` registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[batch_size, row_len]` (f32).
/// * `output`     - Output buffer of shape `[batch_size, row_len]` (u32) — sorted indices.
/// * `batch_size` - Number of rows.
/// * `row_len`    - Number of elements per row (must be <= 256).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `row_len` is 0 or > 256
/// - `batch_size` is 0
/// - Buffers are too small
#[allow(clippy::too_many_arguments)]
pub fn dispatch_argsort_desc_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    batch_size: u32,
    row_len: u32,
) -> Result<()> {
    if row_len == 0 {
        return Err(MlxError::InvalidArgument(
            "argsort_desc_f32: row_len must be > 0".into(),
        ));
    }
    if row_len > 256 {
        return Err(MlxError::InvalidArgument(format!(
            "argsort_desc_f32: row_len {} exceeds max 256 (shared memory limit)",
            row_len
        )));
    }
    if batch_size == 0 {
        return Err(MlxError::InvalidArgument(
            "argsort_desc_f32: batch_size must be > 0".into(),
        ));
    }

    let total = batch_size as usize * row_len as usize;
    let input_bytes = total * 4; // f32
    if input.byte_len() < input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "argsort_desc_f32: input buffer too small: need {} bytes, have {}",
            input_bytes,
            input.byte_len()
        )));
    }
    let output_bytes = total * 4; // u32
    if output.byte_len() < output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "argsort_desc_f32: output buffer too small: need {} bytes, have {}",
            output_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("argsort_desc_f32", device)?;

    let gpu_params = GpuArgsortParams {
        row_len,
        batch_size,
    };

    // One threadgroup per row, threadgroup size = next power of two of row_len.
    let tg_size = std::cmp::min(256, row_len.next_power_of_two()) as u64;

    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(batch_size as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
