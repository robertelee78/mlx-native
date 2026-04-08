//! Fused residual addition + RMS normalization GPU dispatch (bf16).
//!
//! Replaces two separate dispatches — `elementwise_add_bf16` followed by
//! `rms_norm_bf16` — with a single kernel launch per transformer sub-layer.
//! Saves approximately 136 kernel dispatches per Gemma 4 forward pass.
//!
//! The Metal kernel reads each row once, computes the elementwise sum,
//! optionally writes the un-normed sum for subsequent residual use,
//! reduces for RMS, then normalizes and writes the output — the intermediate
//! summed buffer is never materialized in GPU memory between the two ops.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_threadgroups_with_args_and_shared, KernelArg};

/// MSL source embedded at compile time.
pub static FUSED_RESIDUAL_NORM_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_residual_norm_bf16.metal");

/// Register the fused residual-norm shader with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "fused_residual_norm_bf16",
        FUSED_RESIDUAL_NORM_SHADER_SOURCE,
    );
}

/// GPU params struct — must match `FusedResidualNormParams` in the MSL shader exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedResidualNormParams {
    dim:       u32,
    rows:      u32,
    eps:       f32,
    write_sum: u32, // 0 = do not write sum_output, nonzero = write it
}

/// Dispatch a fused residual addition + RMS normalization operation.
///
/// Computes `normed = rms_norm(residual + input, weight, eps)` in a single pass.
/// Optionally also writes the un-normalized sum `residual + input` to `sum_output`
/// so the caller can use it as the residual stream for the next layer without a
/// second elementwise kernel.
///
/// # Arguments
///
/// * `encoder`        - Command encoder to record the dispatch into.
/// * `registry`       - Kernel registry (must have fused_residual_norm_bf16 registered).
/// * `device`         - Metal device for pipeline compilation.
/// * `residual`       - bf16 buffer of shape `[rows, dim]` — the residual stream.
/// * `input`          - bf16 buffer of shape `[rows, dim]` — the sublayer output to add.
/// * `weight`         - bf16 buffer of shape `[dim]` — RMS norm learned scale.
/// * `normed_output`  - bf16 output buffer of shape `[rows, dim]` — normalized result.
/// * `sum_output`     - Optional bf16 output buffer of shape `[rows, dim]`.  When
///                      `Some`, the un-normed sum is written here (for use as the next
///                      layer's residual).  When `None`, that write is skipped.
/// * `rows`           - Number of rows (tokens) in the batch.
/// * `dim`            - Hidden dimension (last axis size).
/// * `eps`            - RMS normalization epsilon (1e-6 for Gemma 4).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_residual_norm_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    normed_output: &MlxBuffer,
    sum_output: Option<&MlxBuffer>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_residual_norm: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if residual.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_residual_norm: residual element count {} != rows({}) * dim({})",
            residual.element_count(),
            rows,
            dim,
        )));
    }
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_residual_norm: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim,
        )));
    }
    if normed_output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_residual_norm: normed_output element count {} != rows({}) * dim({})",
            normed_output.element_count(),
            rows,
            dim,
        )));
    }
    if let Some(sum_buf) = sum_output {
        if sum_buf.element_count() != expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_residual_norm: sum_output element count {} != rows({}) * dim({})",
                sum_buf.element_count(),
                rows,
                dim,
            )));
        }
    }

    let pipeline = registry.get_pipeline("fused_residual_norm_bf16", device)?;

    // One threadgroup per row; size is the smallest power-of-two >= dim,
    // capped at 256.
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;

    // Shared memory: tg_size f32 values — used first to cache element sums,
    // then reused for the parallel sum-of-squares reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    let write_sum = sum_output.is_some();
    let gpu_params = GpuFusedResidualNormParams {
        dim,
        rows,
        eps,
        write_sum: u32::from(write_sum),
    };

    // When sum_output is None we still bind a dummy at buffer 4 to satisfy
    // Metal's requirement that all declared buffers are bound.  The shader
    // checks write_sum before accessing the pointer so no data is written.
    let sum_buf = sum_output.unwrap_or(normed_output);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(normed_output)),
            (4, KernelArg::Buffer(sum_buf)),
            (5, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
