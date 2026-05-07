//! Backward pass for row-wise softmax.
//!
//! Given `y = softmax(x)` along the last dim and upstream gradient
//! `dy` of the same shape, computes
//!
//!   `dx[b, i] = y[b, i] · (dy[b, i] − Σ_j y[b, j] · dy[b, j])`
//!
//! Companion to [`crate::ops::softmax::dispatch_softmax`].  Used by
//! reverse-mode autograd in hf2q's calibrate module (ADR-020 Track 1).
//!
//! Threadgroup-per-row layout matches softmax forward: one threadgroup
//! processes one full row, doing a tree reduction over the columns.

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use metal::MTLSize;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "softmax_backward_f32",
        include_str!("../shaders/softmax_backward.metal"),
    );
}

/// Encode the softmax backward kernel.
///
/// # Arguments
///
/// * `encoder`    — Command encoder.
/// * `registry`   — Kernel registry (must have softmax_backward source registered).
/// * `device`     — Metal device.
/// * `y`          — Forward softmax output `[rows, cols]`, f32.
/// * `dy`         — Upstream gradient `[rows, cols]`, f32.
/// * `dx`         — Output gradient `[rows, cols]`, f32 (must be pre-allocated).
/// * `params_buf` — Params buffer containing `[cols, 0]` as f32.
/// * `rows`       — Row count (one threadgroup per row).
/// * `cols`       — Column count.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if shapes are inconsistent or
/// any buffer is too small.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_softmax_backward(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    y: &MlxBuffer,
    dy: &MlxBuffer,
    dx: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(MlxError::InvalidArgument(
            "softmax_backward: rows and cols must be > 0".into(),
        ));
    }
    let expected = (rows as usize) * (cols as usize);
    for (label, buf) in [("y", y), ("dy", dy), ("dx", dx)] {
        if buf.element_count() != expected {
            return Err(MlxError::InvalidArgument(format!(
                "softmax_backward: {label} element count {} != rows({}) * cols({})",
                buf.element_count(),
                rows,
                cols
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "softmax_backward: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "softmax_backward: params_buf too small (need 8 bytes for float2, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("softmax_backward_f32", device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2
    // for the tree reduction (matches softmax forward convention).
    let tg_size = std::cmp::min(256, cols.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, y), (1, dy), (2, dx), (3, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
