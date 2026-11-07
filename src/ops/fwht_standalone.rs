//! Standalone Fast Walsh-Hadamard Transform dispatch (SIMD shuffle, zero barriers).
//!
//! Pre-rotates Q before TurboQuant SDPA and inverse-rotates the output.
//! FWHT is self-inverse, so the same kernel handles both directions.
//!
//! Uses 1 simdgroup (32 threads) per head with SIMD shuffle for the butterfly.
//! Zero threadgroup barriers.
//!
//! ADR-007 iter-14 D1 SRHT additions:
//! - `dispatch_fwht_sign_premult_f32`: sign * data → FWHT → normalize (for Q pre-rotation with D1).
//! - `dispatch_fwht_sign_undo_f32`: FWHT → normalize → sign * data (for output inverse FWHT + sign undo).
//! - `dispatch_fwht_sign_undo_sigmoid_mul_f32`: the undo and the attention-gate `sigmoid_mul_f32` in one dispatch.

use metal::foreign_types::ForeignType;
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the standalone FWHT kernel.
pub static FWHT_STANDALONE_SHADER_SOURCE: &str =
    include_str!("../shaders/fwht_standalone.metal");

/// GPU params — must match `FwhtParams` in fwht_standalone.metal.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFwhtParams {
    head_dim: u32,
    num_heads: u32,
}

/// Dispatch the standalone FWHT on `[num_heads, head_dim]` F32 data (in-place).
///
/// 1 simdgroup (32 threads) per head. Zero threadgroup barriers.
pub fn dispatch_fwht_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    data: &MlxBuffer,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let kernel_name = match head_dim {
        256 => "fwht_standalone_f32_d256",
        512 => "fwht_standalone_f32_d512",
        _ => return Err(MlxError::InvalidArgument(
            format!("fwht_standalone: unsupported head_dim={}", head_dim),
        )),
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let params = GpuFwhtParams { head_dim, num_heads };

    // 1 simdgroup (32 threads) per head — no shared memory needed.
    let threadgroups = MTLSize::new(num_heads as u64, 1, 1);
    let threads_per_tg = MTLSize::new(32, 1, 1);

    use crate::ops::encode_helpers::{as_bytes, KernelArg};
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(data)),
            (1, KernelArg::Bytes(as_bytes(&params))),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Dispatch FWHT with D1 sign pre-mult (in-place).
///
/// Applies sign * data BEFORE the FWHT transform.
/// Used for Q pre-rotation: sign_j * Q_j → FWHT → normalize.
/// ADR-007 iter-14: mirrors AmesianX cpy-utils.cuh:180-183 sign application.
pub fn dispatch_fwht_sign_premult_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    data: &MlxBuffer,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let kernel_name = match head_dim {
        256 => "fwht_sign_premult_f32_d256",
        512 => "fwht_sign_premult_f32_d512",
        _ => return Err(MlxError::InvalidArgument(
            format!("fwht_sign_premult: unsupported head_dim={}", head_dim),
        )),
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let params = GpuFwhtParams { head_dim, num_heads };
    let threadgroups = MTLSize::new(num_heads as u64, 1, 1);
    let threads_per_tg = MTLSize::new(32, 1, 1);

    use crate::ops::encode_helpers::{as_bytes, KernelArg};
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(data)),
            (1, KernelArg::Bytes(as_bytes(&params))),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Dispatch FWHT with D1 sign undo (in-place).
///
/// Applies the FWHT → normalize → sign_j * elem_j.
/// Used for output inverse rotation: recovers V_weighted from sign * V_weighted
/// that is produced by IWHT of accumulated rotated-domain V.
/// ADR-007 iter-14: sign is self-inverse (sign*sign=1), so same table undoes encode.
pub fn dispatch_fwht_sign_undo_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    data: &MlxBuffer,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let kernel_name = match head_dim {
        256 => "fwht_sign_undo_f32_d256",
        512 => "fwht_sign_undo_f32_d512",
        _ => return Err(MlxError::InvalidArgument(
            format!("fwht_sign_undo: unsupported head_dim={}", head_dim),
        )),
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let params = GpuFwhtParams { head_dim, num_heads };
    let threadgroups = MTLSize::new(num_heads as u64, 1, 1);
    let threads_per_tg = MTLSize::new(32, 1, 1);

    use crate::ops::encode_helpers::{as_bytes, KernelArg};
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(data)),
            (1, KernelArg::Bytes(as_bytes(&params))),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Dispatch [`dispatch_fwht_sign_undo_f32`] and `sigmoid_mul_f32` as one
/// kernel: `output = undo(data) * sigmoid(gate)` over `num_heads` heads.
///
/// Each element runs both kernels' expressions in their order, so the result
/// is bitwise the undo followed by the sigmoid multiply, and a head's result
/// does not depend on `num_heads`. `data` is only read. `data`, `gate` and
/// `output` hold `num_heads * head_dim` F32 values; `output` must not overlap
/// `data` or `gate`. Only `head_dim == 256` is instantiated.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fwht_sign_undo_sigmoid_mul_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    data: &MlxBuffer,
    gate: &MlxBuffer,
    output: &MlxBuffer,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    const OP: &str = "fwht_sign_undo_sigmoid_mul";
    let kernel_name = match head_dim {
        256 => "fwht_sign_undo_sigmoid_mul_f32_d256",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "{OP}: unsupported head_dim={head_dim}"
            )))
        }
    };
    let elements = num_heads as usize * head_dim as usize;
    for (name, buffer) in [("data", data), ("gate", gate), ("output", output)] {
        if buffer.dtype() != DType::F32 || buffer.element_count() < elements {
            return Err(MlxError::InvalidArgument(format!(
                "{OP}: {name} must be F32 with at least {elements} elements"
            )));
        }
    }
    let bytes = elements as u64 * 4;
    let range = |buffer: &MlxBuffer| {
        let start = buffer.byte_offset();
        (buffer.metal_buffer().as_ptr() as usize, start, start + bytes)
    };
    let out = range(output);
    for (name, other) in [("data", range(data)), ("gate", range(gate))] {
        if out.0 == other.0 && out.1 < other.2 && other.1 < out.2 {
            return Err(MlxError::InvalidArgument(format!(
                "{OP}: output overlaps {name}"
            )));
        }
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let params = GpuFwhtParams { head_dim, num_heads };
    use crate::ops::encode_helpers::{as_bytes, KernelArg};
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(data)),
            (1, KernelArg::Buffer(gate)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    Ok(())
}
