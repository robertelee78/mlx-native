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

use metal::MTLSize;

use crate::buffer::MlxBuffer;
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
