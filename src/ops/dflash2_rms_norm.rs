//! Source-ordered DFlash2 RMS normalization with a completed BF16 boundary.
//!
//! The scalar-column accumulation, SIMD reduction and eight ordered partial
//! additions are part of the operation, not interchangeable with a float4 dot
//! reduction. F32 storage carries the caller's values unchanged; only the
//! completed normalized result is rounded to BF16 and widened back to F32.
//! Generic RMS dispatch and its F32/float4 arithmetic are unaffected.
//!
//! Association follows Splash commit 031a25a6125af2d7acf7096580afe56eb33d65bc;
//! see the shader attribution and `docs/peer-benchmarks.md` for evidence.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::{CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub const KERNEL: &str = "dflash2_rms_norm_bf16";
pub static SOURCE: &str = include_str!("../shaders/dflash2_rms_norm.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(KERNEL, SOURCE);
}

/// Normalize `rows` of `dim` values with source-ordered FP32 arithmetic and a
/// BF16 result boundary, retained in F32 storage. No input/weight rounding,
/// conversion buffers, environment switch or alternate precision is added.
///
/// Input/output expose at least `rows * dim` F32 values and weight at least
/// `dim`. `params[0]` supplies epsilon; any further elements (including the
/// generic norm ABI's dimension) are ignored. The checked `dim` is bound inline.
/// Extra output capacity is untouched. Views use their own window, not their
/// backing root's capacity. Zero geometry and invalid storage reject.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_rms_norm_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    // Bound the scalar-column loop and keep all supported widths exactly F32.
    if rows == 0 || dim == 0 || dim > (1 << 24) {
        return Err(MlxError::InvalidArgument(
            "DFlash2 RMS requires positive rows and dim <= 2^24".into(),
        ));
    }
    let bytes = (rows as usize)
        .checked_mul(dim as usize)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 RMS size overflow".into()))?;
    for (name, buffer, required) in [
        ("input", input, bytes),
        ("weight", weight, dim as usize * std::mem::size_of::<f32>()),
        ("output", output, bytes),
        ("params", params, std::mem::size_of::<f32>()),
    ] {
        if buffer.dtype() != DType::F32
            || buffer.byte_offset() % std::mem::align_of::<f32>() as u64 != 0
            || buffer.data_byte_len() < required
        {
            return Err(MlxError::InvalidArgument(format!(
                "DFlash2 RMS {name} requires F32 and at least {required} bytes in its aligned view"
            )));
        }
    }
    let pipeline = registry.get_pipeline(KERNEL, device)?;
    if pipeline.thread_execution_width() != 32
        || pipeline.max_total_threads_per_threadgroup() < 256
        || pipeline.static_threadgroup_memory_length() > device.max_threadgroup_memory_length()
    {
        return Err(MlxError::InvalidArgument(
            "DFlash2 RMS requires eight 32-thread SIMD groups and its static partial storage".into(),
        ));
    }
    // Never fuse through the completed BF16 producer boundary.
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Buffer(params)),
            (4, KernelArg::Bytes(bytemuck::bytes_of(&dim))),
        ],
        MTLSize::new(u64::from(rows), 1, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}
