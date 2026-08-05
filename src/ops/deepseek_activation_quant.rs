//! DeepSeek-V4 activation simulation at the official QAT boundaries.
//!
//! Main KV rows use block-64 E4M3 values with power-of-two E8M0 scales on
//! their non-RoPE prefix. Ratio-four indexer rows use a normalized Hadamard
//! rotation followed by block-32 E2M1 values with E8M0 scales. Both kernels
//! quantize and immediately dequantize in place to BF16, matching inference.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::encode_threadgroups_with_args_and_shared;

pub const DEEPSEEK_MXFP8_KERNEL: &str = "deepseek_mxfp8_fake_quant_bf16";
pub const DEEPSEEK_HADAMARD_MXFP4_KERNEL: &str = "deepseek_hadamard_mxfp4_bf16";
pub const DEEPSEEK_MAIN_WIDTH: usize = 512;
pub const DEEPSEEK_MAIN_QUANTIZED_WIDTH: usize = 448;
pub const DEEPSEEK_INDEX_WIDTH: usize = 128;

pub static DEEPSEEK_ACTIVATION_QUANT_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_activation_quant.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_MXFP8_KERNEL,
        DEEPSEEK_ACTIVATION_QUANT_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_HADAMARD_MXFP4_KERNEL,
        DEEPSEEK_ACTIVATION_QUANT_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekMxfp8Params {
    pub rows: u32,
    pub row_width: u32,
    pub quantized_width: u32,
    pub block_size: u32,
}

fn validate_bf16_rows(buffer: &MlxBuffer, rows: usize, width: usize, label: &str) -> Result<()> {
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| MlxError::InvalidArgument(format!("{label}: rows times width overflows")))?;
    if buffer.dtype() != DType::BF16 || buffer.element_count() != elements {
        return Err(MlxError::InvalidArgument(format!(
            "{label}: buffer must be BF16 with {elements} elements, got {} {:?}",
            buffer.dtype(),
            buffer.shape()
        )));
    }
    Ok(())
}

/// In-place block-64 E4M3/E8M0 fake quantization of a BF16 row prefix.
pub fn dispatch_deepseek_mxfp8_fake_quant_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    data: &MlxBuffer,
    params: &DeepSeekMxfp8Params,
) -> Result<()> {
    if params.rows == 0
        || params.row_width == 0
        || params.block_size != 64
        || params.quantized_width == 0
        || params.quantized_width > params.row_width
        || params.quantized_width % params.block_size != 0
        || params.quantized_width / params.block_size > 16
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_mxfp8: invalid rows/width/quantized/block parameters: {}/{}/{}/{}",
            params.rows, params.row_width, params.quantized_width, params.block_size
        )));
    }
    validate_bf16_rows(
        data,
        params.rows as usize,
        params.row_width as usize,
        "deepseek_mxfp8",
    )?;
    let pipeline = registry.get_pipeline(DEEPSEEK_MXFP8_KERNEL, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(data)),
        ],
        MTLSize::new(params.rows as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}

/// In-place normalized 128-wide Hadamard rotation and block-32
/// E2M1/E8M0 fake quantization of BF16 indexer vectors.
pub fn dispatch_deepseek_hadamard_mxfp4_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    data: &MlxBuffer,
    rows: u32,
) -> Result<()> {
    if rows == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_hadamard_mxfp4: rows must be nonzero".into(),
        ));
    }
    validate_bf16_rows(
        data,
        rows as usize,
        DEEPSEEK_INDEX_WIDTH,
        "deepseek_hadamard_mxfp4",
    )?;
    let pipeline = registry.get_pipeline(DEEPSEEK_HADAMARD_MXFP4_KERNEL, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&rows))),
            (1, KernelArg::Buffer(data)),
        ],
        &[(
            0,
            (DEEPSEEK_INDEX_WIDTH * std::mem::size_of::<f32>()) as u64,
        )],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(DEEPSEEK_INDEX_WIDTH as u64, 1, 1),
    );
    Ok(())
}
