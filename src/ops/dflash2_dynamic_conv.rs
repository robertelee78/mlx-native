//! Grouped two-tap DFlash2 dynamic convolution.

use crate::{
    buffer::MlxBuffer,
    device::MlxDevice,
    dtypes::DType,
    encoder::CommandEncoder,
    error::{MlxError, Result},
    kernel_registry::KernelRegistry,
};
use metal::MTLSize;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub const DFLASH2_DYNAMIC_CONV_KERNEL: &str = "dflash2_dynamic_conv_f32";
pub static DFLASH2_DYNAMIC_CONV_SOURCE: &str =
    include_str!("../shaders/dflash2_dynamic_conv.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(DFLASH2_DYNAMIC_CONV_KERNEL, DFLASH2_DYNAMIC_CONV_SOURCE);
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlash2ConvolutionStage {
    Prepare = 0,
    Residual = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2DynamicConvParams {
    pub lanes: u32,
    pub rows_per_lane: u32,
    pub hidden: u32,
    pub groups: u32,
    pub stage: u32,
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| MlxError::InvalidArgument("DFlash2 convolution shape overflow".into()))
    })
}

fn physical_range(buffer: &MlxBuffer) -> Result<(usize, usize)> {
    let start = (buffer.contents_ptr() as usize)
        .checked_add(
            usize::try_from(buffer.byte_offset())
                .map_err(|_| MlxError::InvalidArgument("DFlash2 buffer offset overflow".into()))?,
        )
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 buffer start overflow".into()))?;
    let end = start
        .checked_add(buffer.data_byte_len())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 buffer end overflow".into()))?;
    Ok((start, end))
}

fn require(buffer: &MlxBuffer, dtype: DType, shape: &[usize], label: &str) -> Result<()> {
    if buffer.dtype() != dtype || buffer.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 convolution {label} must be {dtype} {shape:?}, got {} {:?}",
            buffer.dtype(),
            buffer.shape()
        )));
    }
    let bytes = product(shape)?
        .checked_mul(dtype.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 convolution byte overflow".into()))?;
    if buffer.data_byte_len() != bytes {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 convolution {label} has {} data bytes, expected {bytes}",
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

/// Encode one of the two convolution stages. `residual` is required only for
/// [`DFlash2ConvolutionStage::Residual`]; the prepare stage binds `input` to the
/// unused residual argument without copying it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_dynamic_conv_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    dynamic_weights: &MlxBuffer,
    base_kernel: &MlxBuffer,
    residual: Option<&MlxBuffer>,
    output: &MlxBuffer,
    mut params: DFlash2DynamicConvParams,
    stage: DFlash2ConvolutionStage,
) -> Result<()> {
    params.stage = stage as u32;
    if params.lanes == 0
        || params.rows_per_lane < 2
        || params.rows_per_lane > 16
        || params.hidden == 0
        || params.groups == 0
        || params.hidden % params.groups != 0
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 convolution geometry: lanes={}, rows={}, hidden={}, groups={}",
            params.lanes, params.rows_per_lane, params.hidden, params.groups
        )));
    }
    let rows = (params.lanes as usize)
        .checked_mul(params.rows_per_lane as usize)
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 convolution row overflow".into()))?;
    let activations = [rows, params.hidden as usize];
    let dynamic = [rows, 4 * params.groups as usize];
    let base = [2, 2, params.hidden as usize];
    require(input, DType::F32, &activations, "input")?;
    require(dynamic_weights, DType::F32, &dynamic, "dynamic weights")?;
    require(base_kernel, DType::BF16, &base, "base kernel")?;
    require(output, DType::F32, &activations, "output")?;
    let residual = match stage {
        DFlash2ConvolutionStage::Prepare => input,
        DFlash2ConvolutionStage::Residual => residual.ok_or_else(|| {
            MlxError::InvalidArgument(
                "DFlash2 residual convolution requires a residual buffer".into(),
            )
        })?,
    };
    if stage == DFlash2ConvolutionStage::Residual {
        require(residual, DType::F32, &activations, "residual")?;
    }
    if output.contents_ptr() == input.contents_ptr()
        || output.contents_ptr() == dynamic_weights.contents_ptr()
        || output.contents_ptr() == base_kernel.contents_ptr()
        || (stage == DFlash2ConvolutionStage::Residual
            && output.contents_ptr() == residual.contents_ptr())
    {
        return Err(MlxError::InvalidArgument(
            "DFlash2 convolution output must not alias an input".into(),
        ));
    }

    if encoder.is_capturing() {
        let reads = [input, dynamic_weights, base_kernel, residual]
            .iter()
            .map(|buffer| physical_range(buffer))
            .collect::<Result<Vec<_>>>()?;
        encoder.set_pending_buffer_ranges(reads, vec![physical_range(output)?]);
    }
    let pipeline = registry.get_pipeline(DFLASH2_DYNAMIC_CONV_KERNEL, device.metal_device())?;
    let elements = product(&activations)?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(dynamic_weights)),
            (2, KernelArg::Buffer(base_kernel)),
            (3, KernelArg::Buffer(residual)),
            (4, KernelArg::Buffer(output)),
            (5, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(256.min(elements) as u64, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        input: &[f32],
        dynamic: &[f32],
        base: &[f32],
        residual: &[f32],
        params: DFlash2DynamicConvParams,
        stage: DFlash2ConvolutionStage,
    ) -> Vec<f32> {
        let hidden = params.hidden as usize;
        let groups = params.groups as usize;
        let rows_per_lane = params.rows_per_lane as usize;
        let rows = params.lanes as usize * rows_per_lane;
        let group_size = hidden / groups;
        let stage_tap = stage as usize * 2;
        let dynamic_width = 4 * groups;
        let mut output = vec![0.0; rows * hidden];
        for row in 0..rows {
            for channel in 0..hidden {
                let element = row * hidden + channel;
                let group = channel / group_size;
                let mut value = input[element]
                    * (base[stage_tap * hidden + channel]
                        + dynamic[row * dynamic_width + stage_tap * groups + group]);
                if row % rows_per_lane > 0 {
                    value += input[element - hidden]
                        * (base[(stage_tap + 1) * hidden + channel]
                            + dynamic[row * dynamic_width + (stage_tap + 1) * groups + group]);
                }
                if stage == DFlash2ConvolutionStage::Residual {
                    value += residual[element];
                }
                output[element] = value;
            }
        }
        output
    }

    #[test]
    fn parameter_abi_is_five_u32_words() {
        assert_eq!(std::mem::size_of::<DFlash2DynamicConvParams>(), 20);
    }

    #[test]
    fn gpu_convolution_matches_lane_local_reference() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let params = DFlash2DynamicConvParams {
            lanes: 2,
            rows_per_lane: 2,
            hidden: 2,
            groups: 1,
            stage: 0,
        };
        let input_values = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut input = device.alloc_buffer(32, DType::F32, vec![4, 2]).unwrap();
        input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_values);
        let dynamic = device
            .alloc_buffer(4 * 4 * 4, DType::F32, vec![4, 4])
            .unwrap();
        let mut base = device.alloc_buffer(16, DType::BF16, vec![2, 2, 2]).unwrap();
        let base_values = [1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        for (output, value) in base
            .as_mut_slice::<u16>()
            .unwrap()
            .iter_mut()
            .zip(base_values)
        {
            *output = (value.to_bits() >> 16) as u16;
        }
        let output = device.alloc_buffer(32, DType::F32, vec![4, 2]).unwrap();
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_dynamic_conv_f32(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &dynamic,
            &base,
            None,
            &output,
            params,
            DFlash2ConvolutionStage::Prepare,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let expected = reference(
            &input_values,
            &[0.0; 16],
            &base_values,
            &[0.0; 8],
            params,
            DFlash2ConvolutionStage::Prepare,
        );
        assert_eq!(output.as_slice::<f32>().unwrap(), expected);
    }

    #[test]
    fn reference_does_not_cross_lane_boundaries() {
        let params = DFlash2DynamicConvParams {
            lanes: 2,
            rows_per_lane: 2,
            hidden: 2,
            groups: 1,
            stage: 0,
        };
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let dynamic = vec![0.0; 4 * 4];
        let base = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let output = reference(
            &input,
            &dynamic,
            &base,
            &[0.0; 8],
            params,
            DFlash2ConvolutionStage::Prepare,
        );
        assert_eq!(output, [1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn residual_stage_adds_declared_residual() {
        let params = DFlash2DynamicConvParams {
            lanes: 1,
            rows_per_lane: 2,
            hidden: 2,
            groups: 1,
            stage: 0,
        };
        let output = reference(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 8],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            &[5.0, 6.0, 7.0, 8.0],
            params,
            DFlash2ConvolutionStage::Residual,
        );
        assert_eq!(output, [6.0, 8.0, 10.0, 12.0]);
    }
}
