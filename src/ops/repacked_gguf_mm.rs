//! Experimental fixed-width dense MM over load-once repacked Q4_K/Q6_K weights.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
use crate::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};

pub const REPACK_KERNEL: &str = "hf2q_gguf_repack_q4k_q6k";
pub const Q4K_MM_KERNEL: &str = "hf2q_repacked_mm_q4k_f32";
pub const Q6K_MM_KERNEL: &str = "hf2q_repacked_mm_q6k_f32";
pub const REDUCE_KERNEL: &str = "hf2q_repacked_mm_reduce_f32";
pub static SOURCE: &str = include_str!("../shaders/gguf_repacked_mm.metal");

pub fn register(registry: &mut KernelRegistry) {
    for name in [REPACK_KERNEL, Q4K_MM_KERNEL, Q6K_MM_KERNEL, REDUCE_KERNEL] {
        registry.register_source(name, SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RepackParams {
    rows: u32,
    input_size: u32,
    format: u32,
    plane1_offset: u32,
    meta_offset: u32,
    src_row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MmParams {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    splits: u32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    _pad1: u32,
}

#[derive(Clone)]
pub struct PreparedGgufWeight {
    _source: MlxBuffer,
    bytes: MlxBuffer,
    plane1_offset: u32,
    partials: Option<MlxBuffer>,
    meta_offset: u32,
    n: u32,
    k: u32,
    ggml_type: GgmlType,
    splits: u32,
}

fn target_splits(n: u32, k: u32) -> u32 {
    const M5_MAX_CORES: u32 = 40;
    let column_tiles = n / 64;
    let k_groups = k / 32;
    if column_tiles <= 5 * M5_MAX_CORES && k_groups % 4 == 0 && k / 4 >= 512 {
        4
    } else {
        1
    }
}

pub fn prepare_repacked_gguf_weight(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<PreparedGgufWeight> {
    if weight.dtype() != DType::U8 {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF preparation requires native U8 blocks, got {:?}",
            weight.dtype()
        )));
    }
    let format: u8 = match params.ggml_type {
        GgmlType::Q4_K => 0,
        GgmlType::Q6_K => 1,
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "repacked GGUF MM does not support {other:?}"
            )))
        }
    };
    if params.n % 64 != 0 || params.k % 256 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM requires N % 64 == 0 and K % 256 == 0, got N={} K={}",
            params.n, params.k
        )));
    }
    let rows = params.n as usize;
    let source_bytes = rows
        .checked_mul(params.k as usize / 256)
        .and_then(|value| value.checked_mul(params.ggml_type.block_bytes() as usize))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF source size overflow".into()))?;
    if weight.data_byte_len() < source_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF source needs {source_bytes} bytes, got {}",
            weight.data_byte_len()
        )));
    }

    let groups = params.k as usize / 32;
    let plane0 = rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF plane size overflow".into()))?;
    let plane1 = if format == 1 {
        rows.checked_mul(groups)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                MlxError::InvalidArgument("repacked GGUF high-bit plane size overflow".into())
            })?
    } else {
        0
    };
    let meta = rows
        .checked_mul(groups / 8)
        .and_then(|v| v.checked_mul(if format == 1 { 20 } else { 16 }))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF metadata size overflow".into()))?;
    let total = plane0
        .checked_add(plane1)
        .and_then(|v| v.checked_add(meta))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF size overflow".into()))?;
    let bytes = device.alloc_buffer(total, DType::U8, vec![total])?;
    let splits = target_splits(params.n, params.k);
    let partials = if splits > 1 {
        let elements = splits as usize * params.m as usize * params.n as usize;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MlxError::InvalidArgument("repacked GGUF partial size overflow".into())
            })?;
        Some(device.alloc_buffer(
            bytes,
            DType::F32,
            vec![splits as usize, params.m as usize, params.n as usize],
        )?)
    } else {
        None
    };
    let p = RepackParams {
        rows: params.n,
        input_size: params.k,
        format: u32::from(format),
        plane1_offset: u32::try_from(plane0).map_err(|_| {
            MlxError::InvalidArgument("repacked GGUF plane offset exceeds u32".into())
        })?,
        meta_offset: u32::try_from(plane0 + plane1).map_err(|_| {
            MlxError::InvalidArgument("repacked GGUF meta offset exceeds u32".into())
        })?,
        src_row_bytes: (params.k / 256) * params.ggml_type.block_bytes(),
    };
    let pipeline = registry.get_pipeline(REPACK_KERNEL, device.metal_device())?;
    let threads = rows * groups;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(&bytes)),
            (2, KernelArg::Bytes(as_bytes(&p))),
        ],
        MTLSize::new(threads as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    Ok(PreparedGgufWeight {
        _source: weight.clone(),
        bytes,
        partials,
        plane1_offset: p.plane1_offset,
        meta_offset: p.meta_offset,
        n: params.n,
        k: params.k,
        ggml_type: params.ggml_type,
        splits,
    })
}

pub fn dispatch_repacked_gguf_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    if params.m != 32 || input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(
            "repacked GGUF MM requires M=32 and F32 input/output buffers".into(),
        ));
    }
    if prepared.n != params.n || prepared.k != params.k || prepared.ggml_type != params.ggml_type {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM parameters do not match the prepared weight: prepared N={} K={} {:?}, got N={} K={} {:?}",
            prepared.n, prepared.k, prepared.ggml_type, params.n, params.k, params.ggml_type
        )));
    }
    let input_bytes = params.m as usize * params.k as usize * std::mem::size_of::<f32>();
    let output_bytes = params.m as usize * params.n as usize * std::mem::size_of::<f32>();
    if input.data_byte_len() < input_bytes || output.data_byte_len() < output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM buffer is too small: input {}/{input_bytes}, output {}/{output_bytes}",
            input.data_byte_len(),
            output.data_byte_len()
        )));
    }
    let kernel = match params.ggml_type {
        GgmlType::Q4_K => Q4K_MM_KERNEL,
        GgmlType::Q6_K => Q6K_MM_KERNEL,
        _ => unreachable!(),
    };
    let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
    let p = MmParams {
        ne00: params.k as i32,
        ne02: 1,
        nb01: u64::from(params.k / 256) * u64::from(params.ggml_type.block_bytes()),
        nb02: 0,
        nb03: 0,
        ne12: 1,
        splits: prepared.splits,
        nb10: 4,
        nb11: u64::from(params.k) * 4,
        nb12: u64::from(params.m) * u64::from(params.k) * 4,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        _pad1: 0,
    };
    let offsets = [prepared.plane1_offset, prepared.meta_offset];
    let partials = prepared.partials.as_ref().unwrap_or(output);
    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&p))),
            (1, KernelArg::Buffer(&prepared.bytes)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&offsets))),
            (5, KernelArg::Buffer(partials)),
        ],
        &[(0, 12_288)],
        MTLSize::new(
            u64::from(prepared.splits),
            u64::from(params.n).div_ceil(64),
            1,
        ),
        MTLSize::new(64, 1, 1),
    );
    if prepared.splits > 1 {
        encoder.memory_barrier();
        let reduce = registry.get_pipeline(REDUCE_KERNEL, device.metal_device())?;
        let elements = params.m * params.n;
        let dims = [elements, prepared.splits];
        encode_with_args(
            encoder,
            reduce,
            &[
                (0, KernelArg::Buffer(partials)),
                (1, KernelArg::Buffer(output)),
                (2, KernelArg::Bytes(as_bytes(&dims))),
            ],
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quantized_matmul_ggml::quantized_matmul_ggml;

    fn buffer(device: &MlxDevice, bytes: usize, dtype: DType, shape: Vec<usize>) -> MlxBuffer {
        device.alloc_buffer(bytes, dtype, shape).unwrap()
    }

    fn parity(ggml_type: GgmlType) {
        const M: usize = 32;
        const N: usize = 128;
        const K: usize = 2048;
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let mut input = buffer(&device, M * K * 4, DType::F32, vec![M, K]);
        for (i, value) in input.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
            *value = ((i * 17 % 257) as f32 - 128.0) / 127.0;
        }
        let block_bytes = ggml_type.block_bytes() as usize;
        let mut weight = buffer(
            &device,
            N * (K / ggml_type.block_values() as usize) * block_bytes,
            DType::U8,
            vec![N, K],
        );
        for (block, bytes) in weight
            .as_mut_slice::<u8>()
            .unwrap()
            .chunks_exact_mut(block_bytes)
            .enumerate()
        {
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = ((block * 31 + i * 13 + 7) & 255) as u8;
            }
            match ggml_type {
                GgmlType::Q4_K => {
                    bytes[0..2].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                    bytes[2..4].copy_from_slice(&half::f16::from_f32(0.0078125).to_le_bytes());
                    for scale in &mut bytes[4..16] {
                        *scale &= 0x3f;
                    }
                }
                GgmlType::Q6_K => {
                    for scale in &mut bytes[192..208] {
                        *scale = (*scale as i8).clamp(-31, 31) as u8;
                    }
                    bytes[208..210].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        let stock = buffer(&device, M * N * 4, DType::F32, vec![M, N]);
        let candidate = buffer(&device, M * N * 4, DType::F32, vec![M, N]);
        let params = GgmlQuantizedMatmulParams {
            m: M as u32,
            n: N as u32,
            k: K as u32,
            ggml_type,
        };
        let mut encoder = device.command_encoder().unwrap();
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            &stock,
            &params,
        )
        .unwrap();
        let prepared =
            prepare_repacked_gguf_weight(&mut encoder, &mut registry, &device, &weight, &params)
                .unwrap();
        dispatch_repacked_gguf_mm(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &prepared,
            &candidate,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let a = stock.as_slice::<f32>().unwrap();
        let b = candidate.as_slice::<f32>().unwrap();
        let mut worst_relative = 0.0_f32;
        let mut worst_absolute = 0.0_f32;
        let mut reference_max = 0.0_f32;
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for (&x, &y) in a.iter().zip(b) {
            let error = (x - y).abs();
            worst_relative = worst_relative.max(error / x.abs().max(1.0));
            worst_absolute = worst_absolute.max(error);
            reference_max = reference_max.max(x.abs());
            squared_error += f64::from(error) * f64::from(error);
            squared_reference += f64::from(x) * f64::from(x);
        }
        let relative_rmse = (squared_error / squared_reference).sqrt();
        let normalized_max_error = worst_absolute / reference_max;
        assert!(
            relative_rmse < 3e-4 && normalized_max_error < 5e-4,
            "{ggml_type:?} worst relative {worst_relative}, normalized max {normalized_max_error}, relative RMSE {relative_rmse}"
        );
    }

    #[test]
    fn q4k_matches_stock_tensor_mm() {
        parity(GgmlType::Q4_K);
    }

    #[test]
    fn q6k_matches_stock_tensor_mm() {
        parity(GgmlType::Q6_K);
    }
    #[test]
    fn m5_max_split_policy_keeps_wide_gate_up_unsplit() {
        assert_eq!(target_splits(5_120, 17_408), 4);
        assert_eq!(target_splits(12_288, 5_120), 4);
        assert_eq!(target_splits(17_408, 5_120), 1);
        assert_eq!(target_splits(248_320, 5_120), 1);
    }
}
