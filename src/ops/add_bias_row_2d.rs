//! ADR-021 helper: `out[m, n] = a[m, n] + bias[n]` for a `[M, N]`
//! row-major f32 matrix and a `[N]` f32 bias vector.
//!
//! Lands the patch-embed bias on top of the dual-stem matmul
//! accumulator (qwen3vl.cpp:41-43 equivalent). Lives in its own
//! shader file so ADR-021's surface stays disjoint from any
//! concurrent edits to `elementwise.metal`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub static ADD_BIAS_ROW_2D_SHADER_SOURCE: &str =
    include_str!("../shaders/add_bias_row_2d.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("add_bias_row_2d_f32", ADD_BIAS_ROW_2D_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAddBiasRow2dParams {
    m: u32,
    n: u32,
}

const TG_SIZE: u64 = 256;

/// `output[i, j] = a[i, j] + bias[j]` — bias broadcast across rows.
pub fn dispatch_add_bias_row_2d_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    bias: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "add_bias_row_2d_f32: m ({m}) and n ({n}) must both be > 0"
        )));
    }
    let f32_sz = DType::F32.size_of();
    let mn = (m as usize) * (n as usize);
    let need_a = mn * f32_sz;
    if a.byte_len() < need_a {
        return Err(MlxError::InvalidArgument(format!(
            "add_bias_row_2d_f32: a too small: {} vs {}",
            a.byte_len(),
            need_a
        )));
    }
    if bias.byte_len() < (n as usize) * f32_sz {
        return Err(MlxError::InvalidArgument(format!(
            "add_bias_row_2d_f32: bias too small: {} vs {}",
            bias.byte_len(),
            (n as usize) * f32_sz
        )));
    }
    if output.byte_len() < need_a {
        return Err(MlxError::InvalidArgument(format!(
            "add_bias_row_2d_f32: output too small: {} vs {}",
            output.byte_len(),
            need_a
        )));
    }

    let pipeline = registry.get_pipeline("add_bias_row_2d_f32", device)?;
    let gpu_params = GpuAddBiasRow2dParams { m, n };
    let total = mn as u64;
    let grid = MTLSize::new(total, 1, 1);
    let tg = MTLSize::new(std::cmp::min(TG_SIZE, total), 1, 1);
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(a)),
            (2, KernelArg::Buffer(bias)),
            (3, KernelArg::Buffer(output)),
        ],
        grid,
        tg,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;
    use crate::graph::GraphExecutor;

    #[test]
    fn adr021_add_bias_row_2d_f32_byte_identical() {
        let device = MlxDevice::new().expect("MlxDevice");
        let m: u32 = 17;
        let n: u32 = 33;
        let a_host: Vec<f32> = (0..(m * n))
            .map(|i| ((i as f32) * 0.013_7_f32).sin())
            .collect();
        let bias_host: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.071_f32).cos()).collect();
        let mut expected = vec![0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                expected[i * (n as usize) + j] = a_host[i * (n as usize) + j] + bias_host[j];
            }
        }

        let executor =
            GraphExecutor::new(MlxDevice::new().expect("MlxDevice for executor"));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut a_buf = device
            .alloc_buffer(a_host.len() * 4, DType::F32, vec![m as usize, n as usize])
            .unwrap();
        a_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&a_host);
        let mut bias_buf = device
            .alloc_buffer(bias_host.len() * 4, DType::F32, vec![n as usize])
            .unwrap();
        bias_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&bias_host);
        let out_buf = device
            .alloc_buffer(a_host.len() * 4, DType::F32, vec![m as usize, n as usize])
            .unwrap();

        dispatch_add_bias_row_2d_f32(
            session.encoder_mut(),
            &mut registry,
            device.metal_device(),
            &a_buf,
            &bias_buf,
            &out_buf,
            m,
            n,
        )
        .expect("dispatch bias add");
        session.finish().expect("finish");

        let got: &[f32] = out_buf.as_slice::<f32>().unwrap();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "bias-add drift at {i}: g={g} e={e}");
        }
    }
}
