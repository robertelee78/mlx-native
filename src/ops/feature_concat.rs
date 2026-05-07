//! ADR-021 K5: GPU feature-axis concat (single-chunk strided copy).
//!
//! Each invocation copies one `[T, src_dim]` f32 row-major slab into
//! its slice of the concatenated `[T, dst_stride]` destination, at
//! column offset `dst_offset`. Launching once per chunk (with varying
//! `dst_offset`) builds the full `[T, Σ src_dim_i]` concatenated
//! tensor — exactly the shape qwen3vl.cpp:186
//! `ggml_concat(ctx0, embeddings, deepstack_features, 0)` produces.
//!
//! Pure copy (no FP arithmetic) → AC-1 byte-identical.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub static FEATURE_CONCAT_SHADER_SOURCE: &str =
    include_str!("../shaders/feature_concat.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("feature_concat_f32", FEATURE_CONCAT_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFeatureConcatParams {
    n_tokens: u32,
    src_dim: u32,
    dst_offset: u32,
    dst_stride: u32,
}

const TG_SIZE: u64 = 256;

/// Copy one `[n_tokens, src_dim]` f32 row-major chunk into the
/// `[n_tokens, dst_stride]` destination at column `dst_offset`.
///
/// Caller is responsible for ensuring chunks don't overlap and
/// `dst_offset + src_dim <= dst_stride`.
pub fn dispatch_feature_concat_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    n_tokens: u32,
    src_dim: u32,
    dst_offset: u32,
    dst_stride: u32,
) -> Result<()> {
    if n_tokens == 0 || src_dim == 0 || dst_stride == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "feature_concat_f32: n_tokens ({n_tokens}), src_dim ({src_dim}), \
             dst_stride ({dst_stride}) must all be > 0"
        )));
    }
    if dst_offset.checked_add(src_dim).map(|e| e > dst_stride).unwrap_or(true) {
        return Err(MlxError::InvalidArgument(format!(
            "feature_concat_f32: dst_offset ({dst_offset}) + src_dim ({src_dim}) > \
             dst_stride ({dst_stride}) — chunk overflows the destination row"
        )));
    }
    let f32_sz = DType::F32.size_of();
    let need_src = (n_tokens as usize) * (src_dim as usize) * f32_sz;
    let need_dst = (n_tokens as usize) * (dst_stride as usize) * f32_sz;
    if src.byte_len() < need_src {
        return Err(MlxError::InvalidArgument(format!(
            "feature_concat_f32: src too small: {} vs {} bytes",
            src.byte_len(), need_src
        )));
    }
    if dst.byte_len() < need_dst {
        return Err(MlxError::InvalidArgument(format!(
            "feature_concat_f32: dst too small: {} vs {} bytes",
            dst.byte_len(), need_dst
        )));
    }

    let pipeline = registry.get_pipeline("feature_concat_f32", device)?;
    let gpu_params = GpuFeatureConcatParams {
        n_tokens,
        src_dim,
        dst_offset,
        dst_stride,
    };
    let total = (n_tokens as u64) * (src_dim as u64);
    let grid = MTLSize::new(total, 1, 1);
    let tg = MTLSize::new(std::cmp::min(TG_SIZE, total), 1, 1);
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(src)),
            (2, KernelArg::Buffer(dst)),
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
    fn adr021_k5_feature_concat_f32_byte_identical() {
        let device = MlxDevice::new().expect("MlxDevice");
        let n_tokens: u32 = 11;
        let dim_main: u32 = 32;
        let dim_ds: u32 = 32;
        let dim_total: u32 = dim_main + dim_ds * 3; // base + 3 deepstacks

        let src_main: Vec<f32> = (0..(n_tokens * dim_main))
            .map(|i| ((i as f32) * 0.013_3_f32).sin() * 0.5)
            .collect();
        let src_ds: Vec<Vec<f32>> = (0..3)
            .map(|seed| {
                (0..(n_tokens * dim_ds))
                    .map(|i| ((i as f32 + 100.0 * (seed as f32 + 1.0)) * 0.011_7_f32).cos() * 0.5)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Build CPU oracle.
        let mut expected = vec![0f32; (n_tokens * dim_total) as usize];
        let row_stride = dim_total as usize;
        for t in 0..n_tokens as usize {
            // base
            let dst_base = t * row_stride;
            let src_base = t * dim_main as usize;
            for d in 0..dim_main as usize {
                expected[dst_base + d] = src_main[src_base + d];
            }
            // deepstacks
            for (i, ds) in src_ds.iter().enumerate() {
                let dst_off = (i + 1) * dim_ds as usize;
                let src_off = t * dim_ds as usize;
                for d in 0..dim_ds as usize {
                    expected[dst_base + dst_off + d] = ds[src_off + d];
                }
            }
        }

        // GPU.
        let executor =
            GraphExecutor::new(MlxDevice::new().expect("MlxDevice for executor"));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut main_buf = device
            .alloc_buffer(src_main.len() * 4, DType::F32, vec![n_tokens as usize, dim_main as usize])
            .unwrap();
        main_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&src_main);
        let mut ds_bufs: Vec<MlxBuffer> = (0..3)
            .map(|i| {
                let mut b = device
                    .alloc_buffer(src_ds[i].len() * 4, DType::F32, vec![n_tokens as usize, dim_ds as usize])
                    .unwrap();
                b.as_mut_slice::<f32>().unwrap().copy_from_slice(&src_ds[i]);
                b
            })
            .collect();
        let dst_buf = device
            .alloc_buffer((n_tokens * dim_total * 4) as usize, DType::F32,
                vec![n_tokens as usize, dim_total as usize])
            .unwrap();

        // Copy main at offset 0.
        dispatch_feature_concat_f32(
            session.encoder_mut(), &mut registry, device.metal_device(),
            &main_buf, &dst_buf, n_tokens, dim_main, 0, dim_total,
        ).unwrap();
        session.encoder_mut().memory_barrier();

        // Copy each deepstack at offset (i+1)*dim_ds.
        for (i, ds) in ds_bufs.iter_mut().enumerate() {
            dispatch_feature_concat_f32(
                session.encoder_mut(), &mut registry, device.metal_device(),
                ds, &dst_buf, n_tokens, dim_ds, (i as u32 + 1) * dim_ds, dim_total,
            ).unwrap();
            session.encoder_mut().memory_barrier();
        }

        session.finish().expect("finish");
        let got = dst_buf.as_slice::<f32>().unwrap();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "K5 byte parity violated at {i}");
        }
    }

    #[test]
    fn adr021_k5_feature_concat_f32_input_validation() {
        let device = MlxDevice::new().expect("MlxDevice");
        let executor = GraphExecutor::new(MlxDevice::new().expect("device for executor"));
        let mut session = executor.begin().expect("session");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let s = device.alloc_buffer(64 * 4, DType::F32, vec![16, 4]).unwrap();
        let d = device.alloc_buffer(128 * 4, DType::F32, vec![16, 8]).unwrap();

        // dst_offset + src_dim > dst_stride
        let err = dispatch_feature_concat_f32(
            session.encoder_mut(), &mut registry, device.metal_device(),
            &s, &d, 16, 4, 5, 8,  // 5+4 = 9 > 8
        ).unwrap_err();
        assert!(format!("{err}").contains("overflows the destination row"));
    }
}
