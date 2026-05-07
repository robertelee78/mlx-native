//! ADR-021 K4: GPU 2×2 block-merge reshape for the Qwen3-VL ViT prelude.
//!
//! Permutes a `[ny, nx, n_embd]` row-major f32 tensor into a
//! `[ny*nx, n_embd]` row-major tensor with patches reordered into
//! 2×2-block-major-then-row-major-within-block order. Pure copy
//! (no FP arithmetic) so AC-1 holds byte-identically.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub static BLOCK_MERGE_2X2_SHADER_SOURCE: &str =
    include_str!("../shaders/block_merge_2x2.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("block_merge_2x2_f32", BLOCK_MERGE_2X2_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBlockMerge2x2Params {
    nx: u32,
    ny: u32,
    n_embd: u32,
    half_x: u32,
}

const TG_SIZE: u64 = 256;

pub fn dispatch_block_merge_2x2_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    nx: u32,
    ny: u32,
    n_embd: u32,
) -> Result<()> {
    if nx == 0 || ny == 0 || n_embd == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "block_merge_2x2_f32: nx ({nx}), ny ({ny}), n_embd ({n_embd}) must all be > 0"
        )));
    }
    if nx % 2 != 0 || ny % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "block_merge_2x2_f32: nx ({nx}) and ny ({ny}) must both be even \
             (2x2 block merge)"
        )));
    }
    let f32_sz = DType::F32.size_of();
    let total = (ny as usize) * (nx as usize) * (n_embd as usize);
    let need = total * f32_sz;
    if input.byte_len() < need || output.byte_len() < need {
        return Err(MlxError::InvalidArgument(format!(
            "block_merge_2x2_f32: input/output too small for [{ny}, {nx}, {n_embd}]: \
             input {} / output {} vs need {} bytes",
            input.byte_len(), output.byte_len(), need
        )));
    }
    let pipeline = registry.get_pipeline("block_merge_2x2_f32", device)?;
    let gpu_params = GpuBlockMerge2x2Params {
        nx,
        ny,
        n_embd,
        half_x: nx / 2,
    };
    let total_u64 = total as u64;
    let grid = MTLSize::new(total_u64, 1, 1);
    let tg = MTLSize::new(std::cmp::min(TG_SIZE, total_u64), 1, 1);
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(output)),
        ],
        grid,
        tg,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU oracle (test-only) — mirrors `qwen3vl_2x2_block_merge_reshape`.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn block_merge_2x2_f32_cpu_oracle(
    input: &[f32],
    nx: usize,
    ny: usize,
    n_embd: usize,
) -> Vec<f32> {
    let half_x = nx / 2;
    let mut out = vec![0f32; ny * nx * n_embd];
    for by in 0..(ny / 2) {
        for bx in 0..half_x {
            let block_id = by * half_x + bx;
            for y_in in 0..2 {
                for x_in in 0..2 {
                    let src_y = by * 2 + y_in;
                    let src_x = bx * 2 + x_in;
                    let src_off = (src_y * nx + src_x) * n_embd;
                    let within = y_in * 2 + x_in;
                    let dst_p = block_id * 4 + within;
                    let dst_off = dst_p * n_embd;
                    out[dst_off..dst_off + n_embd]
                        .copy_from_slice(&input[src_off..src_off + n_embd]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;
    use crate::graph::GraphExecutor;

    fn run_kernel(
        device: &MlxDevice,
        input_host: &[f32],
        nx: u32,
        ny: u32,
        n_embd: u32,
    ) -> Vec<f32> {
        let executor =
            GraphExecutor::new(MlxDevice::new().expect("MlxDevice for executor"));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut in_buf = device
            .alloc_buffer(
                input_host.len() * 4,
                DType::F32,
                vec![ny as usize, nx as usize, n_embd as usize],
            )
            .unwrap();
        in_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(input_host);
        let out_buf = device
            .alloc_buffer(
                input_host.len() * 4,
                DType::F32,
                vec![(ny * nx) as usize, n_embd as usize],
            )
            .unwrap();

        dispatch_block_merge_2x2_f32(
            session.encoder_mut(),
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
            nx,
            ny,
            n_embd,
        )
        .expect("dispatch K4");
        session.finish().expect("finish");
        out_buf.as_slice::<f32>().expect("readback").to_vec()
    }

    fn make_seeded(nx: u32, ny: u32, n_embd: u32) -> Vec<f32> {
        let n = (ny * nx * n_embd) as usize;
        (0..n)
            .map(|i| ((i as f32) * 0.013_3_f32).sin() * 0.5)
            .collect()
    }

    /// AC-1 byte parity, square 8×8 patch grid with n_embd=32.
    #[test]
    fn adr021_k4_block_merge_2x2_f32_byte_identical_square_8x8() {
        let device = MlxDevice::new().expect("MlxDevice");
        let nx: u32 = 8;
        let ny: u32 = 8;
        let n_embd: u32 = 32;
        let input = make_seeded(nx, ny, n_embd);
        let oracle = block_merge_2x2_f32_cpu_oracle(&input, nx as usize, ny as usize, n_embd as usize);
        let gpu = run_kernel(&device, &input, nx, ny, n_embd);
        assert_eq!(oracle.len(), gpu.len());
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K4 byte parity violated at element {i}: oracle={a} gpu={b}");
        }
    }

    /// Rectangular wide grid: nx=12, ny=4, n_embd=8.
    #[test]
    fn adr021_k4_block_merge_2x2_f32_byte_identical_rect_wide() {
        let device = MlxDevice::new().expect("MlxDevice");
        let nx: u32 = 12;
        let ny: u32 = 4;
        let n_embd: u32 = 8;
        let input = make_seeded(nx, ny, n_embd);
        let oracle = block_merge_2x2_f32_cpu_oracle(&input, nx as usize, ny as usize, n_embd as usize);
        let gpu = run_kernel(&device, &input, nx, ny, n_embd);
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K4 byte parity violated at element {i} (rect wide): oracle={a} gpu={b}");
        }
    }

    /// Rectangular tall grid: nx=4, ny=12, n_embd=8.
    #[test]
    fn adr021_k4_block_merge_2x2_f32_byte_identical_rect_tall() {
        let device = MlxDevice::new().expect("MlxDevice");
        let nx: u32 = 4;
        let ny: u32 = 12;
        let n_embd: u32 = 8;
        let input = make_seeded(nx, ny, n_embd);
        let oracle = block_merge_2x2_f32_cpu_oracle(&input, nx as usize, ny as usize, n_embd as usize);
        let gpu = run_kernel(&device, &input, nx, ny, n_embd);
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K4 byte parity violated at element {i} (rect tall): oracle={a} gpu={b}");
        }
    }

    /// Reject odd grid sides loud.
    #[test]
    fn adr021_k4_block_merge_2x2_f32_input_validation() {
        let device = MlxDevice::new().expect("MlxDevice");
        let executor = GraphExecutor::new(MlxDevice::new().expect("device for executor"));
        let mut session = executor.begin().expect("session");
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let buf = device.alloc_buffer(64 * 4, DType::F32, vec![4, 4, 4]).unwrap();
        let out_buf = device.alloc_buffer(64 * 4, DType::F32, vec![16, 4]).unwrap();

        let err = dispatch_block_merge_2x2_f32(
            session.encoder_mut(), &mut registry, device.metal_device(),
            &buf, &out_buf, 5, 4, 4,  // nx=5 (odd)
        ).unwrap_err();
        assert!(format!("{err}").contains("must both be even"), "got: {err}");
    }
}
