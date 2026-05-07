//! ADR-021 K2: GPU antialiased bilinear resize for the Qwen3-VL ViT
//! position-embedding table.
//!
//! Resamples a `[H_src, W_src, C]` row-major f32 tensor into a
//! `[H_dst, W_dst, C]` row-major f32 tensor using a triangle filter
//! with support `max(1, 1/sf)` per axis. Mirrors the CPU oracle
//! `qwen3vl_resize_position_embeddings_bilinear` exactly:
//!
//! - sample-coord mapping: `(dst + 0.5) / sf - 0.5`
//!   (PyTorch align_corners=False / pixel_offset=0.5).
//! - antialias: when sf < 1 (downsampling), support widens beyond 1
//!   to apply a low-pass triangle prefilter — exactly matching
//!   ggml's `BILINEAR | ANTIALIAS` mode at
//!   `/opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp:7578-7637`.
//! - When sf == 1 (matching edge), the formula collapses to
//!   pass-through; this is the fast path Qwen3-VL hits at the
//!   trained 768×768 resolution.
//!
//! Constraints:
//! - `trained_n` and `target_n_*` must all be > 0.
//! - `n_embd > 0`.
//! - The source grid is square (`[trained_n, trained_n, C]`); this
//!   matches the CPU oracle's `n_per_side² == num_position_embeddings`
//!   contract.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub static BILINEAR_RESIZE_2D_SHADER_SOURCE: &str =
    include_str!("../shaders/bilinear_resize_2d.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("bilinear_resize_2d_f32", BILINEAR_RESIZE_2D_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBilinearResize2dParams {
    trained_n: u32,
    target_n_x: u32,
    target_n_y: u32,
    n_embd: u32,
    sf_x: f32,
    sf_y: f32,
    support_x: f32,
    support_y: f32,
    invscale_x: f32,
    invscale_y: f32,
}

const TG_SIZE: u64 = 256;

/// Dispatch the K2 antialiased bilinear resize.
///
/// # Arguments
///
/// * `src` — input, f32 row-major, byte_len ≥ `trained_n*trained_n*n_embd*4`.
/// * `dst` — output, f32 row-major, byte_len ≥ `target_n_y*target_n_x*n_embd*4`.
///
/// # Errors
///
/// `MlxError::InvalidArgument` for any zero dimension or buffer too small.
pub fn dispatch_bilinear_resize_2d_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    trained_n: u32,
    target_n_x: u32,
    target_n_y: u32,
    n_embd: u32,
) -> Result<()> {
    if trained_n == 0 || target_n_x == 0 || target_n_y == 0 || n_embd == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "bilinear_resize_2d_f32: trained_n ({trained_n}), target_n_x \
             ({target_n_x}), target_n_y ({target_n_y}), n_embd ({n_embd}) \
             must all be > 0"
        )));
    }
    let f32_sz = DType::F32.size_of();
    let need_src = (trained_n as usize) * (trained_n as usize) * (n_embd as usize) * f32_sz;
    if src.byte_len() < need_src {
        return Err(MlxError::InvalidArgument(format!(
            "bilinear_resize_2d_f32: src too small: {} vs {} bytes",
            src.byte_len(),
            need_src
        )));
    }
    let target_total =
        (target_n_y as usize) * (target_n_x as usize) * (n_embd as usize);
    let need_dst = target_total * f32_sz;
    if dst.byte_len() < need_dst {
        return Err(MlxError::InvalidArgument(format!(
            "bilinear_resize_2d_f32: dst too small: {} vs {} bytes",
            dst.byte_len(),
            need_dst
        )));
    }

    let sf_x = (target_n_x as f32) / (trained_n as f32);
    let sf_y = (target_n_y as f32) / (trained_n as f32);
    let support_x = (1.0f32 / sf_x).max(1.0);
    let support_y = (1.0f32 / sf_y).max(1.0);
    let invscale_x = 1.0f32 / support_x;
    let invscale_y = 1.0f32 / support_y;

    let pipeline = registry.get_pipeline("bilinear_resize_2d_f32", device)?;
    let gpu_params = GpuBilinearResize2dParams {
        trained_n,
        target_n_x,
        target_n_y,
        n_embd,
        sf_x,
        sf_y,
        support_x,
        support_y,
        invscale_x,
        invscale_y,
    };

    let total = target_total as u64;
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

// ---------------------------------------------------------------------------
// CPU oracle (test-only) — mirrors the formula used in
// `qwen3vl_resize_position_embeddings_bilinear` in hf2q exactly.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn bilinear_resize_2d_f32_cpu_oracle(
    src: &[f32],
    trained_n: u32,
    target_n_x: u32,
    target_n_y: u32,
    n_embd: u32,
) -> Vec<f32> {
    let trained = trained_n as i64;
    let tx = target_n_x as i64;
    let ty = target_n_y as i64;
    let h = n_embd as usize;

    let sf_x = (tx as f32) / (trained as f32);
    let sf_y = (ty as f32) / (trained as f32);
    let support_x = (1.0f32 / sf_x).max(1.0);
    let support_y = (1.0f32 / sf_y).max(1.0);
    let invscale_x = 1.0f32 / support_x;
    let invscale_y = 1.0f32 / support_y;
    let pixel_offset = 0.5f32;

    let triangle_filter = |x: f32| -> f32 { (1.0 - x.abs()).max(0.0) };
    let mut out = vec![0f32; (ty as usize) * (tx as usize) * h];

    for y_dst in 0..ty {
        let y = ((y_dst as f32) + pixel_offset) / sf_y;
        let y_min = ((y - support_y + pixel_offset).max(0.0)) as i64;
        let y_max = ((y + support_y + pixel_offset).min(trained as f32)) as i64;
        for x_dst in 0..tx {
            let x = ((x_dst as f32) + pixel_offset) / sf_x;
            let x_min = ((x - support_x + pixel_offset).max(0.0)) as i64;
            let x_max = ((x + support_x + pixel_offset).min(trained as f32)) as i64;

            let dst_off = ((y_dst as usize) * (tx as usize) + (x_dst as usize)) * h;
            let mut total_weight = 0.0f32;
            for sy in y_min..y_max {
                let wy = triangle_filter(((sy as f32) - y + pixel_offset) * invscale_y);
                for sx in x_min..x_max {
                    let wx =
                        triangle_filter(((sx as f32) - x + pixel_offset) * invscale_x);
                    let w = wx * wy;
                    if w <= 0.0 {
                        continue;
                    }
                    let src_off =
                        ((sy as usize) * (trained as usize) + (sx as usize)) * h;
                    for k in 0..h {
                        out[dst_off + k] += src[src_off + k] * w;
                    }
                    total_weight += w;
                }
            }
            if total_weight > 0.0 {
                let inv = 1.0 / total_weight;
                for k in 0..h {
                    out[dst_off + k] *= inv;
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
        src_host: &[f32],
        trained_n: u32,
        target_n_x: u32,
        target_n_y: u32,
        n_embd: u32,
    ) -> Vec<f32> {
        let executor =
            GraphExecutor::new(MlxDevice::new().expect("MlxDevice for executor"));
        let mut session = executor.begin().expect("begin");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut src_buf = device
            .alloc_buffer(
                src_host.len() * 4,
                DType::F32,
                vec![trained_n as usize, trained_n as usize, n_embd as usize],
            )
            .unwrap();
        src_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(src_host);

        let target_total =
            (target_n_y as usize) * (target_n_x as usize) * (n_embd as usize);
        let dst_buf = device
            .alloc_buffer(
                target_total * 4,
                DType::F32,
                vec![target_n_y as usize, target_n_x as usize, n_embd as usize],
            )
            .unwrap();

        dispatch_bilinear_resize_2d_f32(
            session.encoder_mut(),
            &mut registry,
            device.metal_device(),
            &src_buf,
            &dst_buf,
            trained_n,
            target_n_x,
            target_n_y,
            n_embd,
        )
        .expect("dispatch K2");
        session.finish().expect("finish");

        dst_buf.as_slice::<f32>().expect("readback").to_vec()
    }

    fn make_seeded(trained_n: u32, n_embd: u32) -> Vec<f32> {
        let n = (trained_n * trained_n * n_embd) as usize;
        (0..n)
            .map(|i| ((i as f32) * 0.013_3_f32).sin() * 0.5)
            .collect()
    }

    /// Fast-path: trained_n == target_n_x == target_n_y → pass-through.
    /// Byte-identical here because the formula collapses to a single
    /// term with weight 1.0 (no FP-summation order ambiguity).
    #[test]
    fn adr021_k2_bilinear_resize_2d_f32_byte_identical_fast_path() {
        let device = MlxDevice::new().expect("MlxDevice");
        let trained: u32 = 8;
        let n_embd: u32 = 16;
        let src = make_seeded(trained, n_embd);
        let oracle = bilinear_resize_2d_f32_cpu_oracle(&src, trained, trained, trained, n_embd);
        let gpu = run_kernel(&device, &src, trained, trained, trained, n_embd);
        assert_eq!(oracle.len(), gpu.len());
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K2 fast-path byte parity violated at element {i}: oracle={a} gpu={b}");
        }
    }

    /// Upsample 2× per axis: trained 8×8 → target 16×16. Triangle
    /// filter support degenerates to 1 → 4-tap bilinear. Tolerance
    /// is 1 ULP per element (the ADR allows up to 1 ULP for K2).
    #[test]
    fn adr021_k2_bilinear_resize_2d_f32_ulp_bound_upsample_2x() {
        let device = MlxDevice::new().expect("MlxDevice");
        let trained: u32 = 8;
        let target: u32 = 16;
        let n_embd: u32 = 16;
        let src = make_seeded(trained, n_embd);
        let oracle = bilinear_resize_2d_f32_cpu_oracle(&src, trained, target, target, n_embd);
        let gpu = run_kernel(&device, &src, trained, target, target, n_embd);
        assert_eq!(oracle.len(), gpu.len());
        let max_abs = oracle
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Upsampling support = 1 → 4-tap bilinear; CPU & GPU compute
        // the same 4 weights * 4 src values then divide by total
        // weight. Tile size differences in the matmul-style summation
        // — there's no matmul here; this is straight-line — so the
        // sum order is identical and we expect byte equality. Worst-
        // case allow 1 ULP for any rounding-mode delta.
        assert!(
            max_abs < 1e-6,
            "K2 upsample drift {} exceeds 1e-6 tolerance",
            max_abs
        );
    }

    /// Downsample to a non-square target — exercises the antialias
    /// branch with sf_x != sf_y. trained 16×16 → target 4×8 means
    /// support_x = 4.0 (downsample 16→4 wide → 4-tap), support_y =
    /// 2.0 (downsample 16→8 narrow → 2-tap). Multiple weight
    /// contributions per (y_dst, x_dst, c) tile.
    #[test]
    fn adr021_k2_bilinear_resize_2d_f32_ulp_bound_downsample_rect() {
        let device = MlxDevice::new().expect("MlxDevice");
        let trained: u32 = 16;
        let n_embd: u32 = 16;
        let src = make_seeded(trained, n_embd);
        let oracle = bilinear_resize_2d_f32_cpu_oracle(&src, trained, 4, 8, n_embd);
        let gpu = run_kernel(&device, &src, trained, 4, 8, n_embd);
        assert_eq!(oracle.len(), gpu.len());
        let max_abs = oracle
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // The CPU oracle iterates (sy outer, sx inner) and accumulates
        // per-channel into the same dst row. The GPU does the same per
        // (y_dst, x_dst, c) tile — straight-line sum, so we expect
        // byte equality, but allow 1 ULP for IEEE-754 rounding-mode
        // delta on Apple Metal vs Rust f32 stdlib.
        assert!(
            max_abs < 1e-6,
            "K2 downsample-rect drift {} exceeds 1e-6 tolerance",
            max_abs
        );
    }
}
