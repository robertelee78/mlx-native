//! ADR-021 K1: GPU im2col for the Qwen3-VL ViT dual-stem patch embed.
//!
//! Unfolds a `[3, H, W]` row-major f32 pixel buffer into a
//! `[num_patches, 3*p²]` row-major f32 im2col matrix matching
//! `patch_embed_forward_hw`'s inner-kernel iteration order
//! (channel-major, dy-major, dx-major). The output is the `src1`
//! operand of two `dense_matmul_f32_f32_tensor` dispatches that
//! replace the dual-conv patch embed at
//! `vit_gpu_qwen3vl.rs::qwen3vl_dual_conv_patch_embed_cpu_hw`.
//!
//! See ADR-021 §2 K1 / §3 AC-1.
//!
//! Constraints:
//! - `pixel_h % patch_size == 0` and `pixel_w % patch_size == 0`
//!   (mirrors `patch_embed_forward_hw`).
//! - `patch_size` and pixel grid dims must all be > 0.
//!
//! Output layout is byte-identical to the row order
//! `dense_matmul_f32_f32_tensor` consumes for `src1` (`[M, K]`
//! row-major, `M = num_patches`, `K = 3*p²`).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the K1 kernel (embedded at compile time).
pub static IM2COL_2D_3CH_SHADER_SOURCE: &str = include_str!("../shaders/im2col_2d_3ch.metal");

/// Register the K1 shader source. Must be called before
/// [`dispatch_im2col_2d_3ch_f32`] sees this kernel name.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("im2col_2d_3ch_f32", IM2COL_2D_3CH_SHADER_SOURCE);
}

/// GPU-side params struct; matches `Im2col2d3chParams` in
/// `shaders/im2col_2d_3ch.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuIm2col2d3chParams {
    pixel_h: u32,
    pixel_w: u32,
    patch_size: u32,
    nps_x: u32,
    nps_y: u32,
    k_total: u32,
    num_patches: u32,
    _pad: u32,
}

/// Threadgroup size — capped at 256, the standard mlx-native default
/// for 1-D dispatch_threads kernels.
const TG_SIZE: u64 = 256;

/// Dispatch the K1 im2col kernel.
///
/// # Arguments
///
/// * `encoder`    — Command encoder to record the dispatch into.
/// * `registry`   — Kernel registry (must have K1 source registered
///                  via [`register`]).
/// * `device`     — Metal device for pipeline compilation.
/// * `pixels`     — Input buffer, f32, byte_len ≥ `3 * pixel_h * pixel_w * 4`.
///                  Layout: `[3, H, W]` row-major (channel-major).
/// * `output`     — Output buffer, f32, byte_len ≥
///                  `(pixel_h/patch_size) * (pixel_w/patch_size) * 3 * patch_size² * 4`.
///                  Layout: `[num_patches, 3*p²]` row-major.
/// * `pixel_h`    — Image height (pixels). Must be > 0 and a multiple of `patch_size`.
/// * `pixel_w`    — Image width  (pixels). Must be > 0 and a multiple of `patch_size`.
/// * `patch_size` — Square patch side length (pixels). Must be > 0.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] for any zero dimension, a
/// non-multiple of patch_size, or a buffer too small for the output.
pub fn dispatch_im2col_2d_3ch_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    pixels: &MlxBuffer,
    output: &MlxBuffer,
    pixel_h: u32,
    pixel_w: u32,
    patch_size: u32,
) -> Result<()> {
    if patch_size == 0 || pixel_h == 0 || pixel_w == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "im2col_2d_3ch_f32: patch_size ({patch_size}), pixel_h ({pixel_h}), \
             pixel_w ({pixel_w}) must all be > 0"
        )));
    }
    if pixel_h % patch_size != 0 || pixel_w % patch_size != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "im2col_2d_3ch_f32: pixel grid ({pixel_h}x{pixel_w}) must be \
             divisible by patch_size ({patch_size})"
        )));
    }

    let nps_x = pixel_w / patch_size;
    let nps_y = pixel_h / patch_size;
    let p2 = (patch_size as u64) * (patch_size as u64);
    let k_total = 3u64 * p2;
    let num_patches = (nps_x as u64) * (nps_y as u64);
    let total = num_patches * k_total;

    let f32_sz = DType::F32.size_of();
    let expected_pixels_bytes = 3usize * (pixel_h as usize) * (pixel_w as usize) * f32_sz;
    if pixels.byte_len() < expected_pixels_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "im2col_2d_3ch_f32: pixels too small: expected {} bytes for [3, {}, {}], got {}",
            expected_pixels_bytes,
            pixel_h,
            pixel_w,
            pixels.byte_len()
        )));
    }
    let expected_output_bytes = (total as usize) * f32_sz;
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "im2col_2d_3ch_f32: output too small: expected {} bytes for [{}, {}], got {}",
            expected_output_bytes,
            num_patches,
            k_total,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("im2col_2d_3ch_f32", device)?;
    let gpu_params = GpuIm2col2d3chParams {
        pixel_h,
        pixel_w,
        patch_size,
        nps_x: nps_x as u32,
        nps_y: nps_y as u32,
        k_total: k_total as u32,
        num_patches: num_patches as u32,
        _pad: 0,
    };

    let grid = MTLSize::new(total, 1, 1);
    let tg = MTLSize::new(std::cmp::min(TG_SIZE, total), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(pixels)),
            (2, KernelArg::Buffer(output)),
        ],
        grid,
        tg,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU oracle (test-only)
// ---------------------------------------------------------------------------

/// CPU oracle for the K1 kernel. Mirrors the inner-kernel iteration
/// order of `patch_embed_forward_hw` exactly so the parity test can
/// assert byte-identical output.
#[cfg(test)]
pub(crate) fn im2col_2d_3ch_f32_cpu_oracle(
    pixels: &[f32],
    pixel_h: u32,
    pixel_w: u32,
    patch_size: u32,
) -> Vec<f32> {
    let p = patch_size as usize;
    let h = pixel_h as usize;
    let w = pixel_w as usize;
    let nps_x = w / p;
    let nps_y = h / p;
    let num_patches = nps_y * nps_x;
    let k_total = 3 * p * p;
    let mut out = vec![0f32; num_patches * k_total];
    let p2 = p * p;
    let hw = h * w;
    for patch_y in 0..nps_y {
        for patch_x in 0..nps_x {
            let patch_idx = patch_y * nps_x + patch_x;
            let dst_base = patch_idx * k_total;
            for ic in 0..3usize {
                for dy in 0..p {
                    for dx in 0..p {
                        let k = ic * p2 + dy * p + dx;
                        let src_y = patch_y * p + dy;
                        let src_x = patch_x * p + dx;
                        let src_idx = ic * hw + src_y * w + src_x;
                        out[dst_base + k] = pixels[src_idx];
                    }
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

    fn make_pixels_seeded(pixel_h: u32, pixel_w: u32) -> Vec<f32> {
        let n = 3 * (pixel_h as usize) * (pixel_w as usize);
        (0..n)
            .map(|i| (((i as f32) * 0.011_7_f32).sin() * 0.5).clamp(-1.0, 1.0))
            .collect()
    }

    fn run_kernel(
        device: &MlxDevice,
        pixels_host: &[f32],
        pixel_h: u32,
        pixel_w: u32,
        patch_size: u32,
    ) -> Vec<f32> {
        // Mirrors test_graph.rs: separate MlxDevice for the executor;
        // both wrap the same underlying MTLDevice (no Clone on MlxDevice).
        let executor =
            GraphExecutor::new(MlxDevice::new().expect("MlxDevice for executor"));
        let mut session = executor.begin().expect("session begin");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut pixels_buf = device
            .alloc_buffer(
                pixels_host.len() * 4,
                DType::F32,
                vec![3, pixel_h as usize, pixel_w as usize],
            )
            .expect("alloc pixels");
        pixels_buf
            .as_mut_slice::<f32>()
            .expect("pixels mut slice")
            .copy_from_slice(pixels_host);

        let nps_x = (pixel_w / patch_size) as usize;
        let nps_y = (pixel_h / patch_size) as usize;
        let k_total = 3 * (patch_size as usize) * (patch_size as usize);
        let num_patches = nps_x * nps_y;
        let out_buf = device
            .alloc_buffer(
                num_patches * k_total * 4,
                DType::F32,
                vec![num_patches, k_total],
            )
            .expect("alloc out");

        dispatch_im2col_2d_3ch_f32(
            session.encoder_mut(),
            &mut registry,
            device.metal_device(),
            &pixels_buf,
            &out_buf,
            pixel_h,
            pixel_w,
            patch_size,
        )
        .expect("dispatch im2col");

        session.finish().expect("session finish");
        let s: &[f32] = out_buf.as_slice::<f32>().expect("readback");
        s.to_vec()
    }

    /// AC-1 (parity) for K1: GPU output must be BYTE-IDENTICAL to the
    /// CPU oracle. im2col is a pure permutation/copy — no FP reduction
    /// — so byte-identity is the right strict bar.
    #[test]
    fn adr021_k1_im2col_2d_3ch_f32_byte_identical_square_p16() {
        let device = MlxDevice::new().expect("MlxDevice");
        let pixel_h: u32 = 128;
        let pixel_w: u32 = 128;
        let patch_size: u32 = 16;
        let pixels = make_pixels_seeded(pixel_h, pixel_w);
        let oracle = im2col_2d_3ch_f32_cpu_oracle(&pixels, pixel_h, pixel_w, patch_size);
        let gpu = run_kernel(&device, &pixels, pixel_h, pixel_w, patch_size);
        assert_eq!(oracle.len(), gpu.len());
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "K1 byte parity violated at element {i}: oracle={a} gpu={b}"
            );
        }
    }

    /// Rectangular fixture: pixel_h != pixel_w. Catches any
    /// patch_y/patch_x miscomputation that would be invisible to a
    /// square test.
    #[test]
    fn adr021_k1_im2col_2d_3ch_f32_byte_identical_rect_64x128_p16() {
        let device = MlxDevice::new().expect("MlxDevice");
        let pixel_h: u32 = 64;
        let pixel_w: u32 = 128;
        let patch_size: u32 = 16;
        let pixels = make_pixels_seeded(pixel_h, pixel_w);
        let oracle = im2col_2d_3ch_f32_cpu_oracle(&pixels, pixel_h, pixel_w, patch_size);
        let gpu = run_kernel(&device, &pixels, pixel_h, pixel_w, patch_size);
        assert_eq!(oracle.len(), gpu.len());
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K1 byte parity violated at element {i} (rect 64x128): oracle={a} gpu={b}");
        }
    }

    /// Tall fixture: pixel_h > pixel_w. Mirror of the wide rect case.
    #[test]
    fn adr021_k1_im2col_2d_3ch_f32_byte_identical_rect_128x64_p16() {
        let device = MlxDevice::new().expect("MlxDevice");
        let pixel_h: u32 = 128;
        let pixel_w: u32 = 64;
        let patch_size: u32 = 16;
        let pixels = make_pixels_seeded(pixel_h, pixel_w);
        let oracle = im2col_2d_3ch_f32_cpu_oracle(&pixels, pixel_h, pixel_w, patch_size);
        let gpu = run_kernel(&device, &pixels, pixel_h, pixel_w, patch_size);
        assert_eq!(oracle.len(), gpu.len());
        for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "K1 byte parity violated at element {i} (rect 128x64): oracle={a} gpu={b}");
        }
    }

    /// Patch size sweep — tiny (p=4), small (p=8), Qwen3-VL canonical
    /// (p=16). Pin: index math is correct across patch_size values.
    #[test]
    fn adr021_k1_im2col_2d_3ch_f32_byte_identical_patch_sweep() {
        let device = MlxDevice::new().expect("MlxDevice");
        for &p in &[4u32, 8, 16] {
            let pixel_h: u32 = 32;
            let pixel_w: u32 = 48;
            let pixels = make_pixels_seeded(pixel_h, pixel_w);
            let oracle = im2col_2d_3ch_f32_cpu_oracle(&pixels, pixel_h, pixel_w, p);
            let gpu = run_kernel(&device, &pixels, pixel_h, pixel_w, p);
            assert_eq!(oracle.len(), gpu.len());
            for (i, (a, b)) in oracle.iter().zip(gpu.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(),
                    "K1 byte parity violated at element {i} (p={p}): oracle={a} gpu={b}");
            }
        }
    }

    /// Reject zero patch_size + non-multiple grid loud.
    #[test]
    fn adr021_k1_im2col_2d_3ch_f32_input_validation() {
        let device = MlxDevice::new().expect("MlxDevice");
        let executor = GraphExecutor::new(MlxDevice::new().expect("device for executor"));
        let mut session = executor.begin().expect("session");
        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let pixels_buf = device.alloc_buffer(3 * 16 * 16 * 4, DType::F32, vec![3, 16, 16]).unwrap();
        let out_buf = device.alloc_buffer(16 * 16 * 4, DType::F32, vec![16, 16]).unwrap();

        // patch_size = 0
        let err = dispatch_im2col_2d_3ch_f32(
            session.encoder_mut(), &mut registry, device.metal_device(),
            &pixels_buf, &out_buf, 16, 16, 0,
        ).unwrap_err();
        assert!(format!("{err}").contains("> 0"), "got: {err}");

        // pixel_h not divisible
        let err = dispatch_im2col_2d_3ch_f32(
            session.encoder_mut(), &mut registry, device.metal_device(),
            &pixels_buf, &out_buf, 17, 16, 8,
        ).unwrap_err();
        assert!(format!("{err}").contains("divisible"), "got: {err}");
    }
}
