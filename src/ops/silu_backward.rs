//! Elementwise SiLU (swish) forward + reverse-mode backward.
//!
//! Used by hf2q's ADR-020 Track 1 SwiGLU FFN on GpuTape (iter-11b).
//!
//! Forward:  `silu(x) = x · sigmoid(x)`
//! Backward: `dx[i] = dy[i] · silu'(x[i])`
//!           where `silu'(x) = sigmoid(x) · (1 + x · (1 − sigmoid(x)))`
//!
//! Note: mlx-native already has `silu_mul_f32` (= `silu(gate) * up`)
//! used by the inference forward.  This module adds the standalone
//! `silu_f32` + matching backward kernel needed for autograd.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SILU_BACKWARD_SHADER_SOURCE: &str =
    include_str!("../shaders/silu_backward.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("silu_f32", SILU_BACKWARD_SHADER_SOURCE);
    registry.register_source("silu_backward_f32", SILU_BACKWARD_SHADER_SOURCE);
}

/// Encode `output[i] = silu(input[i]) = input[i] · sigmoid(input[i])`.
///
/// `params_buf` must be at least 4 bytes (1 × u32: n).
pub fn dispatch_silu_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
) -> Result<()> {
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "silu_f32: input must have at least one element".into(),
        ));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "silu_f32: output element count {} != input element count {n}",
            output.element_count()
        )));
    }
    for (label, buf) in [("input", input), ("output", output)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "silu_f32: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "silu_f32: params_buf too small (need 4 bytes for u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("silu_f32", device)?;
    let thread_count = n as u64;
    let tg_size = std::cmp::min(256, thread_count);
    encoder.encode(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Encode `dx[i] = dy[i] · silu'(x[i])`.  `x` is the FORWARD INPUT.
///
/// `params_buf` must be at least 4 bytes (1 × u32: n).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_silu_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    dy: &MlxBuffer,
    dx: &MlxBuffer,
    params_buf: &MlxBuffer,
) -> Result<()> {
    let n = x.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "silu_backward_f32: x must have at least one element".into(),
        ));
    }
    for (label, buf) in [("x", x), ("dy", dy), ("dx", dx)] {
        if buf.element_count() != n {
            return Err(MlxError::InvalidArgument(format!(
                "silu_backward_f32: {label} element count {} != x element count {n}",
                buf.element_count(),
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "silu_backward_f32: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "silu_backward_f32: params_buf too small (need 4 bytes for u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("silu_backward_f32", device)?;
    let thread_count = n as u64;
    let tg_size = std::cmp::min(256, thread_count);
    encoder.encode(
        pipeline,
        &[(0, x), (1, dy), (2, dx), (3, params_buf)],
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn silu_cpu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xv| xv / (1.0 + (-xv).exp())).collect()
    }

    fn silu_backward_cpu(x: &[f32], dy: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(dy.iter())
            .map(|(&xv, &dyv)| {
                let s = 1.0 / (1.0 + (-xv).exp());
                let deriv = s * (1.0 + xv * (1.0 - s));
                dyv * deriv
            })
            .collect()
    }

    fn run_silu_forward(input: &[f32]) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let n = input.len();
        let mut in_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc in");
        in_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(input);
        let out_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc out");
        let mut params = device.alloc_buffer(4, DType::F32, vec![1]).expect("params");
        params.as_mut_slice::<u32>().unwrap()[0] = n as u32;
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_silu_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
            &params,
        )
        .expect("dispatch silu");
        encoder.commit_and_wait().expect("commit");
        out_buf.as_slice::<f32>().unwrap().to_vec()
    }

    fn run_silu_backward(input: &[f32], dy: &[f32]) -> Vec<f32> {
        let device = MlxDevice::new().expect("device");
        let n = input.len();
        let mut x_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc x");
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(input);
        let mut dy_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc dy");
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(dy);
        let dx_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc dx");
        let mut params = device.alloc_buffer(4, DType::F32, vec![1]).expect("params");
        params.as_mut_slice::<u32>().unwrap()[0] = n as u32;
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_silu_backward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &dy_buf,
            &dx_buf,
            &params,
        )
        .expect("dispatch silu backward");
        encoder.commit_and_wait().expect("commit");
        dx_buf.as_slice::<f32>().unwrap().to_vec()
    }

    fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}: i={i}: gpu={g} cpu={c} diff={diff}"
            );
        }
    }

    #[test]
    fn silu_forward_parity_with_cpu() {
        let input: Vec<f32> = (0..256)
            .map(|i| (i as f32 - 128.0) * 0.05)
            .collect();
        let gpu = run_silu_forward(&input);
        let cpu = silu_cpu(&input);
        assert_close("silu forward", &gpu, &cpu, 1e-6, 1e-7);
    }

    #[test]
    fn silu_forward_handles_extremes() {
        // Values near the saturation regions: very negative (sigmoid → 0)
        // and very positive (sigmoid → 1).
        let input = vec![-20.0_f32, -10.0, -5.0, -0.5, 0.0, 0.5, 5.0, 10.0, 20.0];
        let gpu = run_silu_forward(&input);
        let cpu = silu_cpu(&input);
        assert_close("silu extremes", &gpu, &cpu, 1e-5, 1e-6);
        // x=0 → silu(0) = 0 · sigmoid(0) = 0 · 0.5 = 0.
        assert_eq!(gpu[4], 0.0);
    }

    #[test]
    fn silu_backward_parity_with_cpu() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let dy: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
        let gpu = run_silu_backward(&input, &dy);
        let cpu = silu_backward_cpu(&input, &dy);
        assert_close("silu backward", &gpu, &cpu, 1e-5, 1e-6);
    }

    #[test]
    fn silu_backward_finite_diff_falsifier() {
        // Acid test: backward analytical gradient must match
        // central finite-difference of the forward.
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.07).collect();
        let h = 1e-3_f32;
        // Pick a few probe indices spanning the saturation regimes.
        for &probe in &[0usize, 7, 15, 16, 24, 31] {
            let mut x_plus = input.clone();
            let mut x_minus = input.clone();
            x_plus[probe] += h;
            x_minus[probe] -= h;
            let f_plus = silu_cpu(&x_plus)[probe];
            let f_minus = silu_cpu(&x_minus)[probe];
            let fd = (f_plus - f_minus) / (2.0 * h);
            // Use dy = e_probe (one-hot) so dx[probe] equals exactly
            // silu'(x[probe]).
            let mut dy = vec![0f32; input.len()];
            dy[probe] = 1.0;
            let dx_gpu = run_silu_backward(&input, &dy)[probe];
            let diff = (dx_gpu - fd).abs();
            let scale = dx_gpu.abs().max(fd.abs()).max(1.0);
            assert!(
                diff <= 1e-3 || diff / scale <= 5e-3,
                "silu finite-diff falsifier failed at probe {probe}: \
                 fd={fd} analytical={dx_gpu} diff={diff}"
            );
        }
    }
}
