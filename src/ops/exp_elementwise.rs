//! ADR-020 iter-11h-c1 — elementwise exp forward + backward.
//!
//! Forward: `y[i] = exp(x[i])`
//! Backward: `dx[i] = dy[i] · y[i]` (caller passes y, the forward
//! output, NOT x — autograd-canonical pattern that avoids recompute).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static EXP_ELEMENTWISE_SHADER_SOURCE: &str =
    include_str!("../shaders/exp_elementwise.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("exp_f32", EXP_ELEMENTWISE_SHADER_SOURCE);
    registry.register_source("exp_backward_f32", EXP_ELEMENTWISE_SHADER_SOURCE);
}

pub fn dispatch_exp_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "exp_f32";
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: input must have at least one element"
        )));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: output element_count {} != input element_count {n}",
            output.element_count()
        )));
    }
    if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: input/output must be f32"
        )));
    }
    if params.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: params < 4 bytes (need 1 × u32 = n)"
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    let tg = std::cmp::min(256, n_u64);
    encoder.encode(
        pipeline,
        &[(0, input), (1, output), (2, params)],
        MTLSize::new(n_u64, 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    Ok(())
}

pub fn dispatch_exp_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    y: &MlxBuffer,
    dy: &MlxBuffer,
    dx: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "exp_backward_f32";
    let n = y.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: y must have at least one element"
        )));
    }
    if dy.element_count() != n || dx.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: dy/dx element_count must match y ({n})"
        )));
    }
    if y.dtype() != DType::F32 || dy.dtype() != DType::F32 || dx.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: y/dy/dx must be f32"
        )));
    }
    if params.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: params < 4 bytes (need 1 × u32 = n)"
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    let tg = std::cmp::min(256, n_u64);
    encoder.encode(
        pipeline,
        &[(0, y), (1, dy), (2, dx), (3, params)],
        MTLSize::new(n_u64, 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
        let mut b = device.alloc_buffer(n * 4, DType::F32, vec![n]).unwrap();
        b.as_mut_slice::<f32>().unwrap().fill(0.0);
        b
    }

    fn make_params(device: &MlxDevice, n: u32) -> MlxBuffer {
        let mut p = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        p.as_mut_slice::<u32>().unwrap()[0] = n;
        p
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 64usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.073 - 1.5)).collect();

        let mut x_buf = alloc_f32(&device, n);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let y_buf = alloc_f32(&device, n);
        let params = make_params(&device, n as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_exp_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &y_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            let cpu = (x[i] as f64).exp() as f32;
            assert!(
                (gpu[i] - cpu).abs() < 1e-5 * cpu.abs().max(1.0),
                "exp y[{i}]: gpu={} cpu={} (x={})",
                gpu[i], cpu, x[i]
            );
        }
    }

    #[test]
    fn backward_dx_equals_dy_times_y() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 32usize;
        let y: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.07).collect();
        let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13 - 0.5).sin()).collect();

        let mut y_buf = alloc_f32(&device, n);
        y_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&y);
        let mut dy_buf = alloc_f32(&device, n);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let dx_buf = alloc_f32(&device, n);
        let params = make_params(&device, n as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_exp_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &y_buf, &dy_buf, &dx_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = dx_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            let expected = dy[i] * y[i];
            assert!(
                (gpu[i] - expected).abs() < 1e-6 * expected.abs().max(1.0),
                "exp dx[{i}]: gpu={} expected={}",
                gpu[i], expected
            );
        }
    }

    /// Finite-difference falsifier: `loss = sum(exp(x))`.  Analytic
    /// gradient is `dx = exp(x)`.  FD must match within 1% rel tol.
    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 16usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.043 - 0.5)).collect();

        // Forward via mlx-native.
        let mut x_buf = alloc_f32(&device, n);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let y_buf = alloc_f32(&device, n);
        let params = make_params(&device, n as u32);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_exp_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &y_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let y = y_buf.as_slice::<f32>().unwrap().to_vec();

        // Analytic backward: dy = ones, dx = exp(x).
        let dy_ones = vec![1.0f32; n];
        let mut dy_buf = alloc_f32(&device, n);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy_ones);
        let dx_buf = alloc_f32(&device, n);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_exp_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &y_buf, &dy_buf, &dx_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let dx = dx_buf.as_slice::<f32>().unwrap().to_vec();

        // FD: loss(xp) - loss(xm) / (2h) for each x[i].
        let h = 1e-4f64;
        for i in 0..n {
            let mut xp = x.clone();
            xp[i] += h as f32;
            let mut xm = x.clone();
            xm[i] -= h as f32;
            let loss_p: f64 = xp.iter().map(|v| (*v as f64).exp()).sum();
            let loss_m: f64 = xm.iter().map(|v| (*v as f64).exp()).sum();
            let fd = (loss_p - loss_m) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (dx[i] as f64 - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={} (y={})",
                dx[i], fd, y[i]
            );
        }
    }

    #[test]
    fn rejects_size_mismatch() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let x = alloc_f32(&device, 16);
        let y = alloc_f32(&device, 8); // wrong size
        let params = make_params(&device, 16);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_exp_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x, &y, &params,
        );
        assert!(res.is_err());
    }
}
