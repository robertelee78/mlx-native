//! ADR-020 iter-11h-misc-3 — elementwise sqrt forward + backward.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SQRT_ELEMENTWISE_SHADER_SOURCE: &str =
    include_str!("../shaders/sqrt_elementwise.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("sqrt_f32", SQRT_ELEMENTWISE_SHADER_SOURCE);
    registry.register_source("sqrt_backward_f32", SQRT_ELEMENTWISE_SHADER_SOURCE);
}

pub fn dispatch_sqrt_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "sqrt_f32";
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(format!("{OP}: empty input")));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: output element_count {} != input {n}",
            output.element_count()
        )));
    }
    if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!("{OP}: must be f32")));
    }
    if params.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!("{OP}: params < 4 bytes")));
    }
    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    encoder.encode(
        pipeline,
        &[(0, input), (1, output), (2, params)],
        MTLSize::new(n_u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_u64), 1, 1),
    );
    Ok(())
}

pub fn dispatch_sqrt_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    y: &MlxBuffer,
    dy: &MlxBuffer,
    dx: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "sqrt_backward_f32";
    let n = y.element_count();
    if dy.element_count() != n || dx.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: shape mismatch (y={n}, dy={}, dx={})",
            dy.element_count(), dx.element_count()
        )));
    }
    if y.dtype() != DType::F32 || dy.dtype() != DType::F32 || dx.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!("{OP}: must be f32")));
    }
    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    encoder.encode(
        pipeline,
        &[(0, y), (1, dy), (2, dx), (3, params)],
        MTLSize::new(n_u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_u64), 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn alloc_f32(d: &MlxDevice, n: usize) -> MlxBuffer {
        let mut b = d.alloc_buffer(n * 4, DType::F32, vec![n]).unwrap();
        b.as_mut_slice::<f32>().unwrap().fill(0.0);
        b
    }
    fn make_params(d: &MlxDevice, n: u32) -> MlxBuffer {
        let mut p = d.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        p.as_mut_slice::<u32>().unwrap()[0] = n;
        p
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 32usize;
        let x: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.3).collect();

        let mut x_buf = alloc_f32(&device, n);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let y_buf = alloc_f32(&device, n);
        let p = make_params(&device, n as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_sqrt_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &y_buf, &p,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            let cpu = (x[i] as f64).sqrt() as f32;
            assert!(
                (gpu[i] - cpu).abs() < 1e-6 * cpu.abs().max(1.0),
                "y[{i}]: gpu={} cpu={} (x={})",
                gpu[i], cpu, x[i]
            );
        }
    }

    /// FD falsifier: loss = sum(sqrt(x)).  dx[i] = 1/(2·sqrt(x[i])).
    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 16usize;
        let x: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();

        let mut x_buf = alloc_f32(&device, n);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let y_buf = alloc_f32(&device, n);
        let p = make_params(&device, n as u32);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_sqrt_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &y_buf, &p,
        ).unwrap();
        // RAW barrier (per feedback_metal_raw_barrier_per_dispatch).
        encoder.memory_barrier();

        let dy_ones = vec![1.0f32; n];
        let mut dy_buf = alloc_f32(&device, n);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy_ones);
        let dx_buf = alloc_f32(&device, n);
        dispatch_sqrt_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &y_buf, &dy_buf, &dx_buf, &p,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let dx = dx_buf.as_slice::<f32>().unwrap().to_vec();

        let h = 1e-3f64;
        for i in 0..n {
            let mut xp = x.clone(); xp[i] += h as f32;
            let mut xm = x.clone(); xm[i] -= h as f32;
            let lp: f64 = xp.iter().map(|v| (*v as f64).sqrt()).sum();
            let lm: f64 = xm.iter().map(|v| (*v as f64).sqrt()).sum();
            let fd = (lp - lm) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (dx[i] as f64 - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={}", dx[i], fd
            );
        }
    }

    #[test]
    fn rejects_size_mismatch() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let x = alloc_f32(&device, 16);
        let y = alloc_f32(&device, 8);
        let p = make_params(&device, 16);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_sqrt_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x, &y, &p,
        );
        assert!(res.is_err());
    }
}
