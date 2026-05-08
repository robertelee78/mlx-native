//! ADR-020 iter-11h-misc-1 — elementwise division forward + backward.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static DIVIDE_ELEMENTWISE_SHADER_SOURCE: &str =
    include_str!("../shaders/divide_elementwise.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("divide_f32", DIVIDE_ELEMENTWISE_SHADER_SOURCE);
    registry.register_source(
        "divide_backward_f32",
        DIVIDE_ELEMENTWISE_SHADER_SOURCE,
    );
}

pub fn dispatch_divide_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    b: &MlxBuffer,
    y: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "divide_f32";
    let n = a.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(format!("{OP}: empty input")));
    }
    if b.element_count() != n || y.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: shape mismatch (a={}, b={}, y={})",
            n, b.element_count(), y.element_count()
        )));
    }
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 || y.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!("{OP}: must be f32")));
    }
    if params.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: params < 4 bytes"
        )));
    }
    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    encoder.encode(
        pipeline,
        &[(0, a), (1, b), (2, y), (3, params)],
        MTLSize::new(n_u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_u64), 1, 1),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_divide_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    b: &MlxBuffer,
    y: &MlxBuffer,
    dy: &MlxBuffer,
    da: &MlxBuffer,
    db: &MlxBuffer,
    params: &MlxBuffer,
) -> Result<()> {
    const OP: &str = "divide_backward_f32";
    let n = b.element_count();
    if y.element_count() != n
        || dy.element_count() != n
        || da.element_count() != n
        || db.element_count() != n
    {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: shape mismatch n={n}, b/y/dy/da/db must match"
        )));
    }
    if b.dtype() != DType::F32 || y.dtype() != DType::F32 || dy.dtype() != DType::F32
        || da.dtype() != DType::F32 || db.dtype() != DType::F32
    {
        return Err(MlxError::InvalidArgument(format!("{OP}: must be f32")));
    }
    let pipeline = registry.get_pipeline(OP, device)?;
    let n_u64 = n as u64;
    encoder.encode(
        pipeline,
        &[(0, b), (1, y), (2, dy), (3, da), (4, db), (5, params)],
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
        let mut bx = d.alloc_buffer(n * 4, DType::F32, vec![n]).unwrap();
        bx.as_mut_slice::<f32>().unwrap().fill(0.0);
        bx
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
        let a: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.07).collect();

        let mut a_buf = alloc_f32(&device, n);
        a_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&a);
        let mut b_buf = alloc_f32(&device, n);
        b_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&b);
        let y_buf = alloc_f32(&device, n);
        let p = make_params(&device, n as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_divide_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &a_buf, &b_buf, &y_buf, &p,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            let cpu = a[i] / b[i];
            assert!(
                (gpu[i] - cpu).abs() < 1e-6 * cpu.abs().max(1.0),
                "y[{i}]: gpu={} cpu={}",
                gpu[i], cpu
            );
        }
    }

    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 16usize;
        let a: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.07).collect();
        let dy: Vec<f32> = vec![1.0; n];

        let mut a_buf = alloc_f32(&device, n);
        a_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&a);
        let mut b_buf = alloc_f32(&device, n);
        b_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&b);
        let y_buf = alloc_f32(&device, n);
        let mut dy_buf = alloc_f32(&device, n);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let da_buf = alloc_f32(&device, n);
        let db_buf = alloc_f32(&device, n);
        let p = make_params(&device, n as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_divide_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &a_buf, &b_buf, &y_buf, &p,
        ).unwrap();
        // RAW barrier — backward reads y written by forward.  Apple Metal
        // compute encoders run threadgroups in parallel by default;
        // without barriers the backward kernel reads zero-init y values
        // (per `feedback_metal_raw_barrier_per_dispatch` memory).
        encoder.memory_barrier();
        dispatch_divide_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &b_buf, &y_buf, &dy_buf, &da_buf, &db_buf, &p,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let da = da_buf.as_slice::<f32>().unwrap().to_vec();
        let db = db_buf.as_slice::<f32>().unwrap().to_vec();

        // FD on a and b. Loss = sum(a/b).
        let h = 1e-3f64;
        let loss = |aa: &[f32], bb: &[f32]| -> f64 {
            (0..n).map(|i| aa[i] as f64 / bb[i] as f64).sum::<f64>()
        };
        for i in 0..n {
            let mut ap = a.clone(); ap[i] += h as f32;
            let mut am = a.clone(); am[i] -= h as f32;
            let fd = (loss(&ap, &b) - loss(&am, &b)) / (2.0 * h);
            let tol = 1e-3 * fd.abs().max(1.0);
            assert!(
                (da[i] as f64 - fd).abs() < tol,
                "FD a[{i}]: analytic={} fd={}", da[i], fd
            );
        }
        for i in 0..n {
            let mut bp = b.clone(); bp[i] += h as f32;
            let mut bm = b.clone(); bm[i] -= h as f32;
            let fd = (loss(&a, &bp) - loss(&a, &bm)) / (2.0 * h);
            let tol = 1e-3 * fd.abs().max(1.0);
            assert!(
                (db[i] as f64 - fd).abs() < tol,
                "FD b[{i}]: analytic={} fd={}", db[i], fd
            );
        }
    }

    #[test]
    fn rejects_size_mismatch() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let a = alloc_f32(&device, 16);
        let b = alloc_f32(&device, 8); // wrong
        let y = alloc_f32(&device, 16);
        let p = make_params(&device, 16);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_divide_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &a, &b, &y, &p,
        );
        assert!(res.is_err());
    }
}
