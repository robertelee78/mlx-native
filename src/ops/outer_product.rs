//! ADR-020 iter-11h-c2 — vector outer product forward + backward.
//!
//! Forward:  `y[i, j] = lhs[i] · rhs[j]`
//! Backward: `dlhs[i] = Σ_j dy[i, j] · rhs[j]`
//!           `drhs[j] = Σ_i dy[i, j] · lhs[i]`
//!
//! Distinct from matmul: matmul kernel has a 32-element floor on each
//! dim (`M, N, K ≥ 32` for dW backward); outer products have
//! inner-dim = 1, falling below that floor.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static OUTER_PRODUCT_SHADER_SOURCE: &str =
    include_str!("../shaders/outer_product.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("outer_product_f32", OUTER_PRODUCT_SHADER_SOURCE);
    registry.register_source(
        "outer_product_backward_lhs_f32",
        OUTER_PRODUCT_SHADER_SOURCE,
    );
    registry.register_source(
        "outer_product_backward_rhs_f32",
        OUTER_PRODUCT_SHADER_SOURCE,
    );
}

fn validate_dims(op: &str, n: u32, m: u32, params: &MlxBuffer) -> Result<()> {
    if n == 0 || m == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: N and M must both be > 0 (got {n}, {m})"
        )));
    }
    if params.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: params < 8 bytes (need 2 × u32)"
        )));
    }
    Ok(())
}

pub fn dispatch_outer_product_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    lhs: &MlxBuffer,
    rhs: &MlxBuffer,
    y: &MlxBuffer,
    params: &MlxBuffer,
    n: u32,
    m: u32,
) -> Result<()> {
    const OP: &str = "outer_product_f32";
    validate_dims(OP, n, m, params)?;
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 || y.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: all buffers must be f32"
        )));
    }
    if lhs.element_count() != n as usize {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: lhs.element_count {} != N {n}",
            lhs.element_count()
        )));
    }
    if rhs.element_count() != m as usize {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: rhs.element_count {} != M {m}",
            rhs.element_count()
        )));
    }
    if y.element_count() != (n as usize) * (m as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: y.element_count {} != N*M = {}",
            y.element_count(),
            n as usize * m as usize
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, lhs), (1, rhs), (2, y), (3, params)],
        MTLSize::new(n as u64, m as u64, 1),
        MTLSize::new(
            std::cmp::min(16, n as u64),
            std::cmp::min(16, m as u64),
            1,
        ),
    );
    Ok(())
}

pub fn dispatch_outer_product_backward_lhs_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    rhs: &MlxBuffer,
    dlhs: &MlxBuffer,
    params: &MlxBuffer,
    n: u32,
    m: u32,
) -> Result<()> {
    const OP: &str = "outer_product_backward_lhs_f32";
    validate_dims(OP, n, m, params)?;
    if dy.element_count() != (n as usize) * (m as usize)
        || rhs.element_count() != m as usize
        || dlhs.element_count() != n as usize
    {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: shape mismatch (dy={}, rhs={}, dlhs={})",
            dy.element_count(),
            rhs.element_count(),
            dlhs.element_count()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, dy), (1, rhs), (2, dlhs), (3, params)],
        MTLSize::new(n as u64, 1, 1),
        MTLSize::new(std::cmp::min(64, n as u64), 1, 1),
    );
    Ok(())
}

pub fn dispatch_outer_product_backward_rhs_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    lhs: &MlxBuffer,
    drhs: &MlxBuffer,
    params: &MlxBuffer,
    n: u32,
    m: u32,
) -> Result<()> {
    const OP: &str = "outer_product_backward_rhs_f32";
    validate_dims(OP, n, m, params)?;
    if dy.element_count() != (n as usize) * (m as usize)
        || lhs.element_count() != n as usize
        || drhs.element_count() != m as usize
    {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: shape mismatch (dy={}, lhs={}, drhs={})",
            dy.element_count(),
            lhs.element_count(),
            drhs.element_count()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, dy), (1, lhs), (2, drhs), (3, params)],
        MTLSize::new(m as u64, 1, 1),
        MTLSize::new(std::cmp::min(64, m as u64), 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn alloc_f32(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        let mut b = device.alloc_buffer(n * 4, DType::F32, shape).unwrap();
        b.as_mut_slice::<f32>().unwrap().fill(0.0);
        b
    }

    fn make_params(device: &MlxDevice, n: u32, m: u32) -> MlxBuffer {
        let mut p = device.alloc_buffer(8, DType::U32, vec![2]).unwrap();
        p.as_mut_slice::<u32>().unwrap().copy_from_slice(&[n, m]);
        p
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 8usize;
        let m = 5usize;
        let lhs: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let rhs: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.137 - 0.3)).collect();

        let mut lhs_buf = alloc_f32(&device, n, vec![n]);
        lhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&lhs);
        let mut rhs_buf = alloc_f32(&device, m, vec![m]);
        rhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&rhs);
        let y_buf = alloc_f32(&device, n * m, vec![n, m]);
        let params = make_params(&device, n as u32, m as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_outer_product_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &lhs_buf, &rhs_buf, &y_buf, &params, n as u32, m as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            for j in 0..m {
                let expected = lhs[i] * rhs[j];
                assert!(
                    (gpu[i * m + j] - expected).abs() < 1e-6 * expected.abs().max(1.0),
                    "y[{i},{j}]: gpu={} expected={}",
                    gpu[i * m + j], expected
                );
            }
        }
    }

    #[test]
    fn backward_dlhs_drhs_match_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 8usize;
        let m = 5usize;
        let lhs: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let rhs: Vec<f32> = (0..m).map(|i| 0.2 + (i as f32) * 0.07).collect();
        let dy: Vec<f32> = (0..(n * m)).map(|i| ((i as f32) * 0.131 - 0.4).sin()).collect();

        let mut lhs_buf = alloc_f32(&device, n, vec![n]);
        lhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&lhs);
        let mut rhs_buf = alloc_f32(&device, m, vec![m]);
        rhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&rhs);
        let mut dy_buf = alloc_f32(&device, n * m, vec![n, m]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let dlhs_buf = alloc_f32(&device, n, vec![n]);
        let drhs_buf = alloc_f32(&device, m, vec![m]);
        let params = make_params(&device, n as u32, m as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_outer_product_backward_lhs_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &rhs_buf, &dlhs_buf, &params, n as u32, m as u32,
        ).unwrap();
        dispatch_outer_product_backward_rhs_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &lhs_buf, &drhs_buf, &params, n as u32, m as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let dlhs = dlhs_buf.as_slice::<f32>().unwrap();
        let drhs = drhs_buf.as_slice::<f32>().unwrap();
        for i in 0..n {
            let expected: f64 = (0..m).map(|j| dy[i * m + j] as f64 * rhs[j] as f64).sum();
            assert!(
                (dlhs[i] as f64 - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "dlhs[{i}]: gpu={} expected={}",
                dlhs[i], expected
            );
        }
        for j in 0..m {
            let expected: f64 = (0..n).map(|i| dy[i * m + j] as f64 * lhs[i] as f64).sum();
            assert!(
                (drhs[j] as f64 - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "drhs[{j}]: gpu={} expected={}",
                drhs[j], expected
            );
        }
    }

    /// FD falsifier: loss = sum(outer(lhs, rhs)) = sum(lhs) * sum(rhs).
    /// Analytic dlhs[i] = sum(rhs), drhs[j] = sum(lhs). Verify FD matches.
    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let n = 6usize;
        let m = 4usize;
        let lhs: Vec<f32> = (0..n).map(|i| 0.3 + (i as f32) * 0.07).collect();
        let rhs: Vec<f32> = (0..m).map(|i| 0.5 + (i as f32) * 0.05).collect();

        let mut lhs_buf = alloc_f32(&device, n, vec![n]);
        lhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&lhs);
        let mut rhs_buf = alloc_f32(&device, m, vec![m]);
        rhs_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&rhs);
        let dy_ones = vec![1.0f32; n * m];
        let mut dy_buf = alloc_f32(&device, n * m, vec![n, m]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy_ones);
        let dlhs_buf = alloc_f32(&device, n, vec![n]);
        let drhs_buf = alloc_f32(&device, m, vec![m]);
        let params = make_params(&device, n as u32, m as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_outer_product_backward_lhs_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &rhs_buf, &dlhs_buf, &params, n as u32, m as u32,
        ).unwrap();
        dispatch_outer_product_backward_rhs_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &lhs_buf, &drhs_buf, &params, n as u32, m as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let dlhs = dlhs_buf.as_slice::<f32>().unwrap().to_vec();
        let drhs = drhs_buf.as_slice::<f32>().unwrap().to_vec();

        let h = 1e-3f64;
        let loss = |l: &[f32], r: &[f32]| -> f64 {
            let mut s = 0.0f64;
            for i in 0..n { for j in 0..m { s += l[i] as f64 * r[j] as f64; } }
            s
        };
        for i in 0..n {
            let mut lp = lhs.clone(); lp[i] += h as f32;
            let mut lm = lhs.clone(); lm[i] -= h as f32;
            let fd = (loss(&lp, &rhs) - loss(&lm, &rhs)) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (dlhs[i] as f64 - fd).abs() < tol,
                "FD lhs[{i}]: analytic={} fd={}", dlhs[i], fd
            );
        }
        for j in 0..m {
            let mut rp = rhs.clone(); rp[j] += h as f32;
            let mut rm = rhs.clone(); rm[j] -= h as f32;
            let fd = (loss(&lhs, &rp) - loss(&lhs, &rm)) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (drhs[j] as f64 - fd).abs() < tol,
                "FD rhs[{j}]: analytic={} fd={}", drhs[j], fd
            );
        }
    }

    #[test]
    fn rejects_zero_dims() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let lhs = alloc_f32(&device, 1, vec![1]);
        let rhs = alloc_f32(&device, 1, vec![1]);
        let y = alloc_f32(&device, 1, vec![1, 1]);
        let params = make_params(&device, 0, 1);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_outer_product_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &lhs, &rhs, &y, &params, 0, 1,
        );
        assert!(res.is_err());
    }
}
