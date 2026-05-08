//! ADR-020 iter-11h-e1 — take_along_axis (gather) + scatter-backward
//! for the GpuTape autograd pipeline.  Forward gathers values along
//! the last axis using a precomputed (non-differentiable) index
//! buffer; backward scatters gradients back into a zero-initialised
//! dx buffer.
//!
//! Used by MoE router on GpuTape (iter-11h-e):
//!   y = take_along_axis(softmax(gate(x)), top_k_indices, axis=-1)

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static TAKE_ALONG_AXIS_SHADER_SOURCE: &str =
    include_str!("../shaders/take_along_axis.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("take_along_axis_f32", TAKE_ALONG_AXIS_SHADER_SOURCE);
    registry.register_source(
        "take_along_axis_backward_f32",
        TAKE_ALONG_AXIS_SHADER_SOURCE,
    );
}

fn validate(
    op: &str,
    rows: u32,
    cols: u32,
    k: u32,
    a: &MlxBuffer,
    indices: &MlxBuffer,
    out: &MlxBuffer,
    params: &MlxBuffer,
    expected_a: usize,
    expected_out: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 || k == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: rows, cols, k must all be > 0 (got {rows}, {cols}, {k})"
        )));
    }
    if k > cols {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: k ({k}) > cols ({cols})"
        )));
    }
    if a.dtype() != DType::F32 || out.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: a/out must be f32"
        )));
    }
    if indices.dtype() != DType::U32 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: indices dtype {} not u32",
            indices.dtype()
        )));
    }
    if a.element_count() != expected_a {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: a element_count {} != {expected_a}",
            a.element_count()
        )));
    }
    if indices.element_count() != (rows as usize) * (k as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: indices element_count {} != rows*k = {}",
            indices.element_count(),
            (rows as usize) * (k as usize)
        )));
    }
    if out.element_count() != expected_out {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: out element_count {} != {expected_out}",
            out.element_count()
        )));
    }
    if params.byte_len() < 12 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: params < 12 bytes (need 3 × u32)"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_take_along_axis_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    indices: &MlxBuffer,
    y: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<()> {
    const OP: &str = "take_along_axis_f32";
    let r = rows as usize;
    let c = cols as usize;
    let k_us = k as usize;
    validate(OP, rows, cols, k, x, indices, y, params, r * c, r * k_us)?;

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, x), (1, indices), (2, y), (3, params)],
        MTLSize::new(rows as u64, k as u64, 1),
        MTLSize::new(
            std::cmp::min(16, rows as u64),
            std::cmp::min(16, k as u64),
            1,
        ),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_take_along_axis_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    indices: &MlxBuffer,
    dx: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<()> {
    const OP: &str = "take_along_axis_backward_f32";
    let r = rows as usize;
    let c = cols as usize;
    let k_us = k as usize;
    validate(OP, rows, cols, k, dx, indices, dy, params, r * c, r * k_us)?;

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, dy), (1, indices), (2, dx), (3, params)],
        MTLSize::new(rows as u64, k as u64, 1),
        MTLSize::new(
            std::cmp::min(16, rows as u64),
            std::cmp::min(16, k as u64),
            1,
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn alloc_f32(d: &MlxDevice, n: usize, sh: Vec<usize>) -> MlxBuffer {
        let mut b = d.alloc_buffer(n * 4, DType::F32, sh).unwrap();
        b.as_mut_slice::<f32>().unwrap().fill(0.0);
        b
    }
    fn alloc_u32(d: &MlxDevice, n: usize, sh: Vec<usize>) -> MlxBuffer {
        let mut b = d.alloc_buffer(n * 4, DType::U32, sh).unwrap();
        b.as_mut_slice::<u32>().unwrap().fill(0);
        b
    }
    fn make_params(d: &MlxDevice, rows: u32, cols: u32, k: u32) -> MlxBuffer {
        let mut p = d.alloc_buffer(12, DType::U32, vec![3]).unwrap();
        p.as_mut_slice::<u32>().unwrap().copy_from_slice(&[rows, cols, k]);
        p
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let rows = 4;
        let cols = 8;
        let k = 3;
        let x: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.137 - 0.4).sin() * 0.7)
            .collect();
        // Per-row top-K indices (must be distinct within a row, and
        // < cols).  Hand-pick non-trivial values.
        let indices: Vec<u32> = vec![
            0, 3, 7,
            1, 4, 6,
            2, 5, 0,
            7, 0, 4,
        ];

        let mut x_buf = alloc_f32(&device, rows * cols, vec![rows, cols]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut idx_buf = alloc_u32(&device, rows * k, vec![rows, k]);
        idx_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&indices);
        let y_buf = alloc_f32(&device, rows * k, vec![rows, k]);
        let params = make_params(&device, rows as u32, cols as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_take_along_axis_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &idx_buf, &y_buf, &params,
            rows as u32, cols as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        for r in 0..rows {
            for j in 0..k {
                let idx = indices[r * k + j] as usize;
                let expected = x[r * cols + idx];
                assert!(
                    (gpu[r * k + j] - expected).abs() < 1e-6 * expected.abs().max(1.0),
                    "y[{r},{j}]: gpu={} expected={} (idx={})",
                    gpu[r * k + j], expected, idx
                );
            }
        }
    }

    #[test]
    fn backward_scatter_matches_cpu_oracle() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let rows = 3;
        let cols = 6;
        let k = 2;
        let dy: Vec<f32> = (0..(rows * k))
            .map(|i| ((i as f32) * 0.231 + 0.1).sin() * 0.6)
            .collect();
        // Distinct indices per row.
        let indices: Vec<u32> = vec![
            0, 4,
            1, 5,
            2, 3,
        ];

        let mut dy_buf = alloc_f32(&device, rows * k, vec![rows, k]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let mut idx_buf = alloc_u32(&device, rows * k, vec![rows, k]);
        idx_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&indices);
        let dx_buf = alloc_f32(&device, rows * cols, vec![rows, cols]);
        let params = make_params(&device, rows as u32, cols as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_take_along_axis_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &idx_buf, &dx_buf, &params,
            rows as u32, cols as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = dx_buf.as_slice::<f32>().unwrap();
        // Build CPU oracle.
        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for j in 0..k {
                let idx = indices[r * k + j] as usize;
                expected[r * cols + idx] = dy[r * k + j];
            }
        }
        for i in 0..(rows * cols) {
            assert!(
                (gpu[i] - expected[i]).abs() < 1e-6,
                "dx[{i}]: gpu={} expected={}",
                gpu[i], expected[i]
            );
        }
    }

    /// FD falsifier: loss = sum(take_along_axis(x, indices)).  Analytic
    /// dx[r, c] = 1 if c is in row r's top-K else 0.  FD must match.
    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let rows = 4;
        let cols = 6;
        let k = 2;
        let x: Vec<f32> = (0..(rows * cols))
            .map(|i| 0.3 + (i as f32) * 0.013)
            .collect();
        let indices: Vec<u32> = vec![
            0, 3,
            1, 5,
            2, 4,
            0, 4,
        ];

        // Forward to get y.
        let mut x_buf = alloc_f32(&device, rows * cols, vec![rows, cols]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut idx_buf = alloc_u32(&device, rows * k, vec![rows, k]);
        idx_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&indices);
        let y_buf = alloc_f32(&device, rows * k, vec![rows, k]);
        let params = make_params(&device, rows as u32, cols as u32, k as u32);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_take_along_axis_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &idx_buf, &y_buf, &params,
            rows as u32, cols as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        // Analytic dx via dy=ones.
        let dy_ones = vec![1.0f32; rows * k];
        let mut dy_buf = alloc_f32(&device, rows * k, vec![rows, k]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy_ones);
        let dx_buf = alloc_f32(&device, rows * cols, vec![rows, cols]);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_take_along_axis_backward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &idx_buf, &dx_buf, &params,
            rows as u32, cols as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let dx = dx_buf.as_slice::<f32>().unwrap().to_vec();

        // FD on every x[i].
        let h = 1e-3f64;
        let loss = |x_in: &[f32]| -> f64 {
            let mut s = 0.0f64;
            for r in 0..rows {
                for j in 0..k {
                    s += x_in[r * cols + indices[r * k + j] as usize] as f64;
                }
            }
            s
        };
        for i in 0..(rows * cols) {
            let mut xp = x.clone(); xp[i] += h as f32;
            let mut xm = x.clone(); xm[i] -= h as f32;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * h);
            let tol = 1e-3 * fd.abs().max(1.0);
            assert!(
                (dx[i] as f64 - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={}", dx[i], fd
            );
        }
    }

    #[test]
    fn rejects_k_greater_than_cols() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let x = alloc_f32(&device, 4, vec![1, 4]);
        let i = alloc_u32(&device, 5, vec![1, 5]);
        let y = alloc_f32(&device, 5, vec![1, 5]);
        let p = make_params(&device, 1, 4, 5);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_take_along_axis_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x, &i, &y, &p, 1, 4, 5,
        );
        assert!(res.is_err());
    }
}
