//! ADR-020 iter-11h-b — depthwise causal 1D convolution forward +
//! backward kernels for the GpuTape autograd pipeline.
//!
//! Distinct from `ssm_conv` (which fuses SiLU + handles autoregressive
//! decode state).  This module is for TRAINING-MODE backward pass:
//!
//! Forward shape contract:
//!   x : `[n_tokens, channels]` row-major (f32)
//!   kernel_w : `[channels, K]` row-major (f32)
//!   y : `[n_tokens, channels]` row-major (f32)
//!
//! Math (per output element `(t, c)`):
//!   y[t, c] = Σ_{k=0..K-1, t+k-(K-1)>=0} kernel_w[c, k] · x[t+k-(K-1), c]
//!
//! Zero-pad on the past: outputs at `t < K-1` see fewer than K input
//! taps (the missing taps default to 0 — equivalent to "no prior
//! decode state", which is the training-time invariant).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static CONV1D_DEPTHWISE_CAUSAL_SHADER_SOURCE: &str =
    include_str!("../shaders/conv1d_depthwise_causal.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "conv1d_depthwise_causal_forward_f32",
        CONV1D_DEPTHWISE_CAUSAL_SHADER_SOURCE,
    );
    registry.register_source(
        "conv1d_depthwise_causal_backward_dx_f32",
        CONV1D_DEPTHWISE_CAUSAL_SHADER_SOURCE,
    );
    registry.register_source(
        "conv1d_depthwise_causal_backward_dw_f32",
        CONV1D_DEPTHWISE_CAUSAL_SHADER_SOURCE,
    );
}

fn validate_shapes(
    op: &str,
    n_tokens: u32,
    channels: u32,
    k: u32,
    x_or_dx: &MlxBuffer,
    w_or_dy_or_dw: &MlxBuffer,
    out: &MlxBuffer,
    params: &MlxBuffer,
    expected_first_count: usize,
    expected_second_count: usize,
    expected_out_count: usize,
) -> Result<()> {
    if n_tokens == 0 || channels == 0 || k == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: n_tokens, channels, K must all be > 0 (got {n_tokens}, {channels}, {k})"
        )));
    }
    if x_or_dx.dtype() != DType::F32
        || w_or_dy_or_dw.dtype() != DType::F32
        || out.dtype() != DType::F32
    {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: all I/O buffers must be f32"
        )));
    }
    if x_or_dx.element_count() != expected_first_count {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: first-buffer element_count {} != expected {expected_first_count}",
            x_or_dx.element_count()
        )));
    }
    if w_or_dy_or_dw.element_count() != expected_second_count {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: second-buffer element_count {} != expected {expected_second_count}",
            w_or_dy_or_dw.element_count()
        )));
    }
    if out.element_count() != expected_out_count {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: out element_count {} != expected {expected_out_count}",
            out.element_count()
        )));
    }
    if params.byte_len() < 12 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: params < 12 bytes (need 3 × u32 = [n_tokens, channels, K])"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_conv1d_depthwise_causal_forward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    kernel_w: &MlxBuffer,
    y: &MlxBuffer,
    params: &MlxBuffer,
    n_tokens: u32,
    channels: u32,
    k: u32,
) -> Result<()> {
    const OP: &str = "conv1d_depthwise_causal_forward_f32";
    let n = n_tokens as usize;
    let c = channels as usize;
    let k_us = k as usize;
    validate_shapes(
        OP, n_tokens, channels, k, x, kernel_w, y, params,
        n * c, c * k_us, n * c,
    )?;

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, x), (1, kernel_w), (2, y), (3, params)],
        MTLSize::new(n_tokens as u64, channels as u64, 1),
        MTLSize::new(
            std::cmp::min(32, n_tokens as u64),
            std::cmp::min(8, channels as u64),
            1,
        ),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_conv1d_depthwise_causal_backward_dx_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    kernel_w: &MlxBuffer,
    dx: &MlxBuffer,
    params: &MlxBuffer,
    n_tokens: u32,
    channels: u32,
    k: u32,
) -> Result<()> {
    const OP: &str = "conv1d_depthwise_causal_backward_dx_f32";
    let n = n_tokens as usize;
    let c = channels as usize;
    let k_us = k as usize;
    validate_shapes(
        OP, n_tokens, channels, k, dy, kernel_w, dx, params,
        n * c, c * k_us, n * c,
    )?;

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, dy), (1, kernel_w), (2, dx), (3, params)],
        MTLSize::new(n_tokens as u64, channels as u64, 1),
        MTLSize::new(
            std::cmp::min(32, n_tokens as u64),
            std::cmp::min(8, channels as u64),
            1,
        ),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_conv1d_depthwise_causal_backward_dw_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    dy: &MlxBuffer,
    dw: &MlxBuffer,
    params: &MlxBuffer,
    n_tokens: u32,
    channels: u32,
    k: u32,
) -> Result<()> {
    const OP: &str = "conv1d_depthwise_causal_backward_dw_f32";
    let n = n_tokens as usize;
    let c = channels as usize;
    let k_us = k as usize;
    validate_shapes(
        OP, n_tokens, channels, k, x, dy, dw, params,
        n * c, n * c, c * k_us,
    )?;

    let pipeline = registry.get_pipeline(OP, device)?;
    encoder.encode(
        pipeline,
        &[(0, x), (1, dy), (2, dw), (3, params)],
        MTLSize::new(channels as u64, k as u64, 1),
        MTLSize::new(
            std::cmp::min(32, channels as u64),
            std::cmp::min(8, k as u64),
            1,
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn alloc_f32(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        let mut b = device
            .alloc_buffer(n * 4, DType::F32, shape)
            .expect("alloc f32");
        b.as_mut_slice::<f32>().unwrap().fill(0.0);
        b
    }

    fn make_params(device: &MlxDevice, n_tokens: u32, channels: u32, k: u32) -> MlxBuffer {
        let mut p = device
            .alloc_buffer(12, DType::U32, vec![3])
            .expect("alloc params");
        p.as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[n_tokens, channels, k]);
        p
    }

    /// CPU oracle: forward causal depthwise conv with zero-pad.
    fn forward_cpu(
        x: &[f32], kernel_w: &[f32], n: usize, c: usize, k: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; n * c];
        for t in 0..n {
            for ch in 0..c {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    let i_signed = (t as isize) + (kk as isize) - (k as isize - 1);
                    if i_signed < 0 {
                        continue;
                    }
                    let i = i_signed as usize;
                    sum += kernel_w[ch * k + kk] as f64 * x[i * c + ch] as f64;
                }
                y[t * c + ch] = sum as f32;
            }
        }
        y
    }

    /// CPU oracle: backward dx.
    fn backward_dx_cpu(
        dy: &[f32], kernel_w: &[f32], n: usize, c: usize, k: usize,
    ) -> Vec<f32> {
        let mut dx = vec![0.0f32; n * c];
        for i in 0..n {
            for ch in 0..c {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    let t_signed = (i as isize) + (k as isize - 1) - (kk as isize);
                    if t_signed < 0 || t_signed >= n as isize {
                        continue;
                    }
                    let t = t_signed as usize;
                    sum += kernel_w[ch * k + kk] as f64 * dy[t * c + ch] as f64;
                }
                dx[i * c + ch] = sum as f32;
            }
        }
        dx
    }

    /// CPU oracle: backward dw.
    fn backward_dw_cpu(
        x: &[f32], dy: &[f32], n: usize, c: usize, k: usize,
    ) -> Vec<f32> {
        let mut dw = vec![0.0f32; c * k];
        for ch in 0..c {
            for kk in 0..k {
                let mut sum = 0.0f64;
                for t in (k - 1 - kk)..n {
                    let i = t + kk - (k - 1);
                    sum += x[i * c + ch] as f64 * dy[t * c + ch] as f64;
                }
                dw[ch * k + kk] = sum as f32;
            }
        }
        dw
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n = 16usize;
        let c = 8usize;
        let k = 4usize;

        let x: Vec<f32> = (0..(n * c))
            .map(|i| ((i as f32) * 0.137 - 0.4).sin() * 0.7)
            .collect();
        let w: Vec<f32> = (0..(c * k))
            .map(|i| ((i as f32) * 0.231 + 0.1).cos() * 0.5)
            .collect();

        let mut x_buf = alloc_f32(&device, n * c, vec![n, c]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut w_buf = alloc_f32(&device, c * k, vec![c, k]);
        w_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&w);
        let y_buf = alloc_f32(&device, n * c, vec![n, c]);
        let params = make_params(&device, n as u32, c as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_conv1d_depthwise_causal_forward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &w_buf, &y_buf, &params,
            n as u32, c as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        let cpu = forward_cpu(&x, &w, n, c, k);
        for i in 0..(n * c) {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-5 * cpu[i].abs().max(1.0),
                "forward y[{i}]: gpu={} cpu={}",
                gpu[i], cpu[i]
            );
        }
    }

    #[test]
    fn backward_dx_matches_cpu_oracle() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n = 16usize;
        let c = 8usize;
        let k = 4usize;

        let dy: Vec<f32> = (0..(n * c)).map(|i| ((i as f32) * 0.073 - 0.3).sin() * 0.6).collect();
        let w: Vec<f32> = (0..(c * k)).map(|i| 0.1 + (i as f32) * 0.013).collect();

        let mut dy_buf = alloc_f32(&device, n * c, vec![n, c]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let mut w_buf = alloc_f32(&device, c * k, vec![c, k]);
        w_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&w);
        let dx_buf = alloc_f32(&device, n * c, vec![n, c]);
        let params = make_params(&device, n as u32, c as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_conv1d_depthwise_causal_backward_dx_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &w_buf, &dx_buf, &params,
            n as u32, c as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = dx_buf.as_slice::<f32>().unwrap();
        let cpu = backward_dx_cpu(&dy, &w, n, c, k);
        for i in 0..(n * c) {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-5 * cpu[i].abs().max(1.0),
                "dx[{i}]: gpu={} cpu={}",
                gpu[i], cpu[i]
            );
        }
    }

    #[test]
    fn backward_dw_matches_cpu_oracle() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n = 32usize;
        let c = 8usize;
        let k = 4usize;

        let x: Vec<f32> = (0..(n * c)).map(|i| ((i as f32) * 0.041 - 0.5).cos() * 0.7).collect();
        let dy: Vec<f32> = (0..(n * c)).map(|i| ((i as f32) * 0.073 - 0.3).sin() * 0.6).collect();

        let mut x_buf = alloc_f32(&device, n * c, vec![n, c]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut dy_buf = alloc_f32(&device, n * c, vec![n, c]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy);
        let dw_buf = alloc_f32(&device, c * k, vec![c, k]);
        let params = make_params(&device, n as u32, c as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_conv1d_depthwise_causal_backward_dw_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &dy_buf, &dw_buf, &params,
            n as u32, c as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = dw_buf.as_slice::<f32>().unwrap();
        let cpu = backward_dw_cpu(&x, &dy, n, c, k);
        for i in 0..(c * k) {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4 * cpu[i].abs().max(1.0),
                "dw[{i}]: gpu={} cpu={}",
                gpu[i], cpu[i]
            );
        }
    }

    /// Finite-difference falsifier: verify analytic dw and dx match
    /// numerical gradient of `loss = sum(forward(x, w))` to within 1%
    /// relative tolerance.  This is THE load-bearing correctness gate.
    #[test]
    fn backward_finite_difference_falsifier() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n = 8usize;
        let c = 4usize;
        let k = 3usize;

        let x: Vec<f32> = (0..(n * c)).map(|i| ((i as f32) * 0.137).sin() * 0.6).collect();
        let w: Vec<f32> = (0..(c * k)).map(|i| 0.2 + (i as f32) * 0.05).collect();

        let forward_loss = |x: &[f32], w: &[f32]| -> f64 {
            let y = forward_cpu(x, w, n, c, k);
            y.iter().map(|v| *v as f64).sum::<f64>()
        };

        // Analytic gradients via dy = ones.
        let dy_ones = vec![1.0f32; n * c];
        let mut dy_buf = alloc_f32(&device, n * c, vec![n, c]);
        dy_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&dy_ones);

        let mut x_buf = alloc_f32(&device, n * c, vec![n, c]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut w_buf = alloc_f32(&device, c * k, vec![c, k]);
        w_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&w);
        let dx_buf = alloc_f32(&device, n * c, vec![n, c]);
        let dw_buf = alloc_f32(&device, c * k, vec![c, k]);
        let params = make_params(&device, n as u32, c as u32, k as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_conv1d_depthwise_causal_backward_dx_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &dy_buf, &w_buf, &dx_buf, &params,
            n as u32, c as u32, k as u32,
        ).unwrap();
        dispatch_conv1d_depthwise_causal_backward_dw_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &dy_buf, &dw_buf, &params,
            n as u32, c as u32, k as u32,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let dx = dx_buf.as_slice::<f32>().unwrap().to_vec();
        let dw = dw_buf.as_slice::<f32>().unwrap().to_vec();

        // FD on x.
        let h = 1e-3f64;
        for i in 0..(n * c) {
            let mut xp = x.clone();
            xp[i] += h as f32;
            let mut xm = x.clone();
            xm[i] -= h as f32;
            let fd = (forward_loss(&xp, &w) - forward_loss(&xm, &w)) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (dx[i] as f64 - fd).abs() < tol,
                "FD x[{i}]: analytic={} fd={}", dx[i], fd
            );
        }
        // FD on w.
        for i in 0..(c * k) {
            let mut wp = w.clone();
            wp[i] += h as f32;
            let mut wm = w.clone();
            wm[i] -= h as f32;
            let fd = (forward_loss(&x, &wp) - forward_loss(&x, &wm)) / (2.0 * h);
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (dw[i] as f64 - fd).abs() < tol,
                "FD w[{i}]: analytic={} fd={}", dw[i], fd
            );
        }
    }

    #[test]
    fn rejects_zero_dimensions() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let x_buf = alloc_f32(&device, 1, vec![1, 1]);
        let w_buf = alloc_f32(&device, 1, vec![1, 1]);
        let y_buf = alloc_f32(&device, 1, vec![1, 1]);
        let params = make_params(&device, 0, 1, 1);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_conv1d_depthwise_causal_forward_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &w_buf, &y_buf, &params, 0, 1, 1,
        );
        assert!(res.is_err());
    }
}
