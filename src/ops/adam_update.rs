//! Adam optimizer step kernel + Rust dispatch.
//!
//! Used by hf2q's ADR-020 Track 2 DWQ-proper training loop (iter 13).
//! Per-element in-place update of `param`, `m`, `v` per the standard
//! Adam algorithm.  See `shaders/adam_update.metal` for the math.
//!
//! Caller pre-computes the bias-correction denominators
//! `1 − β1^t` and `1 − β2^t` for the current step `t` and passes
//! them via the params buffer — keeps the kernel pure-element.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static ADAM_UPDATE_SHADER_SOURCE: &str =
    include_str!("../shaders/adam_update.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("adam_update_f32", ADAM_UPDATE_SHADER_SOURCE);
}

/// Encode one Adam optimizer step.
///
/// `params_buf` must contain `[lr, beta1, beta2, eps,
/// (1 − β1^t), (1 − β2^t)]` as f32 (24 bytes).
/// `meta_buf` must contain `[n_elements]` as u32 (4 bytes).
///
/// All four data buffers (`param`, `grad`, `m`, `v`) must have the
/// same f32 element count.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adam_update_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    param: &MlxBuffer,
    grad: &MlxBuffer,
    m: &MlxBuffer,
    v: &MlxBuffer,
    params_buf: &MlxBuffer,
    meta_buf: &MlxBuffer,
) -> Result<()> {
    let n = param.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "adam_update_f32: param must have at least one element".into(),
        ));
    }
    for (label, buf) in [("grad", grad), ("m", m), ("v", v)] {
        if buf.element_count() != n {
            return Err(MlxError::InvalidArgument(format!(
                "adam_update_f32: {label} element count {} != param element count {n}",
                buf.element_count(),
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "adam_update_f32: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if param.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "adam_update_f32: param dtype {} not f32",
            param.dtype()
        )));
    }
    if params_buf.byte_len() < 24 {
        return Err(MlxError::InvalidArgument(format!(
            "adam_update_f32: params_buf too small (need 24 bytes for 6×f32, got {})",
            params_buf.byte_len()
        )));
    }
    if meta_buf.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "adam_update_f32: meta_buf too small (need 4 bytes for u32, got {})",
            meta_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("adam_update_f32", device)?;
    let thread_count = n as u64;
    let tg_size = std::cmp::min(256, thread_count);
    encoder.encode(
        pipeline,
        &[
            (0, param),
            (1, grad),
            (2, m),
            (3, v),
            (4, params_buf),
            (5, meta_buf),
        ],
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    /// CPU oracle — pure-Rust Adam step.
    fn adam_cpu(
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        omb1_t: f32,
        omb2_t: f32,
    ) {
        for i in 0..param.len() {
            let g = grad[i];
            let m_new = beta1 * m[i] + (1.0 - beta1) * g;
            let v_new = beta2 * v[i] + (1.0 - beta2) * g * g;
            m[i] = m_new;
            v[i] = v_new;
            let m_hat = m_new / omb1_t;
            let v_hat = v_new / omb2_t;
            param[i] = param[i] - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    fn run_adam_step(
        param: &[f32],
        grad: &[f32],
        m: &[f32],
        v: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let device = MlxDevice::new().expect("device");
        let n = param.len();
        let mut p_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc param");
        p_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(param);
        let mut g_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc grad");
        g_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(grad);
        let mut m_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc m");
        m_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(m);
        let mut v_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![n])
            .expect("alloc v");
        v_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(v);
        let omb1_t = 1.0 - beta1.powi(t as i32);
        let omb2_t = 1.0 - beta2.powi(t as i32);
        let mut params_buf = device
            .alloc_buffer(24, DType::F32, vec![6])
            .expect("alloc params");
        params_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&[lr, beta1, beta2, eps, omb1_t, omb2_t]);
        let mut meta_buf = device
            .alloc_buffer(4, DType::F32, vec![1])
            .expect("alloc meta");
        meta_buf.as_mut_slice::<u32>().unwrap()[0] = n as u32;

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_adam_update_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &p_buf,
            &g_buf,
            &m_buf,
            &v_buf,
            &params_buf,
            &meta_buf,
        )
        .expect("dispatch adam");
        encoder.commit_and_wait().expect("commit");
        (
            p_buf.as_slice::<f32>().unwrap().to_vec(),
            m_buf.as_slice::<f32>().unwrap().to_vec(),
            v_buf.as_slice::<f32>().unwrap().to_vec(),
        )
    }

    fn assert_close_vec(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) {
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
    fn adam_step_t1_byte_close_to_cpu() {
        let n = 64;
        let param: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let m = vec![0f32; n];
        let v = vec![0f32; n];
        let lr = 1e-3_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let (p_gpu, m_gpu, v_gpu) =
            run_adam_step(&param, &grad, &m, &v, lr, beta1, beta2, eps, 1);
        let mut p_cpu = param.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        adam_cpu(
            &mut p_cpu,
            &grad,
            &mut m_cpu,
            &mut v_cpu,
            lr,
            beta1,
            beta2,
            eps,
            1.0 - beta1.powi(1),
            1.0 - beta2.powi(1),
        );
        assert_close_vec("adam param t=1", &p_gpu, &p_cpu, 1e-5, 1e-7);
        assert_close_vec("adam m t=1", &m_gpu, &m_cpu, 1e-5, 1e-7);
        assert_close_vec("adam v t=1", &v_gpu, &v_cpu, 1e-5, 1e-7);
    }

    #[test]
    fn adam_step_t10_with_nontrivial_state() {
        // Simulate the state we'd have after 10 prior steps by
        // initializing m, v to non-zero values.
        let n = 32;
        let param: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let m: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001 + 0.001).collect();
        let lr = 5e-4_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let (p_gpu, m_gpu, v_gpu) =
            run_adam_step(&param, &grad, &m, &v, lr, beta1, beta2, eps, 10);
        let mut p_cpu = param.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        adam_cpu(
            &mut p_cpu,
            &grad,
            &mut m_cpu,
            &mut v_cpu,
            lr,
            beta1,
            beta2,
            eps,
            1.0 - beta1.powi(10),
            1.0 - beta2.powi(10),
        );
        assert_close_vec("adam param t=10", &p_gpu, &p_cpu, 1e-5, 1e-7);
        assert_close_vec("adam m t=10", &m_gpu, &m_cpu, 1e-5, 1e-7);
        assert_close_vec("adam v t=10", &v_gpu, &v_cpu, 1e-5, 1e-7);
    }

    #[test]
    fn adam_zero_grad_leaves_param_unchanged() {
        // With grad = 0 and m = v = 0, the update is 0/eps = 0
        // (within fp32) → param unchanged.  Confirms the zero-grad
        // optimization-fixed-point.
        let n = 16;
        let param: Vec<f32> = (0..n).map(|i| (i as f32) - 8.0).collect();
        let grad = vec![0f32; n];
        let m = vec![0f32; n];
        let v = vec![0f32; n];
        let (p_gpu, m_gpu, v_gpu) =
            run_adam_step(&param, &grad, &m, &v, 1e-3, 0.9, 0.999, 1e-8, 1);
        // m, v stay 0 after one step with grad=0; param: 0/(0+eps) = 0
        // → no change.  Allow tiny eps-noise but bit-exact in practice.
        for (i, (p_in, p_out)) in param.iter().zip(p_gpu.iter()).enumerate() {
            assert!(
                (p_in - p_out).abs() < 1e-9,
                "i={i}: param changed from {p_in} to {p_out}"
            );
        }
        assert!(m_gpu.iter().all(|&x| x == 0.0));
        assert!(v_gpu.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn adam_simple_optimization_converges() {
        // Optimize a single-element f(x) = (x - 5)² by running Adam
        // for many steps.  Gradient = 2·(x − 5).  After enough steps
        // x should converge to 5.
        let device = MlxDevice::new().expect("device");
        let mut p_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("p");
        p_buf.as_mut_slice::<f32>().unwrap()[0] = 0.0; // start at 0
        let mut g_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("g");
        let m_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("m");
        let v_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("v");
        // alloc_buffer is zero-fill.
        let mut params_buf = device
            .alloc_buffer(24, DType::F32, vec![6])
            .expect("params");
        let mut meta_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("meta");
        meta_buf.as_mut_slice::<u32>().unwrap()[0] = 1u32;

        let lr = 0.1_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;

        let mut registry = KernelRegistry::new();
        register(&mut registry);

        for step in 1..=200u32 {
            let x = p_buf.as_slice::<f32>().unwrap()[0];
            let g = 2.0 * (x - 5.0);
            g_buf.as_mut_slice::<f32>().unwrap()[0] = g;
            params_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&[
                lr,
                beta1,
                beta2,
                eps,
                1.0 - beta1.powi(step as i32),
                1.0 - beta2.powi(step as i32),
            ]);
            let mut encoder = device.command_encoder().expect("encoder");
            dispatch_adam_update_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &p_buf,
                &g_buf,
                &m_buf,
                &v_buf,
                &params_buf,
                &meta_buf,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }

        let final_x = p_buf.as_slice::<f32>().unwrap()[0];
        // After 200 steps with lr=0.1 on f(x)=(x−5)², x should be near 5.
        assert!(
            (final_x - 5.0).abs() < 0.05,
            "expected x ≈ 5 after 200 Adam steps; got {final_x}"
        );
    }

    #[test]
    fn adam_rejects_mismatched_sizes() {
        let device = MlxDevice::new().expect("device");
        let p = device.alloc_buffer(16, DType::F32, vec![4]).expect("p");
        let g = device.alloc_buffer(32, DType::F32, vec![8]).expect("g"); // wrong size
        let m = device.alloc_buffer(16, DType::F32, vec![4]).expect("m");
        let v = device.alloc_buffer(16, DType::F32, vec![4]).expect("v");
        let params = device.alloc_buffer(24, DType::F32, vec![6]).expect("params");
        let meta = device.alloc_buffer(4, DType::F32, vec![1]).expect("meta");
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        let err = dispatch_adam_update_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &p,
            &g,
            &m,
            &v,
            &params,
            &meta,
        )
        .expect_err("must reject mismatched sizes");
        assert!(format!("{err}").contains("grad element count"));
    }
}
