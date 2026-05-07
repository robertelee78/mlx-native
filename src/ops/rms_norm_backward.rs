//! Backward pass for RMS Normalization (`rms_norm_f32` forward).
//!
//! Forward:
//!   `y[b, i] = x[b, i] · rsqrt(mean(x[b, :]²) + eps) · w[i]`
//!
//! Backward — given `dy`, `x`, `w` → produce `dx` and `dw` via the
//! analytical identities (see `shaders/rms_norm_backward.metal` for
//! the derivation):
//!
//!   `dw[i]    = Σ_b dy[b, i] · x[b, i] · r[b]`
//!   `dx[b, k] = r[b] · (dy[b, k] · w[k] - x[b, k] · s[b] · r[b]² / D)`
//!     where `s[b] = Σ_i dy[b, i] · x[b, i] · w[i]`,
//!           `r[b] = rsqrt(mean(x[b, :]²) + eps)`.
//!
//! Three kernels:
//!   1. `rms_norm_compute_rms_inv_f32` — produces `r[rows]`
//!   2. `rms_norm_backward_dx_f32`     — produces `dx[rows, dim]`
//!   3. `rms_norm_backward_dw_f32`     — produces `dw[dim]`
//!
//! `r[rows]` is computed once and reused across both `dx` and `dw`.
//! Used by hf2q ADR-020 Track 1 (4-layer Qwen35 attention block on
//! GpuTape).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static RMS_NORM_BACKWARD_SHADER_SOURCE: &str =
    include_str!("../shaders/rms_norm_backward.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "rms_norm_compute_rms_inv_f32",
        RMS_NORM_BACKWARD_SHADER_SOURCE,
    );
    registry.register_source("rms_norm_backward_dx_f32", RMS_NORM_BACKWARD_SHADER_SOURCE);
    registry.register_source("rms_norm_backward_dw_f32", RMS_NORM_BACKWARD_SHADER_SOURCE);
}

/// Compute per-row `r[b] = rsqrt(mean(x[b, :]²) + eps)` and write to
/// `r_out[rows]`.  Helper for the two backward kernels.
///
/// `params_buf` must contain `[eps, dim_f]` as f32.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rms_norm_compute_rms_inv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    r_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "rms_norm_compute_rms_inv: rows and dim must be > 0".into(),
        ));
    }
    if x.element_count() != (rows as usize) * (dim as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_compute_rms_inv: x element count {} != rows({}) * dim({})",
            x.element_count(),
            rows,
            dim
        )));
    }
    if r_out.element_count() != rows as usize {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_compute_rms_inv: r_out element count {} != rows ({})",
            r_out.element_count(),
            rows
        )));
    }
    for (label, buf) in [("x", x), ("r_out", r_out)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_compute_rms_inv: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_compute_rms_inv: params_buf too small (need 8 bytes for float2, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_compute_rms_inv_f32", device)?;
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, x), (1, r_out), (2, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Compute `dx[rows, dim]` from `(x, w, dy, r)` using the analytical
/// RMSNorm backward formula.  `r` must have been precomputed via
/// [`dispatch_rms_norm_compute_rms_inv`] (`r[b] = rsqrt(mean(x²) + eps)`).
///
/// `params_buf` must contain `[dim_f, _]` as f32 (only `params[0]`
/// is read; the second slot is reserved padding for future variants).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rms_norm_backward_dx(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    w: &MlxBuffer,
    dy: &MlxBuffer,
    r: &MlxBuffer,
    dx: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "rms_norm_backward_dx: rows and dim must be > 0".into(),
        ));
    }
    let expected_2d = (rows as usize) * (dim as usize);
    for (label, buf) in [("x", x), ("dy", dy), ("dx", dx)] {
        if buf.element_count() != expected_2d {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dx: {label} element count {} != rows({}) * dim({})",
                buf.element_count(),
                rows,
                dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dx: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if w.element_count() != dim as usize {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dx: w element count {} != dim ({})",
            w.element_count(),
            dim
        )));
    }
    if r.element_count() != rows as usize {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dx: r element count {} != rows ({})",
            r.element_count(),
            rows
        )));
    }
    for (label, buf) in [("w", w), ("r", r)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dx: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dx: params_buf too small (need 8 bytes for float2, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_backward_dx_f32", device)?;
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, x), (1, w), (2, dy), (3, r), (4, dx), (5, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Compute `dw[dim]` from `(x, dy, r)` using the analytical RMSNorm
/// backward formula `dw[i] = Σ_b dy[b, i] · x[b, i] · r[b]`.
///
/// `params_buf` must contain `[dim_f, rows_f]` as f32.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rms_norm_backward_dw(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    dy: &MlxBuffer,
    r: &MlxBuffer,
    dw: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "rms_norm_backward_dw: rows and dim must be > 0".into(),
        ));
    }
    let expected_2d = (rows as usize) * (dim as usize);
    for (label, buf) in [("x", x), ("dy", dy)] {
        if buf.element_count() != expected_2d {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dw: {label} element count {} != rows({}) * dim({})",
                buf.element_count(),
                rows,
                dim
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dw: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if r.element_count() != rows as usize {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dw: r element count {} != rows ({})",
            r.element_count(),
            rows
        )));
    }
    if dw.element_count() != dim as usize {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dw: dw element count {} != dim ({})",
            dw.element_count(),
            dim
        )));
    }
    for (label, buf) in [("r", r), ("dw", dw)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "rms_norm_backward_dw: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "rms_norm_backward_dw: params_buf too small (need 8 bytes for float2, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_backward_dw_f32", device)?;
    // One threadgroup per FEATURE; threads in TG stride over rows.
    let tg_size = std::cmp::min(256, rows.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, x), (1, dy), (2, r), (3, dw), (4, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(dim as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    /// Pure-Rust reference implementation of the full RMSNorm forward
    /// + backward used as oracle in parity tests.
    fn rms_norm_forward_backward_cpu(
        x: &[f32],
        w: &[f32],
        dy: &[f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = vec![0f32; rows];
        for b in 0..rows {
            let row = &x[b * dim..(b + 1) * dim];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            r[b] = (ms + eps).sqrt().recip();
        }

        let mut dx = vec![0f32; rows * dim];
        for b in 0..rows {
            let r_b = r[b];
            // s[b] = Σ_i dy * x * w
            let s_b: f32 = (0..dim)
                .map(|i| dy[b * dim + i] * x[b * dim + i] * w[i])
                .sum();
            let coeff = s_b * r_b * r_b / dim as f32;
            for k in 0..dim {
                let idx = b * dim + k;
                dx[idx] = r_b * (dy[idx] * w[k] - x[idx] * coeff);
            }
        }

        let mut dw = vec![0f32; dim];
        for i in 0..dim {
            let mut acc = 0.0f32;
            for b in 0..rows {
                acc += dy[b * dim + i] * x[b * dim + i] * r[b];
            }
            dw[i] = acc;
        }

        (r, dx, dw)
    }

    fn run_backward(
        x: &[f32],
        w: &[f32],
        dy: &[f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let n = rows * dim;

        let mut x_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![rows, dim])
            .expect("alloc x");
        x_buf
            .as_mut_slice::<f32>()
            .expect("x as_mut_slice")
            .copy_from_slice(x);
        let mut w_buf = device
            .alloc_buffer(dim * 4, DType::F32, vec![dim])
            .expect("alloc w");
        w_buf
            .as_mut_slice::<f32>()
            .expect("w as_mut_slice")
            .copy_from_slice(w);
        let mut dy_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![rows, dim])
            .expect("alloc dy");
        dy_buf
            .as_mut_slice::<f32>()
            .expect("dy as_mut_slice")
            .copy_from_slice(dy);

        let r_buf = device
            .alloc_buffer(rows * 4, DType::F32, vec![rows])
            .expect("alloc r");
        let dx_buf = device
            .alloc_buffer(n * 4, DType::F32, vec![rows, dim])
            .expect("alloc dx");
        let dw_buf = device
            .alloc_buffer(dim * 4, DType::F32, vec![dim])
            .expect("alloc dw");

        // params for rms_inv: [eps, dim_f]
        let mut params_inv = device
            .alloc_buffer(8, DType::F32, vec![2])
            .expect("alloc params_inv");
        {
            let s = params_inv.as_mut_slice::<f32>().expect("params_inv as_mut");
            s[0] = eps;
            s[1] = dim as f32;
        }
        // params for dx: [dim_f, _]
        let mut params_dx = device
            .alloc_buffer(8, DType::F32, vec![2])
            .expect("alloc params_dx");
        {
            let s = params_dx.as_mut_slice::<f32>().expect("params_dx as_mut");
            s[0] = dim as f32;
            s[1] = 0.0;
        }
        // params for dw: [dim_f, rows_f]
        let mut params_dw = device
            .alloc_buffer(8, DType::F32, vec![2])
            .expect("alloc params_dw");
        {
            let s = params_dw.as_mut_slice::<f32>().expect("params_dw as_mut");
            s[0] = dim as f32;
            s[1] = rows as f32;
        }

        let mut registry = KernelRegistry::new();
        register(&mut registry);

        let mut encoder = device.command_encoder().expect("command_encoder");
        dispatch_rms_norm_compute_rms_inv(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &r_buf,
            &params_inv,
            rows as u32,
            dim as u32,
        )
        .expect("dispatch rms_inv");
        encoder.memory_barrier();
        dispatch_rms_norm_backward_dx(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &w_buf,
            &dy_buf,
            &r_buf,
            &dx_buf,
            &params_dx,
            rows as u32,
            dim as u32,
        )
        .expect("dispatch dx");
        dispatch_rms_norm_backward_dw(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &dy_buf,
            &r_buf,
            &dw_buf,
            &params_dw,
            rows as u32,
            dim as u32,
        )
        .expect("dispatch dw");
        encoder.commit_and_wait().expect("commit_and_wait");

        let r_out = r_buf.as_slice::<f32>().expect("r as_slice").to_vec();
        let dx_out = dx_buf.as_slice::<f32>().expect("dx as_slice").to_vec();
        let dw_out = dw_buf.as_slice::<f32>().expect("dw as_slice").to_vec();
        (r_out, dx_out, dw_out)
    }

    fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let diff = (g - c).abs();
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!(
                diff <= abs_tol || diff / scale <= rel_tol,
                "{label}: index {i}: gpu={g} cpu={c} diff={diff} (rel_tol={rel_tol}, abs_tol={abs_tol})"
            );
        }
    }

    #[test]
    fn rms_norm_backward_parity_small() {
        // 2 rows, 32 dim — the smallest non-trivial fixture.
        let rows = 2usize;
        let dim = 32usize;
        let eps = 1e-6;
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + 0.1 * (i as f32 - 16.0)).collect();
        let dy: Vec<f32> = (0..rows * dim)
            .map(|i| ((i as f32 * 0.0271).cos()) * 0.3)
            .collect();
        let (r_gpu, dx_gpu, dw_gpu) = run_backward(&x, &w, &dy, rows, dim, eps);
        let (r_cpu, dx_cpu, dw_cpu) =
            rms_norm_forward_backward_cpu(&x, &w, &dy, rows, dim, eps);
        assert_close("r small", &r_gpu, &r_cpu, 1e-5, 1e-6);
        assert_close("dx small", &dx_gpu, &dx_cpu, 1e-4, 1e-6);
        assert_close("dw small", &dw_gpu, &dw_cpu, 1e-4, 1e-6);
    }

    #[test]
    fn rms_norm_backward_parity_realistic_qwen35_shape() {
        // Qwen 3.5 hidden_size = 5120 — but we use 1024 dim x 8 rows
        // for the unit test (representative of attention-block shapes
        // without bloating test wall).  Larger shapes are exercised
        // in hf2q's GpuTape integration tests.
        let rows = 8usize;
        let dim = 1024usize;
        let eps = 1e-6;
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.0073).sin() * 0.42)
            .collect();
        let w: Vec<f32> = (0..dim)
            .map(|i| 1.0 + ((i as f32 * 0.011).cos()) * 0.05)
            .collect();
        let dy: Vec<f32> = (0..rows * dim)
            .map(|i| ((i as f32 * 0.013).cos()) * 0.28)
            .collect();
        let (r_gpu, dx_gpu, dw_gpu) = run_backward(&x, &w, &dy, rows, dim, eps);
        let (r_cpu, dx_cpu, dw_cpu) =
            rms_norm_forward_backward_cpu(&x, &w, &dy, rows, dim, eps);
        assert_close("r realistic", &r_gpu, &r_cpu, 1e-5, 1e-6);
        // Looser tolerance for dx/dw at dim=1024 because TG-tree
        // reduction sums in a different float order than the CPU
        // sequential sum.
        assert_close("dx realistic", &dx_gpu, &dx_cpu, 1e-3, 1e-5);
        assert_close("dw realistic", &dw_gpu, &dw_cpu, 1e-3, 1e-5);
    }

    #[test]
    fn rms_norm_backward_unit_weights_pinning() {
        // With w == 1 everywhere, dw = Σ_b dy * x * r, dx = r·(dy - x·s/D).
        // Pin the simpler form against the general formula.
        let rows = 4usize;
        let dim = 64usize;
        let eps = 1e-6;
        let x: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.001 - 0.1).collect();
        let w: Vec<f32> = vec![1.0; dim];
        let dy: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.0007 - 0.05).collect();
        let (r_gpu, dx_gpu, dw_gpu) = run_backward(&x, &w, &dy, rows, dim, eps);
        let (r_cpu, dx_cpu, dw_cpu) =
            rms_norm_forward_backward_cpu(&x, &w, &dy, rows, dim, eps);
        assert_close("r unit-w", &r_gpu, &r_cpu, 1e-5, 1e-6);
        assert_close("dx unit-w", &dx_gpu, &dx_cpu, 1e-4, 1e-6);
        assert_close("dw unit-w", &dw_gpu, &dw_cpu, 1e-4, 1e-6);
    }

    #[test]
    fn rms_norm_backward_finite_diff_falsifier() {
        // The acid test: backward must match a finite-difference
        // gradient of a scalar loss `L = Σ_{b, i} dy[b, i] * y[b, i]`
        // (whose gradient w.r.t. y is `dy`, w.r.t. x is `dx`, and
        // w.r.t. w is `dw` per chain rule).
        let rows = 2usize;
        let dim = 32usize;
        let eps = 1e-6;
        let det = |s: u64| {
            // Deterministic pseudo-random in [-1, 1].
            let mut state = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            state ^= state >> 33;
            state = state.wrapping_mul(0xff51_afd7_ed55_8ccdu64);
            state ^= state >> 33;
            ((state as i64) as f32) / (i64::MAX as f32)
        };
        let x: Vec<f32> = (0..rows * dim).map(|i| det(i as u64) * 0.5).collect();
        let w: Vec<f32> = (0..dim)
            .map(|i| 1.0 + det((i as u64) + 7919) * 0.1)
            .collect();
        let dy: Vec<f32> = (0..rows * dim)
            .map(|i| det((i as u64) + 12347) * 0.3)
            .collect();
        let (_r_gpu, dx_gpu, dw_gpu) = run_backward(&x, &w, &dy, rows, dim, eps);

        // Finite-diff dx[0] and a few others.
        let h = 1e-3_f32;
        for &probe_idx in &[0usize, 7, 31, 32, 47, 63] {
            if probe_idx >= rows * dim {
                continue;
            }
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[probe_idx] += h;
            x_minus[probe_idx] -= h;
            // Forward + dot-with-dy (the loss is L = Σ dy * y).
            let l_plus = forward_dot_loss(&x_plus, &w, &dy, rows, dim, eps);
            let l_minus = forward_dot_loss(&x_minus, &w, &dy, rows, dim, eps);
            let fd = (l_plus - l_minus) / (2.0 * h);
            let analytical = dx_gpu[probe_idx];
            let diff = (fd - analytical).abs();
            let scale = fd.abs().max(analytical.abs()).max(1.0);
            assert!(
                diff <= 1e-3 || diff / scale <= 5e-3,
                "dx finite-diff falsifier failed at {probe_idx}: fd={fd} analytical={analytical} diff={diff}"
            );
        }
        // Finite-diff dw at a few features.
        for &probe_feat in &[0usize, 5, 15, 31] {
            if probe_feat >= dim {
                continue;
            }
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[probe_feat] += h;
            w_minus[probe_feat] -= h;
            let l_plus = forward_dot_loss(&x, &w_plus, &dy, rows, dim, eps);
            let l_minus = forward_dot_loss(&x, &w_minus, &dy, rows, dim, eps);
            let fd = (l_plus - l_minus) / (2.0 * h);
            let analytical = dw_gpu[probe_feat];
            let diff = (fd - analytical).abs();
            let scale = fd.abs().max(analytical.abs()).max(1.0);
            assert!(
                diff <= 1e-3 || diff / scale <= 5e-3,
                "dw finite-diff falsifier failed at feat {probe_feat}: fd={fd} analytical={analytical} diff={diff}"
            );
        }
    }

    /// Helper for finite-diff falsifier: the scalar loss
    /// `L = Σ dy[b, i] * y[b, i]` whose gradients are exactly
    /// (dx, dw) under the chain rule.
    fn forward_dot_loss(
        x: &[f32],
        w: &[f32],
        dy: &[f32],
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> f32 {
        let mut acc = 0.0f64; // wider accumulator to keep the FD precise
        for b in 0..rows {
            let row = &x[b * dim..(b + 1) * dim];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let r_b = (ms + eps).sqrt().recip();
            for i in 0..dim {
                let y = x[b * dim + i] * r_b * w[i];
                acc += (dy[b * dim + i] as f64) * (y as f64);
            }
        }
        acc as f32
    }
}
