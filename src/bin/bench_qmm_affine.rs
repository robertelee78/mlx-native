//! Quick benchmark: iter-15 per-element vs iter-15b tiled qmm_affine.
//!
//! Measures average per-call wall time over N iterations on a fixed
//! shape, releasing GPU between calls.  Goal: validate the iter-15b
//! "2-5× speedup" claim on a representative shape.
//!
//! Usage: `cargo run --release --bin bench_qmm_affine -- [M] [N] [K]`
//! Default: 64x4096x4096 (close to a real attention out-proj shape).

use std::time::Instant;

use mlx_native::ops::qmm_affine::{
    dispatch_qmm_affine_t_f32, dispatch_qmm_affine_t_f32_simd,
    dispatch_qmm_affine_t_f32_tiled,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let m: u32 = args
        .next()
        .map(|s| s.parse().expect("M"))
        .unwrap_or(64);
    let n: u32 = args
        .next()
        .map(|s| s.parse().expect("N"))
        .unwrap_or(4096);
    let k: u32 = args
        .next()
        .map(|s| s.parse().expect("K"))
        .unwrap_or(4096);
    let group_size: u32 = 32;
    let n_iter: usize = args
        .next()
        .map(|s| s.parse().expect("n_iter"))
        .unwrap_or(20);

    println!("[bench] shape M={m} N={n} K={k} group_size={group_size} iter={n_iter}");

    let device = MlxDevice::new()?;
    let mut registry = KernelRegistry::new();

    let m_us = m as usize;
    let n_us = n as usize;
    let k_us = k as usize;
    let groups = k_us / group_size as usize;

    let x: Vec<f32> = (0..(m_us * k_us))
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let q_int: Vec<u8> = (0..(n_us * k_us)).map(|i| ((i * 7) % 16) as u8).collect();
    let scales: Vec<f32> = (0..(n_us * groups))
        .map(|i| 0.05 + (i as f32) * 1e-5)
        .collect();
    let biases: Vec<f32> = (0..(n_us * groups))
        .map(|i| -0.1 + (i as f32) * 1e-5)
        .collect();

    let mut x_buf = device.alloc_buffer(m_us * k_us * 4, DType::F32, vec![m_us, k_us])?;
    x_buf.as_mut_slice::<f32>()?.copy_from_slice(&x);
    let mut q_buf = device.alloc_buffer(n_us * k_us, DType::U8, vec![n_us, k_us])?;
    q_buf.as_mut_slice::<u8>()?.copy_from_slice(&q_int);
    let mut s_buf = device.alloc_buffer(n_us * groups * 4, DType::F32, vec![n_us, groups])?;
    s_buf.as_mut_slice::<f32>()?.copy_from_slice(&scales);
    let mut b_buf = device.alloc_buffer(n_us * groups * 4, DType::F32, vec![n_us, groups])?;
    b_buf.as_mut_slice::<f32>()?.copy_from_slice(&biases);
    let y_buf = device.alloc_buffer(m_us * n_us * 4, DType::F32, vec![m_us, n_us])?;
    let mut meta = device.alloc_buffer(16, DType::U32, vec![4])?;
    meta.as_mut_slice::<u32>()?
        .copy_from_slice(&[m, n, k, group_size]);

    // Warm up both kernels.
    for _ in 0..3 {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }
    for _ in 0..3 {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32_tiled(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }

    // Per-element kernel timing.
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }
    let dt_pe = t0.elapsed();
    let avg_pe = dt_pe.as_secs_f64() / n_iter as f64;

    // Tiled kernel timing.
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32_tiled(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }
    let dt_tl = t0.elapsed();
    let avg_tl = dt_tl.as_secs_f64() / n_iter as f64;

    // Warm up + time the simdgroup-MMA kernel (iter-15c).
    for _ in 0..3 {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32_simd(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let mut encoder = device.command_encoder()?;
        dispatch_qmm_affine_t_f32_simd(
            &mut encoder, &mut registry, device.metal_device(),
            &x_buf, &q_buf, &s_buf, &b_buf, &y_buf, &meta, m, n, k, group_size,
        )?;
        encoder.commit_and_wait()?;
    }
    let dt_sm = t0.elapsed();
    let avg_sm = dt_sm.as_secs_f64() / n_iter as f64;

    let speedup_tl = avg_pe / avg_tl;
    let speedup_sm = avg_pe / avg_sm;
    let flops_per_call = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let pe_gflops = flops_per_call / avg_pe / 1e9;
    let tl_gflops = flops_per_call / avg_tl / 1e9;
    let sm_gflops = flops_per_call / avg_sm / 1e9;

    println!("[bench] per-element: avg {:.3} ms = {:.1} GFLOPS", avg_pe * 1000.0, pe_gflops);
    println!("[bench] tiled:       avg {:.3} ms = {:.1} GFLOPS", avg_tl * 1000.0, tl_gflops);
    println!("[bench] simd-MMA:    avg {:.3} ms = {:.1} GFLOPS", avg_sm * 1000.0, sm_gflops);
    println!("[bench] speedup tiled / per-element = {speedup_tl:.2}×");
    println!("[bench] speedup simd  / per-element = {speedup_sm:.2}×");
    Ok(())
}
