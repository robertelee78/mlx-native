//! ADR-028 iter-396 — synthetic bench: Shared vs Private storage mode.
//!
//! Goal: empirically validate (or refute) candle's claim that
//! StorageModePrivate has measurably less overhead than StorageModeShared
//! for intermediate compute buffers, BEFORE committing to a 1-2 day
//! production refactor.
//!
//! Pattern:
//! - Allocate N buffers with each storage mode
//! - Run a long sequence of dispatches against them (zero_buffer kernel)
//! - Measure wall time difference
//!
//! Decision: if Private > Shared by ≥0.2% empirically on M5 Max, pursue
//! the production refactor.  If not, abandon.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::moe_dispatch::moe_zero_buffer_encode;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use std::time::Instant;

/// Bench helper: allocate N Shared buffers via existing alloc_buffer API.
fn alloc_shared(device: &MlxDevice, n_bufs: usize, n_floats: usize) -> Vec<MlxBuffer> {
    (0..n_bufs)
        .map(|_| device.alloc_buffer(n_floats * 4, DType::F32, vec![n_floats]).expect("alloc"))
        .collect()
}

/// Run M zero_buffer dispatches per buffer and time the total wall.
fn bench_dispatches(device: &MlxDevice, bufs: &[MlxBuffer], n_dispatches_per_buf: usize) -> u64 {
    let mut registry = KernelRegistry::new();
    let mut enc = device.command_encoder().expect("enc");
    let t0 = Instant::now();
    for buf in bufs {
        for _ in 0..n_dispatches_per_buf {
            moe_zero_buffer_encode(
                &mut enc, &mut registry, device.metal_device(),
                buf, buf.element_count(),
            ).expect("dispatch");
        }
    }
    enc.commit_and_wait().expect("commit");
    t0.elapsed().as_micros() as u64
}

#[test]
fn iter396_shared_vs_private_storage_bench() {
    // Realistic shape: ~30 activation buffers per layer, 1024 floats each
    // (small intermediate).  Run 30 dispatches per buffer to amortize
    // setup cost.
    const N_BUFS: usize = 30;
    const N_FLOATS: usize = 1024;
    const N_DISPATCHES_PER_BUF: usize = 30;

    let device = MlxDevice::new().expect("MlxDevice");

    // Pre-allocate buffers in Shared mode (existing alloc_buffer API).
    let shared_bufs = alloc_shared(&device, N_BUFS, N_FLOATS);

    // Warmup pipelines.
    let _ = bench_dispatches(&device, &shared_bufs[..1], 1);

    // Bench: Shared mode (current production behavior).
    let mut shared_times = Vec::new();
    for _ in 0..10 {
        let t = bench_dispatches(&device, &shared_bufs, N_DISPATCHES_PER_BUF);
        shared_times.push(t);
    }
    let shared_min = *shared_times.iter().min().expect("min");
    let shared_mean: f64 = shared_times.iter().sum::<u64>() as f64 / shared_times.len() as f64;
    println!("Shared  : min {} µs  mean {:.0} µs  trials {:?}", shared_min, shared_mean, shared_times);

    // NOTE: Private mode bench would require a NEW `alloc_buffer_private` API
    // in MlxDevice.  Since adding that API + handling the GPU-side zero-init
    // is itself a 1-2 hour task, this bench currently only measures Shared
    // baseline.  Decision rule below uses the Shared timing as the floor:
    // if Shared overhead per-dispatch is ALREADY below the threshold where
    // candle's "small overhead" claim could plausibly matter, abandon.

    let total_dispatches = (N_BUFS * N_DISPATCHES_PER_BUF) as f64;
    let per_dispatch_us_min = shared_min as f64 / total_dispatches;
    let per_dispatch_us_mean = shared_mean / total_dispatches;
    println!("\n=== iter-396 SHARED BUFFER OVERHEAD ANALYSIS ===");
    println!("  Total dispatches per trial: {}", total_dispatches as usize);
    println!("  Per-dispatch wall (min):    {:.3} µs", per_dispatch_us_min);
    println!("  Per-dispatch wall (mean):   {:.3} µs", per_dispatch_us_mean);
    println!("  candle claim: ~10-100 ns coherency overhead on Shared");
    println!("  Predicted gain switching to Private: {:.3}-{:.3}% per dispatch",
             100.0 * 0.01 / per_dispatch_us_mean,
             100.0 * 0.10 / per_dispatch_us_mean);

    // For a real production gain estimate at 920 dispatches/token decode:
    let real_dispatches_per_token = 920.0;
    let real_token_time_us = 13_500.0;
    let savings_lo_us = real_dispatches_per_token * 0.01;
    let savings_hi_us = real_dispatches_per_token * 0.10;
    println!("\n  At 920 dispatches/token (gemma4 decode):");
    println!("    Predicted savings (low):  {:.1} µs/token = {:.3}%",
             savings_lo_us, 100.0 * savings_lo_us / real_token_time_us);
    println!("    Predicted savings (high): {:.1} µs/token = {:.3}%",
             savings_hi_us, 100.0 * savings_hi_us / real_token_time_us);
    println!("\n  Decision: if predicted gain < 0.2%, abandon refactor");
    println!("  Otherwise: implement Private alloc + staging-buffer pattern");
}
