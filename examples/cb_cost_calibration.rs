//! ADR-015 P2 — Empty-CB encode/submit cost calibration on M5 Max.
//!
//! Refutes or confirms the working assumption from ADR-012 §Optimize that
//! the hf2q→llama.cpp 0.5 ms/token decode gap is dominated by
//! CPU-side per-CB scheduling overhead at ~5 µs/CB × ~100 CB/token.
//!
//! ## Method
//!
//! Three regimes, each timed end-to-end on the host CPU with
//! `std::time::Instant`.  For each regime we run R fresh trials and
//! report median:
//!
//! 1. **`async N`** — `for _ in 0..N { device.command_encoder()?.commit() }`
//!    plus a single terminal `commit_and_wait` to drain the queue.
//!    Measures CB allocation + submission cost when the CPU does not
//!    block per CB.  Closest to hf2q's decode hot path.
//!
//! 2. **`sync N`** — `for _ in 0..N { device.command_encoder()?.commit_and_wait()? }`
//!    Same workload with a CPU sync per CB.  Upper bound on what
//!    GPU-finished round-trip costs.
//!
//! 3. **`alloc-only N`** — `for _ in 0..N { let _ = device.command_encoder()? }`
//!    Encoder allocated and dropped without commit.  Isolates Metal
//!    command-buffer construction cost from submission/sync.
//!
//! N values cover the hf2q decode regime (real 35B-A3B uses ~100
//! CBs / token; we extend to 1000 to see slope linearity).
//!
//! ## Reading the numbers
//!
//! - If `async µs/CB` ≪ 5 µs, the ADR-015 single-CB upper bound shrinks
//!   proportionally and the `qwen35` decode gap may have a different
//!   primary cause than ADR-012's working assumption.
//! - If `async µs/CB` ≈ 5 µs, ADR-015's ~6% headroom is realistic.
//! - If `async µs/CB` ≫ 5 µs, the gap is bigger than ADR-012 estimated
//!   and the win is correspondingly larger.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example cb_cost_calibration
//! ```
//!
//! Run on a cold SoC per `feedback_perf_gate_thermal_methodology`.

use mlx_native::{MlxDevice, Result};
use std::time::Instant;

const REPS: usize = 5;

fn run_async(device: &MlxDevice, n: usize) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..n {
        let mut enc = device.command_encoder()?;
        enc.commit();
    }
    // Drain the Metal serial queue so the time we measure includes the
    // actual cost of N commits and not a partial async pipeline.
    let mut drain = device.command_encoder()?;
    drain.commit_and_wait()?;
    Ok(t0.elapsed().as_secs_f64() * 1_000.0)
}

fn run_sync(device: &MlxDevice, n: usize) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..n {
        let mut enc = device.command_encoder()?;
        enc.commit_and_wait()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1_000.0)
}

fn run_alloc_only(device: &MlxDevice, n: usize) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..n {
        // Allocated and dropped at end of scope.  The drop must commit
        // an empty buffer or the queue stalls — see ADR-015 P2 first
        // attempt that pinned a 442 GB VSZ before we drained.  Empirically
        // the simplest safe alloc-only is: alloc, immediately commit
        // (async, no wait), and let the queue drain at function exit
        // via a final blocking commit.
        let mut enc = device.command_encoder()?;
        enc.commit();
    }
    let mut drain = device.command_encoder()?;
    drain.commit_and_wait()?;
    Ok(t0.elapsed().as_secs_f64() * 1_000.0)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() -> Result<()> {
    let device = MlxDevice::new()?;

    eprintln!("=== ADR-015 P2 — Empty-CB cost calibration ===");
    eprintln!("REPS={REPS} (median reported per cell)");
    eprintln!();

    // Warm-up: 100 sync round-trips primes the Metal pipeline.
    for _ in 0..100 {
        let mut enc = device.command_encoder()?;
        enc.commit_and_wait()?;
    }

    println!("| N    | regime     | wall_ms (median) | µs/CB |");
    println!("|-----:|:-----------|-----------------:|------:|");

    // hf2q decode regime: ~100 CBs / token (measured fa3f9d6).  Cap at
    // 1000 so we see slope linearity without piling state on the queue.
    for &n in &[10usize, 50, 100, 200, 500, 1000] {
        // async
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            samples.push(run_async(&device, n)?);
        }
        let m = median(samples);
        println!(
            "| {n:>4} | async      | {m:>16.3} | {us:>5.2} |",
            us = m * 1_000.0 / (n as f64)
        );

        // sync
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            samples.push(run_sync(&device, n)?);
        }
        let m = median(samples);
        println!(
            "| {n:>4} | sync       | {m:>16.3} | {us:>5.2} |",
            us = m * 1_000.0 / (n as f64)
        );

        // alloc-only-with-commit (drops would stall queue)
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            samples.push(run_alloc_only(&device, n)?);
        }
        let m = median(samples);
        println!(
            "| {n:>4} | alloc+cmit | {m:>16.3} | {us:>5.2} |",
            us = m * 1_000.0 / (n as f64)
        );
    }

    eprintln!();
    eprintln!("Done.  Compare async µs/CB to ADR-015 working assumption (~5 µs/CB).");

    Ok(())
}
