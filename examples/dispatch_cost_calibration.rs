//! ADR-015 P3a — Per-dispatch encode cost calibration on M5 Max.
//!
//! P2 (`cb_cost_calibration.rs`) measured async µs/CB ≈ 1.6 µs and
//! showed that the hf2q→llama.cpp 0.5 ms/token decode gap can only
//! be ~32% explained by CB-count alone.  The remaining ~340 µs/token
//! must live in **per-dispatch** encode cost or **Rust-side
//! orchestration overhead**.
//!
//! This bench separates "per-dispatch encode" from "per-CB submit"
//! by encoding many dispatches into a single CB:
//!
//! 1. Set up a 1-element BF16 buffer (input + output).
//! 2. Encode N copies of `dispatch_scalar_mul_bf16` into one
//!    `CommandEncoder` — no commit, no wait.
//! 3. Time the encoding loop alone.
//! 4. Then commit + wait once and time submit + GPU.
//! 5. Repeat for several N to see linearity.
//!
//! ## Reading the numbers
//!
//! - `µs/dispatch` ≈ 0.14 (matching llama.cpp's implied per-dispatch
//!   cost: ~150 µs CPU encode for ~1070 dispatches per token):
//!   the residual gap is **NOT** per-dispatch.  Lever lives elsewhere
//!   (Rust orchestration, buffer pool, barrier bookkeeping).
//! - `µs/dispatch` ≫ 0.14: hf2q's per-dispatch encoding has overhead
//!   llama.cpp doesn't.  Lever is shader-launch path optimization.
//! - hf2q decode at ~1070 dispatches/token × measured µs/dispatch
//!   gives the actual per-dispatch budget; compare to llama.cpp's
//!   ~150 µs total CPU encode (per ADR-012's analysis).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example dispatch_cost_calibration
//! ```
//!
//! Run on a cold SoC per `feedback_perf_gate_thermal_methodology`.

use mlx_native::ops::elementwise::dispatch_scalar_mul_bf16_with_encoder;
use mlx_native::{DType, KernelRegistry, MlxDevice, Result};
use std::time::Instant;

const REPS: usize = 5;
const N_ELEMENTS: u32 = 1;

fn alloc_bf16_one(device: &MlxDevice) -> Result<mlx_native::MlxBuffer> {
    // BF16 = 2 bytes per element.  N_ELEMENTS=1 => 2 bytes total.
    device.alloc_buffer(2, DType::BF16, vec![1])
}

fn run_encode_only(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &mlx_native::MlxBuffer,
    output: &mlx_native::MlxBuffer,
    n: usize,
) -> Result<(f64, f64)> {
    let mut enc = device.command_encoder()?;
    let metal_dev = device.metal_device();

    let t0 = Instant::now();
    for _ in 0..n {
        dispatch_scalar_mul_bf16_with_encoder(
            &mut enc, registry, metal_dev, input, output, N_ELEMENTS, 1.0_f32,
        )?;
    }
    let encode_ms = t0.elapsed().as_secs_f64() * 1_000.0;

    let t1 = Instant::now();
    enc.commit_and_wait()?;
    let commit_ms = t1.elapsed().as_secs_f64() * 1_000.0;

    Ok((encode_ms, commit_ms))
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() -> Result<()> {
    let device = MlxDevice::new()?;
    let mut registry = KernelRegistry::new();
    let input = alloc_bf16_one(&device)?;
    let output = alloc_bf16_one(&device)?;

    eprintln!("=== ADR-015 P3a — Per-dispatch encode cost calibration ===");
    eprintln!("Op: scalar_mul_bf16 on N_ELEMENTS=1 (minimal GPU work).");
    eprintln!("REPS={REPS} (median reported per cell).");
    eprintln!();

    // Warm-up: encode + commit a small CB to settle pipeline + JIT.
    {
        let mut enc = device.command_encoder()?;
        for _ in 0..10 {
            dispatch_scalar_mul_bf16_with_encoder(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &input,
                &output,
                N_ELEMENTS,
                1.0,
            )?;
        }
        enc.commit_and_wait()?;
    }

    println!("| N        | encode_ms (med) | commit_ms (med) | µs/dispatch (encode) |");
    println!("|---------:|----------------:|----------------:|---------------------:|");

    for &n in &[10usize, 50, 100, 500, 1_000, 5_000] {
        let mut encode_samples = Vec::with_capacity(REPS);
        let mut commit_samples = Vec::with_capacity(REPS);

        for _ in 0..REPS {
            let (enc_ms, com_ms) =
                run_encode_only(&device, &mut registry, &input, &output, n)?;
            encode_samples.push(enc_ms);
            commit_samples.push(com_ms);
        }

        let enc_med = median(encode_samples);
        let com_med = median(commit_samples);
        let us_per = enc_med * 1_000.0 / (n as f64);

        println!(
            "| {n:>8} | {enc_med:>15.3} | {com_med:>15.3} | {us_per:>20.3} |"
        );
    }

    eprintln!();
    eprintln!("Reference: llama.cpp implies ~0.14 µs/dispatch (~150 µs / ~1070 dispatches).");
    eprintln!("If hf2q ≈ 0.14 µs/dispatch, gap residual lives in Rust orchestration.");
    eprintln!("If hf2q ≫ 0.14 µs/dispatch, gap residual lives in shader-launch path.");

    Ok(())
}
