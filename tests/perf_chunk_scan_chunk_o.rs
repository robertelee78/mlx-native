//! Wave 5b.2 iter 2 — RED perf gate for `gated_delta_net_chunk_o_bf16`.
//!
//! Asserts the second-largest chunk-pipeline kernel achieves the iter-2
//! ≥ 3× speedup target on Apple M5 Max relative to the 2026-04-27 baseline
//! of 2.155 ms recorded in `docs/wave5b2-iter1-baseline.md`.
//!
//! Bar: median wall ≤ 720 µs across 50 fresh dispatches at long-prefill
//! shape (B=1, T=4096, Hg=2, H=4, K=128, V=128, BT=64, NT=64). The bench
//! harness at `benches/bench_chunk_scan_pipeline.rs` already measures this
//! median; this test is the RED→GREEN gate that travels with the source
//! tree so `cargo test` can enforce the speedup contract without
//! `cargo bench`.
//!
//! # Methodology
//!
//! - Single-process, single-threadgroup test (no inter-test isolation
//!   needed; the kernel is stateless across dispatches).
//! - Buffer allocation hoisted out of the timed loop.
//! - First 5 dispatches warm the pipeline cache (excluded from the median).
//! - 50 timed dispatches; median used as the wall-time statistic (robust
//!   to OS-jitter outliers).
//! - Wall = `command_buffer.commit + wait_until_completed` per dispatch
//!   (the natural client-observable latency, same as the bench harness).
//!
//! # Why this RED test exists
//!
//! `cargo bench` is opt-in (manual run) and produces criterion HTML
//! reports rather than pass/fail signals. The RED test makes the speedup
//! a tracked invariant: anyone running `cargo test --release` sees the
//! perf contract enforced alongside correctness, and CI gates it without
//! needing to parse criterion JSON.
//!
//! # Bar derivation
//!
//! Baseline: 2.155 ms median (HEAD `cf5f420`, M5 Max, 2026-04-27, post
//! Wave 5b.2 iter 1.5 inter_state work — chunk_o was untouched).
//! Target: ≥ 3× speedup ⇒ wall ≤ 718 µs.
//! Bar (with margin for jitter): **720 µs**.
//!
//! chunk_o has THREE matmul sections (q·h^T, q·k^T, b_A·v_new) vs
//! inter_state's two; should benefit MORE from MMA, not less. If the
//! observed speedup is < 2.5× the hypothesis is falsified — likely
//! root cause is thread-divergence on the causal mask blocking
//! simdgroup uniformity, or the bf16 cast on b_A bottlenecking the
//! third dot.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::time::Instant;

use mlx_native::ops::gated_delta_net_chunk_o::{
    self, build_gated_delta_net_chunk_o_params, dispatch_gated_delta_net_chunk_o,
    GatedDeltaNetChunkOParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// Long-prefill shape — must match `benches/bench_chunk_scan_pipeline.rs`.
const B: u32 = 1;
const T: u32 = 4096;
const HG: u32 = 2;
const H: u32 = 4;
const K: u32 = 128;
const V: u32 = 128;
const BT: u32 = 64;

const WARMUP_DISPATCHES: usize = 5;
const TIMED_DISPATCHES: usize = 50;

/// 3× speedup vs 2.155 ms baseline = 718 µs; +0.3% jitter headroom = 720 µs.
const SPEEDUP_BAR_MS: f64 = 0.720;

fn alloc_bf16(device: &MlxDevice, n_elems: usize, fill: f32) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n_elems * 2, DType::BF16, vec![n_elems])
        .expect("alloc bf16");
    {
        let dst = buf.as_mut_slice::<u16>().expect("mut bf16");
        let bf16_bits = (fill.to_bits() >> 16) as u16;
        for v in dst.iter_mut() {
            *v = bf16_bits;
        }
    }
    buf
}

fn alloc_f32(device: &MlxDevice, n_elems: usize, fill: f32) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n_elems * 4, DType::F32, vec![n_elems])
        .expect("alloc f32");
    {
        let dst = buf.as_mut_slice::<f32>().expect("mut f32");
        for v in dst.iter_mut() {
            *v = fill;
        }
    }
    buf
}

#[test]
fn chunk_o_simdgroup_matrix_speedup() {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available — skipping chunk_o perf gate");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    gated_delta_net_chunk_o::register(&mut registry);

    let scale: f32 = (K as f32).powf(-0.5);
    let p = GatedDeltaNetChunkOParams {
        b: B,
        t: T,
        hg: HG,
        h: H,
        k: K,
        v: V,
        bt: BT,
        scale,
    };
    let nt = p.num_chunks();

    // Inputs: small non-zero fill keeps the kernel from being
    // optimized into a no-op while staying away from overflow.
    let q_buf = alloc_bf16(&device, (B * T * HG * K) as usize, 0.01);
    let k_buf = alloc_bf16(&device, (B * T * HG * K) as usize, 0.01);
    let v_buf = alloc_bf16(&device, (B * T * H * V) as usize, 0.01);
    let h_buf = alloc_bf16(&device, (B * nt * H * V * K) as usize, 0.01);
    let g_buf = alloc_f32(&device, (B * T * H) as usize, 0.0);
    let o_buf = alloc_bf16(&device, (B * T * H * V) as usize, 0.0);

    let params_buf = build_gated_delta_net_chunk_o_params(&device, p).expect("build params");

    // Warm up — let the pipeline cache fill, command-buffer pool stabilize.
    for _ in 0..WARMUP_DISPATCHES {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_chunk_o(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &k_buf,
            &v_buf,
            &h_buf,
            &g_buf,
            &o_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    // Timed loop — record per-dispatch wall.
    let mut samples_us: Vec<u64> = Vec::with_capacity(TIMED_DISPATCHES);
    for _ in 0..TIMED_DISPATCHES {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_chunk_o(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &k_buf,
            &v_buf,
            &h_buf,
            &g_buf,
            &o_buf,
            &params_buf,
            p,
        )
        .expect("dispatch");

        let t0 = Instant::now();
        enc.commit_and_wait().expect("commit");
        let elapsed_us = t0.elapsed().as_micros() as u64;
        samples_us.push(elapsed_us);
    }

    samples_us.sort_unstable();
    let median_us = samples_us[TIMED_DISPATCHES / 2];
    let p10_us = samples_us[TIMED_DISPATCHES / 10];
    let p90_us = samples_us[(TIMED_DISPATCHES * 9) / 10];

    let median_ms = (median_us as f64) / 1000.0;
    let p10_ms = (p10_us as f64) / 1000.0;
    let p90_ms = (p90_us as f64) / 1000.0;

    eprintln!(
        "chunk_o perf:       median = {median_ms:.3} ms   p10 = {p10_ms:.3} ms   p90 = {p90_ms:.3} ms   bar = {SPEEDUP_BAR_MS:.3} ms"
    );
    eprintln!(
        "                    baseline = 2.155 ms   observed speedup = {:.2}x",
        2.155 / median_ms
    );

    assert!(
        median_ms <= SPEEDUP_BAR_MS,
        "chunk_o median wall = {median_ms:.3} ms — failed ≥ 3× speedup gate (bar {SPEEDUP_BAR_MS:.3} ms = 2.155 ms / 3 + jitter headroom). \
         simdgroup_matrix MMA optimization either underperformed or was reverted."
    );
}
