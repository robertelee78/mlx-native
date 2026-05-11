//! ADR-028 iter-386 — synthetic bench: serial vs multi-thread encoding.
//!
//! Goal: prove (or falsify) that multi-thread cmd-buf encoding beats
//! single-thread before committing to the multi-day forward_decode
//! refactor (iter-387+).
//!
//! Pattern:
//! - Set up N independent qmatmul ops (each writes to a different output buf)
//! - Single-thread baseline: main thread encodes all N + commit_and_wait
//! - Parallel variant: main encodes N/2 into buf0; worker encodes N/2 into
//!   buf1; commit both; wait for both
//! - Measure wall time of both
//!
//! Note: this bench measures ENCODING wall time + GPU wait, not pure
//! encoding overhead.  GPU wait should be similar across both variants
//! (same total compute).  The delta = parallel encoding savings.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::encoder_worker::EncoderWorker;
use mlx_native::{
    quantized_matmul_ggml, DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxBuffer,
    MlxDevice,
};
use std::sync::Arc;
use std::time::Instant;

/// Create a Q6_K-packed weight buffer of shape [n_rows, k_cols].
fn make_q6_k_weight(device: &MlxDevice, n_rows: usize, k_cols: usize) -> MlxBuffer {
    assert!(k_cols % 256 == 0, "k_cols must be multiple of QK_K=256");
    let blocks_per_row = k_cols / 256;
    let bytes_per_block = 210;
    let total_bytes = n_rows * blocks_per_row * bytes_per_block;
    // Allocate raw bytes; values are arbitrary for bench (no parity check).
    device
        .alloc_buffer(total_bytes, DType::U8, vec![total_bytes])
        .expect("alloc weight")
}

fn make_f32_buf(device: &MlxDevice, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc f32")
}

#[test]
fn iter386_serial_vs_parallel_encoding_bench() {
    // Synthetic workload: 60 independent qmatmul ops at gemma4-like shapes
    // (m=1, n=2816, k=2816 — typical Q/K/V/O proj). 60 = 2 ops/layer × 30 layers.
    const N_OPS: usize = 60;
    const M: u32 = 1;
    const N: u32 = 2816;
    const K: u32 = 2816;

    let device = MlxDevice::new().expect("MlxDevice");
    let device_arc = Arc::new(device);

    // Pre-allocate everything (don't measure alloc cost).
    let weights: Vec<Arc<MlxBuffer>> = (0..N_OPS)
        .map(|_| Arc::new(make_q6_k_weight(&device_arc, N as usize, K as usize)))
        .collect();
    let inputs: Vec<Arc<MlxBuffer>> = (0..N_OPS)
        .map(|_| Arc::new(make_f32_buf(&device_arc, K as usize)))
        .collect();
    let outputs: Vec<Arc<MlxBuffer>> = (0..N_OPS)
        .map(|_| Arc::new(make_f32_buf(&device_arc, N as usize)))
        .collect();

    let params = GgmlQuantizedMatmulParams {
        m: M,
        n: N,
        k: K,
        ggml_type: GgmlType::Q6_K,
    };

    // ---- Warmup pipelines ----
    {
        let mut registry = KernelRegistry::new();
        let mut enc = device_arc.command_encoder().expect("enc");
        quantized_matmul_ggml(
            &mut enc,
            &mut registry,
            &device_arc,
            &inputs[0],
            &weights[0],
            &outputs[0],
            &params,
        )
        .expect("warmup");
        enc.commit_and_wait().expect("warmup wait");
    }

    // ---- Serial baseline (single-thread encode all N_OPS) ----
    let mut serial_times = Vec::new();
    for _trial in 0..5 {
        let mut registry = KernelRegistry::new();
        let t0 = Instant::now();
        let mut enc = device_arc.command_encoder().expect("enc");
        for i in 0..N_OPS {
            quantized_matmul_ggml(
                &mut enc,
                &mut registry,
                &device_arc,
                &inputs[i],
                &weights[i],
                &outputs[i],
                &params,
            )
            .expect("dispatch");
        }
        enc.commit_and_wait().expect("commit");
        serial_times.push(t0.elapsed().as_micros() as f64);
    }
    let serial_min = serial_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let serial_mean: f64 = serial_times.iter().sum::<f64>() / serial_times.len() as f64;
    println!("Serial   : min {:.1} µs  mean {:.1} µs  trials {:?}",
             serial_min, serial_mean, serial_times.iter().map(|x| *x as u64).collect::<Vec<_>>());

    // ---- Parallel variant (main: N/2 ops; worker: N/2 ops) ----
    let worker = EncoderWorker::spawn();
    let mut parallel_times = Vec::new();
    for _trial in 0..5 {
        let device_main = Arc::clone(&device_arc);
        let device_worker = Arc::clone(&device_arc);
        let inputs_first: Vec<_> = inputs[..N_OPS / 2].iter().map(Arc::clone).collect();
        let weights_first: Vec<_> = weights[..N_OPS / 2].iter().map(Arc::clone).collect();
        let outputs_first: Vec<_> = outputs[..N_OPS / 2].iter().map(Arc::clone).collect();
        let inputs_second: Vec<_> = inputs[N_OPS / 2..].iter().map(Arc::clone).collect();
        let weights_second: Vec<_> = weights[N_OPS / 2..].iter().map(Arc::clone).collect();
        let outputs_second: Vec<_> = outputs[N_OPS / 2..].iter().map(Arc::clone).collect();

        let (worker_done_tx, worker_done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let t0 = Instant::now();

        // Submit second-half encoding to worker (runs concurrently with main encoding).
        worker.submit(move || {
            let result = (|| -> Result<(), String> {
                let mut registry = KernelRegistry::new();
                let mut enc = device_worker.command_encoder()
                    .map_err(|e| format!("enc: {e}"))?;
                for i in 0..inputs_second.len() {
                    quantized_matmul_ggml(
                        &mut enc, &mut registry, &device_worker,
                        &inputs_second[i], &weights_second[i], &outputs_second[i],
                        &params,
                    ).map_err(|e| format!("dispatch: {e}"))?;
                }
                enc.commit_and_wait().map_err(|e| format!("commit: {e}"))?;
                Ok(())
            })();
            worker_done_tx.send(result).ok();
        }).expect("submit");

        // Main thread encodes first half concurrently.
        let mut registry = KernelRegistry::new();
        let mut enc = device_main.command_encoder().expect("enc main");
        for i in 0..inputs_first.len() {
            quantized_matmul_ggml(
                &mut enc, &mut registry, &device_main,
                &inputs_first[i], &weights_first[i], &outputs_first[i],
                &params,
            ).expect("dispatch main");
        }
        enc.commit_and_wait().expect("commit main");

        // Wait for worker to finish too.
        worker_done_rx.recv().expect("worker died").expect("worker error");

        parallel_times.push(t0.elapsed().as_micros() as f64);
    }
    let parallel_min = parallel_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let parallel_mean: f64 = parallel_times.iter().sum::<f64>() / parallel_times.len() as f64;
    println!("Parallel : min {:.1} µs  mean {:.1} µs  trials {:?}",
             parallel_min, parallel_mean, parallel_times.iter().map(|x| *x as u64).collect::<Vec<_>>());

    // ---- Report ----
    let speedup_min = serial_min / parallel_min;
    let speedup_mean = serial_mean / parallel_mean;
    println!("\n=== iter-386 SYNTHETIC BENCH RESULT ===");
    println!("  N ops      = {}", N_OPS);
    println!("  Serial min = {:.1} µs", serial_min);
    println!("  Parallel min= {:.1} µs", parallel_min);
    println!("  Speedup (min) = {:.3}x", speedup_min);
    println!("  Speedup (mean) = {:.3}x", speedup_mean);
    println!("  Verdict = {}",
             if speedup_min > 1.05 { "✓ PARALLEL WINS (>5%) — proceed iter-387+" }
             else if speedup_min > 0.98 { "≈ ROUGHLY EQUAL (-2%..+5%) — marginal, judgment call" }
             else { "✗ PARALLEL LOSES — abandon multi-thread approach" });
}
