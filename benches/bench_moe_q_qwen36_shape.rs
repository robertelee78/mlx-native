//! Microbenchmark: per-shape dense quantized matmul (tensor-ops path)
//! at the exact shapes hf2q dispatches during Qwen3.6 27B DWQ46 prefill
//! at seq_len=4096 on M5 Max.
//!
//! W-5b.13 cross-fence audit (2026-04-27): the W-5b.11 hand-off named the
//! dominant 47.4% bucket as `mlx_native::ops::moe_q::*`, but the production
//! 27B DWQ46 GGUF is **dense FFN with Q4_0** (no `_exps` tensors; per-layer
//! gate/up/down each Q4_0 of shape 5120×17408 / 17408×5120).  The actual
//! kernel is `kernel_mul_mm_q4_0_tensor_f32`, dispatched via
//! `quantized_matmul_ggml::dispatch_mm` for any seq>8.
//!
//! Shape derivation (from /opt/hf2q/models/qwen3.6-27b-dwq46/config.json
//! and direct GGUF tensor inspection):
//!   hidden_size           = 5120
//!   intermediate_size     = 17408
//!   num_hidden_layers     = 64 (16 full_attention + 48 linear_attention)
//!   ffn_gate.weight       Q4_0 (5120, 17408)  — per layer
//!   ffn_up.weight         Q4_0 (5120, 17408)  — per layer
//!   ffn_down.weight       Q4_0 (17408, 5120)  — per layer
//!
//! Each shape is warmed up, then timed over N iterations via
//! `command_encoder().commit_and_wait()`.  Report: median ms per call,
//! effective TFLOP/s (2*M*N*K / time), and per-prefill total.
//!
//! Compare the per-prefill total to the W-5b.11 measurement of
//! `layer.ffn_dispatch` = 9 750 ms (all 64 layers) — the fraction NOT
//! covered by these matmul totals is wrapper/dispatch/barrier overhead.
//!
//! Invoke:
//!   cargo bench --bench bench_moe_q_qwen36_shape

use mlx_native::ops::quantized_matmul_ggml::{
    dispatch_mm_for_test, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct ShapeCase {
    label:     &'static str,
    m:         u32,
    n:         u32,
    k:         u32,
    qtype:     GgmlType,
    per_layer: usize, // calls per dense FFN layer (gate=1, up=1, down=1)
    layers:    usize, // total layers calling this shape
}

/// Qwen3.6 27B DWQ46 prefill shapes at M=seq_len=4096.  All 64 layers
/// run the same dense FFN (gate/up/down Q4_0).  Attention projections
/// vary by attention type but share Q4_0; we focus on FFN (the W-5b.11
/// `layer.ffn_dispatch` bucket).
const SHAPES: &[ShapeCase] = &[
    // --- Dense MLP at production prefill (64 layers, all Q4_0) ---
    ShapeCase { label: "FFN_gate", m: 4096, n: 17408, k: 5120,  qtype: GgmlType::Q4_0, per_layer: 1, layers: 64 },
    ShapeCase { label: "FFN_up",   m: 4096, n: 17408, k: 5120,  qtype: GgmlType::Q4_0, per_layer: 1, layers: 64 },
    ShapeCase { label: "FFN_down", m: 4096, n:  5120, k: 17408, qtype: GgmlType::Q4_0, per_layer: 1, layers: 64 },
];

const WARMUP_ITERS: usize = 3;
const MEASURE_ITERS: usize = 10;

fn qtype_block_bytes(qt: GgmlType) -> u64 {
    match qt {
        GgmlType::Q4_0 => 18,
        GgmlType::Q8_0 => 34,
        GgmlType::Q6_K => 210,
        _ => panic!("unsupported qtype in bench"),
    }
}
fn qtype_block_values(qt: GgmlType) -> u64 {
    match qt {
        GgmlType::Q4_0 | GgmlType::Q8_0 => 32,
        GgmlType::Q6_K => 256,
        _ => panic!("unsupported qtype"),
    }
}

fn alloc_weight(device: &MlxDevice, n: u32, k: u32, qt: GgmlType) -> MlxBuffer {
    let blocks_per_row = (k as u64) / qtype_block_values(qt);
    let total_bytes = (n as u64) * blocks_per_row * qtype_block_bytes(qt);
    device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc weight")
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn bench_one(case: &ShapeCase, device: &MlxDevice, registry: &mut KernelRegistry) -> f64 {
    let input  = alloc_f32(device, (case.m as usize) * (case.k as usize), "input");
    let mut output = alloc_f32(device, (case.m as usize) * (case.n as usize), "output");
    let weight = alloc_weight(device, case.n, case.k, case.qtype);

    let params = GgmlQuantizedMatmulParams {
        m: case.m, n: case.n, k: case.k, ggml_type: case.qtype,
    };

    // Warmup (also triggers kernel pipeline compile on first call).
    for _ in 0..WARMUP_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        dispatch_mm_for_test(&mut enc, registry, device, &input, &weight, &mut output, &params)
            .expect("dispatch warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        dispatch_mm_for_test(&mut enc, registry, device, &input, &weight, &mut output, &params)
            .expect("dispatch measure");
        enc.commit_and_wait().expect("measure commit");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        samples.push(elapsed);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!(
        "{:<10} {:>5} {:>6} {:>6} {:>6} {:>10} {:>9} {:>9} {:>7} {:>11}",
        "shape", "M", "N", "K", "qtype", "median_ms", "GFLOPs", "TFLOP/s", "layers", "prefill_ms"
    );
    println!("{}", "-".repeat(108));

    let mut total_ffn_prefill_ms = 0.0;
    for case in SHAPES {
        let median = bench_one(case, &device, &mut registry);
        let flops = 2.0 * (case.m as f64) * (case.n as f64) * (case.k as f64);
        let gflops = flops / 1e9;
        let tflops = flops / (median / 1000.0) / 1e12;
        let calls = case.per_layer * case.layers;
        let per_prefill = median * calls as f64;
        total_ffn_prefill_ms += per_prefill;
        println!(
            "{:<10} {:>5} {:>6} {:>6} {:>6?} {:>10.3} {:>9.1} {:>9.2} {:>7} {:>11.1}",
            case.label, case.m, case.n, case.k, case.qtype,
            median, gflops, tflops, calls, per_prefill,
        );
    }
    println!("{}", "-".repeat(108));
    println!(
        "Total dense FFN qmatmul time at pp4096 (Qwen3.6 27B DWQ46 dense MLP, 64 layers × 3 calls): {:.1} ms",
        total_ffn_prefill_ms
    );
    println!(
        "\nCompare to W-5b.11 measurement (docs/wave5b3-walkbar-results.md):\n\
         layer.ffn_dispatch (all 64 layers) = 9 750 ms\n\
         If kernel_total < 9 750 ms, the gap is wrapper/dispatch/barrier overhead\n\
         outside the qmatmul kernels themselves."
    );
}
