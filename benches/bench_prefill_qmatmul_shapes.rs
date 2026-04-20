//! Microbenchmark: per-shape dense quantized matmul (tensor-ops path)
//! at the exact shapes hf2q dispatches during Gemma 4 26B DWQ prefill
//! at seq_len=2455 on M5 Max.
//!
//! Purpose: decompose the per-shape wall-clock of `kernel_mul_mm_*_tensor_f32`
//! so we can tell whether the hf2q-vs-llama.cpp prefill gap lives in
//! per-call kernel performance or in dispatch/scheduling overhead.
//!
//! Shapes derived from the Gemma 4 26B A4B DWQ GGUF metadata:
//!   embedding_length = 2816
//!   feed_forward_length (dense MLP) = 2112
//!   num_attention_heads = 16
//!   num_key_value_heads = 8 (sliding) / 2 (global, 5 layers)
//!   head_dim = 256 (sliding) / 512 (global)
//!   num_experts = 128
//!
//! Tensor types from the GGUF:
//!   Sliding-layer Q/K/V/O, router, MLP gate/up: Q6_K
//!   MLP down:                                    Q8_0
//!   Global-layer Q/K/V/O (5 layers):             Q4_0
//!
//! Each shape is warmed up, then timed over N iterations via
//! `command_encoder().commit_and_wait()`.  Report: median ms per call
//! and effective TFLOP/s (2*M*N*K / time).  Running this bench tells us
//! whether our tensor kernel is within the M5 Max tensor-core envelope.
//!
//! Invoke:
//!   cargo bench --bench bench_prefill_qmatmul_shapes --release

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
    per_layer: usize, // how many times per prefill at pp2455
}

/// Actual prefill shapes (M=seq_len=2455).  "count" is calls per prefill
/// so we can translate per-call ms into total prefill ms.
const SHAPES: &[ShapeCase] = &[
    // --- Sliding-layer attention (25 layers) ---
    ShapeCase { label: "Q_sliding", m: 2455, n: 4096, k: 2816, qtype: GgmlType::Q6_K, per_layer: 25 },
    ShapeCase { label: "K_sliding", m: 2455, n: 2048, k: 2816, qtype: GgmlType::Q6_K, per_layer: 25 },
    ShapeCase { label: "V_sliding", m: 2455, n: 2048, k: 2816, qtype: GgmlType::Q6_K, per_layer: 25 },
    ShapeCase { label: "O_sliding", m: 2455, n: 2816, k: 4096, qtype: GgmlType::Q6_K, per_layer: 25 },
    // --- Global-layer attention (5 layers, Q4_0) ---
    ShapeCase { label: "Q_global",  m: 2455, n: 8192, k: 2816, qtype: GgmlType::Q4_0, per_layer: 5 },
    ShapeCase { label: "K_global",  m: 2455, n: 1024, k: 2816, qtype: GgmlType::Q4_0, per_layer: 5 },
    ShapeCase { label: "V_global",  m: 2455, n: 1024, k: 2816, qtype: GgmlType::Q4_0, per_layer: 5 },
    ShapeCase { label: "O_global",  m: 2455, n: 2816, k: 8192, qtype: GgmlType::Q4_0, per_layer: 5 },
    // --- Dense MLP (30 layers, Q6_K for gate/up, Q8_0 for down) ---
    ShapeCase { label: "MLP_gate",  m: 2455, n: 2112, k: 2816, qtype: GgmlType::Q6_K, per_layer: 30 },
    ShapeCase { label: "MLP_up",    m: 2455, n: 2112, k: 2816, qtype: GgmlType::Q6_K, per_layer: 30 },
    ShapeCase { label: "MLP_down",  m: 2455, n: 2816, k: 2112, qtype: GgmlType::Q8_0, per_layer: 30 },
    // --- Router (30 layers, Q6_K) ---
    ShapeCase { label: "Router",    m: 2455, n:  128, k: 2816, qtype: GgmlType::Q6_K, per_layer: 30 },
];

const WARMUP_ITERS: usize = 3;
const MEASURE_ITERS: usize = 10;

fn qtype_block_bytes(qt: GgmlType) -> u64 {
    match qt {
        GgmlType::Q4_0 => 18,   // 32 values / 18 bytes
        GgmlType::Q8_0 => 34,   // 32 values / 34 bytes
        GgmlType::Q6_K => 210,  // 256 values / 210 bytes
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

    // Measure — per-iteration commit_and_wait so each sample is a full
    // kernel dispatch + GPU completion (matches what hf2q's prefill
    // profiler sees).
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
        "{:<12} {:>5} {:>5} {:>5} {:>6} {:>8} {:>9} {:>9} {:>8} {:>8}",
        "shape", "M", "N", "K", "qtype", "median_ms", "GFLOPs", "TFLOP/s", "layers", "prefill_ms"
    );
    println!("{}", "-".repeat(104));

    let mut total_prefill_ms = 0.0;
    for case in SHAPES {
        let median = bench_one(case, &device, &mut registry);
        let flops = 2.0 * (case.m as f64) * (case.n as f64) * (case.k as f64);
        let gflops = flops / 1e9;
        let tflops = flops / (median / 1000.0) / 1e12;
        let per_prefill = median * case.per_layer as f64;
        total_prefill_ms += per_prefill;
        println!(
            "{:<12} {:>5} {:>5} {:>5} {:>6?} {:>8.3} {:>9.1} {:>9.2} {:>8} {:>8.1}",
            case.label, case.m, case.n, case.k, case.qtype,
            median, gflops, tflops, case.per_layer, per_prefill,
        );
    }
    println!("{}", "-".repeat(104));
    println!(
        "Total dense-qmatmul time at pp2455: {:.1} ms\n\
         Compare to [MM_PROFILE] dense qmatmul total in HF2Q_PROFILE_MM run.",
        total_prefill_ms
    );
}
