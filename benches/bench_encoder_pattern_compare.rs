//! README-quote bench — single-encoder vs per-N-flush vs per-op encoder
//! patterns.  Backs the "lower per-token overhead" claim in the Design
//! Principles section of README.md.
//!
//! Three patterns at the same total N=200 dispatches per token:
//!
//!   A. mlx-native (`GraphExecutor`):
//!        new_command_buffer → new_compute_command_encoder → N dispatches
//!        → endEncoding → commit → wait
//!      Total: 1 CB, 1 encoder, 1 commit, 1 wait.
//!
//!   B. candle's Metal backend default (`CANDLE_METAL_COMPUTE_PER_BUFFER=50`,
//!      verified in /opt/candle/candle-metal-kernels/src/metal/commands.rs):
//!        for chunk in N/50:
//!          new_command_buffer → new_compute_command_encoder
//!          → 50 dispatches → endEncoding → commit (no wait)
//!        wait on the last CB.
//!      Total: 4 CBs, 4 encoders, 4 commits, 1 wait.
//!
//!   C. Worst-case per-op encoder:
//!        for each of N: new_command_buffer → new_compute_command_encoder
//!        → 1 dispatch → endEncoding → commit → wait
//!      Total: 200 CBs, 200 encoders, 200 commits, 200 waits.
//!
//! Tested across three shapes spanning the realistic decode workload range:
//!   - Router_Q5K  (n=128, k=2816, ~0.045 MB read) — encoder-overhead-dominated
//!   - Q_sliding_Q5K (n=4096, k=2816, ~1.4 MB read) — typical Q/K/V projection
//!   - lmhead_Q6_K (n=262144, k=2816, ~605 MB read) — bandwidth-bound, GPU-dominated
//!
//! Reports wall-time per dispatch in each pattern + ratios (B/A) and (C/A).
//!
//! Run:
//!   cargo bench -p mlx-native --bench bench_encoder_pattern_compare

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const N_DISPATCHES: usize = 200;
const CANDLE_FLUSH: usize = 50; // DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER
const WARMUP: usize = 5;
const MEASURE: usize = 20;

struct ProbeShape {
    label: &'static str,
    n: u32,
    k: u32,
    qtype: GgmlType,
}

const SHAPES: &[ProbeShape] = &[
    // Tiny GPU work — encoder overhead dominates.
    ProbeShape { label: "Router_Q5K", n: 128, k: 2816, qtype: GgmlType::Q5_K },
    // Typical Q/K/V projection at qwen36 prefill shape.
    ProbeShape { label: "Q_sliding_Q5K", n: 4096, k: 2816, qtype: GgmlType::Q5_K },
    // Bandwidth-bound: lm_head Q6_K (605 MB read) — GPU work dominates.
    ProbeShape { label: "lmhead_Q6K", n: 262144, k: 2816, qtype: GgmlType::Q6_K },
];

fn alloc_weight(device: &MlxDevice, shape: &ProbeShape) -> MlxBuffer {
    let blocks_per_row = (shape.k as u64) / (shape.qtype.block_values() as u64);
    let total_bytes = (shape.n as u64) * blocks_per_row * (shape.qtype.block_bytes() as u64);
    device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc weight")
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64) * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn params(shape: &ProbeShape) -> GgmlQuantizedMatmulParams {
    GgmlQuantizedMatmulParams {
        m: 1,
        n: shape.n,
        k: shape.k,
        ggml_type: shape.qtype,
    }
}

/// Pattern A: one CB, one encoder, N dispatches, one commit, one wait.
fn run_single_encoder(
    shape: &ProbeShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
) -> f64 {
    let p = params(shape);
    let t0 = std::time::Instant::now();
    let mut enc = device.command_encoder().expect("encoder");
    for _ in 0..N_DISPATCHES {
        quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, &p)
            .expect("dispatch A");
    }
    enc.commit_and_wait().expect("commit A");
    t0.elapsed().as_secs_f64() * 1_000_000.0
}

/// Pattern B: N/50 CBs, each a fresh encoder + commit; wait only on the last.
fn run_per_n_flush(
    shape: &ProbeShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
) -> f64 {
    let p = params(shape);
    let n_chunks = N_DISPATCHES.div_ceil(CANDLE_FLUSH);

    let t0 = std::time::Instant::now();
    let mut last: Option<mlx_native::CommandEncoder> = None;
    for chunk in 0..n_chunks {
        let start = chunk * CANDLE_FLUSH;
        let end = ((chunk + 1) * CANDLE_FLUSH).min(N_DISPATCHES);
        let mut enc = device.command_encoder().expect("encoder");
        for _ in start..end {
            quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, &p)
                .expect("dispatch B");
        }
        if chunk + 1 < n_chunks {
            enc.commit();
        } else {
            last = Some(enc);
        }
    }
    last.expect("last").commit_and_wait().expect("commit_and_wait B");
    t0.elapsed().as_secs_f64() * 1_000_000.0
}

/// Pattern C: N CBs, each with a fresh encoder + commit_and_wait. Worst case.
fn run_per_op_encoder(
    shape: &ProbeShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
) -> f64 {
    let p = params(shape);
    let t0 = std::time::Instant::now();
    for _ in 0..N_DISPATCHES {
        let mut enc = device.command_encoder().expect("encoder");
        quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, &p)
            .expect("dispatch C");
        enc.commit_and_wait().expect("commit C");
    }
    t0.elapsed().as_secs_f64() * 1_000_000.0
}

fn measure<F: FnMut() -> f64>(label: &str, mut f: F) -> (f64, f64) {
    for _ in 0..WARMUP {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        samples.push(f());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&samples, 0.5);
    let p10 = percentile(&samples, 0.1);
    let p90 = percentile(&samples, 0.9);
    let per_disp = p50 / N_DISPATCHES as f64;
    println!(
        "  {:<34} | total p50={:>8.2}us [p10={:>8.2}, p90={:>8.2}] | per-dispatch p50={:>6.3}us",
        label, p50, p10, p90, per_disp,
    );
    (p50, per_disp)
}

fn bench_shape(shape: &ProbeShape, device: &MlxDevice, registry: &mut KernelRegistry) {
    let input = alloc_f32(device, shape.k as usize, "input");
    let output = alloc_f32(device, shape.n as usize, "output");
    let weight = alloc_weight(device, shape);

    let bytes_read_mb = (shape.n as f64) * (shape.k as f64) * (shape.qtype.block_bytes() as f64)
        / (shape.qtype.block_values() as f64) / 1_048_576.0;
    println!("[{}] n={} k={} qtype={:?} | read={:.3} MB", shape.label, shape.n, shape.k, shape.qtype, bytes_read_mb);

    let (a_total, a_per) = measure("A. mlx-native single-encoder", || {
        run_single_encoder(shape, device, registry, &input, &weight, &output)
    });
    let (b_total, b_per) = measure(&format!("B. candle default ({}/CB flush)", CANDLE_FLUSH), || {
        run_per_n_flush(shape, device, registry, &input, &weight, &output)
    });
    let (c_total, c_per) = measure("C. per-op encoder (worst case)", || {
        run_per_op_encoder(shape, device, registry, &input, &weight, &output)
    });

    println!("  Ratios vs A: B/A={:.2}× wall, {:.2}×/disp | C/A={:.2}× wall, {:.2}×/disp",
        b_total / a_total, b_per / a_per, c_total / a_total, c_per / a_per);
    println!();
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!("README-quote bench: encoder pattern comparison");
    println!("  N per token: {} dispatches, Warmup: {}, Measure: {}", N_DISPATCHES, WARMUP, MEASURE);
    println!();

    for shape in SHAPES {
        bench_shape(shape, &device, &mut registry);
    }
}
