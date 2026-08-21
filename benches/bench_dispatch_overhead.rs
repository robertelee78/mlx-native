//! ADR-028 iter-254 — per-dispatch CPU encode overhead (H2 isolation test).
//!
//! Hypothesis under test (H2):
//!   "hf2q's per-dispatch CPU overhead (Rust→Metal binding) is a large
//!    fraction of the 14.7 µs/dispatch decode-time average — and a
//!    measurable fraction of the gap to the peer engine (which does
//!    11.4 µs/dispatch via thinner C-Metal binding)."
//!
//! Falsification plan:
//!   - Encode N=200 dispatches into a single command buffer.
//!   - Time JUST the encoding loop (no commit, no wait) → CPU-only.
//!   - Then commit_and_wait → total (CPU + GPU + driver sync).
//!   - GPU+sync amortized per dispatch = (total - CPU) / N.
//!
//! Two shapes contrast minimal vs maximum GPU work at the SAME binding
//! count (8 buffer/byte slots in `quantized_matmul_ggml`):
//!   - Router (n=128, k=2816, Q5_K) — ~0.045 MB read; tiny GPU work.
//!   - lmhead Q6_K (n=262144, k=2816, Q6_K) — ~605 MB read; huge GPU work.
//!
//! If H2 is correct (CPU encoding is the bottleneck), the per-dispatch
//! CPU time will be ~3 µs+ on BOTH shapes (since binding count + FFI
//! cost is identical).  If H2 is wrong, CPU time will be tiny (sub-µs)
//! and the gap is GPU-side.
//!
//! Run:
//!   cargo bench -p mlx-native --bench bench_dispatch_overhead

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct ProbeShape {
    label: &'static str,
    n: u32,
    k: u32,
    qtype: GgmlType,
}

const SHAPES: &[ProbeShape] = &[
    // Tiny GPU work (router shape from iter-180).
    ProbeShape { label: "Router_Q5K", n: 128, k: 2816, qtype: GgmlType::Q5_K },
    // Medium (Q-projection from iter-180).
    ProbeShape { label: "Q_sliding_Q5K", n: 4096, k: 2816, qtype: GgmlType::Q5_K },
    // Maximum GPU work (lm_head Q6_K, 605 MB read).
    ProbeShape { label: "lmhead_Q6_K", n: 262144, k: 2816, qtype: GgmlType::Q6_K },
];

const WARMUP: usize = 5;
const MEASURE: usize = 30;
const N_DISPATCHES: usize = 200;

fn alloc_weight(device: &MlxDevice, n: u32, k: u32, qt: GgmlType) -> MlxBuffer {
    let blocks_per_row = (k as u64) / (qt.block_values() as u64);
    let total_bytes = (n as u64) * blocks_per_row * (qt.block_bytes() as u64);
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

fn bench_one(case: &ProbeShape, device: &MlxDevice, registry: &mut KernelRegistry) {
    let m: u32 = 1;
    let input = alloc_f32(device, (m as usize) * (case.k as usize), "input");
    let mut output = alloc_f32(device, (m as usize) * (case.n as usize), "output");
    let weight = alloc_weight(device, case.n, case.k, case.qtype);

    let params = GgmlQuantizedMatmulParams {
        m,
        n: case.n,
        k: case.k,
        ggml_type: case.qtype,
    };

    // Warmup: triggers pipeline compile + caches.
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        for _ in 0..N_DISPATCHES {
            quantized_matmul_ggml(
                &mut enc, registry, device, &input, &weight, &mut output, &params,
            )
            .expect("dispatch warmup");
        }
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut cpu_per_disp = Vec::with_capacity(MEASURE);
    let mut total_per_disp = Vec::with_capacity(MEASURE);

    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");

        // PHASE 1: time CPU-only encoding loop.
        let t0 = std::time::Instant::now();
        for _ in 0..N_DISPATCHES {
            quantized_matmul_ggml(
                &mut enc, registry, device, &input, &weight, &mut output, &params,
            )
            .expect("dispatch measure");
        }
        let cpu_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        // PHASE 2: commit + wait, get total wall time.
        enc.commit_and_wait().expect("commit measure");
        let total_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        cpu_per_disp.push(cpu_us / N_DISPATCHES as f64);
        total_per_disp.push(total_us / N_DISPATCHES as f64);
    }

    cpu_per_disp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    total_per_disp.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cpu_p50 = percentile(&cpu_per_disp, 0.5);
    let cpu_p10 = percentile(&cpu_per_disp, 0.1);
    let cpu_p90 = percentile(&cpu_per_disp, 0.9);
    let tot_p50 = percentile(&total_per_disp, 0.5);
    let gpu_amortized = tot_p50 - cpu_p50;
    let cpu_pct = 100.0 * cpu_p50 / tot_p50;

    println!(
        "{:<15} | CPU-only p50={:>5.2}us [p10={:>5.2}, p90={:>5.2}] | total p50={:>6.2}us | GPU+sync amort={:>6.2}us | CPU%={:>5.1}%",
        case.label, cpu_p50, cpu_p10, cpu_p90, tot_p50, gpu_amortized, cpu_pct,
    );
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!("ADR-028 iter-254: per-dispatch CPU encoding overhead probe");
    println!("  N_DISPATCHES per CB: {}", N_DISPATCHES);
    println!("  Warmup: {}, Measure: {}", WARMUP, MEASURE);
    println!("  Method: time encode-loop (no commit), then commit_and_wait, diff.");
    println!();
    println!("  CPU-only = pure Rust→Metal binding cost per dispatch.");
    println!("  total = CPU encode + GPU compute + driver sync (amortized over {}).", N_DISPATCHES);
    println!();
    println!(
        "{:<15} | {:^46} | {:^15} | {:^25} | {}",
        "shape", "Rust→Metal CPU encode (per dispatch)", "total per disp", "GPU+sync amort/disp", "CPU%"
    );
    println!("{}", "-".repeat(140));

    for case in SHAPES {
        bench_one(case, &device, &mut registry);
    }

    println!();
    println!("Falsification verdict:");
    println!("  - If CPU-only > 2 µs/dispatch on Router shape → H2 confirmed (binding overhead matters).");
    println!("  - If CPU-only < 1 µs/dispatch on all shapes → H2 falsified (binding is fast; gap is GPU).");
    println!("  - GPU+sync amort scaling with shape size confirms bandwidth-bound kernels.");
}
