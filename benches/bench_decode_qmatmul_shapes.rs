//! ADR-028 iter-180: per-shape mat-VEC throughput at gemma4 decode (M=1).
//!
//! Hypothesis under test (from iter-179):
//!   "22pp end-to-end efficiency gap is mat-mul kernel efficiency, NOT structural."
//!
//! Falsification plan:
//!   - Run each gemma4 26B-A4B-APEX-Q5_K_M decode-time projection shape
//!     through `quantized_matmul_ggml` at M=1.
//!   - Compute bytes-read-per-call (weight bytes; activations are tiny at M=1)
//!     and effective GB/s.
//!   - Compare to M5 Max peak (546 GB/s).
//!   - If our kernels run at >70% of peak, hypothesis REFUTED — gap is
//!     elsewhere (dispatch/scheduling/encoder overhead).
//!   - If at <50%, kernel-impl is the bottleneck → optimization target.
//!
//! Shapes derived from the gemma4 APEX-Q5_K_M load banner:
//!   30 layers, 16 heads (8 kv), head_dim=256, hidden=2816
//!   sliding=24 layers (every-6 = 5 global), moe 128 experts/8 active
//!   moe_intermediate=2112 (gemma4 default; verified vs hf2q config.rs)
//!
//! Tensor types per the APEX gguf (Q5_K dominant per `hf2q load:` banner):
//!   QKV/O proj:   Q5_K
//!   MoE gate_up:  Q5_K (fused 2x intermediate)
//!   MoE down:     Q5_K
//!   Router:       Q5_K (small, n=128)
//!
//! Run:
//!   cargo bench -p mlx-native --bench bench_decode_qmatmul_shapes --release

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// One mat-vec shape at decode (M=1).
struct DecodeShape {
    label: &'static str,
    n: u32,
    k: u32,
    qtype: GgmlType,
    /// Number of times this shape fires per decoded token.
    per_token: usize,
}

/// Gemma4 26B-A4B APEX-Q5_K_M decode shapes.  hidden=2816, n_heads=16,
/// n_kv_heads=8 (sliding) / 2 (global), head_dim=256, moe_int=2112.
const SHAPES: &[DecodeShape] = &[
    // Attention projections (single-token; reads the whole weight matrix).
    // Sliding layers (24 of 30): n_kv_heads=8.
    DecodeShape { label: "Q_sliding",  n: 4096, k: 2816, qtype: GgmlType::Q5_K, per_token: 24 },
    DecodeShape { label: "K_sliding",  n: 2048, k: 2816, qtype: GgmlType::Q5_K, per_token: 24 },
    DecodeShape { label: "V_sliding",  n: 2048, k: 2816, qtype: GgmlType::Q5_K, per_token: 24 },
    DecodeShape { label: "O_sliding",  n: 2816, k: 4096, qtype: GgmlType::Q5_K, per_token: 24 },
    // Global layers (6 of 30): same head shapes as sliding for gemma4
    // (per llama.cpp gemma4.cpp:208 — both use n_head_kv but proj
    // weights live at the same n=2816→4096/2048 dims).  Re-bench at
    // the same shape to confirm Q5_K behaviour.
    DecodeShape { label: "Q_global",   n: 4096, k: 2816, qtype: GgmlType::Q5_K, per_token: 6 },
    DecodeShape { label: "K_global",   n: 2048, k: 2816, qtype: GgmlType::Q5_K, per_token: 6 },
    DecodeShape { label: "V_global",   n: 2048, k: 2816, qtype: GgmlType::Q5_K, per_token: 6 },
    DecodeShape { label: "O_global",   n: 2816, k: 4096, qtype: GgmlType::Q5_K, per_token: 6 },
    // Router (1 per layer, all 30): n=128.
    DecodeShape { label: "Router",     n:  128, k: 2816, qtype: GgmlType::Q5_K, per_token: 30 },

    // ADR-028 iter-187: gemma4 lm_head (token_embd tied) — vocab=262144, hidden=2816.
    // Currently re-quantized at load to Q8_0 (~784 MB).  GGUF stores Q6_K (~605 MB).
    // Measure both qtypes to size the "Q6_K direct" lever vs current Q8_0 path.
    DecodeShape { label: "lmhead_Q6_K", n: 262144, k: 2816, qtype: GgmlType::Q6_K, per_token: 1 },
    DecodeShape { label: "lmhead_Q8_0", n: 262144, k: 2816, qtype: GgmlType::Q8_0, per_token: 1 },
];

const WARMUP: usize = 5;
const MEASURE: usize = 50;

/// M5 Max measured peak unified-memory bandwidth (Apple WWDC '23 benches +
/// our own measurements).  Theoretical peak is ~546 GB/s; sustained is
/// closer to 400 GB/s for memory-bound kernels.  Use the theoretical for
/// the headline ratio so the threshold is conservative.
const M5_MAX_PEAK_GB_S: f64 = 546.0;
const M5_MAX_SUSTAINED_GB_S: f64 = 400.0;

fn alloc_weight(device: &MlxDevice, n: u32, k: u32, qt: GgmlType) -> (MlxBuffer, u64) {
    let blocks_per_row = (k as u64) / (qt.block_values() as u64);
    let total_bytes = (n as u64) * blocks_per_row * (qt.block_bytes() as u64);
    let buf = device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc weight");
    (buf, total_bytes)
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

/// Returns `(per_call_us_batched, weight_bytes)`.  Batched timing
/// amortizes per-CB sync (~80-100 µs on M5 Max) so the result reflects
/// per-call kernel time only — the production-relevant number.
fn bench_one(
    case: &DecodeShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> (f64, u64) {
    let m: u32 = 1;
    let input = alloc_f32(device, (m as usize) * (case.k as usize), "input");
    let mut output = alloc_f32(device, (m as usize) * (case.n as usize), "output");
    let (weight, weight_bytes) = alloc_weight(device, case.n, case.k, case.qtype);

    let params = GgmlQuantizedMatmulParams {
        m,
        n: case.n,
        k: case.k,
        ggml_type: case.qtype,
    };

    // Warm-up: triggers the on-demand pipeline compile for this shape.
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        quantized_matmul_ggml(
            &mut enc, registry, device, &input, &weight, &mut output, &params,
        )
        .expect("dispatch warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    // Two measurements:
    //   1) per-iter sync (one dispatch per CB) — captures launch+sync cost.
    //   2) batched (BATCH dispatches per CB) — amortizes sync, isolates
    //      per-call kernel time (closer to production where the engine
    //      submits ~240 dispatches in a single CB per token).
    //
    // Returning the BATCHED value is the meaningful number for the
    // kernel-efficiency hypothesis.  The unbatched result is reported
    // separately for context.
    const BATCH: usize = 32;

    let mut single_samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        quantized_matmul_ggml(
            &mut enc, registry, device, &input, &weight, &mut output, &params,
        )
        .expect("dispatch measure");
        enc.commit_and_wait().expect("measure commit");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        single_samples.push(elapsed_us);
    }
    single_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let single_median = single_samples[single_samples.len() / 2];

    let mut batched_per_call = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            quantized_matmul_ggml(
                &mut enc, registry, device, &input, &weight, &mut output, &params,
            )
            .expect("dispatch measure");
        }
        enc.commit_and_wait().expect("measure commit");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        batched_per_call.push(elapsed_us / BATCH as f64);
    }
    batched_per_call.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let batched_median = batched_per_call[batched_per_call.len() / 2];

    // Print both side-by-side; return BATCHED for production-relevant
    // throughput aggregation.
    eprintln!(
        "    {:<14}  single_sync={:>6.1}us  batched_per_call={:>6.1}us  delta={:>+6.1}us",
        case.label, single_median, batched_median, single_median - batched_median,
    );
    (batched_median, weight_bytes)
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!(
        "Decode mat-vec throughput at gemma4 26B-A4B APEX-Q5_K_M shapes (M=1)\n\
         M5 Max peak: {:.0} GB/s | sustained-target: {:.0} GB/s\n",
        M5_MAX_PEAK_GB_S, M5_MAX_SUSTAINED_GB_S
    );
    println!(
        "{:<14} {:>5} {:>5} {:>6} {:>9} {:>9} {:>10} {:>9} {:>10}",
        "shape", "N", "K", "qtype", "median_us", "MB_read", "GB/s", "%peak", "%sustain"
    );
    println!("{}", "-".repeat(96));

    let mut total_bytes: u64 = 0;
    let mut total_us: f64 = 0.0;

    for case in SHAPES {
        let (median_us, weight_bytes) = bench_one(case, &device, &mut registry);
        let mb = weight_bytes as f64 / 1.0e6;
        let gb_per_s = (weight_bytes as f64) / (median_us / 1.0e6) / 1.0e9;
        let pct_peak = 100.0 * gb_per_s / M5_MAX_PEAK_GB_S;
        let pct_sustain = 100.0 * gb_per_s / M5_MAX_SUSTAINED_GB_S;
        println!(
            "{:<14} {:>5} {:>5} {:>6?} {:>9.1} {:>9.1} {:>10.1} {:>8.1}% {:>9.1}%",
            case.label, case.n, case.k, case.qtype,
            median_us, mb, gb_per_s, pct_peak, pct_sustain,
        );
        total_bytes += weight_bytes * (case.per_token as u64);
        total_us += median_us * (case.per_token as f64);
    }

    println!("{}", "-".repeat(96));
    let total_gb = total_bytes as f64 / 1.0e9;
    let total_ms = total_us / 1000.0;
    let aggregate_gb_s = total_gb / (total_us / 1.0e6);
    println!(
        "Per-token attention+router weight reads: {:.2} GB in {:.2} ms ({:.1} GB/s aggregate)",
        total_gb, total_ms, aggregate_gb_s,
    );
    println!(
        "Implies decode throughput ceiling from these kernels alone: {:.1} tok/s\n",
        1000.0 / total_ms,
    );
    println!(
        "Note: this bench EXCLUDES the MoE expert matmuls (top-k=8 dynamic\n\
         routing) which dominate gemma4 decode time per the kernel profiler.\n\
         Use bench_moe_q_qwen36_shape (or its gemma4 sibling) to measure MoE.\n"
    );
}
