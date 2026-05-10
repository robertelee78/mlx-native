//! ADR-028 iter-181: per-shape MoE _id mat-vec throughput at decode (M=1).
//!
//! Hypothesis under test: After iter-180 refuted "kernel inefficiency"
//! for dense Q5_K mat-vec (71% of M5 Max peak), the next testable
//! hypothesis is that MoE _id matmul (sparse expert access pattern)
//! runs at much lower bandwidth efficiency, accounting for the
//! end-to-end decode-time gap.
//!
//! Shapes + qtypes derived from gemma4 + qwen3.6 GGUF tensor metadata
//! (read directly via gguf-py at iter-181 — file label "APEX-Q5_K_M" is
//! misleading; internal qtypes are mixed):
//!
//!   gemma4 26B-A4B APEX-Q5_K_M:
//!     ffn_gate_up_exps  Q6_K  shape [k=2816, n=1408, 128 experts]
//!     ffn_down_exps     Q8_0  shape [k=704, n=2816, 128 experts]
//!     gate_up: n_tokens=1, top_k=8, n=1408, k=2816, n_experts=128
//!     down:    n_tokens=8, top_k=1, n=2816, k=704,  n_experts=128
//!     30 layers, all run BOTH dense MLP AND MoE.
//!
//!   qwen3.6 35B-A3B APEX-Q5_K_M (qwen35moe arch):
//!     ffn_gate_exps  Q5_K  shape [k=2048, n=512, 256 experts] (NOT fused)
//!     ffn_up_exps    Q5_K  shape [k=2048, n=512, 256 experts]
//!     ffn_down_exps  Q6_K  shape [k=512,  n=2048, 256 experts]
//!     Plus shared-experts (shexp): Q5_K / Q5_K / Q6_K (1 per layer).
//!     gate:  n_tokens=1, top_k=8, n=512, k=2048, n_experts=256
//!     up:    n_tokens=1, top_k=8, n=512, k=2048, n_experts=256
//!     down:  n_tokens=8, top_k=1, n=2048, k=512, n_experts=256
//!     40 layers, only 10 are full-FA (full_attention_interval=4).
//!
//! Falsification:
//!   - If MoE _id reaches >65% peak (similar to dense), MoE bandwidth
//!     is saturated → optimization gap is in dispatch/scheduling.
//!   - If <40% peak, MoE _id sparse access is the bottleneck → port
//!     llama.cpp's expert tile reuse or build a fused router+gate+up.
//!
//! Run:
//!   cargo bench -p mlx-native --bench bench_decode_moe_id_shapes --release

use mlx_native::ops::quantized_matmul_id_ggml::{
    quantized_matmul_id_ggml, GgmlQuantizedMatmulIdParams,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct MoeShape {
    label: &'static str,
    n_tokens: u32,
    top_k: u32,
    n: u32,
    k: u32,
    n_experts: u32,
    qtype: GgmlType,
    /// dispatches per layer (1 for gate_up + 1 for down = 2)
    per_token: usize,
}

const SHAPES: &[MoeShape] = &[
    // gemma4 (30 layers): real qtypes from GGUF (Q6_K + Q8_0).
    MoeShape { label: "g4_gate_up_Q6K", n_tokens: 1, top_k: 8, n: 1408, k: 2816, n_experts: 128, qtype: GgmlType::Q6_K, per_token: 30 },
    MoeShape { label: "g4_down_Q8_0",   n_tokens: 8, top_k: 1, n: 2816, k:  704, n_experts: 128, qtype: GgmlType::Q8_0, per_token: 30 },
    // qwen3.6 (40 layers, separate gate + up): real qtypes from GGUF.
    MoeShape { label: "q36_gate_Q5K", n_tokens: 1, top_k: 8, n:  512, k: 2048, n_experts: 256, qtype: GgmlType::Q5_K, per_token: 40 },
    MoeShape { label: "q36_up_Q5K",   n_tokens: 1, top_k: 8, n:  512, k: 2048, n_experts: 256, qtype: GgmlType::Q5_K, per_token: 40 },
    MoeShape { label: "q36_down_Q6K", n_tokens: 8, top_k: 1, n: 2048, k:  512, n_experts: 256, qtype: GgmlType::Q6_K, per_token: 40 },
];

const WARMUP: usize = 5;
const MEASURE: usize = 50;
const BATCH: usize = 32;
const M5_MAX_PEAK_GB_S: f64 = 546.0;
const M5_MAX_SUSTAINED_GB_S: f64 = 400.0;

fn alloc_weight_stack(
    device: &MlxDevice,
    n_experts: u32,
    n: u32,
    k: u32,
    qt: GgmlType,
) -> (MlxBuffer, u64, u64) {
    let blocks_per_row = (k as u64) / (qt.block_values() as u64);
    let per_expert_bytes = (n as u64) * blocks_per_row * (qt.block_bytes() as u64);
    let total_bytes = per_expert_bytes * (n_experts as u64);
    let buf = device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc weight stack");
    (buf, per_expert_bytes, total_bytes)
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn alloc_ids(device: &MlxDevice, n_tokens: u32, top_k: u32, n_experts: u32) -> MlxBuffer {
    let count = (n_tokens as usize) * (top_k as usize);
    let bytes = count * std::mem::size_of::<u32>();
    let mut buf = device
        .alloc_buffer(bytes, DType::U32, vec![count])
        .expect("alloc ids");
    let slice: &mut [u32] = buf.as_mut_slice().expect("ids slice");
    // Spread across experts so cache footprint matches realistic routing.
    for (i, v) in slice.iter_mut().enumerate() {
        *v = (i as u32 * 7919) % n_experts;
    }
    buf
}

fn bench_one(
    case: &MoeShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> (f64, f64, u64) {
    let input = alloc_f32(
        device,
        (case.n_tokens as usize) * (case.k as usize),
        "input",
    );
    let mut output = alloc_f32(
        device,
        (case.n_tokens as usize) * (case.top_k as usize) * (case.n as usize),
        "output",
    );
    let (weight, per_expert_bytes, _stack_bytes) =
        alloc_weight_stack(device, case.n_experts, case.n, case.k, case.qtype);
    let ids = alloc_ids(device, case.n_tokens, case.top_k, case.n_experts);

    // Bytes actually READ per call: top_k expert slices touched once each.
    // (The other n_experts - top_k slices stay cold in DRAM.)
    let bytes_read_per_call =
        per_expert_bytes * (case.n_tokens as u64) * (case.top_k as u64);

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: case.n_tokens,
        top_k: case.top_k,
        n: case.n,
        k: case.k,
        n_experts: case.n_experts,
        expert_stride: per_expert_bytes,
        ggml_type: case.qtype,
    };

    // Warmup.
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml(
            &mut enc, registry, device, &input, &weight, &ids, &mut output, &params,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    // Single-sync.
    let mut single = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        quantized_matmul_id_ggml(
            &mut enc, registry, device, &input, &weight, &ids, &mut output, &params,
        )
        .expect("single dispatch");
        enc.commit_and_wait().expect("single commit");
        let us = t0.elapsed().as_secs_f64() * 1.0e6;
        single.push(us);
    }
    single.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let single_median = single[single.len() / 2];

    // Batched (production-relevant).
    let mut batched = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            quantized_matmul_id_ggml(
                &mut enc, registry, device, &input, &weight, &ids, &mut output, &params,
            )
            .expect("batched dispatch");
        }
        enc.commit_and_wait().expect("batched commit");
        let us = t0.elapsed().as_secs_f64() * 1.0e6 / BATCH as f64;
        batched.push(us);
    }
    batched.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let batched_median = batched[batched.len() / 2];

    eprintln!(
        "    {:<14}  single_sync={:>6.1}us  batched={:>6.1}us",
        case.label, single_median, batched_median,
    );
    (batched_median, single_median, bytes_read_per_call)
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!(
        "Decode MoE _id mat-vec throughput at gemma4 + qwen3.6 shapes\n\
         M5 Max peak {:.0} GB/s | sustained-target {:.0} GB/s\n\
         Bytes-per-call = top_k experts × n_tokens × per_expert_bytes\n",
        M5_MAX_PEAK_GB_S, M5_MAX_SUSTAINED_GB_S
    );
    println!(
        "{:<14} {:>4} {:>4} {:>5} {:>5} {:>4} {:>9} {:>9} {:>9} {:>8} {:>9}",
        "shape", "tk", "tok", "n", "k", "Ne", "us_batch", "MB_read", "GB/s", "%peak", "%sustain"
    );
    println!("{}", "-".repeat(112));

    let mut total_g4_us = 0.0;
    let mut total_g4_bytes: u64 = 0;
    let mut total_q36_us = 0.0;
    let mut total_q36_bytes: u64 = 0;

    for case in SHAPES {
        let (batched_us, _single, bytes_per_call) = bench_one(case, &device, &mut registry);
        let mb = bytes_per_call as f64 / 1.0e6;
        let gb_per_s = (bytes_per_call as f64) / (batched_us / 1.0e6) / 1.0e9;
        let pct_peak = 100.0 * gb_per_s / M5_MAX_PEAK_GB_S;
        let pct_sustain = 100.0 * gb_per_s / M5_MAX_SUSTAINED_GB_S;
        println!(
            "{:<14} {:>4} {:>4} {:>5} {:>5} {:>4} {:>9.1} {:>9.1} {:>9.1} {:>7.1}% {:>8.1}%",
            case.label, case.top_k, case.n_tokens, case.n, case.k, case.n_experts,
            batched_us, mb, gb_per_s, pct_peak, pct_sustain,
        );

        let total_us = batched_us * (case.per_token as f64);
        let total_bytes = bytes_per_call * (case.per_token as u64);
        if case.label.starts_with("g4_") {
            total_g4_us += total_us;
            total_g4_bytes += total_bytes;
        } else if case.label.starts_with("q36_") {
            total_q36_us += total_us;
            total_q36_bytes += total_bytes;
        }
    }

    println!("{}", "-".repeat(112));
    let g4_gb = total_g4_bytes as f64 / 1.0e9;
    let g4_ms = total_g4_us / 1000.0;
    let g4_gbs = g4_gb / (total_g4_us / 1.0e6);
    println!(
        "gemma4 (30 layers, 60 _id calls/token): {:.2} GB read in {:.2} ms (aggregate {:.0} GB/s)",
        g4_gb, g4_ms, g4_gbs,
    );

    let q36_gb = total_q36_bytes as f64 / 1.0e9;
    let q36_ms = total_q36_us / 1000.0;
    let q36_gbs = q36_gb / (total_q36_us / 1.0e6);
    println!(
        "qwen3.6 (40 layers, 80 _id calls/token): {:.2} GB read in {:.2} ms (aggregate {:.0} GB/s)",
        q36_gb, q36_ms, q36_gbs,
    );
    println!();
    println!(
        "Per-token MoE-only ceilings:\n\
         gemma4:  {:.0} tok/s\n\
         qwen3.6: {:.0} tok/s",
        1000.0 / g4_ms, 1000.0 / q36_ms,
    );
}
