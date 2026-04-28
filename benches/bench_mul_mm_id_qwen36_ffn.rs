//! Microbenchmark: per-shape `mul_mm_id` MoE expert mat-mul at the exact
//! shapes hf2q dispatches during Qwen3.6 35B-A3B DWQ46 prefill at PP4106
//! on M5 Max.
//!
//! ADR-005 W-5b.23 cross-fence audit (2026-04-27): the W-5b.22
//! instrumentation reported 3,316 ms / 99.9% of the residual is in
//! `dn.outer_ffn_dispatch` for 48 DN-attn-layers at PP4106.  That bucket
//! covers the ENTIRE `build_moe_ffn_layer_gpu_q_into` body (router
//! projections + softmax_topk + 3 expert mm_id mat-muls + silu_mul +
//! shared-expert proj + weighted reduce + residual fold).  This bench
//! isolates the 3 dominant `quantized_matmul_id_ggml` calls (gate, up,
//! down) at production shape, sums them × 48 layers, and compares to the
//! 3,316 ms observed bucket.
//!
//! Decision tree (per W-5b.23 worker prompt):
//!   sum × 48 ≈ 3,316 ms  →  kernel exec IS the bucket; recommend ADR-015
//!                           W2b kernel-internal optimisation.
//!   sum × 48 ≪ 3,316 ms  →  delta is invocation-pattern (commit_and_wait,
//!                           barriers between sub-mat-muls); recommend
//!                           single-CB fusion in W-5b.24.
//!   sum × 48 ≫ 3,316 ms  →  bench harness is wrong or the W-5b.22 bucket
//!                           attribution missed something; STOP.
//!
//! Shape derivation (from /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-
//! abliterated-dwq46/ direct GGUF inspection, cross-referenced with
//! config.json):
//!     hidden_size              = 2048
//!     moe_intermediate_size    =  512
//!     num_experts              =  256
//!     num_experts_per_tok      =    8     (gate/up call)
//!                                  1     (down call, post-expand)
//!     num_hidden_layers        =   40     (W-5b.22 measures over 48 DN-
//!                                          layers — matches the qwen3.6
//!                                          per-doc count, which differs
//!                                          from config.json's 40; we keep
//!                                          48 so the comparison to the
//!                                          3,316 ms observed bucket is
//!                                          apples-to-apples)
//!     ffn_gate_exps   Q4_0  shape [256, 512, 2048]
//!     ffn_up_exps     Q4_0  shape [256, 512, 2048]
//!     ffn_down_exps   Q4_0  shape [256, 2048, 512]
//!
//! At PP4106 the dispatch routes through `dispatch_id_mm` (n_tokens > 8;
//! `MM_ID_ROUTING_THRESHOLD = 8`).  Decode (n_tokens=1) goes through
//! `dispatch_id_mv` instead; this bench targets the prefill measurement.
//!
//! Routing pattern: ids drawn uniformly from `[0, n_experts)`.  At top_k=8
//! the expected per-expert routed count is `n_tokens*top_k/n_experts` =
//! 4106*8/256 ≈ 128, well above the NR1=32 mm_id tile size, so essentially
//! every threadgroup in the (M-tile-block, N-tile, expert) grid runs.
//!
//! Each shape is warmed up `WARMUP_ITERS` times, then timed over
//! `MEASURE_ITERS` iterations via `command_encoder().commit_and_wait()`.
//! Median ms per call + effective TFLOP/s reported, then sum × 48 layers.
//!
//! Read-only audit: this bench does NOT modify production source.  Pure
//! additive criterion-style bench under /opt/mlx-native/benches/.
//!
//! Invoke:
//!   cargo bench --bench bench_mul_mm_id_qwen36_ffn --release

use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_for_test, GgmlIdMmDispatchParams, IdMmScratch,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct ShapeCase {
    label:     &'static str,
    n_tokens:  u32, // M
    top_k:     u32,
    n:         u32, // output cols per expert
    k:         u32, // input dim
    n_experts: u32,
    qtype:     GgmlType,
    layers:    usize, // total layers calling this shape
}

/// Qwen3.6 35B-A3B DWQ46 MoE prefill shapes at PP=4106 (W-5b.22 bench prompt).
///
/// All 3 calls per layer × 48 DN layers = 144 mm_id dispatches per prefill
/// in the W-5b.22 measurement window.
///
/// `down` is run at total_rows = n_tokens*top_k = 32848 with top_k=1, since
/// after the gate*up SwiGLU expansion the FFN gives one routed row per
/// (token, slot) pair (see /opt/hf2q/src/inference/models/qwen35/gpu_ffn.rs:1548-1560).
const SHAPES: &[ShapeCase] = &[
    ShapeCase {
        label: "FFN_gate", n_tokens: 4106, top_k: 8,
        n: 512, k: 2048, n_experts: 256,
        qtype: GgmlType::Q4_0, layers: 48,
    },
    ShapeCase {
        label: "FFN_up",   n_tokens: 4106, top_k: 8,
        n: 512, k: 2048, n_experts: 256,
        qtype: GgmlType::Q4_0, layers: 48,
    },
    ShapeCase {
        label: "FFN_down", n_tokens: 4106 * 8, top_k: 1,
        n: 2048, k: 512, n_experts: 256,
        qtype: GgmlType::Q4_0, layers: 48,
    },
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

fn alloc_weight_stacked(device: &MlxDevice, n_experts: u32, n: u32, k: u32, qt: GgmlType)
    -> MlxBuffer
{
    // Stacked weight: [n_experts, N, packed_K]
    let blocks_per_row = (k as u64) / qtype_block_values(qt);
    let per_expert_bytes = (n as u64) * blocks_per_row * qtype_block_bytes(qt);
    let total_bytes = per_expert_bytes * (n_experts as u64);
    device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc stacked weight")
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

/// Allocate ids buffer with uniform expert distribution across `[0, n_experts)`.
/// Layout: `[n_tokens * top_k]` u32 flat (i32-byte-equivalent).
fn alloc_ids_uniform(device: &MlxDevice, n_tokens: u32, top_k: u32, n_experts: u32) -> MlxBuffer {
    let total = (n_tokens * top_k) as usize;
    let mut buf = device
        .alloc_buffer(total * 4, DType::U32, vec![total])
        .expect("alloc ids");
    let slice: &mut [u32] = buf.as_mut_slice().expect("ids as_mut_slice");
    // Deterministic uniform-ish round-robin so each expert gets ~total/n_experts rows.
    // The mm_id kernel's perf is sensitive to per-expert routed counts (early-exit
    // threadgroups when neh1 < r1), so we want the realistic case where every
    // expert is hit by ~128 rows at PP4106 top_k=8 / n_experts=256.
    for (i, slot) in slice.iter_mut().enumerate() {
        *slot = (i as u32) % n_experts;
    }
    buf
}

fn alloc_id_mm_scratch(device: &MlxDevice, params: &GgmlIdMmDispatchParams) -> IdMmScratch {
    IdMmScratch::alloc(device, params.n_experts, params.n_tokens).expect("alloc IdMmScratch")
}

fn bench_one(case: &ShapeCase, device: &MlxDevice, registry: &mut KernelRegistry) -> f64 {
    let total_rows = (case.n_tokens as usize) * (case.top_k as usize);

    // Buffers
    let input = alloc_f32(device, (case.n_tokens as usize) * (case.k as usize), "input");
    let mut output = alloc_f32(device, total_rows * (case.n as usize), "output");
    let weight = alloc_weight_stacked(device, case.n_experts, case.n, case.k, case.qtype);
    let ids = alloc_ids_uniform(device, case.n_tokens, case.top_k, case.n_experts);

    // expert_stride matches the load-side layout exactly: per-expert byte count.
    let blocks_per_row = (case.k as u64) / qtype_block_values(case.qtype);
    let per_expert_bytes = (case.n as u64) * blocks_per_row * qtype_block_bytes(case.qtype);

    let params = GgmlIdMmDispatchParams {
        n_tokens:      case.n_tokens,
        top_k:         case.top_k,
        n:             case.n,
        k:             case.k,
        n_experts:     case.n_experts,
        expert_stride: per_expert_bytes,
        ggml_type:     case.qtype,
    };
    let mut scratch = alloc_id_mm_scratch(device, &params);

    // Warmup: triggers kernel pipeline compile on first call.
    for _ in 0..WARMUP_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        dispatch_id_mm_for_test(
            &mut enc, registry, device,
            &input, &weight, &ids,
            &mut scratch.htpe, &mut scratch.hids,
            &mut output, &params,
        ).expect("dispatch warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    // Measure: median of MEASURE_ITERS samples.
    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        dispatch_id_mm_for_test(
            &mut enc, registry, device,
            &input, &weight, &ids,
            &mut scratch.htpe, &mut scratch.hids,
            &mut output, &params,
        ).expect("dispatch measure");
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
        "{:<10} {:>9} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>11}",
        "shape", "n_tokens", "top_k", "N", "K",
        "qtype", "median_ms", "GFLOPs", "TFLOP/s", "layers", "prefill_ms",
    );
    println!("{}", "-".repeat(120));

    let mut total_ffn_prefill_ms = 0.0;
    for case in SHAPES {
        let median = bench_one(case, &device, &mut registry);
        // FLOPs: 2 * total_rows * N * K (one mat-vec per routed row, since
        // mm_id is per-expert dense within the routed subset).
        let total_rows = (case.n_tokens as f64) * (case.top_k as f64);
        let flops = 2.0 * total_rows * (case.n as f64) * (case.k as f64);
        let gflops = flops / 1e9;
        let tflops = flops / (median / 1000.0) / 1e12;
        let per_prefill = median * (case.layers as f64);
        total_ffn_prefill_ms += per_prefill;
        println!(
            "{:<10} {:>9} {:>6} {:>6} {:>6} {:>10?} {:>10.3} {:>10.1} {:>9.2} {:>9} {:>11.1}",
            case.label, case.n_tokens, case.top_k, case.n, case.k,
            case.qtype, median, gflops, tflops, case.layers, per_prefill,
        );
    }
    println!("{}", "-".repeat(120));
    println!(
        "Total mm_id MoE FFN kernel exec time at PP4106 (Qwen3.6 35B-A3B DWQ46, 48 DN layers × 3 calls): {:.1} ms",
        total_ffn_prefill_ms,
    );
    println!(
        "\nCompare to W-5b.22 measurement (docs/wave5b3-walkbar-results.md):\n\
         dn.outer_ffn_dispatch (48 DN layers, full MoeQ FFN body) = 3,316 ms\n\
         \n\
         If kernel_sum ≈ 3,316 ms : kernel exec IS the bucket; W-5b.24 = ADR-015 W2b kernel work.\n\
         If kernel_sum ≪ 3,316 ms : delta is invocation-pattern (commit_and_wait, router proj,\n\
                                    softmax_topk, silu_mul, weighted_reduce); W-5b.24 = invocation\n\
                                    fusion in hf2q (similar to W-5b.15 / W-5b.20 pattern).\n\
         If kernel_sum ≫ 3,316 ms : bench harness wrong or W-5b.22 bucket attribution missed\n\
                                    something; STOP and re-instrument."
    );
}
