//! Large-prefill MoE routing-schedule reuse microbenchmark.
//!
//! This is the low-register-pressure follow-up to the falsified fused Q6_K
//! gate+up megakernel experiment. It keeps the existing ordinary `mm_id`
//! kernels and compares two same-command-buffer projection pairs:
//!
//! - independent: gate builds a routing schedule in scratch A, then up builds
//!   the identical schedule again in scratch B;
//! - prepared: gate builds the routing schedule once, then up reuses it from
//!   the same scratch.
//!
//! The benchmark uses separate gate/up weights and production family shapes:
//! Qwen3.6 Q5_K (`K=2048`, `N=512`, 256 experts, top-k 8) and DeepSeek-V4
//! Q2_K (`K=4096`, `N=2048`, 256 experts, top-k 6). It deliberately excludes
//! activation and down projection so the measured delta is attributable to
//! schedule construction, its global barrier, and newly available overlap
//! between the two existing projection kernels. The token ladder is
//! `{64, 256, 1024, 2048}`; tiny decode batches are a separate experiment.
//!
//! Run explicitly on a thermally controlled Apple Silicon host:
//!
//! ```text
//! MLX_RUN_MOE_SCHEDULE_BENCH=1 cargo test --release \
//!   --test moe_prepared_schedule_prefill_microbench -- --nocapture
//! ```

#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    quantized_matmul_id_ggml_pooled, quantized_matmul_id_ggml_pooled_pair, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
};
use std::time::Instant;

const WARMUP_PAIRS: usize = 3;
const SAMPLES: usize = 21;

#[derive(Clone, Copy)]
struct Shape {
    family: &'static str,
    quant: GgmlType,
    n_tokens: u32,
    top_k: u32,
    n_experts: u32,
    n: u32,
    k: u32,
}

fn alloc_u8(device: &MlxDevice, bytes: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(bytes, DType::U8, vec![bytes])
        .unwrap_or_else(|error| panic!("allocate {label} ({bytes} bytes): {error}"))
}

fn alloc_f32(device: &MlxDevice, elements: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .unwrap_or_else(|error| panic!("allocate {label} ({elements} f32): {error}"))
}

fn alloc_ids(device: &MlxDevice, shape: Shape) -> MlxBuffer {
    let rows = (shape.n_tokens * shape.top_k) as usize;
    let mut ids = device
        .alloc_buffer(
            rows * 4,
            DType::U32,
            vec![shape.n_tokens as usize, shape.top_k as usize],
        )
        .expect("allocate routing ids");
    for (index, expert) in ids
        .as_mut_slice::<u32>()
        .expect("map routing ids")
        .iter_mut()
        .enumerate()
    {
        let token = index / shape.top_k as usize;
        let slot = index % shape.top_k as usize;
        // The odd stride distributes adjacent tokens while `slot < top_k`
        // guarantees unique experts inside each token.
        *expert = ((token * 17 + slot) % shape.n_experts as usize) as u32;
    }
    ids
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    sorted[(sorted.len() - 1) * numerator / denominator]
}

fn encode_independent_pair(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    gate_weight: &MlxBuffer,
    up_weight: &MlxBuffer,
    ids: &MlxBuffer,
    gate_scratch: &mut IdMmScratch,
    up_scratch: &mut IdMmScratch,
    gate_output: &MlxBuffer,
    up_output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> f64 {
    let started = Instant::now();
    let mut encoder = device.command_encoder().expect("independent encoder");
    quantized_matmul_id_ggml_pooled(
        &mut encoder,
        registry,
        device,
        input,
        gate_weight,
        ids,
        gate_output,
        gate_scratch,
        params,
    )
    .expect("independent gate dispatch");
    quantized_matmul_id_ggml_pooled(
        &mut encoder,
        registry,
        device,
        input,
        up_weight,
        ids,
        up_output,
        up_scratch,
        params,
    )
    .expect("independent up dispatch");
    encoder
        .commit_and_wait()
        .expect("independent pair completion");
    started.elapsed().as_secs_f64() * 1_000.0
}

fn encode_prepared_pair(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    gate_weight: &MlxBuffer,
    up_weight: &MlxBuffer,
    ids: &MlxBuffer,
    scratch: &mut IdMmScratch,
    gate_output: &MlxBuffer,
    up_output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> f64 {
    let started = Instant::now();
    let mut encoder = device.command_encoder().expect("prepared encoder");
    quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        registry,
        device,
        input,
        gate_weight,
        up_weight,
        ids,
        gate_output,
        up_output,
        scratch,
        params,
    )
    .expect("prepared up dispatch");
    encoder.commit_and_wait().expect("prepared pair completion");
    started.elapsed().as_secs_f64() * 1_000.0
}

fn bench_shape(device: &MlxDevice, registry: &mut KernelRegistry, shape: Shape) {
    let block_values = shape.quant.block_values() as usize;
    let block_bytes = shape.quant.block_bytes() as usize;
    assert_eq!(shape.k as usize % block_values, 0);
    let expert_stride = shape.n as usize * (shape.k as usize / block_values) * block_bytes;
    let weight_bytes = shape.n_experts as usize * expert_stride;
    let routed_rows = (shape.n_tokens * shape.top_k) as usize;

    let input = alloc_f32(
        device,
        shape.n_tokens as usize * shape.k as usize,
        "shared expert input",
    );
    let gate_weight = alloc_u8(device, weight_bytes, "gate expert weights");
    let up_weight = alloc_u8(device, weight_bytes, "up expert weights");
    let ids = alloc_ids(device, shape);
    let gate_output = alloc_f32(device, routed_rows * shape.n as usize, "gate output");
    let up_output = alloc_f32(device, routed_rows * shape.n as usize, "up output");

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: shape.n_tokens,
        top_k: shape.top_k,
        n: shape.n,
        k: shape.k,
        n_experts: shape.n_experts,
        expert_stride: expert_stride as u64,
        ggml_type: shape.quant,
    };
    let mut gate_scratch =
        IdMmScratch::alloc(device, shape.n_experts, shape.n_tokens).expect("gate scratch");
    let mut up_scratch =
        IdMmScratch::alloc(device, shape.n_experts, shape.n_tokens).expect("up scratch");
    let mut prepared_scratch =
        IdMmScratch::alloc(device, shape.n_experts, shape.n_tokens).expect("prepared scratch");

    for _ in 0..WARMUP_PAIRS {
        encode_independent_pair(
            device,
            registry,
            &input,
            &gate_weight,
            &up_weight,
            &ids,
            &mut gate_scratch,
            &mut up_scratch,
            &gate_output,
            &up_output,
            &params,
        );
        encode_prepared_pair(
            device,
            registry,
            &input,
            &gate_weight,
            &up_weight,
            &ids,
            &mut prepared_scratch,
            &gate_output,
            &up_output,
            &params,
        );
    }

    let mut independent = Vec::with_capacity(SAMPLES);
    let mut prepared = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        // Alternate order so slow thermal drift cannot systematically favor
        // either composition.
        if sample % 2 == 0 {
            independent.push(encode_independent_pair(
                device,
                registry,
                &input,
                &gate_weight,
                &up_weight,
                &ids,
                &mut gate_scratch,
                &mut up_scratch,
                &gate_output,
                &up_output,
                &params,
            ));
            prepared.push(encode_prepared_pair(
                device,
                registry,
                &input,
                &gate_weight,
                &up_weight,
                &ids,
                &mut prepared_scratch,
                &gate_output,
                &up_output,
                &params,
            ));
        } else {
            prepared.push(encode_prepared_pair(
                device,
                registry,
                &input,
                &gate_weight,
                &up_weight,
                &ids,
                &mut prepared_scratch,
                &gate_output,
                &up_output,
                &params,
            ));
            independent.push(encode_independent_pair(
                device,
                registry,
                &input,
                &gate_weight,
                &up_weight,
                &ids,
                &mut gate_scratch,
                &mut up_scratch,
                &gate_output,
                &up_output,
                &params,
            ));
        }
    }
    independent.sort_by(f64::total_cmp);
    prepared.sort_by(f64::total_cmp);

    let independent_median = percentile(&independent, 1, 2);
    let prepared_median = percentile(&prepared, 1, 2);
    println!(
        "moe_schedule family={} quant={:?} tokens={} top_k={} experts={} N={} K={} weight_pair_mib={:.1} independent_ms[p10={:.3},median={:.3},p90={:.3}] prepared_ms[p10={:.3},median={:.3},p90={:.3}] speedup={:.4}x",
        shape.family,
        shape.quant,
        shape.n_tokens,
        shape.top_k,
        shape.n_experts,
        shape.n,
        shape.k,
        (2 * weight_bytes) as f64 / (1024.0 * 1024.0),
        percentile(&independent, 1, 10),
        independent_median,
        percentile(&independent, 9, 10),
        percentile(&prepared, 1, 10),
        prepared_median,
        percentile(&prepared, 9, 10),
        independent_median / prepared_median,
    );
}

#[test]
fn large_prefill_reuses_one_moe_routing_schedule() {
    if std::env::var("MLX_RUN_MOE_SCHEDULE_BENCH").ok().as_deref() != Some("1") {
        eprintln!("moe prepared-schedule benchmark gated; set MLX_RUN_MOE_SCHEDULE_BENCH=1");
        return;
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for shape in [
        Shape {
            family: "qwen35moe",
            quant: GgmlType::Q5_K,
            n_tokens: 64,
            top_k: 8,
            n_experts: 256,
            n: 512,
            k: 2048,
        },
        Shape {
            family: "qwen35moe",
            quant: GgmlType::Q5_K,
            n_tokens: 256,
            top_k: 8,
            n_experts: 256,
            n: 512,
            k: 2048,
        },
        Shape {
            family: "qwen35moe",
            quant: GgmlType::Q5_K,
            n_tokens: 1024,
            top_k: 8,
            n_experts: 256,
            n: 512,
            k: 2048,
        },
        Shape {
            family: "qwen35moe",
            quant: GgmlType::Q5_K,
            n_tokens: 2048,
            top_k: 8,
            n_experts: 256,
            n: 512,
            k: 2048,
        },
        Shape {
            family: "deepseek4",
            quant: GgmlType::Q2_K,
            n_tokens: 64,
            top_k: 6,
            n_experts: 256,
            n: 2048,
            k: 4096,
        },
        Shape {
            family: "deepseek4",
            quant: GgmlType::Q2_K,
            n_tokens: 256,
            top_k: 6,
            n_experts: 256,
            n: 2048,
            k: 4096,
        },
        Shape {
            family: "deepseek4",
            quant: GgmlType::Q2_K,
            n_tokens: 1024,
            top_k: 6,
            n_experts: 256,
            n: 2048,
            k: 4096,
        },
        Shape {
            family: "deepseek4",
            quant: GgmlType::Q2_K,
            n_tokens: 2048,
            top_k: 6,
            n_experts: 256,
            n: 2048,
            k: 4096,
        },
    ] {
        bench_shape(&device, &mut registry, shape);
    }
}
