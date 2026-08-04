//! Q2_K decode, dense-prefill, and expert-prefill Metal benchmarks.

use mlx_native::{
    ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml_mv, quantized_matmul_ggml,
    quantized_matmul_id_ggml_pooled, DType, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams,
    GgmlType, IdMmScratch, KernelRegistry, MlxDevice,
};

const N: u32 = 4096;
const K: u32 = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const DISPATCHES_PER_SAMPLE: usize = 16;

fn bench_dense(device: &MlxDevice, registry: &mut KernelRegistry, m: u32, label: &str) {
    let block_values = GgmlType::Q2_K.block_values() as usize;
    let block_bytes = GgmlType::Q2_K.block_bytes() as usize;
    let weight_bytes = N as usize * (K as usize / block_values) * block_bytes;
    let input = device
        .alloc_buffer(
            m as usize * K as usize * 4,
            DType::F32,
            vec![m as usize, K as usize],
        )
        .expect("input");
    let weight = device
        .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
        .expect("weight");
    let output = device
        .alloc_buffer(
            m as usize * N as usize * 4,
            DType::F32,
            vec![m as usize, N as usize],
        )
        .expect("output");
    let params = GgmlQuantizedMatmulParams {
        m,
        n: N,
        k: K,
        ggml_type: GgmlType::Q2_K,
    };

    for _ in 0..WARMUP {
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_ggml(
            &mut encoder,
            registry,
            &device,
            &input,
            &weight,
            &output,
            &params,
        )
        .expect("warmup dispatch");
        encoder.commit_and_wait().expect("warmup completion");
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut encoder = device.command_encoder().expect("encoder");
        let started = std::time::Instant::now();
        for _ in 0..DISPATCHES_PER_SAMPLE {
            quantized_matmul_ggml(
                &mut encoder,
                registry,
                &device,
                &input,
                &weight,
                &output,
                &params,
            )
            .expect("measured dispatch");
        }
        encoder.commit_and_wait().expect("measured completion");
        samples.push(started.elapsed().as_secs_f64() * 1e6 / DISPATCHES_PER_SAMPLE as f64);
    }
    samples.sort_by(f64::total_cmp);
    let median_us = samples[samples.len() / 2];
    let p10_us = samples[samples.len() / 10];
    let p90_us = samples[samples.len() * 9 / 10];
    let gb_per_s = weight_bytes as f64 / (median_us * 1e-6) / 1e9;
    println!(
        "Q2_K dense {label} M={m} N={N} K={K}: median={median_us:.2} us, p10={p10_us:.2} us, p90={p90_us:.2} us, weight={:.2} MB, effective={gb_per_s:.2} GB/s",
        weight_bytes as f64 / 1e6,
    );
}

fn bench_expert(device: &MlxDevice, registry: &mut KernelRegistry) {
    const TOKENS: u32 = 64;
    const TOP_K: u32 = 6;
    const EXPERTS: u32 = 16;
    const EXPERT_N: u32 = 512;
    const EXPERT_DISPATCHES: usize = 8;

    let block_values = GgmlType::Q2_K.block_values() as usize;
    let block_bytes = GgmlType::Q2_K.block_bytes() as usize;
    let expert_stride = EXPERT_N as usize * (K as usize / block_values) * block_bytes;
    let input = device
        .alloc_buffer(
            TOKENS as usize * K as usize * 4,
            DType::F32,
            vec![TOKENS as usize, K as usize],
        )
        .expect("expert input");
    let weight = device
        .alloc_buffer(
            EXPERTS as usize * expert_stride,
            DType::U8,
            vec![EXPERTS as usize, expert_stride],
        )
        .expect("expert weight");
    let mut ids = device
        .alloc_buffer(
            TOKENS as usize * TOP_K as usize * 4,
            DType::U32,
            vec![TOKENS as usize, TOP_K as usize],
        )
        .expect("expert ids");
    for (index, id) in ids
        .as_mut_slice::<u32>()
        .expect("expert ids slice")
        .iter_mut()
        .enumerate()
    {
        let token = index / TOP_K as usize;
        let slot = index % TOP_K as usize;
        *id = ((token * 7 + slot) % EXPERTS as usize) as u32;
    }
    let output = device
        .alloc_buffer(
            TOKENS as usize * TOP_K as usize * EXPERT_N as usize * 4,
            DType::F32,
            vec![TOKENS as usize, TOP_K as usize, EXPERT_N as usize],
        )
        .expect("expert output");
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: TOP_K,
        n: EXPERT_N,
        k: K,
        n_experts: EXPERTS,
        expert_stride: expert_stride as u64,
        ggml_type: GgmlType::Q2_K,
    };
    let mut scratch = IdMmScratch::alloc(device, EXPERTS, TOKENS).expect("expert scratch");
    let mut medians = Vec::new();

    for (label, force_mv) in [("forced-matvec", true), ("production-mm_id", false)] {
        for _ in 0..WARMUP {
            let mut encoder = device.command_encoder().expect("expert encoder");
            if force_mv {
                quantized_matmul_id_ggml_mv(
                    &mut encoder,
                    registry,
                    device,
                    &input,
                    &weight,
                    &ids,
                    &output,
                    &params,
                )
            } else {
                quantized_matmul_id_ggml_pooled(
                    &mut encoder,
                    registry,
                    device,
                    &input,
                    &weight,
                    &ids,
                    &output,
                    &mut scratch,
                    &params,
                )
            }
            .expect("expert warmup dispatch");
            encoder.commit_and_wait().expect("expert warmup completion");
        }

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let mut encoder = device.command_encoder().expect("expert encoder");
            let started = std::time::Instant::now();
            for _ in 0..EXPERT_DISPATCHES {
                if force_mv {
                    quantized_matmul_id_ggml_mv(
                        &mut encoder,
                        registry,
                        device,
                        &input,
                        &weight,
                        &ids,
                        &output,
                        &params,
                    )
                } else {
                    quantized_matmul_id_ggml_pooled(
                        &mut encoder,
                        registry,
                        device,
                        &input,
                        &weight,
                        &ids,
                        &output,
                        &mut scratch,
                        &params,
                    )
                }
                .expect("expert measured dispatch");
            }
            encoder
                .commit_and_wait()
                .expect("expert measured completion");
            samples.push(started.elapsed().as_secs_f64() * 1e6 / EXPERT_DISPATCHES as f64);
        }
        samples.sort_by(f64::total_cmp);
        let median_us = samples[samples.len() / 2];
        let p10_us = samples[samples.len() / 10];
        let p90_us = samples[samples.len() * 9 / 10];
        println!(
            "Q2_K expert {label} tokens={TOKENS} top_k={TOP_K} experts={EXPERTS} N={EXPERT_N} K={K}: median={median_us:.2} us, p10={p10_us:.2} us, p90={p90_us:.2} us"
        );
        medians.push(median_us);
    }
    println!(
        "Q2_K expert top_k=6 MM_ID speedup vs forced matvec: {:.2}x",
        medians[0] / medians[1]
    );
}

fn main() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    bench_dense(&device, &mut registry, 1, "decode");
    bench_dense(&device, &mut registry, 64, "prefill");
    bench_expert(&device, &mut registry);
}
