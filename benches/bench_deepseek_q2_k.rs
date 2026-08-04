//! Correctness-first Q2_K Metal matvec baseline at a DeepSeek-V4 hidden shape.

use mlx_native::{
    quantized_matmul_ggml, DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const M: u32 = 1;
const N: u32 = 4096;
const K: u32 = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const DISPATCHES_PER_SAMPLE: usize = 16;

fn main() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let block_values = GgmlType::Q2_K.block_values() as usize;
    let block_bytes = GgmlType::Q2_K.block_bytes() as usize;
    let weight_bytes = N as usize * (K as usize / block_values) * block_bytes;
    let input = device
        .alloc_buffer(M as usize * K as usize * 4, DType::F32, vec![K as usize])
        .expect("input");
    let weight = device
        .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
        .expect("weight");
    let output = device
        .alloc_buffer(M as usize * N as usize * 4, DType::F32, vec![N as usize])
        .expect("output");
    let params = GgmlQuantizedMatmulParams {
        m: M,
        n: N,
        k: K,
        ggml_type: GgmlType::Q2_K,
    };

    for _ in 0..WARMUP {
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
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
                &mut registry,
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
    let gb_per_s = weight_bytes as f64 / (median_us * 1e-6) / 1e9;
    println!(
        "Q2_K dense decode M={M} N={N} K={K}: median={median_us:.2} us, weight={:.2} MB, effective={gb_per_s:.2} GB/s",
        weight_bytes as f64 / 1e6,
    );
}
