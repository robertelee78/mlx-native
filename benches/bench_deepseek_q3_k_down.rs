//! Q3_K DeepSeek-V4 expert-down decode and prefill benchmark.
//!
//! Logical production shape: 256 experts, K=2048 expert intermediate,
//! N=4096 hidden output. Decode flattens six selected expert activations;
//! prefill models 64 tokens × top-k 6 as 384 routed activation rows.

use mlx_native::{
    quantized_matmul_id_ggml_pooled, DType, GgmlQuantizedMatmulIdParams, GgmlType, IdMmScratch,
    KernelRegistry, MlxDevice,
};

const EXPERTS: u32 = 256;
const N: u32 = 4096;
const K: u32 = 2048;
const WARMUP: usize = 3;
const SAMPLES: usize = 12;

fn bench_down(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weight: &mlx_native::MlxBuffer,
    expert_stride: u64,
    routed_rows: u32,
    label: &str,
) {
    let input = device
        .alloc_buffer(
            routed_rows as usize * K as usize * 4,
            DType::F32,
            vec![routed_rows as usize, K as usize],
        )
        .expect("input");
    let mut ids = device
        .alloc_buffer(
            routed_rows as usize * 4,
            DType::U32,
            vec![routed_rows as usize],
        )
        .expect("ids");
    for (row, id) in ids
        .as_mut_slice::<u32>()
        .expect("ids slice")
        .iter_mut()
        .enumerate()
    {
        *id = (row as u32 * 131 + 17) % EXPERTS;
    }
    let output = device
        .alloc_buffer(
            routed_rows as usize * N as usize * 4,
            DType::F32,
            vec![routed_rows as usize, N as usize],
        )
        .expect("output");
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: routed_rows,
        top_k: 1,
        n: N,
        k: K,
        n_experts: EXPERTS,
        expert_stride,
        ggml_type: GgmlType::Q3_K,
    };
    let mut scratch = IdMmScratch::alloc(device, EXPERTS, routed_rows).expect("scratch");

    for _ in 0..WARMUP {
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_pooled(
            &mut encoder,
            registry,
            device,
            &input,
            weight,
            &ids,
            &output,
            &mut scratch,
            &params,
        )
        .expect("warmup dispatch");
        encoder.commit_and_wait().expect("warmup completion");
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut encoder = device.command_encoder().expect("encoder");
        let started = std::time::Instant::now();
        quantized_matmul_id_ggml_pooled(
            &mut encoder,
            registry,
            device,
            &input,
            weight,
            &ids,
            &output,
            &mut scratch,
            &params,
        )
        .expect("measured dispatch");
        encoder.commit_and_wait().expect("measured completion");
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "Q3_K DeepSeek-V4 down {label}: rows={routed_rows} experts={EXPERTS} N={N} K={K} median={:.2} us p10={:.2} us p90={:.2} us",
        samples[samples.len() / 2],
        samples[samples.len() / 10],
        samples[samples.len() * 9 / 10],
    );
}

fn main() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let blocks_per_row = K as u64 / GgmlType::Q3_K.block_values() as u64;
    let expert_stride = N as u64 * blocks_per_row * GgmlType::Q3_K.block_bytes() as u64;
    let total_weight_bytes = EXPERTS as u64 * expert_stride;
    let mut weight = device
        .alloc_buffer(
            total_weight_bytes as usize,
            DType::U8,
            vec![EXPERTS as usize, expert_stride as usize],
        )
        .expect("Q3_K expert weights");
    weight
        .as_mut_slice::<u8>()
        .expect("Q3_K expert weight slice")
        .fill(0);

    println!(
        "Q3_K DeepSeek-V4 expert-down weights: {:.2} MiB (per expert {:.2} MiB)",
        total_weight_bytes as f64 / 1_048_576.0,
        expert_stride as f64 / 1_048_576.0,
    );
    bench_down(
        &device,
        &mut registry,
        &weight,
        expert_stride,
        6,
        "decode-mv_id",
    );
    bench_down(
        &device,
        &mut registry,
        &weight,
        expert_stride,
        64 * 6,
        "prefill-mm_id",
    );
}
