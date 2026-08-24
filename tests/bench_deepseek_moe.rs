//! Opt-in DeepSeek-V4 0731 MoE activation-path microbenchmarks.
//! Run with `cargo test --release --test bench_deepseek_moe -- --ignored --nocapture`.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_moe_activation::{
    dispatch_deepseek_moe_swiglu, dispatch_deepseek_moe_weighted_reduce, DEEPSEEK_MOE_HIDDEN_DIM,
    DEEPSEEK_MOE_INTER_DIM,
};
use mlx_native::ops::deepseek_moe_routing::{
    dispatch_deepseek_moe_hash_route, dispatch_deepseek_moe_score_route, DEEPSEEK_MOE_EXPERTS,
    DEEPSEEK_MOE_TOP_K,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use std::time::Instant;

const E: usize = DEEPSEEK_MOE_EXPERTS;
const K: usize = DEEPSEEK_MOE_TOP_K;
const I: usize = DEEPSEEK_MOE_INTER_DIM;
const H: usize = DEEPSEEK_MOE_HIDDEN_DIM;
const VOCAB: usize = 129_280;
const ITERATIONS: usize = 20;

fn f32_buffer(device: &MlxDevice, elements: usize, shape: Vec<usize>) -> MlxBuffer {
    device
        .alloc_buffer(elements * 4, DType::F32, shape)
        .unwrap()
}

fn i32_buffer(device: &MlxDevice, values: &[i32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::I32, shape)
        .unwrap();
    buffer.as_mut_slice().unwrap().copy_from_slice(values);
    buffer
}

fn measure(mut operation: impl FnMut()) -> f64 {
    for _ in 0..2 {
        operation();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        operation();
    }
    start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64
}

#[test]
#[ignore = "performance gate"]
fn benchmark_decode_and_prefill_production_shapes() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let bias = f32_buffer(&device, E, vec![E]);
    let table_values = (0..VOCAB * K)
        .map(|index| (index % E) as i32)
        .collect::<Vec<_>>();
    let table = i32_buffer(&device, &table_values, vec![VOCAB, K]);

    for tokens in [1usize, 17] {
        let logits = f32_buffer(&device, tokens * E, vec![tokens, E]);
        let token_ids = i32_buffer(
            &device,
            &(0..tokens).map(|token| token as i32).collect::<Vec<_>>(),
            vec![tokens],
        );
        let route_ids = i32_buffer(&device, &vec![0; tokens * K], vec![tokens, K]);
        let route_weights = f32_buffer(&device, tokens * K, vec![tokens, K]);
        let mut invalid_status = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        invalid_status.as_mut_slice::<u32>().unwrap()[0] = 0;

        let score_ms = measure(|| {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_score_route(
                &mut encoder,
                &mut registry,
                &device,
                &logits,
                &bias,
                &route_ids,
                &route_weights,
                tokens,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        });
        let hash_ms = measure(|| {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_hash_route(
                &mut encoder,
                &mut registry,
                &device,
                &logits,
                &token_ids,
                &table,
                &route_ids,
                &route_weights,
                tokens,
                VOCAB,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        });

        let rows = tokens * K;
        let gate = f32_buffer(&device, rows * I, vec![rows, I]);
        let up = f32_buffer(&device, rows * I, vec![rows, I]);
        let selected_weights = f32_buffer(&device, rows, vec![rows]);
        let activated = f32_buffer(&device, rows * I, vec![rows, I]);
        let swiglu_ms = measure(|| {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_swiglu(
                &mut encoder,
                &mut registry,
                &device,
                &gate,
                &up,
                Some(&selected_weights),
                &activated,
                &invalid_status,
                rows,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        });

        let valid_ids = (0..tokens * K)
            .map(|index| (index % E) as i32)
            .collect::<Vec<_>>();
        let valid_ids = i32_buffer(&device, &valid_ids, vec![tokens, K]);
        let routed = f32_buffer(&device, tokens * K * H, vec![tokens, K, H]);
        let shared = f32_buffer(&device, tokens * H, vec![tokens, H]);
        let reduced = f32_buffer(&device, tokens * H, vec![tokens, H]);
        let reduce_ms = measure(|| {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_weighted_reduce(
                &mut encoder,
                &mut registry,
                &device,
                &valid_ids,
                &route_weights,
                &routed,
                &shared,
                &reduced,
                &invalid_status,
                tokens,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        });

        println!(
            "tokens={tokens} score_route_ms={score_ms:.3} hash_route_ms={hash_ms:.3} \
             swiglu_rows={rows} swiglu_ms={swiglu_ms:.3} reduce_ms={reduce_ms:.3}"
        );
    }
}
