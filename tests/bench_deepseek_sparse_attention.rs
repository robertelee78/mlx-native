//! Opt-in latency checks for the official DeepSeek-V4 sparse-attention shape.
//! Run with `cargo test --release --test bench_deepseek_sparse_attention -- --ignored --nocapture`.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, DeepSeekSparseAttentionParams, DEEPSEEK_SPARSE_HEADS,
    DEEPSEEK_SPARSE_HEAD_DIM,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use std::time::Instant;

#[test]
#[ignore = "performance gate"]
fn benchmark_decode_and_prefill() {
    let device = MlxDevice::new().unwrap();
    for queries in [1usize, 17] {
        let kv_len = 640usize;
        let top_k = 512usize;
        let q = device
            .alloc_buffer(
                queries * DEEPSEEK_SPARSE_HEADS * DEEPSEEK_SPARSE_HEAD_DIM * 2,
                DType::BF16,
                vec![1, queries, DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM],
            )
            .unwrap();
        let kv = device
            .alloc_buffer(
                kv_len * DEEPSEEK_SPARSE_HEAD_DIM * 2,
                DType::BF16,
                vec![1, kv_len, DEEPSEEK_SPARSE_HEAD_DIM],
            )
            .unwrap();
        let sinks = device
            .alloc_buffer(
                DEEPSEEK_SPARSE_HEADS * 4,
                DType::F32,
                vec![DEEPSEEK_SPARSE_HEADS],
            )
            .unwrap();
        let mut indices = device
            .alloc_buffer(queries * top_k * 4, DType::I32, vec![1, queries, top_k])
            .unwrap();
        for (i, value) in indices
            .as_mut_slice::<i32>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *value = (i % kv_len) as i32;
        }
        let output = device
            .alloc_buffer(
                queries * DEEPSEEK_SPARSE_HEADS * DEEPSEEK_SPARSE_HEAD_DIM * 2,
                DType::BF16,
                vec![1, queries, DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM],
            )
            .unwrap();
        let params = DeepSeekSparseAttentionParams {
            batch: 1,
            query_len: queries as u32,
            kv_len: kv_len as u32,
            top_k: top_k as u32,
            heads: DEEPSEEK_SPARSE_HEADS as u32,
            head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
            scale: 1.0 / (DEEPSEEK_SPARSE_HEAD_DIM as f32).sqrt(),
        };
        let mut registry = KernelRegistry::new();
        let start = Instant::now();
        for _ in 0..10 {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_sparse_attention(
                &mut encoder,
                &mut registry,
                &device,
                &q,
                &kv,
                &sinks,
                &indices,
                &output,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        println!(
            "queries={queries} avg_ms={:.3}",
            start.elapsed().as_secs_f64() * 100.0
        );
    }
}
