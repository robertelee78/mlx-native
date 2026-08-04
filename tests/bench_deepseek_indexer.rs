//! Opt-in DeepSeek-V4 indexer latency receipt.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_indexer::{dispatch_deepseek_indexer, DeepSeekIndexerParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use std::time::Instant;

#[test]
#[ignore = "performance gate"]
fn benchmark_prefill_and_production_decode() {
    let device = MlxDevice::new().unwrap();
    for (queries, kv_len, start_pos) in [(128usize, 32usize, 0usize), (1, 640, 2559)] {
        let q = device
            .alloc_buffer(
                queries * 64 * 128 * 2,
                DType::BF16,
                vec![1, queries, 64, 128],
            )
            .unwrap();
        let kv = device
            .alloc_buffer(kv_len * 128 * 2, DType::BF16, vec![1, kv_len, 128])
            .unwrap();
        let weights = device
            .alloc_buffer(queries * 64 * 4, DType::F32, vec![1, queries, 64])
            .unwrap();
        let scratch = device
            .alloc_buffer(queries * kv_len * 4, DType::F32, vec![1, queries, kv_len])
            .unwrap();
        let output = device
            .alloc_buffer(queries * 512 * 4, DType::I32, vec![1, queries, 512])
            .unwrap();
        let params = DeepSeekIndexerParams {
            batch: 1,
            query_len: queries as u32,
            kv_len: kv_len as u32,
            start_pos: start_pos as u32,
            ratio: 4,
            heads: 64,
            head_dim: 128,
            top_k: 512,
            offset: 128,
        };
        let mut registry = KernelRegistry::new();
        let start = Instant::now();
        for _ in 0..10 {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_indexer(
                &mut encoder,
                &mut registry,
                &device,
                &q,
                &kv,
                &weights,
                &scratch,
                &output,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        println!(
            "queries={queries} kv={kv_len} avg_ms={:.3}",
            start.elapsed().as_secs_f64() * 100.0
        );
    }
}
