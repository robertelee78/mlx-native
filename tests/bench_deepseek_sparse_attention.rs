//! Opt-in latency checks for the official DeepSeek-V4 sparse-attention shape.
//! Run with `cargo test --release --test bench_deepseek_sparse_attention -- --ignored --nocapture`.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, dispatch_deepseek_sparse_attention_flash_decode,
    dispatch_deepseek_sparse_attention_flash_prefill, DeepSeekSparseAttentionParams,
    DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM,
};
use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams;
use mlx_native::ops::flash_attn_prefill_d512::{
    dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks,
    dispatch_flash_attn_prefill_bf16_d512_with_sinks,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use std::time::Instant;

#[test]
#[ignore = "performance gate"]
fn benchmark_decode_and_prefill() {
    let device = MlxDevice::new().unwrap();
    const RUNS: usize = 100;
    for (queries, kv_len, top_k) in [
        (1usize, 640usize, 512usize),
        (1, 640, 640),
        (1, 1024, 174),
        (17, 640, 512),
    ] {
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
        {
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
        let start = Instant::now();
        for _ in 0..RUNS {
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
            "queries={queries} kv={kv_len} top_k={top_k} avg_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64
        );

        if queries == 1 {
            let gathered = device
                .alloc_buffer(
                    top_k * DEEPSEEK_SPARSE_HEAD_DIM * 2,
                    DType::BF16,
                    vec![1, 1, top_k, DEEPSEEK_SPARSE_HEAD_DIM],
                )
                .unwrap();
            let flash_params = FlashAttnPrefillParams {
                n_heads: DEEPSEEK_SPARSE_HEADS as u32,
                n_kv_heads: 1,
                head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
                seq_len_q: 1,
                seq_len_k: top_k as u32,
                batch: 1,
                scale: params.scale,
                do_causal: false,
            };
            {
                let mut encoder = device.command_encoder().unwrap();
                dispatch_flash_attn_prefill_bf16_d512_with_sinks(
                    &mut encoder,
                    &device,
                    &mut registry,
                    &q,
                    &gathered,
                    &gathered,
                    None,
                    &sinks,
                    &output,
                    &flash_params,
                )
                .unwrap();
                encoder.commit_and_wait().unwrap();
            }
            let flash_start = Instant::now();
            for _ in 0..RUNS {
                let mut encoder = device.command_encoder().unwrap();
                dispatch_flash_attn_prefill_bf16_d512_with_sinks(
                    &mut encoder,
                    &device,
                    &mut registry,
                    &q,
                    &gathered,
                    &gathered,
                    None,
                    &sinks,
                    &output,
                    &flash_params,
                )
                .unwrap();
                encoder.commit_and_wait().unwrap();
            }
            println!(
                "queries={queries} gathered_kv={top_k} flash_avg_ms={:.3}",
                flash_start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64
            );

            let mask = device
                .alloc_buffer(top_k * 2, DType::BF16, vec![1, top_k])
                .unwrap();
            let mut invalid_global = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
            invalid_global.as_mut_slice::<u32>().unwrap().fill(0);
            let mut invalid_heads = device
                .alloc_buffer(
                    DEEPSEEK_SPARSE_HEADS * 4,
                    DType::U32,
                    vec![DEEPSEEK_SPARSE_HEADS],
                )
                .unwrap();
            invalid_heads.as_mut_slice::<u32>().unwrap().fill(0);
            {
                let mut encoder = device.command_encoder().unwrap();
                dispatch_deepseek_sparse_attention_flash_decode(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &q,
                    &kv,
                    &sinks,
                    &indices,
                    &gathered,
                    &mask,
                    &invalid_global,
                    &invalid_heads,
                    &output,
                    &params,
                )
                .unwrap();
                encoder.commit_and_wait().unwrap();
            }
            let gathered_flash_start = Instant::now();
            for _ in 0..RUNS {
                let mut encoder = device.command_encoder().unwrap();
                dispatch_deepseek_sparse_attention_flash_decode(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &q,
                    &kv,
                    &sinks,
                    &indices,
                    &gathered,
                    &mask,
                    &invalid_global,
                    &invalid_heads,
                    &output,
                    &params,
                )
                .unwrap();
                encoder.commit_and_wait().unwrap();
            }
            println!(
                "queries={queries} kv={kv_len} top_k={top_k} gather_flash_avg_ms={:.3}",
                gathered_flash_start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64
            );
        }
    }

    const TILE_RUNS: usize = 10;
    for queries in [64usize, 128, 256, 512, 1024] {
        let top_k = 640usize;
        let kv_len = 8192usize;
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
        let gathered = device
            .alloc_buffer(
                queries * top_k * DEEPSEEK_SPARSE_HEAD_DIM * 2,
                DType::BF16,
                vec![1, queries, top_k, DEEPSEEK_SPARSE_HEAD_DIM],
            )
            .unwrap();
        let mask = device
            .alloc_buffer(queries * top_k * 2, DType::BF16, vec![queries, 1, top_k])
            .unwrap();
        let mut invalid_global = device
            .alloc_buffer(queries * 4, DType::U32, vec![1, queries])
            .unwrap();
        invalid_global.as_mut_slice::<u32>().unwrap().fill(0);
        let mut invalid_heads = device
            .alloc_buffer(
                queries * DEEPSEEK_SPARSE_HEADS * 4,
                DType::U32,
                vec![1, queries, DEEPSEEK_SPARSE_HEADS],
            )
            .unwrap();
        invalid_heads.as_mut_slice::<u32>().unwrap().fill(0);
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
        let flash_params = FlashAttnPrefillParams {
            n_heads: DEEPSEEK_SPARSE_HEADS as u32,
            n_kv_heads: 1,
            head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
            seq_len_q: 1,
            seq_len_k: top_k as u32,
            batch: queries as u32,
            scale: params.scale,
            do_causal: false,
        };
        let packed_flash_params = FlashAttnPrefillParams {
            n_heads: 8,
            seq_len_q: 8,
            ..flash_params
        };
        let mut registry = KernelRegistry::new();
        {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_flash_attn_prefill_bf16_d512_with_sinks(
                &mut encoder,
                &device,
                &mut registry,
                &q,
                &gathered,
                &gathered,
                Some(&mask),
                &sinks,
                &output,
                &flash_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        let flash_start = Instant::now();
        for _ in 0..TILE_RUNS {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_flash_attn_prefill_bf16_d512_with_sinks(
                &mut encoder,
                &device,
                &mut registry,
                &q,
                &gathered,
                &gathered,
                Some(&mask),
                &sinks,
                &output,
                &flash_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        let flash_elapsed = flash_start.elapsed();
        {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks(
                &mut encoder,
                &device,
                &mut registry,
                &q,
                &gathered,
                &gathered,
                &mask,
                &sinks,
                &output,
                &packed_flash_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        let packed_flash_start = Instant::now();
        for _ in 0..TILE_RUNS {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks(
                &mut encoder,
                &device,
                &mut registry,
                &q,
                &gathered,
                &gathered,
                &mask,
                &sinks,
                &output,
                &packed_flash_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        let packed_flash_elapsed = packed_flash_start.elapsed();
        let start = Instant::now();
        for _ in 0..TILE_RUNS {
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_sparse_attention_flash_prefill(
                &mut encoder,
                &mut registry,
                &device,
                &q,
                &kv,
                &sinks,
                &indices,
                &gathered,
                &mask,
                &invalid_global,
                &invalid_heads,
                &output,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "sparse_flash queries={queries} top_k={top_k} avg_ms={:.3} us_per_query={:.3} legacy_flash_ms={:.3} packed_flash_ms={:.3} flash_speedup={:.2}x adapter_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0 / TILE_RUNS as f64,
            elapsed.as_secs_f64() * 1_000_000.0 / (TILE_RUNS * queries) as f64,
            flash_elapsed.as_secs_f64() * 1000.0 / TILE_RUNS as f64,
            packed_flash_elapsed.as_secs_f64() * 1000.0 / TILE_RUNS as f64,
            flash_elapsed.as_secs_f64() / packed_flash_elapsed.as_secs_f64(),
            (elapsed.as_secs_f64() - packed_flash_elapsed.as_secs_f64()) * 1000.0
                / TILE_RUNS as f64,
        );
    }
}
