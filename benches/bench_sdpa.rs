//! Criterion benchmark for the scaled dot-product attention (SDPA) kernel.
//!
//! Uses Gemma 4 decode-mode dimensions:
//! - seq_len = 1 (single-token decode)
//! - num_heads = 8
//! - head_dim = 256
//! - kv_seq_len = 4096 (sliding window)

use criterion::{criterion_group, criterion_main, Criterion};
use mlx_native::ops::sdpa::{self, SdpaParams};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// Allocate an f32 Metal buffer filled with a constant value.
fn alloc_f32_buffer(device: &MlxDevice, shape: Vec<usize>, fill: f32) -> MlxBuffer {
    let n: usize = shape.iter().copied().product();
    let byte_len = n * 4;
    let mut buf = device.alloc_buffer(byte_len, DType::F32, shape).unwrap();
    {
        let slice: &mut [f32] = buf.as_mut_slice().unwrap();
        for v in slice.iter_mut() {
            *v = fill;
        }
    }
    buf
}

fn bench_sdpa_gemma4_decode(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping benchmarks");
            return;
        }
    };
    let mut registry = KernelRegistry::new();

    // Register the SDPA shader.
    sdpa::register(&mut registry);

    // Gemma 4 decode-mode dimensions.
    let batch_size: u32 = 1;
    let n_heads: u32 = 8;
    let n_kv_heads: u32 = 4; // GQA: 4 KV heads for 8 query heads
    let head_dim: u32 = 256;
    let seq_len: u32 = 1;
    let kv_seq_len: u32 = 4096; // sliding window

    // Q: [batch, n_heads, seq_len, head_dim]
    let q = alloc_f32_buffer(
        &device,
        vec![
            batch_size as usize,
            n_heads as usize,
            seq_len as usize,
            head_dim as usize,
        ],
        0.01,
    );

    // K: [batch, n_kv_heads, kv_seq_len, head_dim]
    let k = alloc_f32_buffer(
        &device,
        vec![
            batch_size as usize,
            n_kv_heads as usize,
            kv_seq_len as usize,
            head_dim as usize,
        ],
        0.01,
    );

    // V: [batch, n_kv_heads, kv_seq_len, head_dim]
    let v = alloc_f32_buffer(
        &device,
        vec![
            batch_size as usize,
            n_kv_heads as usize,
            kv_seq_len as usize,
            head_dim as usize,
        ],
        0.01,
    );

    // Output: [batch, n_heads, seq_len, head_dim]
    let output = alloc_f32_buffer(
        &device,
        vec![
            batch_size as usize,
            n_heads as usize,
            seq_len as usize,
            head_dim as usize,
        ],
        0.0,
    );

    let params = SdpaParams {
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len,
        scale: 1.0 / (head_dim as f32).sqrt(),
        kv_capacity: kv_seq_len,
    };

    // Warm up: compile the pipeline.
    {
        let mut enc = device.command_encoder().unwrap();
        sdpa::sdpa(
            &mut enc,
            &mut registry,
            &device,
            &q,
            &k,
            &v,
            &output,
            &params,
            batch_size,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    c.bench_function(
        &format!(
            "sdpa_decode_heads{n_heads}_kvheads{n_kv_heads}_dim{head_dim}_kvlen{kv_seq_len}"
        ),
        |b| {
            b.iter(|| {
                let mut enc = device.command_encoder().unwrap();
                sdpa::sdpa(
                    &mut enc,
                    &mut registry,
                    &device,
                    &q,
                    &k,
                    &v,
                    &output,
                    &params,
                    batch_size,
                )
                .unwrap();
                enc.commit_and_wait().unwrap();
            });
        },
    );
}

criterion_group!(benches, bench_sdpa_gemma4_decode);
criterion_main!(benches);
