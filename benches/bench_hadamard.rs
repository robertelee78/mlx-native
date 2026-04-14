//! Criterion benchmark for the Fast Walsh-Hadamard Transform (FWHT) kernel.
//!
//! Models Gemma-4-27B per-token decode dimensions:
//!   Sliding layers: 25 layers × 2 (K+V) = 50 dispatches,  num_kv_heads=8,  head_dim=256
//!   Global  layers:  5 layers × 2 (K+V) = 10 dispatches,  num_kv_heads=2,  head_dim=512
//!
//! Gate: total full-model Hadamard overhead ≤ 200 µs (< 2 % of decode budget).

use criterion::{criterion_group, criterion_main, Criterion};
use mlx_native::ops::hadamard;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn alloc_f32(device: &MlxDevice, num_heads: usize, head_dim: usize) -> MlxBuffer {
    let n = num_heads * head_dim;
    let byte_len = n * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![num_heads, head_dim])
        .unwrap();
    {
        let slice: &mut [f32] = buf.as_mut_slice().unwrap();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = (i as f32) * 0.001;
        }
    }
    buf
}

fn bench_hadamard(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping hadamard benchmarks");
            return;
        }
    };
    let mut registry = KernelRegistry::new();
    hadamard::register(&mut registry);

    // ---- Sliding-window layer: 8 heads × 256 dim ----
    let data_d256_h8 = alloc_f32(&device, 8, 256);

    // Warm-up: compile the pipeline.
    {
        let mut enc = device.command_encoder().unwrap();
        hadamard::dispatch_hadamard_transform(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &data_d256_h8,
            256,
            8,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    c.bench_function("hadamard_d256_h8", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().unwrap();
            hadamard::dispatch_hadamard_transform(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &data_d256_h8,
                256,
                8,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
        });
    });

    // ---- Global layer: 2 heads × 512 dim ----
    let data_d512_h2 = alloc_f32(&device, 2, 512);

    // Warm-up.
    {
        let mut enc = device.command_encoder().unwrap();
        hadamard::dispatch_hadamard_transform(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &data_d512_h2,
            512,
            2,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    c.bench_function("hadamard_d512_h2", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().unwrap();
            hadamard::dispatch_hadamard_transform(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &data_d512_h2,
                512,
                2,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
        });
    });

    // ---- Full-model simulation: 50 × d256_h8  +  10 × d512_h2 ----
    //
    // All 60 dispatches go into a single command buffer to match real decode.
    // Gate: total latency ≤ 200 µs.
    c.bench_function("hadamard_full_model_60dispatches", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().unwrap();

            // 25 sliding layers × 2 (K+V) = 50 dispatches at d=256, h=8
            for _ in 0..50 {
                hadamard::dispatch_hadamard_transform(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &data_d256_h8,
                    256,
                    8,
                )
                .unwrap();
            }

            // 5 global layers × 2 (K+V) = 10 dispatches at d=512, h=2
            for _ in 0..10 {
                hadamard::dispatch_hadamard_transform(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &data_d512_h2,
                    512,
                    2,
                )
                .unwrap();
            }

            enc.commit_and_wait().unwrap();
        });
    });
}

criterion_group!(benches, bench_hadamard);
criterion_main!(benches);
