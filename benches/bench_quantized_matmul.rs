//! Criterion benchmark for the quantized matmul kernel.
//!
//! Uses Gemma 4 attention projection shapes: [1, 2816] x [2816, 2816].
//! The benchmark uses synthetic random data since actual model files may not be
//! available.

use criterion::{criterion_group, criterion_main, Criterion};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice, QuantizedMatmulParams};

/// Allocate and fill an F32 buffer with a deterministic pattern.
fn alloc_f32_buffer(device: &MlxDevice, shape: Vec<usize>, fill: f32) -> MlxBuffer {
    let n: usize = shape.iter().copied().product();
    let byte_len = n * DType::F32.size_of();
    let mut buf = device.alloc_buffer(byte_len, DType::F32, shape).unwrap();
    {
        let slice: &mut [f32] = buf.as_mut_slice().unwrap();
        slice.fill(fill);
    }
    buf
}

/// Allocate and fill a BF16 buffer with a deterministic pattern.
fn alloc_bf16_buffer(device: &MlxDevice, shape: Vec<usize>, fill: f32) -> MlxBuffer {
    let n: usize = shape.iter().copied().product();
    let byte_len = n * DType::BF16.size_of();
    let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape).unwrap();
    {
        let slice: &mut [u16] = buf.as_mut_slice().unwrap();
        slice.fill((fill.to_bits() >> 16) as u16);
    }
    buf
}

/// Allocate a packed 4-bit weight buffer filled with a constant quantized value.
fn alloc_packed_4bit(device: &MlxDevice, n: usize, k: usize, qval: u8) -> MlxBuffer {
    let values_per_pack = 8usize;
    let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
    let total_packs = n * packs_per_row;
    let byte_len = total_packs * 4;

    let mut buf = device
        .alloc_buffer(byte_len, DType::U32, vec![n, packs_per_row])
        .unwrap();
    {
        let slice: &mut [u32] = buf.as_mut_slice().unwrap();
        let val = qval as u32 & 0xF;
        // Pack the same value into all nibbles.
        let packed = val
            | (val << 4)
            | (val << 8)
            | (val << 12)
            | (val << 16)
            | (val << 20)
            | (val << 24)
            | (val << 28);
        for v in slice.iter_mut() {
            *v = packed;
        }
    }
    buf
}

fn bench_quantized_matmul_gemma4(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping benchmarks");
            return;
        }
    };
    let mut registry = KernelRegistry::new();

    // Gemma 4 attention projection dimensions.
    let m: u32 = 1;
    let k: u32 = 2816;
    let n: u32 = 2816;
    let group_size: u32 = 64;
    let bits: u32 = 4;

    // Pre-allocate all buffers outside the benchmark loop.
    let input = alloc_f32_buffer(&device, vec![m as usize, k as usize], 0.01);
    let weight = alloc_packed_4bit(&device, n as usize, k as usize, 7);

    let num_groups = ((k + group_size - 1) / group_size) as usize;
    let scales = alloc_bf16_buffer(&device, vec![n as usize, num_groups], 0.1);
    let biases = alloc_bf16_buffer(&device, vec![n as usize, num_groups], 0.0);

    let params = QuantizedMatmulParams {
        m,
        k,
        n,
        group_size,
        bits,
    };

    // Warm up the pipeline compilation.
    {
        let mut enc = device.command_encoder().unwrap();
        let _out = mlx_native::quantized_matmul(
            &mut enc,
            &mut registry,
            &device,
            &input,
            &weight,
            &scales,
            &biases,
            &params,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    c.bench_function(
        &format!("quantized_matmul_4bit_[{m},{k}]x[{k},{n}]_gs{group_size}"),
        |b| {
            b.iter(|| {
                let mut enc = device.command_encoder().unwrap();
                let _out = mlx_native::quantized_matmul(
                    &mut enc,
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    &scales,
                    &biases,
                    &params,
                )
                .unwrap();
                enc.commit_and_wait().unwrap();
            });
        },
    );
}

criterion_group!(benches, bench_quantized_matmul_gemma4);
criterion_main!(benches);
