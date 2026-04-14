//! Criterion benchmark: nibble-gather vs sequential F16 KV-cache read throughput.
//!
//! Simulates Gemma-4-27B global-layer SDPA reads:
//!   head_dim   = 512  (global layer head dimension)
//!   num_kv_heads = 2  (global KV heads)
//!   capacity   = 8192 or 262144
//!
//! ADR-007 gate: nibble-gather throughput ≥ 50% of sequential F16 throughput.

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime};
use mlx_native::ops::gather_bench;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Allocate a u8 buffer (used for nibble-packed and raw byte data).
fn alloc_u8_buffer(device: &MlxDevice, len: usize, fill: u8) -> mlx_native::MlxBuffer {
    let mut buf = device
        .alloc_buffer(len, DType::U8, vec![len])
        .unwrap();
    let slice: &mut [u8] = buf.as_mut_slice().unwrap();
    slice.iter_mut().for_each(|v| *v = fill);
    buf
}

/// Allocate an f32 buffer.
fn alloc_f32_buffer(device: &MlxDevice, n: usize, fill: f32) -> mlx_native::MlxBuffer {
    let byte_len = n * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .unwrap();
    let slice: &mut [f32] = buf.as_mut_slice().unwrap();
    slice.iter_mut().for_each(|v| *v = fill);
    buf
}

/// Allocate an F16 buffer (2 bytes per element); fill is written as raw u16 bits.
fn alloc_f16_buffer_raw(device: &MlxDevice, n: usize) -> mlx_native::MlxBuffer {
    let byte_len = n * 2;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F16, vec![n])
        .unwrap();
    // Fill with a small positive F16 value: 0x3C00 = 1.0 in F16
    let slice: &mut [u8] = buf.as_mut_slice().unwrap();
    for chunk in slice.chunks_exact_mut(2) {
        chunk[0] = 0x00;
        chunk[1] = 0x3C; // little-endian 0x3C00
    }
    buf
}

// ---------------------------------------------------------------------------
// Benchmark function
// ---------------------------------------------------------------------------

fn run_gather_group(
    group: &mut BenchmarkGroup<WallTime>,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    capacity: u32,
    head_dim: u32,
) {
    let cap = capacity as usize;
    let dim = head_dim as usize;

    // --- nibble kernel buffers ---
    // packed: [capacity × head_dim/2] bytes; fill with 0x37 for deterministic nibbles
    let packed = alloc_u8_buffer(device, cap * (dim / 2), 0x37u8);
    // centroids: [16 × head_dim] f32; small constant
    let centroids = alloc_f32_buffer(device, 16 * dim, 0.01f32);
    // output (shared): [capacity × head_dim] f32
    let out_nibble = alloc_f32_buffer(device, cap * dim, 0.0f32);

    // --- F16 kernel buffers ---
    let f16_cache = alloc_f16_buffer_raw(device, cap * dim);
    let out_f16 = alloc_f32_buffer(device, cap * dim, 0.0f32);

    // Effective bytes read by nibble kernel:
    //   packed = capacity * head_dim/2   (1 byte per 2 indices)
    //   centroids touched = capacity * head_dim * 4 bytes (each thread reads 1 f32)
    // We report "effective output bytes" = capacity * head_dim * 4 for both,
    // so the ratio is purely about latency difference.
    let output_bytes = (cap * dim * 4) as u64;

    // Warm up: compile pipelines and run once.
    {
        let mut enc = device.command_encoder().unwrap();
        gather_bench::dispatch_gather_nibble(
            &mut enc,
            registry,
            device.metal_device(),
            &packed,
            &centroids,
            &out_nibble,
            capacity,
            head_dim,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();

        let mut enc = device.command_encoder().unwrap();
        gather_bench::dispatch_gather_f16_seq(
            &mut enc,
            registry,
            device.metal_device(),
            &f16_cache,
            &out_f16,
            capacity,
            head_dim,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    let param = BenchmarkId::new("nibble", format!("cap{capacity}_dim{head_dim}"));
    group.throughput(Throughput::Bytes(output_bytes));
    group.bench_with_input(param, &(capacity, head_dim), |b, _| {
        b.iter(|| {
            let mut enc = device.command_encoder().unwrap();
            gather_bench::dispatch_gather_nibble(
                &mut enc,
                registry,
                device.metal_device(),
                &packed,
                &centroids,
                &out_nibble,
                capacity,
                head_dim,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
        });
    });

    let param = BenchmarkId::new("f16_seq", format!("cap{capacity}_dim{head_dim}"));
    group.throughput(Throughput::Bytes(output_bytes));
    group.bench_with_input(param, &(capacity, head_dim), |b, _| {
        b.iter(|| {
            let mut enc = device.command_encoder().unwrap();
            gather_bench::dispatch_gather_f16_seq(
                &mut enc,
                registry,
                device.metal_device(),
                &f16_cache,
                &out_f16,
                capacity,
                head_dim,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
        });
    });
}

fn bench_gather(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping gather benchmarks");
            return;
        }
    };

    let mut registry = KernelRegistry::new();
    gather_bench::register(&mut registry);

    // Gemma-4-27B global layer parameters.
    let head_dim: u32 = 512;

    let mut group = c.benchmark_group("gather_throughput");
    // Tighten sample count for large allocations to avoid timeout.
    group.sample_size(20);

    // Current cap
    run_gather_group(&mut group, &device, &mut registry, 8192, head_dim);
    // Target full-context cap
    run_gather_group(&mut group, &device, &mut registry, 262144, head_dim);

    group.finish();

    // Print throughput ratio summary from the two capacities' last timing.
    // Criterion's output already contains per-benchmark GB/s; the ADR-007 gate
    // comparison is visible in the "gather_throughput" group output.
    println!("\n--- ADR-007 gate: nibble throughput >= 50% of f16_seq throughput ---");
    println!("See Criterion output above for GB/s values per configuration.");
}

criterion_group!(benches, bench_gather);
criterion_main!(benches);
