//! Single-tenant GPU-time receipt for the Q8_0 NR2 multi-input-row matvec.
//!
//! Reports median GPU time and effective bandwidth (unique weight +
//! activation + output bytes per call / GPU time) for M=1 NR2, M=4 per-row
//! NR2 (`dense_q8_0_mv_r1 = false`), and M=4 multi-row NR2.
//!
//! cargo test --release --locked --test q8_0_mv_nr2_rows_speed -- --ignored --nocapture

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    quantized_matmul_ggml_with_policy, DType, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;
const REPS: usize = 100;
const SAMPLES: usize = 7;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn fill_q8_0(buffer: &mut MlxBuffer, state: &mut u64) {
    for bytes in buffer
        .as_mut_slice::<u8>()
        .expect("Q8_0 buffer")
        .chunks_exact_mut(BLOCK_Q8_0_BYTES)
    {
        let d = half::f16::from_f32(0.001 + (next_u64(state) % 64) as f32 * 0.000_25);
        bytes[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        for byte in &mut bytes[2..] {
            *byte = next_u64(state) as u8;
        }
    }
}

fn fill_f32(buffer: &mut MlxBuffer, state: &mut u64) {
    for value in buffer.as_mut_slice::<f32>().expect("F32 buffer") {
        *value = ((next_u64(state) >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
    }
}

/// Returns (median, min, max) GPU microseconds per call.
fn gpu_us(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    policy: &GgmlRoutingPolicy,
) -> (f64, f64, f64) {
    let mut warmup = device.command_encoder().expect("warmup encoder");
    for _ in 0..10 {
        quantized_matmul_ggml_with_policy(
            &mut warmup,
            registry,
            device,
            input,
            weight,
            output,
            params,
            policy,
        )
        .expect("warmup dispatch");
    }
    warmup.commit_and_wait().expect("warmup GPU execution");

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut encoder = device.command_encoder().expect("measurement encoder");
        for _ in 0..REPS {
            quantized_matmul_ggml_with_policy(
                &mut encoder,
                registry,
                device,
                input,
                weight,
                output,
                params,
                policy,
            )
            .expect("measurement dispatch");
        }
        let (start, end) = encoder
            .commit_wait_with_gpu_time()
            .expect("measurement GPU execution");
        samples.push((end - start) * 1.0e6 / REPS as f64);
    }
    samples.sort_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    )
}

fn measure(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
    policy: &GgmlRoutingPolicy,
) -> f64 {
    let mut state =
        0x5138_5350_4545_4400 ^ m as u64 ^ ((n as u64) << 8) ^ (k as u64).rotate_left(29);
    let weight_bytes = n * (k / QK8_0) * BLOCK_Q8_0_BYTES;
    let mut weight = device
        .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
        .expect("weight buffer");
    fill_q8_0(&mut weight, &mut state);
    let mut input = device
        .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
        .expect("input buffer");
    fill_f32(&mut input, &mut state);
    let output = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
        .expect("output buffer");
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q8_0,
    };
    let (median, min, max) = gpu_us(device, registry, &input, &weight, &output, &params, policy);
    let unique_bytes = (weight_bytes + m * k * 4 + m * n * 4) as f64;
    eprintln!(
        "Q8_0 N={n} K={k} M={m} {label}: median={median:.3} us (min={min:.3}, max={max:.3}), \
         {:.1} GB/s unique, {:.3} us/row; medians of {SAMPLES}x{REPS} GPU-timed calls",
        unique_bytes / (median * 1.0e3),
        median / m as f64,
    );
    median
}

#[test]
#[ignore = "single-tenant performance receipt"]
fn q8_0_multi_row_gpu_bandwidth() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let multi_row = GgmlRoutingPolicy::default();
    let per_row = GgmlRoutingPolicy {
        dense_q8_0_mv_r1: false,
        ..GgmlRoutingPolicy::default()
    };
    for (n, k) in [(4_096, 4_096), (4_096, 16_384), (16_384, 4_096)] {
        let single = measure(
            &device,
            &mut registry,
            "NR2 single-row",
            1,
            n,
            k,
            &multi_row,
        );
        let per_row_us = measure(&device, &mut registry, "NR2 per-row", 4, n, k, &per_row);
        let multi_us = measure(&device, &mut registry, "NR2 multi-row", 4, n, k, &multi_row);
        eprintln!(
            "Q8_0 N={n} K={k} M=4: multi-row {:.3}x faster than per-row, {:.3}x the M=1 time",
            per_row_us / multi_us,
            multi_us / single,
        );
    }
}
