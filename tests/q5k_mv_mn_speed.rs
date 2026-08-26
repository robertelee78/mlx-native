//! Single-tenant GPU-time receipt for Q5_K scalar-tree MV, exact mN, and mv_ext.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    quantized_matmul_ggml_with_policy, DType, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_Q5_K_BYTES: usize = 176;
const REPS: usize = 100;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn fill_q5k(buffer: &mut MlxBuffer, state: &mut u64) {
    for (block, bytes) in buffer
        .as_mut_slice::<u8>()
        .expect("Q5_K buffer")
        .chunks_exact_mut(BLOCK_Q5_K_BYTES)
        .enumerate()
    {
        let d = half::f16::from_f32(0.001 + (block % 31) as f32 * 0.000_031_25);
        let dmin = half::f16::from_f32(0.000_5 + (block % 17) as f32 * 0.000_015_625);
        bytes[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        bytes[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        for byte in &mut bytes[4..] {
            *byte = next_u64(state) as u8;
        }
    }
}

fn fill_f32(buffer: &mut MlxBuffer, state: &mut u64) {
    for value in buffer.as_mut_slice::<f32>().expect("F32 buffer") {
        let unit = (next_u64(state) >> 40) as f32 / (1u32 << 24) as f32;
        *value = unit - 0.5;
    }
}

fn median_gpu_us(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    policy: &GgmlRoutingPolicy,
) -> f64 {
    let mut warmup = device.command_encoder().expect("warmup encoder");
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
    warmup.commit_and_wait().expect("warmup GPU execution");

    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
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
    samples[samples.len() / 2]
}

fn measure_shape(device: &MlxDevice, registry: &mut KernelRegistry, m: usize, n: usize, k: usize) {
    let mut state = 0x5135_5350_4545_4404 ^ m as u64 ^ n as u64 ^ (k as u64).rotate_left(17);

    let blocks = n * (k / QK_K);
    let mut weight = device
        .alloc_buffer(
            blocks * BLOCK_Q5_K_BYTES,
            DType::U8,
            vec![blocks * BLOCK_Q5_K_BYTES],
        )
        .expect("weight buffer");
    fill_q5k(&mut weight, &mut state);
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
        ggml_type: GgmlType::Q5_K,
    };

    let scalar_tree = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: false,
        dense_decode_mvn: false,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let exact_mn = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: false,
        dense_decode_mvn: true,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let mv_ext = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: true,
        dense_decode_mvn: false,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };

    let scalar_us = median_gpu_us(
        device,
        registry,
        &input,
        &weight,
        &output,
        &params,
        &scalar_tree,
    );
    let exact_us = median_gpu_us(
        device, registry, &input, &weight, &output, &params, &exact_mn,
    );
    let mv_ext_us = median_gpu_us(device, registry, &input, &weight, &output, &params, &mv_ext);

    eprintln!(
        "Q5_K m={m} N={n} K={k}: scalar={scalar_us:.3} us, exact_mN={exact_us:.3} us ({:.3}x), canonical={mv_ext_us:.3} us ({:.3}x vs exact); medians of 5x{REPS} GPU-timed calls",
        scalar_us / exact_us,
        exact_us / mv_ext_us,
    );
}

#[test]
#[ignore = "single-tenant performance receipt"]
fn q5k_width_four_gpu_time() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for (n, k) in [
        (5_120, 5_120),
        (6_144, 5_120),
        (10_240, 5_120),
        (17_408, 5_120),
        (5_120, 17_408),
    ] {
        for m in [1, 4] {
            measure_shape(&device, &mut registry, m, n, k);
        }
    }
}
