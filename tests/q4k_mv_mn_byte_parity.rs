//! GPU byte-parity gate for the Q4_K column-amortizing small-batch matvec.
//!
//! Each mN output column must match a separate invocation of the production
//! serial Q4_K matvec at the u32 bit level. The shape uses Qwen-like hidden
//! dimensions so the test crosses many blocks and threadgroups.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    dispatch_mv_q4k_mn_adaptive, quantized_matmul_ggml_with_policy, DType,
    GgmlQuantizedMatmulParams, GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_Q4_K_BYTES: usize = 144;
const N: usize = 5_120;
const K: usize = 5_120;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn random_f32(state: &mut u64) -> f32 {
    let unit = (next_u64(state) >> 40) as f32 / (1u32 << 24) as f32;
    unit - 0.5
}

fn q4k_weight_bytes(n: usize, k: usize, state: &mut u64) -> Vec<u8> {
    let n_blocks = n * (k / QK_K);
    let mut bytes = Vec::with_capacity(n_blocks * BLOCK_Q4_K_BYTES);
    for block in 0..n_blocks {
        let d = half::f16::from_f32(0.001 + (block % 31) as f32 * 0.000_031_25);
        let dmin = half::f16::from_f32(0.000_5 + (block % 17) as f32 * 0.000_015_625);
        bytes.extend_from_slice(&d.to_bits().to_le_bytes());
        bytes.extend_from_slice(&dmin.to_bits().to_le_bytes());
        for _ in 0..12 {
            bytes.push(next_u64(state) as u8);
        }
        for _ in 0..128 {
            bytes.push(next_u64(state) as u8);
        }
    }
    assert_eq!(bytes.len(), n_blocks * BLOCK_Q4_K_BYTES);
    bytes
}

fn assert_q4k_mn_matches_serial(m: usize, seed: u64) {
    let mut state = seed;
    let weight_bytes = q4k_weight_bytes(N, K, &mut state);
    let input: Vec<f32> = (0..m * K).map(|_| random_f32(&mut state)).collect();

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("weight buffer");
    weight
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weight_bytes);

    let serial_policy = GgmlRoutingPolicy {
        dense_decode_mvn: false,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let serial_params = GgmlQuantizedMatmulParams {
        m: 1,
        n: N as u32,
        k: K as u32,
        ggml_type: GgmlType::Q4_K,
    };
    let mut expected = Vec::with_capacity(m * N);
    for col in 0..m {
        let mut col_input = device
            .alloc_buffer(K * 4, DType::F32, vec![1, K])
            .expect("serial input");
        col_input
            .as_mut_slice::<f32>()
            .expect("serial input slice")
            .copy_from_slice(&input[col * K..(col + 1) * K]);
        let serial_output = device
            .alloc_buffer(N * 4, DType::F32, vec![1, N])
            .expect("serial output");
        let mut encoder = device.command_encoder().expect("serial encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &col_input,
            &weight,
            &serial_output,
            &serial_params,
            &serial_policy,
        )
        .expect("serial dispatch");
        encoder.commit_and_wait().expect("serial GPU execution");
        expected.extend_from_slice(serial_output.as_slice::<f32>().expect("serial result"));
    }

    let mut batched_input = device
        .alloc_buffer(m * K * 4, DType::F32, vec![m, K])
        .expect("mN input");
    batched_input
        .as_mut_slice::<f32>()
        .expect("mN input slice")
        .copy_from_slice(&input);
    let output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .expect("mN output");
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: N as u32,
        k: K as u32,
        ggml_type: GgmlType::Q4_K,
    };
    let mut encoder = device.command_encoder().expect("mN encoder");
    dispatch_mv_q4k_mn_adaptive(
        &mut encoder,
        &mut registry,
        &device,
        &batched_input,
        &weight,
        &output,
        &params,
    )
    .expect("mN dispatch");
    encoder.commit_and_wait().expect("mN GPU execution");

    let actual = output.as_slice::<f32>().expect("mN result");
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "Q4_K mN byte mismatch: m={m}, col={}, row={}, serial={want:?}, mN={got:?}",
            index / N,
            index % N,
        );
    }
}

#[test]
fn q4k_adaptive_mn_is_byte_identical_to_serial_for_every_production_width() {
    for m in 2..=8 {
        assert_q4k_mn_matches_serial(m, 0x4a4b_0000 + m as u64);
    }
}
