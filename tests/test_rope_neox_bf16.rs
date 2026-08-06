//! Runtime coverage for the generic NeoX bf16 RoPE binding contract.

#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn neox_bf16_long_theta_matches_host_frequency_schedule() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::rope::register(&mut registry);

    let theta = 1_000_000.0_f32;
    let seq_len = 1_u32;
    let n_heads = 2_u32;
    let head_dim = 16_u32;
    let rope_dim = 16_u32;
    let elements = (seq_len * n_heads * head_dim) as usize;
    let source: Vec<f32> = (0..elements)
        .map(|i| ((i as f32) * 0.17 - 1.0).sin())
        .collect();

    let mut input = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("input");
    input
        .as_mut_slice::<half::bf16>()
        .expect("input slice")
        .iter_mut()
        .zip(&source)
        .for_each(|(dst, &src)| *dst = half::bf16::from_f32(src));
    let output = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("output");
    let mut params = device
        .alloc_buffer(16, DType::F32, vec![4])
        .expect("params");
    params
        .as_mut_slice::<f32>()
        .expect("params slice")
        .copy_from_slice(&[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let mut positions = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("positions");
    positions.as_mut_slice::<u32>().expect("position slice")[0] = 2048;

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        &positions,
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let input_f32: Vec<f32> = input
        .as_slice::<half::bf16>()
        .expect("read input")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let actual: Vec<f32> = output
        .as_slice::<half::bf16>()
        .expect("read output")
        .iter()
        .map(|v| v.to_f32())
        .collect();
    let half_dim = (head_dim / 2) as usize;
    for row in 0..(seq_len * n_heads) as usize {
        let base = row * head_dim as usize;
        for pair in 0..half_dim {
            let ratio = (2 * pair) as f32 / head_dim as f32;
            let angle = 2048.0_f32 * (1.0_f32 / theta.powf(ratio));
            let x0 = input_f32[base + pair];
            let x1 = input_f32[base + pair + half_dim];
            let expected0 = x0 * angle.cos() - x1 * angle.sin();
            let expected1 = x1 * angle.cos() + x0 * angle.sin();
            for (index, expected) in [
                (base + pair, expected0),
                (base + pair + half_dim, expected1),
            ] {
                assert!(
                    (actual[index] - expected).abs() <= 8.0e-3,
                    "NeoX bf16 mismatch at {index}: expected={expected}, got={}",
                    actual[index]
                );
            }
        }
    }
}
