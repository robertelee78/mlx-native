//! ADR-034 task #94 (2026-05-21) — parity test for the fused
//! `kernel_fused_dual_proj_q4_0_f32` kernel.
//!
//! Asserts byte-identical (within F32 tolerance) output vs the unfused
//! 2-dispatch sequence:
//!   1. `quantized_matmul_ggml(Q4_0, weight_a, x)` → tmp_a
//!   2. `quantized_matmul_ggml(Q4_0, weight_b, x)` → tmp_b
//!
//! Falsification gate: if any m ∈ {1, 2, 4} fails, the fused kernel
//! is buggy and any subsequent perf claim is meaningless.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_dual_proj_q4_0::{
    dispatch_fused_dual_proj_q4_0, FusedDualProjQ4_0Args,
};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::{
    DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK4_0: usize = 32;

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Pack a flat F32 array (length multiple of QK4_0=32) into GGUF Q4_0 blocks
/// (18 bytes each: 2-byte half scale + 16 packed 4-bit pairs).
fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK4_0 == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK4_0 * 18);
    for block in values.chunks(QK4_0) {
        // Find max abs * sign (the canonical encoder uses sign of max-abs for d).
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max_val = v;
            }
        }
        // Use signed max as numerator (sign carries through to d).
        let d = max_val / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let half = half::f16::from_f32(d).to_bits();
        bytes.extend_from_slice(&half.to_le_bytes());
        // Pack 32 quants as 16 bytes, lower nibble = quants[i], upper = quants[i+16].
        // Range [-8, 7] biased to [0, 15] before packing.
        for i in 0..16 {
            let q0 = (block[i] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            let q1 = (block[i + 16] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            bytes.push(q0 | (q1 << 4));
        }
    }
    bytes
}

fn run_parity_at_m(m: u32) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let hidden_size: u32 = 256; // 8 blocks
    let output_size: u32 = 64;  // 8 rows/TG × 8 TGs

    // Generate input + 2 weight matrices.
    let input = pseudo_random_f32(0xABCD_1234, (hidden_size * m) as usize);
    let w_a_f32 = pseudo_random_f32(0xDEAD_BEEF, (output_size * hidden_size) as usize);
    let w_b_f32 = pseudo_random_f32(0xF00D_CAFE, (output_size * hidden_size) as usize);

    let w_a_q4_0 = pack_q4_0(&w_a_f32);
    let w_b_q4_0 = pack_q4_0(&w_b_f32);

    let mut input_buf = device
        .alloc_buffer(
            (hidden_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, hidden_size as usize],
        )
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input);

    let mut w_a_buf = device
        .alloc_buffer(
            w_a_q4_0.len(),
            DType::U8,
            vec![output_size as usize, hidden_size as usize],
        )
        .expect("alloc w_a");
    w_a_buf
        .as_mut_slice::<u8>()
        .expect("w_a slice")
        .copy_from_slice(&w_a_q4_0);

    let mut w_b_buf = device
        .alloc_buffer(
            w_b_q4_0.len(),
            DType::U8,
            vec![output_size as usize, hidden_size as usize],
        )
        .expect("alloc w_b");
    w_b_buf
        .as_mut_slice::<u8>()
        .expect("w_b slice")
        .copy_from_slice(&w_b_q4_0);

    // --- Unfused reference: 2 separate quantized_matmul_ggml dispatches ---
    let tmp_a = device
        .alloc_buffer(
            (output_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, output_size as usize],
        )
        .expect("alloc tmp_a");
    let tmp_b = device
        .alloc_buffer(
            (output_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, output_size as usize],
        )
        .expect("alloc tmp_b");

    let mv_params = GgmlQuantizedMatmulParams {
        m,
        n: output_size,
        k: hidden_size,
        ggml_type: GgmlType::Q4_0,
    };

    let mut enc = device.command_encoder().expect("encoder unfused");
    quantized_matmul_ggml(
        &mut enc,
        &mut registry,
        &device,
        &input_buf,
        &w_a_buf,
        &tmp_a,
        &mv_params,
    )
    .expect("matvec a");
    quantized_matmul_ggml(
        &mut enc,
        &mut registry,
        &device,
        &input_buf,
        &w_b_buf,
        &tmp_b,
        &mv_params,
    )
    .expect("matvec b");
    enc.commit_and_wait().expect("commit unfused");

    let unfused_a: Vec<f32> = tmp_a.as_slice::<f32>().expect("read tmp_a").to_vec();
    let unfused_b: Vec<f32> = tmp_b.as_slice::<f32>().expect("read tmp_b").to_vec();

    // --- Fused dispatch ---
    let dst_a = device
        .alloc_buffer(
            (output_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, output_size as usize],
        )
        .expect("alloc dst_a");
    let dst_b = device
        .alloc_buffer(
            (output_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, output_size as usize],
        )
        .expect("alloc dst_b");

    let mut enc2 = device.command_encoder().expect("encoder fused");
    dispatch_fused_dual_proj_q4_0(
        &mut enc2,
        &mut registry,
        &device,
        &w_a_buf,
        &w_b_buf,
        &input_buf,
        &dst_a,
        &dst_b,
        FusedDualProjQ4_0Args {
            m,
            output_size,
            hidden_size,
        },
    )
    .expect("fused dispatch");
    enc2.commit_and_wait().expect("commit fused");

    let fused_a: Vec<f32> = dst_a.as_slice::<f32>().expect("read dst_a").to_vec();
    let fused_b: Vec<f32> = dst_b.as_slice::<f32>().expect("read dst_b").to_vec();

    // Assert byte-identical (within F32 tolerance) — same accumulator order,
    // same block_q4_0_dot_y helper logic.
    assert_eq!(fused_a.len(), unfused_a.len());
    assert_eq!(fused_b.len(), unfused_b.len());

    let check = |label: &str, fused: &[f32], unfused: &[f32]| {
        let mut max_abs = 0.0f32;
        for (i, (&a, &b)) in fused.iter().zip(unfused.iter()).enumerate() {
            let abs = (a - b).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            if a.to_bits() != b.to_bits() && i < 5 {
                eprintln!(
                    "[{label} diff @ row {i}] fused={a:.6e} ({:#010x}) unfused={b:.6e} ({:#010x})",
                    a.to_bits(),
                    b.to_bits(),
                );
            }
        }
        eprintln!("{label} (m={m}): max_abs_diff={max_abs:.3e}");
        assert!(
            max_abs < 1e-5,
            "{label} fused vs unfused max_abs_diff {max_abs:.3e} exceeds 1e-5 (m={m})"
        );
    };

    check("dst_a", &fused_a, &unfused_a);
    check("dst_b", &fused_b, &unfused_b);
}

#[test]
fn fused_dual_proj_q4_0_m_eq_1_byte_identical() {
    run_parity_at_m(1);
}

#[test]
fn fused_dual_proj_q4_0_m_eq_2_byte_identical() {
    run_parity_at_m(2);
}

#[test]
fn fused_dual_proj_q4_0_m_eq_4_byte_identical() {
    run_parity_at_m(4);
}
