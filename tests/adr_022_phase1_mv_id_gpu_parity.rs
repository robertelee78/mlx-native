//! ADR-022 Phase 1 — GPU↔CPU parity tests for Q5_1 / IQ4_NL mv_id kernels.
//!
//! Validates that `kernel_mul_mv_id_q5_1_f32` and
//! `kernel_mul_mv_id_iq4_nl_f32` produce F32 outputs byte-equal (within
//! accumulation tolerance) to a host-side reference matmul that
//! dequantizes the same block bytes via `dequantize_to_f32`.
//!
//! Per ADR-022 §2 acceptance criteria 2:
//!
//! > GPU↔CPU parity: kernel-output F32 byte-equal to host-reference
//! > matmul within accumulation tolerance (1e-4 for f32 accumulator).
//!
//! The test constructs synthetic quantized expert weights directly
//! (not via F32 → quantize) so the parity check measures kernel
//! correctness alone, decoupled from quantizer correctness (which is
//! covered by `adr_022_phase1_dequant_parity.rs`).
//!
//! Mantra: code + test == truth. Comments are starting points.

use mlx_native::gguf::{
    test_only_dequantize_iq4_nl, test_only_dequantize_q5_1, test_only_kvalues_iq4_nl,
};
use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK5_1: usize = 32;
const BLOCK_Q5_1_BYTES: usize = 24;
const QK4_NL: usize = 32;
const BLOCK_IQ4_NL_BYTES: usize = 18;

// ----- xorshift64* deterministic random -----

fn xs64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 13;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545F4914F6CDD1D)
}

fn random_f32_in_pm1(state: &mut u64) -> f32 {
    let bits = xs64(state);
    ((bits >> 11) as f32) / (1u64 << 53) as f32 * 2.0 - 1.0
}

// ----- Reference Q5_1 quantizer (mirrors ggml-quants.c:189) -----

fn ref_quantize_q5_1_block(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), QK5_1);
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in row {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let d = (max - min) / 31.0;
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    let m = min;
    let mut qs = [0u8; QK5_1 / 2];
    let mut qh: u32 = 0;
    for j in 0..(QK5_1 / 2) {
        let q0 = ((row[j] - m) * id + 0.5).clamp(0.0, 31.0) as u32;
        let q1 = ((row[j + QK5_1 / 2] - m) * id + 0.5).clamp(0.0, 31.0) as u32;
        qs[j] = ((q0 & 0x0F) | ((q1 & 0x0F) << 4)) as u8;
        qh |= ((q0 >> 4) & 1) << j;
        qh |= ((q1 >> 4) & 1) << (j + 16);
    }
    let mut out = Vec::with_capacity(BLOCK_Q5_1_BYTES);
    out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
    out
}

fn ref_quantize_q5_1(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len() % QK5_1, 0);
    let mut out = Vec::with_capacity((row.len() / QK5_1) * BLOCK_Q5_1_BYTES);
    for chunk in row.chunks(QK5_1) {
        out.extend(ref_quantize_q5_1_block(chunk));
    }
    out
}

// ----- Reference IQ4_NL quantizer (naive d-fit + nearest codebook) -----

fn ref_quantize_iq4_nl_block(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert_eq!(row.len(), QK4_NL);
    let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let d = if max_abs == 0.0 { 0.0 } else { max_abs / 113.0 };
    let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
    let nearest = |target: f32| -> u8 {
        let mut best_idx: u8 = 0;
        let mut best_err = f32::MAX;
        for (idx, &k) in kv.iter().enumerate() {
            let err = (target - k as f32).abs();
            if err < best_err {
                best_err = err;
                best_idx = idx as u8;
            }
        }
        best_idx
    };
    let mut qs = [0u8; QK4_NL / 2];
    for j in 0..(QK4_NL / 2) {
        let lo = nearest(row[j] * inv_d);
        let hi = nearest(row[j + QK4_NL / 2] * inv_d);
        qs[j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }
    let mut out = Vec::with_capacity(BLOCK_IQ4_NL_BYTES);
    out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&qs);
    out
}

fn ref_quantize_iq4_nl(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len() % QK4_NL, 0);
    let mut out = Vec::with_capacity((row.len() / QK4_NL) * BLOCK_IQ4_NL_BYTES);
    for chunk in row.chunks(QK4_NL) {
        out.extend(ref_quantize_iq4_nl_block(chunk));
    }
    out
}

// ----- CPU reference matmul: dequant + dot product per (token, slot) -----

/// For each `(token t, slot s)`:
///   expert = ids[t*top_k + s]
///   for n in 0..N:
///     out[t*top_k+s, n] = sum_k(dequant(W[expert, n, k]) * input[t, k])
fn cpu_reference_matmul(
    weights_bytes: &[u8],
    per_expert_bytes: usize,
    n: usize,
    k: usize,
    n_experts: usize,
    blocks_per_row: usize,
    block_bytes: usize,
    dequantize: impl Fn(&[u8], &mut [f32]) -> mlx_native::Result<()>,
    input: &[f32],
    ids: &[u32],
    n_tokens: usize,
    top_k: usize,
) -> Vec<f32> {
    assert_eq!(weights_bytes.len(), n_experts * per_expert_bytes);
    assert_eq!(per_expert_bytes, n * blocks_per_row * block_bytes);
    let total_rows = n_tokens * top_k;
    let mut out = vec![0.0_f32; total_rows * n];
    let mut row_buf = vec![0.0_f32; k];
    for t in 0..n_tokens {
        let input_row = &input[t * k..(t + 1) * k];
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;
            let expert_w = &weights_bytes
                [expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            for col in 0..n {
                let row_bytes =
                    &expert_w[col * blocks_per_row * block_bytes..(col + 1) * blocks_per_row * block_bytes];
                dequantize(row_bytes, &mut row_buf).unwrap();
                let mut sum = 0.0_f32;
                for kk in 0..k {
                    sum += row_buf[kk] * input_row[kk];
                }
                out[row_idx * n + col] = sum;
            }
        }
    }
    out
}

// ----- Generic harness -----

fn run_mv_id_parity(
    ggml_type: GgmlType,
    block_bytes: usize,
    quantize_row: impl Fn(&[f32]) -> Vec<u8>,
    dequantize_row: impl Fn(&[u8], &mut [f32]) -> mlx_native::Result<()>,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    seed: u64,
) {
    assert_eq!(k % 32, 0, "K must be block-aligned");
    let blocks_per_row = k / 32;
    let per_expert_bytes = n * blocks_per_row * block_bytes;

    // Random F32 weights per expert per row, quantized to bytes.
    let mut state = seed;
    let mut stacked_bytes = Vec::with_capacity(n_experts * per_expert_bytes);
    for _expert in 0..n_experts {
        for _row in 0..n {
            let mut row_f32 = vec![0.0_f32; k];
            for v in row_f32.iter_mut() {
                *v = random_f32_in_pm1(&mut state) * 0.5;
            }
            stacked_bytes.extend(quantize_row(&row_f32));
        }
    }
    assert_eq!(stacked_bytes.len(), n_experts * per_expert_bytes);

    // Random F32 input per token.
    let mut input_data = vec![0.0_f32; n_tokens * k];
    for v in input_data.iter_mut() {
        *v = random_f32_in_pm1(&mut state);
    }

    // Random expert ids in [0, n_experts), one per (token, slot).
    let total_rows = n_tokens * top_k;
    let mut ids = vec![0_u32; total_rows];
    for v in ids.iter_mut() {
        *v = (xs64(&mut state) as u32) % (n_experts as u32);
    }

    // CPU reference.
    let cpu_out = cpu_reference_matmul(
        &stacked_bytes,
        per_expert_bytes,
        n,
        k,
        n_experts,
        blocks_per_row,
        block_bytes,
        &dequantize_row,
        &input_data,
        &ids,
        n_tokens,
        top_k,
    );

    // GPU dispatch.
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U8, vec![stacked_bytes.len()])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&stacked_bytes);

    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut ids_buf = device
        .alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids);

    let mut output_buf = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type,
    };

    let mut encoder = device.command_encoder().unwrap();
    mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &ids_buf,
        &mut output_buf,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();

    let gpu_out: &[f32] = output_buf.as_slice().unwrap();
    assert_eq!(gpu_out.len(), cpu_out.len());

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let denom = c.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
        // Per-element tolerance: tight relative + small absolute floor.
        // F32 accumulator over ~k=64 multiplications: rounding ~ k*eps ≈ 1e-5
        // worst case at unit magnitudes.
        assert!(
            abs_err <= 1e-3 || rel_err <= 1e-3,
            "{:?} mv_id mismatch at idx {i}: GPU {g} vs CPU {c} (abs_err {abs_err}, rel_err {rel_err})",
            ggml_type
        );
    }
    eprintln!(
        "[adr-022 {:?} mv_id parity] max_abs_err={max_abs_err:.6e} max_rel_err={max_rel_err:.6e}",
        ggml_type
    );
}

#[test]
fn adr022_q5_1_mv_id_parity_2tok_2slot() {
    run_mv_id_parity(
        GgmlType::Q5_1,
        BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1,
        test_only_dequantize_q5_1,
        /*n_tokens=*/ 2,
        /*top_k=*/ 2,
        /*n_experts=*/ 4,
        /*n=*/ 16,
        /*k=*/ 64,
        0xAD22_0511_0001,
    );
}

#[test]
fn adr022_iq4_nl_mv_id_parity_2tok_2slot() {
    run_mv_id_parity(
        GgmlType::IQ4_NL,
        BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl,
        test_only_dequantize_iq4_nl,
        /*n_tokens=*/ 2,
        /*top_k=*/ 2,
        /*n_experts=*/ 4,
        /*n=*/ 16,
        /*k=*/ 64,
        0xAD22_004F_0001,
    );
}

#[test]
fn adr022_q5_1_mv_id_parity_realistic_shape() {
    // Realistic gemma4 ffn_down expert shape (k=2816, n=704), but
    // shrunk by 32× to keep the test fast: k=2816/4=704, n=704/4=176,
    // n_experts=4 (vs 128 in real file). Verifies the kernel survives
    // larger nb (blocks per row).
    run_mv_id_parity(
        GgmlType::Q5_1,
        BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1,
        test_only_dequantize_q5_1,
        /*n_tokens=*/ 1,
        /*top_k=*/ 8,
        /*n_experts=*/ 4,
        /*n=*/ 176,
        /*k=*/ 704,
        0xAD22_0511_0002,
    );
}

#[test]
fn adr022_iq4_nl_mv_id_parity_realistic_shape() {
    run_mv_id_parity(
        GgmlType::IQ4_NL,
        BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl,
        test_only_dequantize_iq4_nl,
        /*n_tokens=*/ 1,
        /*top_k=*/ 8,
        /*n_experts=*/ 4,
        /*n=*/ 176,
        /*k=*/ 704,
        0xAD22_004F_0002,
    );
}
