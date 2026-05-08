//! ADR-022 Phase 1 — GPU↔CPU parity for Q5_1 / IQ4_NL DENSE mv kernels.
//!
//! Sibling to `adr_022_phase1_mv_id_gpu_parity.rs` but exercises the
//! dense (non-id) `kernel_mul_mv_q5_1_f32` and
//! `kernel_mul_mv_iq4_nl_f32` via `quantized_matmul_ggml`.
//!
//! Per ADR-022 §2 acceptance criteria 2.

use mlx_native::gguf::{
    test_only_dequantize_iq4_nl, test_only_dequantize_q5_1, test_only_kvalues_iq4_nl,
};
use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice};

const QK5_1: usize = 32;
const BLOCK_Q5_1_BYTES: usize = 24;
const QK4_NL: usize = 32;
const BLOCK_IQ4_NL_BYTES: usize = 18;

fn xs64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 13;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545F4914F6CDD1D)
}

fn random_pm1(state: &mut u64) -> f32 {
    let bits = xs64(state);
    ((bits >> 11) as f32) / (1u64 << 53) as f32 * 2.0 - 1.0
}

fn ref_quantize_q5_1(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len() % QK5_1, 0);
    let mut out = Vec::with_capacity((row.len() / QK5_1) * BLOCK_Q5_1_BYTES);
    for chunk in row.chunks(QK5_1) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in chunk {
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
            let q0 = ((chunk[j] - m) * id + 0.5).clamp(0.0, 31.0) as u32;
            let q1 = ((chunk[j + QK5_1 / 2] - m) * id + 0.5).clamp(0.0, 31.0) as u32;
            qs[j] = ((q0 & 0x0F) | ((q1 & 0x0F) << 4)) as u8;
            qh |= ((q0 >> 4) & 1) << j;
            qh |= ((q1 >> 4) & 1) << (j + 16);
        }
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

fn ref_quantize_iq4_nl(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert_eq!(row.len() % QK4_NL, 0);
    let mut out = Vec::with_capacity((row.len() / QK4_NL) * BLOCK_IQ4_NL_BYTES);
    for chunk in row.chunks(QK4_NL) {
        let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 113.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        let nearest = |t: f32| -> u8 {
            let mut best_idx: u8 = 0;
            let mut best_err = f32::MAX;
            for (i, &k) in kv.iter().enumerate() {
                let e = (t - k as f32).abs();
                if e < best_err {
                    best_err = e;
                    best_idx = i as u8;
                }
            }
            best_idx
        };
        let mut qs = [0u8; QK4_NL / 2];
        for j in 0..(QK4_NL / 2) {
            let lo = nearest(chunk[j] * inv_d);
            let hi = nearest(chunk[j + QK4_NL / 2] * inv_d);
            qs[j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
        }
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

/// CPU reference dense matmul: out[m_idx, col] = sum_k(W[col, k] * input[m_idx, k]).
/// Weights are stored as N rows of K elements (row-major), each row quantized
/// in `block_bytes`-sized blocks of `qk` elements.
fn cpu_dense_matmul(
    weights_bytes: &[u8],
    n: usize,
    k: usize,
    qk: usize,
    block_bytes: usize,
    dequantize: impl Fn(&[u8], &mut [f32]) -> mlx_native::Result<()>,
    input: &[f32],
    m: usize,
) -> Vec<f32> {
    let blocks_per_row = k / qk;
    let row_bytes = blocks_per_row * block_bytes;
    assert_eq!(weights_bytes.len(), n * row_bytes);
    let mut out = vec![0.0_f32; m * n];
    let mut row_buf = vec![0.0_f32; k];
    for mi in 0..m {
        let in_row = &input[mi * k..(mi + 1) * k];
        for col in 0..n {
            let row = &weights_bytes[col * row_bytes..(col + 1) * row_bytes];
            dequantize(row, &mut row_buf).unwrap();
            let mut sum = 0.0_f32;
            for kk in 0..k {
                sum += row_buf[kk] * in_row[kk];
            }
            out[mi * n + col] = sum;
        }
    }
    out
}

fn run_dense_mv_parity(
    ggml_type: GgmlType,
    qk: usize,
    block_bytes: usize,
    quantize: impl Fn(&[f32]) -> Vec<u8>,
    dequantize: impl Fn(&[u8], &mut [f32]) -> mlx_native::Result<()>,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
) {
    assert_eq!(k % qk, 0);
    let blocks_per_row = k / qk;
    let row_bytes = blocks_per_row * block_bytes;
    let total_weight_bytes = n * row_bytes;

    let mut state = seed;
    let mut weights_bytes = Vec::with_capacity(total_weight_bytes);
    for _ in 0..n {
        let mut row_f32 = vec![0.0_f32; k];
        for v in row_f32.iter_mut() {
            *v = random_pm1(&mut state) * 0.5;
        }
        weights_bytes.extend(quantize(&row_f32));
    }
    assert_eq!(weights_bytes.len(), total_weight_bytes);

    let mut input_data = vec![0.0_f32; m * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    let cpu_out = cpu_dense_matmul(
        &weights_bytes,
        n,
        k,
        qk,
        block_bytes,
        &dequantize,
        &input_data,
        m,
    );

    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();

    let mut weight_buf = device
        .alloc_buffer(weights_bytes.len(), DType::U8, vec![weights_bytes.len()])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&weights_bytes);

    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut output_buf = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m * n])
        .unwrap();

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type,
    };

    let mut encoder = device.command_encoder().unwrap();
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();

    let gpu_out: &[f32] = output_buf.as_slice().unwrap();
    let mut max_abs_err = 0.0_f32;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let denom = c.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        assert!(
            abs_err <= 1e-3 || rel_err <= 1e-3,
            "{:?} dense mv mismatch at idx {i}: GPU {g} vs CPU {c} (abs {abs_err}, rel {rel_err})",
            ggml_type
        );
    }
    eprintln!(
        "[adr-022 {:?} dense mv parity] max_abs_err={max_abs_err:.6e} m={m} n={n} k={k}",
        ggml_type
    );
}

#[test]
fn adr022_q5_1_dense_mv_parity_small() {
    run_dense_mv_parity(
        GgmlType::Q5_1,
        QK5_1,
        BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1,
        test_only_dequantize_q5_1,
        /*m=*/ 1,
        /*n=*/ 16,
        /*k=*/ 64,
        0xAD22_0511_D001,
    );
}

#[test]
fn adr022_iq4_nl_dense_mv_parity_small() {
    run_dense_mv_parity(
        GgmlType::IQ4_NL,
        QK4_NL,
        BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl,
        test_only_dequantize_iq4_nl,
        1,
        16,
        64,
        0xAD22_004F_D001,
    );
}

#[test]
fn adr022_q5_1_dense_mv_parity_realistic() {
    run_dense_mv_parity(
        GgmlType::Q5_1,
        QK5_1,
        BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1,
        test_only_dequantize_q5_1,
        4,    // m = 4 input rows
        176,  // n = output dim
        704,  // k = input dim
        0xAD22_0511_D002,
    );
}

#[test]
fn adr022_iq4_nl_dense_mv_parity_realistic() {
    run_dense_mv_parity(
        GgmlType::IQ4_NL,
        QK4_NL,
        BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl,
        test_only_dequantize_iq4_nl,
        4,
        176,
        704,
        0xAD22_004F_D002,
    );
}
