//! ADR-022 Phase 1 P1.7 — GPU↔CPU parity for `mul_mv_ext` r1 family
//! (Q5_1 + IQ4_NL × r1ptg ∈ {2, 3, 4, 5}).
//!
//! Tests every kernel landed by the Phase-1 P1.7 commit:
//!   - 2 weight types × 4 r1ptg widths = 8 kernels.
//!
//! Each test case constructs a synthetic `[N, K]` quantized weight and a
//! `[M, K]` f32 input, runs CPU reference matmul against runtime
//! dequantize, then dispatches the GPU kernel and asserts the outputs
//! match within F32-accumulator tolerance.

use mlx_native::gguf::{test_only_dequantize_iq4_nl, test_only_dequantize_q5_1, test_only_kvalues_iq4_nl};
use mlx_native::{
    mul_mv_ext_dispatch, DType, GgmlType, KernelRegistry, MlxDevice, MulMvExtParams,
};

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

fn ref_quantize_q5_1_block(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), QK5_1);
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in row {
        if v < min { min = v; }
        if v > max { max = v; }
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
    for chunk in row.chunks(QK5_1) { out.extend(ref_quantize_q5_1_block(chunk)); }
    out
}

fn ref_quantize_iq4_nl_block(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert_eq!(row.len(), QK4_NL);
    let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let d = if max_abs == 0.0 { 0.0 } else { max_abs / 113.0 };
    let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
    let nearest = |target: f32| -> u8 {
        let mut best_idx = 0u8;
        let mut best_err = f32::MAX;
        for (idx, &k) in kv.iter().enumerate() {
            let err = (target - k as f32).abs();
            if err < best_err { best_err = err; best_idx = idx as u8; }
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
    for chunk in row.chunks(QK4_NL) { out.extend(ref_quantize_iq4_nl_block(chunk)); }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_mv_ext_parity(
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

    // Synthetic weight: per-row F32 → quantize via reference quantizer.
    let mut weights_bytes = Vec::with_capacity(total_weight_bytes);
    for _ in 0..n {
        let mut row_f32 = vec![0.0_f32; k];
        for v in row_f32.iter_mut() { *v = random_pm1(&mut state) * 0.5; }
        weights_bytes.extend(quantize(&row_f32));
    }
    assert_eq!(weights_bytes.len(), total_weight_bytes);

    // Synthetic input: m × k f32.
    let mut input_data = vec![0.0_f32; m * k];
    for v in input_data.iter_mut() { *v = random_pm1(&mut state); }

    // CPU reference: dequantize each weight row + dot with each input row.
    let mut cpu_out = vec![0.0_f32; m * n];
    let mut row_buf = vec![0.0_f32; k];
    for row_n in 0..n {
        let row_data = &weights_bytes[row_n * row_bytes..(row_n + 1) * row_bytes];
        dequantize(row_data, &mut row_buf).unwrap();
        for row_m in 0..m {
            let in_row = &input_data[row_m * k..(row_m + 1) * k];
            let mut acc = 0.0_f32;
            for kk in 0..k { acc += row_buf[kk] * in_row[kk]; }
            cpu_out[row_m * n + row_n] = acc;
        }
    }

    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();

    let mut weight_buf = device
        .alloc_buffer(weights_bytes.len(), DType::U8, vec![weights_bytes.len()])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&weights_bytes);

    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);

    let mut output_buf = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m * n])
        .unwrap();
    for v in output_buf.as_mut_slice::<f32>().unwrap().iter_mut() { *v = 0.0; }

    let params = MulMvExtParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        batch: 1,
        ggml_type,
    };

    let mut encoder = device.command_encoder().unwrap();
    mul_mv_ext_dispatch(
        &mut encoder, &mut registry, &device,
        &weight_buf, &input_buf, &mut output_buf, &params,
    ).unwrap();
    encoder.commit_and_wait().unwrap();

    let gpu_out: &[f32] = output_buf.as_slice().unwrap();
    eprintln!(
        "[adr-022 P1.7 {:?} m={} n={} k={}] first GPU: {:?}",
        ggml_type, m, n, k, &gpu_out[..gpu_out.len().min(8)]
    );
    eprintln!(
        "[adr-022 P1.7 {:?} m={} n={} k={}] first CPU: {:?}",
        ggml_type, m, n, k, &cpu_out[..cpu_out.len().min(8)]
    );

    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let denom = c.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        // Tolerance: 5e-3 absorbs F32 accumulator reorder noise across
        // K=128..512 multiplications and Q5_1 / IQ4_NL block-quant rounding.
        assert!(
            abs_err <= 5e-3 || rel_err <= 5e-3,
            "{:?} mv_ext mismatch at idx {i}: GPU {g} vs CPU {c} (abs {abs_err}, rel {rel_err}, m={m} n={n} k={k})",
            ggml_type
        );
    }
    eprintln!(
        "[adr-022 P1.7 {:?} m={} n={} k={}] PASS max_abs_err={:.6e} max_rel_err={:.6e}",
        ggml_type, m, n, k, max_abs_err, max_rel_err
    );
}

// ----- Q5_1 — r1∈{2,3,4,5} (mapped from m via pick_r1ptg) -----

#[test]
fn adr022_p17_q5_1_mv_ext_m2() {
    run_mv_ext_parity(
        GgmlType::Q5_1, QK5_1, BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1, test_only_dequantize_q5_1,
        2, 32, 128, 0xAD22_0511_E002,
    );
}

#[test]
fn adr022_p17_q5_1_mv_ext_m3() {
    run_mv_ext_parity(
        GgmlType::Q5_1, QK5_1, BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1, test_only_dequantize_q5_1,
        3, 32, 128, 0xAD22_0511_E003,
    );
}

#[test]
fn adr022_p17_q5_1_mv_ext_m4() {
    run_mv_ext_parity(
        GgmlType::Q5_1, QK5_1, BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1, test_only_dequantize_q5_1,
        4, 32, 128, 0xAD22_0511_E004,
    );
}

#[test]
fn adr022_p17_q5_1_mv_ext_m5() {
    run_mv_ext_parity(
        GgmlType::Q5_1, QK5_1, BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1, test_only_dequantize_q5_1,
        5, 32, 128, 0xAD22_0511_E005,
    );
}

// ----- IQ4_NL — r1∈{2,3,4,5} -----

#[test]
fn adr022_p17_iq4_nl_mv_ext_m2() {
    run_mv_ext_parity(
        GgmlType::IQ4_NL, QK4_NL, BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl, test_only_dequantize_iq4_nl,
        2, 32, 128, 0xAD22_004F_E002,
    );
}

#[test]
fn adr022_p17_iq4_nl_mv_ext_m3() {
    run_mv_ext_parity(
        GgmlType::IQ4_NL, QK4_NL, BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl, test_only_dequantize_iq4_nl,
        3, 32, 128, 0xAD22_004F_E003,
    );
}

#[test]
fn adr022_p17_iq4_nl_mv_ext_m4() {
    run_mv_ext_parity(
        GgmlType::IQ4_NL, QK4_NL, BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl, test_only_dequantize_iq4_nl,
        4, 32, 128, 0xAD22_004F_E004,
    );
}

#[test]
fn adr022_p17_iq4_nl_mv_ext_m5() {
    run_mv_ext_parity(
        GgmlType::IQ4_NL, QK4_NL, BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl, test_only_dequantize_iq4_nl,
        5, 32, 128, 0xAD22_004F_E005,
    );
}

// ----- Realistic Gemma4-shape sanity: K=2816 (hidden), N=128 (router/expert dim).
//       At m=2, pick_nxpsg returns 16 (256-divisible + M<3). At m=3, falls
//       back to 8. Verifies both nxpsg branches.

#[test]
fn adr022_p17_q5_1_realistic_k2816() {
    run_mv_ext_parity(
        GgmlType::Q5_1, QK5_1, BLOCK_Q5_1_BYTES,
        ref_quantize_q5_1, test_only_dequantize_q5_1,
        2, 128, 2816, 0xAD22_0511_E2816,
    );
}

#[test]
fn adr022_p17_iq4_nl_realistic_k2816() {
    run_mv_ext_parity(
        GgmlType::IQ4_NL, QK4_NL, BLOCK_IQ4_NL_BYTES,
        ref_quantize_iq4_nl, test_only_dequantize_iq4_nl,
        2, 128, 2816, 0xAD22_004F_E2816,
    );
}
