//! ADR-022 Phase 2 — Q5_K dense mv + mm + mm_tensor parity vs CPU reference.
//!
//! Validates the Phase-2 dense kernels:
//!   - kernel_mul_mv_q5_K_f32         (dense mv, m≤8 path)
//!   - kernel_mul_mm_q5_K_f32         (dense mm, m>8 path, simdgroup MMA)
//!   - kernel_mul_mm_q5_K_tensor_f32  (dense mm, m>8 path, M3+ tensor cores)
//!
//! CPU reference uses `mlx_native::gguf::dequantize_to_f32` to dequant
//! the synthetic weight blocks then computes a row-major matmul. The
//! same Q5_K bytes flow through both paths so quantizer fidelity cancels.

use mlx_native::{
    quantized_matmul_ggml, DType, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_Q5_K_BYTES: usize = 176;

// xorshift* deterministic random
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

/// Reference Q5_K quantizer — same body as the mm_id parity test.
/// Encodes a 256-element row to the GGUF Q5_K block format.
fn ref_quantize_q5_k_block(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), QK_K);
    let mut sub_d = [0f32; 8];
    let mut sub_m = [0f32; 8];
    for sb in 0..8 {
        let chunk = &row[sb * 32..(sb + 1) * 32];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in chunk {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        sub_d[sb] = (max - min) / 31.0;
        sub_m[sb] = -min;
    }
    let max_scale = sub_d.iter().cloned().fold(0.0_f32, f32::max);
    let max_min = sub_m.iter().cloned().fold(0.0_f32, f32::max);
    let d_super = if max_scale == 0.0 { 0.0 } else { max_scale / 63.0 };
    let dmin_super = if max_min == 0.0 { 0.0 } else { max_min / 63.0 };

    let mut scales_packed = [0u8; 12];
    for sb in 0..8 {
        let s = if d_super == 0.0 { 0u8 } else {
            ((sub_d[sb] / d_super).round() as i32).clamp(0, 63) as u8
        };
        let m = if dmin_super == 0.0 { 0u8 } else {
            ((sub_m[sb] / dmin_super).round() as i32).clamp(0, 63) as u8
        };
        if sb < 4 {
            scales_packed[sb] = (scales_packed[sb] & 0xC0) | (s & 0x3F);
            scales_packed[4 + sb] = (scales_packed[4 + sb] & 0xC0) | (m & 0x3F);
        } else {
            let i = sb - 4;
            scales_packed[8 + i] = (s & 0x0F) | ((m & 0x0F) << 4);
            scales_packed[i] = (scales_packed[i] & 0x3F) | (((s >> 4) & 0x03) << 6);
            scales_packed[4 + i] = (scales_packed[4 + i] & 0x3F) | (((m >> 4) & 0x03) << 6);
        }
    }

    let mut qs = [0u8; QK_K / 2];
    let mut qh = [0u8; QK_K / 8];
    for sb in 0..8 {
        let chunk = &row[sb * 32..(sb + 1) * 32];
        let inv_d = if sub_d[sb] == 0.0 { 0.0 } else { 1.0 / sub_d[sb] };
        for j in 0..32 {
            let q = ((chunk[j] + sub_m[sb]) * inv_d).round().clamp(0.0, 31.0) as u32;
            let half = sb / 4;
            let in_half = sb % 4;
            let nib = (q & 0x0F) as u8;
            let pos_base = half * 64 + (in_half % 2) * 16 + (j % 16);
            let pos = if (j / 16) == 0 { pos_base } else { pos_base + 32 };
            if (in_half / 2) == 0 {
                qs[pos] = (qs[pos] & 0xF0) | nib;
            } else {
                qs[pos] = (qs[pos] & 0x0F) | (nib << 4);
            }
            let qh_idx = j;
            let qh_bit = sb;
            if (q & 0x10) != 0 {
                qh[qh_idx] |= 1 << qh_bit;
            }
        }
    }

    let mut out = Vec::with_capacity(BLOCK_Q5_K_BYTES);
    out.extend_from_slice(&half::f16::from_f32(d_super).to_bits().to_le_bytes());
    out.extend_from_slice(&half::f16::from_f32(dmin_super).to_bits().to_le_bytes());
    out.extend_from_slice(&scales_packed);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&qs);
    assert_eq!(out.len(), BLOCK_Q5_K_BYTES);
    out
}

fn ref_quantize_q5_k(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len() % QK_K, 0);
    let mut out = Vec::with_capacity((row.len() / QK_K) * BLOCK_Q5_K_BYTES);
    for chunk in row.chunks(QK_K) {
        out.extend(ref_quantize_q5_k_block(chunk));
    }
    out
}

fn run_dense_q5_k_parity(m: usize, n: usize, k: usize, seed: u64, tol: f32) {
    assert_eq!(k % QK_K, 0);
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * BLOCK_Q5_K_BYTES;
    let total_weight_bytes = n * row_bytes;

    let mut state = seed;

    let mut weights_bytes = Vec::with_capacity(total_weight_bytes);
    for _ in 0..n {
        let mut row_f32 = vec![0.0_f32; k];
        for v in row_f32.iter_mut() {
            *v = random_pm1(&mut state) * 0.5;
        }
        weights_bytes.extend(ref_quantize_q5_k(&row_f32));
    }
    assert_eq!(weights_bytes.len(), total_weight_bytes);

    let mut input_data = vec![0.0_f32; m * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    // CPU reference: dequant each weight row then dot-product against each input row.
    let mut cpu_out = vec![0.0_f32; m * n];
    let mut row_buf = vec![0.0_f32; k];
    for row_n in 0..n {
        let row_data = &weights_bytes[row_n * row_bytes..(row_n + 1) * row_bytes];
        mlx_native::gguf::test_only_dequantize(row_data, GgmlType::Q5_K, &mut row_buf).unwrap();
        for row_m in 0..m {
            let in_row = &input_data[row_m * k..(row_m + 1) * k];
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += row_buf[kk] * in_row[kk];
            }
            cpu_out[row_m * n + row_n] = acc;
        }
    }

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
    for v in output_buf.as_mut_slice::<f32>().unwrap().iter_mut() {
        *v = 0.0;
    }

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q5_K,
    };

    let mut encoder = device.command_encoder().unwrap();
    quantized_matmul_ggml(
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
    eprintln!(
        "[adr-022 phase-2 Q5_K dense m={m} n={n} k={k}] first GPU: {:?}",
        &gpu_out[..gpu_out.len().min(8)]
    );
    eprintln!(
        "[adr-022 phase-2 Q5_K dense m={m} n={n} k={k}] first CPU: {:?}",
        &cpu_out[..cpu_out.len().min(8)]
    );

    let mut max_abs_err = 0.0_f32;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let denom = c.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        assert!(
            abs_err <= tol || rel_err <= tol,
            "Q5_K dense m={m} mismatch at idx {i}: GPU {g} vs CPU {c} (abs {abs_err}, rel {rel_err}, tol {tol})"
        );
    }
    eprintln!(
        "[adr-022 phase-2 Q5_K dense m={m} n={n} k={k}] PASS max_abs_err={:.6e}",
        max_abs_err
    );
}

#[test]
fn adr022_phase2_q5_k_dense_mv_decode() {
    // m=1 → mv path. Realistic Qwen 3.6 shexp shape.
    run_dense_q5_k_parity(/*m=*/ 1, /*n=*/ 512, /*k=*/ 2048, 0xAD22_05_D001, 5e-2);
}

#[test]
fn adr022_phase2_q5_k_dense_mv_small_batch() {
    // m=4 → still mv path (≤ MM_ROUTING_THRESHOLD=8).
    run_dense_q5_k_parity(/*m=*/ 4, /*n=*/ 64, /*k=*/ 256, 0xAD22_05_D002, 5e-2);
}

#[test]
fn adr022_phase2_q5_k_dense_mm_prefill() {
    // m=64 → mm path (> MM_ROUTING_THRESHOLD).
    run_dense_q5_k_parity(/*m=*/ 64, /*n=*/ 64, /*k=*/ 256, 0xAD22_05_D003, 5e-2);
}

#[test]
fn adr022_phase2_q5_k_dense_mm_realistic() {
    // m=32 (right at threshold), realistic Qwen3.6 attn_gate shape K=4096.
    run_dense_q5_k_parity(/*m=*/ 32, /*n=*/ 64, /*k=*/ 4096, 0xAD22_05_D004, 5e-2);
}
