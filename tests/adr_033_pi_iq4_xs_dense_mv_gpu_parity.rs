//! ADR-033 §Pi — GPU↔CPU parity for IQ4_XS DENSE mv kernel
//! (`kernel_mul_mv_iq4_xs_f32`).
//!
//! Sibling to `adr_022_phase1_dense_mv_gpu_parity.rs` for IQ4_NL; this
//! test exercises the IQ4_XS super-block variant (256-element block,
//! 8 × 32-element sub-blocks with 6-bit per-sub-block scales) added
//! 2026-05-22 to support apex-i-quality MoE serving.
//!
//! The CPU reference uses `test_only_dequantize_iq4_xs`, which is
//! independently validated against canonical `dequantize_row_iq4_xs`
//! via the hf2q-side `byte_cmp_noim` / `byte_cmp_im` tests on
//! `tests/fixtures/ggml_quants/iq4_xs_512_*` fixtures.

use mlx_native::gguf::{test_only_dequantize_iq4_xs, test_only_kvalues_iq4_nl};
use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice};

const QK_K: usize = 256;
const BLOCK_IQ4_XS_BYTES: usize = 136;
const SUB: usize = 32;

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

/// Minimal IQ4_XS quantizer for the parity test — deliberately NOT the
/// canonical iterative version (that lives in hf2q). Just picks a
/// per-sub-block scale `dl = max_abs / 113` (the codebook's max-abs
/// value) and emits canonical-format bytes. Lossy but byte-format
/// correct, which is all the GPU mv kernel cares about (it just
/// reads the bytes and applies them per the spec).
fn ref_quantize_iq4_xs(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert!(row.len() % QK_K == 0, "row {} not multiple of QK_K", row.len());
    let mut out = Vec::with_capacity((row.len() / QK_K) * BLOCK_IQ4_XS_BYTES);
    for super_chunk in row.chunks(QK_K) {
        // Compute per-sub-block "scale" amax + global max-scale.
        let mut sub_scales = [0.0f32; 8];
        let mut max_scale: f32 = 0.0;
        for ib in 0..8 {
            let sub = &super_chunk[ib * SUB..(ib + 1) * SUB];
            let amax = sub.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let d_sub = if amax == 0.0 { 0.0 } else { -amax / kv[0] as f32 };
            sub_scales[ib] = d_sub;
            if d_sub.abs() > max_scale.abs() {
                max_scale = d_sub;
            }
        }

        // Super-block scale d = -max_scale / 32 (canonical formula).
        let d = -max_scale / 32.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Per-sub-block 6-bit signed scale l ∈ [-32, 31].
        let mut scales_h: u16 = 0;
        let mut scales_l = [0u8; 4];
        let mut qs = [0u8; QK_K / 2]; // 128

        let nearest_codebook = |t: f32| -> u8 {
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

        for ib in 0..8 {
            let l_raw = (id * sub_scales[ib]).round() as i32;
            let l_signed = l_raw.clamp(-32, 31);
            let dl = d * (l_signed as f32);
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

            // Re-fill L_b using the quantized sub-block scale.
            let sub_chunk = &super_chunk[ib * SUB..(ib + 1) * SUB];
            let mut l_buf = [0u8; SUB];
            for j in 0..SUB {
                l_buf[j] = nearest_codebook(idl * sub_chunk[j]);
            }
            // Nibble-pack into qs[16*ib..16*(ib+1)] — bottom half low nibble,
            // top half high nibble (mirrors canonical at ggml-quants.c:4898).
            let qs_sub = &mut qs[16 * ib..16 * (ib + 1)];
            for j in 0..16 {
                qs_sub[j] = l_buf[j] | (l_buf[16 + j] << 4);
            }

            // Pack 6-bit scale: l = l_signed + 32 ∈ [0, 63].
            let l_unsigned = (l_signed + 32) as u8;
            let l_l = l_unsigned & 0xf;
            let l_h = l_unsigned >> 4;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h |= (l_h as u16) << (2 * ib);
        }

        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        out.extend_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&scales_l);
        out.extend_from_slice(&qs);
    }
    out
}

fn cpu_dense_matmul_iq4_xs(weights_bytes: &[u8], n: usize, k: usize, input: &[f32], m: usize) -> Vec<f32> {
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * BLOCK_IQ4_XS_BYTES;
    assert_eq!(weights_bytes.len(), n * row_bytes);
    let mut out = vec![0.0_f32; m * n];
    let mut row_buf = vec![0.0_f32; k];
    for mi in 0..m {
        let in_row = &input[mi * k..(mi + 1) * k];
        for col in 0..n {
            let row = &weights_bytes[col * row_bytes..(col + 1) * row_bytes];
            test_only_dequantize_iq4_xs(row, &mut row_buf).unwrap();
            let mut sum = 0.0_f32;
            for kk in 0..k {
                sum += row_buf[kk] * in_row[kk];
            }
            out[mi * n + col] = sum;
        }
    }
    out
}

fn run_iq4_xs_dense_mv_parity(m: usize, n: usize, k: usize, seed: u64, tol: f32) {
    assert_eq!(k % QK_K, 0);
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * BLOCK_IQ4_XS_BYTES;
    let total_weight_bytes = n * row_bytes;

    let mut state = seed;
    let mut weights_bytes = Vec::with_capacity(total_weight_bytes);
    for _ in 0..n {
        let mut row_f32 = vec![0.0_f32; k];
        for v in row_f32.iter_mut() {
            *v = random_pm1(&mut state) * 0.5;
        }
        weights_bytes.extend(ref_quantize_iq4_xs(&row_f32));
    }
    assert_eq!(weights_bytes.len(), total_weight_bytes);

    let mut input_data = vec![0.0_f32; m * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    let cpu_out = cpu_dense_matmul_iq4_xs(&weights_bytes, n, k, &input_data, m);

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
        ggml_type: GgmlType::IQ4_XS,
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
            abs_err <= tol || rel_err <= tol,
            "IQ4_XS dense mv mismatch at idx {i}: GPU {g} vs CPU {c} (abs {abs_err}, rel {rel_err}, tol {tol})"
        );
    }
    eprintln!(
        "[adr-033 §Pi IQ4_XS dense mv parity] max_abs_err={max_abs_err:.6e} m={m} n={n} k={k}"
    );
}

#[test]
fn adr033_pi_iq4_xs_dense_mv_parity_small() {
    // Minimal: 1 row × 16 cols × 256 K (1 super-block per row).
    run_iq4_xs_dense_mv_parity(1, 16, QK_K, 0xAD33_004F_D001, 1e-3);
}

#[test]
fn adr033_pi_iq4_xs_dense_mv_parity_realistic() {
    // Multi-row, multi-super-block. Tolerance slightly higher to allow
    // F32-accumulator reorder across simdgroup reductions.
    run_iq4_xs_dense_mv_parity(4, 64, QK_K * 4, 0xAD33_004F_D002, 5e-3);
}

#[test]
fn adr033_pi_iq4_xs_dense_mv_parity_qwen_shape() {
    // Approximate Qwen 3.5 35B-A3B routed-expert tensor shape per
    // sub-block: K = 2048 (hidden_size), N = 64 (subset of experts).
    run_iq4_xs_dense_mv_parity(1, 64, 2048, 0xAD33_004F_D003, 5e-3);
}
