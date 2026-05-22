//! ADR-033 §Pi Task #16 — GPU↔CPU parity for IQ4_XS mv_id (MoE-routed) kernel.
//!
//! Validates `kernel_mul_mv_id_iq4_xs_f32` (sibling to the dense
//! `kernel_mul_mv_iq4_xs_f32` shipped at mlx-native@59b0311). This is the
//! critical decode-time hot path for Qwen MoE expert tensors when an
//! apex-quality / apex-i-quality GGUF is served by hf2q.
//!
//! Mantra: code + test == truth. The CPU reference uses
//! `test_only_dequantize_iq4_xs` to dequant per-expert slabs and run
//! a host-side dot product per (token, slot, expert_id).

use mlx_native::gguf::{test_only_dequantize_iq4_xs, test_only_kvalues_iq4_nl};
use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxDevice,
};

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

/// Lossy-but-format-correct IQ4_XS reference quantizer — same as the
/// dense mv test. Used to populate per-expert weight slabs.
fn ref_quantize_iq4_xs(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert!(row.len() % QK_K == 0);
    let mut out = Vec::with_capacity((row.len() / QK_K) * BLOCK_IQ4_XS_BYTES);
    for super_chunk in row.chunks(QK_K) {
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
        let d = -max_scale / 32.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut scales_h: u16 = 0;
        let mut scales_l = [0u8; 4];
        let mut qs = [0u8; QK_K / 2];

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
            let sub_chunk = &super_chunk[ib * SUB..(ib + 1) * SUB];
            let mut l_buf = [0u8; SUB];
            for j in 0..SUB {
                l_buf[j] = nearest_codebook(idl * sub_chunk[j]);
            }
            let qs_sub = &mut qs[16 * ib..16 * (ib + 1)];
            for j in 0..16 {
                qs_sub[j] = l_buf[j] | (l_buf[16 + j] << 4);
            }
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

fn cpu_reference_matmul(
    weights_bytes: &[u8],
    per_expert_bytes: usize,
    n: usize,
    k: usize,
    n_experts: usize,
    blocks_per_row: usize,
    input: &[f32],
    ids: &[u32],
    n_tokens: usize,
    top_k: usize,
) -> Vec<f32> {
    assert_eq!(weights_bytes.len(), n_experts * per_expert_bytes);
    assert_eq!(per_expert_bytes, n * blocks_per_row * BLOCK_IQ4_XS_BYTES);
    let total_rows = n_tokens * top_k;
    let mut out = vec![0.0_f32; total_rows * n];
    let mut row_buf = vec![0.0_f32; k];
    for t in 0..n_tokens {
        let input_row = &input[t * k..(t + 1) * k];
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;
            let expert_w = &weights_bytes[expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            for col in 0..n {
                let row_bytes = &expert_w
                    [col * blocks_per_row * BLOCK_IQ4_XS_BYTES..(col + 1) * blocks_per_row * BLOCK_IQ4_XS_BYTES];
                test_only_dequantize_iq4_xs(row_bytes, &mut row_buf).unwrap();
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

fn run_iq4_xs_mv_id_parity(
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    seed: u64,
    tol: f32,
) {
    assert_eq!(k % QK_K, 0);
    let blocks_per_row = k / QK_K;
    let per_expert_bytes = n * blocks_per_row * BLOCK_IQ4_XS_BYTES;

    let mut state = seed;
    let mut stacked_bytes = Vec::with_capacity(n_experts * per_expert_bytes);
    for _expert in 0..n_experts {
        for _row in 0..n {
            let mut row_f32 = vec![0.0_f32; k];
            for v in row_f32.iter_mut() {
                *v = random_pm1(&mut state) * 0.5;
            }
            stacked_bytes.extend(ref_quantize_iq4_xs(&row_f32));
        }
    }
    assert_eq!(stacked_bytes.len(), n_experts * per_expert_bytes);

    let mut input_data = vec![0.0_f32; n_tokens * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    let total_rows = n_tokens * top_k;
    let mut ids = vec![0_u32; total_rows];
    for v in ids.iter_mut() {
        *v = (xs64(&mut state) as u32) % (n_experts as u32);
    }

    let cpu_out = cpu_reference_matmul(
        &stacked_bytes,
        per_expert_bytes,
        n,
        k,
        n_experts,
        blocks_per_row,
        &input_data,
        &ids,
        n_tokens,
        top_k,
    );

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
        ggml_type: GgmlType::IQ4_XS,
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
    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let denom = c.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        assert!(
            abs_err <= tol || rel_err <= tol,
            "IQ4_XS mv_id mismatch at idx {i}: GPU {g} vs CPU {c} (abs {abs_err}, rel {rel_err}, tol {tol})"
        );
    }
    eprintln!(
        "[adr-033 §Pi IQ4_XS mv_id parity] max_abs_err={max_abs_err:.6e} n_tokens={n_tokens} top_k={top_k} n_experts={n_experts} n={n} k={k}"
    );
}

#[test]
fn adr033_pi_iq4_xs_mv_id_parity_small() {
    // Minimal: 1 token, top_k=2, 4 experts, n=16, k=256.
    run_iq4_xs_mv_id_parity(1, 2, 4, 16, QK_K, 0xAD33_1D4F_D001, 1e-3);
}

#[test]
fn adr033_pi_iq4_xs_mv_id_parity_realistic() {
    // 2 tokens × top_k=4 = 8 routed rows; 16 experts; n=64 rows; k=512.
    run_iq4_xs_mv_id_parity(2, 4, 16, 64, QK_K * 2, 0xAD33_1D4F_D002, 5e-3);
}

#[test]
fn adr033_pi_iq4_xs_mv_id_parity_qwen_shape() {
    // Qwen 3.5 35B-A3B routed-expert shape per dispatch: 1 token, top_k=8,
    // 256 experts, ffn_down_exps has inner dim K=512 (= moe_intermediate_size).
    // Output dim = hidden_size=2048 — large to keep CPU reference under 1 min.
    // Use smaller shape but matching ratio for the regression test.
    run_iq4_xs_mv_id_parity(1, 8, 16, 64, 512, 0xAD33_1D4F_D003, 5e-3);
}
