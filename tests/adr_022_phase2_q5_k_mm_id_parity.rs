//! ADR-022 Phase 2 — Q5_K mm_id + mm_id_tensor parity vs mv_id reference.
//!
//! Validates the Q5_K kernels landed in Phase 2:
//!   - kernel_mul_mm_id_q5_K_f32          (id_mm.metal)
//!   - kernel_mul_mm_id_q5_K_tensor_f32   (id_mm_tensor.metal)
//!
//! mv_id is already proven correct (the qwen35moe path uses it for
//! decode and ships coherent inference per the iter-19 smoke test
//! against `qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf`).
//! Comparing mm_id output against mv_id at the same shape isolates
//! mm_id as the only variable.

use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_for_test, quantized_matmul_id_ggml, GgmlIdMmDispatchParams,
};
use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_Q5_K_BYTES: usize = 176;

// xorshift* deterministic random.
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

/// Reference Q5_K quantizer for synthesizing test weights from F32 rows.
///
/// Encodes a 256-element row using the same packed layout as
/// llama.cpp's `quantize_row_q5_K_ref`: per-32-element sub-block scale
/// + min, packed as 6-bit pairs in `scales[12]`; quants stored as 4-bit
/// `qs[128]` plus 1-bit `qh[32]`. The implementation here is a naive
/// per-sub-block fit (no roundtrip optimization) — sufficient for parity
/// testing at the kernel level; the resulting blocks are valid Q5_K
/// regardless of compression quality.
fn ref_quantize_q5_k_block(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), QK_K);
    let mut sub_d = [0f32; 8];
    let mut sub_m = [0f32; 8];
    for sb in 0..8 {
        let chunk = &row[sb * 32..(sb + 1) * 32];
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
        sub_d[sb] = (max - min) / 31.0;
        sub_m[sb] = -min;
    }
    let max_scale = sub_d.iter().cloned().fold(0.0_f32, f32::max);
    let max_min = sub_m.iter().cloned().fold(0.0_f32, f32::max);
    let d_super = if max_scale == 0.0 { 0.0 } else { max_scale / 63.0 };
    let dmin_super = if max_min == 0.0 { 0.0 } else { max_min / 63.0 };

    let mut scales_packed = [0u8; 12];
    for sb in 0..8 {
        let s = if d_super == 0.0 {
            0u8
        } else {
            ((sub_d[sb] / d_super).round() as i32).clamp(0, 63) as u8
        };
        let m = if dmin_super == 0.0 {
            0u8
        } else {
            ((sub_m[sb] / dmin_super).round() as i32).clamp(0, 63) as u8
        };
        // Layout per ggml-common.h:97 (`get_scale_min_k4_just2`):
        //   scales[0..4] = sub-block scales 0..3 low 6 bits
        //   scales[4..8] = sub-block mins   0..3 low 6 bits
        //   scales[8..12] = sub-block (scale[4..7], min[4..7]) interleaved
        //                  scales[8+i] = (scale[4+i] & 0x0F) | ((min[4+i] & 0x0F) << 4)
        //                  high 2 bits packed into scales[0..8] high bits
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
        let inv_d = if sub_d[sb] == 0.0 {
            0.0
        } else {
            1.0 / sub_d[sb]
        };
        for j in 0..32 {
            let q = ((chunk[j] + sub_m[sb]) * inv_d).round().clamp(0.0, 31.0) as u32;
            // Pack low 4 bits to qs at appropriate position.
            // qs layout: each 64-byte half-block (sub-blocks 0..3 or 4..7)
            // packs qs[i*2] = (q_low_block_a) | (q_low_block_b << 4) where
            // a/b are the two sub-blocks within the half. See dequant
            // formula at ggml-metal.metal:705.
            let half = sb / 4; // 0 or 1
            let in_half = sb % 4;
            let pos = half * 64 + (in_half % 2) * 16 + (j % 16);
            let nib = (q & 0x0F) as u8;
            if (in_half / 2) == 0 && (j / 16) == 0 {
                qs[pos] = (qs[pos] & 0xF0) | nib;
            } else if (in_half / 2) == 1 && (j / 16) == 0 {
                qs[pos] = (qs[pos] & 0x0F) | (nib << 4);
            } else if (in_half / 2) == 0 && (j / 16) == 1 {
                let pos2 = half * 64 + 32 + (in_half % 2) * 16 + (j % 16);
                qs[pos2] = (qs[pos2] & 0xF0) | nib;
            } else {
                let pos2 = half * 64 + 32 + (in_half % 2) * 16 + (j % 16);
                qs[pos2] = (qs[pos2] & 0x0F) | (nib << 4);
            }
            // Pack high bit to qh: bit position is sb (each sub-block's
            // bit goes to the same nibble across the 32-byte qh array).
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

#[allow(clippy::too_many_arguments)]
fn run_q5_k_mm_id_vs_mv_id(
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    seed: u64,
) {
    assert_eq!(k % QK_K, 0);
    assert!(top_k == 1 || top_k == 8);

    let blocks_per_row = k / QK_K;
    let per_expert_bytes = n * blocks_per_row * BLOCK_Q5_K_BYTES;

    let mut state = seed;
    let mut stacked_bytes = Vec::with_capacity(n_experts * per_expert_bytes);
    for _expert in 0..n_experts {
        for _row in 0..n {
            let mut row_f32 = vec![0.0_f32; k];
            for v in row_f32.iter_mut() {
                *v = random_pm1(&mut state) * 0.5;
            }
            stacked_bytes.extend(ref_quantize_q5_k(&row_f32));
        }
    }
    assert_eq!(stacked_bytes.len(), n_experts * per_expert_bytes);

    let mut input_data = vec![0.0_f32; n_tokens * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    let total_rows = n_tokens * top_k;
    let mut ids = vec![0_u32; total_rows];
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids[t * top_k + s] = ((t * 17 + s * 13 + 7) % n_experts) as u32;
        }
    }

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
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    // mv_id reference (chunked at threshold ≤ 32 to force mv path).
    let mut mv_output = vec![0.0_f32; total_rows * n];
    let mv_chunk = 32_usize.min(n_tokens);
    let mut tok_off = 0;
    while tok_off < n_tokens {
        let chunk = (n_tokens - tok_off).min(mv_chunk);
        let mut chunk_in = device
            .alloc_buffer(chunk * k * 4, DType::F32, vec![chunk * k])
            .unwrap();
        chunk_in
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&input_data[tok_off * k..(tok_off + chunk) * k]);
        let mut chunk_ids = device
            .alloc_buffer(chunk * top_k * 4, DType::U32, vec![chunk * top_k])
            .unwrap();
        chunk_ids
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&ids[tok_off * top_k..(tok_off + chunk) * top_k]);
        let mut chunk_out = device
            .alloc_buffer(chunk * top_k * n * 4, DType::F32, vec![chunk * top_k * n])
            .unwrap();
        for v in chunk_out.as_mut_slice::<f32>().unwrap().iter_mut() {
            *v = 0.0;
        }
        let chunk_params = GgmlQuantizedMatmulIdParams {
            n_tokens: chunk as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type: GgmlType::Q5_K,
        };
        let mut enc = device.command_encoder().unwrap();
        quantized_matmul_id_ggml(
            &mut enc,
            &mut registry,
            &device,
            &chunk_in,
            &weight_buf,
            &chunk_ids,
            &mut chunk_out,
            &chunk_params,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
        let chunk_slice: &[f32] = chunk_out.as_slice().unwrap();
        mv_output[tok_off * top_k * n..(tok_off + chunk) * top_k * n]
            .copy_from_slice(chunk_slice);
        tok_off += chunk;
    }

    // mm_id under test (force via dispatch_id_mm_for_test).
    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q5_K,
    };
    let mut htpe = device
        .alloc_buffer(dispatch.htpe_bytes(), DType::U32, vec![n_experts])
        .unwrap();
    let mut hids = device
        .alloc_buffer(dispatch.hids_bytes(), DType::U32, vec![n_experts, n_tokens])
        .unwrap();
    for v in htpe.as_mut_slice::<u32>().unwrap().iter_mut() {
        *v = 0;
    }
    for v in hids.as_mut_slice::<u32>().unwrap().iter_mut() {
        *v = 0;
    }

    let mut mm_output_buf = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    for v in mm_output_buf.as_mut_slice::<f32>().unwrap().iter_mut() {
        *v = 0.0;
    }

    {
        let mut enc = device.command_encoder().unwrap();
        dispatch_id_mm_for_test(
            &mut enc,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut htpe,
            &mut hids,
            &mut mm_output_buf,
            &dispatch,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    let mm_out: &[f32] = mm_output_buf.as_slice().unwrap();
    eprintln!(
        "[adr-022 phase-2 Q5_K mm_id parity] first GPU mm_id: {:?}",
        &mm_out[..mm_out.len().min(8)]
    );
    eprintln!(
        "[adr-022 phase-2 Q5_K mm_id parity] first mv_id ref:  {:?}",
        &mv_output[..mv_output.len().min(8)]
    );

    // Tolerance: 5e-2 matches the existing K-quant test_q4_0_mm_id_matches_mv_id_prefill_shape
    // precedent (test_quantized_matmul_id_mm.rs:476) for K ≥ 256 — F32
    // accumulator reorder + simdgroup partial-sum vs scalar accumulation
    // diverge measurably at K=256 with K-quant dynamic range.
    let mut max_abs_err = 0.0_f32;
    let mut first_bad = None;
    for (i, (mm, mv)) in mm_out.iter().zip(mv_output.iter()).enumerate() {
        let err = (mm - mv).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        if err > 5e-2 && first_bad.is_none() {
            first_bad = Some((i, *mm, *mv, err));
        }
    }
    if let Some((i, mm_v, mv_v, err)) = first_bad {
        let row = i / n;
        let col = i % n;
        let tok = row / top_k;
        let slot = row % top_k;
        let expert = ids[row];
        panic!(
            "Q5_K mm_id vs mv_id mismatch at idx {i} (tok {tok} slot {slot} expert {expert} col {col}): mm {mm_v} vs mv {mv_v} (err {err}, max_abs {max_abs_err})"
        );
    }
    eprintln!(
        "[adr-022 phase-2 Q5_K mm_id parity] PASS max_abs_err={:.6e}",
        max_abs_err
    );
}

#[test]
fn adr022_phase2_q5_k_mm_id_parity_prefill_path() {
    run_q5_k_mm_id_vs_mv_id(
        /*n_tokens=*/ 64,
        /*top_k=*/ 8,
        /*n_experts=*/ 8,
        /*n=*/ 64,
        /*k=*/ 256,
        0xAD22_05_00_6011,
    );
}

#[test]
fn adr022_phase2_q5_k_mm_id_parity_top_k_1() {
    run_q5_k_mm_id_vs_mv_id(
        /*n_tokens=*/ 64,
        /*top_k=*/ 1,
        /*n_experts=*/ 8,
        /*n=*/ 64,
        /*k=*/ 256,
        0xAD22_05_00_6012,
    );
}
