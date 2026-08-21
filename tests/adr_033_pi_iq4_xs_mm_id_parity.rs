//! ADR-033 §Pi Task #20 — IQ4_XS mm_id (batched-prefill) parity vs mv_id.
//!
//! Validates `kernel_mul_mm_id_iq4_xs_f32` shipped at
//! `mlx-native/src/shaders/quantized_matmul_id_mm.metal`. This is the
//! batched matrix-matrix MoE kernel that unblocks hf2q-native prefill
//! of IQ4_XS apex-i-quality GGUFs at m > 8 tokens (apex-i-quality
//! files were previously serveable only via the peer engine because hf2q's
//! own prefill errored out when the dispatcher hit IQ4_XS).
//!
//! mv_id (the decode-path single-token-per-call variant) is already
//! parity-tested in `adr_033_pi_iq4_xs_mv_id_gpu_parity.rs`. Comparing
//! mm_id output against mv_id output at the SAME shape isolates mm_id
//! as the only variable — bit-equality isn't expected (mv_id uses
//! flash_attn-style register loops, mm_id uses simdgroup MMA tiles),
//! but the outputs must agree to numerical tolerance for the shipped
//! correctness claim to hold.
//!
//! Reuses the IQ4_XS reference quantizer + dequantize helper from the
//! mv_id parity test by importing the same module-private constants.

use mlx_native::gguf::test_only_kvalues_iq4_nl;
use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_for_test, quantized_matmul_id_ggml, GgmlIdMmDispatchParams,
};
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

/// IQ4_XS reference quantizer — same as the mv_id parity test
/// (`adr_033_pi_iq4_xs_mv_id_gpu_parity.rs`). Format-correct but
/// lossy; sufficient to populate per-expert weight slabs that both
/// mv_id and mm_id kernels then dequantize identically.
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

/// mm_id-vs-mv_id parity at the same shape. mv_id is the reference
/// (proven correct via the mv_id parity test); mm_id is the
/// kernel-under-test.
fn run_iq4_xs_mm_id_parity(
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    seed: u64,
    tol_abs: f32,
    tol_rel: f32,
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
    // Production MoE routing picks the top_k *distinct* highest-scoring
    // experts per token (real routers do top-k selection over distinct
    // expert scores). The mm_id kernel's `hids` buffer is sized
    // `[n_experts, n_tokens]` accordingly — each token contributes ≤ 1
    // to any single expert's routed list. Generate distinct-per-token
    // ids via a Fisher-Yates partial shuffle over [0, n_experts).
    {
        let mut pool = vec![0_u32; n_experts];
        for t in 0..n_tokens {
            for j in 0..n_experts {
                pool[j] = j as u32;
            }
            // Partial shuffle: pick top_k unique experts for this token.
            for j in 0..top_k.min(n_experts) {
                let r = (xs64(&mut state) as usize) % (n_experts - j);
                let pick = pool[j + r];
                pool[j + r] = pool[j];
                pool[j] = pick;
                ids[t * top_k + j] = pick;
            }
            // If top_k > n_experts (shouldn't happen in practice), pad
            // with expert 0. mm_id only supports top_k ∈ {1, 8} so this
            // path is dormant.
            for j in n_experts..top_k {
                ids[t * top_k + j] = 0;
            }
        }
    }

    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();

    // ---- Shared input buffers ----
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

    // ---- Reference: mv_id (already proven correct vs CPU) ----
    let mut mv_id_out = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    {
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
        quantized_matmul_id_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut mv_id_out,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // ---- Under test: mm_id (the new IQ4_XS port) ----
    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::IQ4_XS,
    };
    let mut htpe_buf = device
        .alloc_buffer(dispatch.htpe_bytes(), DType::U32, vec![n_experts])
        .unwrap();
    // Zero-init htpe — `map0` may accumulate per-expert counts via
    // atomic_increment, requiring a zero starting state. alloc_buffer
    // doesn't guarantee zero-init.
    {
        let s = htpe_buf.as_mut_slice::<u32>().unwrap();
        for v in s.iter_mut() {
            *v = 0;
        }
    }
    let mut hids_buf = device
        .alloc_buffer(
            dispatch.hids_bytes(),
            DType::U32,
            vec![n_experts, n_tokens],
        )
        .unwrap();
    let mut mm_id_out = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    {
        let mut encoder = device.command_encoder().unwrap();
        dispatch_id_mm_for_test(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut htpe_buf,
            &mut hids_buf,
            &mut mm_id_out,
            &dispatch,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // ---- Compare ----
    let mv_out: &[f32] = mv_id_out.as_slice().unwrap();
    let mm_out: &[f32] = mm_id_out.as_slice().unwrap();
    assert_eq!(mv_out.len(), mm_out.len());
    let mut max_abs_err = 0.0_f32;
    let mut max_rel_err = 0.0_f32;
    for (i, (mv, mm)) in mv_out.iter().zip(mm_out.iter()).enumerate() {
        let abs_err = (mv - mm).abs();
        let denom = mv.abs().max(1.0);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
        assert!(
            abs_err <= tol_abs || rel_err <= tol_rel,
            "IQ4_XS mm_id vs mv_id mismatch at idx {i}: mv_id {mv} vs mm_id {mm} \
             (abs {abs_err}, rel {rel_err}, tol_abs {tol_abs}, tol_rel {tol_rel})"
        );
    }
    eprintln!(
        "[adr-033 §Pi Task #20 IQ4_XS mm_id parity] n_tokens={n_tokens} top_k={top_k} \
         n_experts={n_experts} n={n} k={k} max_abs_err={max_abs_err:.6e} \
         max_rel_err={max_rel_err:.6e}"
    );
}

// NOTE on top_k constraint: `kernel_mul_mm_id_map0_ne20_<N>` is only
// instantiated for N=1 and N=8 (see quantized_matmul_id_mm.metal). The
// mm_id dispatch path therefore requires top_k ∈ {1, 8}. Tests below
// honor this constraint; arbitrary top_k still works via the mv_id path.

#[test]
fn adr033_pi_task20_iq4_xs_mm_id_parity_small_batch_top_k1() {
    // mm_id path is engaged at m >= ~32 per the dispatch fast/slow split.
    // n_tokens=32, top_k=1 = 32 routed rows; 4 experts; n=16; k=256.
    run_iq4_xs_mm_id_parity(32, 1, 4, 16, QK_K, 0xAD33_2014_D001, 1e-3, 5e-3);
}

// Regression guard — mm_id at top_k=8 production shape should write
// EVERY output row (never leave any all-zero). Pre-fix (when the test
// used random per-slot ids violating the production-distinct-per-token
// invariant), this surfaced ~45% of output rows as all-zero, which
// guided the root-cause investigation. Post-fix (distinct routing
// matching production MoE), zero count should be 0.
#[test]
fn adr033_pi_task20_iq4_xs_mm_id_no_zero_rows() {
    let n_tokens = 64usize;
    let top_k = 8usize;
    let n_experts = 8usize;
    let n = 64usize;
    let k = QK_K;
    let seed = 0xAD33_2014_D9009u64;

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

    let mut input_data = vec![0.0_f32; n_tokens * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }
    let total_rows = n_tokens * top_k;
    let mut ids = vec![0_u32; total_rows];
    // Production MoE routing picks the top_k *distinct* highest-scoring
    // experts per token (real routers do top-k selection over distinct
    // expert scores). The mm_id kernel's `hids` buffer is sized
    // `[n_experts, n_tokens]` accordingly — each token contributes ≤ 1
    // to any single expert's routed list. Generate distinct-per-token
    // ids via a Fisher-Yates partial shuffle over [0, n_experts).
    {
        let mut pool = vec![0_u32; n_experts];
        for t in 0..n_tokens {
            for j in 0..n_experts {
                pool[j] = j as u32;
            }
            // Partial shuffle: pick top_k unique experts for this token.
            for j in 0..top_k.min(n_experts) {
                let r = (xs64(&mut state) as usize) % (n_experts - j);
                let pick = pool[j + r];
                pool[j + r] = pool[j];
                pool[j] = pick;
                ids[t * top_k + j] = pick;
            }
            // If top_k > n_experts (shouldn't happen in practice), pad
            // with expert 0. mm_id only supports top_k ∈ {1, 8} so this
            // path is dormant.
            for j in n_experts..top_k {
                ids[t * top_k + j] = 0;
            }
        }
    }

    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U8, vec![stacked_bytes.len()])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked_bytes);
    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);
    let mut ids_buf = device
        .alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::IQ4_XS,
    };
    let mut htpe_buf = device.alloc_buffer(dispatch.htpe_bytes(), DType::U32, vec![n_experts]).unwrap();
    {
        let s = htpe_buf.as_mut_slice::<u32>().unwrap();
        for v in s.iter_mut() { *v = 0; }
    }
    let mut hids_buf = device.alloc_buffer(dispatch.hids_bytes(), DType::U32, vec![n_experts, n_tokens]).unwrap();
    let mut mm_id_out = device.alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n]).unwrap();
    {
        let mut encoder = device.command_encoder().unwrap();
        dispatch_id_mm_for_test(
            &mut encoder, &mut registry, &device,
            &input_buf, &weight_buf, &ids_buf,
            &mut htpe_buf, &mut hids_buf, &mut mm_id_out,
            &dispatch,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
    }

    let out: &[f32] = mm_id_out.as_slice().unwrap();
    let mut zero_rows: Vec<usize> = Vec::new();
    for r in 0..total_rows {
        let row = &out[r * n..(r + 1) * n];
        let all_zero = row.iter().all(|&v| v == 0.0);
        if all_zero {
            zero_rows.push(r);
        }
    }
    eprintln!(
        "[adr-033 §Pi Task #20 no-zero-rows guard] {} of {} output rows all-zero",
        zero_rows.len(),
        total_rows
    );
    assert!(
        zero_rows.is_empty(),
        "mm_id at top_k=8 left {} of {} output rows all-zero — regression of the \
         IQ4_XS top_k=8 routing-invariant gap (see commit history for full RCA). \
         First 10: {:?}",
        zero_rows.len(),
        total_rows,
        &zero_rows[..zero_rows.len().min(10)]
    );
}

// ADR-033 §Pi Task #20 — IQ4_XS mm_id at top_k=8 production shape.
//
// PRE-FIX: this test was #[ignore]'d because mm_id at top_k=8 silently
// failed to write certain output rows for IQ4_XS. Localized root cause:
// `dispatch_id_mm_pooled`'s mm grid_x was sized for `n_tokens` routed
// rows per expert, but with top_k > 1 the per-expert routed count
// (htpe[im]) can exceed n_tokens when routing is uneven. Output rows
// beyond grid_x*NR1 stayed at the buffer's initial value (0). Q5_K
// passed by lucky routing distribution; IQ4_XS happened to trigger it.
//
// FIX: grid_x now uses `n_tokens * top_k` (worst-case all routed rows
// to one expert). The kernel's `if (r1 >= neh1) return` early-exit
// handles unused tiles correctly.
//
// This test pins the fix at production-Qwen-MoE-style shape.
#[test]
fn adr033_pi_task20_iq4_xs_mm_id_parity_matches_q5_k_shape() {
    // Cloned exactly from `adr022_phase2_q5_k_mm_id_parity_prefill_path` —
    // n_tokens=64, top_k=8, n_experts=8, n=64, k=256.
    run_iq4_xs_mm_id_parity(64, 8, 8, 64, QK_K, 0xAD33_2014_D002, 1e-3, 5e-3);
}
