//! ADR-028 iter-321 — parity test for `kernel_mul_mv_id_q6_K_f32_nr2`.
//!
//! Mirrors iter-309's non-_id parity test, adapted for MoE `_id`
//! dispatch.  Asserts that the new nr0=2 variant produces output
//! within 1e-4 of the baseline single-row kernel across multiple
//! token + top_k + expert-id permutations.
//!
//! Falsifier for ADR-028 iter-321: this MUST pass for any production
//! perf bench to be load-bearing.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxDevice,
};

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

fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;
        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale { max_scale = amax; }
        }
        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else { 0 };
        }
        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                let v0 = q6[l0_base + l];
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);
                let v1 = q6[l0_base + l + 32];
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);
                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let v0 = q6[128 + l0_base + l];
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);
                let v1 = q6[128 + l0_base + l + 32];
                let v3 = q6[128 + l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);
                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[qh_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }
        buf.extend_from_slice(&ql);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(&sub_scale_int.iter().map(|&s| s as u8).collect::<Vec<_>>());
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

fn run_id_mv(
    use_nr2: bool, n_tokens: u32, top_k: u32, n: u32, k: u32, n_experts: u32,
    weight_bytes: &[u8], input: &[f32], ids: &[u32], expert_stride: usize,
) -> Vec<f32> {
    if use_nr2 {
        std::env::set_var("HF2Q_Q6K_ID_MV_NR2", "1");
    } else {
        std::env::remove_var("HF2Q_Q6K_ID_MV_NR2");
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let mut input_buf = device
        .alloc_buffer((n_tokens as usize) * (k as usize) * 4, DType::F32,
                      vec![n_tokens as usize, k as usize]).expect("alloc input");
    input_buf.as_mut_slice::<f32>().expect("input").copy_from_slice(input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf.as_mut_slice::<u8>().expect("weight").copy_from_slice(weight_bytes);

    let mut ids_buf = device
        .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()]).expect("alloc ids");
    ids_buf.as_mut_slice::<u32>().expect("ids").copy_from_slice(ids);

    let total_rows = (n_tokens as usize) * (top_k as usize);
    let mut output_buf = device
        .alloc_buffer(total_rows * (n as usize) * 4, DType::F32, vec![total_rows, n as usize])
        .expect("alloc output");
    for v in output_buf.as_mut_slice::<f32>().expect("init").iter_mut() { *v = 0.0; }

    let params = GgmlQuantizedMatmulIdParams {
        n_tokens, top_k, n, k, n_experts, expert_stride: expert_stride as u64,
        ggml_type: GgmlType::Q6_K,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
        &mut encoder, &mut registry, &device,
        &input_buf, &weight_buf, &ids_buf, &mut output_buf, &params,
    ).expect("id dispatch");
    encoder.commit_and_wait().expect("GPU");

    let out = output_buf.as_slice::<f32>().expect("read").to_vec();
    std::env::remove_var("HF2Q_Q6K_ID_MV_NR2");
    out
}

fn check_parity(label: &str, n_tokens: u32, top_k: u32, n: u32, k: u32, n_experts: u32, tol: f32) {
    let weight_per_expert_f32 = pseudo_random_f32(0xb33fb33f, (n_experts as usize) * (n as usize) * (k as usize));
    let weight_bytes = pack_q6_k(&weight_per_expert_f32);
    let expert_stride = ((n as usize) * (k as usize) * 6) / 8 + (n as usize) * 16 / 8 + (n as usize) * 16;
    // Actually compute expert_stride from packed bytes / n_experts.
    let expert_stride_real = weight_bytes.len() / n_experts as usize;
    let _ = expert_stride;

    let input = pseudo_random_f32(0xcafe, (n_tokens as usize) * (k as usize));
    let ids: Vec<u32> = (0..(n_tokens * top_k)).map(|i| (i as u32) % n_experts).collect();

    let baseline = run_id_mv(false, n_tokens, top_k, n, k, n_experts, &weight_bytes, &input, &ids, expert_stride_real);
    let nr2 = run_id_mv(true, n_tokens, top_k, n, k, n_experts, &weight_bytes, &input, &ids, expert_stride_real);

    assert_eq!(baseline.len(), nr2.len(), "{label}: len mismatch");
    let mut max_abs = 0f32;
    for (i, (&b, &n2)) in baseline.iter().zip(nr2.iter()).enumerate() {
        let diff = (b - n2).abs();
        if diff > max_abs { max_abs = diff; }
        assert!(diff <= tol, "{label}: idx[{i}] baseline={b} nr2={n2} diff={diff} > tol={tol}");
    }
    eprintln!("[parity-q6k-id-nr2] {label} n_tokens={n_tokens} top_k={top_k} N={n} K={k} experts={n_experts}: max_abs={:.3e}", max_abs);
}

#[test]
fn q6k_id_mv_nr2_parity_n4_k256_e2() {
    // smallest: N=4 (one TG of 4 rows), K=256 (one block), 2 experts.
    check_parity("N=4 K=256 e=2", 1, 2, 4, 256, 2, 1e-4);
}

#[test]
fn q6k_id_mv_nr2_parity_n8_k512_e4() {
    // 2 TGs in N, 2 blocks/row, 4 experts.
    check_parity("N=8 K=512 e=4", 1, 4, 8, 512, 4, 1e-4);
}

#[test]
fn q6k_id_mv_nr2_parity_multi_token() {
    // 4 tokens × top_k=2 = 8 output rows. Tests multi-token routing.
    check_parity("4tok top_k=2 N=4 K=256 e=4", 4, 2, 4, 256, 4, 1e-4);
}

#[test]
fn q6k_id_mv_nr2_parity_n_odd_boundary() {
    // N=6 not divisible by 4 — exercises boundary check.
    check_parity("boundary N=6 K=256", 1, 2, 6, 256, 2, 1e-4);
}
