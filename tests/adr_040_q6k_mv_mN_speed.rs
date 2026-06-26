//! ADR-040 §0.21c — speed sweep for `kernel_mul_mv_q6_K_f32_mN` vs plain mv.
//!
//! Measures GPU-busy time (commit_wait_with_gpu_time) for the bit-identical
//! column-amortizing mN kernel at m=2..8 on an lm_head-like shape (large N,
//! K~2560), vs plain `kernel_mul_mv_q6_K_f32` run m times (once per column).
//! Each measurement batches REPS dispatches into one command buffer so GPU
//! launch overhead amortizes and the busy-time delta reflects kernel work.
//!
//! Ignored by default (`#[ignore]`) — it is a benchmark, run explicitly with
//! `--ignored`. Requires a single-tenant GPU for meaningful numbers.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    dispatch_mv_q6k_mn_adaptive, quantized_matmul_ggml, DType, GgmlQuantizedMatmulParams,
    GgmlType, KernelRegistry, MlxDevice,
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

// Q6_K pack (same as the parity test).
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
            if amax > max_scale {
                max_scale = amax;
            }
        }
        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }
        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                q6[s * 16 + i] = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
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
        buf.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    buf
}

const REPS: usize = 200;

#[test]
#[ignore]
fn q6k_mvN_speed_sweep_lmhead() {
    // Compare against the PRODUCTION default decode kernel: NR2 (default-on).
    std::env::set_var("HF2Q_Q6K_MV_NR2", "1");
    std::env::remove_var("HF2Q_DECODE_MVN");
    std::env::remove_var("HF2Q_DECODE_MV_EXT");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    // lm_head-like: large N, K=2560 (10 Q6_K blocks/row).
    let n = 4096usize;
    let k = 2560usize;

    let weight_f32 = pseudo_random_f32(0xc0ffee, n * k);
    let weight_bytes = pack_q6_k(&weight_f32);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("w mut")
        .copy_from_slice(&weight_bytes);

    eprintln!("[mvN-speed] lm_head N={n} K={k}, REPS={REPS} dispatches/measure");
    eprintln!("[mvN-speed]  m | NR2(m dispatches) us | mN(1 dispatch) us | speedup");

    for m in 2..=8usize {
        let input = pseudo_random_f32(0xbeef ^ (m as u64), m * k);
        let mut input_buf = device
            .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
            .expect("alloc in");
        input_buf
            .as_mut_slice::<f32>()
            .expect("in mut")
            .copy_from_slice(&input);
        let mut out_buf = device
            .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
            .expect("alloc out");
        for v in out_buf.as_mut_slice::<f32>().expect("out mut").iter_mut() {
            *v = 0.0;
        }

        let params = GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: GgmlType::Q6_K,
        };

        // --- plain mv: m separate column dispatches per rep ---
        // warmup
        {
            let mut enc = device.command_encoder().expect("enc");
            quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &weight_buf, &mut out_buf, &params).expect("plain");
            enc.commit_and_wait().expect("gpu");
        }
        let mut plain_best = f64::MAX;
        for _ in 0..3 {
            let mut enc = device.command_encoder().expect("enc");
            for _ in 0..REPS {
                quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &weight_buf, &mut out_buf, &params).expect("plain");
            }
            let (s, e) = enc.commit_wait_with_gpu_time().expect("gpu time");
            let us = (e - s) * 1e6 / REPS as f64;
            if us < plain_best { plain_best = us; }
        }

        // --- mN adaptive: register-safe column-tiled dispatch(es) per rep ---
        {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_mv_q6k_mn_adaptive(&mut enc, &mut registry, &device, &input_buf, &weight_buf, &out_buf, &params).expect("mN");
            enc.commit_and_wait().expect("gpu");
        }
        let mut mn_best = f64::MAX;
        for _ in 0..3 {
            let mut enc = device.command_encoder().expect("enc");
            for _ in 0..REPS {
                dispatch_mv_q6k_mn_adaptive(&mut enc, &mut registry, &device, &input_buf, &weight_buf, &out_buf, &params).expect("mN");
            }
            let (s, e) = enc.commit_wait_with_gpu_time().expect("gpu time");
            let us = (e - s) * 1e6 / REPS as f64;
            if us < mn_best { mn_best = us; }
        }

        eprintln!(
            "[mvN-speed]  {m} | {:8.2} | {:8.2} | {:.3}x",
            plain_best, mn_best, plain_best / mn_best
        );
    }
    std::env::remove_var("HF2Q_Q6K_MV_NR2");
}
