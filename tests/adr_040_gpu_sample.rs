//! ADR-040 §26 iter-M — byte-identity + speed spike for the GPU-side
//! `gpu_sample_argmax_candidates` kernel. Validates it reproduces the host
//! `argmax_f32_first_max` (first-max, lower-index tie-break) + the finalize
//! threshold candidate set (logits >= top1_val - 0.5f) EXACTLY on adversarial
//! inputs (ties, NaN, all -inf, many-candidates/overflow), then measures the
//! GPU time at the production shape (N=8, vocab=262144) — must be << 0.92ms
//! (the host scan it replaces) to net-win.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::gpu_sample::dispatch_gpu_sample_argmax_candidates;
use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Host reference = hf2q `argmax_f32_first_max` + finalize threshold scan.
fn host_ref(xs: &[f32]) -> (u32, f32, Vec<u32>) {
    let maxv = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let bi = xs.iter().position(|&v| v == maxv).unwrap_or(0) as u32;
    let threshold = maxv - 0.5f32;
    let cands: Vec<u32> = xs
        .iter()
        .enumerate()
        .filter(|(_, &v)| v >= threshold)
        .map(|(i, _)| i as u32)
        .collect();
    (bi, maxv, cands)
}

fn run_case(name: &str, rows: &[Vec<f32>], cap: u32) {
    let device = MlxDevice::new().expect("device");
    let mut reg = KernelRegistry::new();
    let n = rows.len();
    let vocab = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == vocab));

    let mut flat = Vec::with_capacity(n * vocab);
    for r in rows {
        flat.extend_from_slice(r);
    }
    let mut logits = device
        .alloc_buffer(n * vocab * 4, DType::F32, vec![n, vocab])
        .expect("logits");
    logits.as_mut_slice::<f32>().unwrap().copy_from_slice(&flat);

    let top1_idx = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let top1_val = device.alloc_buffer(n * 4, DType::F32, vec![n]).unwrap();
    let cand_count = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let overflow = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let cand_ids = device
        .alloc_buffer(n * cap as usize * 4, DType::U32, vec![n, cap as usize])
        .unwrap();
    let mut params = device.alloc_buffer(2 * 4, DType::U32, vec![2]).unwrap();
    params.as_mut_slice::<u32>().unwrap().copy_from_slice(&[vocab as u32, cap]);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_gpu_sample_argmax_candidates(
        &mut enc, &mut reg, device.metal_device(), &logits, &top1_idx, &top1_val, &cand_count,
        &overflow, &cand_ids, &params, n as u32, vocab as u32, cap,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("gpu");

    let g_idx = top1_idx.as_slice::<u32>().unwrap();
    let g_val = top1_val.as_slice::<f32>().unwrap();
    let g_cnt = cand_count.as_slice::<u32>().unwrap();
    let g_ovf = overflow.as_slice::<u32>().unwrap();
    let g_ids = cand_ids.as_slice::<u32>().unwrap();

    for (s, row) in rows.iter().enumerate() {
        let (h_idx, h_val, h_cands) = host_ref(row);
        assert_eq!(g_idx[s], h_idx, "{name} slot{s}: top1_idx GPU {} != host {}", g_idx[s], h_idx);
        assert_eq!(
            g_val[s].to_bits(), h_val.to_bits(),
            "{name} slot{s}: top1_val GPU {} != host {}", g_val[s], h_val
        );
        let overflowed = g_ovf[s] == 1 || g_cnt[s] > cap;
        let host_overflow = h_cands.len() as u32 > cap;
        assert_eq!(overflowed, host_overflow, "{name} slot{s}: overflow GPU {overflowed} != host {host_overflow}");
        if !host_overflow {
            assert_eq!(g_cnt[s] as usize, h_cands.len(), "{name} slot{s}: cand_count GPU {} != host {}", g_cnt[s], h_cands.len());
            let mut gpu_set: Vec<u32> = g_ids[s * cap as usize..s * cap as usize + g_cnt[s] as usize].to_vec();
            gpu_set.sort_unstable();
            let mut host_set = h_cands.clone();
            host_set.sort_unstable();
            assert_eq!(gpu_set, host_set, "{name} slot{s}: candidate SET mismatch");
        }
    }
    eprintln!("[gpu-sample] {name}: PASS ({n} slots, vocab {vocab}, cap {cap})");
}

fn pseudo(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn gpu_sample_byte_identical_adversarial() {
    // 1. Random typical.
    let rows: Vec<Vec<f32>> = (0..8).map(|i| pseudo(0xc0ffee ^ i, 4096)).collect();
    run_case("random", &rows, 1024);

    // 2. Ties — max value repeated; first index must win.
    let mut tie = pseudo(0x1234, 2048);
    tie[100] = 9.0;
    tie[500] = 9.0;
    tie[1000] = 9.0;
    run_case("ties", &[tie], 1024);

    // 3. NaN scattered (must be skipped by argmax + threshold).
    let mut nanrow = pseudo(0x5678, 2048);
    nanrow[3] = f32::NAN;
    nanrow[77] = f32::NAN;
    nanrow[2000] = 5.0; // the real max
    run_case("nan", &[nanrow], 1024);

    // 4. All -inf → (idx 0, -inf), no candidates above threshold (-inf >= -inf true → all candidates!).
    run_case("all_neg_inf", &[vec![f32::NEG_INFINITY; 1024]], 2048);

    // 5. Many candidates near max → exercises a large candidate set within cap.
    let mut flat = vec![10.0f32; 4096];
    flat[0] = 10.4; // max; threshold=9.9 → all 4096 are candidates
    run_case("many_candidates", &[flat.clone()], 8192);

    // 6. Overflow — same flat row but small cap.
    run_case("overflow", &[flat], 256);
}

const REPS: usize = 100;

#[test]
#[ignore] // speed bench — run with --ignored
fn gpu_sample_speed_n8_vocab262144() {
    let device = MlxDevice::new().expect("device");
    let mut reg = KernelRegistry::new();
    let n = 8usize;
    let vocab = 262144usize;
    let cap = 1024u32;

    let flat = pseudo(0xbeef, n * vocab);
    let mut logits = device.alloc_buffer(n * vocab * 4, DType::F32, vec![n, vocab]).unwrap();
    logits.as_mut_slice::<f32>().unwrap().copy_from_slice(&flat);
    let top1_idx = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let top1_val = device.alloc_buffer(n * 4, DType::F32, vec![n]).unwrap();
    let cand_count = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let overflow = device.alloc_buffer(n * 4, DType::U32, vec![n]).unwrap();
    let cand_ids = device.alloc_buffer(n * cap as usize * 4, DType::U32, vec![n, cap as usize]).unwrap();
    let mut params = device.alloc_buffer(8, DType::U32, vec![2]).unwrap();
    params.as_mut_slice::<u32>().unwrap().copy_from_slice(&[vocab as u32, cap]);

    // warmup
    {
        let mut enc = device.command_encoder().unwrap();
        dispatch_gpu_sample_argmax_candidates(&mut enc, &mut reg, device.metal_device(), &logits, &top1_idx, &top1_val, &cand_count, &overflow, &cand_ids, &params, n as u32, vocab as u32, cap).unwrap();
        enc.commit_and_wait().unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..3 {
        let mut enc = device.command_encoder().unwrap();
        for _ in 0..REPS {
            dispatch_gpu_sample_argmax_candidates(&mut enc, &mut reg, device.metal_device(), &logits, &top1_idx, &top1_val, &cand_count, &overflow, &cand_ids, &params, n as u32, vocab as u32, cap).unwrap();
        }
        let (s, e) = enc.commit_wait_with_gpu_time().unwrap();
        let us = (e - s) * 1e6 / REPS as f64;
        if us < best { best = us; }
    }
    eprintln!("[gpu-sample-speed] N={n} vocab={vocab} cap={cap}: {best:.2} us/dispatch GPU-busy (host scan it replaces = ~920us/step for 8 slots)");
}
