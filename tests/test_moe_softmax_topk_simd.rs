//! Correctness gates for `dispatch_moe_softmax_topk_simd` (one SIMD group per row):
//! - F64 reference: same expert ids in the same order (descending probability, ties to the
//!   lower expert index) and weights within 2e-6 absolute;
//! - rows are independent: a batched dispatch equals one dispatch per row bitwise;
//! - against the shipped `dispatch_moe_softmax_topk` (Class B: only the softmax summation
//!   order differs): identical ids, weights within 2e-6;
//! - explicit ties, NaN logits and the shape limits.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::moe_softmax_topk::{dispatch_moe_softmax_topk, dispatch_moe_softmax_topk_simd};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn lcg(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// F64 reference: softmax, then top-k by (probability desc, expert asc), renormalized.
fn reference(logits: &[f32], n_tokens: usize, n_experts: usize, top_k: usize) -> (Vec<u32>, Vec<f64>) {
    let mut ids = Vec::with_capacity(n_tokens * top_k);
    let mut weights = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        let row: Vec<f64> = logits[t * n_experts..(t + 1) * n_experts]
            .iter()
            .map(|&x| if x.is_nan() { f32::MIN as f64 } else { x.clamp(f32::MIN, f32::MAX) as f64 })
            .collect();
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = row.iter().map(|&x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|&x| x / s).collect();
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
        let chosen = &order[..top_k];
        let ts: f64 = chosen.iter().map(|&i| p[i]).sum();
        for &i in chosen {
            ids.push(i as u32);
            weights.push(p[i] / ts);
        }
    }
    (ids, weights)
}

fn upload(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut b = device.alloc_buffer(data.len() * 4, DType::F32, vec![data.len()]).expect("alloc");
    b.as_mut_slice::<f32>().expect("mut").copy_from_slice(data);
    b
}

/// Runs one kernel over `n_tokens` rows; returns (ids, weights).
fn run(simd: bool, logits: &[f32], n_tokens: u32, n_experts: u32, top_k: u32) -> (Vec<u32>, Vec<f32>) {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let lbuf = upload(&device, logits);
    let n = (n_tokens * top_k) as usize;
    let ids = device.alloc_buffer(n * 4, DType::U32, vec![n]).expect("ids");
    let w = device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("w");
    let mut enc = device.command_encoder().expect("encoder");
    if simd {
        dispatch_moe_softmax_topk_simd(&mut enc, &mut registry, &device, &lbuf, &ids, &w, n_tokens, n_experts, top_k)
            .expect("dispatch simd");
    } else {
        dispatch_moe_softmax_topk(&mut enc, &mut registry, &device, &lbuf, &ids, &w, n_tokens, n_experts, top_k)
            .expect("dispatch");
    }
    enc.commit_and_wait().expect("commit");
    (ids.as_slice::<u32>().expect("ids").to_vec(), w.as_slice::<f32>().expect("w").to_vec())
}

fn check_against_reference(logits: &[f32], n_tokens: usize, n_experts: usize, top_k: usize) {
    let (ids, w) = run(true, logits, n_tokens as u32, n_experts as u32, top_k as u32);
    let (rids, rw) = reference(logits, n_tokens, n_experts, top_k);
    assert_eq!(ids, rids, "ids differ from the F64 reference (tokens {n_tokens}, experts {n_experts}, k {top_k})");
    for (i, (&a, &b)) in w.iter().zip(rw.iter()).enumerate() {
        assert!((a as f64 - b).abs() <= 2e-6, "weight {i}: {a} vs reference {b}");
    }
}

#[test]
fn matches_f64_reference_on_serving_and_edge_shapes() {
    for &(n_experts, top_k) in &[(256usize, 8usize), (128, 8), (64, 4), (512, 10), (1024, 32), (32, 32), (40, 3)] {
        for &n_tokens in &[1usize, 2, 3, 4, 7, 64] {
            for seed in 0..4u64 {
                let logits = lcg(seed * 977 + n_experts as u64 * 31 + n_tokens as u64, n_tokens * n_experts, 4.0);
                check_against_reference(&logits, n_tokens, n_experts, top_k);
            }
        }
    }
}

#[test]
fn batched_rows_equal_single_row_dispatches_bitwise() {
    let (n_tokens, n_experts, top_k) = (4usize, 256usize, 8usize);
    let logits = lcg(4242, n_tokens * n_experts, 6.0);
    let (ids, w) = run(true, &logits, n_tokens as u32, n_experts as u32, top_k as u32);
    for t in 0..n_tokens {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        let (rids, rw) = run(true, row, 1, n_experts as u32, top_k as u32);
        assert_eq!(&ids[t * top_k..(t + 1) * top_k], &rids[..], "row {t} ids");
        let a: Vec<u32> = w[t * top_k..(t + 1) * top_k].iter().map(|x| x.to_bits()).collect();
        let b: Vec<u32> = rw.iter().map(|x| x.to_bits()).collect();
        assert_eq!(a, b, "row {t} weights differ bitwise");
    }
}

#[test]
fn agrees_with_the_shipped_kernel() {
    for &n_tokens in &[1u32, 2, 3, 4, 16] {
        for seed in 0..8u64 {
            let logits = lcg(seed + 100 * n_tokens as u64, (n_tokens * 256) as usize, 5.0);
            let (ids_old, w_old) = run(false, &logits, n_tokens, 256, 8);
            let (ids_new, w_new) = run(true, &logits, n_tokens, 256, 8);
            assert_eq!(ids_old, ids_new, "ids differ from the shipped kernel (tokens {n_tokens}, seed {seed})");
            for (i, (&a, &b)) in w_old.iter().zip(w_new.iter()).enumerate() {
                assert!((a - b).abs() <= 2e-6, "weight {i}: shipped {a} vs simd {b}");
            }
        }
    }
}

#[test]
fn ties_go_to_the_lower_expert_and_nan_never_wins() {
    let n_experts = 256usize;
    // every logit equal except a NaN in the middle: the first 8 experts win, equal weights
    let mut logits = vec![0.5f32; n_experts];
    logits[3] = f32::NAN;
    let (ids, w) = run(true, &logits, 1, n_experts as u32, 8);
    assert_eq!(ids, vec![0, 1, 2, 4, 5, 6, 7, 8]);
    for &x in &w {
        assert!((x - 0.125).abs() <= 1e-7, "tied weight {x}");
    }
    // ties spread across lanes and per-lane slots: experts 200 (lane 8, slot 6), 41 (lane 9,
    // slot 1) and 73 (lane 9, slot 2) equal and largest
    let mut logits = lcg(9, n_experts, 1.0);
    for &e in &[200usize, 41, 73] {
        logits[e] = 3.0;
    }
    let (ids, _) = run(true, &logits, 1, n_experts as u32, 8);
    assert_eq!(&ids[..3], &[41, 73, 200]);
    check_against_reference(&logits, 1, n_experts, 8);
}

#[test]
fn infinite_rows_select_real_experts() {
    let n_experts = 256usize;
    // an all -inf row and a row with a +inf logit: every id is a real expert
    let mut logits = vec![f32::NEG_INFINITY; 2 * n_experts];
    for x in logits[n_experts..].iter_mut() {
        *x = 0.25;
    }
    logits[n_experts + 77] = f32::INFINITY;
    let (ids, w) = run(true, &logits, 2, n_experts as u32, 8);
    assert!(ids.iter().all(|&i| (i as usize) < n_experts), "ids {ids:?}");
    assert_eq!(&ids[..8], &[0, 1, 2, 3, 4, 5, 6, 7], "all -inf row ties to the lowest experts");
    assert_eq!(ids[8], 77, "the +inf expert wins its row");
    assert!(w.iter().all(|x| x.is_finite()), "weights {w:?}");
    check_against_reference(&logits, 2, n_experts, 8);
}

#[test]
fn rejects_out_of_range_shapes() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let logits = upload(&device, &vec![0.0f32; 2048]);
    let ids = device.alloc_buffer(64 * 4, DType::U32, vec![64]).expect("ids");
    let w = device.alloc_buffer(64 * 4, DType::F32, vec![64]).expect("w");
    let mut enc = device.command_encoder().expect("encoder");
    assert!(dispatch_moe_softmax_topk_simd(&mut enc, &mut registry, &device, &logits, &ids, &w, 1, 2048, 8).is_err());
    assert!(dispatch_moe_softmax_topk_simd(&mut enc, &mut registry, &device, &logits, &ids, &w, 1, 256, 33).is_err());
    assert!(dispatch_moe_softmax_topk_simd(&mut enc, &mut registry, &device, &logits, &ids, &w, 0, 256, 8).is_err());
    assert!(dispatch_moe_softmax_topk_simd(&mut enc, &mut registry, &device, &logits, &ids, &w, 1, 4, 8).is_err());
}
