//! Correctness gates for `quantized_matmul_id_ggml_mv_q4_k_nr2`
//! (`kernel_mul_mv_id_q4_K_f32_nr2`, the peer's Q4_K mat-vec layout):
//! - F64 reference per (row, expert) pair within the K-scaled bound
//!   `(K / 32 + 16) * 2^-24 * sum_k |x_k| (|d sc q| + |dmin m|)` at the
//!   Qwen3.6 expert gate/up (512 x 2048, w rows x top-8) and down
//!   (2048 x 512, w * 8 rows x top-1) shapes and at odd row tails, covering
//!   every row-group fold of the kernel (K = 256: 4, K = 512: 2, else 1);
//! - pair independence, bitwise: every pair of a dispatch of 1, 8, 16, 24 or
//!   32 pairs equals the one-pair dispatch of that (row, expert), in the
//!   broadcast gate/up layout and in the one-pair-per-row down layout, for
//!   permuted pair orders, repeated experts, other pairs' ids replaced or
//!   invalid, and a NaN/inf activation row in another token;
//! - an out-of-range id writes the stock kernel's NaN poison and nothing
//!   outside the output view; every output element is written;
//! - argument validation (Q4_K only).

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]

#[path = "support/gemma_expert_data.rs"]
mod gemma_expert_data;

use std::collections::HashMap;

use gemma_expert_data::*;
use mlx_native::{
    quantized_matmul_id_ggml_mv_q4_k_nr2, quantized_matmul_id_ggml_mv_with_policy,
    GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxDevice,
};

/// Qwen3.6-35B-A3B expert shapes.
const HIDDEN: usize = 2048;
const MOE_I: usize = 512;
const TOP_K: usize = 8;
/// Fewer experts than the model's 256 keep the buffers small; the kernel
/// only sees `expert_id * expert_stride`.
const N_EXPERTS: usize = 64;

struct Harness {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Harness {
    fn new() -> Self {
        Self {
            device: MlxDevice::new().expect("Metal device"),
            registry: KernelRegistry::new(),
        }
    }

    /// One dispatch; returns the raw u32 of the guarded `[pairs, n]` output.
    fn run(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
        stock: bool,
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        assert_eq!(ids.len(), pairs);
        assert_eq!(input.len(), n_tokens * w.k);
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * w.n]);
        let mut encoder = self.device.command_encoder().expect("encoder");
        let p = params(w, n_tokens, top_k);
        if stock {
            quantized_matmul_id_ggml_mv_with_policy(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input_view,
                &w.view,
                &ids_view,
                &output_view,
                &p,
                &GgmlRoutingPolicy::default(),
            )
            .expect("stock mv_id");
        } else {
            quantized_matmul_id_ggml_mv_q4_k_nr2(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input_view,
                &w.view,
                &ids_view,
                &output_view,
                &p,
            )
            .expect("nr2 mv_id");
        }
        encoder.commit_and_wait().expect("GPU");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }
}

/// Output row `pair` of a guarded `run` result.
fn row(out: &[u32], pair: usize, n: usize) -> &[u32] {
    &out[GUARD + pair * n..GUARD + (pair + 1) * n]
}

// ---- F64 reference ----

fn scale_min(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0x0f) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

/// Dequantized row in F64 with the per-value magnitude `|d sc q| + |dmin m|`.
fn dequant_row(w: &Weights, expert: usize, r: usize) -> (Vec<f64>, Vec<f64>) {
    let bytes = w.view.as_slice::<u8>().expect("weight bytes");
    let nb = w.k / 256;
    let base = expert * w.stride + r * nb * 144;
    let mut values = Vec::with_capacity(w.k);
    let mut magnitude = Vec::with_capacity(w.k);
    for b in 0..nb {
        let block = &bytes[base + b * 144..base + (b + 1) * 144];
        let d = f64::from(half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32());
        let dmin =
            f64::from(half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32());
        let scales = &block[4..16];
        let qs = &block[16..144];
        for j in 0..4 {
            let q = &qs[32 * j..32 * j + 32];
            for (half_index, shift) in [(0usize, 0u8), (1, 4)] {
                let (sc, m) = scale_min(2 * j + half_index, scales);
                for &byte in q {
                    let quant = f64::from((byte >> shift) & 0x0f);
                    let scaled = d * f64::from(sc) * quant;
                    let min = dmin * f64::from(m);
                    values.push(scaled - min);
                    magnitude.push(scaled.abs() + min.abs());
                }
            }
        }
    }
    (values, magnitude)
}

/// Worst `|got - ref| / bound` over every element of every pair.
fn reference_ratio(
    w: &Weights,
    n_tokens: usize,
    top_k: usize,
    ids: &[u32],
    input: &[f32],
    out: &[u32],
    label: &str,
) -> f64 {
    let unit = (w.k as f64 / 32.0 + 16.0) * f64::powi(2.0, -24);
    let mut worst = 0f64;
    for pair in 0..n_tokens * top_k {
        let x = &input[(pair / top_k) * w.k..(pair / top_k + 1) * w.k];
        for r in 0..w.n {
            let (values, magnitude) = dequant_row(w, ids[pair] as usize, r);
            let mut want = 0f64;
            let mut scale = 0f64;
            for k in 0..w.k {
                want += f64::from(x[k]) * values[k];
                scale += f64::from(x[k]).abs() * magnitude[k];
            }
            let got = f64::from(f32::from_bits(row(out, pair, w.n)[r]));
            let bound = unit * scale + f64::from(f32::MIN_POSITIVE) * w.k as f64;
            let err = (got - want).abs();
            assert!(
                err <= bound,
                "{label}: pair {pair} (expert {}) column {r}: got {got:e}, F64 {want:e}, err {err:e} > bound {bound:e}",
                ids[pair]
            );
            worst = worst.max(err / bound);
        }
    }
    worst
}

fn distinct_ids(rng: &mut Rng, n_tokens: usize, top_k: usize, n_experts: usize) -> Vec<u32> {
    let pool: Vec<u32> = (0..n_experts as u32).collect();
    (0..n_tokens).flat_map(|_| pick_distinct(rng, &pool, top_k)).collect()
}

/// Finite activations in [-1, 1) (the F64 test adds the edge classes of
/// `make_input` separately).
fn plain_input(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * k).map(|_| rng.unit() * 2.0 - 1.0).collect()
}

#[test]
fn matches_f64_reference_within_k_scaled_bound() {
    let mut h = Harness::new();
    let mut rng = Rng(0x5157_3336);
    // (N, K, n_tokens, top_k, label)
    let cases = [
        (MOE_I, HIDDEN, 4, TOP_K, "gate/up w=4"),
        (MOE_I, HIDDEN, 1, TOP_K, "gate/up w=1"),
        (HIDDEN, MOE_I, 4 * TOP_K, 1, "down w=4"),
        (7, 768, 3, 2, "tail N=7 K=768"),
        (5, 256, 2, 3, "tail N=5 K=256"),
        (6, 512, 2, 2, "tail N=6 K=512"),
        (13, 512, 3, 2, "tail N=13 K=512"),
        (11, 1024, 2, 2, "tail N=11 K=1024"),
    ];
    for (case, &(n, k, n_tokens, top_k, label)) in cases.iter().enumerate() {
        let w = make_weights(&h.device, GgmlType::Q4_K, n, k, N_EXPERTS, 0xa11 + case as u64);
        let ids = distinct_ids(&mut rng, n_tokens, top_k, N_EXPERTS);
        for (edge, input) in [
            (false, plain_input(n_tokens, k, 0xbee + case as u64)),
            (true, make_input(n_tokens, k, 0xcab + case as u64)),
        ] {
            let out = h.run(&w, n_tokens, top_k, &ids, &input, false);
            assert_stock_frame(&out, n_tokens * top_k, n, label);
            let ratio = reference_ratio(&w, n_tokens, top_k, &ids, &input, &out, label);
            let stock = h.run(&w, n_tokens, top_k, &ids, &input, true);
            let stock_ratio = reference_ratio(&w, n_tokens, top_k, &ids, &input, &stock, label);
            let differ = (GUARD..out.len() - GUARD).filter(|&i| out[i] != stock[i]).count();
            eprintln!(
                "[q4k-nr2 f64] {label} edge={edge}: worst err/bound nr2 {ratio:.3e}, stock {stock_ratio:.3e}; {differ} of {} elements differ from stock",
                n_tokens * top_k * n
            );
        }
    }
}

/// Bitwise one-pair references, cached by (input row values, expert).
struct Singles<'a> {
    w: &'a Weights,
    cache: HashMap<(Vec<u32>, u32), Vec<u32>>,
    dispatches: usize,
}

impl<'a> Singles<'a> {
    fn new(w: &'a Weights) -> Self {
        Self { w, cache: HashMap::new(), dispatches: 0 }
    }

    fn get(&mut self, h: &mut Harness, x: &[f32], id: u32) -> Vec<u32> {
        let key = (x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), id);
        if let Some(found) = self.cache.get(&key) {
            return found.clone();
        }
        let out = h.run(self.w, 1, 1, &[id], x, false);
        let result = row(&out, 0, self.w.n).to_vec();
        self.dispatches += 1;
        self.cache.insert(key, result.clone());
        result
    }
}

/// Runs one dispatch and checks every valid pair against its one-pair
/// dispatch and every invalid pair against all-NaN. Returns pairs checked.
fn check_against_singles(
    h: &mut Harness,
    singles: &mut Singles,
    n_tokens: usize,
    top_k: usize,
    ids: &[u32],
    input: &[f32],
    label: &str,
) -> usize {
    let w = singles.w;
    let out = h.run(w, n_tokens, top_k, ids, input, false);
    assert_stock_frame(&out, n_tokens * top_k, w.n, label);
    for pair in 0..n_tokens * top_k {
        let got = row(&out, pair, w.n);
        if (ids[pair] as usize) >= w.n_experts {
            assert!(
                got.iter().all(|&b| f32::from_bits(b).is_nan()),
                "{label}: invalid id at pair {pair} did not write NaN"
            );
            continue;
        }
        let t = pair / top_k;
        let want = singles.get(h, &input[t * w.k..(t + 1) * w.k], ids[pair]);
        if let Some(c) = (0..w.n).find(|&c| got[c] != want[c]) {
            panic!(
                "{label}: pair {pair} (token {t}, expert {}) column {c}: {:#010x} in the dispatch, {:#010x} alone",
                ids[pair], got[c], want[c]
            );
        }
    }
    n_tokens * top_k
}

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
}

/// Gate/up layout: w token rows broadcast to top-8 experts (8w pairs).
fn gate_up_independence(n: usize, k: usize, seed: u64) {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, n, k, N_EXPERTS, seed);
    let mut singles = Singles::new(&w);
    let mut rng = Rng(seed ^ 0x9e37);
    let mut checked = 0;
    let x = make_input(4, k, seed ^ 0x51);
    let ids = distinct_ids(&mut rng, 4, TOP_K, N_EXPERTS);
    for rows in 1..=4 {
        let label = format!("gate/up N={n} K={k} w={rows}");
        checked += check_against_singles(
            &mut h, &mut singles, rows, TOP_K, &ids[..rows * TOP_K], &x[..rows * k], &label,
        );
        // Permuted token order and slot order.
        let mut tokens: Vec<usize> = (0..rows).collect();
        shuffle(&mut rng, &mut tokens);
        let mut px = Vec::with_capacity(rows * k);
        let mut pids = Vec::with_capacity(rows * TOP_K);
        for &t in &tokens {
            px.extend_from_slice(&x[t * k..(t + 1) * k]);
            let mut slots = ids[t * TOP_K..(t + 1) * TOP_K].to_vec();
            shuffle(&mut rng, &mut slots);
            pids.extend_from_slice(&slots);
        }
        checked += check_against_singles(
            &mut h, &mut singles, rows, TOP_K, &pids, &px, &format!("{label} permuted"),
        );
        // Other pairs' ids replaced: token 0 keeps its experts, the rest take
        // a fresh draw, then one expert for all, then invalid ids.
        let mut fresh = ids[..rows * TOP_K].to_vec();
        fresh[TOP_K..].copy_from_slice(&distinct_ids(&mut rng, rows - 1, TOP_K, N_EXPERTS));
        checked += check_against_singles(
            &mut h, &mut singles, rows, TOP_K, &fresh, &x[..rows * k], &format!("{label} others redrawn"),
        );
        let mut same = ids[..rows * TOP_K].to_vec();
        same[TOP_K..].fill(ids[0]);
        checked += check_against_singles(
            &mut h, &mut singles, rows, TOP_K, &same, &x[..rows * k], &format!("{label} others one expert"),
        );
        let mut invalid = ids[..rows * TOP_K].to_vec();
        for (i, id) in invalid.iter_mut().enumerate().skip(1).step_by(3) {
            *id = if i % 2 == 0 { INVALID_MAX } else { N_EXPERTS as u32 };
        }
        checked += check_against_singles(
            &mut h, &mut singles, rows, TOP_K, &invalid, &x[..rows * k], &format!("{label} invalid ids"),
        );
        // NaN/inf in another token's row leaves token 0 unchanged.
        if rows > 1 {
            let mut poisoned = x[..rows * k].to_vec();
            poison_row(&mut poisoned, rows - 1, k);
            let out = h.run(&w, rows, TOP_K, &ids[..rows * TOP_K], &poisoned, false);
            for pair in 0..(rows - 1) * TOP_K {
                let t = pair / TOP_K;
                let want = singles.get(&mut h, &x[t * k..(t + 1) * k], ids[pair]);
                assert_eq!(row(&out, pair, n), &want[..], "{label}: pair {pair} changed by a NaN row");
            }
            checked += (rows - 1) * TOP_K;
        }
    }
    eprintln!(
        "[q4k-nr2 independence] gate/up N={n} K={k}: {checked} pairs bitwise equal to {} one-pair dispatches",
        singles.dispatches
    );
}

/// Down layout: one row per pair, top-1 (1, 8, 16, 24, 32 pairs).
fn down_independence(n: usize, k: usize, seed: u64) {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, n, k, N_EXPERTS, seed);
    let mut singles = Singles::new(&w);
    let mut rng = Rng(seed ^ 0x7f4a);
    let mut checked = 0;
    let x = make_input(32, k, seed ^ 0x77);
    let ids = distinct_ids(&mut rng, 4, TOP_K, N_EXPERTS);
    for pairs in [1usize, 8, 16, 24, 32] {
        let label = format!("down N={n} K={k} pairs={pairs}");
        checked += check_against_singles(
            &mut h, &mut singles, pairs, 1, &ids[..pairs], &x[..pairs * k], &label,
        );
        let mut order: Vec<usize> = (0..pairs).collect();
        shuffle(&mut rng, &mut order);
        let px: Vec<f32> = order.iter().flat_map(|&p| x[p * k..(p + 1) * k].iter().copied()).collect();
        let pids: Vec<u32> = order.iter().map(|&p| ids[p]).collect();
        checked += check_against_singles(
            &mut h, &mut singles, pairs, 1, &pids, &px, &format!("{label} permuted"),
        );
        let same = vec![ids[0]; pairs];
        checked += check_against_singles(
            &mut h, &mut singles, pairs, 1, &same, &x[..pairs * k], &format!("{label} one expert"),
        );
        let mut invalid = ids[..pairs].to_vec();
        for id in invalid.iter_mut().skip(1).step_by(2) {
            *id = INVALID_MAX;
        }
        checked += check_against_singles(
            &mut h, &mut singles, pairs, 1, &invalid, &x[..pairs * k], &format!("{label} invalid ids"),
        );
    }
    eprintln!(
        "[q4k-nr2 independence] down N={n} K={k}: {checked} pairs bitwise equal to {} one-pair dispatches",
        singles.dispatches
    );
}

#[test]
fn gate_up_pairs_equal_one_pair_dispatches_bitwise() {
    gate_up_independence(MOE_I, HIDDEN, 0x6a7e);
}

#[test]
fn down_pairs_equal_one_pair_dispatches_bitwise() {
    down_independence(HIDDEN, MOE_I, 0xd0e1);
}

#[test]
fn row_tail_pairs_equal_one_pair_dispatches_bitwise() {
    gate_up_independence(7, 768, 0x7a11);
    down_independence(5, 256, 0x5a11);
    down_independence(13, 512, 0xd13a);
    gate_up_independence(11, 1024, 0x11a1);
}

#[test]
fn invalid_ids_write_the_stock_poison_only_in_their_rows() {
    let mut h = Harness::new();
    for (n, k, n_tokens, top_k) in [(MOE_I, HIDDEN, 2, TOP_K), (HIDDEN, MOE_I, 16, 1), (7, 256, 3, 2)] {
        let w = make_weights(&h.device, GgmlType::Q4_K, n, k, N_EXPERTS, 0x9015 + n as u64);
        let mut rng = Rng(0xbad + n as u64);
        let clean_ids = distinct_ids(&mut rng, n_tokens, top_k, N_EXPERTS);
        let input = make_input(n_tokens, k, 0x1d + n as u64);
        let mut ids = clean_ids.clone();
        ids[0] = INVALID_MAX;
        let last = ids.len() - 1;
        ids[last] = N_EXPERTS as u32;
        let label = format!("poison N={n} K={k} tokens={n_tokens} top_k={top_k}");
        let clean = h.run(&w, n_tokens, top_k, &clean_ids, &input, false);
        let got = h.run(&w, n_tokens, top_k, &ids, &input, false);
        let stock = h.run(&w, n_tokens, top_k, &ids, &input, true);
        assert_stock_frame(&got, ids.len(), n, &label);
        assert_stock_frame(&stock, ids.len(), n, &label);
        for pair in 0..ids.len() {
            if pair == 0 || pair == last {
                assert_eq!(row(&got, pair, n), row(&stock, pair, n), "{label}: poison of pair {pair} differs from stock");
                assert!(row(&got, pair, n).iter().all(|&b| f32::from_bits(b).is_nan()), "{label}: pair {pair} not NaN");
            } else {
                assert_eq!(row(&got, pair, n), row(&clean, pair, n), "{label}: valid pair {pair} changed");
            }
        }
    }
}

#[test]
fn rejects_other_types() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, 8, 256, 2, 1);
    let (_ip, input) = guarded_f32(&h.device, &vec![0.0; 256]);
    let (_dp, ids) = guarded_u32(&h.device, &[0]);
    let (_op, output) = guarded_f32(&h.device, &vec![0.0; 8]);
    let mut p = params(&w, 1, 1);
    p.ggml_type = GgmlType::Q6_K;
    let mut encoder = h.device.command_encoder().expect("encoder");
    let err = quantized_matmul_id_ggml_mv_q4_k_nr2(
        &mut encoder, &mut h.registry, &h.device, &input, &w.view, &ids, &output, &p,
    );
    assert!(err.is_err(), "Q6_K params must be rejected");
}
