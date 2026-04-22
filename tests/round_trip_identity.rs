//! C-3 Target 1: round-trip identity triad for TurboQuant KV nibble path.
//!
//! 9 cells = 3 cases x 3 head_dims:
//!   Case A — full pipeline: fwht_inplace -> nibble_quantize -> nibble_dequantize -> fwht_inplace
//!            (actually nibble_quantize already calls fwht internally, so this is just
//!             nibble_quantize -> nibble_dequantize = one FWHT-quant-dequant-FWHT round trip)
//!   Case B — quant-only: encode/decode without any FWHT
//!   Case C — FWHT-only: fwht_inplace twice (self-inverse sanity check)
//!
//! Deterministic seed: 0xC25EED via Xoshiro256 + Box-Muller.
//! CPU-only; no Metal; no #[cfg(target_vendor = "apple")] gate.
//!
//! ADR-007 C-3 subtask.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::turboquant::{fwht_inplace, CODEBOOK_4BIT};

// ---- PRNG (xoshiro256**, same as test_flash_attn_vec_tq.rs) ----

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *si = x ^ (x >> 31);
        }
        Xoshiro256 { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn randn_pair(rng: &mut Xoshiro256) -> (f64, f64) {
    loop {
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();
        if u1 > 1e-30 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}

fn random_f32_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let (a, b) = randn_pair(rng);
        v.push(a as f32);
        if v.len() < n {
            v.push(b as f32);
        }
    }
    v
}

// ---- Decision boundaries for 4-bit (midpoints of adjacent codebook entries) ----

fn boundaries_4bit() -> [f32; 15] {
    let mut b = [0.0f32; 15];
    for i in 0..15 {
        b[i] = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
    }
    b
}

fn nearest_centroid_4bit(value: f32) -> u8 {
    let boundaries = boundaries_4bit();
    let mut idx: u8 = 0;
    for &b in &boundaries {
        if value > b {
            idx += 1;
        }
    }
    idx
}

// ---- Nibble-format quantize/dequantize (matching GPU kernel) ----
// Verbatim copy from tests/test_flash_attn_vec_tq.rs:98-143

/// Quantize a head vector into nibble-packed format matching the GPU kernel.
fn nibble_quantize(x: &[f32], head_dim: usize) -> (Vec<u8>, f32) {
    let mut rotated = x.to_vec();
    fwht_inplace(&mut rotated).unwrap();

    let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-30 {
        return (vec![0u8; head_dim / 2], 0.0);
    }

    let inv_norm = 1.0 / norm;
    let scale = (head_dim as f32).sqrt();

    // Find nearest centroid for each coordinate
    let mut packed = vec![0u8; head_dim / 2];
    for c in 0..head_dim {
        let scaled = rotated[c] * inv_norm * scale;
        let idx = nearest_centroid_4bit(scaled);
        let byte_idx = c / 2;
        if c % 2 == 0 {
            packed[byte_idx] = idx & 0xF;
        } else {
            packed[byte_idx] |= (idx & 0xF) << 4;
        }
    }

    (packed, norm)
}

/// Dequantize from nibble-packed format back to original domain.
fn nibble_dequantize(packed: &[u8], norm: f32, head_dim: usize) -> Vec<f32> {
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let mut rotated = Vec::with_capacity(head_dim);

    for c in 0..head_dim {
        let byte_idx = c / 2;
        let idx = if c % 2 == 0 {
            (packed[byte_idx] & 0xF) as usize
        } else {
            ((packed[byte_idx] >> 4) & 0xF) as usize
        };
        rotated.push(CODEBOOK_4BIT[idx] * inv_scale * norm);
    }

    fwht_inplace(&mut rotated).unwrap();
    rotated
}

// ---- Case B: quant-only (no FWHT; treat input as already in FWHT domain) ----

fn quant_only_encode(x: &[f32], head_dim: usize) -> (Vec<u8>, f32) {
    // Treat x as if already FWHT-rotated; extract L2 norm, unit-normalize, scale by sqrt(d)
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-30 {
        return (vec![0u8; head_dim / 2], 0.0);
    }

    let inv_norm = 1.0 / norm;
    let scale = (head_dim as f32).sqrt();

    let mut packed = vec![0u8; head_dim / 2];
    for c in 0..head_dim {
        let scaled = x[c] * inv_norm * scale;
        let idx = nearest_centroid_4bit(scaled);
        let byte_idx = c / 2;
        if c % 2 == 0 {
            packed[byte_idx] = idx & 0xF;
        } else {
            packed[byte_idx] |= (idx & 0xF) << 4;
        }
    }

    (packed, norm)
}

fn quant_only_decode(packed: &[u8], norm: f32, head_dim: usize) -> Vec<f32> {
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = Vec::with_capacity(head_dim);

    for c in 0..head_dim {
        let byte_idx = c / 2;
        let idx = if c % 2 == 0 {
            (packed[byte_idx] & 0xF) as usize
        } else {
            ((packed[byte_idx] >> 4) & 0xF) as usize
        };
        out.push(CODEBOOK_4BIT[idx] * inv_scale * norm);
    }

    out
}

// ---- NRMSE ----

fn nrmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let num: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let den: f32 = original.iter().map(|a| a * a).sum();
    if den < 1e-30 {
        return 0.0;
    }
    (num / den).sqrt()
}

// ---- Cell ----

#[derive(Debug)]
struct Cell {
    case: char,
    head_dim: usize,
    nrmse_mean: f64,
    nrmse_std: f64,
}

// ---- Verdict logic ----
//
// Lloyd-Max 4-bit Gaussian rate-distortion floor ~0.097 (source: turboquant.rs:391-449
// compute_lloyd_max_codebook + Cover&Thomas Ch.10). 0.15 is the upper bound on
// "healthy" quant-only nrmse accounting for finite-sample 1000-vector noise.
fn decide_verdict(cells: &[Cell]) -> &'static str {
    let case_c_nrmse_max = cells
        .iter()
        .filter(|c| c.case == 'C')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);

    let case_b_nrmse_max = cells
        .iter()
        .filter(|c| c.case == 'B')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);

    // Match same head_dim ordering for A-over-B ratio
    let head_dims = [128usize, 256, 512];
    let case_a_over_b_ratio_max = head_dims
        .iter()
        .map(|&hd| {
            let a = cells
                .iter()
                .find(|c| c.case == 'A' && c.head_dim == hd)
                .map(|c| c.nrmse_mean)
                .unwrap_or(0.0);
            let b = cells
                .iter()
                .find(|c| c.case == 'B' && c.head_dim == hd)
                .map(|c| c.nrmse_mean)
                .unwrap_or(1.0);
            if b < 1e-10 { 0.0 } else { a / b }
        })
        .fold(f64::NEG_INFINITY, f64::max);

    let case_a_nrmse_max = cells
        .iter()
        .filter(|c| c.case == 'A')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);

    // Band around the first-principles Lloyd-Max 4-bit N(0,1) floor.
    // Analytic derivation using CODEBOOK_4BIT: MSE = 0.009501008, RMSE = 0.097473.
    // Codex read-only review independently re-derived the same value (cfa-20260422-C3).
    // Band [0.085, 0.11] = analytic floor +/- ~12%, tight enough to detect
    // codebook drift or scale-pairing regressions, loose enough for finite-N jitter.
    const FLOOR_LO: f64 = 0.085;
    const FLOOR_HI: f64 = 0.11;
    const RATIO_LO: f64 = 0.90;
    const RATIO_HI: f64 = 1.10;

    if case_c_nrmse_max > 1e-5 {
        return "FWHT_non_reversible";
    }
    if case_b_nrmse_max > FLOOR_HI {
        return "CODEBOOK_bug";
    }
    if case_a_over_b_ratio_max > RATIO_HI {
        return "FWHT_normalization_bug";
    }
    let in_floor_band = case_a_nrmse_max >= FLOOR_LO
        && case_a_nrmse_max <= FLOOR_HI
        && case_a_over_b_ratio_max >= RATIO_LO
        && case_a_over_b_ratio_max <= RATIO_HI;
    if in_floor_band {
        return "representation_floor_confirmed";
    }
    "unexpected_pattern"
}

// ---- Main test ----

#[test]
fn round_trip_identity() {
    const SEED: u64 = 0xC25EED;
    const N_VECTORS: usize = 1000;
    const HEAD_DIMS: [usize; 3] = [128, 256, 512];

    // Lloyd-Max 4-bit N(0,1) theoretical floor
    const LLOYD_MAX_FLOOR: f64 = 0.097;

    let mut cells: Vec<Cell> = Vec::new();

    for &head_dim in &HEAD_DIMS {
        // One RNG per head_dim block, but seeded identically for reproducibility
        // (spec says single stream; we advance through the same stream per head_dim
        // by using a fresh seeded rng for each head_dim so ordering doesn't matter)
        // Actually per spec: "single stream of Gaussian floats" seeded with SEED.
        // We use three independent rngs (one per head_dim) all seeded from SEED,
        // so vectors at each head_dim are deterministic.
        let mut rng = Xoshiro256::new(SEED);

        // Collect nrmse for each case
        let mut nrmse_a = Vec::with_capacity(N_VECTORS);
        let mut nrmse_b = Vec::with_capacity(N_VECTORS);
        let mut nrmse_c = Vec::with_capacity(N_VECTORS);

        for _ in 0..N_VECTORS {
            let x = random_f32_vec(&mut rng, head_dim);

            // Case A: full pipeline = nibble_quantize -> nibble_dequantize
            // (nibble_quantize internally applies fwht_inplace; nibble_dequantize applies it again)
            let (packed_a, norm_a) = nibble_quantize(&x, head_dim);
            let x_recon_a = nibble_dequantize(&packed_a, norm_a, head_dim);
            nrmse_a.push(nrmse(&x, &x_recon_a) as f64);

            // Case B: quant-only (no FWHT)
            let (packed_b, norm_b) = quant_only_encode(&x, head_dim);
            let x_recon_b = quant_only_decode(&packed_b, norm_b, head_dim);
            nrmse_b.push(nrmse(&x, &x_recon_b) as f64);

            // Case C: FWHT-only (two applications = identity since H_norm * H_norm = I)
            let mut x_fwht = x.clone();
            fwht_inplace(&mut x_fwht).unwrap();
            fwht_inplace(&mut x_fwht).unwrap();
            nrmse_c.push(nrmse(&x, &x_fwht) as f64);
        }

        let mean_nrmse = |v: &[f64]| -> f64 {
            v.iter().sum::<f64>() / v.len() as f64
        };
        let std_nrmse = |v: &[f64]| -> f64 {
            let m = mean_nrmse(v);
            let var: f64 = v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
            var.sqrt()
        };

        cells.push(Cell {
            case: 'A',
            head_dim,
            nrmse_mean: mean_nrmse(&nrmse_a),
            nrmse_std: std_nrmse(&nrmse_a),
        });
        cells.push(Cell {
            case: 'B',
            head_dim,
            nrmse_mean: mean_nrmse(&nrmse_b),
            nrmse_std: std_nrmse(&nrmse_b),
        });
        cells.push(Cell {
            case: 'C',
            head_dim,
            nrmse_mean: mean_nrmse(&nrmse_c),
            nrmse_std: std_nrmse(&nrmse_c),
        });
    }

    // Compute verdict
    let verdict = decide_verdict(&cells);

    // Compute ratio_to_case_b for cases A and C
    let ratio_for = |case: char, hd: usize| -> f64 {
        let a = cells
            .iter()
            .find(|c| c.case == case && c.head_dim == hd)
            .map(|c| c.nrmse_mean)
            .unwrap_or(0.0);
        let b = cells
            .iter()
            .find(|c| c.case == 'B' && c.head_dim == hd)
            .map(|c| c.nrmse_mean)
            .unwrap_or(1.0);
        if b < 1e-10 { 0.0 } else { a / b }
    };

    // Print markdown table to stdout
    println!();
    println!("# C-3 Round-Trip Identity Triad Results");
    println!();
    println!("| case | head_dim | n_vectors | nrmse_mean | nrmse_std | ratio_to_case_b |");
    println!("|------|----------|-----------|------------|-----------|-----------------|");

    for cell in &cells {
        let ratio = if cell.case == 'B' {
            1.0
        } else {
            ratio_for(cell.case, cell.head_dim)
        };
        println!(
            "| {}    | {}       | {}        | {:.7}   | {:.7}   | {:.4}           |",
            cell.case, cell.head_dim, N_VECTORS, cell.nrmse_mean, cell.nrmse_std, ratio
        );
    }
    println!();
    println!("Verdict: {}", verdict);
    println!();
    println!("Lloyd-Max 4-bit Gaussian theoretical floor reference: {}", LLOYD_MAX_FLOOR);

    // Determine verdict rationale
    let case_c_max = cells
        .iter()
        .filter(|c| c.case == 'C')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let case_b_max = cells
        .iter()
        .filter(|c| c.case == 'B')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let case_a_over_b_max = [128usize, 256, 512]
        .iter()
        .map(|&hd| ratio_for('A', hd))
        .fold(f64::NEG_INFINITY, f64::max);
    let case_a_max = cells
        .iter()
        .filter(|c| c.case == 'A')
        .map(|c| c.nrmse_mean)
        .fold(f64::NEG_INFINITY, f64::max);

    let verdict_rationale = match verdict {
        "FWHT_non_reversible" => format!(
            "Case C max nrmse {:.2e} > 1e-5 threshold — FWHT is not self-inverse",
            case_c_max
        ),
        "CODEBOOK_bug" => format!(
            "Case B max nrmse {:.4} > 0.11 floor-band ceiling — codebook quantization error exceeds analytic Lloyd-Max floor 0.09747",
            case_b_max
        ),
        "FWHT_normalization_bug" => format!(
            "Case A/B ratio max {:.4} > 1.10 — FWHT introduces extra error beyond quant floor (encode/decode scale pairing drift)",
            case_a_over_b_max
        ),
        "representation_floor_confirmed" => format!(
            "Case A max nrmse {:.4} within floor band [0.085, 0.11] and A/B ratio max {:.4} within [0.90, 1.10] — error is at analytic Lloyd-Max 4-bit N(0,1) floor (RMSE=0.09747, independently verified by CFA queen + Codex review 2026-04-22)",
            case_a_max, case_a_over_b_max
        ),
        _ => format!(
            "No threshold triggered (case_c_max={:.2e}, case_b_max={:.4}, a_over_b_max={:.4}, case_a_max={:.4}); band [0.085,0.11] x ratio [0.90,1.10]",
            case_c_max, case_b_max, case_a_over_b_max, case_a_max
        ),
    };

    println!("Verdict rationale: {}", verdict_rationale);

    // Build JSON cells
    let mut json_cells = String::from("[\n");
    let head_dims_ordered = [128usize, 256, 512];
    let cases = ['A', 'B', 'C'];
    let mut first = true;
    for &hd in &head_dims_ordered {
        for &case in &cases {
            if let Some(cell) = cells.iter().find(|c| c.case == case && c.head_dim == hd) {
                let ratio = if case == 'B' {
                    1.0f64
                } else {
                    ratio_for(case, hd)
                };
                if !first {
                    json_cells.push_str(",\n");
                }
                first = false;
                json_cells.push_str(&format!(
                    "    {{\"case\": \"{}\", \"head_dim\": {}, \"n_vectors\": {}, \"nrmse_mean\": {:.7}, \"nrmse_std\": {:.7}, \"ratio_to_case_b\": {:.6}}}",
                    cell.case, cell.head_dim, N_VECTORS, cell.nrmse_mean, cell.nrmse_std, ratio
                ));
            }
        }
    }
    json_cells.push_str("\n  ]");

    let result_json = format!(
        r#"{{
  "session": "cfa-20260422-C3-roundtrip",
  "seed": "0xC25EED",
  "n_vectors_per_cell": {n_vectors},
  "lloyd_max_4bit_gaussian_floor_ref": {floor},
  "cells": {cells},
  "verdict": "{verdict}",
  "verdict_rationale": "{rationale}"
}}"#,
        n_vectors = N_VECTORS,
        floor = LLOYD_MAX_FLOOR,
        cells = json_cells,
        verdict = verdict,
        rationale = verdict_rationale.replace('"', "'"),
    );

    // Build markdown table string
    let mut md_rows = String::new();
    for &hd in &head_dims_ordered {
        for &case in &cases {
            if let Some(cell) = cells.iter().find(|c| c.case == case && c.head_dim == hd) {
                let ratio = if case == 'B' {
                    1.0f64
                } else {
                    ratio_for(case, hd)
                };
                md_rows.push_str(&format!(
                    "| {} | {} | {} | {:.7} | {:.7} | {:.6} |\n",
                    cell.case, cell.head_dim, N_VECTORS, cell.nrmse_mean, cell.nrmse_std, ratio
                ));
            }
        }
    }

    let result_md = format!(
        "# C-3 Round-Trip Identity Triad Results\n\nSession: cfa-20260422-C3-roundtrip  \nSeed: 0xC25EED  \nN vectors per cell: {}  \nLloyd-Max 4-bit Gaussian floor reference: {}\n\n| case | head_dim | n_vectors | nrmse_mean | nrmse_std | ratio_to_case_b |\n|------|----------|-----------|------------|-----------|------------------|\n{}  \n**Verdict: {}**  \n**Verdict rationale:** {}\n",
        N_VECTORS, LLOYD_MAX_FLOOR, md_rows, verdict, verdict_rationale
    );

    // Write output files
    std::fs::create_dir_all("/tmp/cfa-20260422-C3-roundtrip").expect("create output dir");
    std::fs::write("/tmp/cfa-20260422-C3-roundtrip/result.json", &result_json)
        .expect("write result.json");
    std::fs::write("/tmp/cfa-20260422-C3-roundtrip/result.md", &result_md)
        .expect("write result.md");

    println!("Written: /tmp/cfa-20260422-C3-roundtrip/result.json");
    println!("Written: /tmp/cfa-20260422-C3-roundtrip/result.md");

    // Regression gates — test FAILS if any of these drift. Each assertion
    // pairs with one of the decision-tree branches so a future regression
    // lands on the right hypothesis without requiring manual interpretation.
    for c in &cells {
        if c.case == 'C' {
            assert!(
                c.nrmse_mean < 1e-5,
                "FWHT non-reversible at head_dim={} (nrmse={:.3e}); orthogonal FWHT should round-trip to machine epsilon",
                c.head_dim, c.nrmse_mean
            );
        }
        if c.case == 'A' || c.case == 'B' {
            assert!(
                c.nrmse_mean >= 0.085 && c.nrmse_mean <= 0.11,
                "Case {} at head_dim={} nrmse={:.5} outside Lloyd-Max 4-bit N(0,1) floor band [0.085, 0.11] (analytic RMSE=0.09747); indicates CODEBOOK drift or scale-pairing regression",
                c.case, c.head_dim, c.nrmse_mean
            );
        }
        if c.case == 'A' {
            let case_b_match = cells
                .iter()
                .find(|o| o.case == 'B' && o.head_dim == c.head_dim)
                .map(|o| o.nrmse_mean)
                .expect("Case B cell for this head_dim");
            let ratio = c.nrmse_mean / case_b_match;
            assert!(
                ratio >= 0.90 && ratio <= 1.10,
                "Case A/B ratio at head_dim={} = {:.4} outside [0.90, 1.10]; indicates FWHT normalization convention has drifted from its round-trip-identity pair",
                c.head_dim, ratio
            );
        }
    }
    assert_eq!(
        verdict, "representation_floor_confirmed",
        "Verdict regression: expected representation_floor_confirmed (per ADR-007 C-3 outcome 2026-04-22), got {}",
        verdict
    );
}
