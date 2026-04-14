//! Validation tests for TurboQuant KV cache compression (ADR-007 Phase 0.1).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::turboquant::{
    fwht_inplace, turboquant_dequantize, turboquant_quantize, BitWidth, TurboQuantConfig,
    CODEBOOK_2BIT, CODEBOOK_3BIT, CODEBOOK_4BIT,
};

// ---- xoshiro256** PRNG ----

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 to initialize state from a single seed
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

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---- Box-Muller for Gaussian sampling ----

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

fn random_normal_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let (a, b) = randn_pair(rng);
        v.push(a as f32);
        i += 1;
        if i < n {
            v.push(b as f32);
            i += 1;
        }
    }
    v
}

/// Generate a random unit vector on S^{d-1}.
fn random_unit_vec(rng: &mut Xoshiro256, d: usize) -> Vec<f32> {
    let v = random_normal_vec(rng, d);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-30 {
        // Retry (extremely unlikely)
        return random_unit_vec(rng, d);
    }
    v.iter().map(|x| x / norm).collect()
}

// ---- Gram-Schmidt QR for random orthogonal matrix ----

fn random_orthogonal_matrix(rng: &mut Xoshiro256, d: usize) -> Vec<Vec<f64>> {
    // Generate d x d random Gaussian matrix, then orthogonalize via Gram-Schmidt
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(d);
    for _ in 0..d {
        let mut col = Vec::with_capacity(d);
        let mut i = 0;
        while i < d {
            let (a, b) = randn_pair(rng);
            col.push(a);
            i += 1;
            if i < d {
                col.push(b);
                i += 1;
            }
        }
        cols.push(col);
    }

    // Modified Gram-Schmidt
    for j in 0..d {
        // Normalize column j
        let norm: f64 = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for k in 0..d {
                cols[j][k] /= norm;
            }
        }
        // Subtract projection from subsequent columns
        for jj in (j + 1)..d {
            let dot: f64 = (0..d).map(|k| cols[j][k] * cols[jj][k]).sum();
            for k in 0..d {
                cols[jj][k] -= dot * cols[j][k];
            }
        }
    }

    cols
}

/// Apply orthogonal matrix Q (given as columns) to vector x: result = Q^T * x
fn apply_ortho_matrix(q_cols: &[Vec<f64>], x: &[f32]) -> Vec<f32> {
    let d = x.len();
    let mut result = vec![0.0f32; d];
    for (i, col) in q_cols.iter().enumerate() {
        let dot: f64 = (0..d).map(|k| col[k] * x[k] as f64).sum();
        result[i] = dot as f32;
    }
    result
}

/// Apply orthogonal matrix Q: result = Q * x (inverse of Q^T)
fn apply_ortho_matrix_inv(q_cols: &[Vec<f64>], x: &[f32]) -> Vec<f32> {
    let d = x.len();
    let mut result = vec![0.0f32; d];
    for (i, col) in q_cols.iter().enumerate() {
        for k in 0..d {
            result[k] += col[k] as f32 * x[i];
        }
    }
    result
}

/// Quantize-dequantize using a dense random orthogonal matrix instead of Hadamard.
fn quantize_with_random_rotation(
    x: &[f32],
    q_cols: &[Vec<f64>],
    codebook: &[f32],
    d: usize,
) -> Vec<f32> {
    // Rotate
    let rotated = apply_ortho_matrix(q_cols, x);

    // Compute norm
    let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-30 {
        return vec![0.0; d];
    }

    let inv_norm = 1.0 / norm;
    let scale = (d as f32).sqrt();
    let inv_scale = 1.0 / scale;

    // Quantize
    let mut reconstructed = vec![0.0f32; d];
    for (i, &v) in rotated.iter().enumerate() {
        let normalized = v * inv_norm;
        let scaled = normalized * scale;
        // Find nearest centroid
        let mut best_idx = 0;
        let mut best_dist = (scaled - codebook[0]).abs();
        for (j, &c) in codebook.iter().enumerate().skip(1) {
            let dist = (scaled - c).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = j;
            }
        }
        reconstructed[i] = codebook[best_idx] * inv_scale * norm;
    }

    // Inverse rotate
    apply_ortho_matrix_inv(q_cols, &reconstructed)
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / n as f32
}

// ====================================================================
// Test (a): MSE bounds
// ====================================================================

#[test]
fn test_mse_bounds() {
    let mut rng = Xoshiro256::new(42);

    for (bw, bw_name, mse_bound) in [
        (BitWidth::Two, "2-bit", 0.117),
        (BitWidth::Three, "3-bit", 0.030),
        (BitWidth::Four, "4-bit", 0.009),
    ] {
        let d = 128;
        let config = TurboQuantConfig {
            bit_width: bw,
            head_dim: d,
        };

        let n_vectors = 10_000;
        let mut total_mse = 0.0f64;

        for _ in 0..n_vectors {
            // Random vector ~ N(0, 1/d) per coordinate
            let x = random_normal_vec(&mut rng, d);
            let x_scaled: Vec<f32> = x.iter().map(|&v| v / (d as f32).sqrt()).collect();

            let (packed, norm) = turboquant_quantize(&x_scaled, &config).unwrap();
            let reconstructed = turboquant_dequantize(&packed, norm, &config).unwrap();

            let m = mse(&x_scaled, &reconstructed);
            total_mse += m as f64;
        }

        let avg_mse_per_coord = total_mse / n_vectors as f64;
        // The MSE bound is for quantizing N(0,1) directly.
        // Our vectors are N(0, 1/d), but after normalization and scaling by sqrt(d),
        // the quantization error maps back. The per-coordinate MSE should be
        // bounded by the N(0,1) MSE times (1/d) * norm^2_avg.
        // For unit-variance check, we look at the relative reconstruction quality.

        // Actually: let's compute MSE on the unit-sphere representation directly.
        // The paper's MSE bounds are for the normalized case.
        // MSE(x, x_hat) / ||x||^2 should be bounded.
        let mut total_rel_mse = 0.0f64;
        for _ in 0..n_vectors {
            let x = random_normal_vec(&mut rng, d);
            let x_scaled: Vec<f32> = x.iter().map(|&v| v / (d as f32).sqrt()).collect();
            let norm_sq: f32 = x_scaled.iter().map(|v| v * v).sum();

            let (packed, norm) = turboquant_quantize(&x_scaled, &config).unwrap();
            let reconstructed = turboquant_dequantize(&packed, norm, &config).unwrap();

            let err_sq: f32 = x_scaled
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();

            if norm_sq > 1e-10 {
                // MSE per coordinate in N(0,1) space = (err_sq / norm_sq) * d * (1/d) = err_sq/norm_sq
                // But the codebook MSE is per-coordinate: sum((x_i - c_i)^2) / d
                // After scaling from unit sphere by sqrt(d), the per-coord MSE in N(0,1) space is:
                // err_sq / (norm_sq) per coordinate would be the normalized error
                // The paper's bound applies to: E[||x - x_hat||^2 / ||x||^2] * d
                // which equals the per-coordinate MSE in the codebook domain.
                let per_coord_normalized = (err_sq as f64 / norm_sq as f64) * d as f64 / d as f64;
                total_rel_mse += per_coord_normalized;
            }
        }
        let avg_rel_mse = total_rel_mse / n_vectors as f64;

        println!(
            "{bw_name}: avg_mse_per_coord = {avg_mse_per_coord:.6}, \
             avg_rel_mse (||e||^2/||x||^2) = {avg_rel_mse:.6}, \
             bound = {mse_bound}"
        );

        // The relative MSE ||e||^2 / ||x||^2 should be bounded by the codebook MSE
        // for the N(0,1) distribution divided by d (since each coordinate contributes MSE/d).
        // Actually the total relative error = sum of per-coord quantization errors / ||x||^2
        // = sum_i (c_i - x_i_hat)^2 * (norm/sqrt(d))^2 / norm^2 = sum_i err_i^2 / d
        // where err_i is the quantization error in N(0,1) space.
        // So avg_rel_mse should be approximately equal to the codebook MSE.
        assert!(
            avg_rel_mse <= mse_bound * 1.15, // 15% tolerance for finite sample
            "GATE FAIL: {bw_name} relative MSE {avg_rel_mse:.6} exceeds bound {mse_bound} * 1.15 = {:.6}",
            mse_bound * 1.15
        );
    }
}

// ====================================================================
// Test (b): Round-trip at 4-bit
// ====================================================================

#[test]
fn test_roundtrip_4bit() {
    let d = 128;
    let config = TurboQuantConfig {
        bit_width: BitWidth::Four,
        head_dim: d,
    };

    let mut rng = Xoshiro256::new(123);
    let x = random_normal_vec(&mut rng, d);
    let x_scaled: Vec<f32> = x.iter().map(|&v| v / (d as f32).sqrt()).collect();

    let (packed, norm) = turboquant_quantize(&x_scaled, &config).unwrap();
    let reconstructed = turboquant_dequantize(&packed, norm, &config).unwrap();

    let m = mse(&x_scaled, &reconstructed);
    println!("4-bit round-trip MSE per coordinate: {m:.6}");
    assert!(
        m < 0.01,
        "4-bit round-trip MSE {m} >= 0.01"
    );
}

// ====================================================================
// Test (c): Hadamard orthogonality (FWHT applied twice = identity)
// ====================================================================

#[test]
fn test_hadamard_orthogonality() {
    for d in [128, 256, 512] {
        let original: Vec<f32> = (0..d).map(|i| (i as f32) * 0.01 - (d as f32 * 0.005)).collect();
        let mut data = original.clone();
        fwht_inplace(&mut data).unwrap();
        // Verify it changed
        let same = original
            .iter()
            .zip(data.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(!same, "FWHT should change the vector (d={d})");

        fwht_inplace(&mut data).unwrap();
        for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "FWHT roundtrip mismatch at d={d}, idx={i}: {a} vs {b}, diff={}",
                (a - b).abs()
            );
        }
        println!("Hadamard orthogonality verified for d={d}");
    }
}

// ====================================================================
// Test (d): Hadamard vs random rotation MSE comparison
// ====================================================================

#[test]
fn test_hadamard_vs_random_rotation() {
    let mut rng = Xoshiro256::new(999);
    let n_vectors = 1000;

    for (bw, bw_name, codebook) in [
        (BitWidth::Two, "2-bit", &CODEBOOK_2BIT[..]),
        (BitWidth::Three, "3-bit", &CODEBOOK_3BIT[..]),
        (BitWidth::Four, "4-bit", &CODEBOOK_4BIT[..]),
    ] {
        // Only test d=128 for speed (random orthogonal matrix is O(d^2))
        let d = 128;
        let config = TurboQuantConfig {
            bit_width: bw,
            head_dim: d,
        };

        // Generate one random orthogonal matrix (expensive for large d)
        let q = random_orthogonal_matrix(&mut rng, d);

        let mut hadamard_mse_total = 0.0f64;
        let mut random_mse_total = 0.0f64;

        for _ in 0..n_vectors {
            let x = random_normal_vec(&mut rng, d);
            let x_scaled: Vec<f32> = x.iter().map(|&v| v / (d as f32).sqrt()).collect();

            // Hadamard quantization
            let (packed, norm) = turboquant_quantize(&x_scaled, &config).unwrap();
            let recon_h = turboquant_dequantize(&packed, norm, &config).unwrap();
            hadamard_mse_total += mse(&x_scaled, &recon_h) as f64;

            // Random rotation quantization
            let recon_r = quantize_with_random_rotation(&x_scaled, &q, codebook, d);
            random_mse_total += mse(&x_scaled, &recon_r) as f64;
        }

        let h_avg = hadamard_mse_total / n_vectors as f64;
        let r_avg = random_mse_total / n_vectors as f64;
        let ratio = if r_avg > 1e-15 { h_avg / r_avg } else { 1.0 };

        println!(
            "{bw_name} d={d}: Hadamard MSE={h_avg:.6}, Random MSE={r_avg:.6}, ratio={ratio:.4}"
        );

        assert!(
            ratio <= 1.2,
            "GATE FAIL: {bw_name} Hadamard/Random MSE ratio {ratio:.4} > 1.2"
        );
    }
}

// ====================================================================
// Test (e): Gaussian vs Beta codebook validation
// ====================================================================

#[test]
fn test_gaussian_vs_beta_codebook() {
    use mlx_native::turboquant::compute_lloyd_max_beta_codebook;

    let mut rng = Xoshiro256::new(777);
    let n_vectors = 2000;

    for d in [128, 256, 512] {
        for (bits, bw, gauss_cb) in [
            (2usize, BitWidth::Two, &CODEBOOK_2BIT[..]),
            (3, BitWidth::Three, &CODEBOOK_3BIT[..]),
            (4, BitWidth::Four, &CODEBOOK_4BIT[..]),
        ] {
            let num_levels = 1 << bits;
            let _config = TurboQuantConfig {
                bit_width: bw,
                head_dim: d,
            };

            // Compute Beta-optimal codebook
            let beta_cb_f64 = compute_lloyd_max_beta_codebook(d, num_levels);
            // Scale Beta codebook: Beta coords are on [-1,1] with variance ~1/d,
            // so scale by sqrt(d) to match the N(0,1) domain of our Gaussian codebook.
            let beta_cb: Vec<f32> = beta_cb_f64
                .iter()
                .map(|&c| (c * (d as f64).sqrt()) as f32)
                .collect();

            let mut gauss_mse_total = 0.0f64;
            let mut beta_mse_total = 0.0f64;

            for _ in 0..n_vectors {
                let x = random_unit_vec(&mut rng, d);

                // Hadamard rotate
                let mut rotated = x.clone();
                fwht_inplace(&mut rotated).unwrap();

                let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
                let inv_norm = if norm > 1e-30 { 1.0 / norm } else { 1.0 };
                let scale = (d as f32).sqrt();
                let inv_scale = 1.0 / scale;

                // Gaussian codebook quantization
                let mut recon_g = vec![0.0f32; d];
                for (i, &v) in rotated.iter().enumerate() {
                    let normalized = v * inv_norm;
                    let scaled = normalized * scale;
                    let mut best = 0;
                    let mut best_d = f32::MAX;
                    for (j, &c) in gauss_cb.iter().enumerate() {
                        let dd = (scaled - c).abs();
                        if dd < best_d {
                            best_d = dd;
                            best = j;
                        }
                    }
                    recon_g[i] = gauss_cb[best] * inv_scale * norm;
                }
                fwht_inplace(&mut recon_g).unwrap();
                gauss_mse_total += mse(&x, &recon_g) as f64;

                // Beta codebook quantization (in same scaled domain)
                let mut recon_b = vec![0.0f32; d];
                for (i, &v) in rotated.iter().enumerate() {
                    let normalized = v * inv_norm;
                    let scaled = normalized * scale;
                    let mut best = 0;
                    let mut best_d = f32::MAX;
                    for (j, &c) in beta_cb.iter().enumerate() {
                        let dd = (scaled - c).abs();
                        if dd < best_d {
                            best_d = dd;
                            best = j;
                        }
                    }
                    recon_b[i] = beta_cb[best] * inv_scale * norm;
                }
                fwht_inplace(&mut recon_b).unwrap();
                beta_mse_total += mse(&x, &recon_b) as f64;
            }

            let g_avg = gauss_mse_total / n_vectors as f64;
            let b_avg = beta_mse_total / n_vectors as f64;
            let ratio = if b_avg > 1e-15 { g_avg / b_avg } else { 1.0 };

            println!(
                "{bits}-bit d={d}: Gaussian MSE={g_avg:.6}, Beta MSE={b_avg:.6}, ratio={ratio:.4}"
            );

            assert!(
                ratio <= 1.05,
                "GATE FAIL: {bits}-bit d={d} Gaussian/Beta MSE ratio {ratio:.4} > 1.05"
            );
        }
    }
}

// ====================================================================
// Test (f): Fixed channel split validation
// ====================================================================

#[test]
fn test_fixed_channel_split() {
    let mut rng = Xoshiro256::new(555);
    let n_vectors = 1000;

    for d in [128, 256, 512] {
        let quarter = d / 4;

        // For each vector, find the top-d/4 highest-magnitude coordinates after Hadamard
        let mut top_quarter_counts = vec![0u32; d];

        let mut variances = vec![0.0f64; d];
        let mut means = vec![0.0f64; d];

        for _ in 0..n_vectors {
            let x = random_unit_vec(&mut rng, d);
            let mut rotated = x;
            fwht_inplace(&mut rotated).unwrap();

            for (i, &v) in rotated.iter().enumerate() {
                means[i] += v as f64;
            }

            // Find top-d/4 by magnitude
            let mut mag_idx: Vec<(f32, usize)> = rotated
                .iter()
                .enumerate()
                .map(|(i, &v)| (v.abs(), i))
                .collect();
            mag_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(_, idx) in mag_idx.iter().take(quarter) {
                top_quarter_counts[idx] += 1;
            }
        }

        // Compute variance of per-coordinate magnitudes
        for i in 0..d {
            means[i] /= n_vectors as f64;
        }

        // Re-pass to compute variance (or use the mean-of-squares approach)
        let mut rng2 = Xoshiro256::new(555); // same seed
        for _ in 0..n_vectors {
            let x = random_unit_vec(&mut rng2, d);
            let mut rotated = x;
            fwht_inplace(&mut rotated).unwrap();

            for (i, &v) in rotated.iter().enumerate() {
                let diff = v as f64 - means[i];
                variances[i] += diff * diff;
            }
        }
        for v in variances.iter_mut() {
            *v /= n_vectors as f64;
        }

        let var_mean: f64 = variances.iter().sum::<f64>() / d as f64;
        let var_std: f64 = (variances
            .iter()
            .map(|v| (v - var_mean) * (v - var_mean))
            .sum::<f64>()
            / d as f64)
            .sqrt();

        // Overlap: if coordinates were uniformly selected, each would be in top-quarter
        // with probability 0.25. The expected count is n_vectors * 0.25 = 250.
        let expected_count = n_vectors as f64 * 0.25;
        let count_std: f64 = (top_quarter_counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected_count;
                diff * diff
            })
            .sum::<f64>()
            / d as f64)
            .sqrt();

        // Coefficient of variation: low means uniform spread
        let cv = count_std / expected_count;

        println!(
            "d={d}: var_mean={var_mean:.6}, var_std={var_std:.6}, \
             expected_top_count={expected_count:.0}, count_std={count_std:.1}, CV={cv:.4}"
        );

        // After Hadamard rotation, the top-quarter selection should be fairly uniform
        // (CV < 0.5 would indicate non-concentrated behavior)
        assert!(
            cv < 0.5,
            "d={d}: coefficient of variation {cv:.4} >= 0.5 — \
             top-quarter coordinates are too concentrated, fixed split may be problematic"
        );
        println!(
            "d={d}: Fixed channel split confirmed OK — \
             top-quarter coordinates are spread uniformly (CV={cv:.4})"
        );
    }
}

// ====================================================================
// Test (g): Codebook symmetry
// ====================================================================

#[test]
fn test_codebook_symmetry() {
    for (name, cb) in [
        ("2-bit", &CODEBOOK_2BIT[..]),
        ("3-bit", &CODEBOOK_3BIT[..]),
        ("4-bit", &CODEBOOK_4BIT[..]),
    ] {
        let n = cb.len();
        for i in 0..n / 2 {
            let sum = cb[i] + cb[n - 1 - i];
            assert!(
                sum.abs() < 1e-5,
                "{name} codebook not symmetric: c[{i}]={} + c[{}]={} = {sum}",
                cb[i],
                n - 1 - i,
                cb[n - 1 - i]
            );
        }
        println!("{name} codebook symmetry verified");
    }
}

// ====================================================================
// Additional: TwoPointFive bit-width smoke test
// ====================================================================

#[test]
fn test_two_point_five_bit() {
    let d = 128;
    let config = TurboQuantConfig {
        bit_width: BitWidth::TwoPointFive,
        head_dim: d,
    };

    let mut rng = Xoshiro256::new(314);
    let x = random_normal_vec(&mut rng, d);
    let x_scaled: Vec<f32> = x.iter().map(|&v| v / (d as f32).sqrt()).collect();

    let (packed, norm) = turboquant_quantize(&x_scaled, &config).unwrap();
    let reconstructed = turboquant_dequantize(&packed, norm, &config).unwrap();

    let m = mse(&x_scaled, &reconstructed);
    println!("2.5-bit round-trip MSE per coordinate: {m:.6}");
    // MSE should be between 2-bit and 3-bit bounds
    assert!(m < 0.2, "2.5-bit MSE {m} unreasonably high");
}

// ====================================================================
// Additional: Error handling tests
// ====================================================================

#[test]
fn test_fwht_non_power_of_two() {
    let mut data = vec![1.0f32; 100];
    assert!(fwht_inplace(&mut data).is_err());
}

#[test]
fn test_quantize_wrong_length() {
    let config = TurboQuantConfig {
        bit_width: BitWidth::Two,
        head_dim: 128,
    };
    let x = vec![0.0f32; 64]; // wrong length
    assert!(turboquant_quantize(&x, &config).is_err());
}
