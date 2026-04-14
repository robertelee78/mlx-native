//! TurboQuant KV cache compression — CPU reference implementation.
//!
//! Implements the TurboQuant_mse algorithm:
//! 1. Walsh-Hadamard rotation for incoherence
//! 2. Per-head norm extraction
//! 3. Lloyd-Max scalar quantization against N(0,1) codebooks
//!
//! This module is CPU-only math — no Metal GPU dispatch.

// ---- Lloyd-Max Codebooks for N(0,1) ----
//
// Precomputed via iterative Lloyd-Max algorithm with convergence tolerance 1e-12.
// Each codebook is symmetric around zero.

/// 2-bit Lloyd-Max centroids for N(0,1): 4 reconstruction levels.
pub const CODEBOOK_2BIT: [f32; 4] = [
    -1.5104176, -0.4527800, 0.4527800, 1.5104176,
];

/// 3-bit Lloyd-Max centroids for N(0,1): 8 reconstruction levels.
pub const CODEBOOK_3BIT: [f32; 8] = [
    -2.1519457, -1.3439093, -0.7560053, -0.2450942,
    0.2450942, 0.7560053, 1.3439093, 2.1519457,
];

/// 4-bit Lloyd-Max centroids for N(0,1): 16 reconstruction levels.
pub const CODEBOOK_4BIT: [f32; 16] = [
    -2.7325896, -2.0690172, -1.6180464, -1.2562312,
    -0.9423405, -0.6567591, -0.3880483, -0.1283950,
    0.1283950, 0.3880483, 0.6567591, 0.9423405,
    1.2562312, 1.6180464, 2.0690172, 2.7325896,
];

// ---- BitWidth enum ----

/// Quantization bit-width for TurboQuant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    /// 2-bit uniform: all coordinates use 4-level codebook.
    Two,
    /// 3-bit uniform: all coordinates use 8-level codebook.
    Three,
    /// 4-bit uniform: all coordinates use 16-level codebook.
    Four,
    /// 2.5-bit mixed: first d/4 coordinates at 3-bit, remaining 3d/4 at 2-bit.
    TwoPointFive,
}

/// Configuration for TurboQuant quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Quantization bit-width.
    pub bit_width: BitWidth,
    /// Head dimension (must be a power of 2: 128, 256, or 512).
    pub head_dim: usize,
}

// ---- Fast Walsh-Hadamard Transform ----

/// In-place normalized Fast Walsh-Hadamard Transform.
///
/// The normalization ensures H * H = I, so the inverse transform is the
/// same function applied again.
///
/// # Arguments
/// * `x` — mutable slice of length `n` where `n` is a power of 2.
///
/// # Returns
/// `Ok(())` on success, or an error if the length is not a power of 2.
pub fn fwht_inplace(x: &mut [f32]) -> crate::Result<()> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "FWHT requires power-of-two length, got {n}"
        )));
    }

    let mut h = 1;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += step;
        }
        h *= 2;
    }

    // Normalize so that H * H = I
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }

    Ok(())
}

// ---- Standard Normal PDF / CDF ----

/// Standard normal probability density function: phi(x) = exp(-x^2/2) / sqrt(2*pi).
#[inline]
fn std_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2*pi)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF using the Abramowitz & Stegun rational approximation
/// (formula 26.2.17, maximum error < 7.5e-8).
#[inline]
fn std_normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x_abs = x.abs();

    // Horner form of the rational approximation
    const P: f64 = 0.231_641_9;
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;

    let t = 1.0 / (1.0 + P * x_abs);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = B1 * t + B2 * t2 + B3 * t3 + B4 * t4 + B5 * t5;
    let phi = std_normal_pdf(x_abs);

    let result = 1.0 - phi * poly;

    if sign < 0.0 {
        1.0 - result
    } else {
        result
    }
}

// ---- Nearest centroid lookup ----

/// Find the index of the nearest centroid in a sorted codebook.
#[inline]
fn nearest_centroid(value: f32, codebook: &[f32]) -> u8 {
    // Binary-search style: codebook is sorted, find nearest by checking boundaries
    let n = codebook.len();
    if n <= 1 {
        return 0;
    }

    let mut best_idx = 0u8;
    let mut best_dist = (value - codebook[0]).abs();

    for (i, &c) in codebook.iter().enumerate().skip(1) {
        let dist = (value - c).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Get the codebook for a specific coordinate index under the given config.
#[inline]
fn codebook_for_coord(coord_idx: usize, config: &TurboQuantConfig) -> &'static [f32] {
    match config.bit_width {
        BitWidth::Two => &CODEBOOK_2BIT,
        BitWidth::Three => &CODEBOOK_3BIT,
        BitWidth::Four => &CODEBOOK_4BIT,
        BitWidth::TwoPointFive => {
            let boundary = config.head_dim / 4;
            if coord_idx < boundary {
                &CODEBOOK_3BIT // first d/4 channels at 3-bit
            } else {
                &CODEBOOK_2BIT // remaining 3d/4 at 2-bit
            }
        }
    }
}

/// Bits per index for a coordinate under the given config.
#[inline]
fn bits_for_coord(coord_idx: usize, config: &TurboQuantConfig) -> usize {
    match config.bit_width {
        BitWidth::Two => 2,
        BitWidth::Three => 3,
        BitWidth::Four => 4,
        BitWidth::TwoPointFive => {
            if coord_idx < config.head_dim / 4 {
                3
            } else {
                2
            }
        }
    }
}

// ---- Pack / Unpack indices ----

/// Pack variable-width indices into a byte vector using bit-packing.
///
/// Indices are packed MSB-first into consecutive bytes.
fn pack_indices(indices: &[u8], config: &TurboQuantConfig) -> Vec<u8> {
    let total_bits: usize = (0..indices.len())
        .map(|i| bits_for_coord(i, config))
        .sum();
    let num_bytes = (total_bits + 7) / 8;
    let mut packed = vec![0u8; num_bytes];

    let mut bit_offset = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let nbits = bits_for_coord(i, config);
        // Write `nbits` bits of `idx` starting at `bit_offset`
        for b in (0..nbits).rev() {
            let bit_val = (idx >> b) & 1;
            let byte_pos = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8);
            if byte_pos < packed.len() {
                packed[byte_pos] |= bit_val << bit_pos;
            }
            bit_offset += 1;
        }
    }

    packed
}

/// Unpack variable-width indices from a packed byte vector.
fn unpack_indices(packed: &[u8], config: &TurboQuantConfig) -> Vec<u8> {
    let d = config.head_dim;
    let mut indices = Vec::with_capacity(d);

    let mut bit_offset = 0usize;
    for i in 0..d {
        let nbits = bits_for_coord(i, config);
        let mut val = 0u8;
        for _ in 0..nbits {
            let byte_pos = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8);
            let bit_val = if byte_pos < packed.len() {
                (packed[byte_pos] >> bit_pos) & 1
            } else {
                0
            };
            val = (val << 1) | bit_val;
            bit_offset += 1;
        }
        indices.push(val);
    }

    indices
}

// ---- Quantize / Dequantize ----

/// Quantize a single head vector using TurboQuant_mse.
///
/// Steps:
/// 1. Apply FWHT (Walsh-Hadamard rotation) for incoherence
/// 2. Extract L2 norm
/// 3. Normalize to unit vector
/// 4. Quantize each coordinate against the appropriate Lloyd-Max codebook
/// 5. Pack indices
///
/// # Arguments
/// * `x` — input vector of length `config.head_dim`
/// * `config` — quantization configuration
///
/// # Returns
/// `(packed_indices, norm)` on success.
pub fn turboquant_quantize(
    x: &[f32],
    config: &TurboQuantConfig,
) -> crate::Result<(Vec<u8>, f32)> {
    let d = config.head_dim;
    if x.len() != d {
        return Err(crate::MlxError::InvalidArgument(format!(
            "Expected vector of length {d}, got {}",
            x.len()
        )));
    }
    if !d.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "head_dim must be power of 2, got {d}"
        )));
    }

    // 1. Copy and apply FWHT
    let mut rotated = x.to_vec();
    fwht_inplace(&mut rotated)?;

    // 2. Compute L2 norm of rotated vector (same as original since Hadamard is orthogonal)
    let norm_sq: f32 = rotated.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();

    if norm < 1e-30 {
        // Zero vector: all indices = 0, norm = 0
        let indices = vec![0u8; d];
        let packed = pack_indices(&indices, config);
        return Ok((packed, 0.0));
    }

    // 3. Normalize to unit vector on S^{d-1}
    let inv_norm = 1.0 / norm;
    for v in rotated.iter_mut() {
        *v *= inv_norm;
    }

    // 4. Quantize: each coordinate needs to be scaled to N(0,1) domain.
    // A unit vector on S^{d-1} has coordinates ~ N(0, 1/d) for large d.
    // Scale by sqrt(d) to map to N(0,1) for codebook lookup.
    let scale = (d as f32).sqrt();
    let mut indices = Vec::with_capacity(d);
    for (i, &v) in rotated.iter().enumerate() {
        let scaled = v * scale;
        let cb = codebook_for_coord(i, config);
        indices.push(nearest_centroid(scaled, cb));
    }

    // 5. Pack
    let packed = pack_indices(&indices, config);

    Ok((packed, norm))
}

/// Dequantize a TurboQuant-compressed head vector.
///
/// Steps:
/// 1. Unpack indices
/// 2. Look up centroid values, scale back from N(0,1) domain
/// 3. Multiply by norm
/// 4. Apply inverse FWHT (same as forward)
///
/// # Arguments
/// * `packed` — packed index bytes
/// * `norm` — the L2 norm stored during quantization
/// * `config` — quantization configuration
///
/// # Returns
/// Reconstructed vector of length `config.head_dim`.
pub fn turboquant_dequantize(
    packed: &[u8],
    norm: f32,
    config: &TurboQuantConfig,
) -> crate::Result<Vec<f32>> {
    let d = config.head_dim;
    if !d.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "head_dim must be power of 2, got {d}"
        )));
    }

    // 1. Unpack indices
    let indices = unpack_indices(packed, config);

    // 2. Look up centroids and scale back from N(0,1) to unit-sphere scale
    let inv_scale = 1.0 / (d as f32).sqrt();
    let mut reconstructed = Vec::with_capacity(d);
    for (i, &idx) in indices.iter().enumerate() {
        let cb = codebook_for_coord(i, config);
        let idx_usize = idx as usize;
        let centroid = if idx_usize < cb.len() {
            cb[idx_usize]
        } else {
            0.0 // fallback for out-of-range (shouldn't happen)
        };
        reconstructed.push(centroid * inv_scale * norm);
    }

    // 3. Apply inverse FWHT (same as forward since H^{-1} = H with normalization)
    fwht_inplace(&mut reconstructed)?;

    Ok(reconstructed)
}

// ---- Lloyd-Max computation utilities (used by tests for validation) ----

/// Compute Lloyd-Max codebook for N(0,1) with the given number of levels.
///
/// Returns the sorted centroid array. This is used in tests to validate the
/// hardcoded codebooks.
pub fn compute_lloyd_max_codebook(num_levels: usize) -> Vec<f64> {
    // Initialize with uniform quantile boundaries
    let mut boundaries = Vec::with_capacity(num_levels + 1);
    boundaries.push(-10.0_f64); // approx -inf
    for i in 1..num_levels {
        let p = i as f64 / num_levels as f64;
        boundaries.push(quantile_normal(p));
    }
    boundaries.push(10.0_f64); // approx +inf

    // Initial centroids from conditional expectations
    let mut centroids = vec![0.0_f64; num_levels];
    for i in 0..num_levels {
        let a = boundaries[i];
        let b = boundaries[i + 1];
        let prob = std_normal_cdf(b) - std_normal_cdf(a);
        if prob > 1e-30 {
            centroids[i] = (std_normal_pdf(a) - std_normal_pdf(b)) / prob;
        }
    }

    // Iterate
    for _iter in 0..50_000 {
        let old = centroids.clone();

        // Update boundaries to midpoints
        boundaries[0] = -10.0;
        for i in 1..num_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }
        *boundaries.last_mut().unwrap_or(&mut 0.0) = 10.0;

        // Update centroids
        for i in 0..num_levels {
            let a = boundaries[i];
            let b = boundaries[i + 1];
            let prob = std_normal_cdf(b) - std_normal_cdf(a);
            if prob > 1e-30 {
                centroids[i] = (std_normal_pdf(a) - std_normal_pdf(b)) / prob;
            }
        }

        // Check convergence
        let max_change = centroids
            .iter()
            .zip(old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_change < 1e-12 {
            break;
        }
    }

    centroids
}

/// Approximate quantile (inverse CDF) of N(0,1) using rational approximation.
///
/// Uses the Beasley-Springer-Moro algorithm.
fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return -10.0;
    }
    if p >= 1.0 {
        return 10.0;
    }

    // Rational approximation (Peter Acklam's algorithm)
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Compute Lloyd-Max codebook for Beta((d-1)/2, (d-1)/2) scaled to [-1, 1].
///
/// The exact distribution of a coordinate of a unit vector uniform on S^{d-1}
/// is Beta((d-1)/2, (d-1)/2) on [-1, 1]. For large d this converges to N(0, 1/d).
///
/// Uses numerical integration via trapezoidal rule for the conditional expectations.
pub fn compute_lloyd_max_beta_codebook(dim: usize, num_levels: usize) -> Vec<f64> {
    let alpha = (dim as f64 - 1.0) / 2.0;

    // Beta PDF on [-1,1] with parameters (alpha, alpha) — symmetric
    // f(x) = C * (1-x^2)^(alpha-1)  for x in [-1, 1]
    // where C normalizes to 1.

    // Use log-space for numerical stability
    let log_norm = log_beta_norm_const(alpha);

    let beta_pdf = |x: f64| -> f64 {
        if x <= -1.0 || x >= 1.0 {
            return 0.0;
        }
        let val = 1.0 - x * x;
        if val <= 0.0 {
            return 0.0;
        }
        (log_norm + (alpha - 1.0) * val.ln()).exp()
    };

    // Numerical CDF via cumulative trapezoidal integration
    let n_grid = 10_000;
    let grid_lo = -1.0_f64;
    let grid_hi = 1.0_f64;
    let dx = (grid_hi - grid_lo) / n_grid as f64;

    // Build CDF table
    let mut cdf_vals = vec![0.0_f64; n_grid + 1];
    let mut pdf_vals = vec![0.0_f64; n_grid + 1];
    for i in 0..=n_grid {
        let x = grid_lo + i as f64 * dx;
        pdf_vals[i] = beta_pdf(x);
    }
    for i in 1..=n_grid {
        cdf_vals[i] = cdf_vals[i - 1] + 0.5 * (pdf_vals[i - 1] + pdf_vals[i]) * dx;
    }
    // Normalize CDF to [0, 1]
    let cdf_total = cdf_vals[n_grid];
    if cdf_total > 1e-30 {
        for v in cdf_vals.iter_mut() {
            *v /= cdf_total;
        }
        for v in pdf_vals.iter_mut() {
            *v /= cdf_total;
        }
    }

    // Helper: interpolated CDF and conditional expectation on [a, b]
    let interp_cdf = |x: f64| -> f64 {
        let frac = (x - grid_lo) / dx;
        let idx = frac as usize;
        if idx >= n_grid {
            return 1.0;
        }
        let t = frac - idx as f64;
        cdf_vals[idx] * (1.0 - t) + cdf_vals[idx + 1] * t
    };

    let conditional_expectation = |a: f64, b: f64| -> f64 {
        // E[X | a <= X <= b] via numerical integration
        let prob = interp_cdf(b) - interp_cdf(a);
        if prob < 1e-30 {
            return (a + b) / 2.0;
        }

        let n_sub = 500;
        let sub_dx = (b - a) / n_sub as f64;
        let mut integral = 0.0_f64;
        for j in 0..=n_sub {
            let x = a + j as f64 * sub_dx;
            let w = if j == 0 || j == n_sub { 0.5 } else { 1.0 };
            let frac = (x - grid_lo) / dx;
            let idx = frac as usize;
            let pdf_val = if idx >= n_grid {
                0.0
            } else {
                let t = frac - idx as f64;
                pdf_vals[idx] * (1.0 - t) + pdf_vals[idx + 1] * t
            };
            integral += w * x * pdf_val * sub_dx;
        }
        integral / prob
    };

    // Initialize with uniform quantile boundaries
    let mut boundaries = Vec::with_capacity(num_levels + 1);
    boundaries.push(-1.0_f64);
    for i in 1..num_levels {
        let target_p = i as f64 / num_levels as f64;
        // Binary search for quantile
        let mut lo = -1.0_f64;
        let mut hi = 1.0_f64;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if interp_cdf(mid) < target_p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        boundaries.push((lo + hi) / 2.0);
    }
    boundaries.push(1.0_f64);

    // Initial centroids
    let mut centroids = vec![0.0_f64; num_levels];
    for i in 0..num_levels {
        centroids[i] = conditional_expectation(boundaries[i], boundaries[i + 1]);
    }

    // Lloyd-Max iteration
    for _iter in 0..5000 {
        let old = centroids.clone();

        // Update boundaries
        boundaries[0] = -1.0;
        for i in 1..num_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }
        if let Some(last) = boundaries.last_mut() {
            *last = 1.0;
        }

        // Update centroids
        for i in 0..num_levels {
            centroids[i] = conditional_expectation(boundaries[i], boundaries[i + 1]);
        }

        let max_change = centroids
            .iter()
            .zip(old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_change < 1e-10 {
            break;
        }
    }

    centroids
}

/// Log of the normalization constant for the symmetric Beta PDF on [-1, 1].
fn log_beta_norm_const(alpha: f64) -> f64 {
    // Beta(alpha, alpha) on [0,1] has norm B(alpha, alpha) = Gamma(alpha)^2 / Gamma(2*alpha)
    // On [-1,1] we scale by 1/2, so norm = B(alpha,alpha) * 2^(2*alpha-1)
    // log C = -log(B(alpha,alpha)) - (2*alpha-1)*log(2)
    //       = log(Gamma(2*alpha)) - 2*log(Gamma(alpha)) - (2*alpha-1)*log(2)
    ln_gamma(2.0 * alpha) - 2.0 * ln_gamma(alpha) - (2.0 * alpha - 1.0) * 2.0_f64.ln()
}

/// Lanczos approximation for ln(Gamma(x)), x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation with g=7, n=9
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut ag = COEFF[0];
    for i in 1..9 {
        ag += COEFF[i] / (x + i as f64);
    }

    let tmp = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn test_codebook_values_match_lloyd_max() {
        for (bits, hardcoded) in [
            (2, &CODEBOOK_2BIT[..]),
            (3, &CODEBOOK_3BIT[..]),
            (4, &CODEBOOK_4BIT[..]),
        ] {
            let computed = compute_lloyd_max_codebook(1 << bits);
            assert_eq!(computed.len(), hardcoded.len());
            for (i, (&h, &c)) in hardcoded.iter().zip(computed.iter()).enumerate() {
                let diff = (h as f64 - c).abs();
                assert!(
                    diff < 1e-4,
                    "{bits}-bit codebook mismatch at {i}: hardcoded={h}, computed={c}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_fwht_roundtrip() {
        let original: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let mut data = original.clone();
        fwht_inplace(&mut data).unwrap();
        fwht_inplace(&mut data).unwrap();
        for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "FWHT roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }
}
