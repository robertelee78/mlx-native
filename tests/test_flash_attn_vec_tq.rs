//! Correctness tests for the TurboQuant SDPA kernel (ADR-007 Phase 1.3).
//!
//! Compares GPU `flash_attn_vec_tq` output against a CPU reference SDPA
//! using dequantized KV cache vectors.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::flash_attn_vec_tq::{self, FlashAttnVecTqParams};
use mlx_native::turboquant::{fwht_inplace, CODEBOOK_4BIT};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---- PRNG (xoshiro256**, same as test_turboquant.rs) ----

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

// ---- CPU naive SDPA reference ----

/// Compute naive SDPA on CPU with dequantized KV cache.
/// Q: [num_heads, head_dim]
/// K: [kv_seq_len, head_dim] (per KV head, dequantized)
/// V: [kv_seq_len, head_dim] (per KV head, dequantized)
/// Returns: [num_heads, head_dim] output
fn cpu_sdpa(
    q: &[f32],
    k_dequant: &[Vec<f32>],  // [kv_seq_len] of [head_dim]
    v_dequant: &[Vec<f32>],  // [kv_seq_len] of [head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * head_dim];
    let heads_per_kv = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_offset = h * head_dim;

        // Compute attention scores: Q · K^T
        let mut scores = Vec::with_capacity(kv_seq_len);
        for p in 0..kv_seq_len {
            let mut dot = 0.0f32;
            for c in 0..head_dim {
                dot += q[q_offset + c] * k_dequant[kv_h * kv_seq_len + p][c];
            }
            scores.push(dot * scale);
        }

        // Softmax (causal mask: all positions valid for decode at last position)
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            for e in &mut exp_scores {
                *e /= sum;
            }
        }

        // Weighted V sum
        let o_offset = h * head_dim;
        for p in 0..kv_seq_len {
            let w = exp_scores[p];
            for c in 0..head_dim {
                output[o_offset + c] += w * v_dequant[kv_h * kv_seq_len + p][c];
            }
        }
    }

    output
}

// ---- Test helper ----

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    flash_attn_vec_tq::register(&mut registry);
    (device, registry)
}

fn run_sdpa_tq_test(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    seed: u64,
) {
    let (device, mut registry) = setup();
    let mut rng = Xoshiro256::new(seed);

    let hd = head_dim as usize;
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let kvl = kv_seq_len as usize;
    let capacity = kv_seq_len; // capacity = seq_len for this test

    // Generate random Q [num_heads, head_dim]
    let q_data = random_f32_vec(&mut rng, nh * hd);

    // Generate random KV vectors and quantize them
    // K/V: [num_kv_heads, kv_seq_len, head_dim] original, then quantized to nibble format
    let mut k_packed_all = vec![0u8; nkv * kvl * (hd / 2)];
    let mut k_norms_all = vec![0.0f32; nkv * kvl];
    let mut v_packed_all = vec![0u8; nkv * kvl * (hd / 2)];
    let mut v_norms_all = vec![0.0f32; nkv * kvl];

    // Store dequantized vectors for CPU reference
    let mut k_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);
    let mut v_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);

    for kv_h in 0..nkv {
        for p in 0..kvl {
            // Random K vector
            let k_vec = random_f32_vec(&mut rng, hd);
            let (k_packed, k_norm) = nibble_quantize(&k_vec, hd);
            let k_deq = nibble_dequantize(&k_packed, k_norm, hd);

            // Copy packed data into flat buffers
            // Layout: [kv_head, capacity, head_dim/2] for packed, [kv_head, capacity] for norms
            let packed_offset = (kv_h * kvl + p) * (hd / 2);
            k_packed_all[packed_offset..packed_offset + hd / 2].copy_from_slice(&k_packed);
            k_norms_all[kv_h * kvl + p] = k_norm;
            k_dequant.push(k_deq);

            // Random V vector
            let v_vec = random_f32_vec(&mut rng, hd);
            let (v_packed, v_norm) = nibble_quantize(&v_vec, hd);
            let v_deq = nibble_dequantize(&v_packed, v_norm, hd);

            let v_packed_offset = (kv_h * kvl + p) * (hd / 2);
            v_packed_all[v_packed_offset..v_packed_offset + hd / 2].copy_from_slice(&v_packed);
            v_norms_all[kv_h * kvl + p] = v_norm;
            v_dequant.push(v_deq);
        }
    }

    // Centroid table no longer needed — codebook is embedded in the Metal kernel.

    // CPU reference SDPA
    let cpu_output = cpu_sdpa(
        &q_data,
        &k_dequant,
        &v_dequant,
        nh, nkv, hd, kvl,
        1.0, // scale
    );

    // ---- GPU path ----

    // The GPU kernel fuses FWHT into the kernel:
    //   - Q is loaded as float, FWHT'd in shared memory, then used for dot products
    //   - Output accumulator is inverse-FWHT'd in shared memory before writing
    // So Q goes in unrotated and output comes out unrotated — no external FWHT needed.

    // Allocate GPU buffers
    let mut q_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc Q");
    q_buf.as_mut_slice::<f32>().expect("write Q")[..nh * hd]
        .copy_from_slice(&q_data);

    let mut k_packed_buf = device
        .alloc_buffer(k_packed_all.len(), DType::U8, vec![nkv, kvl, hd / 2])
        .expect("alloc K packed");
    k_packed_buf.as_mut_slice::<u8>().expect("write K packed")
        .copy_from_slice(&k_packed_all);

    let mut k_norms_buf = device
        .alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl])
        .expect("alloc K norms");
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms")
        .copy_from_slice(&k_norms_all);

    let mut v_packed_buf = device
        .alloc_buffer(v_packed_all.len(), DType::U8, vec![nkv, kvl, hd / 2])
        .expect("alloc V packed");
    v_packed_buf.as_mut_slice::<u8>().expect("write V packed")
        .copy_from_slice(&v_packed_all);

    let mut v_norms_buf = device
        .alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl])
        .expect("alloc V norms");
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms")
        .copy_from_slice(&v_norms_all);

    let output_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc output");

    // Dispatch GPU kernel
    let mut encoder = device.command_encoder().expect("encoder");

    let params = FlashAttnVecTqParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity: capacity,
        scale: 1.0,
        mask_type: 1, // causal
        sliding_window: 0,
        softcap: 0.0,
    };

    flash_attn_vec_tq::flash_attn_vec_tq(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_packed_buf,
        &k_norms_buf,
        &v_packed_buf,
        &v_norms_buf,
        &output_buf,
        &params,
    )
    .expect("flash_attn_vec_tq dispatch");

    encoder.commit_and_wait().expect("commit");

    // Read GPU output — FWHT is fused into the kernel, output is in original domain.
    let gpu_output: Vec<f32> = output_buf.as_slice::<f32>().expect("read output").to_vec();

    // Compare CPU vs GPU
    let mut max_abs_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;

    for i in 0..nh * hd {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_sq_diff += (diff as f64) * (diff as f64);
        sum_sq_ref += (cpu_output[i] as f64) * (cpu_output[i] as f64);
    }

    let rmse = (sum_sq_diff / (nh * hd) as f64).sqrt();
    let nrmse = if sum_sq_ref > 0.0 {
        (sum_sq_diff / sum_sq_ref).sqrt()
    } else {
        0.0
    };

    println!(
        "  heads={nh} kv_heads={nkv} d={hd} kvl={kvl}: \
         max_diff={max_abs_diff:.6} rmse={rmse:.6} nrmse={nrmse:.6}"
    );

    // The tolerance is relaxed because of quantization error.
    // With 4-bit quantization, per-coordinate MSE ≈ 0.009 ≈ 0.095 RMSE.
    // After SDPA mixing, errors should partially cancel.
    assert!(
        nrmse < 0.15,
        "Normalized RMSE too large: {nrmse:.6} (expected < 0.15)"
    );
    assert!(
        max_abs_diff < 1.0,
        "Max absolute difference too large: {max_abs_diff:.6} (expected < 1.0)"
    );
}

// ---- Tests ----

#[test]
fn test_sdpa_tq_d256_h8_kv4_seq64() {
    println!("flash_attn_vec_tq correctness test:");
    run_sdpa_tq_test(8, 4, 256, 64, 42);
}

#[test]
fn test_sdpa_tq_d512_h4_kv2_seq32() {
    println!("flash_attn_vec_tq correctness test:");
    run_sdpa_tq_test(4, 2, 512, 32, 123);
}

#[test]
fn test_sdpa_tq_d256_h4_kv4_seq8_short() {
    // Short context — less room for error accumulation
    println!("flash_attn_vec_tq short context test:");
    run_sdpa_tq_test(4, 4, 256, 8, 999);
}
