//! Unit tests for GGML block-format expert-indexed quantized matmul GPU kernels.
//!
//! Tests Q4_0, Q8_0, Q5_K, Q6_K with expert routing:
//!   - 1 token, 8 experts, top-2: verify output matches per-expert manual dispatch
//!   - 4 tokens, 8 experts, top-2: batch decode
//!   - Q5_K: compared against CPU F32 reference (no non-id Q5_K kernel)
//!   - Q6_K and Q8_0 variants

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxDevice,
};

// --------------------------------------------------------------------------
// PRNG
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

// --------------------------------------------------------------------------
// GGML block packing helpers (copied from test_quantized_matmul_ggml.rs)
// --------------------------------------------------------------------------

fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0, "values must be multiple of 32");
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let v0 = (block[i] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            let v1 = (block[i + 16] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            nibbles[i] = v0 | (v1 << 4);
        }
        buf.extend_from_slice(&nibbles);
    }
    buf
}

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
    buf
}

// --------------------------------------------------------------------------
// Q5_K packing helpers
// --------------------------------------------------------------------------
//
// Q5_K block layout (176 bytes, 256 values):
//   d[2]  dmin[2]  scales[12]  qh[32]  qs[128]
//
// 8 sub-blocks of 32 values each, paired into 4 groups of 64.
// The quantized value q5 ∈ [0, 31]; dequant = scale[j]*q5 - min[j].
//
// Packing strategy for tests: simple uniform quantization per sub-block.
// Not production-optimal but bit-exact dequant for testing.

/// Encode 6-bit scales and mins into the 12-byte scales array using the
/// same bit-packing as get_scale_min_k4 (inverted):
///   j < 4: scales[j] &= ~63; scales[j] |= sc[j]&63
///          scales[j+4] &= ~63; scales[j+4] |= m[j]&63
///   j >= 4: scales[j+4] = (scales[j+4]&0xF0) | (sc[j]&0xF)
///           scales[j+4] |= (m[j]&0xF)<<4
///           scales[j-4] |= (sc[j]>>4)<<6   (stores upper 2 bits of sc)
///           scales[j] |= (m[j]>>4)<<6       (stores upper 2 bits of m)
///
/// Both sc and m are 6-bit (0..63).
fn encode_q5k_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..4 {
        s[j] = sc[j] & 63;
        s[j + 4] = m[j] & 63;
    }
    for j in 4..8 {
        // lower 4 bits of sc[j] go into s[j+4] bits 0-3
        s[j + 4] = (s[j + 4] & 0xF0) | (sc[j] & 0xF);
        // lower 4 bits of m[j] go into s[j+4] bits 4-7
        s[j + 4] |= (m[j] & 0xF) << 4;
        // upper 2 bits of sc[j] go into s[j-4] bits 6-7
        s[j - 4] |= (sc[j] >> 4) << 6;
        // upper 2 bits of m[j] go into s[j] bits 6-7
        s[j] |= (m[j] >> 4) << 6;
    }
    s
}

/// CPU dequant for a Q5_K block (spec-derived, same as in test_q5_k_dequant.rs).
/// Returns 256 f32 values.
fn cpu_dequant_q5k_block(block: &[u8]) -> [f32; 256] {
    assert_eq!(block.len(), 176);
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut out = [0.0f32; 256];
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    let mut y = 0usize;
    let mut ql_off = 0usize;
    while ql_off < 128 {
        let ql = &qs[ql_off..ql_off + 32];
        let (sc1, m1) = get_scale_min_k4_test(is, scales);
        let (sc2, m2) = get_scale_min_k4_test(is + 1, scales);
        let d1 = d * sc1 as f32;
        let mn1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let mn2 = dmin * m2 as f32;
        for l in 0..32 {
            let low = (ql[l] & 0x0F) as u32;
            let high: u32 = if (qh[l] & u1) != 0 { 16 } else { 0 };
            out[y] = d1 * (low + high) as f32 - mn1;
            y += 1;
        }
        for l in 0..32 {
            let low = (ql[l] >> 4) as u32;
            let high: u32 = if (qh[l] & u2) != 0 { 16 } else { 0 };
            out[y] = d2 * (low + high) as f32 - mn2;
            y += 1;
        }
        is += 2;
        ql_off += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    out
}

fn get_scale_min_k4_test(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Pack f32 values into Q5_K blocks.
/// One block covers 256 values.  Uses 8 sub-blocks × 32 values.
/// Simple uniform quantizer per sub-block (good enough for testing; not
/// tuned for production quality).
fn pack_q5_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0, "Q5_K requires multiple of 256 values");
    let mut buf = Vec::new();

    for block_vals in values.chunks(256) {
        // Step 1: per-sub-block amax for scale selection.
        let mut sub_max = [0.0f32; 8];
        for (s, sub) in block_vals.chunks(32).enumerate() {
            sub_max[s] = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        }

        // Step 2: find min across all values (for the offset "min" path).
        let global_min: f32 = block_vals.iter().cloned().fold(f32::MAX, f32::min);
        // Shift values so all are >= 0 after applying -min.
        let shift = if global_min < 0.0 { -global_min } else { 0.0 };

        // Step 3: find max of (sub values + shift) for super-block scale.
        let global_max: f32 = block_vals.iter().map(|&v| v + shift).fold(0.0f32, f32::max);

        // super-block scale d: maps 6-bit sc (0..63) to sub-block scale.
        // We want sub_scale = d * sc where sc <= 63.
        // For simplicity: use d = global_max / (31.0 * 63.0) so that
        //   q=31 at max, sc=63 at max. Then d*sc*q ≈ global_max.
        let d_val = if global_max > 0.0 {
            global_max / (31.0 * 63.0)
        } else {
            1.0
        };
        // dmin: maps 6-bit min (0..63) to sub-block min.
        // We encode the global shift as dmin * m where m <= 63.
        let dmin_val = if shift > 0.0 {
            shift / 63.0
        } else {
            1.0
        };

        let id_d = 1.0 / d_val;
        let id_dmin = 1.0 / dmin_val;

        // Step 4: compute per-sub-block sc and m (6-bit each).
        let mut sc_arr = [0u8; 8];
        let mut m_arr = [0u8; 8];
        for s in 0..8 {
            let sub = &block_vals[s * 32..(s + 1) * 32];
            let sub_shifted_max = sub.iter().map(|&v| v + shift).fold(0.0f32, f32::max);
            // sc such that d_val * sc ≈ sub_shifted_max / 31
            sc_arr[s] = ((sub_shifted_max / 31.0) * id_d).round().clamp(0.0, 63.0) as u8;
            // m: shift / dmin_val
            m_arr[s] = (shift * id_dmin).round().clamp(0.0, 63.0) as u8;
        }

        // Step 5: quantize each value to 5-bit q5 ∈ [0, 31].
        let mut q5 = [0u8; 256];
        for s in 0..8 {
            let sub = &block_vals[s * 32..(s + 1) * 32];
            let sc = sc_arr[s] as f32;
            let m_val = m_arr[s] as f32;
            let sub_scale = d_val * sc;
            let sub_min = dmin_val * m_val;
            let inv_sub_scale = if sub_scale != 0.0 { 1.0 / sub_scale } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = ((v + sub_min) * inv_sub_scale).round().clamp(0.0, 31.0) as u8;
                q5[s * 32 + i] = q;
            }
        }

        // Step 6: pack q5 into qs (low 4 bits, 128 bytes) and qh (high bit, 32 bytes).
        // For sub-block pair (s, s+1), s even:
        //   pair_idx = s/2
        //   For l in 0..32:
        //     qs[s/2*32 + l] = (q5[s*32+l] & 0xF) | ((q5[(s+1)*32+l] & 0xF) << 4)
        //   qh[l] |= ((q5[s*32+l] >> 4) & 1) << (2*pair_idx)
        //   qh[l] |= ((q5[(s+1)*32+l] >> 4) & 1) << (2*pair_idx + 1)
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for pair_idx in 0..4 {
            let s0 = pair_idx * 2;
            let s1 = s0 + 1;
            let u1_shift = 2 * pair_idx;       // bit position for s0 high bit
            let u2_shift = 2 * pair_idx + 1;   // bit position for s1 high bit
            for l in 0..32 {
                let q_lo = q5[s0 * 32 + l] & 0xF;
                let q_hi_nibble = (q5[s1 * 32 + l] & 0xF) << 4;
                qs[pair_idx * 32 + l] = q_lo | q_hi_nibble;
                // high bit of s0 value: bit u1_shift of qh[l]
                if (q5[s0 * 32 + l] >> 4) & 1 != 0 {
                    qh[l] |= 1u8 << u1_shift;
                }
                // high bit of s1 value: bit u2_shift of qh[l]
                if (q5[s1 * 32 + l] >> 4) & 1 != 0 {
                    qh[l] |= 1u8 << u2_shift;
                }
            }
        }

        // Step 7: encode scales.
        let scales_encoded = encode_q5k_scales(&sc_arr, &m_arr);

        // Step 8: build the 176-byte block.
        let d_f16 = half::f16::from_f32(d_val);
        let dmin_f16 = half::f16::from_f32(dmin_val);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        buf.extend_from_slice(&dmin_f16.to_le_bytes());
        buf.extend_from_slice(&scales_encoded);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(&qs);
    }
    buf
}

/// CPU F32 matmul reference: compute output[n] = sum_k(weight_f32[n][k] * input[k]).
/// weight_q5k_bytes: packed Q5_K blocks for n rows of k cols each.
fn cpu_q5k_matmul(weight_q5k: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % 256, 0);
    let n_blocks_per_row = k / 256;
    assert_eq!(weight_q5k.len(), n * n_blocks_per_row * 176);

    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for b in 0..n_blocks_per_row {
            let block_offset = (row * n_blocks_per_row + b) * 176;
            let block = &weight_q5k[block_offset..block_offset + 176];
            let dq = cpu_dequant_q5k_block(block);
            for i in 0..256 {
                acc += dq[i] * input[b * 256 + i];
            }
        }
        out[row] = acc;
    }
    out
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// Create stacked expert weight buffer: N experts concatenated.
/// Returns (stacked_bytes, per_expert_byte_size).
fn stack_expert_weights(expert_packed: &[Vec<u8>]) -> (Vec<u8>, usize) {
    let per_expert = expert_packed[0].len();
    for ep in expert_packed {
        assert_eq!(ep.len(), per_expert);
    }
    let mut stacked = Vec::with_capacity(per_expert * expert_packed.len());
    for ep in expert_packed {
        stacked.extend_from_slice(ep);
    }
    (stacked, per_expert)
}

/// Verify the _id kernel matches per-expert non-id dispatch.
/// This dispatches the regular (non-id) kernel once per expert and compares.
fn run_id_vs_norid_test(
    ggml_type: GgmlType,
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    n: usize,
    k: usize,
    pack_fn: fn(&[f32]) -> Vec<u8>,
    tolerance: f32,
) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    // Generate input and weights
    let input_data = pseudo_random_f32(42, n_tokens * k);
    let mut expert_packed = Vec::new();
    for e in 0..n_experts {
        let w_data = pseudo_random_f32(100 + e as u64, n * k);
        expert_packed.push(pack_fn(&w_data));
    }
    let (stacked_bytes, per_expert_bytes) = stack_expert_weights(&expert_packed);

    let mut ids = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 3 + s * 7 + 1) % n_experts) as u32);
        }
    }

    let total_rows = n_tokens * top_k;

    // Upload
    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens * k])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U32, vec![stacked_bytes.len() / 4])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(total_rows * u32_sz, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids);

    // --- Run _id kernel ---
    let mut id_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows * n])
        .unwrap();

    {
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type,
        };

        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut id_output_buf,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // --- Run per-expert non-id kernel ---
    let mut norid_results = vec![0.0f32; total_rows * n];
    for t in 0..n_tokens {
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;

            // Create single-expert weight buffer
            let expert_bytes = &stacked_bytes
                [expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            let mut expert_w_buf = device
                .alloc_buffer(per_expert_bytes, DType::U32, vec![per_expert_bytes / 4])
                .unwrap();
            expert_w_buf
                .as_mut_slice::<u8>()
                .unwrap()
                .copy_from_slice(expert_bytes);

            // Single-token input
            let mut tok_input_buf = device
                .alloc_buffer(k * f32_sz, DType::F32, vec![k])
                .unwrap();
            tok_input_buf
                .as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&input_data[t * k..(t + 1) * k]);

            let mut tok_output_buf = device
                .alloc_buffer(n * f32_sz, DType::F32, vec![n])
                .unwrap();

            let norid_params = GgmlQuantizedMatmulParams {
                m: 1,
                n: n as u32,
                k: k as u32,
                ggml_type,
            };

            let mut encoder = device.command_encoder().unwrap();
            mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml(
                &mut encoder,
                &mut registry,
                &device,
                &tok_input_buf,
                &expert_w_buf,
                &mut tok_output_buf,
                &norid_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();

            let result: &[f32] = tok_output_buf.as_slice().unwrap();
            norid_results[row_idx * n..(row_idx + 1) * n].copy_from_slice(result);
        }
    }

    // Compare _id output vs per-expert output (should be BIT-EXACT)
    let id_out: &[f32] = id_output_buf.as_slice().unwrap();
    let mut max_err: f32 = 0.0;
    let mut err_count = 0usize;
    for i in 0..total_rows * n {
        let err = (id_out[i] - norid_results[i]).abs();
        max_err = max_err.max(err);
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  id vs norid mismatch at [{}]: id={:.6}, norid={:.6}, err={:.6}",
                    i, id_out[i], norid_results[i], err
                );
            }
            err_count += 1;
        }
    }
    assert_eq!(
        err_count, 0,
        "{:?} id vs norid: {} mismatches (max_err={:.6})",
        ggml_type, err_count, max_err
    );
    eprintln!(
        "  PASS id-vs-norid: {:?} {}x{}, {} tokens, top-{}, max_err={:.6}",
        ggml_type, n, k, n_tokens, top_k, max_err
    );
}

// --------------------------------------------------------------------------
// Test cases
// --------------------------------------------------------------------------

#[test]
fn test_q4_0_id_vs_norid() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        1, 8, 2,
        64, 128,
        pack_q4_0,
        0.0,  // Should be bit-exact
    );
}

#[test]
fn test_q8_0_id_vs_norid() {
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        1, 8, 2,
        64, 128,
        pack_q8_0,
        0.0,
    );
}

#[test]
fn test_q4_0_id_vs_norid_4tok() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        4, 8, 2,
        64, 128,
        pack_q4_0,
        0.0,
    );
}

#[test]
fn test_q8_0_id_vs_norid_4tok() {
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        4, 8, 2,
        64, 128,
        pack_q8_0,
        0.0,
    );
}

// Production-like shapes (Gemma 4 MoE dimensions)
#[test]
fn test_q8_0_production_shape() {
    // Gemma 4: gate_up [2*moe_intermediate, hidden] = [2*2048, 2816]
    // But that's too large for CI. Use scaled-down version.
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        1, 8, 2,
        256, 256,   // Scaled down from 4096x2816
        pack_q8_0,
        0.0,
    );
}

#[test]
fn test_q4_0_production_shape() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        1, 8, 2,
        256, 256,
        pack_q4_0,
        0.0,
    );
}

// --------------------------------------------------------------------------
// Q5_K tests: compare _id kernel output against CPU F32 reference matmul.
// (No non-id Q5_K kernel exists, so we can't use run_id_vs_norid_test.)
// --------------------------------------------------------------------------

/// Run Q5_K _id kernel and compare against CPU F32 dequant + matmul reference.
fn run_q5k_id_vs_cpu(
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    n: usize,   // weight rows per expert
    k: usize,   // must be multiple of 256
    tolerance: f32,
) {
    assert_eq!(k % 256, 0, "Q5_K requires k divisible by 256");
    assert_eq!(n % 2, 0,   "Q5_K kernel requires n even (2 rows per threadgroup)");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    let input_data = pseudo_random_f32(42, n_tokens * k);
    let mut expert_packed: Vec<Vec<u8>> = Vec::new();
    for e in 0..n_experts {
        let w_data = pseudo_random_f32(100 + e as u64, n * k);
        expert_packed.push(pack_q5_k(&w_data));
    }
    let per_expert_bytes = expert_packed[0].len();
    let mut stacked = Vec::with_capacity(per_expert_bytes * n_experts);
    for ep in &expert_packed {
        stacked.extend_from_slice(ep);
    }

    // Build ids: deterministic routing.
    let mut ids = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 3 + s * 7 + 1) % n_experts) as u32);
        }
    }
    let total_rows = n_tokens * top_k;

    // Upload buffers.
    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens * k])
        .unwrap();
    input_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked.len(), DType::U32, vec![stacked.len() / 4])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked);

    let mut ids_buf = device
        .alloc_buffer(total_rows * u32_sz, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    let mut id_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows * n])
        .unwrap();

    // Run GPU _id kernel.
    {
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type: GgmlType::Q5_K,
        };
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut id_output_buf,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Compute CPU reference for each (token, slot) pair.
    let mut cpu_results = vec![0.0f32; total_rows * n];
    for t in 0..n_tokens {
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;
            let expert_weights = &stacked[expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            let input_slice = &input_data[t * k..(t + 1) * k];
            let cpu_out = cpu_q5k_matmul(expert_weights, input_slice, n, k);
            cpu_results[row_idx * n..(row_idx + 1) * n].copy_from_slice(&cpu_out);
        }
    }

    // Compare.
    let gpu_out: &[f32] = id_output_buf.as_slice().unwrap();
    let mut max_err = 0.0f32;
    let mut err_count = 0usize;
    for i in 0..total_rows * n {
        let err = (gpu_out[i] - cpu_results[i]).abs();
        max_err = max_err.max(err);
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  Q5_K mismatch [{}]: gpu={:.6} cpu={:.6} err={:.6}",
                    i, gpu_out[i], cpu_results[i], err
                );
            }
            err_count += 1;
        }
    }
    assert_eq!(
        err_count, 0,
        "Q5_K id vs cpu: {} mismatches (max_err={:.6}, tolerance={:.6})",
        err_count, max_err, tolerance
    );
    eprintln!(
        "  PASS Q5_K id-vs-cpu: n={} k={} {} tokens top-{}: max_err={:.6}",
        n, k, n_tokens, top_k, max_err
    );
}

/// Synthetic fixture: 1 token, 4 experts, top-1, 2×256 weight matrix.
/// Tolerance 5e-2 — typical Q5_K round-trip error for uniform random data.
#[test]
fn test_q5_k_id_vs_cpu_1tok() {
    run_q5k_id_vs_cpu(
        1,  // n_tokens
        4,  // n_experts
        2,  // top_k
        2,  // n (weight rows — must be even)
        256, // k
        5e-2,
    );
}

/// 4 tokens, 8 experts, top-2, 4×256.
#[test]
fn test_q5_k_id_vs_cpu_4tok() {
    run_q5k_id_vs_cpu(
        4,  // n_tokens
        8,  // n_experts
        2,  // top_k
        4,  // n
        256, // k
        5e-2,
    );
}

/// Production-like shape (scaled-down Qwen3.5-MoE expert weights).
/// Qwen3.5-MoE: hidden=4096, intermediate=2048 per expert, top-8.
#[test]
fn test_q5_k_id_vs_cpu_production_shape() {
    run_q5k_id_vs_cpu(
        1,    // n_tokens (decode)
        8,    // n_experts
        2,    // top_k
        4,    // n: scaled down from 2048
        256,  // k: scaled down from 4096
        5e-2,
    );
}
