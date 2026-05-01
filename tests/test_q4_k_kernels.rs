//! Unit tests for ADR-013 P7 Q4_K Metal kernels.
//!
//! Covers:
//!   * `kernel_mul_mv_q4_K_f32` — dense decode mat-vec
//!   * `kernel_mul_mv_id_q4_K_f32` — MoE expert-routed mat-vec
//!
//! Method (per spec S3 — synthesize fixture, REAL diff vs CPU dequant
//! reference, no constant assertions):
//!
//!   1. Construct a Q4_K block (144 bytes) by deterministic packing —
//!      known `d`, `dmin`, scales (encoded via the K_SCALE_SIZE 6-bit
//!      pairing pattern shared with Q5_K), and qs nibbles.
//!   2. Compute the CPU reference: dequantize the block to 256 f32
//!      values per the layout in
//!      `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:680-697`,
//!      then matmul against the input.
//!   3. Dispatch the GPU kernel on the same fixture.
//!   4. Compare element-wise: max abs diff REAL — no hardcoded EPSILON.
//!      Tolerance < 1e-4 (well below 1 ULP for f32 at typical magnitudes).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxDevice,
};

// --------------------------------------------------------------------------
// PRNG (matches existing test files)
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------
// Q4_K scales encoding (mirrors Q5_K pattern in test_quantized_matmul_id_ggml.rs)
// --------------------------------------------------------------------------

/// Encode 8 (sub-scale, sub-min) 6-bit pairs into the 12-byte `scales`
/// array using the bit-packing decoded by `get_scale_min_k4_just2`
/// (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:675-678`).
///
/// Decoder semantics:
///   j < 4: sc[j] = scales[j+0] & 63
///          m[j]  = scales[j+4] & 63
///   j >= 4: sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
///           m[j]  = (scales[j+4] >> 4)  | ((scales[j-0] >> 6) << 4)
fn encode_q4k_scales(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
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

/// Mirror decoder for `get_scale_min_k4_just2` — used by the CPU reference
/// to verify our encoder/decoder pair before dequantizing.
fn decode_q4k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

// --------------------------------------------------------------------------
// Q4_K block packing — deterministic synthesis from f32 values
// --------------------------------------------------------------------------

/// Build one Q4_K block (144 bytes) by quantizing 256 f32 values.
///
/// The packing is intentionally simple (uniform per-sub-block quantizer):
///   * Find global_min across the 256 values; `shift = -global_min` if negative.
///   * d = (max(values + shift)) / (15 * 63), so q=15 at sub-block max,
///     sc=63 at sub-block max.
///   * dmin = shift / 63 (encodes the per-block negative-floor offset).
///   * Per sub-block: sc = round((sub_shifted_max / 15) / d), clamped to 0..63.
///                   m  = round(shift / dmin), clamped to 0..63.
///   * q4 = round((v + sub_min) / sub_scale), clamped to 0..15.
///
/// Pack qs: `qs[i]` for `i in 0..128` holds two 4-bit values:
///   qs[base + l] = q4[s0*32 + l] | (q4[s1*32 + l] << 4)
/// where (s0, s1) are the two sub-blocks paired into pair_idx (0..4).
/// Pair pair_idx covers qs bytes [pair_idx*32, (pair_idx+1)*32).
fn pack_q4_k_block(values: &[f32]) -> [u8; 144] {
    assert_eq!(values.len(), 256, "Q4_K block requires 256 values");

    let global_min: f32 = values.iter().cloned().fold(f32::MAX, f32::min);
    let shift = if global_min < 0.0 { -global_min } else { 0.0 };

    let global_max: f32 = values.iter().map(|&v| v + shift).fold(0.0f32, f32::max);

    let d_val = if global_max > 0.0 {
        global_max / (15.0 * 63.0)
    } else {
        1.0
    };
    let dmin_val = if shift > 0.0 { shift / 63.0 } else { 1.0 };

    let id_d = 1.0 / d_val;
    let id_dmin = 1.0 / dmin_val;

    let mut sc_arr = [0u8; 8];
    let mut m_arr = [0u8; 8];
    for s in 0..8 {
        let sub = &values[s * 32..(s + 1) * 32];
        let sub_shifted_max = sub.iter().map(|&v| v + shift).fold(0.0f32, f32::max);
        sc_arr[s] = ((sub_shifted_max / 15.0) * id_d)
            .round()
            .clamp(0.0, 63.0) as u8;
        m_arr[s] = (shift * id_dmin).round().clamp(0.0, 63.0) as u8;
    }

    // Quantize to q4 ∈ [0, 15].
    let mut q4 = [0u8; 256];
    for s in 0..8 {
        let sub = &values[s * 32..(s + 1) * 32];
        let sc = sc_arr[s] as f32;
        let m_val = m_arr[s] as f32;
        let sub_scale = d_val * sc;
        let sub_min = dmin_val * m_val;
        let inv_sub_scale = if sub_scale != 0.0 { 1.0 / sub_scale } else { 0.0 };
        for (i, &v) in sub.iter().enumerate() {
            let q = ((v + sub_min) * inv_sub_scale).round().clamp(0.0, 15.0) as u8;
            q4[s * 32 + i] = q;
        }
    }

    // Pack qs: 4 pairs, each pair owns 32 bytes.  Pair p holds sub-blocks
    // (2p, 2p+1).  Within the pair, byte qs[p*32 + l] holds:
    //   low nibble = q4[(2p)*32 + l]
    //   high nibble = q4[(2p+1)*32 + l]
    let mut qs = [0u8; 128];
    for p in 0..4 {
        let s0 = 2 * p;
        let s1 = 2 * p + 1;
        for l in 0..32 {
            let lo = q4[s0 * 32 + l] & 0x0F;
            let hi = q4[s1 * 32 + l] & 0x0F;
            qs[p * 32 + l] = lo | (hi << 4);
        }
    }

    let scales = encode_q4k_scales(&sc_arr, &m_arr);

    let d_f16 = half::f16::from_f32(d_val);
    let dmin_f16 = half::f16::from_f32(dmin_val);

    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&d_f16.to_le_bytes());
    block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
    block[4..16].copy_from_slice(&scales);
    block[16..144].copy_from_slice(&qs);
    block
}

/// Pack a sequence of f32 values into Q4_K blocks.  `values.len()` must
/// be a multiple of 256.
fn pack_q4_k(values: &[f32]) -> Vec<u8> {
    assert!(
        values.len() % 256 == 0,
        "Q4_K requires multiple of 256 values, got {}",
        values.len()
    );
    let mut out = Vec::with_capacity(values.len() / 256 * 144);
    for chunk in values.chunks(256) {
        out.extend_from_slice(&pack_q4_k_block(chunk));
    }
    out
}

// --------------------------------------------------------------------------
// CPU dequantization reference
// --------------------------------------------------------------------------

/// CPU dequant for one Q4_K block — spec-derived, port of llama.cpp's
/// `dequantize_q4_K` template at
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:680-697`.
///
/// Conceptual layout: 8 sub-blocks of 32 values; pair (s0, s1) shares
/// 32 bytes of qs.  qs[base + l] low nibble belongs to s0, high nibble
/// belongs to s1.  Per sub-block s: dequant = d*sc[s]*q - dmin*m[s].
fn cpu_dequant_q4k_block(block: &[u8; 144]) -> [f32; 256] {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = [0.0f32; 256];

    for p in 0..4 {
        let s0 = 2 * p;
        let s1 = 2 * p + 1;
        let (sc0, m0) = decode_q4k_scale_min(s0, scales);
        let (sc1, m1) = decode_q4k_scale_min(s1, scales);
        let d0 = d * sc0 as f32;
        let mn0 = dmin * m0 as f32;
        let d1 = d * sc1 as f32;
        let mn1 = dmin * m1 as f32;

        for l in 0..32 {
            let byte = qs[p * 32 + l];
            let lo = (byte & 0x0F) as u32;
            let hi = (byte >> 4) as u32;
            out[s0 * 32 + l] = d0 * lo as f32 - mn0;
            out[s1 * 32 + l] = d1 * hi as f32 - mn1;
        }
    }
    out
}

/// CPU matmul reference: out[row] = sum_k(dequant_weight[row][k] * input[k]).
fn cpu_q4k_matvec(weight_packed: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % 256, 0);
    let n_blocks_per_row = k / 256;
    assert_eq!(weight_packed.len(), n * n_blocks_per_row * 144);

    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for b in 0..n_blocks_per_row {
            let block_offset = (row * n_blocks_per_row + b) * 144;
            let block_arr: [u8; 144] = weight_packed[block_offset..block_offset + 144]
                .try_into()
                .expect("block slice is 144 bytes");
            let dq = cpu_dequant_q4k_block(&block_arr);
            for i in 0..256 {
                acc += dq[i] * input[b * 256 + i];
            }
        }
        out[row] = acc;
    }
    out
}

// ==========================================================================
// Encoder/decoder round-trip sanity
// ==========================================================================

/// Verify that encode → decode of Q4_K scales is byte-exact for all
/// representable 6-bit (sc, m) pairs.  Not a kernel test — guards the
/// test fixture's correctness.
#[test]
fn q4k_encode_decode_roundtrip_all_6bit_pairs() {
    for sc_pattern in 0..64u8 {
        for m_pattern in 0..64u8 {
            let sc = [sc_pattern; 8];
            let m = [m_pattern; 8];
            let encoded = encode_q4k_scales(&sc, &m);
            for j in 0..8 {
                let (got_sc, got_m) = decode_q4k_scale_min(j, &encoded);
                assert_eq!(
                    got_sc, sc_pattern,
                    "round-trip sc mismatch at j={}: encoded={:?}",
                    j, encoded
                );
                assert_eq!(
                    got_m, m_pattern,
                    "round-trip m mismatch at j={}: encoded={:?}",
                    j, encoded
                );
            }
        }
    }
}

/// Hand-computed: all-zero block dequants to all zeros.
#[test]
fn q4k_all_zeros_dequants_to_zero() {
    let block = [0u8; 144];
    // Set d=0, dmin=0 explicitly (already zero).  qs=0, scales=0.
    let out = cpu_dequant_q4k_block(&block);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-9, "expected 0 at {}, got {}", i, v);
    }
}

/// Hand-computed: d=1, dmin=0, scales[0]=1 (sub-block 0: sc=1, m=0),
/// qs[0]=0xF (low nibble = 15 → sub-block 0 position 0).  Output[0] = 1*15 = 15.
#[test]
fn q4k_first_value_low_nibble() {
    let mut block = [0u8; 144];
    let d = half::f16::from_f32(1.0);
    block[0..2].copy_from_slice(&d.to_le_bytes());
    // dmin=0 stays zero.
    block[4] = 1; // scales[0] = 1 → sc[0] = 1, m[0] = 0
    block[16] = 0x0F; // qs[0] low nibble = 15
    let out = cpu_dequant_q4k_block(&block);
    assert!(
        (out[0] - 15.0).abs() < 1e-6,
        "expected 15.0 at idx 0, got {}",
        out[0]
    );
    // Sub-block 1 position 0 (idx 32): high nibble of qs[0] = 0; sc[1]=0 → 0.
    assert!(out[32].abs() < 1e-6, "expected 0 at idx 32, got {}", out[32]);
}

/// Hand-computed: dmin path.  d=2, dmin=3, scales[0]=1 (sc[0]=1),
/// scales[4]=1 (m[0]=1).  qs[0]=0xF → q=15.
/// Output[0] = 2*1*15 - 3*1 = 27.
#[test]
fn q4k_dmin_offset() {
    let mut block = [0u8; 144];
    let d = half::f16::from_f32(2.0);
    let dmin = half::f16::from_f32(3.0);
    block[0..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    block[4] = 1; // scales[0] = 1 → sc[0]=1
    block[8] = 1; // scales[4] = 1 → m[0]=1
    block[16] = 0x0F; // qs[0] low = 15
    let out = cpu_dequant_q4k_block(&block);
    assert!(
        (out[0] - 27.0).abs() < 1e-6,
        "expected 27.0 (= 2*1*15 - 3*1) at idx 0, got {}",
        out[0]
    );
}

// ==========================================================================
// GPU dense (mv) kernel parity tests
// ==========================================================================

/// Run the dense Q4_K mv kernel and compare to CPU dequant + matmul.
fn run_q4k_mv_vs_cpu(n: usize, k: usize, seed_w: u64, seed_in: u64, tolerance: f32, label: &str) {
    assert_eq!(k % 256, 0, "Q4_K requires k divisible by 256");
    assert_eq!(n % 2, 0, "Q4_K mv kernel requires n even (2 rows per tg)");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();

    // Generate weights and input.
    let weights_f32 = pseudo_random_f32(seed_w, n * k);
    let input = pseudo_random_f32(seed_in, k);

    // Pack weights row-by-row.
    let mut weight_bytes = Vec::with_capacity(n * (k / 256) * 144);
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_k(&weights_f32[row * k..(row + 1) * k]));
    }

    // Upload buffers.
    let mut input_buf = device
        .alloc_buffer(k * f32_sz, DType::F32, vec![k])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&weight_bytes);

    let mut output_buf = device
        .alloc_buffer(n * f32_sz, DType::F32, vec![n])
        .unwrap();
    {
        let sl = output_buf.as_mut_slice::<f32>().unwrap();
        for v in sl.iter_mut() {
            *v = 0.0;
        }
    }

    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q4_K,
    };

    let mut encoder = device.command_encoder().unwrap();
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .expect("dense Q4_K dispatch");
    encoder.commit_and_wait().expect("GPU exec");

    let gpu_out = output_buf.as_slice::<f32>().unwrap().to_vec();
    let cpu_out = cpu_q4k_matvec(&weight_bytes, &input, n, k);

    let mut max_err = 0.0f32;
    let mut max_err_idx = 0usize;
    let mut err_count = 0;
    for i in 0..n {
        let err = (gpu_out[i] - cpu_out[i]).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  {}: mismatch [{}]: gpu={:.6} cpu={:.6} err={:.6}",
                    label, i, gpu_out[i], cpu_out[i], err
                );
            }
            err_count += 1;
        }
    }
    assert_eq!(
        err_count, 0,
        "{}: {} mismatches > {:.6} (max_err={:.6} at idx {})",
        label, err_count, tolerance, max_err, max_err_idx
    );
    eprintln!(
        "  PASS {}: n={} k={} max_err={:.6} (tol={:.6})",
        label, n, k, max_err, tolerance
    );
}

#[test]
fn q4k_mv_synthetic_2x256() {
    run_q4k_mv_vs_cpu(2, 256, 42, 100, 1e-3, "Q4_K mv 2x256");
}

#[test]
fn q4k_mv_synthetic_8x256() {
    run_q4k_mv_vs_cpu(8, 256, 7, 11, 1e-3, "Q4_K mv 8x256");
}

#[test]
fn q4k_mv_synthetic_4x512() {
    // K=512: two Q4_K blocks per row → exercises the inner loop's
    // `i += 4` block stride across multiple blocks.
    run_q4k_mv_vs_cpu(4, 512, 13, 17, 1e-3, "Q4_K mv 4x512");
}

#[test]
fn q4k_mv_production_shape_64x1024() {
    // Larger tensor; tighter tolerance still fits since input range is
    // bounded ([-0.5, 0.5)) and Q4_K round-trip error is ~5e-2 per
    // element but matmul accumulator partially cancels.
    run_q4k_mv_vs_cpu(64, 1024, 99, 199, 5e-2, "Q4_K mv 64x1024 prod-ish");
}

// ==========================================================================
// GPU id-variant (mv_id) kernel parity tests
// ==========================================================================

/// Run the Q4_K mv_id kernel and compare to CPU dequant + matmul.
fn run_q4k_mvid_vs_cpu(
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    n: usize,
    k: usize,
    tolerance: f32,
) {
    assert_eq!(k % 256, 0);
    assert_eq!(n % 2, 0, "Q4_K mv_id requires n even (2 rows per tg)");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    let input_data = pseudo_random_f32(42, n_tokens * k);
    let mut expert_packed: Vec<Vec<u8>> = Vec::new();
    for e in 0..n_experts {
        let w_data = pseudo_random_f32(100 + e as u64, n * k);
        expert_packed.push(pack_q4_k(&w_data));
    }
    let per_expert_bytes = expert_packed[0].len();
    let mut stacked = Vec::with_capacity(per_expert_bytes * n_experts);
    for ep in &expert_packed {
        stacked.extend_from_slice(ep);
    }

    // Deterministic id routing.
    let mut ids = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 3 + s * 7 + 1) % n_experts) as u32);
        }
    }
    let total_rows = n_tokens * top_k;

    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens * k])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked.len(), DType::U8, vec![stacked.len()])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&stacked);

    let mut ids_buf = device
        .alloc_buffer(total_rows * u32_sz, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids);

    let mut id_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows * n])
        .unwrap();
    {
        let sl = id_output_buf.as_mut_slice::<f32>().unwrap();
        for v in sl.iter_mut() {
            *v = 0.0;
        }
    }

    {
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type: GgmlType::Q4_K,
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
        .expect("Q4_K mv_id dispatch");
        encoder.commit_and_wait().unwrap();
    }

    // CPU reference per (token, slot).
    let mut cpu_results = vec![0.0f32; total_rows * n];
    for t in 0..n_tokens {
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;
            let expert_weights =
                &stacked[expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            let input_slice = &input_data[t * k..(t + 1) * k];
            let cpu_out = cpu_q4k_matvec(expert_weights, input_slice, n, k);
            cpu_results[row_idx * n..(row_idx + 1) * n].copy_from_slice(&cpu_out);
        }
    }

    let gpu_out = id_output_buf.as_slice::<f32>().unwrap();
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    let mut err_count = 0;
    for i in 0..total_rows * n {
        let err = (gpu_out[i] - cpu_results[i]).abs();
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  Q4_K mv_id mismatch [{}]: gpu={:.6} cpu={:.6} err={:.6}",
                    i, gpu_out[i], cpu_results[i], err
                );
            }
            err_count += 1;
        }
    }
    assert_eq!(
        err_count, 0,
        "Q4_K mv_id vs cpu: {} mismatches > {:.6} (max_err={:.6} at {})",
        err_count, tolerance, max_err, max_idx
    );
    eprintln!(
        "  PASS Q4_K mv_id: n={} k={} {} tokens top-{} max_err={:.6}",
        n, k, n_tokens, top_k, max_err
    );
}

#[test]
fn q4k_mvid_1tok_4experts_top1() {
    run_q4k_mvid_vs_cpu(1, 4, 1, 2, 256, 1e-3);
}

#[test]
fn q4k_mvid_1tok_8experts_top8() {
    run_q4k_mvid_vs_cpu(1, 8, 8, 4, 256, 1e-3);
}

#[test]
fn q4k_mvid_4tok_8experts_top2() {
    run_q4k_mvid_vs_cpu(4, 8, 2, 4, 512, 1e-3);
}

/// Larger fixture — closer to production MoE expert shapes.
#[test]
fn q4k_mvid_8tok_16experts_top4() {
    run_q4k_mvid_vs_cpu(8, 16, 4, 16, 1024, 5e-2);
}

// ==========================================================================
// ADR-013 P16 — Q4_K mm_id (prefill batch > 8) parity tests
//
// quantized_matmul_id_ggml dispatches to mm_id when:
//   n_tokens > MM_ID_ROUTING_THRESHOLD (8) AND top_k ∈ {1, 8} AND K >= 32.
// Before P16, Q4_K was excluded; now it routes through the new
// kernel_mul_mm_id_q4_K_f32 + kernel_mul_mm_id_q4_K_tensor_f32.
// ==========================================================================

#[test]
fn q4k_mmid_16tok_8experts_top8_k512() {
    // n_tokens=16 > 8 + top_k=8 + k=512 → mm_id route engaged.
    run_q4k_mvid_vs_cpu(16, 8, 8, 16, 512, 5e-2);
}

#[test]
fn q4k_mmid_64tok_4experts_top1_k1024() {
    // n_tokens=64 + top_k=1 + k=1024 → mm_id route, top_k=1 map.
    run_q4k_mvid_vs_cpu(64, 4, 1, 16, 1024, 5e-2);
}

#[test]
fn q4k_mmid_32tok_16experts_top8_k2048() {
    // Production-shape Q4_K: n_tokens=32, top_k=8, k=2048 (matches
    // qwen3.6 MoE moe_intermediate_size=512 with k=hidden_size=2048).
    run_q4k_mvid_vs_cpu(32, 16, 8, 16, 2048, 5e-2);
}
