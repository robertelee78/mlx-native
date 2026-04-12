//! Unit tests for GGML block-format quantized matmul GPU kernels.
//!
//! Tests Q4_0, Q8_0, Q6_K with:
//!   - Known values (hand-constructed blocks, exact verification)
//!   - Random values (CPU reference dequant + matmul)
//!   - Production shapes (M=1, N=4096, K=4096)

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice};

// --------------------------------------------------------------------------
// PRNG (matches other test files)
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
// GGML block packing helpers (quantization direction: f32 -> blocks)
// --------------------------------------------------------------------------

/// Pack f32 values into Q4_0 blocks.
/// Each block: 2 bytes f16 scale + 16 bytes nibbles = 18 bytes, 32 values.
/// Dequant: val = d * (nibble - 8)
fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0, "values must be multiple of 32");
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        // Find scale: d = max(|v|) / 7 (maps range [-8d, 7d])
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Write f16 scale
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());

        // Quantize to 4-bit unsigned with zero point at 8
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

/// CPU dequant for Q4_0 (reference).
fn dequant_q4_0(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 32 == 0);
    let blocks = k / 32;
    let mut out = Vec::with_capacity(k);
    for b in 0..blocks {
        let offset = b * 18;
        let d = half::f16::from_le_bytes([packed[offset], packed[offset + 1]]).to_f32();
        let qs = &packed[offset + 2..offset + 18];
        for i in 0..16 {
            let lo = (qs[i] & 0x0F) as f32;
            let hi = (qs[i] >> 4) as f32;
            out.push(d * (lo - 8.0));
            out.push(d * (hi - 8.0));
        }
    }
    // Fix ordering: the nibbles are stored as [lo_0, lo_1, ..., lo_15]
    // with hi nibbles interleaved. But our pack function stores:
    //   nibbles[i] = v0(value[i]) | v1(value[i+16]) << 4
    // So dequant order is: value[i] from lo nibble, value[i+16] from hi nibble.
    // Let's re-order to match.
    let mut result = Vec::with_capacity(k);
    for b in 0..blocks {
        let base = b * 32;
        for i in 0..16 {
            result.push(out[base + i * 2]); // lo nibble = values 0..15
        }
        for i in 0..16 {
            result.push(out[base + i * 2 + 1]); // hi nibble = values 16..31
        }
    }
    result
}

/// Pack f32 values into Q8_0 blocks.
/// Each block: 2 bytes f16 scale + 32 bytes int8 quants = 34 bytes, 32 values.
/// Dequant: val = d * qs[i]
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

/// CPU dequant for Q8_0 (reference).
fn dequant_q8_0(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 32 == 0);
    let blocks = k / 32;
    let mut out = Vec::with_capacity(k);
    for b in 0..blocks {
        let offset = b * 34;
        let d = half::f16::from_le_bytes([packed[offset], packed[offset + 1]]).to_f32();
        for i in 0..32 {
            let q = packed[offset + 2 + i] as i8;
            out.push(d * q as f32);
        }
    }
    out
}

/// Pack f32 values into Q6_K blocks.
/// Each block: 128 ql + 64 qh + 16 scales + 2 d = 210 bytes, 256 values.
/// Dequant: val = d * scale[sub] * (6bit_val - 32)
fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        // Q6_K has 16 sub-blocks of 16 values each.
        // Super-block scale d, sub-block scales[16].
        // 6-bit quants stored split: low 4 bits in ql, high 2 bits in qh.

        // Compute per-sub-block scales
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
            let iscale = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
            sub_scale_int[s] = iscale;
        }

        // Quantize values to 6-bit with zero point at 32
        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        // Pack into ql (lower 4 bits) and qh (upper 2 bits)
        // ql layout: 128 bytes = low 4 bits of each 6-bit value
        //   ql[i] for i<128: low nibble = q6[i]&0xF, stored as-is per byte
        // Actually the layout matches the kernel's access pattern:
        //   ql[0..128] stores lower 4 bits of all 256 values
        //   ql[i] = q6[i] & 0xF  for i in 0..128  (first 128 values)
        //   ql[i] = q6[i] & 0xF  ... but wait, 256 values need 256 entries
        //   Each ql byte stores TWO values: ql[i] = (q6[i]&0xF) | (q6[i+128]&0xF)<<4
        //   No wait, looking at kernel dequant:
        //     q1[l] & 0xF  and  q1[l] >> 4
        //   So ql stores pairs: lower nibble and upper nibble.
        //   ql[i] for i in 0..128 = (q6[2*group_offset + ...] & 0xF) in lower,
        //                           (q6[...] >> 4) in upper half...
        //
        // Let me match the kernel access pattern exactly. The kernel reads:
        //   q1 = x[i].ql + q_offset_l    (q_offset_l = 64*ip + l0)
        //   q2 = q1 + 32
        //   qh = x[i].qh + q_offset_h    (q_offset_h = 32*ip + l0)
        //   y_offset = 128*ip + l0
        //
        //   sums[0] += y[l+ 0] * ((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32)
        //   sums[1] += y[l+32] * ((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32)
        //   sums[2] += y[l+64] * ((q1[l]  >> 4) | ((qh[l] & 0x30) << 0)) - 32)
        //   sums[3] += y[l+96] * ((q2[l]  >> 4) | ((qh[l] & 0xC0) >> 2)) - 32)
        //
        // For ip=0, l0=0..28 (il=0..7, n=4):
        //   y_offset = l0, so y indices: l0, l0+32, l0+64, l0+96
        //   q_offset_l = l0, q_offset_h = l0
        //
        // For ip=1, l0=0..28:
        //   y_offset = 128+l0, so y indices: 128+l0, 160+l0, 192+l0, 224+l0
        //   q_offset_l = 64+l0, q_offset_h = 32+l0
        //
        // So the mapping from y-index to 6-bit value:
        //   y[l0+l]     -> ql[l0+l] low nibble    | qh[l0+l] bits 0-1 << 4
        //   y[l0+l+32]  -> ql[l0+l+32] low nibble | qh[l0+l] bits 2-3 << 2
        //   y[l0+l+64]  -> ql[l0+l] hi nibble      | qh[l0+l] bits 4-5 << 0
        //   y[l0+l+96]  -> ql[l0+l+32] hi nibble   | qh[l0+l] bits 6-7 >> 2
        //
        //   y[128+l0+l]     -> ql[64+l0+l] low  | qh[32+l0+l] bits 0-1 << 4
        //   y[128+l0+l+32]  -> ql[64+l0+l+32] lo| qh[32+l0+l] bits 2-3 << 2
        //   y[128+l0+l+64]  -> ql[64+l0+l] hi   | qh[32+l0+l] bits 4-5
        //   y[128+l0+l+96]  -> ql[64+l0+l+32] hi| qh[32+l0+l] bits 6-7 >> 2
        //
        // Scale mapping:
        //   is = 8*ip + l0/16
        //   sc[0] = scales[is], sc[2] = scales[is+2], sc[4] = scales[is+4], sc[6] = scales[is+6]
        //
        // For ip=0: is = l0/16 (0 or 1 for l0=0..28, n=4, il=0..7)
        //   Actually l0 = 4*il, so l0 = 0,4,8,12,16,20,24,28
        //   is = 0,0,0,0,1,1,1,1
        //   sc[0]=scales[0], sc[2]=scales[2], sc[4]=scales[4], sc[6]=scales[6]  (is=0)
        //   sc[0]=scales[1], sc[2]=scales[3], sc[4]=scales[5], sc[6]=scales[7]  (is=1)
        //
        // For ip=1: is = 8 + l0/16
        //   sc indices: 8,9 + offsets 0,2,4,6
        //
        // So the sub-block-to-scale mapping:
        //   Sub-block j (16 values from y[j*16..(j+1)*16]):
        //   j=0 (y[0..16]):   uses scales[0] (via is=0, sc[0])
        //   j=1 (y[16..32]):  uses scales[1] (via is=1, sc[0])
        //   j=2 (y[32..48]):  uses scales[2] (via is=0, sc[2])
        //   j=3 (y[48..64]):  uses scales[3] (via is=1, sc[2])
        //   j=4 (y[64..80]):  uses scales[4] (via is=0, sc[4])
        //   j=5 (y[80..96]):  uses scales[5] (via is=1, sc[4])
        //   j=6 (y[96..112]): uses scales[6] (via is=0, sc[6])
        //   j=7 (y[112..128]):uses scales[7] (via is=1, sc[6])
        //   j=8 (y[128..144]):uses scales[8]  (via is=8, sc[0])
        //   j=9 (y[144..160]):uses scales[9]  (via is=9, sc[0])
        //   j=10(y[160..176]):uses scales[10] (via is=8, sc[2])
        //   j=11(y[176..192]):uses scales[11] (via is=9, sc[2])
        //   j=12(y[192..208]):uses scales[12] (via is=8, sc[4])
        //   j=13(y[208..224]):uses scales[13] (via is=9, sc[4])
        //   j=14(y[224..240]):uses scales[14] (via is=8, sc[6])
        //   j=15(y[240..256]):uses scales[15] (via is=9, sc[6])

        // Now pack ql, qh based on the y->q mapping above.
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        // First half (ip=0): y indices 0..128
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                // y[l0+l] -> ql[ql_idx] low nibble
                let v0 = q6[l0_base + l];
                // y[l0+l+64] -> ql[ql_idx] hi nibble
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                // y[l0+l+32] -> ql[ql_idx+32] low nibble
                let v1 = q6[l0_base + l + 32];
                // y[l0+l+96] -> ql[ql_idx+32] hi nibble
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                // qh[ql_idx] stores high 2 bits of 4 values
                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        // Second half (ip=1): y indices 128..256
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                // y[128+l0+l] -> ql[64+l0+l] low nibble
                let v0 = q6[128 + l0_base + l];
                // y[128+l0+l+64] -> ql[64+l0+l] hi nibble
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                // y[128+l0+l+32] -> ql[64+l0+l+32] low nibble
                let v1 = q6[128 + l0_base + l + 32];
                // y[128+l0+l+96] -> ql[64+l0+l+32] hi nibble
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
        buf.extend_from_slice(
            &sub_scale_int
                .iter()
                .map(|&s| s as u8)
                .collect::<Vec<_>>(),
        );
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

/// CPU dequant for Q6_K (reference) — matches the kernel's access pattern.
fn dequant_q6_k(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 256 == 0);
    let blocks = k / 256;
    let mut out = vec![0.0f32; k];

    for b in 0..blocks {
        let offset = b * 210;
        let ql = &packed[offset..offset + 128];
        let qh = &packed[offset + 128..offset + 192];
        let scales = &packed[offset + 192..offset + 208];
        let d = half::f16::from_le_bytes([packed[offset + 208], packed[offset + 209]]).to_f32();

        let base = b * 256;

        // First half (ip=0): y indices 0..128
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let idx = l0_base + l;
                let is = l0_base / 16; // sub-block scale index offset

                let v0_lo = ql[idx] & 0x0F;
                let v0_hi = (qh[idx] & 0x03) << 4;
                let q0 = (v0_lo | v0_hi) as i8 - 32;
                let sc0 = scales[is] as i8;
                out[base + l0_base + l] = d * sc0 as f32 * q0 as f32;

                let v1_lo = ql[idx + 32] & 0x0F;
                let v1_hi = (qh[idx] & 0x0C) << 2;
                let q1 = (v1_lo | v1_hi) as i8 - 32;
                let sc2 = scales[is + 2] as i8;
                out[base + l0_base + l + 32] = d * sc2 as f32 * q1 as f32;

                let v2_lo = (ql[idx] >> 4) & 0x0F;
                let v2_hi = (qh[idx] & 0x30) >> 0;
                let q2 = (v2_lo | v2_hi) as i8 - 32;
                let sc4 = scales[is + 4] as i8;
                out[base + l0_base + l + 64] = d * sc4 as f32 * q2 as f32;

                let v3_lo = (ql[idx + 32] >> 4) & 0x0F;
                let v3_hi = (qh[idx] & 0xC0) >> 2;
                let q3 = (v3_lo | v3_hi) as i8 - 32;
                let sc6 = scales[is + 6] as i8;
                out[base + l0_base + l + 96] = d * sc6 as f32 * q3 as f32;
            }
        }

        // Second half (ip=1): y indices 128..256
        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let is = 8 + l0_base / 16;

                let v0_lo = ql[ql_idx] & 0x0F;
                let v0_hi = (qh[qh_idx] & 0x03) << 4;
                let q0 = (v0_lo | v0_hi) as i8 - 32;
                let sc0 = scales[is] as i8;
                out[base + 128 + l0_base + l] = d * sc0 as f32 * q0 as f32;

                let v1_lo = ql[ql_idx + 32] & 0x0F;
                let v1_hi = (qh[qh_idx] & 0x0C) << 2;
                let q1 = (v1_lo | v1_hi) as i8 - 32;
                let sc2 = scales[is + 2] as i8;
                out[base + 128 + l0_base + l + 32] = d * sc2 as f32 * q1 as f32;

                let v2_lo = (ql[ql_idx] >> 4) & 0x0F;
                let v2_hi = (qh[qh_idx] & 0x30) >> 0;
                let q2 = (v2_lo | v2_hi) as i8 - 32;
                let sc4 = scales[is + 4] as i8;
                out[base + 128 + l0_base + l + 64] = d * sc4 as f32 * q2 as f32;

                let v3_lo = (ql[ql_idx + 32] >> 4) & 0x0F;
                let v3_hi = (qh[qh_idx] & 0xC0) >> 2;
                let q3 = (v3_lo | v3_hi) as i8 - 32;
                let sc6 = scales[is + 6] as i8;
                out[base + 128 + l0_base + l + 96] = d * sc6 as f32 * q3 as f32;
            }
        }
    }
    out
}

// --------------------------------------------------------------------------
// CPU reference matmul: output = input @ dequant(weight)^T
// output[m][n] = sum_k(input[m][k] * dequant_weight[n][k])
// --------------------------------------------------------------------------

fn cpu_matvec(input: &[f32], dequant_weight: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += input[row * k + ki] * dequant_weight[col * k + ki];
            }
            output[row * n + col] = acc;
        }
    }
    output
}

// --------------------------------------------------------------------------
// Test helper: run GPU kernel and compare to CPU reference
// --------------------------------------------------------------------------

fn run_ggml_matmul_test(
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    dequant_weights: &[f32],
    input: &[f32],
    tolerance: f32,
    label: &str,
) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    // Allocate and fill input buffer
    let input_bytes = m * k * 4;
    let mut input_buf = device
        .alloc_buffer(input_bytes, DType::F32, vec![m, k])
        .expect("alloc input");
    {
        let sl = input_buf
            .as_mut_slice::<f32>()
            .expect("input mut slice");
        sl.copy_from_slice(input);
    }

    // Allocate and fill weight buffer
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    {
        let sl = weight_buf
            .as_mut_slice::<u8>()
            .expect("weight mut slice");
        sl.copy_from_slice(weight_bytes);
    }

    // Allocate output buffer
    let output_bytes = m * n * 4;
    let mut output_buf = device
        .alloc_buffer(output_bytes, DType::F32, vec![m, n])
        .expect("alloc output");

    // Zero output
    {
        let sl = output_buf
            .as_mut_slice::<f32>()
            .expect("output mut slice");
        for v in sl.iter_mut() {
            *v = 0.0;
        }
    }

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("GPU execution");

    // Read GPU output
    let gpu_output = output_buf
        .as_slice::<f32>()
        .expect("read output")
        .to_vec();

    // CPU reference
    let cpu_output = cpu_matvec(input, dequant_weights, m, n, k);

    // Compare
    let mut max_err: f32 = 0.0;
    for i in 0..gpu_output.len() {
        let err = (gpu_output[i] - cpu_output[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if err > tolerance {
            panic!(
                "{}: mismatch at index {}: GPU={} CPU={} err={} (tol={})",
                label, i, gpu_output[i], cpu_output[i], err, tolerance
            );
        }
    }
    eprintln!(
        "{}: PASS (max_err={:.6}, tol={}, M={}, N={}, K={})",
        label, max_err, tolerance, m, n, k
    );
}

// ==========================================================================
// Q4_0 tests
// ==========================================================================

#[test]
fn test_q4_0_known_values() {
    let k = 32usize;
    let n = 8usize; // must be multiple of 8 (align)
    let m = 1usize;

    // Create known weight values: simple ramp
    let mut weights_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            weights_f32[row * k + col] = (row as f32 * 0.1) * ((col as f32) - 16.0) / 16.0;
        }
    }

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        let row_data = &weights_f32[row * k..(row + 1) * k];
        weight_bytes.extend_from_slice(&pack_q4_0(row_data));
    }

    // Dequant for reference
    let mut dequant = Vec::new();
    for row in 0..n {
        let row_offset = row * 18;
        let row_bytes = &weight_bytes[row_offset..row_offset + 18];
        dequant.extend_from_slice(&dequant_q4_0(row_bytes, k));
    }

    let input = vec![1.0f32; k];

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q4_0,
        &weight_bytes,
        &dequant,
        &input,
        1e-4,
        "Q4_0 known values",
    );
}

#[test]
fn test_q4_0_random() {
    let k = 256usize;
    let n = 32usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(42, n * k);
    let input = pseudo_random_f32(123, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }

    let mut dequant = Vec::new();
    for row in 0..n {
        let block_size = 18usize;
        let blocks = k / 32;
        let row_offset = row * blocks * block_size;
        let row_bytes = &weight_bytes[row_offset..row_offset + blocks * block_size];
        dequant.extend_from_slice(&dequant_q4_0(row_bytes, k));
    }

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q4_0,
        &weight_bytes,
        &dequant,
        &input,
        1e-3,
        "Q4_0 random",
    );
}

#[test]
fn test_q4_0_production_shape() {
    let k = 4096usize;
    let n = 4096usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(1, n * k);
    let input = pseudo_random_f32(2, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }

    let mut dequant = Vec::new();
    for row in 0..n {
        let blocks = k / 32;
        let row_offset = row * blocks * 18;
        let row_bytes = &weight_bytes[row_offset..row_offset + blocks * 18];
        dequant.extend_from_slice(&dequant_q4_0(row_bytes, k));
    }

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q4_0,
        &weight_bytes,
        &dequant,
        &input,
        0.5, // wider tolerance for large accumulations with 4-bit quantization
        "Q4_0 production 4096x4096",
    );
}

// ==========================================================================
// Q8_0 tests
// ==========================================================================

#[test]
fn test_q8_0_known_values() {
    let k = 32usize;
    let n = 8usize;
    let m = 1usize;

    let mut weights_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            weights_f32[row * k + col] = (row as f32 + 1.0) * ((col as f32) - 16.0) / 32.0;
        }
    }

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }

    let mut dequant = Vec::new();
    for row in 0..n {
        let row_offset = row * 34;
        let row_bytes = &weight_bytes[row_offset..row_offset + 34];
        dequant.extend_from_slice(&dequant_q8_0(row_bytes, k));
    }

    let input = vec![1.0f32; k];

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q8_0,
        &weight_bytes,
        &dequant,
        &input,
        1e-4,
        "Q8_0 known values",
    );
}

#[test]
fn test_q8_0_random() {
    let k = 256usize;
    let n = 32usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(55, n * k);
    let input = pseudo_random_f32(66, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }

    let mut dequant = Vec::new();
    for row in 0..n {
        let blocks = k / 32;
        let row_offset = row * blocks * 34;
        let row_bytes = &weight_bytes[row_offset..row_offset + blocks * 34];
        dequant.extend_from_slice(&dequant_q8_0(row_bytes, k));
    }

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q8_0,
        &weight_bytes,
        &dequant,
        &input,
        1e-3,
        "Q8_0 random",
    );
}

#[test]
fn test_q8_0_production_shape() {
    let k = 4096usize;
    let n = 4096usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(77, n * k);
    let input = pseudo_random_f32(88, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }

    let mut dequant = Vec::new();
    for row in 0..n {
        let blocks = k / 32;
        let row_offset = row * blocks * 34;
        let row_bytes = &weight_bytes[row_offset..row_offset + blocks * 34];
        dequant.extend_from_slice(&dequant_q8_0(row_bytes, k));
    }

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q8_0,
        &weight_bytes,
        &dequant,
        &input,
        0.1, // Q8_0 has better precision than Q4_0
        "Q8_0 production 4096x4096",
    );
}

// ==========================================================================
// Q6_K tests
// ==========================================================================

#[test]
fn test_q6_k_known_values() {
    let k = 256usize;
    let n = 2usize; // must be multiple of 2 (align for Q6_K)
    let m = 1usize;

    let mut weights_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for col in 0..k {
            weights_f32[row * k + col] = (row as f32 + 1.0) * ((col as f32) - 128.0) / 256.0;
        }
    }

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }

    let dequant = dequant_q6_k(&weight_bytes, n * k);

    let input = vec![1.0f32; k];

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q6_K,
        &weight_bytes,
        &dequant,
        &input,
        1e-3,
        "Q6_K known values",
    );
}

#[test]
fn test_q6_k_random() {
    let k = 256usize;
    let n = 16usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(99, n * k);
    let input = pseudo_random_f32(100, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }

    let dequant = dequant_q6_k(&weight_bytes, n * k);

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q6_K,
        &weight_bytes,
        &dequant,
        &input,
        1e-2,
        "Q6_K random",
    );
}

#[test]
fn test_q6_k_production_shape() {
    let k = 4096usize;
    let n = 4096usize;
    let m = 1usize;

    let weights_f32 = pseudo_random_f32(200, n * k);
    let input = pseudo_random_f32(201, m * k);

    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }

    let dequant = dequant_q6_k(&weight_bytes, n * k);

    run_ggml_matmul_test(
        m, n, k,
        GgmlType::Q6_K,
        &weight_bytes,
        &dequant,
        &input,
        1.0, // wider for large accumulations
        "Q6_K production 4096x4096",
    );
}
