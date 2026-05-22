//! ADR-034 task #93 cont. 24 (2026-05-21) — parity test for the fused
//! `kernel_fused_gate_up_silu_q4_K_f32` kernel.
//!
//! Asserts byte-identical (within F32 tolerance) output vs the unfused
//! 3-dispatch sequence:
//!   1. `quantized_matmul_ggml(Q4_K, gate_w, x)` → tmp_gate
//!   2. `quantized_matmul_ggml(Q4_K, up_w, x)`   → tmp_up
//!   3. `dispatch_silu_mul(tmp_gate, tmp_up)`     → out_unfused
//!
//! Falsification gate at m∈{1,2,4}.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, non_snake_case)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_q4_K::{
    dispatch_fused_gate_up_silu_q4_K, FusedGateUpSiluQ4_KArgs,
};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

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

/// Pack a flat F32 array (length multiple of QK_K=256) into Q4_K super-blocks.
/// 144 bytes each: 2*sizeof(half) + 12 scales + 128 packed nibbles.
///
/// Mirrors llama.cpp `quantize_row_q4_K_ref`. Uses per-sub-block (32-element)
/// affine quantization with 6-bit signed scales and 6-bit unsigned mins,
/// packed into the 12-byte `scales` field via the standard k-mask layout.
fn pack_q4_K(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK_K == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK_K * 144);
    for block in values.chunks(QK_K) {
        // 8 sub-blocks of 32 elements each. For each sub-block compute
        // affine quant: q[i] = round((v[i] - min) / d), 4-bit unsigned.
        let mut sub_d  = [0.0f32; 8];
        let mut sub_m  = [0.0f32; 8];
        for s in 0..8 {
            let sub = &block[s*32 .. (s+1)*32];
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for &v in sub {
                if v < min { min = v; }
                if v > max { max = v; }
            }
            sub_m[s] = min;
            sub_d[s] = (max - min) / 15.0;
            if sub_d[s] == 0.0 { sub_d[s] = 1e-30; }
        }
        // Outer scales: per llama.cpp, d (super-scale) = max(sub_d) / 63
        // and dmin = max(-sub_m) / 63.
        let outer_d_max  = sub_d.iter().cloned().fold(0.0f32, f32::max);
        let outer_dm_max = sub_m.iter().cloned().fold(0.0f32, |acc, m| acc.max(-m));
        let d  = outer_d_max  / 63.0;
        let dm = outer_dm_max / 63.0;
        let id  = if d  != 0.0 { 1.0 / d  } else { 0.0 };
        let idm = if dm != 0.0 { 1.0 / dm } else { 0.0 };

        // Encode 6-bit unsigned scale + 6-bit unsigned (negated) min for
        // each sub-block, packed as in ggml-common.h Q4_K layout.
        let mut sc6 = [0u8; 8];
        let mut mn6 = [0u8; 8];
        for s in 0..8 {
            sc6[s] = (sub_d[s] * id).round().clamp(0.0, 63.0) as u8;
            mn6[s] = (-sub_m[s] * idm).round().clamp(0.0, 63.0) as u8;
        }

        // Pack 6-bit scales + mins into 12-byte `scales` field
        // (kmask1=0x3f3f, kmask2=0x0f0f, kmask3=0xc0c0 layout).
        // ggml-quants.c get_scale_min_k4 inverse:
        //   if (j < 4) { d = scales[j]&63; m = scales[j+4]&63; }
        //   else { d = (scales[j+4]&0xf) | ((scales[j-4]>>6)<<4);
        //          m = (scales[j+4]>> 4) | ((scales[j+0]>>6)<<4); }
        // We need to produce `scales[0..12]` so the inverse yields our sc6/mn6.
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j]     = sc6[j];           // low 6 bits = d
            scales[j + 4] = mn6[j];           // low 6 bits = m
        }
        for j in 4..8 {
            // scales[j+4] holds (m_high << 4) | d_low_4
            scales[j + 4] = (sc6[j] & 0xf) | ((mn6[j] & 0xf) << 4);
            // top 2 bits of scales[j-4] hold high 2 bits of d
            scales[j - 4] |= (sc6[j] >> 4) << 6;
            // top 2 bits of scales[j+0] hold high 2 bits of m
            scales[j]     |= (mn6[j] >> 4) << 6;
        }

        // Write d, dmin (half), scales, qs.
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(dm).to_bits().to_le_bytes());
        bytes.extend_from_slice(&scales);

        // Pack 256 quants as 128 bytes (nibble pairs).
        let mut qs = [0u8; QK_K/2];
        for s in 0..8 {
            let sub = &block[s*32 .. (s+1)*32];
            let sub_d_real = d * (sc6[s] as f32);
            let sub_m_real = dm * (mn6[s] as f32);
            let inv_sub_d = if sub_d_real != 0.0 { 1.0 / sub_d_real } else { 0.0 };
            // Q4_K nibble layout: for sub-block s, the 32 quants pack into
            // 16 bytes. Byte b in sub_block s has:
            //   lower nibble: quants[s*32 + b]
            //   upper nibble: quants[s*32 + 16 + b]
            // The Metal kernel reads via q_offset = 32*iq + 8*ir, where
            // iq=s/4 (0 or 1) and ir=s%4 selects which 8-byte stripe.
            // BUT the packing in qs[] follows a different stride — see
            // ggml-quants.c quantize_row_q4_K_ref:
            //   for (int j = 0; j < QK_K; j += 64) {
            //     for (int l = 0; l < 32; ++l) {
            //       q[l] = ... low nibble of (vals[j+l] - min) / d
            //       q[l] |= ... high nibble of (vals[j+l+32] - min) / d
            //     }
            //     q += 32;
            //   }
            // So for sub-block pair (s, s+1) where s%2==0: bytes [s*16 .. s*16+32)
            // hold first 32 vals as low nibbles, next 32 vals as high.
            // We handle this in the outer pack loop below.
            let _ = (sub, sub_d_real, sub_m_real, inv_sub_d);
        }
        // Re-do with correct ggml-quants pairing: pair sub-blocks (s, s+1) where
        // s%2==0 share a 32-byte stripe in qs.
        for pair in 0..4 {
            let s0 = pair * 2;
            let s1 = s0 + 1;
            let sub0_d = d * (sc6[s0] as f32);
            let sub0_m = dm * (mn6[s0] as f32);
            let sub1_d = d * (sc6[s1] as f32);
            let sub1_m = dm * (mn6[s1] as f32);
            let id0 = if sub0_d != 0.0 { 1.0 / sub0_d } else { 0.0 };
            let id1 = if sub1_d != 0.0 { 1.0 / sub1_d } else { 0.0 };
            for l in 0..32 {
                let v0 = block[s0*32 + l];
                let v1 = block[s1*32 + l];
                let q0 = ((v0 + sub0_m) * id0).round().clamp(0.0, 15.0) as u8;
                let q1 = ((v1 + sub1_m) * id1).round().clamp(0.0, 15.0) as u8;
                qs[pair * 32 + l] = q0 | (q1 << 4);
            }
        }
        bytes.extend_from_slice(&qs);
    }
    bytes
}

fn run_parity_at_m(m: u32) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let hidden_size: u32 = 256; // 1 super-block
    let intermediate_size: u32 = 64;

    let input = pseudo_random_f32(0x_BEEF_F00D, (hidden_size * m) as usize);
    let gate_w_f32 = pseudo_random_f32(0xDEAD_BEEF, (intermediate_size * hidden_size) as usize);
    let up_w_f32   = pseudo_random_f32(0xCAFE_F00D, (intermediate_size * hidden_size) as usize);

    let gate_q4_K = pack_q4_K(&gate_w_f32);
    let up_q4_K   = pack_q4_K(&up_w_f32);

    let mut input_buf = device
        .alloc_buffer(
            (hidden_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, hidden_size as usize],
        )
        .expect("alloc input");
    input_buf.as_mut_slice::<f32>().expect("input slice").copy_from_slice(&input);

    let mut gate_w_buf = device
        .alloc_buffer(
            gate_q4_K.len(),
            DType::F32,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc gate_w");
    gate_w_buf.as_mut_slice::<u8>().expect("gw slice").copy_from_slice(&gate_q4_K);

    let mut up_w_buf = device
        .alloc_buffer(
            up_q4_K.len(),
            DType::F32,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc up_w");
    up_w_buf.as_mut_slice::<u8>().expect("uw slice").copy_from_slice(&up_q4_K);

    // ---- Unfused reference ----
    let tmp_gate = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc tmp_gate");
    let tmp_up = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc tmp_up");
    let out_unfused = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc out_unfused");
    let mut params_buf_silu = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc silu params");
    let silu_n: u32 = intermediate_size * m;
    params_buf_silu.as_mut_slice::<u32>().expect("write params")[0] = silu_n;

    let mv_params = GgmlQuantizedMatmulParams {
        m,
        n: intermediate_size,
        k: hidden_size,
        ggml_type: GgmlType::Q4_K,
    };

    let mut enc = device.command_encoder().expect("encoder unfused");
    quantized_matmul_ggml(
        &mut enc, &mut registry, &device,
        &input_buf, &gate_w_buf, &tmp_gate, &mv_params,
    )
    .expect("gate matvec");
    quantized_matmul_ggml(
        &mut enc, &mut registry, &device,
        &input_buf, &up_w_buf, &tmp_up, &mv_params,
    )
    .expect("up matvec");
    enc.memory_barrier();
    dispatch_silu_mul(
        &mut enc, &mut registry, device.metal_device(),
        &tmp_gate, &tmp_up, &out_unfused, &params_buf_silu, silu_n,
    )
    .expect("silu_mul");
    enc.commit_and_wait().expect("commit unfused");

    let unfused_result: Vec<f32> = out_unfused.as_slice::<f32>().expect("read unfused").to_vec();

    // ---- Fused dispatch ----
    let out_fused = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc out_fused");

    let mut enc2 = device.command_encoder().expect("encoder fused");
    dispatch_fused_gate_up_silu_q4_K(
        &mut enc2, &mut registry, &device,
        &gate_w_buf, &up_w_buf, &input_buf, &out_fused,
        FusedGateUpSiluQ4_KArgs {
            m,
            intermediate_size,
            hidden_size,
        },
    )
    .expect("fused dispatch");
    enc2.commit_and_wait().expect("commit fused");

    let fused_result: Vec<f32> = out_fused.as_slice::<f32>().expect("read fused").to_vec();

    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in fused_result.iter().zip(unfused_result.iter()).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs { max_abs = abs; }
        if a.to_bits() != b.to_bits() && i < 5 {
            eprintln!(
                "[diff @ row {i}] fused={a:.6e} ({:#010x}) unfused={b:.6e} ({:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }
    eprintln!("Q4_K m={m}: max_abs_diff={max_abs:.3e}");
    assert!(
        max_abs < 1e-5,
        "fused Q4_K vs unfused max_abs_diff {max_abs:.3e} exceeds 1e-5 (m={m})"
    );
}

#[test]
fn fused_q4_K_m_eq_1_byte_identical() { run_parity_at_m(1); }

#[test]
fn fused_q4_K_m_eq_2_byte_identical() { run_parity_at_m(2); }

#[test]
fn fused_q4_K_m_eq_4_byte_identical() { run_parity_at_m(4); }
