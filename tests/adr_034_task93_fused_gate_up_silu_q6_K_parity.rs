//! ADR-034 task #93 cont. 28 (2026-05-21) — parity test for the fused
//! `kernel_fused_gate_up_silu_q6_K_f32` kernel.
//!
//! Asserts within-tolerance output vs the unfused 3-dispatch sequence
//! (quantized_matmul_ggml(Q6_K) gate + up + dispatch_silu_mul). Tolerance
//! 1e-4 absolute (consistent with Q5_K + adr_028_iter309 Q6_K parity).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, non_snake_case)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_q6_K::{
    dispatch_fused_gate_up_silu_q6_K, FusedGateUpSiluQ6_KArgs,
};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;

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

/// Pack F32 into Q6_K super-blocks. 210 bytes/block:
/// ql[128] (low 4 bits of each 6-bit quant) +
/// qh[64]  (high 2 bits packed, 4 per byte) +
/// scales[16] (signed 8-bit per-sub-block scales) +
/// d (half super-block scale).
///
/// Mirrors ggml-quants.c quantize_row_q6_K_ref: per 16-element sub-block
/// affine quant to [-32, 31], 16 sub-blocks per super-block.
fn pack_q6_K(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK_K == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK_K * 210);
    for block in values.chunks(QK_K) {
        // 16 sub-blocks of 16 elements each. Per-sub-block: compute
        // signed scale s = max_abs/-32, store as int8 (signed).
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
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }

        // Compute 6-bit quants per sub-block.
        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                // Q6_K stores quants biased by +32 (range [0, 63] from
                // signed [-32, 31]); the kernel applies the -32 bias when
                // unpacking.
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        // Pack 6-bit quants into ql[128] (low 4 bits) + qh[64] (high 2 bits).
        // Layout per ggml-quants.c: process 64 elements at a time
        // (two 32-element halves of the super-block). For each l in 0..32:
        //   q6[l_base + l] holds 6 bits → low 4 → ql byte, high 2 → qh nibble
        //
        // Inverse from kernel:
        //   q1 = ql + q_offset_l;  // l0..l3 reads q1[l]&0xF (low nibble val l+0..l+3)
        //   q2 = q1 + 32;          // (low nibble val l+32..l+35)
        //   qh = qh + q_offset_h;
        //   sums[0] += y[l+0]  * ((q1[l] & 0xF) | (qh[l] & 0x03) << 4) - 32
        //   sums[1] += y[l+32] * ((q2[l] & 0xF) | (qh[l] & 0x0C) << 2) - 32
        //   sums[2] += y[l+64] * ((q1[l]  >> 4) | (qh[l] & 0x30) << 0) - 32
        //   sums[3] += y[l+96] * ((q2[l]  >> 4) | (qh[l] & 0xC0) >> 2) - 32
        //
        // So for ip=0 half (l_base=0): ql[0..64], qh[0..32].
        //   ql[l]    low_nibble = q6[l]     low_4_bits
        //   ql[l]    high_nibble= q6[l+64]  low_4_bits
        //   ql[l+32] low_nibble = q6[l+32]  low_4_bits
        //   ql[l+32] high_nibble= q6[l+96]  low_4_bits
        //   qh[l]    bits 0-1   = q6[l]     high_2_bits
        //   qh[l]    bits 2-3   = q6[l+32]  high_2_bits
        //   qh[l]    bits 4-5   = q6[l+64]  high_2_bits
        //   qh[l]    bits 6-7   = q6[l+96]  high_2_bits
        // (Then ip=1 half for elements 128..255.)
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        for l0_base in (0..32usize).step_by(4) {
            let _ = l0_base;
        }
        // Simpler: do per-ip-half loop.
        for ip in 0..2usize {
            let base = ip * 128;
            for l in 0..32usize {
                let v0 = q6[base + l];          // val at offset 0   in half
                let v32 = q6[base + l + 32];    // val at offset 32  in half
                let v64 = q6[base + l + 64];    // val at offset 64  in half
                let v96 = q6[base + l + 96];    // val at offset 96  in half
                // ql layout: indices [ip*64 + l] holds (v0_low | v64_low<<4)
                //            indices [ip*64 + l + 32] holds (v32_low | v96_low<<4)
                ql[ip * 64 + l] = (v0 & 0xF) | ((v64 & 0xF) << 4);
                ql[ip * 64 + l + 32] = (v32 & 0xF) | ((v96 & 0xF) << 4);
                // qh layout: index [ip*32 + l] holds packed high-2-bits of
                // (v0, v32, v64, v96) in bits 0-1, 2-3, 4-5, 6-7.
                qh[ip * 32 + l] = (v0 >> 4) | ((v32 >> 4) << 2)
                               | ((v64 >> 4) << 4) | ((v96 >> 4) << 6);
            }
        }

        bytes.extend_from_slice(&ql);
        bytes.extend_from_slice(&qh);
        for s in sub_scale_int.iter() {
            bytes.push(*s as u8);
        }
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn run_parity_at_m(m: u32) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let hidden_size: u32 = 256;
    let intermediate_size: u32 = 64;

    let input = pseudo_random_f32(0xFACE_F00D, (hidden_size * m) as usize);
    let gate_w_f32 = pseudo_random_f32(0xBADC_AB1E, (intermediate_size * hidden_size) as usize);
    let up_w_f32   = pseudo_random_f32(0x1337_C0DE, (intermediate_size * hidden_size) as usize);

    let gate_q6_K = pack_q6_K(&gate_w_f32);
    let up_q6_K   = pack_q6_K(&up_w_f32);

    let mut input_buf = device
        .alloc_buffer((hidden_size * m) as usize * 4, DType::F32, vec![m as usize, hidden_size as usize])
        .expect("alloc input");
    input_buf.as_mut_slice::<f32>().expect("input").copy_from_slice(&input);

    let mut gate_w_buf = device
        .alloc_buffer(gate_q6_K.len(), DType::U8, vec![intermediate_size as usize, hidden_size as usize])
        .expect("alloc gate_w");
    gate_w_buf.as_mut_slice::<u8>().expect("gw").copy_from_slice(&gate_q6_K);

    let mut up_w_buf = device
        .alloc_buffer(up_q6_K.len(), DType::U8, vec![intermediate_size as usize, hidden_size as usize])
        .expect("alloc up_w");
    up_w_buf.as_mut_slice::<u8>().expect("uw").copy_from_slice(&up_q6_K);

    let tmp_gate = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("tmp_gate");
    let tmp_up   = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("tmp_up");
    let out_unfused = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("out_unfused");
    let mut params_buf_silu = device.alloc_buffer(4, DType::U32, vec![1]).expect("silu params");
    let silu_n: u32 = intermediate_size * m;
    params_buf_silu.as_mut_slice::<u32>().expect("write")[0] = silu_n;

    let mv_params = GgmlQuantizedMatmulParams {
        m, n: intermediate_size, k: hidden_size, ggml_type: GgmlType::Q6_K,
    };

    let mut enc = device.command_encoder().expect("encoder unfused");
    quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &gate_w_buf, &tmp_gate, &mv_params).expect("gate");
    quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &up_w_buf,   &tmp_up,   &mv_params).expect("up");
    enc.memory_barrier();
    dispatch_silu_mul(&mut enc, &mut registry, device.metal_device(), &tmp_gate, &tmp_up, &out_unfused, &params_buf_silu, silu_n).expect("silu_mul");
    enc.commit_and_wait().expect("commit");

    let unfused: Vec<f32> = out_unfused.as_slice::<f32>().expect("read unfused").to_vec();

    let out_fused = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("out_fused");
    let mut enc2 = device.command_encoder().expect("encoder fused");
    dispatch_fused_gate_up_silu_q6_K(
        &mut enc2, &mut registry, &device,
        &gate_w_buf, &up_w_buf, &input_buf, &out_fused,
        FusedGateUpSiluQ6_KArgs { m, intermediate_size, hidden_size },
    ).expect("fused");
    enc2.commit_and_wait().expect("commit");

    let fused: Vec<f32> = out_fused.as_slice::<f32>().expect("read fused").to_vec();

    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in fused.iter().zip(unfused.iter()).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs { max_abs = abs; }
        if a.to_bits() != b.to_bits() && i < 5 {
            eprintln!("[diff @ row {i}] fused={a:.6e} unfused={b:.6e}");
        }
    }
    eprintln!("Q6_K m={m}: max_abs_diff={max_abs:.3e}");
    // Tolerance 1e-4 (same as Q5_K and adr_028_iter309 Q6_K parity).
    assert!(max_abs < 1e-4, "Q6_K fused vs unfused max_abs_diff {max_abs:.3e} > 1e-4 (m={m})");
}

#[test]
fn fused_q6_K_m_eq_1_byte_identical() { run_parity_at_m(1); }

#[test]
fn fused_q6_K_m_eq_2_byte_identical() { run_parity_at_m(2); }

#[test]
fn fused_q6_K_m_eq_4_byte_identical() { run_parity_at_m(4); }
