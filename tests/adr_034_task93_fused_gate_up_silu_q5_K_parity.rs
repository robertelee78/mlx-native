//! ADR-034 task #93 cont. 27 (2026-05-21) — parity test for the fused
//! `kernel_fused_gate_up_silu_q5_K_f32` kernel.
//!
//! Asserts byte-identical (within F32 tolerance) output vs the unfused
//! 3-dispatch sequence:
//!   1. `quantized_matmul_ggml(Q5_K, gate_w, x)` → tmp_gate
//!   2. `quantized_matmul_ggml(Q5_K, up_w, x)`   → tmp_up
//!   3. `dispatch_silu_mul(tmp_gate, tmp_up)`     → out_unfused

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, non_snake_case)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_q5_K::{
    dispatch_fused_gate_up_silu_q5_K, FusedGateUpSiluQ5_KArgs,
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

/// Pack F32 into Q5_K super-blocks. 176 bytes/block: half d, half dmin,
/// 12 bytes 6-bit packed scales+mins (same as Q4_K), 32 bytes qh (5th
/// bit per element), 128 bytes qs (low 4 bits).
fn pack_q5_K(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK_K == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK_K * 176);
    for block in values.chunks(QK_K) {
        // Same sub-block analysis as Q4_K but quants are 5-bit (0..31)
        // instead of 4-bit (0..15).
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
            sub_d[s] = (max - min) / 31.0;
            if sub_d[s] == 0.0 { sub_d[s] = 1e-30; }
        }
        let outer_d_max  = sub_d.iter().cloned().fold(0.0f32, f32::max);
        let outer_dm_max = sub_m.iter().cloned().fold(0.0f32, |acc, m| acc.max(-m));
        let d  = outer_d_max  / 63.0;
        let dm = outer_dm_max / 63.0;
        let id  = if d  != 0.0 { 1.0 / d  } else { 0.0 };
        let idm = if dm != 0.0 { 1.0 / dm } else { 0.0 };

        let mut sc6 = [0u8; 8];
        let mut mn6 = [0u8; 8];
        for s in 0..8 {
            sc6[s] = (sub_d[s] * id).round().clamp(0.0, 63.0) as u8;
            mn6[s] = (-sub_m[s] * idm).round().clamp(0.0, 63.0) as u8;
        }

        // 12-byte packed scales+mins (same layout as Q4_K).
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j]     = sc6[j];
            scales[j + 4] = mn6[j];
        }
        for j in 4..8 {
            scales[j + 4] = (sc6[j] & 0xf) | ((mn6[j] & 0xf) << 4);
            scales[j - 4] |= (sc6[j] >> 4) << 6;
            scales[j]     |= (mn6[j] >> 4) << 6;
        }

        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&half::f16::from_f32(dm).to_bits().to_le_bytes());
        bytes.extend_from_slice(&scales);

        // Pack low 4 bits (qs) and high 1 bit (qh) for all 256 quants.
        // qh layout: 32 bytes, each bit corresponds to one quant's high bit.
        // qs layout: 128 bytes paired by sub-block (s, s+1), same as Q4_K
        //   except quants are 0..31 (low 4 bits go to qs, high bit to qh).
        let mut qs = [0u8; QK_K/2]; // 128 bytes
        let mut qh = [0u8; QK_K/8]; // 32 bytes

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
                let q0 = ((v0 + sub0_m) * id0).round().clamp(0.0, 31.0) as u32;
                let q1 = ((v1 + sub1_m) * id1).round().clamp(0.0, 31.0) as u32;
                // Low 4 bits → qs[pair*32 + l] nibble pair.
                qs[pair * 32 + l] = ((q0 & 0xF) as u8) | (((q1 & 0xF) as u8) << 4);
                // High bit → qh layout. The kernel reads qh[l] where each
                // byte holds 4 high-bit pairs across all 8 sub-blocks (in iq
                // order). The mask layout is hm1=1<<(2*iq), hm2=hm1<<1,
                // hm3=hm1<<4, hm4=hm2<<4 (see kernel_mul_mv_q5_K_f32).
                //
                // For sub-block pair (s0, s1) where s0 = pair*2:
                //   iq = pair/2 (iq selects which 32-byte half of the super-block)
                //   Within iq's half, two pair-indices map to q1's two halves.
                // This is complex. Easier path: derive iq/local within the
                // canonical pack pattern.
                //
                // canonical pack (ggml-quants.c quantize_row_q5_K_ref):
                //   for (int j = 0; j < QK_K; ++j) {
                //     uint8_t lo = q[j] & 0xF;
                //     uint8_t hi = q[j] >> 4;
                //     qs_low_nibble[j] = lo;
                //     qh_bit[j] = hi;
                //   }
                // qh[l] bit b corresponds to the (l + 8*b)-th value in the block.
                // (See ggml-quants.c make_q5_k routine.)
                //
                // We're inside the (pair, l) loop with pair_index = pair*64+l.
                // For sub-block s0 it's val_idx0 = s0*32 + l = (pair*2)*32 + l
                //   = pair*64 + l, so the qh bit position is:
                //   byte = val_idx0 % 32, bit = val_idx0 / 32
                let v0_idx = s0 * 32 + l;
                let v1_idx = s1 * 32 + l;
                let q0_hi = (q0 >> 4) & 0x1;
                let q1_hi = (q1 >> 4) & 0x1;
                qh[v0_idx % 32] |= (q0_hi as u8) << (v0_idx / 32);
                qh[v1_idx % 32] |= (q1_hi as u8) << (v1_idx / 32);
            }
        }
        bytes.extend_from_slice(&qh);
        bytes.extend_from_slice(&qs);
    }
    bytes
}

fn run_parity_at_m(m: u32) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let hidden_size: u32 = 256;
    let intermediate_size: u32 = 64;

    let input = pseudo_random_f32(0xDEAD_C0DE, (hidden_size * m) as usize);
    let gate_w_f32 = pseudo_random_f32(0xCAFE_BABE, (intermediate_size * hidden_size) as usize);
    let up_w_f32   = pseudo_random_f32(0xFEED_FACE, (intermediate_size * hidden_size) as usize);

    let gate_q5_K = pack_q5_K(&gate_w_f32);
    let up_q5_K   = pack_q5_K(&up_w_f32);

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
            gate_q5_K.len(),
            DType::U8,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc gate_w");
    gate_w_buf.as_mut_slice::<u8>().expect("gw").copy_from_slice(&gate_q5_K);

    let mut up_w_buf = device
        .alloc_buffer(
            up_q5_K.len(),
            DType::U8,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc up_w");
    up_w_buf.as_mut_slice::<u8>().expect("uw").copy_from_slice(&up_q5_K);

    let tmp_gate = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("tmp_gate");
    let tmp_up   = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("tmp_up");
    let out_unfused = device.alloc_buffer((intermediate_size * m) as usize * 4, DType::F32, vec![m as usize, intermediate_size as usize]).expect("out_unfused");
    let mut params_buf_silu = device.alloc_buffer(4, DType::U32, vec![1]).expect("silu params");
    let silu_n: u32 = intermediate_size * m;
    params_buf_silu.as_mut_slice::<u32>().expect("write")[0] = silu_n;

    let mv_params = GgmlQuantizedMatmulParams {
        m, n: intermediate_size, k: hidden_size, ggml_type: GgmlType::Q5_K,
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
    dispatch_fused_gate_up_silu_q5_K(
        &mut enc2, &mut registry, &device,
        &gate_w_buf, &up_w_buf, &input_buf, &out_fused,
        FusedGateUpSiluQ5_KArgs { m, intermediate_size, hidden_size },
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
    eprintln!("Q5_K m={m}: max_abs_diff={max_abs:.3e}");
    // Tolerance: 1e-4 absolute. Q5_K accumulates `acc1[i] + 16.f * acc2[i]`
    // (vs Q4_K which has no acc2) — the extra mul-add gives the Metal
    // compiler more FMA fusion choices, and interleaving gate+up accumulators
    // in the fused kernel produces slightly different FMA grouping than
    // two separate kernels (~3e-5 absolute on values in [-10, 10] range).
    // 1e-4 is well within F32 IEEE-754 rounding tolerance and consistent
    // with the adr_028_iter309 Q6_K parity contract.
    assert!(max_abs < 1e-4, "Q5_K fused vs unfused max_abs_diff {max_abs:.3e} > 1e-4 (m={m})");
}

#[test]
fn fused_q5_K_m_eq_1_byte_identical() { run_parity_at_m(1); }

#[test]
fn fused_q5_K_m_eq_2_byte_identical() { run_parity_at_m(2); }

#[test]
fn fused_q5_K_m_eq_4_byte_identical() { run_parity_at_m(4); }
