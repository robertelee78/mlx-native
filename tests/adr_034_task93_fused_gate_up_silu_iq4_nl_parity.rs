//! ADR-034 task #93 cont. 26 (2026-05-21) — parity test for the fused
//! `kernel_fused_gate_up_silu_iq4_nl_f32` kernel.
//!
//! Asserts byte-identical (within F32 tolerance) output vs the unfused
//! 3-dispatch sequence:
//!   1. `quantized_matmul_ggml(IQ4_NL, gate_w, x)` → tmp_gate
//!   2. `quantized_matmul_ggml(IQ4_NL, up_w, x)`   → tmp_up
//!   3. `dispatch_silu_mul(tmp_gate, tmp_up)`       → out_unfused

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_iq4_nl::{
    dispatch_fused_gate_up_silu_iq4_nl, FusedGateUpSiluIq4NlArgs,
};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK4_0: usize = 32;

// Frozen IQ4_NL codebook (matches ggml-common.h:1109-1112 + the kernel).
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
];

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

/// Pack F32 values into IQ4_NL blocks (18 bytes each: half scale + 16 packed
/// codebook indices). Mirrors the canonical quantize_row_iq4_nl_ref logic:
/// per block, find d such that max|v|/d ∈ codebook range, then nearest-index
/// lookup for each value.
fn pack_iq4_nl(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK4_0 == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK4_0 * 18);
    for block in values.chunks(QK4_0) {
        // d = max|v| / 113 (max-abs of codebook = max(113, -(-127)) = 127,
        //    but the reference pivot uses max signed extent properly).
        // Simpler: choose d so that v/d lies in codebook range,
        // then nearest-neighbor lookup.
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let id = 1.0 / d;
        let half = half::f16::from_f32(d).to_bits();
        bytes.extend_from_slice(&half.to_le_bytes());
        // Pack 32 values as 16 bytes (low nibble = idx[i], high = idx[i+16])
        // following the same layout the IQ4_NL kernel reads:
        //   qs[i].low_nibble  → values[i +  0]
        //   qs[i].high_nibble → values[i + 16]
        let mut qs = [0u8; 16];
        for i in 0..16 {
            // Nearest codebook index for values[i] and values[i+16].
            let v0_scaled = block[i] * id;
            let v1_scaled = block[i + 16] * id;
            let nearest = |target: f32| -> u8 {
                let mut best = 0u8;
                let mut best_dist = f32::MAX;
                for (idx, &cv) in KVALUES_IQ4_NL.iter().enumerate() {
                    let dist = (cv as f32 - target).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best = idx as u8;
                    }
                }
                best
            };
            let q0 = nearest(v0_scaled);
            let q1 = nearest(v1_scaled);
            qs[i] = q0 | (q1 << 4);
        }
        bytes.extend_from_slice(&qs);
    }
    bytes
}

fn run_parity_at_m(m: u32) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let hidden_size: u32 = 256;     // 8 blocks
    let intermediate_size: u32 = 64; // 8 rows/TG × 8 TGs

    let input = pseudo_random_f32(0xC0FF_EE17, (hidden_size * m) as usize);
    let gate_w_f32 = pseudo_random_f32(0xDEAD_BEEF, (intermediate_size * hidden_size) as usize);
    let up_w_f32   = pseudo_random_f32(0xCAFE_F00D, (intermediate_size * hidden_size) as usize);

    let gate_iq4_nl = pack_iq4_nl(&gate_w_f32);
    let up_iq4_nl   = pack_iq4_nl(&up_w_f32);

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
            gate_iq4_nl.len(),
            DType::U8,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc gate_w");
    gate_w_buf.as_mut_slice::<u8>().expect("gw slice").copy_from_slice(&gate_iq4_nl);

    let mut up_w_buf = device
        .alloc_buffer(
            up_iq4_nl.len(),
            DType::U8,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc up_w");
    up_w_buf.as_mut_slice::<u8>().expect("uw slice").copy_from_slice(&up_iq4_nl);

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
        ggml_type: GgmlType::IQ4_NL,
    };

    let mut enc = device.command_encoder().expect("encoder unfused");
    quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &gate_w_buf, &tmp_gate, &mv_params).expect("gate");
    quantized_matmul_ggml(&mut enc, &mut registry, &device, &input_buf, &up_w_buf,   &tmp_up,   &mv_params).expect("up");
    enc.memory_barrier();
    dispatch_silu_mul(&mut enc, &mut registry, device.metal_device(), &tmp_gate, &tmp_up, &out_unfused, &params_buf_silu, silu_n).expect("silu_mul");
    enc.commit_and_wait().expect("commit unfused");

    let unfused: Vec<f32> = out_unfused.as_slice::<f32>().expect("read unfused").to_vec();

    // ---- Fused dispatch ----
    let out_fused = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc out_fused");

    let mut enc2 = device.command_encoder().expect("encoder fused");
    dispatch_fused_gate_up_silu_iq4_nl(
        &mut enc2, &mut registry, &device,
        &gate_w_buf, &up_w_buf, &input_buf, &out_fused,
        FusedGateUpSiluIq4NlArgs { m, intermediate_size, hidden_size },
    ).expect("fused");
    enc2.commit_and_wait().expect("commit fused");

    let fused: Vec<f32> = out_fused.as_slice::<f32>().expect("read fused").to_vec();

    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in fused.iter().zip(unfused.iter()).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs { max_abs = abs; }
        if a.to_bits() != b.to_bits() && i < 5 {
            eprintln!(
                "[diff @ row {i}] fused={a:.6e} ({:#010x}) unfused={b:.6e} ({:#010x})",
                a.to_bits(), b.to_bits()
            );
        }
    }
    eprintln!("IQ4_NL m={m}: max_abs_diff={max_abs:.3e}");
    assert!(max_abs < 1e-5, "fused IQ4_NL vs unfused max_abs_diff {max_abs:.3e} > 1e-5 (m={m})");
}

#[test]
fn fused_iq4_nl_m_eq_1_byte_identical() { run_parity_at_m(1); }

#[test]
fn fused_iq4_nl_m_eq_2_byte_identical() { run_parity_at_m(2); }

#[test]
fn fused_iq4_nl_m_eq_4_byte_identical() { run_parity_at_m(4); }
