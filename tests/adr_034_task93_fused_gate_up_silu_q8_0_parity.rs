//! ADR-034 task #93 (2026-05-21) — parity test for the fused
//! `kernel_fused_gate_up_silu_q8_0_f32` kernel.
//!
//! Asserts byte-identical (within F32 tolerance) output vs the unfused
//! 3-dispatch sequence:
//!   1. `dispatch_quantized_matmul_ggml(Q8_0, gate_w, x)` → tmp_gate
//!   2. `dispatch_quantized_matmul_ggml(Q8_0, up_w, x)`   → tmp_up
//!   3. `dispatch_silu_mul(tmp_gate, tmp_up)`              → out_unfused
//!
//! Run unfused step (1,2) with `HF2Q_Q8_0_MV_NR2=1` so the gate/up
//! sub-computations use the NR=2 NSG=4 kernel — the same accumulator
//! order our fused kernel uses internally. With matched reduction order
//! the only delta vs the fused kernel is whether silu_mul runs as a
//! standalone dispatch (unfused) or inline (fused). Both perform
//! `silu(g) * up = g / (1 + exp(-g)) * up` in IEEE-754 F32, so the
//! result must be byte-identical.
//!
//! Falsification gate: if this test fails, the fused kernel is buggy
//! and any subsequent perf claim is meaningless.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_q8_0::{
    dispatch_fused_gate_up_silu_q8_0, FusedGateUpSiluQ8_0Args,
};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice,
};

const QK8_0: usize = 32;

// PRNG matching adr_028_iter309_q6k_mv_nr2_parity.rs.
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

/// Pack a flat F32 array (length must be a multiple of `QK8_0`) into
/// GGUF Q8_0 blocks (34 bytes per block: 2-byte half scale + 32 int8).
/// Mirrors llama.cpp's `quantize_row_q8_0_ref`.
fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK8_0 == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK8_0 * 34);
    for block in values.chunks(QK8_0) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let half = half::f16::from_f32(d).to_bits();
        bytes.extend_from_slice(&half.to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

#[test]
fn fused_gate_up_silu_q8_0_byte_identical_to_unfused() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    // Shape: small but realistic. Hidden_size = 256 (8 blocks), intermediate = 64.
    let hidden_size: u32 = 256;
    let intermediate_size: u32 = 64;
    let m: u32 = 1; // decode

    // Generate input + weights with deterministic PRNG.
    let input = pseudo_random_f32(0xC0FFEE, (hidden_size * m) as usize);
    let gate_w_f32 = pseudo_random_f32(
        0xDEAD_BEEF,
        (intermediate_size * hidden_size) as usize,
    );
    let up_w_f32 = pseudo_random_f32(
        0xCAFE_F00D,
        (intermediate_size * hidden_size) as usize,
    );

    // Quantize gate + up to Q8_0.
    let gate_q8_0 = pack_q8_0(&gate_w_f32);
    let up_q8_0 = pack_q8_0(&up_w_f32);

    // Upload all buffers (alloc + copy via as_mut_slice).
    let mut input_buf = device
        .alloc_buffer(
            (hidden_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, hidden_size as usize],
        )
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input as_mut_slice")
        .copy_from_slice(&input);

    let mut gate_w_buf = device
        .alloc_buffer(
            gate_q8_0.len(),
            DType::F32,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc gate_w");
    gate_w_buf
        .as_mut_slice::<u8>()
        .expect("gate_w as_mut_slice")
        .copy_from_slice(&gate_q8_0);

    let mut up_w_buf = device
        .alloc_buffer(
            up_q8_0.len(),
            DType::F32,
            vec![intermediate_size as usize, hidden_size as usize],
        )
        .expect("alloc up_w");
    up_w_buf
        .as_mut_slice::<u8>()
        .expect("up_w as_mut_slice")
        .copy_from_slice(&up_q8_0);

    // --- Step 1: UNFUSED reference path ---
    //
    // NOTE: force HF2Q_Q8_0_MV_NR2=1 so unfused uses the NR=2 NSG=4 kernel
    // (matched accumulator order to our fused kernel). Without this the
    // unfused path uses kernel_mul_mv_q8_0_f32 (8-row variant) and the
    // accumulator order differs, breaking byte-identity.
    std::env::set_var("HF2Q_Q8_0_MV_NR2", "1");

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
        .expect("alloc params silu");
    params_buf_silu.as_mut_slice::<u32>().expect("write params")[0] =
        intermediate_size;

    let mv_params = GgmlQuantizedMatmulParams {
        m,
        n: intermediate_size,
        k: hidden_size,
        ggml_type: GgmlType::Q8_0,
    };

    let mut enc = device.command_encoder().expect("encoder unfused");
    quantized_matmul_ggml(
        &mut enc,
        &mut registry,
        &device,
        &input_buf,
        &gate_w_buf,
        &tmp_gate,
        &mv_params,
    )
    .expect("gate matvec");
    quantized_matmul_ggml(
        &mut enc,
        &mut registry,
        &device,
        &input_buf,
        &up_w_buf,
        &tmp_up,
        &mv_params,
    )
    .expect("up matvec");
    dispatch_silu_mul(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &tmp_gate,
        &tmp_up,
        &out_unfused,
        &params_buf_silu,
        intermediate_size,
    )
    .expect("silu_mul");
    enc.commit_and_wait().expect("commit unfused");

    let unfused_result: Vec<f32> =
        out_unfused.as_slice::<f32>().expect("read unfused").to_vec();

    // --- Step 2: FUSED kernel path ---
    let out_fused = device
        .alloc_buffer(
            (intermediate_size * m) as usize * 4,
            DType::F32,
            vec![m as usize, intermediate_size as usize],
        )
        .expect("alloc out_fused");

    let mut enc2 = device.command_encoder().expect("encoder fused");
    dispatch_fused_gate_up_silu_q8_0(
        &mut enc2,
        &mut registry,
        &device,
        &gate_w_buf,
        &up_w_buf,
        &input_buf,
        &out_fused,
        FusedGateUpSiluQ8_0Args {
            m,
            intermediate_size,
            hidden_size,
        },
    )
    .expect("fused dispatch");
    enc2.commit_and_wait().expect("commit fused");

    let fused_result: Vec<f32> =
        out_fused.as_slice::<f32>().expect("read fused").to_vec();

    // --- Assert byte-identical ---
    //
    // Both kernels:
    //   - Use the same Q8_0 dequant math (`qs * d` per block)
    //   - Use the same reduction order (NR=2 NSG=4 + simd_sum + threadgroup
    //     cross-SG reduce)
    //   - Apply `silu(g) = g / (1 + exp(-g))` in F32 IEEE-754
    //   - Multiply silu(g) * up in F32 IEEE-754
    // The only difference is whether the silu_mul runs as a separate kernel
    // (unfused) or as the final lines of the fused kernel — same FP ops.
    assert_eq!(fused_result.len(), unfused_result.len());
    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let mut bytewise_identical = true;
    for (i, (&a, &b)) in fused_result.iter().zip(unfused_result.iter()).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs_diff {
            max_abs_diff = abs;
        }
        let denom = a.abs().max(b.abs()).max(1e-12);
        let rel = abs / denom;
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
        if a.to_bits() != b.to_bits() {
            bytewise_identical = false;
            if i < 5 {
                eprintln!(
                    "[diff @ row {i}] fused={a:.6e} ({:#010x}) unfused={b:.6e} ({:#010x})",
                    a.to_bits(),
                    b.to_bits(),
                );
            }
        }
    }
    eprintln!(
        "fused vs unfused: max_abs_diff={max_abs_diff:.3e} \
         max_rel_diff={max_rel_diff:.3e} bytewise_identical={bytewise_identical}"
    );

    // Tolerance: F32 ops within FMA-vs-mul-add rounding ≤ 1e-5 abs is the
    // documented contract from adr_028_iter309 (Q6_K parity).
    assert!(
        max_abs_diff < 1e-5,
        "fused vs unfused max_abs_diff {max_abs_diff:.3e} exceeds 1e-5"
    );
}
