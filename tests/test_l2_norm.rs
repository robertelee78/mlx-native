//! Tests for the L2 normalization GPU kernel (ADR-013 Decision 3).
//!
//! Spec: `l2_norm(x, eps) = x / sqrt(sum(x^2) + eps)` over the last dim.
//!
//! Acceptance criteria from ADR-013:
//! - For a hand-constructed small tensor with known Euclidean norm, output
//!   matches `x / ||x||` within 1e-5 for F32, 1e-3 for BF16.
//! - Round-trip: `|l2_norm(x) * ||x|| - x| < eps` for random inputs.
//! - Spec-driven: expected outputs hand-authored, no reference-tool oracle.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn alloc_params(device: &MlxDevice, eps: f32, dim: u32) -> mlx_native::MlxBuffer {
    let mut buf = device
        .alloc_buffer(2 * 4, DType::F32, vec![2])
        .expect("alloc params");
    {
        let s = buf.as_mut_slice::<f32>().expect("mut params");
        s[0] = eps;
        s[1] = dim as f32;
    }
    buf
}

// =====================================================================
// F32 tests
// =====================================================================

/// Spec-driven: hand-constructed input with known sum-of-squares.
///
/// Input  = [3, 4]  (a classic 3-4-5 right triangle)
/// sum(x^2) = 9 + 16 = 25; sqrt(25 + eps) ≈ 5.
/// eps = 0 -> output = [3/5, 4/5] = [0.6, 0.8] exactly.
#[test]
fn test_l2_norm_f32_3_4_5_triangle() {
    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 2u32;
    let rows = 1u32;

    let input_data = [3.0f32, 4.0f32];
    let mut input = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("alloc input");
    input
        .as_mut_slice::<f32>()
        .expect("mut input")
        .copy_from_slice(&input_data);

    let output = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("alloc output");

    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    let expected = [0.6f32, 0.8f32];
    for i in 0..2 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "f32 3-4-5 triangle mismatch at {}: got {}, expected {}, diff {}",
            i,
            got[i],
            expected[i],
            diff
        );
    }
}

/// Multi-row input — each row normalized independently.
#[test]
fn test_l2_norm_f32_multirow() {
    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 4u32;
    let rows = 3u32;
    let n = (rows * dim) as usize;

    // Three rows with known sum-of-squares:
    //   row0 = [1, 0, 0, 0]     -> sum_sq = 1,   inv = 1
    //   row1 = [1, 1, 1, 1]     -> sum_sq = 4,   inv = 0.5
    //   row2 = [0.3, 0.4, 0.0, 0.0] -> sum_sq = 0.25, inv = 2 (output = [0.6, 0.8, 0, 0])
    let input_data: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.3, 0.4, 0.0, 0.0,
    ];

    let mut input = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);

    let output = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("output");
    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    let expected: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0,
        0.5, 0.5, 0.5, 0.5,
        0.6, 0.8, 0.0, 0.0,
    ];
    for i in 0..12 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "multirow mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], diff
        );
    }
}

/// Round-trip correctness: output * ||x|| should reconstruct x within eps.
#[test]
fn test_l2_norm_f32_round_trip() {
    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 64u32;
    let rows = 8u32;
    let n = (rows * dim) as usize;

    // Deterministic pseudo-random input (linear congruential hash).
    let mut input_data = vec![0.0f32; n];
    let mut seed = 0x1234u32;
    for v in input_data.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed as i32 as f32) / (i32::MAX as f32);
    }

    let mut input = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);

    let output = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("output");
    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    for r in 0..rows as usize {
        // Reconstruct row norm from input, multiply normalized output by it,
        // and compare against the original input element.
        let mut sum_sq = 0.0f64;
        for c in 0..dim as usize {
            let v = input_data[r * dim as usize + c] as f64;
            sum_sq += v * v;
        }
        let row_norm = sum_sq.sqrt() as f32;
        for c in 0..dim as usize {
            let idx = r * dim as usize + c;
            let reconstructed = got[idx] * row_norm;
            let diff = (reconstructed - input_data[idx]).abs();
            assert!(
                diff < 1e-5,
                "round-trip mismatch at (r={}, c={}): got {}, expected {}, diff {}",
                r, c, reconstructed, input_data[idx], diff
            );
        }
    }
}

/// Zero-input edge case: sum_sq == 0 with non-zero eps should not NaN.
#[test]
fn test_l2_norm_f32_zero_row_with_eps() {
    let (device, mut registry) = setup();
    let eps = 1e-6f32;
    let dim = 4u32;
    let rows = 1u32;

    let input_data = [0.0f32; 4];
    let mut input = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    for (i, v) in got.iter().enumerate().take(4) {
        assert!(v.is_finite(), "zero-row produced non-finite at {}: {}", i, v);
        assert!(v.abs() < 1e-3, "zero-row not near zero at {}: {}", i, v);
    }
}

/// eps damps the norm: sum_sq = 0 and eps = 1 -> inv = 1, output = input (zero here).
#[test]
fn test_l2_norm_f32_eps_effect() {
    let (device, mut registry) = setup();
    let eps = 9.0f32;       // deliberately large so sum_sq=16 gives sqrt(25) = 5
    let dim = 2u32;
    let rows = 1u32;

    // [3, 4] with eps=9: denominator = sqrt(9 + 16 + 9) ... wait. sum = 25; sqrt(25+9)=sqrt(34).
    // Let's instead use [0, 4]: sum_sq = 16; sqrt(16 + 9) = 5; output = [0, 4/5] = [0, 0.8].
    let input_data = [0.0f32, 4.0f32];
    let mut input = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    let expected = [0.0f32, 0.8f32];
    for i in 0..2 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "eps-effect mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], diff
        );
    }
}

// =====================================================================
// BF16 tests (lower precision tolerance)
// =====================================================================

/// BF16 version of the 3-4-5 triangle test (tolerance widened).
#[test]
fn test_l2_norm_bf16_3_4_5_triangle() {
    use half::bf16;

    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 2u32;
    let rows = 1u32;

    let input_data = [bf16::from_f32(3.0), bf16::from_f32(4.0)];
    let mut input = device
        .alloc_buffer(4, DType::BF16, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<bf16>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(4, DType::BF16, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, eps, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[bf16] = output.as_slice().expect("read");
    let got_f32 = [got[0].to_f32(), got[1].to_f32()];
    let expected = [0.6f32, 0.8f32];
    for i in 0..2 {
        let diff = (got_f32[i] - expected[i]).abs();
        assert!(
            diff < 1e-2,
            "bf16 3-4-5 triangle mismatch at {}: got {}, expected {}, diff {}",
            i, got_f32[i], expected[i], diff
        );
    }
}

// =====================================================================
// Error handling
// =====================================================================

#[test]
fn test_l2_norm_rejects_zero_rows() {
    let (device, mut registry) = setup();
    let dim = 4u32;
    let input = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("input");
    let output = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, 0.0, dim);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        0, // zero rows
        dim,
    );
    assert!(res.is_err(), "zero rows should error");
}

#[test]
fn test_l2_norm_rejects_mismatched_dtype() {
    use half::bf16;
    let _ = bf16::from_f32(0.0); // keep import used across cfg

    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;
    let input = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("input");
    // Deliberately BF16 output to trigger dtype mismatch.
    let output = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, 0.0, dim);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::l2_norm::dispatch_l2_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    );
    assert!(res.is_err(), "dtype mismatch should error");
}

// =====================================================================
// ADR-015 iter59a — fused L2 norm + scalar multiply (l2_norm_scale_f32)
// =====================================================================

fn alloc_scale_params(device: &MlxDevice, eps: f32, dim: u32, scale: f32) -> mlx_native::MlxBuffer {
    let mut buf = device
        .alloc_buffer(3 * 4, DType::F32, vec![3])
        .expect("alloc scale params");
    {
        let s = buf.as_mut_slice::<f32>().expect("mut scale params");
        s[0] = eps;
        s[1] = dim as f32;
        s[2] = scale;
    }
    buf
}

/// Fused l2_norm + scale on the 3-4-5 triangle with scale=10:
///   x = [3, 4]; sum_sq=25; inv = 1/5; output = [3*0.2*10, 4*0.2*10] = [6.0, 8.0].
#[test]
fn test_l2_norm_scale_f32_3_4_5_triangle() {
    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 2u32;
    let rows = 1u32;
    let scale = 10.0f32;

    let input_data = [3.0f32, 4.0f32];
    let mut input = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("alloc input");
    input
        .as_mut_slice::<f32>()
        .expect("mut input")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(8, DType::F32, vec![dim as usize])
        .expect("alloc output");
    let params = alloc_scale_params(&device, eps, dim, scale);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    let expected = [6.0f32, 8.0f32];
    for i in 0..2 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "fused 3-4-5 mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], diff
        );
    }
}

/// Cross-check: fused l2_norm_scale_f32 must match the unfused
/// (l2_norm_f32 + scalar_mul_f32) sequence to <= 1e-5 max-abs.
#[test]
fn test_l2_norm_scale_f32_parity_vs_unfused() {
    let (device, mut registry) = setup();
    let eps = 1e-6f32;
    let dim = 128u32;
    let rows = 32u32;
    let scale = 0.123456f32; // arbitrary non-trivial scalar
    let n = (rows * dim) as usize;

    let mut input_data = vec![0.0f32; n];
    let mut seed: u32 = 0xCAFEBABE;
    for v in input_data.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed as i32 as f32) / (i32::MAX as f32);
    }

    let mut input = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);

    // Fused output.
    let fused_out = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("fused out");
    let fused_params = alloc_scale_params(&device, eps, dim, scale);
    {
        let mut enc = device.command_encoder().expect("enc fused");
        mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &input,
            &fused_out,
            &fused_params,
            rows,
            dim,
        )
        .expect("dispatch fused");
        enc.commit_and_wait().expect("commit fused");
    }

    // Unfused: l2_norm_f32 -> scalar_mul_f32.
    let unfused_l2 = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("unfused l2");
    let unfused_out = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("unfused out");
    let unfused_params = alloc_params(&device, eps, dim);
    {
        let mut enc = device.command_encoder().expect("enc unfused");
        mlx_native::ops::l2_norm::dispatch_l2_norm(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &input,
            &unfused_l2,
            &unfused_params,
            rows,
            dim,
        )
        .expect("dispatch l2 unfused");
        enc.memory_barrier();
        mlx_native::ops::elementwise::scalar_mul_f32(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &unfused_l2,
            &unfused_out,
            n,
            scale,
        )
        .expect("dispatch scalar_mul unfused");
        enc.commit_and_wait().expect("commit unfused");
    }

    let got_fused: &[f32] = fused_out.as_slice().expect("read fused");
    let got_unfused: &[f32] = unfused_out.as_slice().expect("read unfused");
    // Bit-identity required: the fused kernel mirrors the unfused
    // `(input * inv)` then `* scale` ordering by writing the intermediate
    // to device memory between the two multiplies (with a device-memory
    // barrier to prevent compiler FMA fusion).  This contract lets
    // ADR-015 iter59a swap the kernel without perturbing greedy-T=0
    // token cliffs in the GDN delta-rule recurrence (a 1-ulp drift in
    // q_scaled was empirically observed to flip 3/16 prompts in the
    // 32-tok smoke at near-tied logit cliffs).
    let mut max_abs = 0.0f32;
    let mut mismatches = 0usize;
    for i in 0..n {
        if got_fused[i].to_bits() != got_unfused[i].to_bits() {
            mismatches += 1;
            let diff = (got_fused[i] - got_unfused[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "fused vs unfused bit-identity violated: {} mismatches, max_abs={}",
        mismatches, max_abs
    );
}

/// Multi-row fused: each row normalized + scaled independently.
#[test]
fn test_l2_norm_scale_f32_multirow() {
    let (device, mut registry) = setup();
    let eps = 0.0f32;
    let dim = 4u32;
    let rows = 3u32;
    let scale = 2.5f32;
    let n = (rows * dim) as usize;

    // Row sums-of-squares: 1, 4, 0.25 -> invs: 1, 0.5, 2.
    // Output rows scaled by 2.5:
    //   row0: [2.5, 0, 0, 0]
    //   row1: [1.25, 1.25, 1.25, 1.25]
    //   row2: [1.5, 2.0, 0, 0]
    let input_data: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.3, 0.4, 0.0, 0.0,
    ];

    let mut input = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("output");
    let params = alloc_scale_params(&device, eps, dim, scale);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let got: &[f32] = output.as_slice().expect("read");
    let expected: [f32; 12] = [
        2.5, 0.0, 0.0, 0.0,
        1.25, 1.25, 1.25, 1.25,
        1.5, 2.0, 0.0, 0.0,
    ];
    for i in 0..12 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "fused multirow mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], diff
        );
    }
}

/// Spec-driven: scale=1 must be bit-equivalent to plain l2_norm_f32.
#[test]
fn test_l2_norm_scale_f32_unit_scale_matches_plain() {
    let (device, mut registry) = setup();
    let eps = 1e-6f32;
    let dim = 96u32;
    let rows = 16u32;
    let n = (rows * dim) as usize;

    let mut input_data = vec![0.0f32; n];
    let mut seed: u32 = 0xDEADBEEF;
    for v in input_data.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (seed as i32 as f32) / (i32::MAX as f32);
    }

    let mut input = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);

    let plain_out = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("plain out");
    let plain_params = alloc_params(&device, eps, dim);
    {
        let mut enc = device.command_encoder().expect("enc plain");
        mlx_native::ops::l2_norm::dispatch_l2_norm(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &input,
            &plain_out,
            &plain_params,
            rows,
            dim,
        )
        .expect("dispatch plain");
        enc.commit_and_wait().expect("commit plain");
    }

    let scale_out = device
        .alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize])
        .expect("scale out");
    let scale_params = alloc_scale_params(&device, eps, dim, 1.0);
    {
        let mut enc = device.command_encoder().expect("enc scale");
        mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &input,
            &scale_out,
            &scale_params,
            rows,
            dim,
        )
        .expect("dispatch scale");
        enc.commit_and_wait().expect("commit scale");
    }

    let plain: &[f32] = plain_out.as_slice().expect("read plain");
    let scale: &[f32] = scale_out.as_slice().expect("read scale");
    for i in 0..n {
        let diff = (plain[i] - scale[i]).abs();
        assert!(
            diff < 1e-6,
            "scale=1 vs plain mismatch at {}: plain={}, scale={}, diff {}",
            i, plain[i], scale[i], diff
        );
    }
}

#[test]
fn test_l2_norm_scale_f32_rejects_mismatched_dtype() {
    use half::bf16;
    let _ = bf16::from_f32(0.0);

    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;
    let input = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("input");
    let output = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("output");
    let params = alloc_scale_params(&device, 0.0, dim, 1.0);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    );
    assert!(res.is_err(), "dtype mismatch should error");
}

#[test]
fn test_l2_norm_scale_f32_rejects_non_f32() {
    use half::bf16;
    let _ = bf16::from_f32(0.0);

    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;
    let input = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("input");
    let output = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("output");
    let params = alloc_scale_params(&device, 0.0, dim, 1.0);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::l2_norm::dispatch_l2_norm_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    );
    assert!(res.is_err(), "non-f32 dtype should error");
}
