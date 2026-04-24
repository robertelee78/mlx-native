//! Tests for the inclusive prefix-sum GPU kernel (ADR-013 Decision 4).
//!
//! Spec: `out[r, i] = sum(x[r, 0..=i])` applied per-row.
//!
//! Acceptance criteria from ADR-013:
//! - `cumsum([1,2,3,4]) = [1,3,6,10]` — hand-derived spec.
//! - Multi-batched input correctness.
//! - Spec-driven expected values in test comments; no reference oracle.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn alloc_params(device: &MlxDevice, dim: u32) -> mlx_native::MlxBuffer {
    let mut buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc params");
    {
        let s = buf.as_mut_slice::<u32>().expect("mut params");
        s[0] = dim;
    }
    buf
}

// =====================================================================
// F32
// =====================================================================

/// Spec-driven: textbook example. cumsum([1,2,3,4]) = [1, 3, 6, 10].
#[test]
fn test_cumsum_f32_textbook() {
    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;

    let input_data = [1.0f32, 2.0, 3.0, 4.0];
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
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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
    let expected = [1.0f32, 3.0, 6.0, 10.0];
    for i in 0..4 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "textbook mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], diff
        );
    }
}

/// Negative values and zeros present.
///
/// Input:  [1, -1, 2, 0, 3]
/// Cumsum: [1,  0, 2, 2, 5]
#[test]
fn test_cumsum_f32_negatives_and_zeros() {
    let (device, mut registry) = setup();
    let dim = 5u32;
    let rows = 1u32;

    let input_data = [1.0f32, -1.0, 2.0, 0.0, 3.0];
    let mut input = device
        .alloc_buffer(20, DType::F32, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(20, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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
    let expected = [1.0f32, 0.0, 2.0, 2.0, 5.0];
    for i in 0..5 {
        let diff = (got[i] - expected[i]).abs();
        assert!(diff < 1e-6, "neg mismatch at {}: got {}", i, got[i]);
    }
}

/// Multi-row independent scans.
#[test]
fn test_cumsum_f32_multirow() {
    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 3u32;
    let n = (rows * dim) as usize;

    // row 0: [1, 2, 3, 4]       -> [1, 3, 6, 10]
    // row 1: [0, 0, 0, 0]       -> [0, 0, 0, 0]
    // row 2: [-1, 2, -3, 4]     -> [-1, 1, -2, 2]
    let input_data: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0,
        0.0, 0.0, 0.0, 0.0,
        -1.0, 2.0, -3.0, 4.0,
    ];
    let expected: [f32; 12] = [
        1.0, 3.0, 6.0, 10.0,
        0.0, 0.0, 0.0, 0.0,
        -1.0, 1.0, -2.0, 2.0,
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
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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
    for i in 0..12 {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "multirow mismatch at {}: got {}, expected {}",
            i, got[i], expected[i]
        );
    }
}

/// Larger dim that exercises the multi-thread scan (dim > 1).
///
/// With `dim=512`, `tg_size=256`, `chunk=2`: each thread owns 2 elements.
/// Input = all 1.0 -> output[i] = i+1.
#[test]
fn test_cumsum_f32_large_ones() {
    let (device, mut registry) = setup();
    let dim = 512u32;
    let rows = 1u32;

    let input_data = vec![1.0f32; dim as usize];
    let mut input = device
        .alloc_buffer((dim as usize) * 4, DType::F32, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer((dim as usize) * 4, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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
    for i in 0..dim as usize {
        let expected = (i + 1) as f32;
        let diff = (got[i] - expected).abs();
        assert!(
            diff < 1e-4,
            "large-ones mismatch at {}: got {}, expected {}",
            i, got[i], expected
        );
    }
}

/// Random-input parity against a scalar CPU reference implementation.
#[test]
fn test_cumsum_f32_random_cpu_parity() {
    let (device, mut registry) = setup();
    let dim = 257u32; // deliberately not a power of two
    let rows = 5u32;
    let n = (rows * dim) as usize;

    // Deterministic pseudo-random input.
    let mut input_data = vec![0.0f32; n];
    let mut seed = 0xabcd1234u32;
    for v in input_data.iter_mut() {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *v = ((seed as i32 as f64) / (i32::MAX as f64)) as f32;
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
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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

    // CPU reference in f64 for tight tolerance.
    for r in 0..rows as usize {
        let mut acc = 0.0f64;
        for c in 0..dim as usize {
            acc += input_data[r * dim as usize + c] as f64;
            let expected = acc as f32;
            let idx = r * dim as usize + c;
            let diff = (got[idx] - expected).abs();
            assert!(
                diff < 5e-4,
                "random parity mismatch at (r={}, c={}): got {}, expected {}, diff {}",
                r, c, got[idx], expected, diff
            );
        }
    }
}

// =====================================================================
// BF16
// =====================================================================

/// BF16 textbook.
#[test]
fn test_cumsum_bf16_textbook() {
    use half::bf16;
    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;

    let input_data = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ];
    let mut input = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("input");
    input
        .as_mut_slice::<bf16>()
        .expect("mut")
        .copy_from_slice(&input_data);
    let output = device
        .alloc_buffer(8, DType::BF16, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::cumsum::dispatch_cumsum(
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
    let got_f32 = got.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let expected = [1.0f32, 3.0, 6.0, 10.0];
    for i in 0..4 {
        let diff = (got_f32[i] - expected[i]).abs();
        assert!(diff < 1e-1, "bf16 mismatch at {}: got {}", i, got_f32[i]);
    }
}

// =====================================================================
// Error handling
// =====================================================================

#[test]
fn test_cumsum_rejects_zero_dims() {
    let (device, mut registry) = setup();
    let dim = 4u32;
    let rows = 1u32;
    let input = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("input");
    let output = device
        .alloc_buffer(16, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::cumsum::dispatch_cumsum(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        0,
        dim,
    );
    assert!(res.is_err(), "zero rows should error");

    let mut encoder = device.command_encoder().expect("enc2");
    let res = mlx_native::ops::cumsum::dispatch_cumsum(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        0,
    );
    assert!(res.is_err(), "zero dim should error");
}

#[test]
fn test_cumsum_rejects_dim_over_limit() {
    let (device, mut registry) = setup();
    // tg_size caps at 256, chunk caps at 32 in shader -> max dim = 8192.
    let dim = 16384u32;
    let rows = 1u32;
    let byte_len = (dim as usize) * 4;
    let input = device
        .alloc_buffer(byte_len, DType::F32, vec![dim as usize])
        .expect("input");
    let output = device
        .alloc_buffer(byte_len, DType::F32, vec![dim as usize])
        .expect("output");
    let params = alloc_params(&device, dim);

    let mut encoder = device.command_encoder().expect("enc");
    let res = mlx_native::ops::cumsum::dispatch_cumsum(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
        rows,
        dim,
    );
    assert!(res.is_err(), "oversized dim should error cleanly");
}
