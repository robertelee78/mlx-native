//! Tests for elementwise add, multiply, cast, and transpose GPU kernels.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

// ---- elementwise_add f32 ----

#[test]
fn test_elementwise_add_f32_basic() {
    let (device, mut registry) = setup();

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 100.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 1.0, 0.0, -0.5, -100.0];
    let n = a_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("alloc a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("alloc b");
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("alloc out");

    a_buf.as_mut_slice::<f32>().expect("mut a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("mut b").copy_from_slice(&b_data);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::elementwise::elementwise_add(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &a_buf,
        &b_buf,
        &out_buf,
        n,
        DType::F32,
    )
    .expect("elementwise_add");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = out_buf.as_slice().expect("read");
    for i in 0..n {
        let expected = a_data[i] + b_data[i];
        let diff = (output[i] - expected).abs();
        assert!(
            diff < 1e-6,
            "add f32 mismatch at {}: expected {}, got {}, diff {}",
            i, expected, output[i], diff
        );
    }
}

#[test]
fn test_elementwise_add_f32_zeros() {
    let (device, mut registry) = setup();

    let a_data: Vec<f32> = vec![0.0; 16];
    let b_data: Vec<f32> = vec![0.0; 16];
    let n = 16;
    let byte_len = n * 4;

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::elementwise::elementwise_add(
        &mut encoder, &mut registry, device.metal_device(),
        &a_buf, &b_buf, &out_buf, n, DType::F32,
    ).expect("add");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = out_buf.as_slice().expect("read");
    for &val in output {
        assert!(val.abs() < 1e-7, "expected 0, got {}", val);
    }
}

// ---- elementwise_mul f32 ----

#[test]
fn test_elementwise_mul_f32() {
    let (device, mut registry) = setup();

    let a_data: Vec<f32> = vec![2.0, 3.0, -1.0, 0.5];
    let b_data: Vec<f32> = vec![4.0, -2.0, 5.0, 10.0];
    let n = 4;
    let byte_len = n * 4;

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::elementwise::elementwise_mul(
        &mut encoder, &mut registry, device.metal_device(),
        &a_buf, &b_buf, &out_buf, n, DType::F32,
    ).expect("mul");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = out_buf.as_slice().expect("read");
    let expected: Vec<f32> = vec![8.0, -6.0, -5.0, 5.0];
    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(diff < 1e-6, "mul mismatch at {}: expected {}, got {}", i, expected[i], output[i]);
    }
}

// ---- cast f32 <-> f16 ----

#[test]
fn test_cast_f32_to_f16_and_back() {
    let (device, mut registry) = setup();

    let data: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001, 65504.0];
    let n = data.len();

    let mut f32_buf = device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("f32");
    let f16_buf = device.alloc_buffer(n * 2, DType::F16, vec![n]).expect("f16");
    let f32_back = device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("f32 back");

    f32_buf.as_mut_slice::<f32>().expect("write").copy_from_slice(&data);

    // f32 -> f16
    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::elementwise::cast(
        &mut encoder, &mut registry, device.metal_device(),
        &f32_buf, &f16_buf, n,
        mlx_native::ops::elementwise::CastDirection::F32ToF16,
    ).expect("cast f32->f16");
    encoder.commit_and_wait().expect("commit");

    // f16 -> f32
    let mut encoder2 = device.command_encoder().expect("enc2");
    mlx_native::ops::elementwise::cast(
        &mut encoder2, &mut registry, device.metal_device(),
        &f16_buf, &f32_back, n,
        mlx_native::ops::elementwise::CastDirection::F16ToF32,
    ).expect("cast f16->f32");
    encoder2.commit_and_wait().expect("commit2");

    let output: &[f32] = f32_back.as_slice().expect("read");
    for i in 0..n {
        // f16 has limited precision, so we allow some tolerance
        let diff = (output[i] - data[i]).abs();
        let rel_tol = data[i].abs() * 1e-3 + 1e-4;
        assert!(
            diff <= rel_tol,
            "cast roundtrip mismatch at {}: original {}, got {}, diff {}",
            i, data[i], output[i], diff
        );
    }
}

// ---- bf16↔f16 cast (added 2026-05-16 for F16 FA migration step 2) ----

/// Round-trip test: bf16 → f16 → bf16 should preserve values within BF16's
/// natural precision band.  BF16 has 7-bit mantissa, F16 has 10-bit, so
/// the F16 intermediate carries the BF16 precision losslessly; the only
/// rounding step is the final F16 → BF16 cast.
#[test]
fn test_cast_bf16_to_f16_and_back() {
    use half::{bf16, f16};

    let (device, mut registry) = setup();

    let data_f32: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001, 15.0];
    let data_bf: Vec<bf16> = data_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let n = data_bf.len();

    let mut bf_in = device.alloc_buffer(n * 2, DType::BF16, vec![n]).expect("bf in");
    let f16_buf = device.alloc_buffer(n * 2, DType::F16, vec![n]).expect("f16");
    let bf_back = device.alloc_buffer(n * 2, DType::BF16, vec![n]).expect("bf back");

    bf_in.as_mut_slice::<bf16>().expect("write").copy_from_slice(&data_bf);

    // bf16 -> f16
    let mut encoder = device.command_encoder().expect("enc1");
    mlx_native::ops::elementwise::cast(
        &mut encoder, &mut registry, device.metal_device(),
        &bf_in, &f16_buf, n,
        mlx_native::ops::elementwise::CastDirection::BF16ToF16,
    ).expect("cast bf16->f16");
    encoder.commit_and_wait().expect("commit1");

    // f16 -> bf16
    let mut encoder2 = device.command_encoder().expect("enc2");
    mlx_native::ops::elementwise::cast(
        &mut encoder2, &mut registry, device.metal_device(),
        &f16_buf, &bf_back, n,
        mlx_native::ops::elementwise::CastDirection::F16ToBF16,
    ).expect("cast f16->bf16");
    encoder2.commit_and_wait().expect("commit2");

    let f16_mid: &[f16] = f16_buf.as_slice().expect("read mid");
    let bf_out: &[bf16] = bf_back.as_slice().expect("read out");

    // Verify the intermediate F16 representation is at least as precise as
    // the BF16 input (F16's 10-bit mantissa > BF16's 7-bit).
    for i in 0..n {
        let bf_val = data_bf[i].to_f32();
        let f16_val = f16_mid[i].to_f32();
        let err = (bf_val - f16_val).abs();
        let tol = bf_val.abs() * 1e-3 + 1e-4;
        assert!(
            err <= tol,
            "bf16→f16 lost precision at idx {i}: bf16={bf_val:.6e} f16={f16_val:.6e} err={err:.3e}"
        );
    }

    // Round-trip: bf16 → f16 → bf16 should recover the original BF16 exactly
    // (no rounding-trip loss because F16 preserves all BF16 precision).
    for i in 0..n {
        assert_eq!(
            bf_out[i].to_bits(),
            data_bf[i].to_bits(),
            "bf16→f16→bf16 round-trip not bit-exact at idx {i}: \
             in=0x{:04x} out=0x{:04x} (f32 in={} f32 out={})",
            data_bf[i].to_bits(),
            bf_out[i].to_bits(),
            data_bf[i].to_f32(),
            bf_out[i].to_f32(),
        );
    }
}

// ---- transpose 2d f32 ----

#[test]
fn test_transpose_2d_f32() {
    let (device, mut registry) = setup();

    // 3x4 matrix
    let rows = 3;
    let cols = 4;
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let n = rows * cols;
    let byte_len = n * 4;

    let mut input = device.alloc_buffer(byte_len, DType::F32, vec![rows, cols]).expect("in");
    let output = device.alloc_buffer(byte_len, DType::F32, vec![cols, rows]).expect("out");

    input.as_mut_slice::<f32>().expect("write").copy_from_slice(&data);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::transpose::transpose_2d(
        &mut encoder, &mut registry, device.metal_device(),
        &input, &output, rows, cols, DType::F32,
    ).expect("transpose");
    encoder.commit_and_wait().expect("commit");

    let result: &[f32] = output.as_slice().expect("read");

    // Verify: output[col][row] == input[row][col]
    for r in 0..rows {
        for c in 0..cols {
            let input_val = data[r * cols + c];
            let output_val = result[c * rows + r];
            assert!(
                (input_val - output_val).abs() < 1e-7,
                "transpose mismatch: input[{},{}]={} != output[{},{}]={}",
                r, c, input_val, c, r, output_val
            );
        }
    }
}

#[test]
fn test_transpose_2d_f32_square() {
    let (device, mut registry) = setup();

    let n = 4;
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let byte_len = 16 * 4;

    let mut input = device.alloc_buffer(byte_len, DType::F32, vec![n, n]).expect("in");
    let output = device.alloc_buffer(byte_len, DType::F32, vec![n, n]).expect("out");

    input.as_mut_slice::<f32>().expect("write").copy_from_slice(&data);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::transpose::transpose_2d(
        &mut encoder, &mut registry, device.metal_device(),
        &input, &output, n, n, DType::F32,
    ).expect("transpose");
    encoder.commit_and_wait().expect("commit");

    let result: &[f32] = output.as_slice().expect("read");
    for r in 0..n {
        for c in 0..n {
            let expected = data[r * n + c];
            let actual = result[c * n + r];
            assert!(
                (expected - actual).abs() < 1e-7,
                "square transpose mismatch at [{},{}]",
                r, c
            );
        }
    }
}

// ---- Validation tests ----

#[test]
fn test_elementwise_add_zero_elements_error() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(16, DType::F32, vec![4]).expect("buf");
    let mut encoder = device.command_encoder().expect("enc");
    let result = mlx_native::ops::elementwise::elementwise_add(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, 0, DType::F32,
    );
    assert!(result.is_err(), "zero elements should error");
}

#[test]
fn test_transpose_zero_rows_error() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(16, DType::F32, vec![4]).expect("buf");
    let mut encoder = device.command_encoder().expect("enc");
    let result = mlx_native::ops::transpose::transpose_2d(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, 0, 4, DType::F32,
    );
    assert!(result.is_err(), "zero rows should error");
}

#[test]
fn test_cast_zero_elements_error() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(16, DType::F32, vec![4]).expect("buf");
    let mut encoder = device.command_encoder().expect("enc");
    let result = mlx_native::ops::elementwise::cast(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, 0,
        mlx_native::ops::elementwise::CastDirection::F32ToF16,
    );
    assert!(result.is_err(), "zero elements should error");
}
