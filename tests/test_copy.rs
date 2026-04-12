//! Tests for the strided copy GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::copy::{self, StridedCopyParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    copy::register(&mut registry);
    (device, registry)
}

#[test]
fn test_copy_contiguous_identity() {
    let (device, mut registry) = setup();
    let rows: u32 = 4;
    let cols: u32 = 8;
    let total = rows as usize * cols as usize;

    // Fill source with sequential values, contiguous layout.
    let src_data: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let byte_len = total * 4;
    let mut src_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(&src_data);

    let dst_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc dst");

    let params = StridedCopyParams {
        rows,
        cols,
        stride_row: cols,  // contiguous: stride = cols
        stride_col: 1,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    copy::dispatch_strided_copy_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &dst_buf,
        &params,
    )
    .expect("dispatch_strided_copy_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output = dst_buf.as_slice::<f32>().expect("read dst");

    // Should be identity copy.
    for i in 0..total {
        assert!(
            (output[i] - src_data[i]).abs() < 1e-6,
            "contiguous copy mismatch at {}: GPU={}, expected={}",
            i, output[i], src_data[i]
        );
    }
}

#[test]
fn test_copy_transposed() {
    let (device, mut registry) = setup();
    // Original layout: [4, 8] with stride [8, 1] (contiguous).
    // After transpose: logically [8, 4] with stride [1, 8] (transposed).
    // Strided copy should produce contiguous [8, 4] with stride [4, 1].

    let orig_rows: usize = 4;
    let orig_cols: usize = 8;
    let total = orig_rows * orig_cols;

    // Source data in row-major [4, 8] layout.
    let src_data: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let byte_len = total * 4;
    let mut src_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![orig_rows, orig_cols])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(&src_data);

    // Output: [8, 4] contiguous.
    let out_rows: u32 = 8;
    let out_cols: u32 = 4;
    let dst_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![out_rows as usize, out_cols as usize])
        .expect("alloc dst");

    // Transposed strides: stride_row=1 (columns in original), stride_col=8 (rows in original).
    let params = StridedCopyParams {
        rows: out_rows,
        cols: out_cols,
        stride_row: 1,   // step 1 element for each row of the transposed view
        stride_col: 8,   // step 8 elements for each col of the transposed view
    };

    let mut encoder = device.command_encoder().expect("encoder");
    copy::dispatch_strided_copy_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &dst_buf,
        &params,
    )
    .expect("dispatch_strided_copy_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output = dst_buf.as_slice::<f32>().expect("read dst");

    // Verify: output[r][c] should equal src_data[c * orig_cols + r]
    // which is src_data[r * stride_row + c * stride_col] = src_data[r * 1 + c * 8]
    for r in 0..out_rows as usize {
        for c in 0..out_cols as usize {
            let expected = src_data[r * 1 + c * 8];
            let actual = output[r * out_cols as usize + c];
            assert!(
                (actual - expected).abs() < 1e-6,
                "transposed copy mismatch at [{}, {}]: GPU={}, expected={}",
                r, c, actual, expected
            );
        }
    }
}
