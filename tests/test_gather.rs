//! Tests for the gather / index_select GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::gather;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    gather::register(&mut registry);
    (device, registry)
}

#[test]
fn test_gather_basic() {
    let (device, mut registry) = setup();
    let src_rows: u32 = 8;
    let row_width: u32 = 64;
    let n_indices: u32 = 4;

    // Fill source with known pattern: src[r][c] = r * 100 + c
    let mut src_data = vec![0.0f32; src_rows as usize * row_width as usize];
    for r in 0..src_rows as usize {
        for c in 0..row_width as usize {
            src_data[r * row_width as usize + c] = r as f32 * 100.0 + c as f32;
        }
    }

    let indices_data: Vec<u32> = vec![3, 7, 1, 5];

    let src_bytes = src_data.len() * 4;
    let mut src_buf = device
        .alloc_buffer(src_bytes, DType::F32, vec![src_rows as usize, row_width as usize])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(&src_data);

    let idx_bytes = n_indices as usize * 4;
    let mut idx_buf = device
        .alloc_buffer(idx_bytes, DType::U32, vec![n_indices as usize])
        .expect("alloc indices");
    idx_buf
        .as_mut_slice::<u32>()
        .expect("write indices")
        .copy_from_slice(&indices_data);

    let out_bytes = n_indices as usize * row_width as usize * 4;
    let out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_indices as usize, row_width as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    gather::dispatch_gather_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &idx_buf,
        &out_buf,
        src_rows,
        row_width,
        n_indices,
    )
    .expect("dispatch_gather_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output = out_buf.as_slice::<f32>().expect("read output");

    // Verify output[i, :] == src[indices[i], :]
    for (i, &idx) in indices_data.iter().enumerate() {
        for c in 0..row_width as usize {
            let expected = idx as f32 * 100.0 + c as f32;
            let actual = output[i * row_width as usize + c];
            assert!(
                (actual - expected).abs() < 1e-6,
                "gather mismatch at [{}][{}]: GPU={}, expected={}",
                i, c, actual, expected
            );
        }
    }
}

#[test]
fn test_gather_sequential_identity() {
    let (device, mut registry) = setup();
    let src_rows: u32 = 8;
    let row_width: u32 = 4;
    let n_indices: u32 = 4;

    let mut src_data = vec![0.0f32; src_rows as usize * row_width as usize];
    for r in 0..src_rows as usize {
        for c in 0..row_width as usize {
            src_data[r * row_width as usize + c] = (r * 10 + c) as f32;
        }
    }

    // Sequential indices [0, 1, 2, 3] — should be identity for first 4 rows.
    let indices_data: Vec<u32> = vec![0, 1, 2, 3];

    let src_bytes = src_data.len() * 4;
    let mut src_buf = device
        .alloc_buffer(src_bytes, DType::F32, vec![src_rows as usize, row_width as usize])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(&src_data);

    let idx_bytes = n_indices as usize * 4;
    let mut idx_buf = device
        .alloc_buffer(idx_bytes, DType::U32, vec![n_indices as usize])
        .expect("alloc indices");
    idx_buf
        .as_mut_slice::<u32>()
        .expect("write indices")
        .copy_from_slice(&indices_data);

    let out_bytes = n_indices as usize * row_width as usize * 4;
    let out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_indices as usize, row_width as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    gather::dispatch_gather_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &idx_buf,
        &out_buf,
        src_rows,
        row_width,
        n_indices,
    )
    .expect("dispatch_gather_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output = out_buf.as_slice::<f32>().expect("read output");

    // Output should match first 4 rows of source exactly.
    for i in 0..n_indices as usize * row_width as usize {
        assert!(
            (output[i] - src_data[i]).abs() < 1e-6,
            "identity gather mismatch at flat index {}: GPU={}, src={}",
            i, output[i], src_data[i]
        );
    }
}

#[test]
fn test_gather_repeated_indices() {
    let (device, mut registry) = setup();
    let src_rows: u32 = 8;
    let row_width: u32 = 4;
    let n_indices: u32 = 4;

    let mut src_data = vec![0.0f32; src_rows as usize * row_width as usize];
    for r in 0..src_rows as usize {
        for c in 0..row_width as usize {
            src_data[r * row_width as usize + c] = (r * 10 + c) as f32;
        }
    }

    // All indices point to row 2.
    let indices_data: Vec<u32> = vec![2, 2, 2, 2];

    let src_bytes = src_data.len() * 4;
    let mut src_buf = device
        .alloc_buffer(src_bytes, DType::F32, vec![src_rows as usize, row_width as usize])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(&src_data);

    let idx_bytes = n_indices as usize * 4;
    let mut idx_buf = device
        .alloc_buffer(idx_bytes, DType::U32, vec![n_indices as usize])
        .expect("alloc indices");
    idx_buf
        .as_mut_slice::<u32>()
        .expect("write indices")
        .copy_from_slice(&indices_data);

    let out_bytes = n_indices as usize * row_width as usize * 4;
    let out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![n_indices as usize, row_width as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    gather::dispatch_gather_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &idx_buf,
        &out_buf,
        src_rows,
        row_width,
        n_indices,
    )
    .expect("dispatch_gather_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output = out_buf.as_slice::<f32>().expect("read output");

    // All 4 output rows should equal src[2].
    let row2 = &src_data[2 * row_width as usize..3 * row_width as usize];
    for i in 0..n_indices as usize {
        for c in 0..row_width as usize {
            assert!(
                (output[i * row_width as usize + c] - row2[c]).abs() < 1e-6,
                "repeated gather: output[{}][{}]={}, expected src[2][{}]={}",
                i, c, output[i * row_width as usize + c], c, row2[c]
            );
        }
    }
}
