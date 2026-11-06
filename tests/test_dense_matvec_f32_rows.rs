//! Correctness gates for `dispatch_dense_matvec_f32_rows`:
//! - F64 reference within the K-scaled bound `(K / 32 + 8) * 2^-24 * sum |x w|`
//!   at the Qwen3.6 router (256 x 2048), shared-gate logit (1 x 2048) and an
//!   odd tail shape (7 x 132), rows 1..8;
//! - row exactness: every row of a `rows = R` dispatch (R = 2..8) is bitwise the
//!   `rows = 1` dispatch of that row, for any row order and for row views;
//! - argument validation.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::dense_matvec_f32_rows::dispatch_dense_matvec_f32_rows;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn lcg(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn upload(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut b = device.alloc_buffer(data.len() * 4, DType::F32, shape).expect("alloc");
    b.as_mut_slice::<f32>().expect("mut").copy_from_slice(data);
    b
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// One dispatch per entry of `batches`: each is a list of row indices into `x`.
/// Returns the output rows of every dispatch, in dispatch order.
fn run(x: &[f32], w: &[f32], n: usize, k: usize, batches: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let wbuf = upload(&device, w, vec![n, k]);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for rows in batches {
        let mut packed = Vec::with_capacity(rows.len() * k);
        for &r in rows {
            packed.extend_from_slice(&x[r * k..(r + 1) * k]);
        }
        inputs.push(upload(&device, &packed, vec![rows.len(), k]));
        outputs.push(
            device
                .alloc_buffer(rows.len() * n * 4, DType::F32, vec![rows.len(), n])
                .expect("out"),
        );
    }
    let mut enc = device.command_encoder().expect("enc");
    for (i, rows) in batches.iter().enumerate() {
        dispatch_dense_matvec_f32_rows(
            &mut enc,
            &mut registry,
            &device,
            &inputs[i],
            &wbuf,
            &outputs[i],
            rows.len() as u32,
            n as u32,
            k as u32,
        )
        .expect("dispatch");
    }
    enc.commit_and_wait().expect("commit");
    outputs
        .iter()
        .map(|o| o.as_slice::<f32>().expect("read").to_vec())
        .collect()
}

fn check_reference(n: usize, k: usize, rows: usize, seed: u64) {
    let x = lcg(seed, rows * k, 1.0);
    let w = lcg(seed ^ 0x9e37, n * k, 0.05);
    let out = run(&x, &w, n, k, &[(0..rows).collect()]);
    let unit = (2.0f64).powi(-24);
    let mut worst = 0.0f64;
    for r in 0..rows {
        for o in 0..n {
            let mut exact = 0.0f64;
            let mut magnitude = 0.0f64;
            for i in 0..k {
                let p = x[r * k + i] as f64 * w[o * k + i] as f64;
                exact += p;
                magnitude += p.abs();
            }
            let got = out[0][r * n + o] as f64;
            let bound = (k as f64 / 32.0 + 8.0) * unit * magnitude;
            let err = (got - exact).abs();
            assert!(
                err <= bound,
                "n={n} k={k} rows={rows} r={r} o={o}: |{got} - {exact}| = {err} > {bound}"
            );
            worst = worst.max(err / magnitude.max(f64::MIN_POSITIVE));
        }
    }
    eprintln!("dense_matvec_f32_rows n={n} k={k} rows={rows}: worst |err|/sum|xw| = {worst:.3e}");
}

#[test]
fn matches_f64_reference_within_k_scaled_bound() {
    for rows in 1..=8 {
        check_reference(256, 2048, rows, 11 + rows as u64);
        check_reference(1, 2048, rows, 101 + rows as u64);
        check_reference(7, 132, rows, 1001 + rows as u64);
    }
}

#[test]
fn batched_rows_equal_one_row_dispatches_bitwise() {
    for &(n, k) in &[(256usize, 2048usize), (1, 2048), (7, 132), (2048, 512)] {
        let x = lcg(7 + n as u64, 8 * k, 1.0);
        let w = lcg(13 + k as u64, n * k, 0.05);
        let singles: Vec<Vec<usize>> = (0..8).map(|r| vec![r]).collect();
        let single = run(&x, &w, n, k, &singles);
        let mut batches: Vec<Vec<usize>> = (2..=8).map(|r| (0..r).collect()).collect();
        // Permuted and shifted row sets: a row's result must not depend on its
        // position or on which rows share the dispatch.
        batches.push(vec![3, 1, 2]);
        batches.push(vec![7, 0, 5, 2]);
        batches.push(vec![6, 4]);
        let batched = run(&x, &w, n, k, &batches);
        for (rows, out) in batches.iter().zip(&batched) {
            for (slot, &r) in rows.iter().enumerate() {
                assert_eq!(
                    bits(&out[slot * n..(slot + 1) * n]),
                    bits(&single[r]),
                    "n={n} k={k} rows={rows:?}: row {r} differs from its rows=1 dispatch"
                );
            }
        }
    }
}

#[test]
fn row_views_are_honoured() {
    let (n, k) = (256usize, 2048usize);
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let x = lcg(5, 4 * k, 1.0);
    let w = lcg(6, n * k, 0.05);
    let xbuf = upload(&device, &x, vec![4, k]);
    let wbuf = upload(&device, &w, vec![n, k]);
    let whole = device.alloc_buffer(4 * n * 4, DType::F32, vec![4, n]).expect("out");
    let parts = device.alloc_buffer(4 * n * 4, DType::F32, vec![4, n]).expect("out");
    let mut enc = device.command_encoder().expect("enc");
    dispatch_dense_matvec_f32_rows(&mut enc, &mut registry, &device, &xbuf, &wbuf, &whole, 4, n as u32, k as u32)
        .expect("whole");
    for r in 0..4 {
        let xin = xbuf.slice_view((r * k * 4) as u64, k);
        let yout = parts.slice_view((r * n * 4) as u64, n);
        dispatch_dense_matvec_f32_rows(&mut enc, &mut registry, &device, &xin, &wbuf, &yout, 1, n as u32, k as u32)
            .expect("row");
    }
    enc.commit_and_wait().expect("commit");
    assert_eq!(
        bits(whole.as_slice::<f32>().unwrap()),
        bits(parts.as_slice::<f32>().unwrap())
    );
}

#[test]
fn rejects_invalid_arguments() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let x = upload(&device, &vec![0.0; 16 * 132], vec![16, 132]);
    let w = upload(&device, &vec![0.0; 8 * 132], vec![8, 132]);
    let y = device.alloc_buffer(16 * 8 * 4, DType::F32, vec![16, 8]).expect("out");
    let mut enc = device.command_encoder().expect("enc");
    let mut call = |input: &MlxBuffer, rows: u32, n: u32, k: u32| {
        dispatch_dense_matvec_f32_rows(&mut enc, &mut registry, &device, input, &w, &y, rows, n, k)
    };
    assert!(call(&x, 0, 8, 132).is_err());
    assert!(call(&x, 9, 8, 132).is_err());
    assert!(call(&x, 1, 8, 130).is_err());
    assert!(call(&x, 1, 0, 132).is_err());
    assert!(call(&x, 1, 9, 132).is_err(), "weight too small");
    assert!(call(&x.slice_view(4, 132), 1, 8, 132).is_err(), "misaligned view");
}
