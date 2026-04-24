//! Tests for the lower-triangular unit-diagonal solve kernel
//! (ADR-013 Decision 5).
//!
//! Spec: `X = L \ B` where `L[i][i] = 1` is implicit and only the strict
//! lower triangle of L is read.
//!
//! Forward substitution:
//!
//! ```text
//! x[0, :] = b[0, :]
//! x[i, :] = b[i, :] - sum_{j=0..i-1} L[i, j] * x[j, :]
//! ```
//!
//! Acceptance criteria from ADR-013:
//! 1. Shader + op wrapper exist (covered by this file compiling).
//! 2. 4×4 L with random strict-lower entries, 4×8 B: X = solve(L, B),
//!    verify `|L · X - B| < 1e-4` for F32, `<1e-2` for BF16.
//! 3. Spec-driven 3×3 with hand-computed golden X in the test file.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::tri_solve::TriSolveParams;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn alloc_params(device: &MlxDevice, p: TriSolveParams) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(4 * 4, DType::U32, vec![4])
        .expect("alloc params");
    {
        let s = buf.as_mut_slice::<u32>().expect("mut params");
        s[0] = p.n;
        s[1] = p.m;
        s[2] = p.batch;
        s[3] = 0;
    }
    buf
}

fn upload_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc");
    buf.as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

// Compute `L @ X - B`'s max abs residual to verify correctness.
fn residual_linf(l: &[f32], x: &[f32], b: &[f32], n: usize, m: usize, batch: usize) -> f32 {
    let mut max_abs = 0.0f32;
    for bi in 0..batch {
        for i in 0..n {
            for col in 0..m {
                let mut acc = 0.0f32;
                // L is row-major per batch: L[bi][i][j] at bi*n*n + i*n + j.
                // Unit diagonal: L[i][i] = 1 implicit — include it.
                for j in 0..=i {
                    let l_ij = if j == i { 1.0 } else { l[bi * n * n + i * n + j] };
                    acc += l_ij * x[bi * n * m + j * m + col];
                }
                let expected = b[bi * n * m + i * m + col];
                let d = (acc - expected).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
    }
    max_abs
}

// =================================================================
// Spec-driven 3×3 with hand-computed golden
// =================================================================

/// Hand-computed forward substitution.
///
/// L = [[1,    0,   0],
///      [2,    1,   0],
///      [0.5, -1,   1]]          (unit diagonal; strict-lower in the kernel)
///
/// B = [[10],
///      [4],
///      [0]]
///
/// Forward substitution:
///   x[0] = 10
///   x[1] = 4  - 2 * x[0]         = 4 - 20       = -16
///   x[2] = 0  - 0.5 * x[0] - (-1)*x[1]
///        = 0 - 5 + (-16)          = -21
///
/// Verify: L · X should reproduce B.
///   L@X[0] = x[0]                             = 10  ✓
///   L@X[1] = 2*x[0] + x[1]        = 20 - 16   = 4   ✓
///   L@X[2] = 0.5*x[0] - x[1] + x[2]
///          = 5 + 16 - 21          = 0   ✓
#[test]
fn test_tri_solve_3x3_hand_computed() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 3, m: 1, batch: 1 };

    // L stored row-major; strict lower is what the kernel reads (diagonal is
    // IMPLICIT unit — we write 0 on the diagonal defensively to prove the
    // kernel really ignores it).
    let l_data = [
        0.0_f32, 0.0, 0.0,   // row 0
        2.0, 0.0, 0.0,       // row 1 — L[1,0] = 2
        0.5, -1.0, 0.0,      // row 2 — L[2,0] = 0.5, L[2,1] = -1
    ];
    let b_data = [10.0_f32, 4.0, 0.0];
    let x_expected = [10.0_f32, -16.0, -21.0];

    let l_buf = upload_f32(&device, &l_data);
    let b_buf = upload_f32(&device, &b_data);
    let x_buf = device.alloc_buffer(3 * 4, DType::F32, vec![3]).expect("x");
    let p_buf = alloc_params(&device, p);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let x_got: &[f32] = x_buf.as_slice().expect("read");
    for i in 0..3 {
        let d = (x_got[i] - x_expected[i]).abs();
        assert!(
            d < 1e-5,
            "x[{}]: got {}, expected {}, diff {}",
            i, x_got[i], x_expected[i], d
        );
    }
}

// =================================================================
// Acceptance criterion: 4×4 L × 4×8 RHS, residual L·X - B < 1e-4
// =================================================================

#[test]
fn test_tri_solve_4x4_8rhs_residual() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 4, m: 8, batch: 1 };

    // Deterministic pseudo-random strict-lower triangle; diagonal set to 0.
    let mut seed = 0x4242u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.8
    };

    let mut l_data = vec![0.0_f32; 16];
    for i in 0..4 {
        for j in 0..i {
            l_data[i * 4 + j] = rand();
        }
        // Diagonal and upper-triangle stay 0; kernel ignores.
    }

    let mut b_data = vec![0.0_f32; 4 * 8];
    for v in b_data.iter_mut() {
        *v = rand();
    }

    let l_buf = upload_f32(&device, &l_data);
    let b_buf = upload_f32(&device, &b_data);
    let x_buf = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("x");
    let p_buf = alloc_params(&device, p);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let x_got: Vec<f32> = x_buf.as_slice::<f32>().expect("read").to_vec();
    let res = residual_linf(&l_data, &x_got, &b_data, 4, 8, 1);
    assert!(
        res < 1e-4,
        "|L · X - B|_inf = {} exceeds 1e-4 tolerance",
        res
    );
}

// =================================================================
// Batched case
// =================================================================

#[test]
fn test_tri_solve_batched_4x4_3rhs_5batch() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 4, m: 3, batch: 5 };

    let mut seed = 0xBEEFu32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.6
    };

    let n = p.n as usize;
    let m = p.m as usize;
    let batch = p.batch as usize;

    let mut l_data = vec![0.0_f32; batch * n * n];
    for bi in 0..batch {
        for i in 0..n {
            for j in 0..i {
                l_data[bi * n * n + i * n + j] = rand();
            }
        }
    }

    let mut b_data = vec![0.0_f32; batch * n * m];
    for v in b_data.iter_mut() {
        *v = rand();
    }

    let l_buf = upload_f32(&device, &l_data);
    let b_buf = upload_f32(&device, &b_data);
    let x_buf = device
        .alloc_buffer(b_data.len() * 4, DType::F32, vec![b_data.len()])
        .expect("x");
    let p_buf = alloc_params(&device, p);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let x_got: Vec<f32> = x_buf.as_slice::<f32>().expect("read").to_vec();
    let res = residual_linf(&l_data, &x_got, &b_data, n, m, batch);
    assert!(
        res < 1e-4,
        "batched |L·X - B|_inf = {} exceeds 1e-4 tolerance",
        res
    );

    // Spot-check: results should be different across batches because L and B
    // were generated differently per batch.
    let mut any_differ = false;
    for col in 0..m {
        if (x_got[col] - x_got[n * m + col]).abs() > 1e-6 {
            any_differ = true;
            break;
        }
    }
    assert!(any_differ, "batches produced identical row 0 — likely bug");
}

// =================================================================
// Identity-L case (strict lower all zero -> X should equal B exactly)
// =================================================================

#[test]
fn test_tri_solve_identity_returns_b_unchanged() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 5, m: 4, batch: 1 };

    let l_data = vec![0.0_f32; 25]; // L = identity implicitly (zero lower)
    let b_data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.37 - 3.0).collect();

    let l_buf = upload_f32(&device, &l_data);
    let b_buf = upload_f32(&device, &b_data);
    let x_buf = device
        .alloc_buffer(20 * 4, DType::F32, vec![20])
        .expect("x");
    let p_buf = alloc_params(&device, p);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let x: &[f32] = x_buf.as_slice().expect("read");
    for i in 0..20 {
        let d = (x[i] - b_data[i]).abs();
        assert!(
            d < 1e-6,
            "identity solve mismatch at {}: x={}, b={}",
            i, x[i], b_data[i]
        );
    }
}

// =================================================================
// BF16 path
// =================================================================

#[test]
fn test_tri_solve_bf16_4x4_4rhs() {
    use half::bf16;
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 4, m: 4, batch: 1 };

    // Small off-diagonal magnitudes keep BF16 numerics reasonable.
    let mut seed = 0xCAFEu32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.3
    };

    let mut l_f32 = vec![0.0_f32; 16];
    for i in 0..4 {
        for j in 0..i {
            l_f32[i * 4 + j] = rand();
        }
    }
    let b_f32: Vec<f32> = (0..16).map(|_| rand()).collect();

    let l_bf: Vec<bf16> = l_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let b_bf: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut l_buf = device.alloc_buffer(32, DType::BF16, vec![16]).expect("l");
    l_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&l_bf);
    let mut b_buf = device.alloc_buffer(32, DType::BF16, vec![16]).expect("b");
    b_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&b_bf);
    let x_buf = device.alloc_buffer(32, DType::BF16, vec![16]).expect("x");
    let p_buf = alloc_params(&device, p);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let x_bf: &[bf16] = x_buf.as_slice().expect("read");
    let x_f32: Vec<f32> = x_bf.iter().map(|v| v.to_f32()).collect();

    let res = residual_linf(&l_f32, &x_f32, &b_f32, 4, 4, 1);
    assert!(
        res < 1e-2,
        "bf16 residual {} exceeds 1e-2 tolerance",
        res
    );
}

// =================================================================
// Error handling
// =================================================================

#[test]
fn test_tri_solve_rejects_zero_n() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 0, m: 1, batch: 1 };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let p_buf = alloc_params(&device, p);
    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy,
        &dummy,
        &dummy,
        &p_buf,
        p,
    );
    assert!(res.is_err(), "n=0 should error");
}

#[test]
fn test_tri_solve_rejects_element_count_mismatch() {
    let (device, mut registry) = setup();
    let p = TriSolveParams { n: 4, m: 2, batch: 1 };
    // Wrong element count for L (should be 16, give 10).
    let l_buf = device.alloc_buffer(40, DType::F32, vec![10]).expect("l");
    let b_buf = device.alloc_buffer(32, DType::F32, vec![8]).expect("b");
    let x_buf = device.alloc_buffer(32, DType::F32, vec![8]).expect("x");
    let p_buf = alloc_params(&device, p);
    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::tri_solve::dispatch_tri_solve(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &l_buf,
        &b_buf,
        &x_buf,
        &p_buf,
        p,
    );
    assert!(res.is_err(), "element count mismatch should error");
}
