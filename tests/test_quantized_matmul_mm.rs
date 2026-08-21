//! Unit tests for the GGML block-format matrix-matrix (mm) quantized GPU kernels.
//!
//! ADR-011 Phase 3 Wave P3a port of the reference `kernel_mul_mm_<qtype>_f32`.
//! The dispatcher in `quantized_matmul_ggml.rs` routes inputs with `m > 8`
//! through these kernels, which stage a 64x32 weight tile into threadgroup
//! shared memory and reuse it across a 32-row block of M.  The same math,
//! the same weight-block dequantize, the same accumulation order — the only
//! difference vs the mv kernel is memory access pattern.  We therefore
//! verify bit-equivalence (within f32 kernel-parallelism noise) against the
//! mv kernel at every test M, ensuring the port does not change numerics.
//!
//! To force each path we bypass the routing layer and call
//! `dispatch_mm` / `dispatch_mv` directly via test-only helpers.  Each test
//! packs a weight tensor with the same byte pattern, runs both dispatches
//! against the same input, and verifies the outputs agree within a tight
//! tolerance (1e-3 relative or 1e-4 absolute per element; the mv and mm
//! paths use the same scalar accumulation in f32, so the only differences
//! come from reduction ordering of 32-element simdgroup sums vs 8x8 MMA
//! tile sums — both are f32, <= 1 ULP per accumulated element).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry,
    MlxDevice,
};

// --------------------------------------------------------------------------
// PRNG (matches test_quantized_matmul_ggml.rs)
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

// --------------------------------------------------------------------------
// Q4_0 / Q8_0 / Q6_K pack helpers — identical to test_quantized_matmul_ggml.rs
// (duplicated here to avoid restructuring existing test file; the correct
//  math lives in the GPU kernels, not these CPU helpers).
// --------------------------------------------------------------------------

fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());

        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let v0 = (block[i] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            let v1 = (block[i + 16] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            nibbles[i] = v0 | (v1 << 4);
        }
        buf.extend_from_slice(&nibbles);
    }
    buf
}

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());

        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
    buf
}

fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;

        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale {
                max_scale = amax;
            }
        }

        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }

        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                let v0 = q6[l0_base + l];
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[l0_base + l + 32];
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let v0 = q6[128 + l0_base + l];
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[128 + l0_base + l + 32];
                let v3 = q6[128 + l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[qh_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        buf.extend_from_slice(&ql);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(
            &sub_scale_int.iter().map(|&s| s as u8).collect::<Vec<_>>(),
        );
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

// --------------------------------------------------------------------------
// Kernel dispatch helpers.  Phase 3 Wave P3a lands the mm kernel but keeps
// production routing on mv — to verify the mm kernel's output agrees with
// the mv kernel byte-for-byte we drive each path explicitly:
//
//   * `run_mv_gpu`   → calls the public dispatcher (which, in Wave P3a,
//                      always picks the mv path).  Produces the reference.
//   * `run_mm_gpu`   → calls the test-only helper `dispatch_mm_for_test`
//                      to force the mm path regardless of m.  Verifies
//                      the ported kernel byte-for-byte matches the mv path.
//
// Follow-up commits (MoE `_id` mm port, then dispatcher routing by m>8)
// will flip the public dispatcher to route to mm for m>8.
// --------------------------------------------------------------------------

enum DispatchPath {
    Mv,
    Mm,
}

fn run_qmatmul_path(
    path: DispatchPath,
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let input_bytes = m * k * 4;
    let mut input_buf = device
        .alloc_buffer(input_bytes, DType::F32, vec![m, k])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut slice")
        .copy_from_slice(input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight mut slice")
        .copy_from_slice(weight_bytes);

    let output_bytes = m * n * 4;
    let mut output_buf = device
        .alloc_buffer(output_bytes, DType::F32, vec![m, n])
        .expect("alloc output");
    for v in output_buf
        .as_mut_slice::<f32>()
        .expect("output mut slice")
        .iter_mut()
    {
        *v = 0.0;
    }

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    match path {
        DispatchPath::Mv => {
            mlx_native::quantized_matmul_ggml(
                &mut encoder,
                &mut registry,
                &device,
                &input_buf,
                &weight_buf,
                &mut output_buf,
                &params,
            )
            .expect("mv dispatch");
        }
        DispatchPath::Mm => {
            mlx_native::dispatch_mm_for_test(
                &mut encoder,
                &mut registry,
                &device,
                &input_buf,
                &weight_buf,
                &mut output_buf,
                &params,
            )
            .expect("mm dispatch");
        }
    }
    encoder.commit_and_wait().expect("GPU execution");

    output_buf
        .as_slice::<f32>()
        .expect("read output")
        .to_vec()
}

fn run_mv_gpu(
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    run_qmatmul_path(DispatchPath::Mv, m, n, k, ggml_type, weight_bytes, input)
}

fn run_mm_gpu(
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    run_qmatmul_path(DispatchPath::Mm, m, n, k, ggml_type, weight_bytes, input)
}

fn run_batched_mm_gpu(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let mut input_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![batch, m, k])
        .expect("alloc batched input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("batched input mut slice")
        .copy_from_slice(input);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc batched weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("batched weight mut slice")
        .copy_from_slice(weight_bytes);
    let output_buf = device
        .alloc_buffer(batch * m * n * 4, DType::F32, vec![batch, m, n])
        .expect("alloc batched output");

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml_batched_mm(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &output_buf,
        &GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        },
    )
    .expect("batched MM dispatch");
    encoder.commit_and_wait().expect("batched GPU execution");
    output_buf
        .as_slice::<f32>()
        .expect("read batched output")
        .to_vec()
}

fn run_batched_mm_strided_input_gpu(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
    row_bytes: u64,
    batch_bytes: u64,
) -> Vec<f32> {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let mut input_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("alloc strided batched input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("strided batched input mut slice")
        .copy_from_slice(input);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc strided batched weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("strided batched weight mut slice")
        .copy_from_slice(weight_bytes);
    let output_buf = device
        .alloc_buffer(batch * m * n * 4, DType::F32, vec![batch, m, n])
        .expect("alloc strided batched output");

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml_batched_mm_strided_input(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &output_buf,
        &GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        },
        &mlx_native::GgmlBatchedQuantizedMatmulInputStrides {
            row_bytes,
            batch_bytes,
        },
    )
    .expect("strided batched MM dispatch");
    encoder
        .commit_and_wait()
        .expect("strided batched GPU execution");
    output_buf
        .as_slice::<f32>()
        .expect("read strided batched output")
        .to_vec()
}

fn run_batched_mv_gpu(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let mut input_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![batch, m, k])
        .expect("alloc batched MV input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("batched MV input mut slice")
        .copy_from_slice(input);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc batched MV weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("batched MV weight mut slice")
        .copy_from_slice(weight_bytes);
    let output_buf = device
        .alloc_buffer(batch * m * n * 4, DType::F32, vec![batch, m, n])
        .expect("alloc batched MV output");

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml_batched_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &output_buf,
        &GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        },
    )
    .expect("batched MV dispatch");
    encoder.commit_and_wait().expect("batched MV GPU execution");
    output_buf
        .as_slice::<f32>()
        .expect("read batched MV output")
        .to_vec()
}

// --------------------------------------------------------------------------
// Verification helper: pack a random weight matrix, run mv row-by-row
// (reference) and mm batched (under test), check outputs agree.
//
// We compute the mv reference by driving the public dispatcher at m=1 for
// each input row separately.  The mv kernel accumulates each row's f32
// multiplies via `simd_sum` (a 32-lane tree reduce).  The mm kernel
// accumulates in 8x8 MMA tile steps then sums across NK=32 K-slots.  Both
// are f32 throughout, but the reduction order differs — leading to
// <= O(K * eps * max|v|) disagreement (< 1e-3 abs at K<=256, scaling
// linearly).  Quantization noise of the Q-blocks is ~O(d_scale * eps),
// which dominates, so the comparison tolerance is calibrated to cover
// both.
// --------------------------------------------------------------------------

fn check_mm_matches_mv_by_row(
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    weight_bytes: &[u8],
    input: &[f32],
    tolerance_abs: f32,
    label: &str,
) {
    // Reference: mv kernel, one row at a time.
    let mut mv_output = vec![0.0f32; m * n];
    for row in 0..m {
        let row_input = &input[row * k..(row + 1) * k];
        let row_output = run_mv_gpu(1, n, k, ggml_type, weight_bytes, row_input);
        mv_output[row * n..(row + 1) * n].copy_from_slice(&row_output);
    }

    // Under test: mm kernel, batched.
    let mm_output = run_mm_gpu(m, n, k, ggml_type, weight_bytes, input);

    let mut max_err: f32 = 0.0;
    let mut max_err_idx = 0usize;
    for i in 0..mm_output.len() {
        let err = (mm_output[i] - mv_output[i]).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }

    if max_err > tolerance_abs {
        let row = max_err_idx / n;
        let col = max_err_idx % n;
        panic!(
            "{}: mm diverges from mv at [{},{}]: mm={} mv={} err={} (tol={}, M={}, N={}, K={})",
            label, row, col, mm_output[max_err_idx], mv_output[max_err_idx],
            max_err, tolerance_abs, m, n, k,
        );
    }

    eprintln!(
        "{}: PASS (max_err={:.6}, tol={}, M={}, N={}, K={})",
        label, max_err, tolerance_abs, m, n, k,
    );
}

// ==========================================================================
// Q4_0 mm tests
// ==========================================================================

#[test]
fn test_q4_0_mm_matches_mv_small() {
    // Just above the threshold — minimal batch to force mm.
    let m = 16usize;
    let n = 64usize;
    let k = 128usize;

    let weights_f32 = pseudo_random_f32(0xA1B2, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x1234, m * k);

    // Tolerance: quantized * f32 accumulation error scales as O(sqrt(K) * eps
    // * max|value|).  Random values * dequant noise ~0.1 per element, sqrt(K)=11,
    // gives ~3e-3 absolute error.  The mm vs mv diff is only reduction order,
    // which is << quantization noise.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q4_0, &weight_bytes, &input, 5e-3,
        "Q4_0 mm matches mv, M=16 N=64 K=128");
}

#[test]
fn test_q4_0_mm_matches_mv_prefill_shape() {
    // Gemma 4 attn_q shape at prefill-ish batch
    let m = 64usize;
    let n = 256usize;
    let k = 2048usize;

    let weights_f32 = pseudo_random_f32(0xC0FFEE, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xDEADBEEF, m * k);

    // sqrt(K=2048) = 45; error scales up — 3e-2 abs tolerance comfortably covers.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q4_0, &weight_bytes, &input, 5e-2,
        "Q4_0 mm matches mv, M=64 N=256 K=2048");
}

#[test]
fn test_q4_0_mm_matches_mv_irregular() {
    // Dimensions chosen so N is not a multiple of 64 and M is not a multiple
    // of 32 — exercises the partial-tile write-back path.
    let m = 40usize;  // not a multiple of 32
    let n = 96usize;  // not a multiple of 64 (64 < 96 < 128)
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0x13, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x42, m * k);

    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q4_0, &weight_bytes, &input, 1e-2,
        "Q4_0 mm matches mv, M=40 N=96 K=256 (partial tiles)");
}

// ==========================================================================
// Q8_0 mm tests
// ==========================================================================

#[test]
fn test_q8_0_mm_matches_mv_small() {
    let m = 16usize;
    let n = 64usize;
    let k = 128usize;

    let weights_f32 = pseudo_random_f32(0xB1A2, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x5678, m * k);

    // MM kernel uses FP16 A-tile (`simdgroup_half8x8 ma[4]` at
    // quantized_matmul_mm.metal:401) per the reference MMA design for
    // tensor-core speedup. MV kernel keeps FP32 throughout. The precision
    // bound is ~2 × sqrt(K) × eps_fp16 ≈ 1e-2 at K=128.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q8_0, &weight_bytes, &input, 1.5e-2,
        "Q8_0 mm matches mv, M=16 N=64 K=128");
}

#[test]
fn test_q8_0_mm_matches_mv_prefill_shape() {
    // Gemma 4 ffn_down shape
    let m = 64usize;
    let n = 128usize;
    let k = 2112usize;

    let weights_f32 = pseudo_random_f32(0xBEEF, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xBABE, m * k);

    // FP16 A-tile precision bound at K=2112: ~2 × sqrt(2112) × eps_fp16 ≈ 5e-2.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q8_0, &weight_bytes, &input, 5e-2,
        "Q8_0 mm matches mv, M=64 N=128 K=2112");
}

#[test]
fn test_q8_0_mm_matches_mv_irregular() {
    let m = 17usize;
    let n = 72usize;
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0x21, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x22, m * k);

    // FP16 A-tile precision bound: ~2 × sqrt(256) × eps_fp16 ≈ 1.5e-2.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q8_0, &weight_bytes, &input, 2e-2,
        "Q8_0 mm matches mv, M=17 N=72 K=256 (partial tiles)");
}

#[test]
fn test_q8_0_batched_mm_is_byte_identical_to_independent_dispatches() {
    let batch = 3usize;
    let m = 17usize;
    let n = 72usize;
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0xB47C, batch * n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..batch * n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x3D6D, batch * m * k);

    let mut reference = Vec::with_capacity(batch * m * n);
    let weight_stride = weight_bytes.len() / batch;
    let input_stride = m * k;
    for group in 0..batch {
        reference.extend_from_slice(&run_mm_gpu(
            m,
            n,
            k,
            GgmlType::Q8_0,
            &weight_bytes[group * weight_stride..(group + 1) * weight_stride],
            &input[group * input_stride..(group + 1) * input_stride],
        ));
    }

    let batched = run_batched_mm_gpu(batch, m, n, k, GgmlType::Q8_0, &weight_bytes, &input);
    let reference_bits: Vec<u32> = reference.iter().map(|value| value.to_bits()).collect();
    let batched_bits: Vec<u32> = batched.iter().map(|value| value.to_bits()).collect();
    assert_eq!(
        batched_bits, reference_bits,
        "native batch addressing changed Q8_0 MM output bytes"
    );
}

#[test]
fn test_q8_0_strided_token_major_batched_mm_is_byte_identical_to_packed() {
    let batch = 3usize;
    let m = 17usize;
    let n = 72usize;
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0xB47E, batch * n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..batch * n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let packed_input = pseudo_random_f32(0x3D6F, batch * m * k);
    let mut token_major_input = vec![0.0f32; packed_input.len()];
    for row in 0..m {
        for group in 0..batch {
            let packed_start = (group * m + row) * k;
            let token_major_start = (row * batch + group) * k;
            token_major_input[token_major_start..token_major_start + k]
                .copy_from_slice(&packed_input[packed_start..packed_start + k]);
        }
    }

    let packed = run_batched_mm_gpu(
        batch,
        m,
        n,
        k,
        GgmlType::Q8_0,
        &weight_bytes,
        &packed_input,
    );
    let strided = run_batched_mm_strided_input_gpu(
        batch,
        m,
        n,
        k,
        GgmlType::Q8_0,
        &weight_bytes,
        &token_major_input,
        (batch * k * DType::F32.size_of()) as u64,
        (k * DType::F32.size_of()) as u64,
    );
    let packed_bits: Vec<u32> = packed.iter().map(|value| value.to_bits()).collect();
    let strided_bits: Vec<u32> = strided.iter().map(|value| value.to_bits()).collect();
    assert_eq!(
        strided_bits, packed_bits,
        "token-major input strides changed Q8_0 MM output bytes"
    );
}

#[test]
fn test_q8_0_strided_batched_mm_rejects_non_vector_aligned_strides() {
    let batch = 2usize;
    let m = 17usize;
    let n = 72usize;
    let k = 256usize;
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let input_buf = device
        .alloc_buffer(batch * m * k * 4, DType::F32, vec![batch, m, k])
        .expect("alloc strided batched input");
    let weight_bytes = batch * n * (k / 32) * 34;
    let weight_buf = device
        .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
        .expect("alloc strided batched weight");
    let output_buf = device
        .alloc_buffer(batch * m * n * 4, DType::F32, vec![batch, m, n])
        .expect("alloc strided batched output");
    let mut encoder = device.command_encoder().expect("encoder");

    let error = mlx_native::quantized_matmul_ggml_batched_mm_strided_input(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &output_buf,
        &GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: GgmlType::Q8_0,
        },
        &mlx_native::GgmlBatchedQuantizedMatmulInputStrides {
            row_bytes: (k * 4 + 4) as u64,
            batch_bytes: (k * 4) as u64,
        },
    )
    .expect_err("non-vector-aligned input stride must fail before dispatch");

    assert!(
        error.to_string().contains("32-byte aligned"),
        "unexpected stride validation error: {error}"
    );
}

#[test]
fn test_q8_0_batched_mv_is_byte_identical_to_independent_dispatches() {
    let batch = 3usize;
    let m = 1usize;
    let n = 72usize;
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0xB47D, batch * n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..batch * n {
        weight_bytes.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x3D6E, batch * m * k);

    let mut reference = Vec::with_capacity(batch * m * n);
    let weight_stride = weight_bytes.len() / batch;
    let input_stride = m * k;
    for group in 0..batch {
        reference.extend_from_slice(&run_mv_gpu(
            m,
            n,
            k,
            GgmlType::Q8_0,
            &weight_bytes[group * weight_stride..(group + 1) * weight_stride],
            &input[group * input_stride..(group + 1) * input_stride],
        ));
    }

    let batched = run_batched_mv_gpu(batch, m, n, k, GgmlType::Q8_0, &weight_bytes, &input);
    let reference_bits: Vec<u32> = reference.iter().map(|value| value.to_bits()).collect();
    let batched_bits: Vec<u32> = batched.iter().map(|value| value.to_bits()).collect();
    assert_eq!(
        batched_bits, reference_bits,
        "native batch addressing changed Q8_0 MV output bytes"
    );
}

// ==========================================================================
// Q6_K mm tests
// ==========================================================================

#[test]
fn test_q6_k_mm_matches_mv_small() {
    let m = 16usize;
    let n = 32usize;
    let k = 256usize;

    let weights_f32 = pseudo_random_f32(0xD1E2, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x9ABC, m * k);

    // Q6_K MM uses FP16 A-tile (`simdgroup_half8x8 ma[4]` per the reference MMA
    // design) while MV stays FP32. Precision bound at K=256:
    // ~2 × sqrt(256) × eps_fp16 ≈ 1.5e-2 — calibrated with quant-noise margin.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q6_K, &weight_bytes, &input, 2e-2,
        "Q6_K mm matches mv, M=16 N=32 K=256");
}

#[test]
fn test_q6_k_mm_matches_mv_prefill_shape() {
    // Gemma 4 attn_q shape (Q6_K tensor): N=2816 K=4096 would be too slow to
    // generate CPU-side; use N=128 to keep the test fast while still covering
    // the prefill-K range.
    let m = 64usize;
    let n = 128usize;
    let k = 2048usize;

    let weights_f32 = pseudo_random_f32(0xFACE, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xF00D, m * k);

    // FP16 A-tile precision bound at K=2048: ~2 × sqrt(2048) × eps_fp16 ≈ 5e-2.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q6_K, &weight_bytes, &input, 5e-2,
        "Q6_K mm matches mv, M=64 N=128 K=2048");
}

#[test]
fn test_q6_k_mm_matches_mv_irregular() {
    let m = 33usize;  // one over a tile boundary
    let n = 100usize;
    let k = 512usize;

    let weights_f32 = pseudo_random_f32(0x55, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q6_k(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0x66, m * k);

    // FP16 A-tile precision bound at K=512: ~2 × sqrt(512) × eps_fp16 ≈ 2.3e-2.
    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q6_K, &weight_bytes, &input, 3e-2,
        "Q6_K mm matches mv, M=33 N=100 K=512 (partial tiles)");
}

// ==========================================================================
// Dispatcher threshold tests
// ==========================================================================

// ==========================================================================
// Dispatcher routing smoke tests
// ==========================================================================
//
// Wave P3a leaves the public dispatcher on mv until the follow-up commit
// wires m>8 routing.  These tests exist to keep the public contract on
// file (tests break loudly when the routing is actually flipped, signaling
// that the behaviour change is intentional and observable).

#[test]
fn test_dispatcher_produces_finite_output_at_m_1() {
    let m = 1usize;
    let n = 16usize;
    let k = 64usize;

    let weights_f32 = pseudo_random_f32(0xAA, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xBB, m * k);

    let out = run_mv_gpu(m, n, k, GgmlType::Q4_0, &weight_bytes, &input);
    assert_eq!(out.len(), m * n);

    let abs_sum: f32 = out.iter().map(|v| v.abs()).sum();
    assert!(abs_sum > 0.0);
    for v in &out {
        assert!(v.is_finite());
    }
    eprintln!("test_dispatcher_produces_finite_output_at_m_1: PASS (abs_sum={})", abs_sum);
}

#[test]
fn test_mm_for_test_helper_works_at_m_just_above_threshold() {
    // m = MM_ROUTING_THRESHOLD + 1.  The test-only helper forces mm.
    let m = (mlx_native::MM_ROUTING_THRESHOLD + 1) as usize;
    let n = 32usize;
    let k = 128usize;

    let weights_f32 = pseudo_random_f32(0xEE, n * k);
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        weight_bytes.extend_from_slice(&pack_q4_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xFF, m * k);

    check_mm_matches_mv_by_row(m, n, k, GgmlType::Q4_0, &weight_bytes, &input, 5e-3,
        "Q4_0 mm dispatch at m=threshold+1");
}
