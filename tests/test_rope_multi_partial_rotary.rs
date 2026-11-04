//! Partial-rotary pairing contract for `rope_multi` (MROPE / IMROPE).
//!
//! The reference implementation (its CPU op and Metal kernel) rotates the
//! first `n_dims = rope_dim` dims of each row as NeoX pairs
//! `(x[ic], x[ic + n_dims/2])`, `ic < n_dims/2`, with
//! `theta = pos[axis(ic)] * freq_base^(-2*ic/n_dims)`, and copies dims
//! `[n_dims, head_dim)` unchanged. Before this fix `rope_multi` paired
//! `(x[p], x[p + head_dim/2])`, which is only equal when
//! `rope_dim == head_dim`. Qwen3.5 / Qwen3.6 full-attention layers use
//! `head_dim = 256`, `rope_dim = 64`, IMROPE sections `[11, 11, 10, 0]`,
//! `freq_base = 1e7`.
//!
//! Tests:
//! 1. F64 CPU reference with the reference implementation's semantics vs the GPU kernel at the
//!    Qwen3.6 shape (F32 and BF16, IMROPE and MROPE, text and distinct-axis
//!    positions, positions 0 .. 262143). The unrotated tail is bit-exact and
//!    every output element is written.
//! 2. Regression: the pre-fix kernel (`tests/fixtures/
//!    rope_multi_pre_partial_rotary_fix.metal`) fails the same parity check
//!    and matches the old `(p, p + head_dim/2)` pairing instead.
//! 3. Full rotary (`rope_dim == head_dim`, incl. VISION): the fixed kernel is
//!    bitwise identical to the pre-fix kernel.
//!
//! Run on both library paths: default (precompiled metallib) and
//! `MLX_PRECOMPILED_METALLIB=0` (runtime source compile).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use half::bf16;
use metal::MTLSize;
use mlx_native::ops::rope_multi::{
    build_rope_multi_buffers, dispatch_rope_multi, RopeMultiMode, RopeMultiParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

static PRE_FIX_SOURCE: &str = include_str!("fixtures/rope_multi_pre_partial_rotary_fix.metal");
const PRE_FIX_F32: &str = "rope_multi_prefix_f32";
const PRE_FIX_BF16: &str = "rope_multi_prefix_bf16";

/// Sentinel written to the output before dispatch to prove full coverage.
const SENTINEL_F32: u32 = 0x7fc0_dead;
const SENTINEL_BF16: u16 = 0x7fc1;

/// Qwen3.6 full-attention positions to probe: identity, small, mid, and the
/// largest 256K-context positions.
const QWEN36_POSITIONS: [i32; 8] = [0, 1, 2, 31, 1000, 4095, 131_071, 262_143];

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    registry.register_source(PRE_FIX_F32, PRE_FIX_SOURCE);
    registry.register_source(PRE_FIX_BF16, PRE_FIX_SOURCE);
    (device, registry)
}

fn qwen36(n_heads: u32, seq_len: u32, mode: RopeMultiMode) -> RopeMultiParams {
    RopeMultiParams {
        head_dim: 256,
        rope_dim: 64,
        n_heads,
        seq_len,
        freq_base: 1e7,
        mode,
        sections: [11, 11, 10, 0],
    }
}

fn lcg_input(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (seed as i32 as f32) / (i32::MAX as f32) * 2.0
        })
        .collect()
}

/// Positions buffer `[t; h; w; e]`, each of length `seq_len`.
fn positions_for(token_pos: &[i32], distinct_axes: bool) -> Vec<i32> {
    let mut out = Vec::with_capacity(4 * token_pos.len());
    for axis in 0..4 {
        for &p in token_pos {
            out.push(if !distinct_axes {
                p
            } else {
                match axis {
                    0 => p,
                    1 => p / 3 + 5,
                    2 => p / 7 + 11,
                    _ => 0,
                }
            });
        }
    }
    out
}

// -----------------------------------------------------------------
// F64 reference (the reference implementation's semantics)
// -----------------------------------------------------------------

fn axis_for(ic: u32, mode: RopeMultiMode, s: [u32; 4]) -> u32 {
    let sect_dims = s.iter().sum::<u32>().max(1);
    let sector = ic % sect_dims;
    match mode {
        RopeMultiMode::Imrope => {
            if sector % 3 == 1 && sector < 3 * s[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * s[2] {
                2
            } else if sector % 3 == 0 && sector < 3 * s[0] {
                0
            } else {
                3
            }
        }
        RopeMultiMode::Mrope => {
            if sector < s[0] {
                0
            } else if sector < s[0] + s[1] {
                1
            } else if sector < s[0] + s[1] + s[2] {
                2
            } else {
                3
            }
        }
        RopeMultiMode::Vision => panic!("vision is covered by the full-rotary bitwise test"),
    }
}

/// F64 multi-section NeoX rope. `pair_offset` is the partner distance:
/// `rope_dim / 2` is the reference; `head_dim / 2` reproduces the pre-fix
/// kernel. Dims not touched by a rotated pair are copied.
fn ref_rope_multi_f64(
    input: &[f32],
    positions: &[i32],
    p: &RopeMultiParams,
    pair_offset: usize,
) -> Vec<f64> {
    let head_dim = p.head_dim as usize;
    let n_dims = p.rope_dim as usize;
    let n_rows = (p.seq_len * p.n_heads) as usize;
    let mut out: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    for row in 0..n_rows {
        let base = row * head_dim;
        let seq_idx = row / p.n_heads as usize;
        for i0 in (0..n_dims).step_by(2) {
            let ic = i0 / 2;
            let axis = axis_for(ic as u32, p.mode, p.sections) as usize;
            let pos = f64::from(positions[axis * p.seq_len as usize + seq_idx]);
            let theta = pos * f64::from(p.freq_base).powf(-(i0 as f64) / n_dims as f64);
            let (s, c) = theta.sin_cos();
            let x0 = f64::from(input[base + ic]);
            let x1 = f64::from(input[base + ic + pair_offset]);
            out[base + ic] = x0 * c - x1 * s;
            out[base + ic + pair_offset] = x0 * s + x1 * c;
        }
    }
    out
}

/// F32 kernel tolerance. The kernel evaluates `pow`/`cos`/`sin` in f32, so
/// the angle error grows with the position (pair 0 has frequency 1, so
/// `|theta| <= pos`).
fn f32_tol(max_pos: i32) -> f64 {
    2e-5 + 1e-7 * f64::from(max_pos)
}

// -----------------------------------------------------------------
// GPU helpers
// -----------------------------------------------------------------

fn upload_i32(device: &MlxDevice, data: &[i32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::I32, vec![data.len()])
        .expect("alloc i32");
    buf.as_mut_slice::<i32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    /// Production path: `dispatch_rope_multi` (fixed shader).
    Fixed,
    /// Pre-fix shader fixture, runtime-compiled, same grid as the old dispatcher.
    PreFix,
}

fn dispatch(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kernel: Kernel,
    input: &MlxBuffer,
    output: &MlxBuffer,
    positions: &[i32],
    p: RopeMultiParams,
) {
    let positions_buf = upload_i32(device, positions);
    let (params_buf, rope_params_buf, sections_buf) =
        build_rope_multi_buffers(device, p).expect("build bufs");
    let mut enc = device.command_encoder().expect("enc");
    match kernel {
        Kernel::Fixed => dispatch_rope_multi(
            &mut enc,
            registry,
            device.metal_device(),
            input,
            output,
            &positions_buf,
            &params_buf,
            &rope_params_buf,
            &sections_buf,
            p,
        )
        .expect("dispatch"),
        Kernel::PreFix => {
            let name = match input.dtype() {
                DType::F32 => PRE_FIX_F32,
                DType::BF16 => PRE_FIX_BF16,
                other => panic!("unsupported dtype {other}"),
            };
            let pipeline = registry
                .get_pipeline(name, device.metal_device())
                .expect("pre-fix pipeline");
            let half_dim = p.head_dim / 2;
            let n_rows = p.seq_len * p.n_heads;
            let tg_x = half_dim.min(256).max(1);
            let tg_y = n_rows.min((256 / tg_x).max(1)).max(1);
            enc.encode(
                pipeline,
                &[
                    (0, input),
                    (1, output),
                    (2, &params_buf),
                    (3, &positions_buf),
                    (4, &rope_params_buf),
                    (5, &sections_buf),
                ],
                MTLSize::new(u64::from(half_dim), u64::from(n_rows), 1),
                MTLSize::new(u64::from(tg_x), u64::from(tg_y), 1),
            );
        }
    }
    enc.commit_and_wait().expect("commit");
}

fn run_f32(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kernel: Kernel,
    input: &[f32],
    positions: &[i32],
    p: RopeMultiParams,
) -> Vec<f32> {
    let mut in_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("alloc in");
    in_buf
        .as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(input);
    let mut out_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("alloc out");
    out_buf
        .as_mut_slice::<u32>()
        .expect("mut")
        .fill(SENTINEL_F32);
    dispatch(device, registry, kernel, &in_buf, &out_buf, positions, p);
    out_buf.as_slice::<f32>().expect("read").to_vec()
}

fn run_bf16(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kernel: Kernel,
    input: &[bf16],
    positions: &[i32],
    p: RopeMultiParams,
) -> Vec<bf16> {
    let mut in_buf = device
        .alloc_buffer(input.len() * 2, DType::BF16, vec![input.len()])
        .expect("alloc in");
    in_buf
        .as_mut_slice::<bf16>()
        .expect("mut")
        .copy_from_slice(input);
    let mut out_buf = device
        .alloc_buffer(input.len() * 2, DType::BF16, vec![input.len()])
        .expect("alloc out");
    out_buf
        .as_mut_slice::<u16>()
        .expect("mut")
        .fill(SENTINEL_BF16);
    dispatch(device, registry, kernel, &in_buf, &out_buf, positions, p);
    out_buf.as_slice::<bf16>().expect("read").to_vec()
}

fn max_abs_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    got.iter()
        .zip(want)
        .enumerate()
        .map(|(i, (&g, &w))| ((f64::from(g) - w).abs(), i))
        .fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a })
}

fn assert_tail_bit_exact_f32(got: &[f32], input: &[f32], p: &RopeMultiParams, label: &str) {
    let hd = p.head_dim as usize;
    for (row, (g, x)) in got.chunks(hd).zip(input.chunks(hd)).enumerate() {
        for d in p.rope_dim as usize..hd {
            assert_eq!(
                g[d].to_bits(),
                x[d].to_bits(),
                "{label}: tail dim {d} of row {row} changed"
            );
        }
    }
}

// -----------------------------------------------------------------
// 1. Qwen3.6-shape parity vs the F64 reference
// -----------------------------------------------------------------

fn qwen36_f32_case(mode: RopeMultiMode, distinct_axes: bool, n_heads: u32) {
    let (device, mut registry) = setup();
    let p = qwen36(n_heads, QWEN36_POSITIONS.len() as u32, mode);
    let n = (p.seq_len * p.n_heads * p.head_dim) as usize;
    let input = lcg_input(n, 0x36_0256 ^ n_heads);
    let positions = positions_for(&QWEN36_POSITIONS, distinct_axes);

    let got = run_f32(&device, &mut registry, Kernel::Fixed, &input, &positions, p);
    assert!(
        got.iter().all(|v| v.to_bits() != SENTINEL_F32),
        "kernel left output elements unwritten"
    );
    assert_tail_bit_exact_f32(&got, &input, &p, "qwen36 f32");

    let want = ref_rope_multi_f64(&input, &positions, &p, p.rope_dim as usize / 2);
    let row_len = (p.n_heads * p.head_dim) as usize;
    for (t, &pos) in QWEN36_POSITIONS.iter().enumerate() {
        let span = t * row_len..(t + 1) * row_len;
        let (err, at) = max_abs_err(&got[span.clone()], &want[span]);
        let tol = f32_tol(pos);
        println!("{mode:?} distinct={distinct_axes} heads={n_heads} pos={pos}: max_abs_err={err:.3e} (tol {tol:.3e}) at {at}");
        assert!(
            err <= tol,
            "{mode:?} pos={pos}: max_abs_err {err:e} > tol {tol:e} at {at}"
        );
        if pos == 0 && !distinct_axes {
            // Identity rotation: theta = 0 on every axis for every pair.
            assert!(err == 0.0, "pos 0 must be the identity, err {err:e}");
        }
    }
}

#[test]
fn qwen36_imrope_text_f32_matches_f64_reference() {
    qwen36_f32_case(RopeMultiMode::Imrope, false, 16); // Q heads
    qwen36_f32_case(RopeMultiMode::Imrope, false, 2); // KV heads
}

#[test]
fn qwen36_imrope_distinct_axes_f32_matches_f64_reference() {
    qwen36_f32_case(RopeMultiMode::Imrope, true, 4);
}

#[test]
fn qwen36_mrope_distinct_axes_f32_matches_f64_reference() {
    qwen36_f32_case(RopeMultiMode::Mrope, true, 4);
}

#[test]
fn qwen36_imrope_bf16_matches_f64_reference() {
    let (device, mut registry) = setup();
    let p = qwen36(4, QWEN36_POSITIONS.len() as u32, RopeMultiMode::Imrope);
    let n = (p.seq_len * p.n_heads * p.head_dim) as usize;
    let input_bf: Vec<bf16> = lcg_input(n, 0xbf16)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let input_f32: Vec<f32> = input_bf.iter().map(|v| v.to_f32()).collect();
    let positions = positions_for(&QWEN36_POSITIONS, true);

    let got = run_bf16(
        &device,
        &mut registry,
        Kernel::Fixed,
        &input_bf,
        &positions,
        p,
    );
    assert!(
        got.iter().all(|v| v.to_bits() != SENTINEL_BF16),
        "kernel left output elements unwritten"
    );
    let hd = p.head_dim as usize;
    for (row, (g, x)) in got.chunks(hd).zip(input_bf.chunks(hd)).enumerate() {
        for d in p.rope_dim as usize..hd {
            assert_eq!(
                g[d].to_bits(),
                x[d].to_bits(),
                "bf16 tail dim {d} of row {row} changed"
            );
        }
    }

    let want = ref_rope_multi_f64(&input_f32, &positions, &p, p.rope_dim as usize / 2);
    let row_len = (p.n_heads * p.head_dim) as usize;
    for (t, &pos) in QWEN36_POSITIONS.iter().enumerate() {
        let mut err = 0.0f64;
        for i in t * row_len..(t + 1) * row_len {
            // One bf16 rounding of a value bounded by |x0| + |x1| <= 4.
            let e = (f64::from(got[i].to_f32()) - want[i]).abs();
            err = err.max(e - want[i].abs() * 2f64.powi(-8));
        }
        let tol = f32_tol(pos);
        println!("bf16 pos={pos}: max excess err={err:.3e} (tol {tol:.3e})");
        assert!(
            err <= tol,
            "bf16 pos={pos}: excess err {err:e} > tol {tol:e}"
        );
    }
}

// -----------------------------------------------------------------
// 2. Regression: the pre-fix pairing fails the parity check
// -----------------------------------------------------------------

#[test]
fn qwen36_pre_fix_pairing_fails_parity_and_fixed_passes() {
    let (device, mut registry) = setup();
    let token_pos = [1, 1000, 262_143];
    let p = qwen36(4, token_pos.len() as u32, RopeMultiMode::Imrope);
    let n = (p.seq_len * p.n_heads * p.head_dim) as usize;
    let input = lcg_input(n, 0x0bad_0128);
    let positions = positions_for(&token_pos, false);
    let tol = f32_tol(*token_pos.iter().max().unwrap());

    let reference = ref_rope_multi_f64(&input, &positions, &p, p.rope_dim as usize / 2);
    let old_pairing = ref_rope_multi_f64(&input, &positions, &p, p.head_dim as usize / 2);

    let fixed = run_f32(&device, &mut registry, Kernel::Fixed, &input, &positions, p);
    let pre_fix = run_f32(
        &device,
        &mut registry,
        Kernel::PreFix,
        &input,
        &positions,
        p,
    );

    let (fixed_err, _) = max_abs_err(&fixed, &reference);
    let (pre_err, pre_at) = max_abs_err(&pre_fix, &reference);
    let (pre_vs_old, _) = max_abs_err(&pre_fix, &old_pairing);
    println!("fixed vs ref {fixed_err:.3e}; pre-fix vs ref {pre_err:.3e} at {pre_at}; pre-fix vs old pairing {pre_vs_old:.3e}");

    assert!(fixed_err <= tol, "fixed kernel: {fixed_err:e} > {tol:e}");
    assert!(
        pre_err > 0.1,
        "the parity check must reject the pre-fix pairing (err {pre_err:e})"
    );
    assert!(
        pre_vs_old <= tol,
        "the pre-fix kernel should match the (p, p + head_dim/2) pairing (err {pre_vs_old:e})"
    );
    // The pre-fix kernel rotated dims 128..160 and left 32..64 untouched.
    let hd = p.head_dim as usize;
    let row1 = &pre_fix[hd..2 * hd]; // token pos 1, head 0
    assert_eq!(row1[32].to_bits(), input[hd + 32].to_bits());
    assert_ne!(row1[128].to_bits(), input[hd + 128].to_bits());
    let fixed_row1 = &fixed[hd..2 * hd];
    assert_ne!(fixed_row1[32].to_bits(), input[hd + 32].to_bits());
    assert_eq!(fixed_row1[128].to_bits(), input[hd + 128].to_bits());
}

// -----------------------------------------------------------------
// 3. Full rotary: bitwise identical to the pre-fix kernel
// -----------------------------------------------------------------

fn full_rotary_cases() -> Vec<RopeMultiParams> {
    let base = |head_dim, mode, sections, freq_base| RopeMultiParams {
        head_dim,
        rope_dim: head_dim,
        n_heads: 3,
        seq_len: QWEN36_POSITIONS.len() as u32,
        freq_base,
        mode,
        sections,
    };
    vec![
        base(128, RopeMultiMode::Imrope, [24, 20, 20, 0], 5e6),
        base(128, RopeMultiMode::Mrope, [16, 24, 24, 0], 1e6),
        base(256, RopeMultiMode::Imrope, [44, 42, 42, 0], 1e7),
        base(64, RopeMultiMode::Imrope, [11, 11, 10, 0], 1e7),
        base(64, RopeMultiMode::Vision, [16, 16, 0, 0], 1e4),
        base(72, RopeMultiMode::Vision, [18, 18, 0, 0], 1e4),
    ]
}

#[test]
fn full_rotary_f32_bitwise_equals_pre_fix_kernel() {
    let (device, mut registry) = setup();
    for (k, p) in full_rotary_cases().into_iter().enumerate() {
        let n = (p.seq_len * p.n_heads * p.head_dim) as usize;
        let input = lcg_input(n, 0xf011 + k as u32);
        for distinct in [false, true] {
            let positions = positions_for(&QWEN36_POSITIONS, distinct);
            let fixed = run_f32(&device, &mut registry, Kernel::Fixed, &input, &positions, p);
            let pre = run_f32(
                &device,
                &mut registry,
                Kernel::PreFix,
                &input,
                &positions,
                p,
            );
            for (i, (a, b)) in fixed.iter().zip(&pre).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{p:?} distinct={distinct}: element {i} differs ({a} vs {b})"
                );
            }
        }
    }
    let origin = registry
        .pipeline_identity("rope_multi_f32")
        .expect("identity")
        .origin;
    println!("rope_multi_f32 origin: {origin:?}");
}

#[test]
fn full_rotary_bf16_bitwise_equals_pre_fix_kernel() {
    let (device, mut registry) = setup();
    for (k, p) in full_rotary_cases().into_iter().enumerate() {
        let n = (p.seq_len * p.n_heads * p.head_dim) as usize;
        let input: Vec<bf16> = lcg_input(n, 0xb0f1 + k as u32)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let positions = positions_for(&QWEN36_POSITIONS, true);
        let fixed = run_bf16(&device, &mut registry, Kernel::Fixed, &input, &positions, p);
        let pre = run_bf16(
            &device,
            &mut registry,
            Kernel::PreFix,
            &input,
            &positions,
            p,
        );
        for (i, (a, b)) in fixed.iter().zip(&pre).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{p:?}: element {i} differs");
        }
    }
    let origin = registry
        .pipeline_identity("rope_multi_bf16")
        .expect("identity")
        .origin;
    println!("rope_multi_bf16 origin: {origin:?}");
}

#[test]
fn library_path_matches_env() {
    let (device, mut registry) = setup();
    let p = qwen36(1, 1, RopeMultiMode::Imrope);
    let input = vec![0.5f32; p.head_dim as usize];
    run_f32(
        &device,
        &mut registry,
        Kernel::Fixed,
        &input,
        &[7, 7, 7, 7],
        p,
    );
    let origin = registry
        .pipeline_identity("rope_multi_f32")
        .expect("identity")
        .origin;
    let runtime = matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    );
    println!("MLX_PRECOMPILED_METALLIB runtime={runtime}; rope_multi_f32 origin {origin:?}");
    if runtime {
        assert_eq!(origin, mlx_native::KernelPipelineOrigin::RuntimeSource);
    }
}
