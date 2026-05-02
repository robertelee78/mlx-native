//! Tests for rope_multi (MROPE / IMROPE) kernel — ADR-013 Decision 10.
//!
//! Acceptance criteria:
//! 1. Shader + op wrapper exist (covered by this file compiling).
//! 2. Spec-driven small case: n_rot=8, sections=[2,2,1,0]; hand-compute the
//!    cos/sin table per rotary index and the output for a synthetic Q tensor;
//!    assert the kernel matches within 1e-5 (F32).
//! 3. Integration: n_rot=64, sections=[11,11,10,0], freq_base=1e7 produces
//!    bit-stable output across repeated calls (determinism).
//! 4. CPU reference parity: pure-Rust scalar implementation of the spec
//!    matches the Metal kernel to 1e-5 (F32).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::rope_multi::{
    build_rope_multi_buffers, RopeMultiMode, RopeMultiParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
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

fn upload_i32(device: &MlxDevice, data: &[i32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::I32, vec![data.len()])
        .expect("alloc");
    buf.as_mut_slice::<i32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

// -----------------------------------------------------------------
// Pure-Rust scalar reference implementation (spec-driven).
// -----------------------------------------------------------------

fn pick_axis_cpu(sector: u32, mode: RopeMultiMode, s: [u32; 4]) -> u32 {
    match mode {
        RopeMultiMode::Imrope => {
            if sector % 3 == 0 && sector < 3 * s[0] {
                0
            } else if sector % 3 == 1 && sector < 3 * s[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * s[2] {
                2
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
        RopeMultiMode::Vision => {
            // Vision ignores sections 2 and 3; only y/x axes used.
            // Mirrors /opt/llama.cpp/ggml/src/ggml-cuda/rope.cu:308-313.
            if sector < s[0] {
                0
            } else {
                1
            }
        }
    }
}

/// CPU reference — returns full output vector of length `n_rows * head_dim`.
///
/// Handles all three modes. For VISION (mode 24), `sect_dims = s0 + s1`
/// (last two sections ignored), the theta exponent is the *per-section*
/// `local_p` rather than the global `pair_idx`, and the denominator is
/// `n_dims = head_dim/2` rather than `rope_dim`. Mirrors the CUDA
/// reference at /opt/llama.cpp/ggml/src/ggml-cuda/rope.cu:268-328 and the
/// `indep_sects` branch of /opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5660-5710.
fn cpu_rope_multi(
    input: &[f32],
    positions: &[i32],
    p: RopeMultiParams,
) -> Vec<f32> {
    let n_rows = (p.seq_len * p.n_heads) as usize;
    let head_dim = p.head_dim as usize;
    let half_dim = head_dim / 2;
    let rope_dim = p.rope_dim as usize;
    let half_rope = rope_dim / 2;
    let is_vision = matches!(p.mode, RopeMultiMode::Vision);
    let sect_dims = if is_vision {
        (p.sections[0] + p.sections[1]).max(1)
    } else {
        p.sections.iter().sum::<u32>().max(1)
    };

    let mut out = input.to_vec();
    for row in 0..n_rows {
        let base = row * head_dim;
        let seq_idx = (row as u32) / p.n_heads;

        for pair in 0..half_dim {
            if (pair as usize) < half_rope {
                let sector = (pair as u32) % sect_dims;
                let axis = pick_axis_cpu(sector, p.mode, p.sections);
                let pos = positions[(axis * p.seq_len + seq_idx) as usize];

                let (theta_p, denom) = if is_vision {
                    // Per-section local index; denom = n_dims = half_rope
                    // (since for vision rope_dim == head_dim, half_rope ==
                    // head_dim / 2 == n_dims).
                    let local_p = if sector < p.sections[0] {
                        sector
                    } else {
                        sector - p.sections[0]
                    };
                    (local_p, half_rope as u32)
                } else {
                    (pair as u32, p.rope_dim)
                };

                let dim_ratio = 2.0 * (theta_p as f32) / (denom as f32);
                let freq = 1.0 / (p.freq_base.powf(dim_ratio));
                let theta = (pos as f32) * freq;
                let (cos_a, sin_a) = (theta.cos(), theta.sin());

                let x0 = input[base + pair];
                let x1 = input[base + pair + half_dim];
                out[base + pair] = x0 * cos_a - x1 * sin_a;
                out[base + pair + half_dim] = x0 * sin_a + x1 * cos_a;
            } else {
                // pass through (partial rotary tail)
                out[base + pair] = input[base + pair];
                out[base + pair + half_dim] = input[base + pair + half_dim];
            }
        }
    }
    out
}

fn run_rope_multi(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input_data: &[f32],
    positions_data: &[i32],
    p: RopeMultiParams,
) -> Vec<f32> {
    let input_buf = upload_f32(device, input_data);
    let output_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .expect("output");
    let positions_buf = upload_i32(device, positions_data);
    let (params_buf, rope_params_buf, sections_buf) =
        build_rope_multi_buffers(device, p).expect("build bufs");

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc,
        registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &positions_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    output_buf.as_slice::<f32>().expect("read").to_vec()
}

// =================================================================
// Spec-driven: n_rot=8, sections=[2,2,1,0], IMROPE
// =================================================================
//
// Sect assignment (sect_dims = 5, pair_idx 0..3 rotated):
//   pair 0: sector=0, 0%3==0 && 0 < 6  -> axis 0 (t)
//   pair 1: sector=1, 1%3==1 && 1 < 6  -> axis 1 (h)
//   pair 2: sector=2, 2%3==2 && 2 < 3  -> axis 2 (w)
//   pair 3: sector=3, 3%3==0 && 3 < 6  -> axis 0 (t)
//
// For head_dim=16, rope_dim=8, n_rot/2=4 pairs rotated, the rest (pairs 4..7)
// pass-through. Positions: t=5, h=7, w=9, e=0 (unused).
//
// Per-pair theta = position[axis] * freq_base^(-2*pair/rope_dim).
// For freq_base=2, rope_dim=8:
//   pair 0: axis t, pos=5, ratio=0/8=0, freq=1/2^0=1, theta=5
//   pair 1: axis h, pos=7, ratio=2/8=0.25, freq=1/2^0.25 ≈ 0.8408964, theta ≈ 5.88627
//   pair 2: axis w, pos=9, ratio=4/8=0.5, freq=1/2^0.5 ≈ 0.70710678, theta ≈ 6.36396
//   pair 3: axis t, pos=5, ratio=6/8=0.75, freq=1/2^0.75 ≈ 0.59460356, theta ≈ 2.97302
#[test]
fn test_rope_multi_imrope_spec_driven_n8_sections_2_2_1_0() {
    let (device, mut registry) = setup();

    let p = RopeMultiParams {
        head_dim: 16,
        rope_dim: 8,
        n_heads: 1,
        seq_len: 1,
        freq_base: 2.0,
        mode: RopeMultiMode::Imrope,
        sections: [2, 2, 1, 0],
    };

    // Simple input: x[i] = (i+1)*0.1 for i in 0..16.
    let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    // positions tensor laid out as [p_t, p_h, p_w, p_e] each of length seq_len=1.
    let positions = [5i32, 7, 9, 0];

    let got = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let want = cpu_rope_multi(&input, &positions, p);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 1e-5,
            "spec-driven mismatch at {}: got {}, want {}, diff {}",
            i, g, w, d
        );
    }

    // Independently verify pair 0 manually to pin down the spec.
    // pair 0 uses axis t (pos=5), freq=1, theta=5. cos(5) ≈ 0.28366; sin(5) ≈ -0.95892.
    // x0 = 0.1, x1 = x[0 + half_dim=8] = 0.9.
    let pair0 = (got[0], got[8]);
    let expected0 = (
        0.1 * 5.0_f32.cos() - 0.9 * 5.0_f32.sin(),
        0.1 * 5.0_f32.sin() + 0.9 * 5.0_f32.cos(),
    );
    assert!((pair0.0 - expected0.0).abs() < 1e-5, "pair0.x mismatch");
    assert!((pair0.1 - expected0.1).abs() < 1e-5, "pair0.y mismatch");

    // Pair 4 (beyond half_rope=4): pass-through.
    assert!((got[4] - input[4]).abs() < 1e-7, "pass-through x0 broken");
    assert!((got[12] - input[12]).abs() < 1e-7, "pass-through x1 broken");
}

// =================================================================
// MROPE (contiguous) vs IMROPE: distinct axis assignments
// =================================================================

#[test]
fn test_rope_multi_mrope_vs_imrope_differ_for_distinct_positions() {
    let (device, mut registry) = setup();

    let p_imrope = RopeMultiParams {
        head_dim: 16,
        rope_dim: 8,
        n_heads: 1,
        seq_len: 1,
        freq_base: 2.0,
        mode: RopeMultiMode::Imrope,
        sections: [2, 2, 1, 0],
    };
    let mut p_mrope = p_imrope;
    p_mrope.mode = RopeMultiMode::Mrope;

    let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let positions = [5i32, 7, 9, 0];

    let got_imrope = run_rope_multi(&device, &mut registry, &input, &positions, p_imrope);
    let got_mrope = run_rope_multi(&device, &mut registry, &input, &positions, p_mrope);

    // Pair 1 picks a different axis between modes:
    //   IMROPE pair 1 sector=1 -> axis h (pos 7)
    //   MROPE  pair 1 sector=1 -> axis t (sections[0]=2 so sector<2 is axis 0 = t)
    // So outputs should differ at indices 1 and 9.
    let d1 = (got_imrope[1] - got_mrope[1]).abs();
    let d9 = (got_imrope[9] - got_mrope[9]).abs();
    assert!(
        d1 > 1e-3 || d9 > 1e-3,
        "MROPE and IMROPE produced identical pair 1 — axis-mapping may be wrong"
    );
}

// =================================================================
// CPU-reference parity with random inputs
// =================================================================

#[test]
fn test_rope_multi_random_cpu_parity() {
    let (device, mut registry) = setup();

    let p = RopeMultiParams {
        head_dim: 32,
        rope_dim: 16,
        n_heads: 4,
        seq_len: 6,
        freq_base: 10000.0,
        mode: RopeMultiMode::Imrope,
        sections: [3, 3, 2, 0],
    };

    let n_rows = (p.seq_len * p.n_heads) as usize;
    let n_elem = n_rows * (p.head_dim as usize);

    let mut seed = 0xdeadbeefu32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 1.5
    };
    let input: Vec<f32> = (0..n_elem).map(|_| rand()).collect();

    // Positions: 4 axes × seq_len; text-like (all same positions).
    let positions: Vec<i32> = (0..p.seq_len as i32)
        .cycle()
        .take(4 * p.seq_len as usize)
        .collect();

    let got = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let want = cpu_rope_multi(&input, &positions, p);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 1e-5,
            "random parity mismatch at {}: got {}, want {}",
            i, g, w
        );
    }
}

// =================================================================
// Qwen3.5 integration shape (ADR-013 integration test)
// =================================================================

#[test]
fn test_rope_multi_qwen35_shape_determinism() {
    let (device, mut registry) = setup();

    // Qwen3.5 full-attention layer: head_dim=256, rope_dim=64 (partial
    // rotary 0.25), freq_base=1e7, sections=[11,11,10,0], mode IMROPE.
    let p = RopeMultiParams {
        head_dim: 256,
        rope_dim: 64,
        n_heads: 4, // small for a test
        seq_len: 3,
        freq_base: 1e7,
        mode: RopeMultiMode::Imrope,
        sections: [11, 11, 10, 0],
    };

    let n_rows = (p.seq_len * p.n_heads) as usize;
    let n_elem = n_rows * (p.head_dim as usize);

    let mut seed = 0xfeed1234u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32)
    };
    let input: Vec<f32> = (0..n_elem).map(|_| rand()).collect();

    // Text positions: all 4 axes identical (token index).
    let positions: Vec<i32> = (0..p.seq_len as i32)
        .cycle()
        .take(4 * p.seq_len as usize)
        .collect();

    let got1 = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let got2 = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let got3 = run_rope_multi(&device, &mut registry, &input, &positions, p);

    // Determinism across repeated dispatches.
    for i in 0..n_elem {
        assert_eq!(
            got1[i].to_bits(),
            got2[i].to_bits(),
            "non-deterministic at {}: {} vs {}",
            i, got1[i], got2[i]
        );
        assert_eq!(got1[i].to_bits(), got3[i].to_bits(), "non-deterministic run3");
    }

    // Parity vs CPU reference.
    let want = cpu_rope_multi(&input, &positions, p);
    for (i, (&g, &w)) in got1.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 1e-4,
            "qwen35-shape parity mismatch at {}: got {}, want {}",
            i, g, w
        );
    }

    // Partial-rotary tail (pairs 32..127): must pass through unchanged.
    for row in 0..n_rows {
        let base = row * (p.head_dim as usize);
        for pair in 32..128 {
            let ix0 = base + pair;
            let ix1 = base + pair + (p.head_dim as usize / 2);
            assert!(
                (got1[ix0] - input[ix0]).abs() < 1e-7,
                "partial-rotary tail modified at row={}, pair={}",
                row, pair
            );
            assert!(
                (got1[ix1] - input[ix1]).abs() < 1e-7,
                "partial-rotary tail modified at row={}, pair+half",
                row
            );
        }
    }
}

// =================================================================
// Text-only degeneracy: IMROPE with identical axes == NeoX RoPE
// =================================================================

/// For text-only Qwen3.5 all 4 position axes equal the token index, so
/// IMROPE's output MUST equal plain NeoX RoPE's output. This is a smoke
/// test that our sector-cycling doesn't introduce spurious axis picks when
/// the axes don't differ.
#[test]
fn test_rope_multi_imrope_text_equals_neox_rope() {
    let (device, mut registry) = setup();
    let p = RopeMultiParams {
        head_dim: 32,
        rope_dim: 16,
        n_heads: 2,
        seq_len: 4,
        freq_base: 1e4,
        mode: RopeMultiMode::Imrope,
        sections: [3, 3, 2, 0],
    };

    let n_rows = (p.seq_len * p.n_heads) as usize;
    let n_elem = n_rows * (p.head_dim as usize);
    let input: Vec<f32> = (0..n_elem).map(|i| (i as f32) * 0.01).collect();

    // All 4 axes = token index.
    let positions: Vec<i32> = (0..p.seq_len as i32)
        .cycle()
        .take(4 * p.seq_len as usize)
        .collect();

    let got = run_rope_multi(&device, &mut registry, &input, &positions, p);

    // Reference: plain NeoX RoPE with the same per-pair frequency.
    let mut want = input.clone();
    let half_dim = (p.head_dim / 2) as usize;
    let half_rope = (p.rope_dim / 2) as usize;
    for row in 0..n_rows {
        let base = row * p.head_dim as usize;
        let seq_idx = row as u32 / p.n_heads;
        let pos = seq_idx as f32;
        for pair in 0..half_rope {
            let dim_ratio = 2.0 * pair as f32 / p.rope_dim as f32;
            let freq = 1.0 / p.freq_base.powf(dim_ratio);
            let theta = pos * freq;
            let (ca, sa) = (theta.cos(), theta.sin());
            let x0 = input[base + pair];
            let x1 = input[base + pair + half_dim];
            want[base + pair] = x0 * ca - x1 * sa;
            want[base + pair + half_dim] = x0 * sa + x1 * ca;
        }
    }

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 1e-5,
            "text IMROPE != NeoX at {}: got {}, want {}",
            i, g, w
        );
    }
}

// =================================================================
// BF16 path
// =================================================================

#[test]
fn test_rope_multi_bf16_matches_f32_within_tolerance() {
    use half::bf16;
    let (device, mut registry) = setup();
    let p = RopeMultiParams {
        head_dim: 16,
        rope_dim: 8,
        n_heads: 2,
        seq_len: 3,
        freq_base: 1e4,
        mode: RopeMultiMode::Imrope,
        sections: [2, 2, 1, 0],
    };

    let n_rows = (p.seq_len * p.n_heads) as usize;
    let n_elem = n_rows * (p.head_dim as usize);
    let input_f32: Vec<f32> = (0..n_elem).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let positions: Vec<i32> = (0..p.seq_len as i32)
        .cycle()
        .take(4 * p.seq_len as usize)
        .collect();

    // F32 ground truth via kernel.
    let f32_out = run_rope_multi(&device, &mut registry, &input_f32, &positions, p);

    // BF16 run.
    let input_bf: Vec<bf16> = input_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut in_buf = device
        .alloc_buffer(n_elem * 2, DType::BF16, vec![n_elem])
        .expect("input bf16");
    in_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&input_bf);
    let out_buf = device
        .alloc_buffer(n_elem * 2, DType::BF16, vec![n_elem])
        .expect("output bf16");
    let positions_buf = upload_i32(&device, &positions);
    let (params_buf, rope_params_buf, sections_buf) =
        build_rope_multi_buffers(&device, p).expect("bufs");

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &in_buf,
        &out_buf,
        &positions_buf,
        &params_buf,
        &rope_params_buf,
        &sections_buf,
        p,
    )
    .expect("dispatch bf16");
    enc.commit_and_wait().expect("commit");

    let bf_out: Vec<bf16> = out_buf.as_slice::<bf16>().expect("read").to_vec();
    for (i, (bf, f)) in bf_out.iter().zip(f32_out.iter()).enumerate() {
        let diff = (bf.to_f32() - f).abs();
        assert!(
            diff < 5e-2,
            "bf16 drift at {}: bf={}, f32={}, diff={}",
            i, bf.to_f32(), f, diff
        );
    }
}

// =================================================================
// Error handling
// =================================================================

#[test]
fn test_rope_multi_rejects_odd_head_dim() {
    let (device, mut registry) = setup();
    let p = RopeMultiParams {
        head_dim: 15,
        rope_dim: 4,
        n_heads: 1,
        seq_len: 1,
        freq_base: 1e4,
        mode: RopeMultiMode::Imrope,
        sections: [1, 1, 0, 0],
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let pos = device.alloc_buffer(16, DType::I32, vec![4]).expect("p");
    let (params, rope_params, sections) = build_rope_multi_buffers(&device, p).expect("b");

    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy,
        &dummy,
        &pos,
        &params,
        &rope_params,
        &sections,
        p,
    );
    assert!(res.is_err(), "odd head_dim should error");
}

#[test]
fn test_rope_multi_rejects_rope_dim_gt_head_dim() {
    let (device, mut registry) = setup();
    let p = RopeMultiParams {
        head_dim: 8,
        rope_dim: 16,
        n_heads: 1,
        seq_len: 1,
        freq_base: 1e4,
        mode: RopeMultiMode::Imrope,
        sections: [1, 1, 0, 0],
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let pos = device.alloc_buffer(16, DType::I32, vec![4]).expect("p");
    let (params, rope_params, sections) = build_rope_multi_buffers(&device, p).expect("b");

    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy,
        &dummy,
        &pos,
        &params,
        &rope_params,
        &sections,
        p,
    );
    assert!(res.is_err(), "rope_dim > head_dim should error");
}

// =================================================================
// dispatch_rope_multi_cached parity (ADR-015 P3b rank-4)
// =================================================================
//
// The cached dispatch path reuses pre-built parameter buffers across
// calls.  This test verifies it is bit-exact to the per-call path on
// the qwen3.5 decode hot-path shape (seq_len=1, head_dim=256, rope_dim=64,
// IMROPE), and that re-issuing the same dispatch returns the same bytes
// (cache hit path).

fn run_rope_multi_cached(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input_data: &[f32],
    positions_data: &[i32],
    p: RopeMultiParams,
) -> Vec<f32> {
    let input_buf = upload_f32(device, input_data);
    let output_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .expect("output");
    let positions_buf = upload_i32(device, positions_data);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::rope_multi::dispatch_rope_multi_cached(
        &mut enc,
        registry,
        device,
        &input_buf,
        &output_buf,
        &positions_buf,
        p,
    )
    .expect("dispatch_cached");
    enc.commit_and_wait().expect("commit");

    output_buf.as_slice::<f32>().expect("read").to_vec()
}

#[test]
fn test_rope_multi_cached_matches_uncached_qwen35_decode_shape() {
    let (device, mut registry) = setup();
    mlx_native::ops::rope_multi::clear_rope_pack_cache();

    // qwen3.5 FullAttn decode: seq_len=1 (the steady-state hot path).
    let p_q = RopeMultiParams {
        head_dim: 256,
        rope_dim: 64,
        n_heads: 32, // n_heads (Q)
        seq_len: 1,
        freq_base: 1e7,
        mode: RopeMultiMode::Imrope,
        sections: [11, 11, 10, 0],
    };
    let p_k = RopeMultiParams {
        n_heads: 4, // n_kv_heads (K)
        ..p_q
    };

    let n_q = (p_q.seq_len * p_q.n_heads) as usize * p_q.head_dim as usize;
    let n_k = (p_k.seq_len * p_k.n_heads) as usize * p_k.head_dim as usize;

    let mut seed = 0xc0ffee99u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32)
    };

    let input_q: Vec<f32> = (0..n_q).map(|_| rand()).collect();
    let input_k: Vec<f32> = (0..n_k).map(|_| rand()).collect();
    // Text positions: all 4 axes = same token index (decode position 42).
    let positions: Vec<i32> = vec![42, 42, 42, 42];

    let want_q = run_rope_multi(&device, &mut registry, &input_q, &positions, p_q);
    let want_k = run_rope_multi(&device, &mut registry, &input_k, &positions, p_k);

    // Same shape via the cached path — first call populates, second hits.
    let got_q1 = run_rope_multi_cached(&device, &mut registry, &input_q, &positions, p_q);
    let got_k1 = run_rope_multi_cached(&device, &mut registry, &input_k, &positions, p_k);
    let got_q2 = run_rope_multi_cached(&device, &mut registry, &input_q, &positions, p_q);
    let got_k2 = run_rope_multi_cached(&device, &mut registry, &input_k, &positions, p_k);

    // Bit-exact (same kernel, same buffers; only param-buf source differs).
    for i in 0..n_q {
        assert_eq!(
            got_q1[i].to_bits(),
            want_q[i].to_bits(),
            "cached-vs-uncached mismatch Q[{}] {} vs {}",
            i, got_q1[i], want_q[i]
        );
        assert_eq!(
            got_q2[i].to_bits(),
            want_q[i].to_bits(),
            "cache-hit re-dispatch mismatch Q[{}]",
            i
        );
    }
    for i in 0..n_k {
        assert_eq!(
            got_k1[i].to_bits(),
            want_k[i].to_bits(),
            "cached-vs-uncached mismatch K[{}] {} vs {}",
            i, got_k1[i], want_k[i]
        );
        assert_eq!(
            got_k2[i].to_bits(),
            want_k[i].to_bits(),
            "cache-hit re-dispatch mismatch K[{}]",
            i
        );
    }

    // After the 4 cached calls (Q, K, Q, K — 2 distinct keys), the cache
    // should contain exactly 2 entries (one per (n_heads, ...) tuple).
    assert_eq!(
        mlx_native::ops::rope_multi::rope_pack_cache_len(),
        2,
        "expected 2 cache entries (Q + K), got {}",
        mlx_native::ops::rope_multi::rope_pack_cache_len()
    );

    mlx_native::ops::rope_multi::clear_rope_pack_cache();
    assert_eq!(
        mlx_native::ops::rope_multi::rope_pack_cache_len(),
        0,
        "cache should be empty after clear"
    );
}

#[test]
fn test_rope_multi_cached_seq_len_variation() {
    // Verify the cache key includes seq_len: prefill (seq>1) gets its
    // own entry, decode (seq=1) keeps a separate one.  Both must remain
    // bit-exact to the per-call path.
    let (device, mut registry) = setup();
    mlx_native::ops::rope_multi::clear_rope_pack_cache();

    let p_decode = RopeMultiParams {
        head_dim: 64,
        rope_dim: 32,
        n_heads: 4,
        seq_len: 1,
        freq_base: 1e6,
        mode: RopeMultiMode::Imrope,
        sections: [5, 5, 6, 0],
    };
    let p_prefill = RopeMultiParams {
        seq_len: 7,
        ..p_decode
    };

    let mut seed = 0xfacefeedu32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32)
    };

    let n_decode = (p_decode.seq_len * p_decode.n_heads) as usize * p_decode.head_dim as usize;
    let n_prefill = (p_prefill.seq_len * p_prefill.n_heads) as usize * p_prefill.head_dim as usize;

    let input_decode: Vec<f32> = (0..n_decode).map(|_| rand()).collect();
    let input_prefill: Vec<f32> = (0..n_prefill).map(|_| rand()).collect();

    let positions_decode: Vec<i32> = vec![3, 3, 3, 3];
    // Prefill: 4 * seq_len = 28 entries; cycle through token indices.
    let positions_prefill: Vec<i32> = (0..p_prefill.seq_len as i32)
        .cycle()
        .take(4 * p_prefill.seq_len as usize)
        .collect();

    let want_d = run_rope_multi(&device, &mut registry, &input_decode, &positions_decode, p_decode);
    let want_p = run_rope_multi(
        &device,
        &mut registry,
        &input_prefill,
        &positions_prefill,
        p_prefill,
    );

    let got_d =
        run_rope_multi_cached(&device, &mut registry, &input_decode, &positions_decode, p_decode);
    let got_p = run_rope_multi_cached(
        &device,
        &mut registry,
        &input_prefill,
        &positions_prefill,
        p_prefill,
    );

    for i in 0..n_decode {
        assert_eq!(got_d[i].to_bits(), want_d[i].to_bits(), "decode {}", i);
    }
    for i in 0..n_prefill {
        assert_eq!(got_p[i].to_bits(), want_p[i].to_bits(), "prefill {}", i);
    }
    assert_eq!(
        mlx_native::ops::rope_multi::rope_pack_cache_len(),
        2,
        "expected 2 entries (decode seq_len=1 + prefill seq_len=7)"
    );

    mlx_native::ops::rope_multi::clear_rope_pack_cache();
}

// =================================================================
// VISION mode (mode == 24, RopeMultiMode::Vision) — Qwen3-VL ViT
// =================================================================
//
// Spec sources:
//   /opt/llama.cpp/ggml/include/ggml.h:253                (mode = 24)
//   /opt/llama.cpp/ggml/include/ggml.h:1840-1846          ([yyyyxxxx] layout)
//   /opt/llama.cpp/ggml/src/ggml-cuda/rope.cu:268-328     (kernel ref)
//   /opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5643-5711    (cache_init w/ indep_sects)
//   /opt/llama.cpp/tools/mtmd/models/qwen3vl.cpp:14,111   (call site)
//
// Acceptance for the mlx-native PR: synthetic input where [yyyyxxxx] dim
// layout produces specific output values from a hand-computed reference;
// bitwise tolerance ≤ 1 ULP at FP32 (we use ≤ 1e-5 to allow `pow`/`cos`
// implementation variance between Metal and host f32, matching the
// existing IMROPE spec-driven test).

/// Hand-computed expected output for the [yyyyxxxx] layout.
///
/// Setup: head_dim=8, rope_dim=8 (vision: rope_dim == head_dim),
/// sections=[s0=2, s1=2, s2=0, s3=0], n_heads=1, seq_len=2,
/// freq_base=10000.
///
/// Per-pair theta (n_dims = head_dim/2 = 4):
///
///   pair 0 -> axis 0 (y), local_p=0, freq = 10000^(-0/4) = 1
///   pair 1 -> axis 0 (y), local_p=1, freq = 10000^(-2/4) = 1/100 = 0.01
///   pair 2 -> axis 1 (x), local_p=0, freq = 1
///   pair 3 -> axis 1 (x), local_p=1, freq = 0.01
///
/// With y_pos = [3, 5] and x_pos = [7, 11]:
///
///   row 0: thetas = [3.0, 0.03, 7.0, 0.07]
///   row 1: thetas = [5.0, 0.05, 11.0, 0.11]
///
/// Pairing is NeoX-style: rotate (x[pair], x[pair+head_dim/2]).
#[test]
fn test_rope_multi_vision_yyyyxxxx_layout_spec_driven() {
    let (device, mut registry) = setup();

    let p = RopeMultiParams {
        head_dim: 8,
        rope_dim: 8, // vision invariant: must equal head_dim
        n_heads: 1,
        seq_len: 2,
        freq_base: 10000.0,
        mode: RopeMultiMode::Vision,
        // s0=2 (y), s1=2 (x), last two ignored.
        sections: [2, 2, 0, 0],
    };

    // Input: 16 values, row-major [seq=0; seq=1] x head_dim=8.
    // Use 0.1 * (i + 1) so values are distinct and bounded.
    let input: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    // positions layout: 4 axes x seq_len=2 = 8 entries.
    //   [y_0, y_1, x_0, x_1, _, _, _, _]
    let positions = [3i32, 5, 7, 11, 0, 0, 0, 0];

    let got = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let want = cpu_rope_multi(&input, &positions, p);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 1e-5,
            "vision spec-driven mismatch at {}: got {}, want {}, diff {}",
            i,
            g,
            w,
            d
        );
    }

    // Independently pin pair 0 of row 0 (axis 0=y, local_p=0, theta=3).
    // x0 = input[0] = 0.1, x1 = input[0 + half_dim=4] = 0.5.
    // expected = (0.1*cos(3) - 0.5*sin(3),  0.1*sin(3) + 0.5*cos(3))
    let pair0_row0 = (got[0], got[4]);
    let expected_pair0_row0 = (
        0.1_f32 * 3.0_f32.cos() - 0.5_f32 * 3.0_f32.sin(),
        0.1_f32 * 3.0_f32.sin() + 0.5_f32 * 3.0_f32.cos(),
    );
    assert!(
        (pair0_row0.0 - expected_pair0_row0.0).abs() < 1e-5,
        "row0 pair0.x: got {}, want {}",
        pair0_row0.0,
        expected_pair0_row0.0
    );
    assert!(
        (pair0_row0.1 - expected_pair0_row0.1).abs() < 1e-5,
        "row0 pair0.y: got {}, want {}",
        pair0_row0.1,
        expected_pair0_row0.1
    );

    // Independently pin pair 2 of row 1 (axis 1=x, local_p=0, theta=11).
    // Index in flattened buffer: row 1 base = 8; pair 2 -> indices 8+2=10
    // and 8+2+4=14. x0 = input[10] = 1.1, x1 = input[14] = 1.5.
    let pair2_row1 = (got[10], got[14]);
    let expected_pair2_row1 = (
        1.1_f32 * 11.0_f32.cos() - 1.5_f32 * 11.0_f32.sin(),
        1.1_f32 * 11.0_f32.sin() + 1.5_f32 * 11.0_f32.cos(),
    );
    assert!(
        (pair2_row1.0 - expected_pair2_row1.0).abs() < 1e-5,
        "row1 pair2.x: got {}, want {}",
        pair2_row1.0,
        expected_pair2_row1.0
    );
    assert!(
        (pair2_row1.1 - expected_pair2_row1.1).abs() < 1e-5,
        "row1 pair2.y: got {}, want {}",
        pair2_row1.1,
        expected_pair2_row1.1
    );

    // Cross-check that vision actually differs from MROPE at the same
    // shape. MROPE on the same input uses unified pair_idx as exponent,
    // so pair 1 of row 0 should differ between the two modes.
    // Switch only the mode and compute MROPE: with sections [2,2,0,0],
    // sect_dims=4 in both modes, but for MROPE pair 1 -> axis 0, and the
    // dim_ratio uses pair_idx=1 / rope_dim=8 = 0.125 (NOT local_p/n_dims).
    let mut p_mrope = p;
    p_mrope.mode = RopeMultiMode::Mrope;
    let got_mrope = run_rope_multi(&device, &mut registry, &input, &positions, p_mrope);
    let d_pair1_row0 = (got[1] - got_mrope[1]).abs();
    let d_pair1_row0_b = (got[5] - got_mrope[5]).abs();
    assert!(
        d_pair1_row0 > 1e-3 || d_pair1_row0_b > 1e-3,
        "vision and mrope produced identical pair 1 — exponent denominator may not have switched (got_v=({},{}), got_m=({},{}))",
        got[1],
        got[5],
        got_mrope[1],
        got_mrope[5]
    );
}

// =================================================================
// Vision rope: text-degenerate (y_pos == x_pos for all positions)
// =================================================================
//
// When the y and x positions are identical, both axes contribute the
// same `pos` value. The resulting output is well-defined and finite —
// this test pins that property and acts as a smoke test for the
// kernel's per-section-restart path under degenerate-axis inputs.
// (We do NOT claim equivalence with IMROPE because the theta-exponent
// denominator differs — n_dims=head_dim/2 for vision vs rope_dim for
// mrope/imrope — so even with equal positions the outputs diverge by
// design.)

#[test]
fn test_rope_multi_vision_degenerate_positions_finite() {
    let (device, mut registry) = setup();

    let p = RopeMultiParams {
        head_dim: 16,
        rope_dim: 16, // vision: rope_dim == head_dim
        n_heads: 2,
        seq_len: 4,
        freq_base: 10000.0,
        mode: RopeMultiMode::Vision,
        // n_dims = 8 = s0 + s1; pick s0=s1=4 (matches Qwen3-VL's d_head/4 split).
        sections: [4, 4, 0, 0],
    };

    let n_rows = (p.seq_len * p.n_heads) as usize;
    let n_elem = n_rows * (p.head_dim as usize);

    let mut seed = 0xa1b2c3d4u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 1.5
    };
    let input: Vec<f32> = (0..n_elem).map(|_| rand()).collect();

    // y_pos == x_pos: both axes are the token index, so every pair
    // (y or x branch) gets the same `pos` value at a given seq_idx.
    let mut positions = Vec::with_capacity(4 * p.seq_len as usize);
    let token_pos: Vec<i32> = (0..p.seq_len as i32).collect();
    positions.extend_from_slice(&token_pos); // axis 0 (y)
    positions.extend_from_slice(&token_pos); // axis 1 (x)
    positions.extend_from_slice(&token_pos); // axis 2 (unused)
    positions.extend_from_slice(&token_pos); // axis 3 (unused)

    let got = run_rope_multi(&device, &mut registry, &input, &positions, p);
    let want = cpu_rope_multi(&input, &positions, p);

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            g.is_finite(),
            "non-finite output at {}: {}",
            i,
            g
        );
        let d = (g - w).abs();
        assert!(
            d < 1e-5,
            "vision degenerate mismatch at {}: got {}, want {}",
            i,
            g,
            w
        );
    }

    // Also pin: with equal y/x positions, swapping s0<->s1 should NOT
    // change the output (each pair maps to "the same pos" either way).
    let mut p_swapped = p;
    p_swapped.sections = [4, 4, 0, 0]; // s0=s1 already; trivially identical.
    let got_swapped = run_rope_multi(&device, &mut registry, &input, &positions, p_swapped);
    for (i, (&g, &gs)) in got.iter().zip(got_swapped.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            gs.to_bits(),
            "section swap with equal positions changed output at {}: {} vs {}",
            i,
            g,
            gs
        );
    }
}

// =================================================================
// Vision rope: invalid section sum is rejected by host validation
// =================================================================
//
// Vision requires sections[0] + sections[1] == head_dim/2. Any other
// sum must surface MlxError::InvalidArgument before the kernel
// dispatches. Also covers the rope_dim != head_dim rejection branch.

#[test]
fn test_rope_multi_vision_invalid_section_sum_rejected() {
    let (device, mut registry) = setup();

    // head_dim=4 -> n_dims=2, but sections sum to 4+4=8. Should fail
    // BEFORE any dispatch.
    let p_bad_sum = RopeMultiParams {
        head_dim: 4,
        rope_dim: 4,
        n_heads: 1,
        seq_len: 1,
        freq_base: 10000.0,
        mode: RopeMultiMode::Vision,
        sections: [4, 4, 0, 0],
    };

    // The validation runs inside dispatch_rope_multi. Build the small
    // buffers and a dummy input/output sized for the *requested* shape.
    let n_elem = (p_bad_sum.seq_len * p_bad_sum.n_heads * p_bad_sum.head_dim) as usize;
    let dummy_in = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![n_elem])
        .expect("dummy input alloc");
    let dummy_out = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![n_elem])
        .expect("dummy output alloc");
    let pos = device
        .alloc_buffer(16, DType::I32, vec![4])
        .expect("pos alloc");
    let (params, rope_params, sections) =
        build_rope_multi_buffers(&device, p_bad_sum).expect("build bufs");

    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy_in,
        &dummy_out,
        &pos,
        &params,
        &rope_params,
        &sections,
        p_bad_sum,
    );
    assert!(
        res.is_err(),
        "vision: sections[0]+sections[1] != head_dim/2 must error"
    );
    let msg = format!("{:?}", res.err().expect("err present"));
    assert!(
        msg.contains("Vision") || msg.contains("vision"),
        "error must mention Vision; got: {}",
        msg
    );

    // Also assert rope_dim != head_dim is rejected for vision.
    let p_partial = RopeMultiParams {
        head_dim: 8,
        rope_dim: 4, // vision forbids rope_dim < head_dim (no partial rotary)
        n_heads: 1,
        seq_len: 1,
        freq_base: 10000.0,
        mode: RopeMultiMode::Vision,
        sections: [2, 2, 0, 0],
    };
    let n_elem2 = (p_partial.seq_len * p_partial.n_heads * p_partial.head_dim) as usize;
    let dummy_in2 = device
        .alloc_buffer(n_elem2 * 4, DType::F32, vec![n_elem2])
        .expect("dummy input2 alloc");
    let dummy_out2 = device
        .alloc_buffer(n_elem2 * 4, DType::F32, vec![n_elem2])
        .expect("dummy output2 alloc");
    let pos2 = device
        .alloc_buffer(16, DType::I32, vec![4])
        .expect("pos2 alloc");
    let (params2, rope_params2, sections2) =
        build_rope_multi_buffers(&device, p_partial).expect("build bufs2");
    let mut enc2 = device.command_encoder().expect("enc2");
    let res2 = mlx_native::ops::rope_multi::dispatch_rope_multi(
        &mut enc2,
        &mut registry,
        device.metal_device(),
        &dummy_in2,
        &dummy_out2,
        &pos2,
        &params2,
        &rope_params2,
        &sections2,
        p_partial,
    );
    assert!(
        res2.is_err(),
        "vision: rope_dim != head_dim must error"
    );
    let msg2 = format!("{:?}", res2.err().expect("err2 present"));
    assert!(
        msg2.contains("Vision") || msg2.contains("vision"),
        "rope_dim error must mention Vision; got: {}",
        msg2
    );
}
