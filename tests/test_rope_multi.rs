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
    }
}

/// CPU reference — returns full output vector of length `n_rows * head_dim`.
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
    let sect_dims = p.sections.iter().sum::<u32>().max(1);

    let mut out = input.to_vec();
    for row in 0..n_rows {
        let base = row * head_dim;
        let seq_idx = (row as u32) / p.n_heads;

        for pair in 0..half_dim {
            if (pair as usize) < half_rope {
                let sector = (pair as u32) % sect_dims;
                let axis = pick_axis_cpu(sector, p.mode, p.sections);
                let pos = positions[(axis * p.seq_len + seq_idx) as usize];

                let dim_ratio = 2.0 * (pair as f32) / (rope_dim as f32);
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
