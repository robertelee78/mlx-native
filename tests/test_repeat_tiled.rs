//! Tests for the tiled-GQA broadcast GPU kernel (ADR-005 W-5b.19).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::repeat_tiled::{self, RepeatTiledParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    repeat_tiled::register(&mut registry);
    (device, registry)
}

/// Reference CPU implementation: the exact loop hf2q's
/// `gpu_delta_net::apply_gated_delta_net_chunk` GQA pre-expansion runs.
fn cpu_repeat_tiled_reference(
    src: &[f32],
    seq: usize,
    hg: usize,
    h: usize,
    k: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; seq * h * k];
    for t in 0..seq {
        let src_t_base = t * hg * k;
        let dst_t_base = t * h * k;
        for hi in 0..h {
            let kh = hi % hg;
            let src_off = src_t_base + kh * k;
            let dst_off = dst_t_base + hi * k;
            dst[dst_off..dst_off + k]
                .copy_from_slice(&src[src_off..src_off + k]);
        }
    }
    dst
}

fn run_repeat_at(seq: usize, hg: usize, h: usize, k: usize) {
    let (device, mut registry) = setup();

    // Distinguishable per-element values so byte-level mis-routing fails.
    let src_data: Vec<f32> = (0..(seq * hg * k))
        .map(|i| i as f32 * 0.5 + 1.0)
        .collect();

    // Allocate input buffer.
    let mut src_buf = device
        .alloc_buffer(src_data.len() * 4, DType::F32, vec![seq, hg, k])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("src mut_slice")
        .copy_from_slice(&src_data);

    // Allocate output buffer (zero-initialised by Metal).
    let dst_buf = device
        .alloc_buffer(seq * h * k * 4, DType::F32, vec![seq, h, k])
        .expect("alloc dst");

    let params = RepeatTiledParams {
        seq: seq as u32,
        hg: hg as u32,
        h: h as u32,
        k: k as u32,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    repeat_tiled::dispatch_repeat_tiled_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &dst_buf,
        &params,
    )
    .expect("dispatch_repeat_tiled_f32");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Compute CPU reference and compare bit-identically.
    let dst_ref = cpu_repeat_tiled_reference(&src_data, seq, hg, h, k);
    let dst_gpu = dst_buf.as_slice::<f32>().expect("read dst");

    assert_eq!(dst_gpu.len(), dst_ref.len(), "dst length mismatch");

    // Strided-copy = bit-identical (no math). Use exact equality.
    for (i, (g, r)) in dst_gpu.iter().zip(dst_ref.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            r.to_bits(),
            "dst bit-mismatch at i={i}: gpu={g}, cpu={r} \
             (seq={seq}, hg={hg}, h={h}, k={k})"
        );
    }
}

#[test]
fn test_repeat_tiled_qwen36_27b_shape_seq128() {
    // Qwen3.6-27B Gated DeltaNet shape (chunk-pipeline aligned):
    //   T = 128, Hg = n_k_heads = 2, H = n_v_heads = 16, K = d_k = 128
    //   group_ratio = H / Hg = 8 (W-5b.19 prompt cited 16/48=3 — but the
    //   actual qwen35::build_delta_net_layer test uses nk=2, nv=16 per
    //   gpu_delta_net.rs:2476-2479; we cover the actual production shape).
    run_repeat_at(128, 2, 16, 128);
}

#[test]
fn test_repeat_tiled_qwen36_27b_shape_pp4106() {
    // Same head config at the W-5b.16/17/18 PP4106 prefill seq length.
    run_repeat_at(4106, 2, 16, 128);
}

#[test]
fn test_repeat_tiled_group_ratio_3() {
    // Group ratio 3 case (matches the W-5b.19 prompt's cited config:
    // n_k_heads=16, n_v_heads=48). Validates the kh = h % Hg path on a
    // non-power-of-two ratio.
    run_repeat_at(64, 16, 48, 128);
}

#[test]
fn test_repeat_tiled_group_ratio_1_no_op() {
    // Hg == H — degenerate no-broadcast case (every dst head maps to the
    // same src head). Output must equal input verbatim.
    run_repeat_at(8, 4, 4, 32);
}

#[test]
fn test_repeat_tiled_seq_one() {
    // Edge case: seq == 1 (single-token boundary).
    run_repeat_at(1, 2, 16, 128);
}

#[test]
fn test_repeat_tiled_rejects_zero_dims() {
    let (device, mut registry) = setup();
    let buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc");
    let mut encoder = device.command_encoder().expect("encoder");
    let params = RepeatTiledParams { seq: 0, hg: 1, h: 1, k: 1 };
    let res = repeat_tiled::dispatch_repeat_tiled_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &params,
    );
    assert!(res.is_err(), "seq=0 should be rejected");
}

#[test]
fn test_repeat_tiled_rejects_non_multiple_h() {
    let (device, mut registry) = setup();
    let buf = device
        .alloc_buffer(4 * 4, DType::F32, vec![4])
        .expect("alloc");
    let mut encoder = device.command_encoder().expect("encoder");
    // h=5, hg=2 → 5 % 2 != 0
    let params = RepeatTiledParams { seq: 1, hg: 2, h: 5, k: 1 };
    let res = repeat_tiled::dispatch_repeat_tiled_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &params,
    );
    assert!(res.is_err(), "h not multiple of hg should be rejected");
}
