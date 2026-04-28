//! Tests for the fused-QKV split GPU kernel (ADR-005 W-5b.18).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::qkv_split::{self, QkvSplitParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    qkv_split::register(&mut registry);
    (device, registry)
}

/// Reference CPU implementation: the exact loop hf2q's
/// `gpu_delta_net::layer_qkv_deinterleave` CPU path runs.
fn cpu_qkv_split_reference(
    qkv: &[f32],
    seq: usize,
    q_sp: usize,
    k_sp: usize,
    v_sp: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let qkv_ch = q_sp + k_sp + v_sp;
    let mut q = vec![0.0f32; seq * q_sp];
    let mut k = vec![0.0f32; seq * k_sp];
    let mut v = vec![0.0f32; seq * v_sp];
    for t in 0..seq {
        let base = t * qkv_ch;
        q[t * q_sp..(t + 1) * q_sp]
            .copy_from_slice(&qkv[base..base + q_sp]);
        k[t * k_sp..(t + 1) * k_sp]
            .copy_from_slice(&qkv[base + q_sp..base + q_sp + k_sp]);
        v[t * v_sp..(t + 1) * v_sp]
            .copy_from_slice(&qkv[base + q_sp + k_sp..base + qkv_ch]);
    }
    (q, k, v)
}

fn run_split_at(seq: usize, q_sp: usize, k_sp: usize, v_sp: usize) {
    let (device, mut registry) = setup();
    let qkv_ch = q_sp + k_sp + v_sp;

    // Distinguishable per-element values so byte-level mis-routing fails.
    let qkv_data: Vec<f32> = (0..(seq * qkv_ch))
        .map(|i| i as f32 * 0.5 + 1.0)
        .collect();

    // Allocate input buffer.
    let mut qkv_buf = device
        .alloc_buffer(qkv_data.len() * 4, DType::F32, vec![seq, qkv_ch])
        .expect("alloc qkv");
    qkv_buf
        .as_mut_slice::<f32>()
        .expect("qkv mut_slice")
        .copy_from_slice(&qkv_data);

    // Allocate output buffers (zero-initialised by Metal).
    let q_buf = device
        .alloc_buffer(seq * q_sp * 4, DType::F32, vec![seq, q_sp])
        .expect("alloc q");
    let k_buf = device
        .alloc_buffer(seq * k_sp * 4, DType::F32, vec![seq, k_sp])
        .expect("alloc k");
    let v_buf = device
        .alloc_buffer(seq * v_sp * 4, DType::F32, vec![seq, v_sp])
        .expect("alloc v");

    let params = QkvSplitParams {
        seq: seq as u32,
        q_sp: q_sp as u32,
        k_sp: k_sp as u32,
        v_sp: v_sp as u32,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    qkv_split::dispatch_qkv_split_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &qkv_buf,
        &q_buf,
        &k_buf,
        &v_buf,
        &params,
    )
    .expect("dispatch_qkv_split_f32");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Compute CPU reference and compare bit-identically.
    let (q_ref, k_ref, v_ref) =
        cpu_qkv_split_reference(&qkv_data, seq, q_sp, k_sp, v_sp);

    let q_gpu = q_buf.as_slice::<f32>().expect("read q");
    let k_gpu = k_buf.as_slice::<f32>().expect("read k");
    let v_gpu = v_buf.as_slice::<f32>().expect("read v");

    assert_eq!(q_gpu.len(), q_ref.len(), "Q length mismatch");
    assert_eq!(k_gpu.len(), k_ref.len(), "K length mismatch");
    assert_eq!(v_gpu.len(), v_ref.len(), "V length mismatch");

    // Strided-copy = bit-identical (no math).  Use exact equality.
    for (i, (g, r)) in q_gpu.iter().zip(q_ref.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            r.to_bits(),
            "Q bit-mismatch at i={i}: gpu={g}, cpu={r} (seq={seq}, q_sp={q_sp})"
        );
    }
    for (i, (g, r)) in k_gpu.iter().zip(k_ref.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            r.to_bits(),
            "K bit-mismatch at i={i}: gpu={g}, cpu={r} (seq={seq}, k_sp={k_sp})"
        );
    }
    for (i, (g, r)) in v_gpu.iter().zip(v_ref.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            r.to_bits(),
            "V bit-mismatch at i={i}: gpu={g}, cpu={r} (seq={seq}, v_sp={v_sp})"
        );
    }
}

#[test]
fn test_qkv_split_qwen36_27b_shape_seq128() {
    // Qwen3.6-27B Gated DeltaNet shape (the exact W-5b.18 production target):
    //   n_k_heads = 2, n_v_heads = 16, d_k = 128, d_v = 128
    //   q_sp = k_sp = 2 * 128 = 256
    //   v_sp = 16 * 128 = 2048
    //   qkv_ch = 256 + 256 + 2048 = 2560
    //   seq = 128 (chunk-pipeline aligned)
    run_split_at(128, 256, 256, 2048);
}

#[test]
fn test_qkv_split_qwen36_27b_shape_pp4106() {
    // Same head config but at the W-5b.16/17 PP4106 prefill seq length.
    run_split_at(4106, 256, 256, 2048);
}

#[test]
fn test_qkv_split_small_balanced_shape() {
    // Small balanced case — exercises all 3 routing branches at every column.
    run_split_at(4, 8, 8, 8);
}

#[test]
fn test_qkv_split_unbalanced_v_dominant() {
    // V span dominant (mirrors GQA / MQA architectures).
    run_split_at(7, 16, 16, 96);
}

#[test]
fn test_qkv_split_seq_one() {
    // Edge case: seq == 1 (single-token boundary; not the production path
    // — that's the decode branch — but the kernel must still be correct).
    run_split_at(1, 256, 256, 2048);
}

#[test]
fn test_qkv_split_rejects_zero_dims() {
    let (device, mut registry) = setup();
    let buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc");
    let mut encoder = device.command_encoder().expect("encoder");
    let params = QkvSplitParams { seq: 0, q_sp: 1, k_sp: 1, v_sp: 1 };
    let res = qkv_split::dispatch_qkv_split_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf, &buf, &buf, &buf,
        &params,
    );
    assert!(res.is_err(), "seq=0 should be rejected");
}
