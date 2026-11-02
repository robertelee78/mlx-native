//! Determinism stress for the fused kernels that run two RMS reductions
//! through the same threadgroup scratch: the decode end-of-layer
//! (`fused_post_ff_norm2_endlayer_f32{,_v2}`,
//! `fused_moe_wsum_post_ff_norm2_endlayer_f32_v2`) and the post-attention
//! triple norm (`fused_post_attn_triple_norm_f32{,_v2}`).
//!
//! Phase 1 ends with every thread broadcast-reading the total from
//! `shared[0]`; Phase 2's partial-sum store (`shared[tid]` or
//! `shared[sgitg]`) overwrites `shared[0]` from thread 0. Without a
//! threadgroup barrier between them, a simdgroup that reads late normalizes
//! its slice with thread 0's partial sum. Every row of one dispatch gets the
//! same input, so every row must produce identical output bits.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use mlx_native::ops::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};
use mlx_native::{DType, KernelRegistry, MTLSize, MlxBuffer, MlxDevice};

const DIM: usize = 2816;
const ROWS: usize = 2048;
const TOP_K: usize = 8;
const COMMANDS: usize = 16;
const TG: u64 = 256;

fn pattern(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            offset + scale * ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        })
        .collect()
}

fn upload(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut b = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc");
    b.as_mut_slice::<f32>()
        .expect("write")
        .copy_from_slice(data);
    b
}

fn rows_of(device: &MlxDevice, row: &[f32]) -> MlxBuffer {
    upload(device, &row.repeat(ROWS))
}

fn zeros(device: &MlxDevice, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc")
}

/// Rows whose output bits differ from the modal row, over all commands.
fn stress(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kernel: &str,
    inputs: &[&MlxBuffer],
    outputs: &[(&MlxBuffer, usize)],
    params: &[u8],
    shared_bytes: u64,
) -> usize {
    let pipeline = registry
        .get_pipeline(kernel, device.metal_device())
        .expect("pipeline")
        .clone();
    let mut diverged = 0;
    for _ in 0..COMMANDS {
        let mut bindings: Vec<(u64, KernelArg)> = Vec::new();
        for b in inputs
            .iter()
            .copied()
            .chain(outputs.iter().map(|(b, _)| *b))
        {
            bindings.push((bindings.len() as u64, KernelArg::Buffer(b)));
        }
        bindings.push((bindings.len() as u64, KernelArg::Bytes(params)));
        let mut enc = device.command_encoder().expect("encoder");
        encode_threadgroups_with_args_and_shared(
            &mut enc,
            &pipeline,
            &bindings,
            &[(0, shared_bytes)],
            MTLSize::new(ROWS as u64, 1, 1),
            MTLSize::new(TG, 1, 1),
        );
        enc.commit_and_wait().expect("commit");
        let hashes: Vec<u64> = (0..ROWS)
            .map(|r| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for &(b, width) in outputs {
                    let v = b.as_slice::<u32>().expect("read");
                    v[r * width..(r + 1) * width].hash(&mut h);
                }
                h.finish()
            })
            .collect();
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &h in &hashes {
            *counts.entry(h).or_default() += 1;
        }
        let modal = counts.values().copied().max().unwrap();
        diverged += ROWS - modal;
    }
    eprintln!(
        "{kernel}: {diverged} of {} rows diverged from the modal row",
        ROWS * COMMANDS
    );
    diverged
}

#[test]
fn two_reduction_norm_kernels_are_deterministic_under_concurrency() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let attn_out = rows_of(&device, &pattern(1, DIM, 2.0, 0.0));
    let moe_accum = rows_of(&device, &pattern(2, DIM, 6.0, 0.5));
    let residual = rows_of(&device, &pattern(3, DIM, 4.0, 0.0));
    let w2 = upload(&device, &pattern(4, DIM, 0.5, 1.0));
    let w3 = upload(&device, &pattern(5, DIM, 0.5, 1.0));
    let w4 = upload(&device, &pattern(6, DIM, 0.5, 1.0));
    let w5 = upload(&device, &pattern(7, DIM, 0.5, 1.0));
    let scalar = upload(&device, &[0.75]);
    let experts = upload(&device, &pattern(8, TOP_K * DIM, 3.0, 0.0).repeat(ROWS));
    let weights = upload(&device, &pattern(9, TOP_K, 0.2, 0.125).repeat(ROWS));
    let (o1, o2, o3, o4) = (
        zeros(&device, ROWS * DIM),
        zeros(&device, ROWS * DIM),
        zeros(&device, ROWS * DIM),
        zeros(&device, ROWS * DIM),
    );
    let bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    let eps = 1e-6f32.to_bits();
    let endlayer = bytes(&[eps, DIM as u32, 0]);
    let wsum = bytes(&[eps, DIM as u32, TOP_K as u32, 0]);
    let triple = bytes(&[eps, DIM as u32]);
    let n_sg = TG / 32;
    let mut failures = Vec::new();
    let mut check = |kernel: &'static str, n: usize| {
        if n > 0 {
            failures.push((kernel, n));
        }
    };
    let end_in = [&attn_out, &moe_accum, &residual, &w2, &w3, &scalar];
    let end_out = [(&o1, DIM), (&o2, DIM)];
    for (kernel, shared) in [
        ("fused_post_ff_norm2_endlayer_f32", TG * 4),
        ("fused_post_ff_norm2_endlayer_f32_v2", n_sg * 4),
    ] {
        let n = stress(
            &device,
            &mut registry,
            kernel,
            &end_in,
            &end_out,
            &endlayer,
            shared,
        );
        check(kernel, n);
    }
    let kernel = "fused_moe_wsum_post_ff_norm2_endlayer_f32_v2";
    let n = stress(
        &device,
        &mut registry,
        kernel,
        &[&experts, &weights, &attn_out, &residual, &w2, &w3, &scalar],
        &end_out,
        &wsum,
        (n_sg.max(32) + DIM as u64) * 4,
    );
    check(kernel, n);
    let triple_in = [&residual, &attn_out, &w2, &w3, &w4, &w5];
    let triple_out = [(&o1, DIM), (&o2, DIM), (&o3, DIM), (&o4, DIM)];
    for (kernel, shared) in [
        ("fused_post_attn_triple_norm_f32", TG * 4),
        ("fused_post_attn_triple_norm_f32_v2", n_sg.max(32) * 4),
    ] {
        let n = stress(
            &device,
            &mut registry,
            kernel,
            &triple_in,
            &triple_out,
            &triple,
            shared,
        );
        check(kernel, n);
    }
    assert!(
        failures.is_empty(),
        "nondeterministic two-reduction norms: {failures:?}"
    );
}
