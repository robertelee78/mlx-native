//! Determinism stress for the single-token (decode) fused MoE routing
//! kernels and their selection-witness twins.
//!
//! Each kernel broadcasts the softmax max through threadgroup memory and then
//! reuses that memory in Step 2: V2/V3 store `exp` values over
//! `shared[num_experts + i]` (thread 0 overwrites the broadcast slot
//! `shared[num_experts + 0]`), V1 stores `shared[tid] = local_sum` (thread 0
//! overwrites `shared[0]`). Without a threadgroup barrier between the
//! broadcast read and those stores, a simdgroup that reaches the read late
//! uses a clobbered max: every expert it owns (`tid`-strided, so experts
//! 32..63 and 96..127 for simdgroup 1 at `tg = 64`) is rescaled relative to
//! the other simdgroup's experts, and top-K selects the wrong experts.
//!
//! The logits are adversarial for that failure: the true top-8 lives in
//! simdgroup 0's experts and logit 0 is far below the max, so a clobbered max
//! promotes simdgroup 1's experts into the top-8. Many independent instances
//! run concurrently in one command buffer to create scheduling skew between
//! simdgroups; every instance must be bit-identical to the CPU-checked
//! reference instance.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};
use mlx_native::{DType, KernelRegistry, MTLSize, MlxBuffer, MlxDevice};

const EXPERTS: usize = 128;
const TOP_K: usize = 8;
const TG: u64 = 64;
const INSTANCES_PER_COMMAND: usize = 8192;
const COMMANDS: usize = 64;

fn adversarial_logits() -> Vec<f32> {
    let mut logits: Vec<f32> = (0..EXPERTS).map(|i| -2.0 + 0.001 * i as f32).collect();
    // Simdgroup 1 owns experts 32..63 and 96..127 at tg = 64.
    for (i, l) in logits.iter_mut().enumerate() {
        if (32..64).contains(&i) || (96..128).contains(&i) {
            *l = 4.0 + 0.01 * i as f32;
        }
    }
    // True top-8: experts 1..=8, all owned by simdgroup 0.
    for j in 0..TOP_K {
        logits[1 + j] = 8.0 - 0.25 * j as f32;
    }
    // Thread 0's Step-2 store (exp(logits[0] - max) or its partial sum) is
    // tiny, so a clobbered broadcast max inflates simdgroup 1 by ~e^8.
    logits[0] = -20.0;
    logits
}

fn cpu_reference(logits: &[f32], scale: &[f32]) -> (Vec<u32>, Vec<f64>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap());
    let ids: Vec<u32> = order[..TOP_K].iter().map(|&i| i as u32).collect();
    let topk_sum: f64 = order[..TOP_K].iter().map(|&i| exps[i] / sum).sum();
    let weights = order[..TOP_K]
        .iter()
        .map(|&i| exps[i] / sum / topk_sum * scale[i] as f64)
        .collect();
    (ids, weights)
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

/// Returns the number of instances whose output bits differ from instance 0
/// (after checking instance 0 against the CPU reference).
fn stress(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    kernel: &str,
    shared_slots: u64,
    witness: bool,
) -> usize {
    let logits_data = adversarial_logits();
    let scale_data: Vec<f32> = (0..EXPERTS).map(|i| 1.0 + 0.001 * i as f32).collect();
    let (ref_ids, ref_weights) = cpu_reference(&logits_data, &scale_data);
    let logits = upload(device, &logits_data);
    let scale = upload(device, &scale_data);
    let n = INSTANCES_PER_COMMAND;
    let alloc = |dtype: DType, per: usize| {
        device
            .alloc_buffer(n * per * 4, dtype, vec![n * per])
            .expect("alloc output")
    };
    let ids = alloc(DType::U32, TOP_K);
    let weights = alloc(DType::F32, TOP_K);
    let w_scores = alloc(DType::F32, 2);
    let w_ids = alloc(DType::U32, 2);
    let w_status = alloc(DType::U32, 1);
    let params: [u32; 2] = [EXPERTS as u32, TOP_K as u32];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let mut reference: Option<(Vec<u32>, Vec<u32>)> = None;
    let mut diverged = 0usize;
    for _ in 0..COMMANDS {
        let pipeline = registry
            .get_pipeline(kernel, device.metal_device())
            .expect("pipeline")
            .clone();
        let mut enc = device.command_encoder().expect("encoder");
        for i in 0..n {
            let o8 = (i * TOP_K * 4) as u64;
            let mut bindings = vec![
                (0, KernelArg::Buffer(&logits)),
                (1, KernelArg::BufferWithOffset(&ids, o8)),
                (2, KernelArg::BufferWithOffset(&weights, o8)),
                (3, KernelArg::Buffer(&scale)),
                (4, KernelArg::Bytes(&params_bytes)),
            ];
            if witness {
                bindings.push((5, KernelArg::BufferWithOffset(&w_scores, (i * 8) as u64)));
                bindings.push((6, KernelArg::BufferWithOffset(&w_ids, (i * 8) as u64)));
                bindings.push((7, KernelArg::BufferWithOffset(&w_status, (i * 4) as u64)));
            }
            encode_threadgroups_with_args_and_shared(
                &mut enc,
                &pipeline,
                &bindings,
                &[(0, shared_slots * 4)],
                MTLSize::new(1, 1, 1),
                MTLSize::new(TG, 1, 1),
            );
        }
        enc.commit_and_wait().expect("commit");
        let got_ids = ids.as_slice::<u32>().expect("ids");
        let got_w = weights.as_slice::<f32>().expect("weights");
        // Reference: the first instance that matches the CPU softmax/top-K.
        let reference = reference.get_or_insert_with(|| {
            let matches_cpu = |i: usize| {
                let s = i * TOP_K..(i + 1) * TOP_K;
                got_ids[s.clone()] == ref_ids[..]
                    && got_w[s]
                        .iter()
                        .zip(&ref_weights)
                        .all(|(&w, &r)| ((w as f64) - r).abs() <= 1e-5 * r.abs())
            };
            let i = (0..n)
                .find(|&i| matches_cpu(i))
                .unwrap_or_else(|| panic!("{kernel}: no instance matches the CPU reference"));
            let s = i * TOP_K..(i + 1) * TOP_K;
            (
                got_ids[s.clone()].to_vec(),
                got_w[s].iter().map(|w| w.to_bits()).collect(),
            )
        });
        for i in 0..n {
            let s = i * TOP_K..(i + 1) * TOP_K;
            let same_ids = got_ids[s.clone()] == reference.0[..];
            let same_w = got_w[s]
                .iter()
                .zip(&reference.1)
                .all(|(w, r)| w.to_bits() == *r);
            if !(same_ids && same_w) {
                if diverged == 0 {
                    eprintln!(
                        "{kernel}: first divergent top-K {:?} (reference {:?})",
                        &got_ids[i * TOP_K..(i + 1) * TOP_K],
                        reference.0
                    );
                }
                diverged += 1;
            }
        }
    }
    eprintln!(
        "{kernel}: {diverged} of {} concurrent instances diverged from the reference",
        n * COMMANDS
    );
    diverged
}

#[test]
fn decode_moe_routing_kernels_are_deterministic_under_concurrency() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let e = EXPERTS as u64;
    let n_sg = TG.div_ceil(32);
    let cases = [
        ("fused_moe_routing_f32", 2 * e + TG, false),
        ("fused_moe_routing_f32_v2", 2 * e + n_sg, false),
        ("fused_moe_routing_f32_v3", 2 * e + 2 * n_sg, false),
        ("fused_moe_routing_f32_selection_witness", 2 * e + TG, true),
        (
            "fused_moe_routing_f32_v2_selection_witness",
            2 * e + n_sg,
            true,
        ),
        (
            "fused_moe_routing_f32_v3_selection_witness",
            2 * e + 2 * n_sg,
            true,
        ),
    ];
    let failures: Vec<(&str, usize)> = cases
        .iter()
        .map(|&(kernel, slots, witness)| {
            (
                kernel,
                stress(&device, &mut registry, kernel, slots, witness),
            )
        })
        .filter(|&(_, diverged)| diverged > 0)
        .collect();
    assert!(
        failures.is_empty(),
        "nondeterministic decode routing: {failures:?}"
    );
}
