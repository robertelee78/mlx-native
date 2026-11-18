//! Bounded Apple-Silicon hardware witnesses, not physical-page-saving evidence.
//!
//! Launch this SAME target in two fresh processes (the native flag is cached):
//! MLX_UNRETAINED_REFS=0 cargo test --release --locked --test completed_expert_storage -- --ignored --nocapture --test-threads=1
//! MLX_UNRETAINED_REFS=1 cargo test --release --locked --test completed_expert_storage -- --ignored --nocapture --test-threads=1
//!
//! Requires the grouped-M8 Metal tensor family and prepared MM kernels. No
//! global environment mutation, model loads, resource counters, source-text
//! assertions, sleeps, GPU faults, or synthetic timeout/completion equivalence.
//! SourceArena owns its own queue/memberships; its pool commits their removal
//! and its device is dropped inside an autorelease pool before Weak expiry is
//! tested. A successful preparation wait alone is NOT residency retirement.
//!
//! Every post-release projection is compared with an independent integer-valued
//! reference: weight[e,n,k] = Spec::weight(e,n), input[t,k] = 1 + t%5.
//! This intentionally simple lifetime fixture is not the numerical release gate.
//! Validation errors are exercised before submission, not by provoking a GPU
//! error. Unexpected nonterminal wait errors require producer-side resource
//! coverage; they are not safely fault-injected by these consumer tests.

#![cfg(target_os = "macos")]

#[path = "support/completed_expert_storage.rs"]
mod support;

use mlx_native::ops::quantized_matmul_id_ggml::IdMmScheduleOptions;
use mlx_native::{
    command_buffer_uses_unretained_references,
    dispatch_prepared_id_mm_pooled_with_options_for_test,
    prepare_quantized_matmul_id_ggml_grouped_m8,
    prepare_quantized_matmul_id_ggml_grouped_m8_batch,
    DType, ExpertMmSchedule, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType,
    GroupedM8ExpertPlanes, GroupedM8Preparation, IdMmScratch, KernelRegistry,
    MlxDevice, MlxError,
};
use support::{assert_released, SourceArena, Spec};

const SPECS: [Spec; 3] = [
    Spec { ty: GgmlType::Q6_K, n: 64, k: 256, seed: 1 },
    Spec { ty: GgmlType::Q4_K, n: 128, k: 512, seed: 2 },
    Spec { ty: GgmlType::Q6_K, n: 64, k: 256, seed: 4 },
];

fn device() -> MlxDevice {
    let expected = std::env::var("MLX_UNRETAINED_REFS").as_deref() == Ok("1");
    assert_eq!(command_buffer_uses_unretained_references(), expected);
    eprintln!("completed expert storage: native unretained references={expected}");
    MlxDevice::new().expect("Apple-Silicon hardware for completed expert storage tests")
}

fn project(device: &MlxDevice, registry: &mut KernelRegistry,
    planes: &GroupedM8ExpertPlanes, spec: Spec) {
    objc::rc::autoreleasepool(|| {
        let params = spec.params();
        let tokens = params.n_tokens as usize;
        let top_k = params.top_k as usize;
        let k = params.k as usize;
        let n = params.n as usize;
        let mut input = device.alloc_buffer(tokens * k * 4, DType::F32, vec![tokens, k])
            .expect("projection input");
        for (token, row) in input.as_mut_slice::<f32>().expect("input slice")
            .chunks_exact_mut(k).enumerate() {
            row.fill((1 + token % 5) as f32);
        }
        let mut ids = device.alloc_buffer(tokens * top_k * 4, DType::U32, vec![tokens, top_k])
            .expect("projection ids");
        for (index, id) in ids.as_mut_slice::<u32>().expect("ids slice").iter_mut().enumerate() {
            *id = ((index / top_k + index % top_k) % params.n_experts as usize) as u32;
        }
        let words = tokens * top_k * n;
        let mut output = device.alloc_buffer(words * 4, DType::F32, vec![tokens, top_k, n])
            .expect("projection output");
        output.as_mut_slice::<f32>().expect("output slice").fill(f32::NAN);
        let mut scratch = IdMmScratch::alloc(device, params.n_experts, params.n_tokens)
            .expect("projection scratch");
        let routing = GgmlRoutingPolicy {
            expert_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            expert_mm_schedule: ExpertMmSchedule::UniformGrid,
            ..GgmlRoutingPolicy::default()
        };
        assert!(params.n_tokens > routing.expert_mm_threshold);
        let mut encoder = device.command_encoder().expect("post-release consumer encoder");
        dispatch_prepared_id_mm_pooled_with_options_for_test(
            &mut encoder, registry, device, &input, planes, &ids, &output, &mut scratch,
            &params, false, false, &routing,
            IdMmScheduleOptions::production(ExpertMmSchedule::UniformGrid),
        ).expect("actual prepared projection after source retirement");
        encoder.commit_and_wait_labeled("test.completed_expert_storage.project")
            .expect("post-release projection completion");
        let actual = output.as_slice::<f32>().expect("completed output");
        for token in 0..tokens {
            for slot in 0..top_k {
                let expert = ((token + slot) % params.n_experts as usize) as u32;
                for row in 0..n {
                    let expected = spec.weight(expert, row as u32)
                        * (1 + token % 5) as f32 * params.k as f32;
                    let index = (token * top_k + slot) * n + row;
                    assert_eq!(actual[index].to_bits(), expected.to_bits(),
                        "post-release {:?} seed={} token={token} slot={slot} row={row}",
                        spec.ty, spec.seed);
                }
            }
        }
    });
}

#[test]
#[ignore = "requires grouped-M8-capable Apple Silicon; run explicitly in retained and unretained modes"]
fn completed_batch_releases_mixed_sources_and_clones_share_reusable_planes() {
    objc::rc::autoreleasepool(|| {
        let device = device();
        let mut registry = KernelRegistry::new();
        let (clones, witnesses) = objc::rc::autoreleasepool(|| {
            let mut sources = SourceArena::new(&device);
            let mut witnesses = Vec::new();
            let requests = SPECS.iter().map(|&spec| {
                let (source, witness) = sources.source(spec);
                witnesses.push(witness);
                GroupedM8Preparation { source, params: spec.params() }
            }).collect();
            let completed = prepare_quantized_matmul_id_ggml_grouped_m8_batch(
                &device, &mut registry, requests,
            ).expect("complete all mixed Q6/Q4/Q6 requests in original order");
            assert_eq!(completed.len(), SPECS.len());
            let clones: Vec<_> = completed.iter().map(|owner| {
                let cloned = owner.clone();
                assert!(std::ptr::eq::<GroupedM8ExpertPlanes>(&**owner, &*cloned),
                    "completed clones must share the very same plane object");
                cloned
            }).collect();
            drop(completed);
            // This ends the actual source registration owners, not just the
            // preparation encoder. No other test's pending additions exist here.
            drop(sources);
            (clones, witnesses)
        });
        assert_released(&witnesses);
        for (&spec, owner) in SPECS.iter().zip(&clones) {
            project(&device, &mut registry, owner, spec);
        }
        let survivors: Vec<_> = clones.iter().rev().cloned().collect();
        drop(clones);
        assert_released(&witnesses);
        for (&spec, owner) in SPECS.iter().rev().zip(&survivors) {
            project(&device, &mut registry, owner, spec);
        }
    });
}

#[test]
#[ignore = "requires grouped-M8-capable Apple Silicon; run explicitly in retained and unretained modes"]
fn asynchronous_prepared_owner_and_its_clone_legitimately_retain_input() {
    for spec in [SPECS[0], SPECS[1]] {
        objc::rc::autoreleasepool(|| {
            let device = device();
            let mut registry = KernelRegistry::new();
            let (pending, witness) = objc::rc::autoreleasepool(|| {
                let mut sources = SourceArena::new(&device);
                let (source, witness) = sources.source(spec);
                let mut encoder = device.command_encoder().expect("asynchronous producer encoder");
                let pending = prepare_quantized_matmul_id_ggml_grouped_m8(
                    &mut encoder, &mut registry, &device, &source, &spec.params(),
                ).expect("encode-only preparation");
                drop(source);
                drop(sources);
                // The source fixture keeps no escrow. In unretained mode this
                // also catches a missing async source owner before submission.
                assert!(witness.upgrade().is_some(), "async input lost before submission");
                encoder.commit_and_wait_labeled("test.completed_expert_storage.async")
                    .expect("settle real async producer before access or final release");
                drop(encoder);
                (pending, witness)
            });
            // Encoder/CB, source pool/device, originals and autoreleases ended.
            assert!(witness.upgrade().is_some(), "async owner must still retain its input");
            let clone = pending.clone();
            assert!(std::ptr::eq::<GroupedM8ExpertPlanes>(&*pending, &*clone));
            project(&device, &mut registry, &pending, spec);
            drop(pending);
            assert!(witness.upgrade().is_some(), "async clone must retain its input");
            project(&device, &mut registry, &clone, spec);
            objc::rc::autoreleasepool(|| drop(clone));
            assert_released(&[witness]);
        });
    }
}

#[derive(Clone, Copy, Debug)]
enum InvalidLater {
    Dtype,
    Codec,
    Geometry,
    Alignment,
    LogicalExtent,
    Stride,
    ExtentOverflow,
}

#[test]
#[ignore = "requires grouped-M8-capable Apple Silicon; run explicitly in retained and unretained modes"]
fn invalid_later_requests_release_all_consumed_sources_and_allow_recovery() {
    objc::rc::autoreleasepool(|| {
        let device = device();
        let mut registry = KernelRegistry::new();
        for invalid in [InvalidLater::Dtype, InvalidLater::Codec, InvalidLater::Geometry,
            InvalidLater::Alignment, InvalidLater::LogicalExtent, InvalidLater::Stride,
            InvalidLater::ExtentOverflow] {
            let (witnesses, caller_alias) = objc::rc::autoreleasepool(|| {
                let mut sources = SourceArena::new(&device);
                let (first, first_weak) = sources.source(SPECS[0]);
                let (mut second, second_weak) = sources.source(SPECS[1]);
                let (unvisited, unvisited_weak) = sources.source(SPECS[2]);
                let caller_alias = first.clone();
                let mut params = SPECS[1].params();
                match invalid {
                    InvalidLater::Dtype => {
                        second = second.view_dtype(DType::F16).expect("invalid typed view");
                    }
                    InvalidLater::Codec => {
                        params.ggml_type = GgmlType::Q8_0;
                    }
                    InvalidLater::Geometry => {
                        params.n -= 1;
                    }
                    InvalidLater::Alignment => {
                        second = second.slice_view(1, second.data_byte_len() - 1);
                    }
                    InvalidLater::LogicalExtent => {
                        second = second.slice_view(0, second.data_byte_len() - 1);
                    }
                    InvalidLater::Stride => {
                        params.expert_stride -= 64;
                    }
                    InvalidLater::ExtentOverflow => {
                        params.n_experts = u32::MAX;
                        params.expert_stride = u64::MAX - 1;
                    }
                };
                let requests = vec![
                    GroupedM8Preparation { source: first, params: SPECS[0].params() },
                    GroupedM8Preparation { source: second, params },
                    // Neither this tail nor the repeated first source is
                    // validated after the error, but both are consumed owners.
                    GroupedM8Preparation { source: unvisited, params: SPECS[2].params() },
                    GroupedM8Preparation { source: caller_alias.clone(), params: SPECS[0].params() },
                ];
                match prepare_quantized_matmul_id_ggml_grouped_m8_batch(
                    &device, &mut registry, requests,
                ) {
                    Err(MlxError::InvalidArgument(_)) => {}
                    Err(error) => panic!("{invalid:?}: wrong failure boundary: {error}"),
                    Ok(_) => panic!("{invalid:?}: must not publish any completed results"),
                }
                drop(sources);
                (vec![first_weak, second_weak, unvisited_weak], caller_alias)
            });
            assert_released(&witnesses[1..]);
            assert!(witnesses[0].upgrade().is_some(), "caller alias must remain a real owner");
            objc::rc::autoreleasepool(|| drop(caller_alias));
            assert_released(&witnesses);

            // Reuse the SAME consumer device and registry after each failed
            // batch. Release recovery inputs before actually using its output.
            let (completed, recovery_witnesses) = objc::rc::autoreleasepool(|| {
                let mut sources = SourceArena::new(&device);
                let mut recovery_witnesses = Vec::new();
                let requests = SPECS[..2].iter().map(|&spec| {
                    let (source, witness) = sources.source(spec);
                    recovery_witnesses.push(witness);
                    GroupedM8Preparation { source, params: spec.params() }
                }).collect();
                let completed = prepare_quantized_matmul_id_ggml_grouped_m8_batch(
                    &device, &mut registry, requests,
                ).expect("valid batch after partial validation failure");
                assert_eq!(completed.len(), 2);
                drop(sources);
                (completed, recovery_witnesses)
            });
            assert_released(&recovery_witnesses);
            for (&spec, owner) in SPECS[..2].iter().zip(&completed) {
                project(&device, &mut registry, owner, spec);
            }
        }
    });
}
