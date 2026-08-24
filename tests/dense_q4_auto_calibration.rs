#![cfg(mlx_native_has_metal_tensor_artifact)]

use half::f16;
use mlx_native::{
    calibrate_dense_q4_routes, trace_dense_q4_auto, DType, DenseQ4BaseShape,
    DenseQ4CalibrationCase, DenseQ4CalibrationLimits, DenseQ4DecisionSource, DenseQ4InputLayout,
    DenseQ4Route, GgmlQuantizedMatmulParams, GgmlRoutingPolicy, GgmlType, KernelRegistry,
    MlxBuffer, MlxDevice,
};

fn q4_bytes(n: usize, k: usize, salt: u8) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let mut bytes = Vec::with_capacity(n * (k / 32) * 18);
    for block in 0..n * (k / 32) {
        let scale = f16::from_f32(0.015625 + f32::from(salt) / 8192.0);
        bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
        for index in 0..16 {
            let low = ((block + index + usize::from(salt)) % 15 + 1) as u8;
            let high = ((block * 3 + index * 5 + usize::from(salt)) % 15 + 1) as u8;
            bytes.push(low | (high << 4));
        }
    }
    bytes
}

fn weight(device: &MlxDevice, n: usize, k: usize, salt: u8) -> MlxBuffer {
    let bytes = q4_bytes(n, k, salt);
    let mut weight = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("Q4 weight");
    weight
        .as_mut_slice::<u8>()
        .expect("map Q4 weight")
        .copy_from_slice(&bytes);
    weight
}

fn nan_scale_weight(device: &MlxDevice, n: usize, k: usize) -> MlxBuffer {
    let mut bytes = q4_bytes(n, k, 23);
    for block in bytes.chunks_exact_mut(18) {
        block[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    }
    let mut weight = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("NaN-scale Q4 weight");
    weight
        .as_mut_slice::<u8>()
        .expect("map NaN-scale Q4 weight")
        .copy_from_slice(&bytes);
    weight
}

fn input(device: &MlxDevice, m: usize, k: usize) -> MlxBuffer {
    let mut input = device
        .alloc_buffer(m * k * size_of::<f32>(), DType::F32, vec![m, k])
        .expect("F32 input");
    for (index, value) in input
        .as_mut_slice::<f32>()
        .expect("map input")
        .iter_mut()
        .enumerate()
    {
        *value = ((index * 29 % 251) as f32 - 125.0) / 1003.0;
    }
    input
}

fn output(device: &MlxDevice, m: usize, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(m * n * size_of::<f32>(), DType::F32, vec![m, n])
        .expect("F32 output")
}

fn base(n: u32, k: u32) -> DenseQ4BaseShape {
    DenseQ4BaseShape {
        n,
        k,
        batch: 1,
        input_layout: DenseQ4InputLayout::Contiguous,
    }
}

fn assert_cleanup_boundary(receipt: &mlx_native::DenseQ4CalibrationBatchReceipt) {
    let authorized_shape_weight_pairs: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.authorized_weight_buffers)
        .sum();
    let proof_submissions: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.proof_submissions)
        .sum();
    let proof_route_dispatches: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.proof_route_dispatches)
        .sum();
    let proof_auxiliary_dispatches: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.proof_auxiliary_dispatches)
        .sum();
    let peak_proof_scratch_bytes = receipt
        .decisions
        .iter()
        .map(|decision| decision.proof_scratch_bytes)
        .max()
        .unwrap_or(0);
    let proof_gpu_us: f64 = receipt
        .decisions
        .iter()
        .map(|decision| decision.proof_gpu_us)
        .sum();
    let timing_submissions: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.timing_submissions)
        .sum();
    let decision_submissions: u32 = receipt
        .decisions
        .iter()
        .map(|decision| decision.calibration_submissions)
        .sum();
    assert_eq!(
        receipt.authorized_shape_weight_pairs,
        authorized_shape_weight_pairs
    );
    assert_eq!(receipt.proof_submissions, proof_submissions);
    assert_eq!(receipt.proof_route_dispatches, proof_route_dispatches);
    assert_eq!(
        receipt.proof_auxiliary_dispatches,
        proof_auxiliary_dispatches
    );
    assert_eq!(receipt.peak_proof_scratch_bytes, peak_proof_scratch_bytes);
    assert_eq!(receipt.proof_gpu_us.to_bits(), proof_gpu_us.to_bits());
    assert_eq!(receipt.timing_submissions, timing_submissions);
    assert_eq!(decision_submissions, proof_submissions + timing_submissions);
    assert_eq!(receipt.cleanup_submissions, 1);
    assert_eq!(
        receipt.calibration_submissions,
        decision_submissions + receipt.cleanup_submissions,
        "one final empty submission must flush dropped calibration scratch"
    );
}

#[test]
fn exact_union_freezes_once_and_undeclared_shape_falls_back() {
    let device = MlxDevice::new().expect("Metal device");
    let weight = weight(&device, 3072, 768, 3);
    let mut registry = KernelRegistry::new();
    let cases = [
        DenseQ4CalibrationCase {
            weight: &weight,
            shape: base(3072, 768),
            reachable_m: &[32],
        },
        DenseQ4CalibrationCase {
            weight: &weight,
            shape: base(3072, 768),
            reachable_m: &[129],
        },
    ];
    let (plan, receipt) = calibrate_dense_q4_routes(
        &mut registry,
        &device,
        101,
        DenseQ4CalibrationLimits {
            max_elapsed_ms: 20_000,
            max_shapes: 2,
        },
        &cases,
    )
    .expect("calibrate Q4 routes");
    assert_cleanup_boundary(&receipt);
    assert_eq!(receipt.declared_shapes, 2);
    assert_eq!(plan.decision_count(), 2);
    assert_eq!(plan.activation_epoch(), 101);
    assert!(receipt.decisions.iter().all(|decision| {
        decision
            .timings
            .iter()
            .any(|timing| timing.route == DenseQ4Route::Tensor64x32)
    }), "the candidate must be proved and timed even when the stable-winner gate selects compatibility: {:?}", receipt.decisions);

    for m in [32u32, 129] {
        let input = input(&device, m as usize, 768);
        let output = output(&device, m as usize, 3072);
        let params = GgmlQuantizedMatmulParams {
            m,
            n: 3072,
            k: 768,
            ggml_type: GgmlType::Q4_0,
        };
        let mut encoder = device.command_encoder().expect("encoder");
        let trace = trace_dense_q4_auto(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            &output,
            &params,
            &GgmlRoutingPolicy::default(),
        )
        .expect("trace declared shape");
        assert_eq!(trace.decision.source, DenseQ4DecisionSource::FrozenPlan);
        assert_eq!(trace.plan_id.as_deref(), Some(plan.plan_id()));
        encoder.commit_and_wait().expect("execute declared shape");
    }

    let policy_input = input(&device, 32, 768);
    let policy_output = output(&device, 32, 3072);
    let params = GgmlQuantizedMatmulParams {
        m: 32,
        n: 3072,
        k: 768,
        ggml_type: GgmlType::Q4_0,
    };
    let mut v1_policy = GgmlRoutingPolicy::default();
    v1_policy.allow_dense_large_tile_mm = false;
    let mut encoder = device.command_encoder().expect("policy encoder");
    let trace = trace_dense_q4_auto(
        &mut encoder,
        &mut registry,
        &device,
        &policy_input,
        &weight,
        &policy_output,
        &params,
        &v1_policy,
    )
    .expect("trace policy-disabled shape");
    assert_eq!(trace.decision.route, DenseQ4Route::CompatibilityTensorV1);
    assert_eq!(
        trace.decision.source,
        DenseQ4DecisionSource::IneligibleCompatibilityFallback
    );
    assert_eq!(trace.pipeline.kernel_name, "kernel_mul_mm_q4_0_tensor_f32");
    encoder
        .commit_and_wait()
        .expect("execute policy-disabled shape");

    let input = input(&device, 33, 768);
    let output = output(&device, 33, 3072);
    let params = GgmlQuantizedMatmulParams {
        m: 33,
        n: 3072,
        k: 768,
        ggml_type: GgmlType::Q4_0,
    };
    let mut encoder = device.command_encoder().expect("fallback encoder");
    let trace = trace_dense_q4_auto(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &params,
        &GgmlRoutingPolicy::default(),
    )
    .expect("trace undeclared shape");
    assert_eq!(trace.decision.route, DenseQ4Route::CompatibilityV2);
    assert_eq!(
        trace.decision.source,
        DenseQ4DecisionSource::UndeclaredCompatibilityFallback
    );
    encoder.commit_and_wait().expect("execute fallback shape");

    assert!(calibrate_dense_q4_routes(
        &mut registry,
        &device,
        102,
        DenseQ4CalibrationLimits::default(),
        &cases,
    )
    .is_err());
    assert!(registry
        .try_register_source(
            "kernel_mul_mm_q4_0_tensor_64x32_f32",
            "kernel void replacement() {}",
        )
        .is_err());
}

#[test]
fn same_address_new_epoch_and_a_b_a_reuse_only_timing_metadata() {
    let device = MlxDevice::new().expect("Metal device");
    let mut borrowed_weight = weight(&device, 67, 160, 5);
    let stable_address = &borrowed_weight as *const MlxBuffer as usize;
    let limits = DenseQ4CalibrationLimits {
        max_elapsed_ms: 20_000,
        max_shapes: 1,
    };

    let mut registry_a1 = KernelRegistry::new();
    let (plan_a1, first) = calibrate_dense_q4_routes(
        &mut registry_a1,
        &device,
        201,
        limits,
        &[DenseQ4CalibrationCase {
            weight: &borrowed_weight,
            shape: base(67, 160),
            reachable_m: &[33],
        }],
    )
    .expect("first A activation");
    assert_cleanup_boundary(&first);
    assert_eq!(first.declared_shapes, 1);

    let mut registry_b_transfer_attempt = KernelRegistry::new();
    let transfer_error = registry_b_transfer_attempt
        .freeze_dense_q4_plan(&device, plan_a1.clone())
        .expect_err("A's plan must not authorize a distinct registry/model activation");
    assert!(transfer_error
        .to_string()
        .contains("different registry activation authority"));
    assert!(registry_b_transfer_attempt.dense_q4_plan().is_none());

    let first_native_bytes = borrowed_weight
        .as_slice::<u8>()
        .expect("read first native Q4 bytes")
        .to_vec();
    let replacement_native_bytes = q4_bytes(67, 160, 11);
    assert_ne!(first_native_bytes, replacement_native_bytes);
    borrowed_weight
        .as_mut_slice::<u8>()
        .expect("mutate same allocation")
        .copy_from_slice(&replacement_native_bytes);
    assert_eq!(
        &borrowed_weight as *const MlxBuffer as usize,
        stable_address
    );
    let mut registry_a2 = KernelRegistry::new();
    let (plan_a2, second) = calibrate_dense_q4_routes(
        &mut registry_a2,
        &device,
        202,
        limits,
        &[DenseQ4CalibrationCase {
            weight: &borrowed_weight,
            shape: base(67, 160),
            reachable_m: &[33],
        }],
    )
    .expect("same-address second epoch");
    assert_cleanup_boundary(&second);
    assert_eq!(second.process_cache_hits, 1);
    assert_eq!(second.authorized_shape_weight_pairs, 1);
    assert_eq!(second.proof_submissions, 1);
    assert_eq!(second.proof_route_dispatches, 2);
    assert_eq!(second.proof_auxiliary_dispatches, 2);
    assert_eq!(second.timing_submissions, 0);
    assert_eq!(second.calibration_submissions, 2);
    assert_eq!(second.decisions[0].proof_submissions, 1);
    assert_eq!(second.decisions[0].timing_submissions, 0);
    assert_eq!(second.decisions[0].calibration_submissions, 1);
    assert!(second.decisions[0].process_cache_hit);
    assert_ne!(plan_a1.plan_id(), plan_a2.plan_id());

    let weight_b = weight(&device, 68, 160, 13);
    let mut registry_b = KernelRegistry::new();
    let (_plan_b, b) = calibrate_dense_q4_routes(
        &mut registry_b,
        &device,
        203,
        limits,
        &[DenseQ4CalibrationCase {
            weight: &weight_b,
            shape: base(68, 160),
            reachable_m: &[33],
        }],
    )
    .expect("B activation");
    assert_cleanup_boundary(&b);
    assert_eq!(b.declared_shapes, 1);

    let mut registry_a3 = KernelRegistry::new();
    let (plan_a3, third) = calibrate_dense_q4_routes(
        &mut registry_a3,
        &device,
        204,
        limits,
        &[DenseQ4CalibrationCase {
            weight: &borrowed_weight,
            shape: base(67, 160),
            reachable_m: &[33],
        }],
    )
    .expect("A reactivation");
    assert_cleanup_boundary(&third);
    assert_eq!(third.process_cache_hits, 1);
    assert_eq!(third.authorized_shape_weight_pairs, 1);
    assert_eq!(third.proof_submissions, 1);
    assert_eq!(third.proof_route_dispatches, 2);
    assert_eq!(third.proof_auxiliary_dispatches, 2);
    assert_eq!(third.timing_submissions, 0);
    assert_eq!(third.calibration_submissions, 2);
    assert_eq!(third.decisions[0].proof_submissions, 1);
    assert_eq!(third.decisions[0].timing_submissions, 0);
    assert_eq!(third.decisions[0].calibration_submissions, 1);
    assert!(third.decisions[0].process_cache_hit);
    assert_ne!(plan_a2.plan_id(), plan_a3.plan_id());

    drop(borrowed_weight);
    assert_eq!(plan_a3.activation_epoch(), 204);
    assert_eq!(plan_a3.decision_count(), 1);
}

#[test]
fn same_shape_distinct_weights_are_all_proved_but_timed_once() {
    let device = MlxDevice::new().expect("Metal device");
    let first_weight = weight(&device, 69, 160, 29);
    let second_weight = weight(&device, 69, 160, 31);
    assert_ne!(first_weight.contents_ptr(), second_weight.contents_ptr());
    assert_ne!(
        first_weight.as_slice::<u8>().expect("first Q4 bytes"),
        second_weight.as_slice::<u8>().expect("second Q4 bytes")
    );
    let mut registry = KernelRegistry::new();
    let (_plan, receipt) = calibrate_dense_q4_routes(
        &mut registry,
        &device,
        251,
        DenseQ4CalibrationLimits {
            max_elapsed_ms: 20_000,
            max_shapes: 1,
        },
        &[
            DenseQ4CalibrationCase {
                weight: &first_weight,
                shape: base(69, 160),
                reachable_m: &[35],
            },
            DenseQ4CalibrationCase {
                weight: &second_weight,
                shape: base(69, 160),
                reachable_m: &[35],
            },
        ],
    )
    .expect("calibrate both same-shape weights");

    assert_eq!(receipt.declared_shapes, 1);
    assert_eq!(receipt.calibrated_decisions, 1);
    assert_eq!(receipt.process_cache_hits, 0);
    assert_eq!(receipt.decisions.len(), 1);
    assert_eq!(receipt.decisions[0].timings.len(), 2);
    assert_eq!(receipt.authorized_shape_weight_pairs, 2);
    assert_eq!(receipt.proof_submissions, 1);
    assert_eq!(receipt.proof_route_dispatches, 4);
    assert_eq!(receipt.proof_auxiliary_dispatches, 4);
    assert_eq!(receipt.timing_submissions, 10);
    assert_eq!(receipt.cleanup_submissions, 1);
    assert_eq!(receipt.decisions[0].authorized_weight_buffers, 2);
    assert_eq!(receipt.decisions[0].proof_submissions, 1);
    assert_eq!(receipt.decisions[0].proof_route_dispatches, 4);
    assert_eq!(receipt.decisions[0].proof_auxiliary_dispatches, 4);
    assert_eq!(receipt.decisions[0].timing_submissions, 10);
    assert_eq!(receipt.decisions[0].calibration_submissions, 11);
    assert_eq!(receipt.calibration_submissions, 12);
    assert_cleanup_boundary(&receipt);
}

#[test]
fn every_exact_shape_weight_pair_is_proved_in_batched_submissions() {
    let device = MlxDevice::new().expect("Metal device");
    let first_weight = weight(&device, 73, 192, 37);
    let second_weight = weight(&device, 73, 192, 41);
    let mut registry = KernelRegistry::new();
    let (_plan, receipt) = calibrate_dense_q4_routes(
        &mut registry,
        &device,
        253,
        DenseQ4CalibrationLimits {
            max_elapsed_ms: 20_000,
            max_shapes: 2,
        },
        &[
            DenseQ4CalibrationCase {
                weight: &first_weight,
                shape: base(73, 192),
                reachable_m: &[37, 9],
            },
            DenseQ4CalibrationCase {
                weight: &second_weight,
                shape: base(73, 192),
                reachable_m: &[9, 37],
            },
        ],
    )
    .expect("factorized exact-geometry/current-weight calibration");

    assert_eq!(receipt.declared_shapes, 2);
    assert_eq!(receipt.authorized_shape_weight_pairs, 4);
    assert_eq!(receipt.proof_submissions, 2);
    assert_eq!(receipt.proof_route_dispatches, 8);
    assert_eq!(receipt.proof_auxiliary_dispatches, 8);
    assert_eq!(receipt.timing_submissions, 20);
    assert_eq!(receipt.cleanup_submissions, 1);
    assert_eq!(receipt.calibration_submissions, 23);
    assert_eq!(receipt.decisions.len(), 2);
    assert_eq!(receipt.decisions[0].shape.m, 9);
    assert_eq!(receipt.decisions[0].authorized_weight_buffers, 2);
    assert_eq!(receipt.decisions[0].proof_submissions, 1);
    assert_eq!(receipt.decisions[0].proof_route_dispatches, 4);
    assert_eq!(receipt.decisions[0].proof_auxiliary_dispatches, 4);
    assert_eq!(receipt.decisions[1].shape.m, 37);
    assert_eq!(receipt.decisions[1].authorized_weight_buffers, 2);
    assert_eq!(receipt.decisions[1].proof_submissions, 1);
    assert_eq!(receipt.decisions[1].proof_route_dispatches, 4);
    assert_eq!(receipt.decisions[1].proof_auxiliary_dispatches, 4);
    assert_eq!(receipt.decisions[0].timing_submissions, 10);
    assert_eq!(receipt.decisions[1].timing_submissions, 10);
    assert_cleanup_boundary(&receipt);
}

#[test]
fn required_v2_nonfinite_output_hard_fails_without_frozen_plan() {
    let device = MlxDevice::new().expect("Metal device");
    let weight = nan_scale_weight(&device, 71, 160);
    let mut registry = KernelRegistry::new();
    let error = calibrate_dense_q4_routes(
        &mut registry,
        &device,
        252,
        DenseQ4CalibrationLimits {
            max_elapsed_ms: 20_000,
            max_shapes: 1,
        },
        &[DenseQ4CalibrationCase {
            weight: &weight,
            shape: base(71, 160),
            reachable_m: &[35],
        }],
    )
    .expect_err("non-finite required V2 output must abort calibration");

    assert!(error
        .to_string()
        .contains("required dense Q4 compatibility V2 proof failed"));
    assert!(error.to_string().contains("non-finite"));
    assert!(registry.dense_q4_plan().is_none());
}

#[test]
fn invalid_required_v2_and_oversized_native_weight_fail_before_freeze() {
    let device = MlxDevice::new().expect("Metal device");
    let exact_weight = weight(&device, 67, 160, 17);
    let case = DenseQ4CalibrationCase {
        weight: &exact_weight,
        shape: base(67, 160),
        reachable_m: &[33],
    };

    let mut invalid_v2 = KernelRegistry::new();
    invalid_v2
        .try_register_source(
            "kernel_mul_mm_q4_0_tensor_v2_f32",
            "kernel void deliberately_invalid_v2(",
        )
        .expect("register invalid required V2 source");
    let error = calibrate_dense_q4_routes(
        &mut invalid_v2,
        &device,
        301,
        DenseQ4CalibrationLimits::default(),
        &[case],
    )
    .expect_err("required V2 preparation must fail");
    assert!(error
        .to_string()
        .contains("required dense Q4 compatibility V2 pipeline is unavailable"));
    assert!(invalid_v2.dense_q4_plan().is_none());

    let oversized_weight = weight(&device, 68, 160, 19);
    let mut oversized = KernelRegistry::new();
    let error = calibrate_dense_q4_routes(
        &mut oversized,
        &device,
        302,
        DenseQ4CalibrationLimits::default(),
        &[DenseQ4CalibrationCase {
            weight: &oversized_weight,
            shape: base(67, 160),
            reachable_m: &[33],
        }],
    )
    .expect_err("oversized native Q4 weight must fail");
    assert!(error.to_string().contains("must be exactly"));
    assert!(oversized.dense_q4_plan().is_none());
}
