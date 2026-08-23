//! Activation-plan tests for native scalar expert-ID matmul.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use std::sync::{Arc, Mutex, OnceLock};

use half::{bf16, f16};
use mlx_native::{
    calibrate_dense_matmul_id_routes, dense_matmul_id, dense_matmul_id_auto,
    dense_matmul_id_value_independence_theorem_sha256, dispatch_count, reset_counters,
    resolve_dense_matmul_id_auto_route, trace_dense_matmul_id_auto, DType,
    DenseMatmulIdCalibrationBatchReceipt, DenseMatmulIdCalibrationCase,
    DenseMatmulIdCalibrationLimits, DenseMatmulIdDecisionSource, DenseMatmulIdDispatchTrace,
    DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams, DenseMatmulIdRoute,
    DenseMatmulIdRoutingProfile, DenseMatmulIdScratch, DenseMatmulIdSelectionStatus,
    DenseMatmulIdShape, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice,
    DENSE_MATMUL_ID_VALUE_INDEPENDENCE_THEOREM,
};

fn calibration_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn weight(
    device: &MlxDevice,
    n: u32,
    k: u32,
    n_experts: u32,
    seed: usize,
) -> (Arc<MlxBuffer>, u64) {
    scalar_weight(device, DType::BF16, n, k, n_experts, seed)
}

fn scalar_weight(
    device: &MlxDevice,
    dtype: DType,
    n: u32,
    k: u32,
    n_experts: u32,
    seed: usize,
) -> (Arc<MlxBuffer>, u64) {
    let scalar_bytes = dtype.size_of() as u64;
    let matrix_elements = u64::from(n) * u64::from(k);
    let matrix_bytes = matrix_elements * scalar_bytes;
    let stride = matrix_bytes + scalar_bytes * 7;
    let bytes = (u64::from(n_experts - 1) * stride + matrix_bytes) as usize;
    let mut buffer = device
        .alloc_buffer(bytes, dtype, vec![bytes / scalar_bytes as usize])
        .expect("weight");
    buffer.as_mut_slice::<u8>().unwrap().fill(0xD7);
    let value = |expert: usize, index: usize| {
        (((index * 29 + expert * 71 + seed * 17) % 1021) as f32 - 510.0) / 4096.0
    };
    match dtype {
        DType::BF16 => {
            let words = buffer.as_mut_slice::<u16>().unwrap();
            let stride_elements = stride as usize / 2;
            for expert in 0..n_experts as usize {
                for index in 0..matrix_elements as usize {
                    words[expert * stride_elements + index] =
                        bf16::from_f32(value(expert, index)).to_bits();
                }
            }
        }
        DType::F16 => {
            let words = buffer.as_mut_slice::<u16>().unwrap();
            let stride_elements = stride as usize / 2;
            for expert in 0..n_experts as usize {
                for index in 0..matrix_elements as usize {
                    words[expert * stride_elements + index] =
                        f16::from_f32(value(expert, index)).to_bits();
                }
            }
        }
        DType::F32 => {
            let values = buffer.as_mut_slice::<f32>().unwrap();
            let stride_elements = stride as usize / 4;
            for expert in 0..n_experts as usize {
                for index in 0..matrix_elements as usize {
                    values[expert * stride_elements + index] = value(expert, index);
                }
            }
        }
        _ => unreachable!("native scalar test dtype"),
    }
    (Arc::new(buffer), stride)
}

fn params(
    m: u32,
    n: u32,
    k: u32,
    n_experts: u32,
    stride: u64,
    layout: DenseMatmulIdInputLayout,
    multiplicity: DenseMatmulIdMultiplicity,
) -> DenseMatmulIdParams {
    DenseMatmulIdParams {
        m,
        n,
        k,
        top_k: 6.min(n_experts),
        n_experts,
        expert_stride_bytes: stride,
        input_layout: layout,
        id_multiplicity: multiplicity,
        // Calibration owns this field and auto dispatch ignores it.
        route: DenseMatmulIdRoute::Direct,
    }
}

fn shape(weight: &MlxBuffer, params: &DenseMatmulIdParams) -> DenseMatmulIdShape {
    DenseMatmulIdShape {
        weight_dtype: weight.dtype(),
        m: params.m,
        n: params.n,
        k: params.k,
        top_k: params.top_k,
        n_experts: params.n_experts,
        expert_stride_bytes: params.expert_stride_bytes,
        input_layout: params.input_layout,
        id_multiplicity: params.id_multiplicity,
    }
}

struct CallBuffers {
    input: MlxBuffer,
    ids: MlxBuffer,
    output: MlxBuffer,
    scratch: DenseMatmulIdScratch,
}

fn call_buffers(device: &MlxDevice, params: &DenseMatmulIdParams) -> CallBuffers {
    let input_rows = match params.input_layout {
        DenseMatmulIdInputLayout::SharedPerToken => params.m,
        DenseMatmulIdInputLayout::Slotted => params.m * params.top_k,
    } as usize;
    let mut input = device
        .alloc_buffer(
            input_rows * params.k as usize * 4,
            DType::F32,
            vec![input_rows, params.k as usize],
        )
        .unwrap();
    for (index, value) in input.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *value = ((index * 31 % 251) as f32 - 125.0) / 1003.0;
    }
    let mut ids = device
        .alloc_buffer(
            params.m as usize * params.top_k as usize * 4,
            DType::U32,
            vec![params.m as usize, params.top_k as usize],
        )
        .unwrap();
    for token in 0..params.m as usize {
        for slot in 0..params.top_k as usize {
            ids.as_mut_slice::<u32>().unwrap()[token * params.top_k as usize + slot] =
                ((token + slot) % params.n_experts as usize) as u32;
        }
    }
    CallBuffers {
        input,
        ids,
        output: device
            .alloc_buffer(
                params.m as usize * params.top_k as usize * params.n as usize * 4,
                DType::F32,
                vec![params.m as usize, params.top_k as usize, params.n as usize],
            )
            .unwrap(),
        scratch: DenseMatmulIdScratch::new(device, params.n_experts, params.m).unwrap(),
    }
}

fn limits() -> DenseMatmulIdCalibrationLimits {
    DenseMatmulIdCalibrationLimits {
        max_elapsed_ms: 5_000,
        max_cases: 8,
        max_submissions: 128,
    }
}

#[test]
fn activation_plan_is_one_shot_pointer_free_and_reusable_across_epochs_and_swaps() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");

    // Union/freeze: duplicate declarations collapse to one exact shape.
    let (union_weight, union_stride) = weight(&device, 11, 37, 8, 1);
    let (union_weight_other, union_stride_other) = weight(&device, 11, 37, 8, 2);
    assert_eq!(union_stride_other, union_stride);
    assert_ne!(
        union_weight.as_slice::<u8>().unwrap(),
        union_weight_other.as_slice::<u8>().unwrap()
    );
    let union_params = params(
        9,
        11,
        37,
        8,
        union_stride,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let union_cases = [
        DenseMatmulIdCalibrationCase {
            weight: &union_weight,
            params: union_params,
        },
        DenseMatmulIdCalibrationCase {
            weight: &union_weight,
            params: union_params,
        },
        DenseMatmulIdCalibrationCase {
            weight: &union_weight_other,
            params: union_params,
        },
    ];
    let mut union_registry = KernelRegistry::new();
    let (union_plan, union_receipt) = calibrate_dense_matmul_id_routes(
        &mut union_registry,
        &device,
        10_001,
        limits(),
        &union_cases,
    )
    .expect("union calibration");
    assert_eq!(union_plan.decision_count(), 1);
    assert_eq!(union_receipt.declared_cases, 1);
    assert_eq!(union_receipt.declared_weight_identities, 2);
    assert_eq!(
        union_receipt.value_independence_theorem_sha256,
        dense_matmul_id_value_independence_theorem_sha256()
    );
    assert_eq!(
        union_plan.value_independence_theorem_sha256(),
        dense_matmul_id_value_independence_theorem_sha256()
    );
    assert!(DENSE_MATMUL_ID_VALUE_INDEPENDENCE_THEOREM.contains("no-cross-dtype"));
    assert_eq!(
        union_receipt.activation_authority_digest,
        union_plan.activation_authority_digest()
    );
    assert_eq!(union_receipt.decisions[0].declared_weight_identities, 2);
    assert_eq!(union_receipt.cleanup_submissions, 1);
    assert!(union_receipt.calibration_submissions <= limits().max_submissions);
    assert!(union_receipt.deadline_overrun_ms <= 1.0);
    assert_eq!(union_receipt.decisions[0].timings.len(), 4);
    if union_receipt.decisions[0].selected_route == DenseMatmulIdRoute::GroupedPrefill {
        assert_eq!(union_receipt.calibration_submissions, 25);
        assert_eq!(union_receipt.calibration_dispatches, 36);
        assert_eq!(union_receipt.empirical_shape_proof_submissions, 4);
        assert_eq!(union_receipt.empirical_shape_proof_dispatches, 6);
        assert_eq!(union_receipt.theorem_authorized_weight_identities, 2);
    } else {
        assert_eq!(union_receipt.calibration_submissions, 25);
        assert_eq!(union_receipt.calibration_dispatches, 36);
        assert_eq!(union_receipt.empirical_shape_proof_submissions, 4);
        assert_eq!(union_receipt.empirical_shape_proof_dispatches, 6);
        assert_eq!(union_receipt.theorem_authorized_weight_identities, 0);
    }
    assert_eq!(union_receipt.current_timing_submissions, 20);
    assert_eq!(union_receipt.current_timing_dispatches, 30);
    assert_eq!(union_receipt.cached_timing_submissions, 0);
    assert_eq!(union_receipt.cached_timing_dispatches, 0);
    assert_eq!(
        union_receipt.decisions[0]
            .timings
            .iter()
            .map(|timing| timing.profile)
            .collect::<Vec<_>>(),
        vec![
            DenseMatmulIdRoutingProfile::Balanced,
            DenseMatmulIdRoutingProfile::Balanced,
            DenseMatmulIdRoutingProfile::MaximallySkewedDistinct,
            DenseMatmulIdRoutingProfile::MaximallySkewedDistinct,
        ]
    );
    assert_eq!(
        union_receipt.decisions[0]
            .timings
            .iter()
            .map(|timing| timing.route)
            .collect::<Vec<_>>(),
        vec![
            DenseMatmulIdRoute::Direct,
            DenseMatmulIdRoute::GroupedPrefill,
            DenseMatmulIdRoute::Direct,
            DenseMatmulIdRoute::GroupedPrefill,
        ]
    );
    for timing in &union_receipt.decisions[0].timings {
        let expected_dispatches = if timing.route == DenseMatmulIdRoute::Direct {
            1
        } else {
            2
        };
        assert_eq!(timing.encoded.len(), expected_dispatches);
        assert_eq!(timing.pipelines.len(), expected_dispatches);
        assert!(timing
            .encoded
            .iter()
            .zip(&timing.pipelines)
            .all(|(dispatch, pipeline)| dispatch.pipeline_label == pipeline.pipeline_label));
    }
    println!("activation union receipt: {union_receipt:?}");

    // Read-only route resolution validates the same plan/weight contract and
    // cannot mutate an encoder or submit GPU work, on success or failure.
    reset_counters();
    let resolved = resolve_dense_matmul_id_auto_route(
        &union_registry,
        &device,
        10_001,
        &union_weight,
        &union_params,
    )
    .expect("read-only route resolution");
    assert_eq!(resolved.1, DenseMatmulIdDecisionSource::FrozenPlan);
    assert_eq!(dispatch_count(), 0);
    assert!(resolve_dense_matmul_id_auto_route(
        &union_registry,
        &device,
        10_002,
        &union_weight,
        &union_params,
    )
    .is_err());
    let short_weight = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("short weight");
    assert!(resolve_dense_matmul_id_auto_route(
        &union_registry,
        &device,
        10_001,
        &short_weight,
        &union_params,
    )
    .is_err());
    assert_eq!(dispatch_count(), 0);

    // A secondary execution registry may reuse the exact selected plan only
    // for the same activation epoch and borrowed logical weight-identity set.
    let mut worker_registry = KernelRegistry::new();
    worker_registry
        .freeze_dense_matmul_id_plan_for_cases(&device, 10_001, union_plan.clone(), &union_cases)
        .expect("same-activation worker plan freeze");
    assert_eq!(
        worker_registry
            .dense_matmul_id_plan()
            .expect("worker plan")
            .plan_id(),
        union_plan.plan_id()
    );
    let mut wrong_epoch_registry = KernelRegistry::new();
    assert!(wrong_epoch_registry
        .freeze_dense_matmul_id_plan_for_cases(&device, 10_002, union_plan.clone(), &union_cases,)
        .is_err());
    let (model_b_weight, model_b_stride) = weight(&device, 11, 37, 8, 777);
    let (model_b_weight_other, model_b_stride_other) = weight(&device, 11, 37, 8, 778);
    assert_eq!(model_b_stride, union_stride);
    assert_eq!(model_b_stride_other, union_stride);
    let model_b_cases = [
        DenseMatmulIdCalibrationCase {
            weight: &model_b_weight,
            params: union_params,
        },
        DenseMatmulIdCalibrationCase {
            weight: &model_b_weight_other,
            params: union_params,
        },
    ];
    let mut model_b_registry = KernelRegistry::new();
    assert!(model_b_registry
        .freeze_dense_matmul_id_plan_for_cases(&device, 10_001, union_plan.clone(), &model_b_cases,)
        .is_err());

    assert!(calibrate_dense_matmul_id_routes(
        &mut union_registry,
        &device,
        10_001,
        limits(),
        &union_cases,
    )
    .is_err());

    // A declared shape executes, a same-base late width safely uses Direct,
    // and a new epoch still fails closed.
    let union_buffers = call_buffers(&device, &union_params);
    let executor = GraphExecutor::new(device.clone());
    let mut session = executor.begin().unwrap();
    let dispatch = session
        .dense_matmul_id_auto(
            &mut union_registry,
            &device,
            10_001,
            &union_weight,
            &union_buffers.input,
            &union_buffers.ids,
            &union_buffers.output,
            Some(&union_buffers.scratch),
            &union_params,
        )
        .expect("declared graph auto dispatch");
    session.finish().unwrap();
    assert_eq!(
        dispatch.route,
        union_plan
            .route_for(shape(&union_weight, &union_params))
            .unwrap()
    );
    assert_eq!(
        dispatch.decision_source,
        DenseMatmulIdDecisionSource::FrozenPlan
    );
    assert!(union_buffers
        .output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    let union_other_buffers = call_buffers(&device, &union_params);
    let mut union_other_encoder = device.command_encoder().unwrap();
    let union_other_dispatch = dense_matmul_id_auto(
        &mut union_other_encoder,
        &mut union_registry,
        &device,
        10_001,
        &union_weight_other,
        &union_other_buffers.input,
        &union_other_buffers.ids,
        &union_other_buffers.output,
        Some(&union_other_buffers.scratch),
        &union_params,
    )
    .expect("second same-shape current weight auto dispatch");
    union_other_encoder.commit_and_wait().unwrap();
    assert_eq!(
        union_other_dispatch.decision_source,
        DenseMatmulIdDecisionSource::FrozenPlan
    );
    let union_other_direct = call_buffers(&device, &union_params);
    let mut union_other_direct_encoder = device.command_encoder().unwrap();
    dense_matmul_id(
        &mut union_other_direct_encoder,
        &mut union_registry,
        &device,
        &union_weight_other,
        &union_other_direct.input,
        &union_other_direct.ids,
        &union_other_direct.output,
        Some(&union_other_direct.scratch),
        &DenseMatmulIdParams {
            route: DenseMatmulIdRoute::Direct,
            ..union_params
        },
    )
    .expect("second same-shape current weight Direct reference");
    union_other_direct_encoder.commit_and_wait().unwrap();
    assert_eq!(
        union_other_buffers.output.as_slice::<u32>().unwrap(),
        union_other_direct.output.as_slice::<u32>().unwrap()
    );
    let mut wrong_epoch_encoder = device.command_encoder().unwrap();
    assert!(dense_matmul_id_auto(
        &mut wrong_epoch_encoder,
        &mut union_registry,
        &device,
        10_002,
        &union_weight,
        &union_buffers.input,
        &union_buffers.ids,
        &union_buffers.output,
        Some(&union_buffers.scratch),
        &union_params,
    )
    .is_err());
    let late = DenseMatmulIdParams {
        m: 10,
        ..union_params
    };
    let late_buffers = call_buffers(&device, &late);
    let mut late_encoder = device.command_encoder().unwrap();
    let late_receipt = dense_matmul_id_auto(
        &mut late_encoder,
        &mut union_registry,
        &device,
        10_001,
        &union_weight,
        &late_buffers.input,
        &late_buffers.ids,
        &late_buffers.output,
        Some(&late_buffers.scratch),
        &late,
    )
    .expect("same-base undeclared width");
    late_encoder.commit_and_wait().unwrap();
    assert_eq!(late_receipt.route, DenseMatmulIdRoute::Direct);
    assert_eq!(
        late_receipt.decision_source,
        DenseMatmulIdDecisionSource::UndeclaredDirect
    );
    let unseen_base = DenseMatmulIdParams {
        input_layout: DenseMatmulIdInputLayout::Slotted,
        ..late
    };
    let unseen_buffers = call_buffers(&device, &unseen_base);
    let mut unseen_encoder = device.command_encoder().unwrap();
    assert!(dense_matmul_id_auto(
        &mut unseen_encoder,
        &mut union_registry,
        &device,
        10_001,
        &union_weight,
        &unseen_buffers.input,
        &unseen_buffers.ids,
        &unseen_buffers.output,
        Some(&unseen_buffers.scratch),
        &unseen_base,
    )
    .is_err());

    // Same address, new epoch: metadata is reused, while the plan identity is
    // activation-scoped and no timing submission is repeated.
    let mut same_registry = KernelRegistry::new();
    let same_case = [DenseMatmulIdCalibrationCase {
        weight: &union_weight,
        params: union_params,
    }];
    let (same_plan, same_receipt) =
        calibrate_dense_matmul_id_routes(&mut same_registry, &device, 10_002, limits(), &same_case)
            .expect("same-address new-epoch calibration");
    assert_eq!(same_receipt.process_cache_hits, 1);
    if same_receipt.decisions[0].selected_route == DenseMatmulIdRoute::GroupedPrefill {
        assert_eq!(same_receipt.empirical_shape_proof_submissions, 4);
        assert_eq!(same_receipt.empirical_shape_proof_dispatches, 6);
        assert_eq!(same_receipt.calibration_submissions, 5);
        assert_eq!(same_receipt.theorem_authorized_weight_identities, 1);
        assert_eq!(same_receipt.cleanup_submissions, 1);
    } else {
        assert_eq!(same_receipt.empirical_shape_proof_submissions, 0);
        assert_eq!(same_receipt.empirical_shape_proof_dispatches, 0);
        assert_eq!(same_receipt.calibration_submissions, 0);
        assert_eq!(same_receipt.theorem_authorized_weight_identities, 0);
        assert_eq!(same_receipt.cleanup_submissions, 0);
    }
    assert_eq!(same_receipt.current_timing_submissions, 0);
    assert_eq!(same_receipt.cached_timing_submissions, 20);
    assert_eq!(same_receipt.cached_timing_dispatches, 30);
    assert_ne!(union_plan.plan_id(), same_plan.plan_id());

    // A -> B -> A with a new A allocation reuses pointer-free timing metadata.
    let (weight_a, stride_a) = weight(&device, 17, 41, 8, 2);
    let weight_a_bytes = weight_a.as_slice::<u8>().unwrap().to_vec();
    let weak_a = Arc::downgrade(&weight_a);
    let params_a = params(
        9,
        17,
        41,
        8,
        stride_a,
        DenseMatmulIdInputLayout::Slotted,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let mut registry_a = KernelRegistry::new();
    let case_a = [DenseMatmulIdCalibrationCase {
        weight: &weight_a,
        params: params_a,
    }];
    let (_plan_a, receipt_a) =
        calibrate_dense_matmul_id_routes(&mut registry_a, &device, 20_001, limits(), &case_a)
            .expect("A calibration");
    assert_eq!(receipt_a.process_cache_hits, 0);
    drop(case_a);
    drop(weight_a);
    assert!(
        weak_a.upgrade().is_none(),
        "plan/cache retained model weight A"
    );

    let (weight_b, stride_b) = weight(&device, 19, 43, 8, 3);
    let params_b = params(
        9,
        19,
        43,
        8,
        stride_b,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let mut registry_b = KernelRegistry::new();
    let case_b = [DenseMatmulIdCalibrationCase {
        weight: &weight_b,
        params: params_b,
    }];
    calibrate_dense_matmul_id_routes(&mut registry_b, &device, 20_002, limits(), &case_b)
        .expect("B calibration");

    let (weight_a2, stride_a2) = weight(&device, 17, 41, 8, 99);
    assert_eq!(stride_a, stride_a2);
    let mut registry_a2 = KernelRegistry::new();
    let case_a2 = [DenseMatmulIdCalibrationCase {
        weight: &weight_a2,
        params: params_a,
    }];
    let (_plan_a2, receipt_a2) =
        calibrate_dense_matmul_id_routes(&mut registry_a2, &device, 20_003, limits(), &case_a2)
            .expect("A2 calibration");
    assert_eq!(receipt_a2.process_cache_hits, 1);
    assert_ne!(
        weight_a2.as_slice::<u8>().unwrap(),
        weight_a_bytes,
        "A2 must use distinct bytes for the exact-shape representative proof"
    );
    let a2_buffers = call_buffers(&device, &params_a);
    let mut a2_encoder = device.command_encoder().unwrap();
    let a2_dispatch = dense_matmul_id_auto(
        &mut a2_encoder,
        &mut registry_a2,
        &device,
        20_003,
        &weight_a2,
        &a2_buffers.input,
        &a2_buffers.ids,
        &a2_buffers.output,
        Some(&a2_buffers.scratch),
        &params_a,
    )
    .expect("A2 auto dispatch");
    a2_encoder.commit_and_wait().unwrap();
    assert_eq!(
        a2_dispatch.decision_source,
        DenseMatmulIdDecisionSource::FrozenPlan
    );
    let direct_buffers = call_buffers(&device, &params_a);
    let direct_params = DenseMatmulIdParams {
        route: DenseMatmulIdRoute::Direct,
        ..params_a
    };
    let mut direct_encoder = device.command_encoder().unwrap();
    dense_matmul_id(
        &mut direct_encoder,
        &mut registry_a2,
        &device,
        &weight_a2,
        &direct_buffers.input,
        &direct_buffers.ids,
        &direct_buffers.output,
        Some(&direct_buffers.scratch),
        &direct_params,
    )
    .expect("A2 Direct reference");
    direct_encoder.commit_and_wait().unwrap();
    assert_eq!(
        a2_buffers.output.as_slice::<u32>().unwrap(),
        direct_buffers.output.as_slice::<u32>().unwrap(),
        "A2 cached-route output must be bit-identical to Direct on current bytes"
    );
    if receipt_a2.decisions[0].selected_route == DenseMatmulIdRoute::GroupedPrefill {
        assert_eq!(receipt_a2.empirical_shape_proof_submissions, 4);
        assert_eq!(receipt_a2.empirical_shape_proof_dispatches, 6);
        assert_eq!(receipt_a2.calibration_submissions, 5);
        assert_eq!(receipt_a2.theorem_authorized_weight_identities, 1);
        assert_eq!(receipt_a2.cleanup_submissions, 1);
    } else {
        assert_eq!(receipt_a2.empirical_shape_proof_submissions, 0);
        assert_eq!(receipt_a2.empirical_shape_proof_dispatches, 0);
        assert_eq!(receipt_a2.calibration_submissions, 0);
        assert_eq!(receipt_a2.theorem_authorized_weight_identities, 0);
        assert_eq!(receipt_a2.cleanup_submissions, 0);
    }
    assert_eq!(receipt_a2.current_timing_submissions, 0);
    assert_eq!(receipt_a2.cached_timing_submissions, 20);
    assert_eq!(receipt_a2.cached_timing_dispatches, 30);
}

#[test]
fn calibration_proves_every_native_scalar_dtype_before_route_selection() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");
    for (index, dtype) in [DType::F32, DType::F16, DType::BF16]
        .into_iter()
        .enumerate()
    {
        let (native_weight, stride) = scalar_weight(&device, dtype, 23, 35, 8, 101 + index);
        let native_params = params(
            9,
            23,
            35,
            8,
            stride,
            DenseMatmulIdInputLayout::Slotted,
            DenseMatmulIdMultiplicity::DistinctPerToken,
        );
        let mut registry = KernelRegistry::new();
        let (_plan, receipt) = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            40_001 + index as u64,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 1_000,
                max_cases: 1,
                max_submissions: 25,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &native_weight,
                params: native_params,
            }],
        )
        .expect("native dtype calibration");
        assert_eq!(receipt.decisions[0].shape.weight_dtype, dtype);
        assert_eq!(receipt.decisions[0].timings.len(), 4);
        assert!(!matches!(
            receipt.decisions[0].status,
            DenseMatmulIdSelectionStatus::IncoherentGrouped
                | DenseMatmulIdSelectionStatus::BudgetFallback
                | DenseMatmulIdSelectionStatus::ErrorFallback
        ));
        assert_eq!(receipt.calibration_submissions, 25);
        assert_eq!(receipt.calibration_dispatches, 36);
        assert_eq!(receipt.cleanup_submissions, 1);
        assert_eq!(receipt.deadline_overrun_ms, 0.0);
        println!("native dtype {dtype:?} calibration receipt: {receipt:?}");
    }
}

#[test]
fn admitted_base_allows_only_unseen_m_across_scalar_dtypes_and_layouts() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");
    for (dtype_index, dtype) in [DType::F32, DType::F16, DType::BF16]
        .into_iter()
        .enumerate()
    {
        for (layout_index, layout) in [
            DenseMatmulIdInputLayout::SharedPerToken,
            DenseMatmulIdInputLayout::Slotted,
        ]
        .into_iter()
        .enumerate()
        {
            let (weights, stride) = scalar_weight(
                &device,
                dtype,
                5,
                13,
                4,
                700 + dtype_index * 2 + layout_index,
            );
            let declared = params(
                2,
                5,
                13,
                4,
                stride,
                layout,
                DenseMatmulIdMultiplicity::MayRepeat,
            );
            if dtype_index == 0 && layout_index == 0 {
                let mut empty_registry = KernelRegistry::new();
                let buffers = call_buffers(&device, &declared);
                let mut encoder = device.command_encoder().unwrap();
                assert!(dense_matmul_id_auto(
                    &mut encoder,
                    &mut empty_registry,
                    &device,
                    60_000,
                    &weights,
                    &buffers.input,
                    &buffers.ids,
                    &buffers.output,
                    Some(&buffers.scratch),
                    &declared,
                )
                .is_err());
            }
            let mut registry = KernelRegistry::new();
            calibrate_dense_matmul_id_routes(
                &mut registry,
                &device,
                60_000 + (dtype_index * 2 + layout_index) as u64,
                DenseMatmulIdCalibrationLimits {
                    max_elapsed_ms: 1_000,
                    max_cases: 1,
                    max_submissions: 1,
                },
                &[DenseMatmulIdCalibrationCase {
                    weight: &weights,
                    params: declared,
                }],
            )
            .expect("direct-only base admission");

            for (candidate, expected_source) in [
                (declared, DenseMatmulIdDecisionSource::FrozenPlan),
                (
                    DenseMatmulIdParams { m: 9, ..declared },
                    DenseMatmulIdDecisionSource::UndeclaredDirect,
                ),
            ] {
                let buffers = call_buffers(&device, &candidate);
                let mut encoder = device.command_encoder().unwrap();
                let receipt = dense_matmul_id_auto(
                    &mut encoder,
                    &mut registry,
                    &device,
                    60_000 + (dtype_index * 2 + layout_index) as u64,
                    &weights,
                    &buffers.input,
                    &buffers.ids,
                    &buffers.output,
                    Some(&buffers.scratch),
                    &candidate,
                )
                .expect("declared base width dispatch");
                encoder.commit_and_wait().unwrap();
                assert_eq!(receipt.route, DenseMatmulIdRoute::Direct);
                assert_eq!(receipt.decision_source, expected_source);
            }

            let rejected = DenseMatmulIdParams {
                top_k: declared.top_k - 1,
                ..declared
            };
            let buffers = call_buffers(&device, &rejected);
            let mut encoder = device.command_encoder().unwrap();
            assert!(dense_matmul_id_auto(
                &mut encoder,
                &mut registry,
                &device,
                60_000 + (dtype_index * 2 + layout_index) as u64,
                &weights,
                &buffers.input,
                &buffers.ids,
                &buffers.output,
                Some(&buffers.scratch),
                &rejected,
            )
            .is_err());
        }
    }
}

#[test]
fn illegal_grouped_and_exhausted_budget_freeze_direct_without_hidden_work() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");

    let (repeat_weight, repeat_stride) = weight(&device, 37, 49, 8, 4);
    let repeat_params = params(
        9,
        37,
        49,
        8,
        repeat_stride,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::MayRepeat,
    );
    let repeat_cases = [DenseMatmulIdCalibrationCase {
        weight: &repeat_weight,
        params: repeat_params,
    }];
    let mut repeat_registry = KernelRegistry::new();
    let (repeat_plan, repeat_receipt) = calibrate_dense_matmul_id_routes(
        &mut repeat_registry,
        &device,
        30_001,
        DenseMatmulIdCalibrationLimits {
            max_elapsed_ms: 1_000,
            max_cases: 1,
            max_submissions: 1,
        },
        &repeat_cases,
    )
    .expect("repeat direct-only calibration");
    assert_eq!(repeat_receipt.calibration_submissions, 0);
    assert_eq!(repeat_receipt.cleanup_submissions, 0);
    assert_eq!(repeat_receipt.calibrated_decisions, 0);
    assert_eq!(repeat_receipt.process_cache_hits, 0);
    assert_eq!(repeat_receipt.fallback_decisions, 1);
    assert_eq!(
        repeat_receipt.calibrated_decisions
            + repeat_receipt.process_cache_hits
            + repeat_receipt.fallback_decisions,
        repeat_receipt.declared_cases
    );
    assert_eq!(
        repeat_receipt.decisions[0].status,
        DenseMatmulIdSelectionStatus::DirectOnly
    );
    assert_eq!(
        repeat_plan.route_for(shape(&repeat_weight, &repeat_params)),
        Some(DenseMatmulIdRoute::Direct)
    );

    let (budget_weight, budget_stride) = weight(&device, 31, 47, 8, 5);
    let budget_params = params(
        9,
        31,
        47,
        8,
        budget_stride,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let budget_cases = [DenseMatmulIdCalibrationCase {
        weight: &budget_weight,
        params: budget_params,
    }];
    let mut budget_registry = KernelRegistry::new();
    let (budget_plan, budget_receipt) = calibrate_dense_matmul_id_routes(
        &mut budget_registry,
        &device,
        30_002,
        DenseMatmulIdCalibrationLimits {
            max_elapsed_ms: 1_000,
            max_cases: 1,
            max_submissions: 2,
        },
        &budget_cases,
    )
    .expect("budget fallback calibration");
    assert!(budget_receipt.calibration_submissions <= 2);
    assert_eq!(
        budget_receipt.decisions[0].status,
        DenseMatmulIdSelectionStatus::BudgetFallback
    );
    assert_eq!(
        budget_plan.route_for(shape(&budget_weight, &budget_params)),
        Some(DenseMatmulIdRoute::Direct)
    );
}

#[test]
fn nonfinite_required_direct_proof_hard_errors_without_publishing_a_plan() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");
    let (mut weights, stride) = scalar_weight(&device, DType::BF16, 29, 31, 8, 901);
    Arc::get_mut(&mut weights)
        .expect("unique test weight")
        .as_mut_slice::<u16>()
        .unwrap()
        .fill(bf16::from_f32(f32::NAN).to_bits());
    let params = params(
        7,
        29,
        31,
        8,
        stride,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let mut registry = KernelRegistry::new();
    let error = calibrate_dense_matmul_id_routes(
        &mut registry,
        &device,
        80_001,
        limits(),
        &[DenseMatmulIdCalibrationCase {
            weight: &weights,
            params,
        }],
    )
    .expect_err("nonfinite Direct proof must abort activation");
    assert!(
        error.to_string().contains("finitely overwritten"),
        "unexpected error: {error}"
    );
    assert!(
        registry.dense_matmul_id_plan().is_none(),
        "failed required Direct proof published a plan"
    );
}

#[test]
fn frozen_plan_trace_is_exact_serializable_and_protects_every_expert_shader() {
    let _serial = calibration_lock().lock().unwrap();
    let device = MlxDevice::new().expect("device");
    let (weights, stride) = weight(&device, 13, 39, 8, 501);
    let params = params(
        10,
        13,
        39,
        8,
        stride,
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdMultiplicity::DistinctPerToken,
    );
    let mut registry = KernelRegistry::new();
    let (plan, receipt) = calibrate_dense_matmul_id_routes(
        &mut registry,
        &device,
        50_001,
        limits(),
        &[DenseMatmulIdCalibrationCase {
            weight: &weights,
            params,
        }],
    )
    .expect("calibration");
    assert!(!receipt.pipeline_set_fingerprint.is_empty());
    assert!(!receipt.pipeline_identities.is_empty());
    let receipt_json = serde_json::to_string(&receipt).expect("serialize calibration receipt");
    let receipt_round_trip: DenseMatmulIdCalibrationBatchReceipt =
        serde_json::from_str(&receipt_json).expect("deserialize calibration receipt");
    assert_eq!(receipt_round_trip.schema_version, receipt.schema_version);
    assert_eq!(receipt_round_trip.plan_id, receipt.plan_id);
    assert_eq!(
        receipt_round_trip.empirical_shape_proof_submissions,
        receipt.empirical_shape_proof_submissions
    );
    assert_eq!(
        receipt_round_trip.cached_timing_submissions,
        receipt.cached_timing_submissions
    );
    assert_eq!(receipt_round_trip.decisions.len(), receipt.decisions.len());
    assert_eq!(
        receipt_round_trip.decisions[0].shape,
        receipt.decisions[0].shape
    );

    let buffers = call_buffers(&device, &params);
    let mut encoder = device.command_encoder().expect("encoder");
    let trace = trace_dense_matmul_id_auto(
        &mut encoder,
        &mut registry,
        &device,
        50_001,
        &weights,
        &buffers.input,
        &buffers.ids,
        &buffers.output,
        Some(&buffers.scratch),
        &params,
    )
    .expect("auto trace");
    encoder.commit_and_wait().expect("trace completion");
    assert_eq!(trace.device_name, device.name());
    assert_eq!(trace.device_registry_id, device.registry_id());
    assert_eq!(trace.plan_id.as_deref(), Some(plan.plan_id()));
    assert_eq!(
        trace.plan_value_independence_theorem_sha256.as_deref(),
        Some(dense_matmul_id_value_independence_theorem_sha256())
    );
    assert_eq!(
        trace.plan_activation_authority_digest.as_deref(),
        Some(plan.activation_authority_digest())
    );
    assert_eq!(trace.activation_epoch, Some(50_001));
    assert_eq!(
        trace.decision_source,
        Some(DenseMatmulIdDecisionSource::FrozenPlan)
    );
    assert_eq!(trace.encoded.len(), trace.pipelines.len());
    assert_eq!(
        trace.encoded.len(),
        if trace.route == DenseMatmulIdRoute::Direct {
            1
        } else {
            2
        }
    );
    let json = serde_json::to_string(&trace).expect("serialize trace");
    let round_trip: DenseMatmulIdDispatchTrace =
        serde_json::from_str(&json).expect("deserialize trace");
    assert_eq!(round_trip, trace);
    let mut value = serde_json::to_value(&trace).expect("trace value");
    value
        .as_object_mut()
        .expect("trace object")
        .insert("unexpected".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<DenseMatmulIdDispatchTrace>(value).is_err());

    const SOURCE: &str = include_str!("../src/shaders/dense_matmul_id.metal");
    for name in [
        "dense_matmul_id_direct_f32_f32",
        "dense_matmul_id_direct_f16_f32",
        "dense_matmul_id_direct_bf16_f32",
        "dense_matmul_id_map_distinct",
        "dense_matmul_id_grouped_f32_f32",
        "dense_matmul_id_grouped_f16_f32",
        "dense_matmul_id_grouped_bf16_f32",
    ] {
        registry
            .try_register_source(name, SOURCE)
            .expect("identical frozen source remains admissible");
        let changed: &'static str =
            Box::leak(format!("{SOURCE}\n// deliberate mutation for {name}").into_boxed_str());
        let error = registry
            .try_register_source(name, changed)
            .expect_err("frozen expert source mutation must fail");
        assert!(error.to_string().contains("after its route plan is frozen"));
    }
}
