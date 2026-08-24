use super::*;

fn params(m: u32, n: u32, k: u32, ggml_type: GgmlType) -> GgmlQuantizedMatmulParams {
    GgmlQuantizedMatmulParams { m, n, k, ggml_type }
}

#[test]
fn exact_shape_is_q4_single_batch_contiguous_mm_only() {
    let valid = params(9, 769, 800, GgmlType::Q4_0);
    assert_eq!(
        exact_shape(&valid, 1, true),
        Some(DenseQ4Shape {
            m: 9,
            n: 769,
            k: 800,
            batch: 1,
            input_layout: DenseQ4InputLayout::Contiguous,
        })
    );
    assert!(exact_shape(&params(8, 769, 800, GgmlType::Q4_0), 1, true).is_none());
    assert!(exact_shape(&params(9, 769, 801, GgmlType::Q4_0), 1, true).is_none());
    assert!(exact_shape(&params(9, 769, 800, GgmlType::Q8_0), 1, true).is_none());
    assert!(exact_shape(&valid, 2, true).is_none());
    assert!(exact_shape(&valid, 1, false).is_none());
}

#[test]
fn expected_geometry_covers_every_compatibility_and_candidate_tile() {
    let shape = DenseQ4Shape {
        m: 129,
        n: 769,
        k: 800,
        batch: 1,
        input_layout: DenseQ4InputLayout::Contiguous,
    };
    let simdgroup = expected_dispatch(DenseQ4Route::CompatibilitySimdgroup, shape);
    let tensor_v1 = expected_dispatch(DenseQ4Route::CompatibilityTensorV1, shape);
    let tensor_v2 = expected_dispatch(DenseQ4Route::CompatibilityV2, shape);
    let candidate = expected_dispatch(DenseQ4Route::Tensor64x32, shape);
    assert_eq!(simdgroup.grid, [5, 13, 1]);
    assert_eq!(tensor_v1.grid, [5, 13, 1]);
    assert_eq!(tensor_v2.grid, [2, 13, 1]);
    assert_eq!(candidate.grid, [5, 13, 1]);
    assert_eq!(simdgroup.threads_per_threadgroup, [128, 1, 1]);
    assert_eq!(tensor_v1.threads_per_threadgroup, [128, 1, 1]);
    assert_eq!(tensor_v2.threads_per_threadgroup, [128, 1, 1]);
    assert_eq!(candidate.threads_per_threadgroup, [128, 1, 1]);
    assert_eq!(simdgroup.threadgroup_memory, vec![(0, 8192)]);
    assert_eq!(tensor_v1.threadgroup_memory, vec![(0, 8192)]);
    assert_eq!(tensor_v2.threadgroup_memory, vec![(0, 4096)]);
    assert_eq!(candidate.threadgroup_memory, vec![(0, 4096)]);
    assert_ne!(simdgroup.pipeline_label, tensor_v1.pipeline_label);
    assert_ne!(tensor_v1.pipeline_label, tensor_v2.pipeline_label);
    assert_ne!(tensor_v2.pipeline_label, candidate.pipeline_label);
}

#[test]
fn route_plan_is_pointer_free_metadata() {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<DenseQ4RoutePlan>();
    assert_send_sync_static::<DenseQ4CalibrationDecision>();
}

fn acceptance_timing(route: DenseQ4Route, shape: DenseQ4Shape) -> DenseQ4RouteTiming {
    let distribution = DenseQ4TimingDistribution {
        p25_us: 90.0,
        median_us: 100.0,
        p75_us: 110.0,
        samples: 5,
    };
    DenseQ4RouteTiming {
        route,
        wall: distribution,
        gpu: distribution,
        encoded: expected_dispatch(route, shape),
        pipeline: KernelPipelineIdentity {
            schema_version: crate::kernel_registry::KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION,
            pipeline_label: route.pipeline_label(),
            kernel_name: route.kernel_name().into(),
            origin: crate::kernel_registry::KernelPipelineOrigin::RuntimeSource,
            runtime_source_sha256: Some("c".repeat(64)),
            embedded_metallib_sha256: None,
            precise_fp32_math: true,
            threadgroup_size_multiple_hint: true,
        },
    }
}

fn acceptance_receipt(epoch: u64, process_cache_hit: bool) -> DenseQ4CalibrationBatchReceipt {
    let short = DenseQ4Shape {
        m: 9,
        n: 768,
        k: 768,
        batch: 1,
        input_layout: DenseQ4InputLayout::Contiguous,
    };
    let long = DenseQ4Shape { m: 2048, ..short };
    let timing_submissions = if process_cache_hit { 0 } else { 10 };
    let decision = |shape, selected_route| DenseQ4CalibrationDecision {
        shape,
        selected_route,
        status: if selected_route == DenseQ4Route::Tensor64x32 {
            DenseQ4SelectionStatus::CalibratedWinner
        } else {
            DenseQ4SelectionStatus::CompatibilityFastest
        },
        diagnostic: None,
        timings: [DenseQ4Route::CompatibilityV2, DenseQ4Route::Tensor64x32]
            .into_iter()
            .map(|route| acceptance_timing(route, shape))
            .collect(),
        process_cache_hit,
        authorized_weight_buffers: 2,
        proof_submissions: 1,
        proof_route_dispatches: 4,
        proof_auxiliary_dispatches: 4,
        proof_scratch_bytes: 4096,
        proof_gpu_us: 100.0,
        timing_submissions,
        calibration_submissions: 1 + timing_submissions,
    };
    DenseQ4CalibrationBatchReceipt {
        schema_version: DENSE_Q4_ROUTE_SCHEMA_VERSION,
        mlx_native_version: env!("CARGO_PKG_VERSION").into(),
        build_fingerprint: "b".repeat(64),
        plan_id: format!("{epoch:064x}"),
        activation_epoch: epoch,
        device_name: "acceptance-device".into(),
        device_registry_id: 7,
        registry_authority_id: epoch,
        declared_shapes: 2,
        calibrated_decisions: if process_cache_hit { 0 } else { 2 },
        process_cache_hits: if process_cache_hit { 2 } else { 0 },
        compatibility_route_decisions: 1,
        authorized_shape_weight_pairs: 4,
        proof_submissions: 2,
        proof_route_dispatches: 8,
        proof_auxiliary_dispatches: 8,
        peak_proof_scratch_bytes: 4096,
        proof_gpu_us: 200.0,
        timing_submissions: timing_submissions * 2,
        cleanup_submissions: 1,
        calibration_submissions: 3 + timing_submissions * 2,
        elapsed_ms: 1000.0,
        deadline_overrun_ms: 0.0,
        decisions: vec![
            decision(short, DenseQ4Route::Tensor64x32),
            decision(long, DenseQ4Route::CompatibilityV2),
        ],
    }
}

fn acceptance_requirements() -> DenseQ4CartesianAcceptanceRequirements {
    DenseQ4CartesianAcceptanceRequirements {
        expected_base_shapes: 1,
        expected_weight_buffers_per_base: 2,
        reachable_m: vec![9, 2048],
        required_compatibility_m: vec![2048],
        minimum_candidate_decisions: 1,
        maximum_elapsed_ms: 15_000,
    }
}

#[test]
fn cartesian_acceptance_rejects_fallback_only_long_candidate_and_false_counts() {
    let cold = acceptance_receipt(11, false);
    let warm = acceptance_receipt(12, true);
    let requirements = acceptance_requirements();
    validate_dense_q4_cartesian_acceptance(&cold, &warm, &requirements)
        .expect("complete Cartesian evidence must pass");

    let mut fallback_only_cold = cold.clone();
    let mut fallback_only_warm = warm.clone();
    for receipt in [&mut fallback_only_cold, &mut fallback_only_warm] {
        receipt.decisions[0].selected_route = DenseQ4Route::CompatibilityV2;
        receipt.decisions[0].status = DenseQ4SelectionStatus::CompatibilityFastest;
        receipt.compatibility_route_decisions = 2;
    }
    assert!(validate_dense_q4_cartesian_acceptance(
        &fallback_only_cold,
        &fallback_only_warm,
        &requirements
    )
    .is_err());

    let mut fallback_status = cold.clone();
    fallback_status.decisions[1].status = DenseQ4SelectionStatus::CandidateUnavailable;
    fallback_status.decisions[1].diagnostic = Some("candidate pipeline unavailable".into());
    assert!(
        validate_dense_q4_cartesian_acceptance(&fallback_status, &warm, &requirements).is_err()
    );

    let mut long_candidate_cold = cold.clone();
    let mut long_candidate_warm = warm.clone();
    for receipt in [&mut long_candidate_cold, &mut long_candidate_warm] {
        receipt.decisions[1].selected_route = DenseQ4Route::Tensor64x32;
        receipt.decisions[1].status = DenseQ4SelectionStatus::CalibratedWinner;
        receipt.compatibility_route_decisions = 0;
    }
    assert!(validate_dense_q4_cartesian_acceptance(
        &long_candidate_cold,
        &long_candidate_warm,
        &requirements
    )
    .is_err());

    let mut false_counts = cold.clone();
    false_counts.authorized_shape_weight_pairs -= 1;
    assert!(validate_dense_q4_cartesian_acceptance(&false_counts, &warm, &requirements).is_err());

    let mut full_profile_requirements = requirements.clone();
    full_profile_requirements.expected_base_shapes = 2;
    assert!(
        validate_dense_q4_cartesian_acceptance(&cold, &warm, &full_profile_requirements).is_err()
    );

    let mut non_cartesian = cold.clone();
    non_cartesian.decisions[1].shape.n += 1;
    assert!(validate_dense_q4_cartesian_acceptance(&non_cartesian, &warm, &requirements).is_err());

    let mut false_dispatch = cold.clone();
    false_dispatch.decisions[0].timings[0].encoded.grid[0] += 1;
    assert!(validate_dense_q4_cartesian_acceptance(&false_dispatch, &warm, &requirements).is_err());

    let mut deadline_fallback = cold;
    deadline_fallback.deadline_overrun_ms = 0.25;
    assert!(
        validate_dense_q4_cartesian_acceptance(&deadline_fallback, &warm, &requirements).is_err()
    );

    let mut slow_with_redefined_budget = acceptance_receipt(11, false);
    slow_with_redefined_budget.elapsed_ms = 15_000.25;
    assert!(validate_dense_q4_cartesian_acceptance(
        &slow_with_redefined_budget,
        &warm,
        &requirements
    )
    .is_err());
}

#[test]
fn route_plan_cannot_cross_registry_activation_authority() {
    let source = KernelRegistry::new();
    let mut target = KernelRegistry::new();
    assert_ne!(
        source.dense_q4_auto.registry_authority_id(),
        target.dense_q4_auto.registry_authority_id()
    );
    let plan = DenseQ4RoutePlan {
        plan_id: "source-plan".into(),
        build_fingerprint: "source-build".into(),
        device_name: "source-device".into(),
        device_registry_id: 7,
        registry_authority_id: source.dense_q4_auto.registry_authority_id(),
        activation_epoch: 9,
        decisions: HashMap::new(),
    };
    source
        .dense_q4_auto
        .validate_plan_authority(&plan)
        .expect("the source registry owns its plan");
    assert!(target.dense_q4_auto.validate_plan_authority(&plan).is_err());
    assert!(target
        .install_prevalidated_dense_q4_plan(Arc::new(plan))
        .is_err());
    assert!(target.dense_q4_plan().is_none());
}
