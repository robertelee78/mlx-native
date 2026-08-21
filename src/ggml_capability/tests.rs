use super::*;

fn dense_request(m: u32, workload: GgmlWorkloadClass) -> GgmlCapabilityRequest {
    GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::DenseAuto {
            m,
            n: 5_120,
            k: 5_120,
        },
        ggml_type: GgmlType::Q4_K,
        workload,
        routing: GgmlRoutingPolicy::default(),
    }
}

fn expert_shape(n_tokens: u32) -> GgmlExpertShape {
    let matrix_bytes = ggml_matrix_bytes(GgmlType::Q4_K, 5_120, 5_120).unwrap();
    GgmlExpertShape {
        n_tokens,
        n: 5_120,
        k: 5_120,
        top_k: 6,
        n_experts: 256,
        expert_stride_bytes: matrix_bytes,
        ids_are_distinct_per_token: true,
        ids_within_expert_range: true,
    }
}

#[test]
fn dense_decode_reports_specialized_mv_and_exact_bytes() {
    let capability = ggml_capability(dense_request(1, GgmlWorkloadClass::DecodeSingle));
    assert!(capability.executable);
    assert!(capability.specialized_for_workload);
    assert!(!capability.correctness_fallback);
    assert_eq!(capability.route, Some(GgmlKernelRoute::DenseMv));
    assert_eq!(capability.block_values, 256);
    assert_eq!(capability.block_bytes, 144);
    assert_eq!(capability.weight_buffer_count, 1);
    assert_eq!(
        capability.minimum_total_weight_bytes,
        5_120 * (5_120 / 256) * 144
    );
    assert_eq!(capability.dispatches, 1);
    assert_eq!(capability.barriers, 0);
}

#[test]
fn dense_auto_routing_policy_is_explicit() {
    let mut request = dense_request(4, GgmlWorkloadClass::ContinuousWidth);
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::DenseQ4kWidthMn)
    );

    request.ggml_type = GgmlType::Q6_K;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::DenseQ6kWidthMn)
    );

    request.routing.dense_decode_mvn = false;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::DenseMvNr2)
    );

    request.routing.dense_decode_mv_ext = true;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::DenseWidthMvExt)
    );
}

#[test]
fn mv_ext_widths_are_type_sensitive() {
    let mut request = dense_request(2, GgmlWorkloadClass::ContinuousWidth);
    request.routing.dense_decode_mvn = false;
    request.routing.dense_decode_mv_ext = true;

    for ggml_type in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        request.ggml_type = ggml_type;
        for width in 2..=3 {
            request.invocation = GgmlInvocation::DenseAuto {
                m: width,
                n: 5_120,
                k: 5_120,
            };
            assert_ne!(
                ggml_capability(request).route,
                Some(GgmlKernelRoute::DenseWidthMvExt),
                "{ggml_type:?} width {width} must stay off mul_mv_ext"
            );
        }
        request.invocation = GgmlInvocation::DenseAuto {
            m: 4,
            n: 5_120,
            k: 5_120,
        };
        assert_eq!(
            ggml_capability(request).route,
            Some(GgmlKernelRoute::DenseWidthMvExt),
            "{ggml_type:?} width 4 must use mul_mv_ext"
        );
    }

    for ggml_type in [GgmlType::Q4_0, GgmlType::Q8_0] {
        request.ggml_type = ggml_type;
        request.invocation = GgmlInvocation::DenseAuto {
            m: 2,
            n: 5_120,
            k: 5_120,
        };
        assert_eq!(
            ggml_capability(request).route,
            Some(GgmlKernelRoute::DenseWidthMvExt),
            "legacy {ggml_type:?} width 2 remains eligible for mul_mv_ext"
        );
    }
}

#[test]
fn prompt_device_selection_is_not_misreported_as_resolved() {
    let mut request = dense_request(2_048, GgmlWorkloadClass::Prompt);
    let capability = ggml_capability(request);
    assert_eq!(
        capability.route,
        Some(GgmlKernelRoute::DenseMmDeviceSelected)
    );
    assert!(capability.requires_device_probe);

    request.routing.dense_tensor_mm = GgmlTensorMmPreference::ForceSimd;
    let capability = ggml_capability(request);
    assert_eq!(capability.route, Some(GgmlKernelRoute::DenseMmSimdgroup));
    assert!(!capability.requires_device_probe);
}

#[test]
fn short_prompt_is_valid_but_reports_width_route_as_fallback() {
    let request = dense_request(4, GgmlWorkloadClass::Prompt);
    let capability = ggml_capability(request);
    assert!(capability.executable);
    assert_eq!(capability.route, Some(GgmlKernelRoute::DenseQ4kWidthMn));
    assert!(!capability.specialized_for_workload);
    assert!(capability.correctness_fallback);
}

#[test]
fn iq4_xs_prompt_exposes_matvec_fallback() {
    let mut request = dense_request(2_048, GgmlWorkloadClass::Prompt);
    request.ggml_type = GgmlType::IQ4_XS;
    let capability = ggml_capability(request);
    assert!(capability.executable);
    assert_eq!(capability.route, Some(GgmlKernelRoute::DenseMv));
    assert!(!capability.specialized_for_workload);
    assert!(capability.correctness_fallback);
}

#[test]
fn fused_gate_up_is_a_two_weight_atomic_operation() {
    let mut request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    request.invocation = GgmlInvocation::DenseGateUpSiluPair {
        m: 1,
        n: 17_408,
        k: 5_120,
    };
    request.ggml_type = GgmlType::Q5_K;
    let capability = ggml_capability(request);
    assert_eq!(capability.route, Some(GgmlKernelRoute::FusedGateUpSilu));
    assert_eq!(capability.weight_buffer_count, 2);
    assert_eq!(capability.dispatches, 1);
    assert_eq!(
        capability.minimum_total_weight_bytes,
        capability.minimum_weight_buffer_bytes * 2
    );

    request.ggml_type = GgmlType::Q4_0;
    assert!(!ggml_capability(request).executable);
}

#[test]
fn perm021_support_and_head_layout_are_exact() {
    let mut request = dense_request(512, GgmlWorkloadClass::Prompt);
    request.invocation = GgmlInvocation::DensePerm021Bf16 {
        m: 512,
        n: 5_120,
        k: 6_144,
        head_dim: 128,
    };
    request.ggml_type = GgmlType::Q8_0;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::DensePerm021TensorMm)
    );

    request.ggml_type = GgmlType::Q5_K;
    assert!(!ggml_capability(request).executable);
}

#[test]
fn batched_entrypoints_are_not_conflated_with_dense_auto() {
    let mut request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    request.invocation = GgmlInvocation::DenseBatchedMv {
        batch: 4,
        m: 1,
        n: 5_120,
        k: 5_120,
    };
    request.ggml_type = GgmlType::Q8_0;
    let capability = ggml_capability(request);
    assert_eq!(capability.route, Some(GgmlKernelRoute::DenseBatchedMvNr2));
    assert_eq!(capability.weight_buffer_count, 1);
    assert_eq!(
        capability.minimum_total_weight_bytes,
        ggml_batched_matrix_bytes(GgmlType::Q8_0, 4, 5_120, 5_120).unwrap()
    );

    request.invocation = GgmlInvocation::DenseBatchedMm {
        batch: 3,
        m: 2_048,
        n: 5_120,
        k: 5_120,
        input_layout: GgmlBatchedInputLayout::Strided {
            row_bytes: 20_480,
            batch_bytes: 41_943_040,
        },
    };
    request.workload = GgmlWorkloadClass::Prompt;
    let capability = ggml_capability(request);
    assert_eq!(
        capability.route,
        Some(GgmlKernelRoute::DenseBatchedMmDeviceSelected)
    );
    assert_eq!(capability.weight_buffer_count, 1);
    assert_eq!(
        capability.minimum_total_weight_bytes,
        ggml_batched_matrix_bytes(GgmlType::Q8_0, 3, 5_120, 5_120).unwrap()
    );
}

#[test]
fn batched_mv_accepts_width_eight_and_rejects_width_nine_or_prompt_contract() {
    let mut request = dense_request(8, GgmlWorkloadClass::ContinuousWidth);
    request.invocation = GgmlInvocation::DenseBatchedMv {
        batch: 2,
        m: 8,
        n: 5_120,
        k: 5_120,
    };
    request.ggml_type = GgmlType::Q8_0;
    assert!(ggml_capability(request).executable);

    request.invocation = GgmlInvocation::DenseBatchedMv {
        batch: 2,
        m: 9,
        n: 5_120,
        k: 5_120,
    };
    request.workload = GgmlWorkloadClass::Prompt;
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::InvalidOperationContract)
    );

    request.invocation = GgmlInvocation::DenseBatchedMv {
        batch: 2,
        m: 8,
        n: 5_120,
        k: 5_120,
    };
    let capability = ggml_capability(request);
    assert!(capability.executable);
    assert!(capability.correctness_fallback);
}

#[test]
fn q4_and_q6_width_dispatch_count_matches_runtime_tiling() {
    for (m, expected) in [(2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2), (8, 2)] {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let mut request = dense_request(m, GgmlWorkloadClass::ContinuousWidth);
            request.ggml_type = ggml_type;
            let capability = ggml_capability(request);
            assert_eq!(capability.dispatches, expected, "{ggml_type:?} M={m}");
        }
    }
}

#[test]
fn expert_entrypoint_stride_and_scratch_define_capability() {
    let mut request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    request.invocation = GgmlInvocation::ExpertForceMv {
        shape: expert_shape(1),
    };
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::ExpertMv)
    );

    let shape = expert_shape(2_048);
    request.invocation = GgmlInvocation::ExpertPooledPair { shape };
    request.workload = GgmlWorkloadClass::Prompt;
    let capability = ggml_capability(request);
    assert_eq!(
        capability.route,
        Some(GgmlKernelRoute::ExpertPooledPairMmDeviceSelected)
    );
    assert_eq!(capability.weight_buffer_count, 2);
    assert_eq!(capability.dispatches, 3);
    assert_eq!(capability.barriers, 1);
    assert!(matches!(
        capability.scratch,
        GgmlScratchRequirement::ExpertMm {
            caller_owned: true,
            schedule_reused: true,
            ..
        }
    ));

    let mut bad_shape = shape;
    bad_shape.expert_stride_bytes -= 1;
    request.invocation = GgmlInvocation::ExpertPooledPair { shape: bad_shape };
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::UnsupportedLayout)
    );
}

#[test]
fn expert_mm_requires_distinct_ids() {
    let mut request = dense_request(2_048, GgmlWorkloadClass::Prompt);
    let mut shape = expert_shape(2_048);
    shape.ids_are_distinct_per_token = false;
    request.invocation = GgmlInvocation::ExpertAutoAllocated { shape };
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::InvalidOperationContract)
    );
}

#[test]
fn expert_contract_rejects_out_of_range_ids_and_impossible_top_k() {
    let mut request = dense_request(2_048, GgmlWorkloadClass::Prompt);
    let mut shape = expert_shape(2_048);
    shape.ids_within_expert_range = false;
    request.invocation = GgmlInvocation::ExpertAutoAllocated { shape };
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::InvalidOperationContract)
    );

    shape.ids_within_expert_range = true;
    shape.top_k = shape.n_experts + 1;
    request.invocation = GgmlInvocation::ExpertAutoAllocated { shape };
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::InvalidOperationContract)
    );
}

#[test]
fn perm021_requires_a_device_probe_and_marks_non_prompt_as_fallback() {
    let mut request = dense_request(4, GgmlWorkloadClass::ContinuousWidth);
    request.invocation = GgmlInvocation::DensePerm021Bf16 {
        m: 4,
        n: 5_120,
        k: 6_144,
        head_dim: 128,
    };
    request.ggml_type = GgmlType::Q8_0;
    let capability = ggml_capability(request);
    assert!(capability.executable);
    assert!(capability.requires_device_probe);
    assert!(!capability.specialized_for_workload);
    assert!(capability.correctness_fallback);
}

#[test]
fn slotted_expert_path_rejects_matvec_shape() {
    let mut request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    request.invocation = GgmlInvocation::ExpertPooled {
        shape: expert_shape(1),
        input_layout: GgmlExpertInputLayout::Slotted,
    };
    assert!(!ggml_capability(request).executable);
}

#[test]
fn expert_swiglu_q4_is_a_distinct_invocation() {
    let mut request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    request.ggml_type = GgmlType::Q4_0;
    request.invocation = GgmlInvocation::ExpertSwiGluDownQ4 {
        shape: expert_shape(1),
    };
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::ExpertSwiGluDownQ4)
    );
    request.ggml_type = GgmlType::Q8_0;
    assert!(!ggml_capability(request).executable);
}

#[test]
fn embedding_contract_is_exact() {
    let mut request = dense_request(16, GgmlWorkloadClass::Embedding);
    request.invocation = GgmlInvocation::EmbeddingGather {
        n_tokens: 16,
        vocab_size: 151_936,
        embed_dim: 5_120,
    };
    request.ggml_type = GgmlType::Q2_K;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::EmbeddingQ2K)
    );
    request.ggml_type = GgmlType::Q4_K;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::EmbeddingQ4K)
    );
    request.ggml_type = GgmlType::Q8_0;
    assert_eq!(
        ggml_capability(request).route,
        Some(GgmlKernelRoute::EmbeddingQ8_0)
    );
    request.ggml_type = GgmlType::Q5_K;
    assert!(!ggml_capability(request).executable);
}

#[test]
fn workload_shape_mismatch_fails_closed() {
    let request = dense_request(1, GgmlWorkloadClass::ContinuousWidth);
    assert_eq!(
        ggml_capability(request).rejection_code,
        Some(GgmlRejectionCode::InvalidOperationContract)
    );
}

#[test]
fn checked_layout_helpers_use_padded_expert_stride() {
    let matrix = ggml_matrix_bytes(GgmlType::Q4_K, 5_120, 5_120).unwrap();
    let padded = matrix + 4_096;
    assert_eq!(
        ggml_expert_bytes(GgmlType::Q4_K, 3, 5_120, 5_120, padded).unwrap(),
        padded * 2 + matrix
    );
    assert!(ggml_packed_row_bytes(GgmlType::Q4_K, 5_121).is_err());
}

#[test]
fn total_weight_overflow_is_a_typed_rejection_not_a_panic() {
    let request = GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::DenseGateUpSiluPair {
            m: 1,
            n: u32::MAX,
            k: u32::MAX - 255,
        },
        ggml_type: GgmlType::Q4_K,
        workload: GgmlWorkloadClass::DecodeSingle,
        routing: GgmlRoutingPolicy::default(),
    };
    let result = std::panic::catch_unwind(|| ggml_capability(request));
    let capability = result.expect("capability query must not panic");
    assert_eq!(
        capability.rejection_code,
        Some(GgmlRejectionCode::ArithmeticOverflow)
    );
}

#[test]
fn request_and_receipt_round_trip_as_json() {
    let request = dense_request(1, GgmlWorkloadClass::DecodeSingle);
    let request_json = serde_json::to_string(&request).expect("serialize request");
    let decoded_request: GgmlCapabilityRequest =
        serde_json::from_str(&request_json).expect("deserialize request");
    assert_eq!(decoded_request, request);

    let capability = ggml_capability(request);
    let json = serde_json::to_string(&capability).expect("serialize capability");
    let decoded: GgmlCapability = serde_json::from_str(&json).expect("deserialize capability");
    assert_eq!(decoded, capability);
}
