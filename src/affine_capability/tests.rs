use super::*;

fn request() -> PackedAffineRequest {
    PackedAffineRequest {
        operation: AffineOperation::Dense,
        regime: AffineExecutionRegime::DecodeQmv,
        io_dtype: AffineIoDType::Bf16,
        bits: 4,
        group_size: 64,
        m: 1,
        n: 5120,
        k: 5120,
        has_biases: true,
    }
}

#[test]
fn q4_bf16_decode_reports_specialized_qmv() {
    let capability = packed_affine_capability(request());
    assert!(capability.executable);
    assert!(capability.specialized_for_regime);
    assert_eq!(
        capability.route,
        Some(PackedAffineKernelRoute::DenseRowWiseSimdBf16)
    );
}

#[test]
fn same_kernel_is_not_misreported_as_prompt_qmm() {
    let mut request = request();
    request.regime = AffineExecutionRegime::PromptQmm;
    request.m = 128;
    let capability = packed_affine_capability(request);
    assert!(capability.executable);
    assert!(!capability.specialized_for_regime);
    assert!(capability.diagnostic.contains("not a specialized QMM"));
}

#[test]
fn q6_dense_is_an_explicit_scalar_fallback() {
    let mut request = request();
    request.bits = 6;
    let capability = packed_affine_capability(request);
    assert!(capability.executable);
    assert!(!capability.specialized_for_regime);
    assert_eq!(
        capability.route,
        Some(PackedAffineKernelRoute::DenseScalarViaF32)
    );
}

#[test]
fn unsupported_bf16_shape_falls_back_instead_of_claiming_qmv() {
    let mut request = request();
    request.k = 2816;
    let capability = packed_affine_capability(request);
    assert!(capability.executable);
    assert!(!capability.specialized_for_regime);
    assert_eq!(
        capability.route,
        Some(PackedAffineKernelRoute::DenseScalarViaF32)
    );
}

#[test]
fn group_larger_than_simd_block_falls_back() {
    let mut request = request();
    request.group_size = 1024;
    let capability = packed_affine_capability(request);
    assert_eq!(
        capability.route,
        Some(PackedAffineKernelRoute::DenseScalarViaF32)
    );
}

#[test]
fn expert_offset_has_no_scalar_fallback() {
    let mut request = request();
    request.operation = AffineOperation::ExpertOffset;
    request.bits = 6;
    let capability = packed_affine_capability(request);
    assert!(!capability.executable);
    assert_eq!(capability.route, None);
    assert_eq!(
        capability.rejection_code,
        Some(PackedAffineRejectionCode::UnsupportedLayout)
    );
}

#[test]
fn expert_id_scalar_f32_supports_six_bit() {
    let mut request = request();
    request.operation = AffineOperation::ExpertRoutedId;
    request.io_dtype = AffineIoDType::F32;
    request.bits = 6;
    let capability = packed_affine_capability(request);
    assert_eq!(
        capability.route,
        Some(PackedAffineKernelRoute::ExpertRoutedIdScalarF32)
    );
    assert!(!capability.specialized_for_regime);
}

#[test]
fn four_and_six_bit_embeddings_are_reported() {
    let mut request = request();
    request.operation = AffineOperation::Embedding;
    request.regime = AffineExecutionRegime::EmbeddingGather;
    request.io_dtype = AffineIoDType::F32;
    request.m = 16;
    request.n = 151_936;
    for bits in [4, 6] {
        request.bits = bits;
        let capability = packed_affine_capability(request);
        assert_eq!(
            capability.route,
            Some(PackedAffineKernelRoute::EmbeddingGatherF32)
        );
        assert!(capability.specialized_for_regime);
    }
}

#[test]
fn embedding_width_must_match_four_bit_packing_quantum() {
    let mut request = request();
    request.operation = AffineOperation::Embedding;
    request.regime = AffineExecutionRegime::EmbeddingGather;
    request.io_dtype = AffineIoDType::F32;
    request.group_size = 4;
    request.k = 36;
    let capability = packed_affine_capability(request);
    assert!(!capability.executable);
    assert_eq!(
        capability.rejection_code,
        Some(PackedAffineRejectionCode::UnsupportedLayout)
    );
}

#[test]
fn embedding_width_must_match_six_bit_packing_quantum() {
    let mut request = request();
    request.operation = AffineOperation::Embedding;
    request.regime = AffineExecutionRegime::EmbeddingGather;
    request.io_dtype = AffineIoDType::F32;
    request.bits = 6;
    request.group_size = 2;
    request.k = 6;
    let capability = packed_affine_capability(request);
    assert!(!capability.executable);
    assert_eq!(
        capability.rejection_code,
        Some(PackedAffineRejectionCode::UnsupportedLayout)
    );
}

#[test]
fn eight_bit_embedding_is_rejected_by_exact_bit_contract() {
    let mut request = request();
    request.operation = AffineOperation::Embedding;
    request.regime = AffineExecutionRegime::EmbeddingGather;
    request.io_dtype = AffineIoDType::F32;
    request.bits = 8;
    let capability = packed_affine_capability(request);
    assert!(!capability.executable);
    assert_eq!(
        capability.rejection_code,
        Some(PackedAffineRejectionCode::UnsupportedBits)
    );
}

#[test]
fn missing_biases_fail_closed() {
    let mut request = request();
    request.has_biases = false;
    let capability = packed_affine_capability(request);
    assert!(!capability.executable);
}

#[test]
fn receipt_round_trips_as_json() {
    let capability = packed_affine_capability(request());
    let decoded = serde_json::to_string(&capability)
        .and_then(|json| serde_json::from_str::<PackedAffineCapability>(&json));
    assert!(decoded.is_ok());
    if let Ok(decoded) = decoded {
        assert_eq!(decoded, capability);
    }
}
