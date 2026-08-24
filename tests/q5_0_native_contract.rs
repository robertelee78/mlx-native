use half::f16;
use mlx_native::ggml_capability::{
    ggml_capability, ggml_matrix_bytes, GgmlBatchedInputLayout, GgmlCapability,
    GgmlCapabilityRequest, GgmlExpertInputLayout, GgmlExpertShape, GgmlInvocation, GgmlKernelRoute,
    GgmlRoutingPolicy, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};
use mlx_native::gguf::{
    test_only_compute_byte_len, test_only_dequantize, test_only_ggml_type_from_u32,
    test_only_raw_tensor_dtype,
};
use mlx_native::ops::dequant_to_f16::test_only_dequant_to_f16_kernel_name;
use mlx_native::{DType, GgmlType};

const QK5_0: usize = 32;
const BLOCK_BYTES: usize = 22;

fn block(scale: f32, qh: u32, qs: [u8; 16]) -> [u8; BLOCK_BYTES] {
    let mut bytes = [0u8; BLOCK_BYTES];
    bytes[..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    bytes[2..6].copy_from_slice(&qh.to_le_bytes());
    bytes[6..].copy_from_slice(&qs);
    bytes
}

fn expected(block: &[u8; BLOCK_BYTES]) -> [f32; QK5_0] {
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let mut out = [0.0; QK5_0];
    for lane in 0..16 {
        let low = u32::from(block[6 + lane] & 0x0f) | (((qh >> lane) & 1) << 4);
        let high = u32::from(block[6 + lane] >> 4) | (((qh >> (lane + 16)) & 1) << 4);
        out[lane] = (low as i32 - 16) as f32 * scale;
        out[lane + 16] = (high as i32 - 16) as f32 * scale;
    }
    out
}

#[test]
fn q5_0_id_size_raw_storage_and_high_bit_order_are_exact() {
    assert_eq!(test_only_ggml_type_from_u32(6).unwrap(), GgmlType::Q5_0);
    assert_eq!(GgmlType::Q5_0.block_values(), 32);
    assert_eq!(GgmlType::Q5_0.block_bytes(), 22);
    assert_eq!(
        test_only_compute_byte_len(&[3, 64], GgmlType::Q5_0).unwrap(),
        132
    );
    assert!(test_only_compute_byte_len(&[3, 63], GgmlType::Q5_0).is_err());
    assert_eq!(test_only_raw_tensor_dtype(GgmlType::Q5_0), DType::U8);

    let mut qs = [0u8; 16];
    for (lane, byte) in qs.iter_mut().enumerate() {
        *byte = (lane as u8 & 0x0f) | (((15 - lane) as u8 & 0x0f) << 4);
    }
    let bytes = block(-0.25, (1 << 0) | (1 << 7) | (1 << 16) | (1 << 31), qs);
    let want = expected(&bytes);
    let mut got = [f32::NAN; QK5_0];
    test_only_dequantize(&bytes, GgmlType::Q5_0, &mut got).unwrap();
    for (index, (actual, expected)) in got.into_iter().zip(want).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "Q5_0 value {index}");
    }

    let mut moved = bytes;
    moved[2..6].copy_from_slice(&(1u32 << 1).to_le_bytes());
    let mut moved_out = [0.0; QK5_0];
    test_only_dequantize(&moved, GgmlType::Q5_0, &mut moved_out).unwrap();
    assert_ne!(got[0].to_bits(), moved_out[0].to_bits());
    assert_ne!(got[1].to_bits(), moved_out[1].to_bits());
    assert!(test_only_dequantize(&bytes[..21], GgmlType::Q5_0, &mut got).is_err());
    assert!(test_only_dequantize(&bytes, GgmlType::Q5_0, &mut got[..31]).is_err());
}

fn request(invocation: GgmlInvocation, workload: GgmlWorkloadClass) -> GgmlCapabilityRequest {
    GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation,
        ggml_type: GgmlType::Q5_0,
        workload,
        routing: GgmlRoutingPolicy::default(),
    }
}

fn expert_shape(n_tokens: u32) -> GgmlExpertShape {
    let matrix_bytes = ggml_matrix_bytes(GgmlType::Q5_0, 96, 64).unwrap();
    GgmlExpertShape {
        n_tokens,
        n: 96,
        k: 64,
        top_k: 6,
        n_experts: 8,
        expert_stride_bytes: matrix_bytes + 10,
        ids_are_distinct_per_token: true,
        ids_within_expert_range: true,
    }
}

#[test]
fn q5_0_capability_covers_every_native_public_shape_family() {
    for (m, workload) in [
        (1, GgmlWorkloadClass::DecodeSingle),
        (2, GgmlWorkloadClass::ContinuousWidth),
        (8, GgmlWorkloadClass::ContinuousWidth),
        (9, GgmlWorkloadClass::Prompt),
        (33, GgmlWorkloadClass::Prompt),
        (129, GgmlWorkloadClass::Prompt),
    ] {
        let capability = ggml_capability(request(
            GgmlInvocation::DenseAuto { m, n: 96, k: 64 },
            workload,
        ));
        assert!(
            capability.executable,
            "dense M={m}: {}",
            capability.diagnostic
        );
        assert_eq!(capability.block_values, 32);
        assert_eq!(capability.block_bytes, 22);
    }

    for m in [1, 2, 8] {
        let workload = if m == 1 {
            GgmlWorkloadClass::DecodeSingle
        } else {
            GgmlWorkloadClass::ContinuousWidth
        };
        let capability = ggml_capability(request(
            GgmlInvocation::DenseBatchedMv {
                batch: 3,
                m,
                n: 96,
                k: 64,
            },
            workload,
        ));
        assert!(
            capability.executable,
            "batched MV M={m}: {}",
            capability.diagnostic
        );
    }

    for input_layout in [
        GgmlBatchedInputLayout::Contiguous,
        GgmlBatchedInputLayout::Strided {
            row_bytes: 288,
            batch_bytes: 9 * 288,
        },
    ] {
        let capability = ggml_capability(request(
            GgmlInvocation::DenseBatchedMm {
                batch: 3,
                m: 9,
                n: 96,
                k: 64,
                input_layout,
            },
            GgmlWorkloadClass::Prompt,
        ));
        assert!(
            capability.executable,
            "batched MM: {}",
            capability.diagnostic
        );
    }

    let perm021 = ggml_capability(request(
        GgmlInvocation::DensePerm021Bf16 {
            m: 9,
            n: 96,
            k: 64,
            head_dim: 32,
        },
        GgmlWorkloadClass::Prompt,
    ));
    assert_eq!(perm021.route, Some(GgmlKernelRoute::DensePerm021TensorMm));

    let decode_shape = expert_shape(1);
    for invocation in [
        GgmlInvocation::ExpertAutoAllocated {
            shape: decode_shape,
        },
        GgmlInvocation::ExpertForceMv {
            shape: decode_shape,
        },
    ] {
        let capability = ggml_capability(request(invocation, GgmlWorkloadClass::DecodeSingle));
        assert!(
            capability.executable,
            "expert MV: {}",
            capability.diagnostic
        );
    }

    let prompt_shape = expert_shape(33);
    for invocation in [
        GgmlInvocation::ExpertAutoAllocated {
            shape: prompt_shape,
        },
        GgmlInvocation::ExpertPooled {
            shape: prompt_shape,
            input_layout: GgmlExpertInputLayout::SharedPerToken,
        },
        GgmlInvocation::ExpertPooled {
            shape: prompt_shape,
            input_layout: GgmlExpertInputLayout::Slotted,
        },
        GgmlInvocation::ExpertPooledPair {
            shape: prompt_shape,
        },
    ] {
        let capability = ggml_capability(request(invocation, GgmlWorkloadClass::Prompt));
        assert!(
            capability.executable,
            "expert MM: {}",
            capability.diagnostic
        );
    }

    let embedding = ggml_capability(request(
        GgmlInvocation::EmbeddingGather {
            n_tokens: 9,
            vocab_size: 32_000,
            embed_dim: 768,
        },
        GgmlWorkloadClass::Embedding,
    ));
    assert_eq!(embedding.route, Some(GgmlKernelRoute::EmbeddingQ5_0));

    let fused = ggml_capability(request(
        GgmlInvocation::DenseGateUpSiluPair { m: 1, n: 96, k: 64 },
        GgmlWorkloadClass::DecodeSingle,
    ));
    assert!(
        !fused.executable,
        "Q5_0 must use two native dense calls, not a codec-substitution fallback"
    );
    let expert_fused = ggml_capability(request(
        GgmlInvocation::ExpertSwiGluDownQ4 {
            shape: decode_shape,
        },
        GgmlWorkloadClass::DecodeSingle,
    ));
    assert!(
        !expert_fused.executable,
        "Q5_0 must not enter the Q4_0-only fused expert route"
    );

    let mut invalid_bounds = prompt_shape;
    invalid_bounds.ids_within_expert_range = false;
    assert!(
        !ggml_capability(request(
            GgmlInvocation::ExpertAutoAllocated {
                shape: invalid_bounds,
            },
            GgmlWorkloadClass::Prompt,
        ))
        .executable
    );
    let mut duplicate_ids = prompt_shape;
    duplicate_ids.ids_are_distinct_per_token = false;
    assert!(
        !ggml_capability(request(
            GgmlInvocation::ExpertAutoAllocated {
                shape: duplicate_ids,
            },
            GgmlWorkloadClass::Prompt,
        ))
        .executable
    );

    let encoded = serde_json::to_string(&embedding).expect("serialize Q5_0 capability");
    let decoded: GgmlCapability =
        serde_json::from_str(&encoded).expect("deserialize Q5_0 capability");
    assert_eq!(decoded, embedding);
}

#[test]
fn q5_0_shader_source_contract_has_every_native_route() {
    let dense_mv = include_str!("../src/shaders/quantized_matmul_ggml.metal");
    let dense_mm = include_str!("../src/shaders/quantized_matmul_mm.metal");
    let dense_tensor = include_str!("../src/shaders/quantized_matmul_mm_tensor.metal");
    let mv_ext = include_str!("../src/shaders/mul_mv_ext.metal");
    let expert_mv = include_str!("../src/shaders/quantized_matmul_id_ggml.metal");
    let expert_mm = include_str!("../src/shaders/quantized_matmul_id_mm.metal");
    let expert_tensor = include_str!("../src/shaders/quantized_matmul_id_mm_tensor.metal");
    let embedding = include_str!("../src/shaders/embedding_q5_0.metal");

    for (source, required) in [
        (dense_mv, &["block_q5_0", "kernel_mul_mv_q5_0_f32"][..]),
        (dense_mm, &["block_q5_0", "kernel_mul_mm_q5_0_f32"][..]),
        (
            dense_tensor,
            &[
                "kernel_mul_mm_q5_0_tensor_f32",
                "kernel_mul_mm_q5_0_tensor_v2_f32",
                "kernel_mul_mm_q5_0_tensor_bf16_perm021",
            ][..],
        ),
        (
            mv_ext,
            &[
                "kernel_mul_mv_ext_q5_0_f32_r1_2",
                "kernel_mul_mv_ext_q5_0_f32_r1_3",
                "kernel_mul_mv_ext_q5_0_f32_r1_4",
                "kernel_mul_mv_ext_q5_0_f32_r1_5",
            ][..],
        ),
        (expert_mv, &["block_q5_0", "kernel_mul_mv_id_q5_0_f32"][..]),
        (expert_mm, &["block_q5_0", "kernel_mul_mm_id_q5_0_f32"][..]),
        (
            expert_tensor,
            &["block_q5_0", "kernel_mul_mm_id_q5_0_tensor_f32"][..],
        ),
        (embedding, &["block_q5_0", "embedding_gather_q5_0_f32"][..]),
    ] {
        for symbol in required {
            assert!(
                source.contains(symbol),
                "missing Q5_0 source symbol {symbol}"
            );
        }
        assert!(
            source.contains("-16") || source.contains("- 16"),
            "Q5_0 source must preserve signed zero point"
        );
    }

    assert!(expert_mv.contains("poison_invalid_expert_id"));
    assert!(expert_mv.contains("expert_id, output_row, first_row, nr"));
    assert!(expert_mm.contains("threadgroup uint32_t * sids"));
    assert!(!expert_mm.contains("threadgroup uint16_t * sids"));
    assert!(expert_mm.contains("sids[i20] >= ntg"));
    assert!(expert_mm.contains("sids[i20] == sids[other]"));
    for consumer in [expert_mm, expert_tensor] {
        assert!(consumer.contains("route_state & 0x80000000u"));
        assert!(consumer.contains("out[index] = NAN"));
    }
}

#[test]
fn q5_0_cannot_enter_the_f16_shadow_materialization_route() {
    assert!(test_only_dequant_to_f16_kernel_name(GgmlType::Q5_0).is_err());
    assert_eq!(
        test_only_dequant_to_f16_kernel_name(GgmlType::Q5_K).unwrap(),
        "hf2q_dequant_q5_K_to_f16"
    );

    // Source absence is supplemental: the executable selector assertion
    // above fails even if a wildcard arm silently admits Q5_0.
    let shader = include_str!("../src/shaders/dequant_to_f16.metal");
    assert!(!shader.contains("dequant_q5_0"));
    assert!(!shader.contains("block_q5_0"));
}
