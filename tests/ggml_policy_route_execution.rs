//! The serialized routing policy must select the same production kernel route
//! that a capability/cost receipt names. Capture records the actual public
//! dispatch without executing large prompt kernels.

#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml_with_policy, CapturedNode, DType,
    GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, KernelRegistry, MlxDevice,
};

fn labels(nodes: &[CapturedNode]) -> Vec<String> {
    nodes
        .iter()
        .filter_map(|node| match node {
            CapturedNode::Dispatch { pipeline, .. } => Some(pipeline.label().to_owned()),
            CapturedNode::Barrier => None,
        })
        .collect()
}

fn capture_dense(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    params: GgmlQuantizedMatmulParams,
    policy: GgmlRoutingPolicy,
) -> Vec<String> {
    let block_bytes = params.ggml_type.block_bytes() as usize;
    let block_values = params.ggml_type.block_values() as usize;
    let weight_bytes = params.n as usize * (params.k as usize / block_values) * block_bytes;
    let input = device
        .alloc_buffer(
            params.m as usize * params.k as usize * 4,
            DType::F32,
            vec![params.m as usize, params.k as usize],
        )
        .expect("input");
    let weight = device
        .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
        .expect("weight");
    let output = device
        .alloc_buffer(
            params.m as usize * params.n as usize * 4,
            DType::F32,
            vec![params.m as usize, params.n as usize],
        )
        .expect("output");
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_capture();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        registry,
        device,
        &input,
        &weight,
        &output,
        &params,
        &policy,
    )
    .expect("dense capture");
    labels(&encoder.take_capture().expect("dense graph"))
}

#[test]
fn width_and_dense_mm_policy_knobs_bind_production_dispatch() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let width = GgmlQuantizedMatmulParams {
        m: 4,
        n: 32,
        k: 256,
        ggml_type: GgmlType::Q6_K,
    };
    let width_mn = capture_dense(&device, &mut registry, width, GgmlRoutingPolicy::default());
    assert!(width_mn.iter().all(|label| label.contains("q6_K_f32_mN")));

    let mv_ext = capture_dense(
        &device,
        &mut registry,
        width,
        GgmlRoutingPolicy {
            dense_decode_mvn: false,
            dense_decode_mv_ext: true,
            ..GgmlRoutingPolicy::default()
        },
    );
    assert!(mv_ext.iter().all(|label| label.contains("mul_mv_ext_q6_K")));
    assert_ne!(width_mn, mv_ext);

    let prompt = GgmlQuantizedMatmulParams {
        m: 9,
        n: 64,
        k: 256,
        ggml_type: GgmlType::Q8_0,
    };
    let simd = capture_dense(
        &device,
        &mut registry,
        prompt,
        GgmlRoutingPolicy {
            dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            allow_dense_large_tile_mm: false,
            ..GgmlRoutingPolicy::default()
        },
    );
    assert!(
        simd.iter()
            .all(|label| label.contains("kernel_mul_mm_q8_0_f32")),
        "{simd:?}"
    );

    let tensor_v1 = capture_dense(
        &device,
        &mut registry,
        prompt,
        GgmlRoutingPolicy {
            allow_dense_large_tile_mm: false,
            ..GgmlRoutingPolicy::default()
        },
    );
    let tensor_v2 = capture_dense(&device, &mut registry, prompt, GgmlRoutingPolicy::default());
    if tensor_v1.iter().any(|label| label.contains("tensor")) {
        assert!(tensor_v1.iter().all(|label| label.contains("tensor")));
        assert!(tensor_v2.iter().all(|label| label.contains("tensor_v2")));
        assert_ne!(tensor_v1, tensor_v2);
    } else {
        // M1/M2 and metallibs without Tensor API support correctly resolve
        // AutoProbe to the same SIMD route; large-tile is then inapplicable.
        assert_eq!(tensor_v1, simd);
        assert_eq!(tensor_v2, simd);
    }
}

fn capture_expert(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    policy: GgmlRoutingPolicy,
) -> Vec<String> {
    let rows = 9usize;
    let matrix_bytes = 32 * 34;
    let input = device
        .alloc_buffer(rows * 32 * 4, DType::F32, vec![rows, 32])
        .expect("input");
    let weight = device
        .alloc_buffer(2 * matrix_bytes, DType::U8, vec![2 * matrix_bytes])
        .expect("weight");
    let ids = device
        .alloc_buffer(rows * 4, DType::U32, vec![rows])
        .expect("ids");
    let output = device
        .alloc_buffer(rows * 32 * 4, DType::F32, vec![rows, 32])
        .expect("output");
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: rows as u32,
        top_k: 1,
        n: 32,
        k: 32,
        n_experts: 2,
        expert_stride: matrix_bytes as u64,
        ggml_type: GgmlType::Q8_0,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_with_policy(
        &mut encoder,
        registry,
        device,
        &input,
        &weight,
        &ids,
        &output,
        &params,
        &policy,
    )
    .expect("expert capture");
    labels(&encoder.take_capture().expect("expert graph"))
}

#[test]
fn expert_threshold_and_tensor_policy_bind_production_dispatch() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mv = capture_expert(
        &device,
        &mut registry,
        GgmlRoutingPolicy {
            expert_mm_threshold: 9,
            ..GgmlRoutingPolicy::default()
        },
    );
    assert_eq!(mv.len(), 1);
    assert!(mv[0].contains("mul_mv_id_q8_0"));

    let simd = capture_expert(
        &device,
        &mut registry,
        GgmlRoutingPolicy {
            expert_mm_threshold: 8,
            expert_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            ..GgmlRoutingPolicy::default()
        },
    );
    assert!(simd
        .iter()
        .any(|label| label.contains("mul_mm_id_q8_0_f32")));
    assert!(simd.iter().all(|label| !label.contains("tensor")));

    let tensor = capture_expert(
        &device,
        &mut registry,
        GgmlRoutingPolicy {
            expert_mm_threshold: 8,
            ..GgmlRoutingPolicy::default()
        },
    );
    if tensor.iter().any(|label| label.contains("tensor")) {
        assert!(tensor
            .iter()
            .any(|label| label.contains("mul_mm_id_q8_0_tensor")));
        assert_ne!(simd, tensor);
    } else {
        assert_eq!(tensor, simd);
    }
}
