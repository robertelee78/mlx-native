#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_gate_up_silu_q4_K::{
    dispatch_fused_gate_up_silu_q4_K_with_trace, FusedGateUpSiluQ4_KArgs,
};
use mlx_native::{
    quantized_matmul_ggml_with_policy_and_trace, DType, GgmlQuantizedMatmulParams,
    GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType,
    GgmlWorkloadClass, KernelPipelineOrigin, KernelRegistry, MlxDevice,
};

fn buffers(
    device: &MlxDevice,
    params: GgmlQuantizedMatmulParams,
) -> (
    mlx_native::MlxBuffer,
    mlx_native::MlxBuffer,
    mlx_native::MlxBuffer,
) {
    let weight_bytes = params.n as usize
        * (params.k as usize / params.ggml_type.block_values() as usize)
        * params.ggml_type.block_bytes() as usize;
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
    (input, weight, output)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[test]
fn trace_binds_actual_dense_routes_and_drains_failures() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();

    let prompt = GgmlQuantizedMatmulParams {
        m: 9,
        n: 64,
        k: 256,
        ggml_type: GgmlType::Q8_0,
    };
    let (input, weight, output) = buffers(&device, prompt);
    let mut encoder = device.command_encoder().expect("encoder");
    let force_simd = GgmlRoutingPolicy {
        dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    };
    let simd = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &prompt,
        &force_simd,
        GgmlWorkloadClass::Prompt,
    )
    .expect("ForceSimd trace");
    assert_eq!(
        simd.resolved_route,
        GgmlResolvedKernelRoute::DenseMmSimdgroup
    );
    assert_eq!(simd.device.registry_id, device.registry_id());
    assert_eq!(simd.dispatches.len(), 1);
    for dispatch in &simd.dispatches {
        match dispatch.pipeline.origin {
            KernelPipelineOrigin::PrecompiledMetallib => {
                assert!(dispatch.pipeline.runtime_source_sha256.is_none());
                assert!(is_lowercase_sha256(
                    dispatch
                        .pipeline
                        .embedded_metallib_sha256
                        .as_deref()
                        .expect("precompiled metallib digest")
                ));
            }
            KernelPipelineOrigin::RuntimeSource => {
                assert!(dispatch.pipeline.embedded_metallib_sha256.is_none());
                assert!(is_lowercase_sha256(
                    dispatch
                        .pipeline
                        .runtime_source_sha256
                        .as_deref()
                        .expect("runtime-source digest")
                ));
            }
        }
    }
    if std::env::var("MLX_PRECOMPILED_METALLIB").as_deref() == Ok("0") {
        assert!(simd
            .dispatches
            .iter()
            .all(|dispatch| dispatch.pipeline.origin == KernelPipelineOrigin::RuntimeSource));
    }

    let auto = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &prompt,
        &GgmlRoutingPolicy::default(),
        GgmlWorkloadClass::Prompt,
    )
    .expect("AutoProbe trace");
    assert!(matches!(
        auto.resolved_route,
        GgmlResolvedKernelRoute::DenseMmSimdgroup | GgmlResolvedKernelRoute::DenseMmTensorV2
    ));
    assert!(auto.capability.requires_device_probe);

    let width = GgmlQuantizedMatmulParams {
        m: 7,
        n: 65,
        k: 256,
        ggml_type: GgmlType::Q6_K,
    };
    let (width_input, width_weight, width_output) = buffers(&device, width);
    let width_trace = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &width_input,
        &width_weight,
        &width_output,
        &width,
        &GgmlRoutingPolicy::default(),
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("Q6_K width trace");
    assert_eq!(
        width_trace.resolved_route,
        GgmlResolvedKernelRoute::DenseQ6kWidthMn
    );
    assert_eq!(width_trace.dispatches.len(), 2);

    let short_weight = device
        .alloc_buffer(1, DType::U8, vec![1])
        .expect("short weight");
    assert!(quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &short_weight,
        &output,
        &prompt,
        &force_simd,
        GgmlWorkloadClass::Prompt,
    )
    .is_err());
    // The failed call must drain its scoped recorder; the next call succeeds.
    quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &prompt,
        &force_simd,
        GgmlWorkloadClass::Prompt,
    )
    .expect("trace after failure");

    encoder
        .commit_and_wait()
        .expect("execute traced dispatches");
}

#[test]
fn trace_binds_baseline_nr2_mv_ext_and_small_tile_device_routes() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("encoder");

    let decode = GgmlQuantizedMatmulParams {
        m: 1,
        n: 16,
        k: 32,
        ggml_type: GgmlType::Q8_0,
    };
    let (decode_input, decode_weight, decode_output) = buffers(&device, decode);
    let baseline_policy = GgmlRoutingPolicy {
        dense_q8_0_mv_nr2: false,
        ..GgmlRoutingPolicy::default()
    };
    let baseline = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &decode_input,
        &decode_weight,
        &decode_output,
        &decode,
        &baseline_policy,
        GgmlWorkloadClass::DecodeSingle,
    )
    .expect("baseline matvec trace");
    assert_eq!(baseline.resolved_route, GgmlResolvedKernelRoute::DenseMv);

    let nr2 = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &decode_input,
        &decode_weight,
        &decode_output,
        &decode,
        &GgmlRoutingPolicy::default(),
        GgmlWorkloadClass::DecodeSingle,
    )
    .expect("NR2 matvec trace");
    assert_eq!(nr2.resolved_route, GgmlResolvedKernelRoute::DenseMvNr2);

    let width = GgmlQuantizedMatmulParams {
        m: 2,
        n: 16,
        k: 256,
        ggml_type: GgmlType::Q4_K,
    };
    let (width_input, width_weight, width_output) = buffers(&device, width);
    let mv_ext_policy = GgmlRoutingPolicy {
        dense_decode_mvn: false,
        dense_decode_mv_ext: true,
        ..GgmlRoutingPolicy::default()
    };
    let mv_ext = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &width_input,
        &width_weight,
        &width_output,
        &width,
        &mv_ext_policy,
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("MV_EXT trace");
    assert_eq!(
        mv_ext.resolved_route,
        GgmlResolvedKernelRoute::DenseWidthMvExt
    );

    let prompt = GgmlQuantizedMatmulParams {
        m: 9,
        n: 64,
        k: 256,
        ggml_type: GgmlType::Q8_0,
    };
    let (prompt_input, prompt_weight, prompt_output) = buffers(&device, prompt);
    let small_tile_policy = GgmlRoutingPolicy {
        allow_dense_large_tile_mm: false,
        ..GgmlRoutingPolicy::default()
    };
    let small_tile = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &prompt_input,
        &prompt_weight,
        &prompt_output,
        &prompt,
        &small_tile_policy,
        GgmlWorkloadClass::Prompt,
    )
    .expect("small-tile AutoProbe trace");
    assert!(matches!(
        small_tile.resolved_route,
        GgmlResolvedKernelRoute::DenseMmSimdgroup | GgmlResolvedKernelRoute::DenseMmTensorV1
    ));

    encoder.commit_and_wait().expect("execute traced routes");
}

#[test]
fn trace_rejects_graph_capture_that_would_not_execute() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: 8,
        k: 32,
        ggml_type: GgmlType::Q8_0,
    };
    let (input, weight, output) = buffers(&device, params);
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_capture();
    assert!(quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &params,
        &GgmlRoutingPolicy::default(),
        GgmlWorkloadClass::DecodeSingle,
    )
    .is_err());
    assert!(encoder
        .take_capture()
        .expect("capture still active")
        .is_empty());
}

#[test]
fn trace_binds_the_actual_fused_gate_up_dispatch_once() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let args = FusedGateUpSiluQ4_KArgs {
        m: 1,
        intermediate_size: 8,
        hidden_size: 256,
    };
    let gate = device
        .alloc_buffer(8 * 144, DType::U8, vec![8 * 144])
        .expect("gate");
    let up = device
        .alloc_buffer(8 * 144, DType::U8, vec![8 * 144])
        .expect("up");
    let input = device
        .alloc_buffer(256 * 4, DType::F32, vec![1, 256])
        .expect("input");
    let output = device
        .alloc_buffer(8 * 4, DType::F32, vec![1, 8])
        .expect("output");
    let mut encoder = device.command_encoder().expect("encoder");
    let trace = dispatch_fused_gate_up_silu_q4_K_with_trace(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        &input,
        &output,
        args,
        &GgmlRoutingPolicy::default(),
        GgmlWorkloadClass::DecodeSingle,
    )
    .expect("fused trace");
    assert_eq!(
        trace.resolved_route,
        GgmlResolvedKernelRoute::DenseGateUpSilu
    );
    assert_eq!(trace.dispatches.len(), 1);
    assert_eq!(
        trace.dispatches[0].pipeline.kernel_name,
        "kernel_fused_gate_up_silu_q4_K_f32"
    );
    encoder.commit_and_wait().expect("execute fused dispatch");
}
