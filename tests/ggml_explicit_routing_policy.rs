//! The serialized GGUF routing policy must control the public dispatch that
//! a cost receipt measures. Process-global diagnostic environment must not be
//! able to silently select a different kernel for the explicit entry point.

#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    embedding_gather_q2_k, embedding_gather_q8_0, quantized_matmul_ggml_with_policy,
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_pooled_pair_with_policy,
    quantized_matmul_id_ggml_pooled_with_policy, quantized_matmul_id_swiglu_q4_0, CapturedNode,
    DType, EmbeddingQ2KParams, EmbeddingQ8_0Params, GgmlQuantizedMatmulIdParams,
    GgmlQuantizedMatmulParams, GgmlRoutingPolicy, GgmlType, IdMmScratch, KernelRegistry, MlxDevice,
};

use mlx_native::ops::fused_gate_up_silu_iq4_nl::{
    dispatch_fused_gate_up_silu_iq4_nl, FusedGateUpSiluIq4NlArgs,
};
use mlx_native::ops::fused_gate_up_silu_q4_K::{
    dispatch_fused_gate_up_silu_q4_K, FusedGateUpSiluQ4_KArgs,
};
use mlx_native::ops::fused_gate_up_silu_q5_K::{
    dispatch_fused_gate_up_silu_q5_K, FusedGateUpSiluQ5_KArgs,
};
use mlx_native::ops::fused_gate_up_silu_q6_K::{
    dispatch_fused_gate_up_silu_q6_K, FusedGateUpSiluQ6_KArgs,
};
use mlx_native::ops::fused_gate_up_silu_q8_0::{
    dispatch_fused_gate_up_silu_q8_0, FusedGateUpSiluQ8_0Args,
};
use mlx_native::ops::quantized_matmul_ggml::{
    build_q6k_nr2_m1_record, build_q6k_nr2_m1_record_with_policy,
};
use mlx_native::ops::quantized_matmul_id_ggml::{
    build_q6k_id_nr2_m1_record, build_q6k_id_nr2_m1_record_with_policy,
    build_q8_0_id_decode_record, build_q8_0_id_decode_record_with_policy,
};

#[test]
fn fused_and_embedding_entrypoints_reject_short_logical_views() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();

    macro_rules! reject_short_fused_weight {
        ($dispatch:ident, $args:expr, $weight_bytes:expr, $hidden:expr, $intermediate:expr) => {{
            let weight_bytes = $weight_bytes;
            let storage = device
                .alloc_buffer(weight_bytes + 17, DType::U8, vec![weight_bytes + 17])
                .expect("weight storage");
            let exact = storage.slice_view(16, weight_bytes);
            let short = storage.slice_view(16, weight_bytes - 1);
            let input = device
                .alloc_buffer($hidden * 4, DType::F32, vec![$hidden])
                .expect("input");
            let output = device
                .alloc_buffer($intermediate * 4, DType::F32, vec![$intermediate])
                .expect("output");
            let mut encoder = device.command_encoder().expect("encoder");
            assert!($dispatch(
                &mut encoder,
                &mut registry,
                &device,
                &short,
                &exact,
                &input,
                &output,
                $args,
            )
            .is_err());
        }};
    }

    reject_short_fused_weight!(
        dispatch_fused_gate_up_silu_q8_0,
        FusedGateUpSiluQ8_0Args {
            m: 1,
            intermediate_size: 32,
            hidden_size: 32,
        },
        32 * 34,
        32,
        32
    );
    reject_short_fused_weight!(
        dispatch_fused_gate_up_silu_q4_K,
        FusedGateUpSiluQ4_KArgs {
            m: 1,
            intermediate_size: 1,
            hidden_size: 256,
        },
        144,
        256,
        1
    );
    reject_short_fused_weight!(
        dispatch_fused_gate_up_silu_q5_K,
        FusedGateUpSiluQ5_KArgs {
            m: 1,
            intermediate_size: 1,
            hidden_size: 256,
        },
        176,
        256,
        1
    );
    reject_short_fused_weight!(
        dispatch_fused_gate_up_silu_q6_K,
        FusedGateUpSiluQ6_KArgs {
            m: 1,
            intermediate_size: 1,
            hidden_size: 256,
        },
        210,
        256,
        1
    );
    reject_short_fused_weight!(
        dispatch_fused_gate_up_silu_iq4_nl,
        FusedGateUpSiluIq4NlArgs {
            m: 1,
            intermediate_size: 32,
            hidden_size: 32,
        },
        32 * 18,
        32,
        32
    );

    let q2_weight = device
        .alloc_buffer(168, DType::U8, vec![168])
        .expect("q2 weight");
    let q2_ids_storage = device
        .alloc_buffer(12, DType::U32, vec![3])
        .expect("q2 ids storage");
    let q2_ids_short = q2_ids_storage.slice_view(4, 1);
    let q2_output = device
        .alloc_buffer(2 * 256 * 4, DType::F32, vec![2, 256])
        .expect("q2 output");
    let mut encoder = device.command_encoder().expect("q2 encoder");
    assert!(embedding_gather_q2_k(
        &mut encoder,
        &mut registry,
        &device,
        &q2_weight,
        &q2_ids_short,
        &q2_output,
        &EmbeddingQ2KParams {
            vocab_size: 2,
            embed_dim: 256,
            n_tokens: 2,
        },
    )
    .is_err());

    let q8_weight = device
        .alloc_buffer(68, DType::U8, vec![68])
        .expect("q8 weight");
    let q8_ids = device.alloc_buffer(8, DType::U32, vec![2]).expect("q8 ids");
    let q8_output_storage = device
        .alloc_buffer(2 * 32 * 4 + 17, DType::F32, vec![2 * 32 + 4])
        .expect("q8 output storage");
    let q8_output_short = q8_output_storage.slice_view(16, 2 * 32 - 1);
    let mut encoder = device.command_encoder().expect("q8 encoder");
    assert!(embedding_gather_q8_0(
        &mut encoder,
        &mut registry,
        &device,
        &q8_weight,
        &q8_ids,
        &q8_output_short,
        &EmbeddingQ8_0Params {
            vocab_size: 2,
            embed_dim: 32,
            n_tokens: 2,
        },
    )
    .is_err());
}

#[test]
fn explicit_q8_policy_selects_baseline_or_nr2_geometry() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let weight = device
        .alloc_buffer(34 * 32, DType::U8, vec![34 * 32])
        .expect("weight");
    let input = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("input");
    let output = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("output");
    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: 32,
        k: 32,
        ggml_type: GgmlType::Q8_0,
    };

    let mut baseline = GgmlRoutingPolicy::default();
    baseline.dense_q8_0_mv_nr2 = false;
    let mut encoder = device.command_encoder().expect("baseline encoder");
    encoder.start_capture();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &params,
        &baseline,
    )
    .expect("baseline capture");
    let captured = encoder.take_capture().expect("baseline graph");
    match &captured[0] {
        CapturedNode::Dispatch {
            threads_per_threadgroup,
            threadgroup_memory,
            ..
        } => {
            assert_eq!(
                (
                    threads_per_threadgroup.width,
                    threads_per_threadgroup.height
                ),
                (8, 8)
            );
            assert!(threadgroup_memory.is_empty());
        }
        CapturedNode::Barrier => panic!("expected baseline dispatch"),
    }

    let mut nr2 = baseline;
    nr2.dense_q8_0_mv_nr2 = true;
    let mut encoder = device.command_encoder().expect("nr2 encoder");
    encoder.start_capture();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &params,
        &nr2,
    )
    .expect("nr2 capture");
    let captured = encoder.take_capture().expect("nr2 graph");
    match &captured[0] {
        CapturedNode::Dispatch {
            threads_per_threadgroup,
            threadgroup_memory,
            ..
        } => {
            assert_eq!(
                (
                    threads_per_threadgroup.width,
                    threads_per_threadgroup.height
                ),
                (32, 4)
            );
            assert_eq!(threadgroup_memory, &vec![(0, 2 * 32 * 4)]);
        }
        CapturedNode::Barrier => panic!("expected NR2 dispatch"),
    }
}

#[test]
fn padded_expert_entrypoints_reject_short_logical_views() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let policy = GgmlRoutingPolicy::default();

    let q8_matrix_bytes = 32 * 34;
    let q8_stride = q8_matrix_bytes + 64;
    let q8_required = q8_stride + q8_matrix_bytes;
    let q8_storage = device
        .alloc_buffer(q8_required + 17, DType::U8, vec![q8_required + 17])
        .expect("q8 storage");
    let q8_exact = q8_storage.slice_view(16, q8_required);
    let q8_short = q8_storage.slice_view(16, q8_required - 1);

    let mv_input = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("mv input");
    let mv_ids = device.alloc_buffer(4, DType::U32, vec![1]).expect("mv ids");
    let mv_output = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("mv output");
    let mv_params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: 1,
        n: 32,
        k: 32,
        n_experts: 2,
        expert_stride: q8_stride as u64,
        ggml_type: GgmlType::Q8_0,
    };
    let mut encoder = device.command_encoder().expect("mv encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mv_input,
        &q8_exact,
        &mv_ids,
        &mv_output,
        &mv_params,
        &policy,
    )
    .expect("exact padded MV");
    assert_eq!(encoder.take_capture().expect("mv graph").len(), 1);

    let mut encoder = device.command_encoder().expect("short mv encoder");
    assert!(quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mv_input,
        &q8_short,
        &mv_ids,
        &mv_output,
        &mv_params,
        &policy,
    )
    .is_err());

    let rows = 33usize;
    let mm_input = device
        .alloc_buffer(rows * 32 * 4, DType::F32, vec![rows, 32])
        .expect("mm input");
    let mm_ids = device
        .alloc_buffer(rows * 4, DType::U32, vec![rows])
        .expect("mm ids");
    let first_output = device
        .alloc_buffer(rows * 32 * 4, DType::F32, vec![rows, 32])
        .expect("first output");
    let second_output = device
        .alloc_buffer(rows * 32 * 4, DType::F32, vec![rows, 32])
        .expect("second output");
    let mm_params = GgmlQuantizedMatmulIdParams {
        n_tokens: rows as u32,
        ..mv_params
    };
    let mut scratch = IdMmScratch::alloc(&device, 2, rows as u32).expect("scratch");
    let mut encoder = device.command_encoder().expect("pooled encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_pooled_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mm_input,
        &q8_exact,
        &mm_ids,
        &first_output,
        &mut scratch,
        &mm_params,
        &policy,
    )
    .expect("exact padded pooled MM");
    assert_eq!(encoder.take_capture().expect("pooled graph").len(), 3);

    let mut encoder = device.command_encoder().expect("short pooled encoder");
    assert!(quantized_matmul_id_ggml_pooled_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mm_input,
        &q8_short,
        &mm_ids,
        &first_output,
        &mut scratch,
        &mm_params,
        &policy,
    )
    .is_err());

    let mut encoder = device.command_encoder().expect("pair encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_pooled_pair_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mm_input,
        &q8_exact,
        &q8_exact,
        &mm_ids,
        &first_output,
        &second_output,
        &mut scratch,
        &mm_params,
        &policy,
    )
    .expect("exact padded pair");
    assert_eq!(encoder.take_capture().expect("pair graph").len(), 4);

    let mut encoder = device.command_encoder().expect("short pair encoder");
    assert!(quantized_matmul_id_ggml_pooled_pair_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mm_input,
        &q8_exact,
        &q8_short,
        &mm_ids,
        &first_output,
        &second_output,
        &mut scratch,
        &mm_params,
        &policy,
    )
    .is_err());

    let q4_matrix_bytes = 32 * 18;
    let q4_stride = q4_matrix_bytes + 64;
    let q4_required = q4_stride + q4_matrix_bytes;
    let q4_storage = device
        .alloc_buffer(q4_required + 17, DType::U8, vec![q4_required + 17])
        .expect("q4 storage");
    let q4_exact = q4_storage.slice_view(16, q4_required);
    let q4_short = q4_storage.slice_view(16, q4_required - 1);
    let q4_params = GgmlQuantizedMatmulIdParams {
        expert_stride: q4_stride as u64,
        ggml_type: GgmlType::Q4_0,
        ..mv_params
    };
    let gate = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("gate");
    let up = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("up");
    let mut encoder = device.command_encoder().expect("swiglu encoder");
    encoder.start_capture();
    quantized_matmul_id_swiglu_q4_0(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        &q4_exact,
        &mv_ids,
        &mv_output,
        &q4_params,
    )
    .expect("exact padded SwiGLU");
    assert_eq!(encoder.take_capture().expect("swiglu graph").len(), 1);

    let mut encoder = device.command_encoder().expect("short swiglu encoder");
    assert!(quantized_matmul_id_swiglu_q4_0(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        &q4_short,
        &mv_ids,
        &mv_output,
        &q4_params,
    )
    .is_err());
}

#[test]
fn explicit_expert_q8_policy_selects_baseline_or_nr2_geometry() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let per_expert_bytes = 32 * 34;
    let weight = device
        .alloc_buffer(2 * per_expert_bytes, DType::U8, vec![2 * per_expert_bytes])
        .expect("weight");
    let input = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("input");
    let ids = device.alloc_buffer(4, DType::U32, vec![1]).expect("ids");
    let output = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("output");
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: 1,
        n: 32,
        k: 32,
        n_experts: 2,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q8_0,
    };

    let mut baseline = GgmlRoutingPolicy::default();
    baseline.expert_q8_0_mv_nr2 = false;
    let mut encoder = device.command_encoder().expect("baseline encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &output,
        &params,
        &baseline,
    )
    .expect("baseline capture");
    let captured = encoder.take_capture().expect("baseline graph");
    match &captured[0] {
        CapturedNode::Dispatch {
            threads_per_threadgroup,
            threadgroup_memory,
            ..
        } => {
            assert_eq!(
                (
                    threads_per_threadgroup.width,
                    threads_per_threadgroup.height
                ),
                (8, 8)
            );
            assert!(threadgroup_memory.is_empty());
        }
        CapturedNode::Barrier => panic!("expected baseline dispatch"),
    }

    let mut nr2 = baseline;
    nr2.expert_q8_0_mv_nr2 = true;
    let mut encoder = device.command_encoder().expect("nr2 encoder");
    encoder.start_capture();
    quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &output,
        &params,
        &nr2,
    )
    .expect("nr2 capture");
    let captured = encoder.take_capture().expect("nr2 graph");
    match &captured[0] {
        CapturedNode::Dispatch {
            threads_per_threadgroup,
            threadgroup_memory,
            ..
        } => {
            assert_eq!(
                (
                    threads_per_threadgroup.width,
                    threads_per_threadgroup.height
                ),
                (32, 4)
            );
            assert_eq!(threadgroup_memory, &vec![(0, 2 * 32 * 4)]);
        }
        CapturedNode::Barrier => panic!("expected NR2 dispatch"),
    }
}

#[test]
fn explicit_dispatch_record_policy_helper() {
    if std::env::var_os("MLX_NATIVE_RECORD_POLICY_TEST_CHILD").is_none() {
        return;
    }
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();

    let mut policy = GgmlRoutingPolicy::default();
    policy.dense_q6k_mv_nr2 = true;
    policy.expert_q6k_mv_nr2 = true;
    policy.expert_q8_0_mv_nr2 = false;

    assert!(build_q6k_nr2_m1_record_with_policy(
        &mut registry,
        device.metal_device(),
        4,
        256,
        &policy,
    )
    .expect("explicit dense Q6 record")
    .is_some());
    assert!(build_q6k_id_nr2_m1_record_with_policy(
        &mut registry,
        device.metal_device(),
        4,
        256,
        1,
        4 * 210,
        &policy,
    )
    .expect("explicit expert Q6 record")
    .is_some());
    assert!(build_q8_0_id_decode_record_with_policy(
        &mut registry,
        device.metal_device(),
        32,
        32,
        1,
        32 * 34,
        &policy,
    )
    .expect("explicit expert Q8 record")
    .is_some());

    // The child environment requests the opposite routes. Legacy builders
    // must preserve those environment semantics while the explicit builders
    // above remain controlled solely by the serialized policy.
    assert!(
        build_q6k_nr2_m1_record(&mut registry, device.metal_device(), 4, 256)
            .expect("legacy dense Q6 record")
            .is_none()
    );
    assert!(
        build_q6k_id_nr2_m1_record(&mut registry, device.metal_device(), 4, 256, 1, 4 * 210,)
            .expect("legacy expert Q6 record")
            .is_none()
    );
    assert!(
        build_q8_0_id_decode_record(&mut registry, device.metal_device(), 32, 32, 1, 32 * 34,)
            .expect("legacy expert Q8 record")
            .is_none()
    );
}

#[test]
fn explicit_dispatch_records_ignore_conflicting_environment() {
    let status = std::process::Command::new(std::env::current_exe().expect("current test exe"))
        .arg("--exact")
        .arg("explicit_dispatch_record_policy_helper")
        .arg("--nocapture")
        .env("MLX_NATIVE_RECORD_POLICY_TEST_CHILD", "1")
        .env("HF2Q_Q6K_MV_NR2", "0")
        .env("HF2Q_Q6K_ID_MV_NR2", "0")
        .env("HF2Q_Q8_0_ID_MV_NR2", "1")
        .status()
        .expect("run isolated dispatch-record helper");
    assert!(status.success());
}
