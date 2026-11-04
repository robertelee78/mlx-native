//! Every kernel in the embedded metallib that production dispatches must
//! build from runtime source when the precompiled metallib is disabled, from
//! `KernelRegistry::new()` alone. This covers the kernels an op's `register`
//! function adds (constructor registration fallbacks), checked against the
//! source text that `register` installs, and the IQ4_XS matmul kernels the
//! constructor carries directly.
//!
//! One test per binary: `MLX_PRECOMPILED_METALLIB` is latched on first use.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used)]

use mlx_native::ops::{
    add_bias_row_2d, bilinear_resize_2d, block_merge_2x2, deepseek_moe_routing, deepseek_tail_rope,
    embedding_kquant, embedding_q2_k, embedding_q8_0, feature_concat, flash_attn_prefill,
    flash_attn_prefill_blk, flash_attn_prefill_d512, flash_attn_prefill_mask, flash_attn_train,
    flash_attn_vec_tq, fused_residual_norm, gated_delta_net_decode, gather_bench, hadamard,
    hadamard_quantize_kv, im2col_2d_3ch, ssm_conv, vision_2d_rope,
};
use mlx_native::{KernelPipelineOrigin, KernelRegistry, MlxDevice};
use sha2::{Digest, Sha256};

const NO_BOOLS: &[(usize, bool)] = &[];
const NO_INTS: &[(usize, i32)] = &[];
/// `flash_attn_prefill.metal`: align_Q, align_K, has_mask, do_causal, has_blk.
const PREFILL_BOOLS: &[(usize, bool)] = &[
    (200, false),
    (201, false),
    (300, false),
    (301, false),
    (303, false),
];
/// `flash_attn_prefill_d512.metal`: the prefill flags plus 304 and 305.
const D512_BOOLS: &[(usize, bool)] = &[
    (200, false),
    (201, false),
    (300, false),
    (301, false),
    (303, false),
    (304, false),
    (305, false),
];
/// `flash_attn_prefill_d512.metal`: fc_nsg.
const D512_INTS: &[(usize, i32)] = &[(322, 8)];
/// `flash_attn_prefill_blk.metal`: BQ, BK for D=256.
const BLK_INTS: &[(usize, i32)] = &[(400, 32), (401, 16)];
/// `flash_attn_train_{fwd,bwd}.metal`: align_Q, align_K, has_mask, do_causal.
const TRAIN_BOOLS: &[(usize, bool)] = &[(200, false), (201, false), (300, false), (301, false)];
/// `quantized_matmul_ggml.metal`: ne12, r2, r3 as `dispatch_mv` sets them.
const QMATMUL_MV_INTS: &[(usize, i32)] = &[(700, 1), (701, 1), (702, 1)];

type Case = (
    String,
    &'static str,
    &'static [(usize, bool)],
    &'static [(usize, i32)],
);

fn cases() -> Vec<Case> {
    let fixed: [(
        &str,
        &'static str,
        &'static [(usize, bool)],
        &'static [(usize, i32)],
    ); 32] = [
        (
            "add_bias_row_2d_f32",
            add_bias_row_2d::ADD_BIAS_ROW_2D_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "bilinear_resize_2d_f32",
            bilinear_resize_2d::BILINEAR_RESIZE_2D_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "block_merge_2x2_f32",
            block_merge_2x2::BLOCK_MERGE_2X2_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "deepseek_moe_sanitize_indices",
            deepseek_moe_routing::DEEPSEEK_MOE_ROUTING_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "deepseek_tail_rope_f32_to_f16",
            deepseek_tail_rope::DEEPSEEK_TAIL_ROPE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "deepseek_tail_rope_f16_to_bf16",
            deepseek_tail_rope::DEEPSEEK_TAIL_ROPE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "embedding_gather_q5_k_f32",
            embedding_kquant::EMBEDDING_KQUANT_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "embedding_gather_q6_k_f32",
            embedding_kquant::EMBEDDING_KQUANT_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "embedding_gather_q2_k_f32",
            embedding_q2_k::EMBEDDING_Q2_K_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "embedding_gather_q8_0_f32",
            embedding_q8_0::EMBEDDING_Q8_0_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "feature_concat_f32",
            feature_concat::FEATURE_CONCAT_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_prefill_blk_bf16",
            flash_attn_prefill_blk::FLASH_ATTN_PREFILL_BLK_SHADER_SOURCE,
            NO_BOOLS,
            BLK_INTS,
        ),
        (
            "flash_attn_prefill_blk_f16",
            flash_attn_prefill_blk::FLASH_ATTN_PREFILL_BLK_SHADER_SOURCE,
            NO_BOOLS,
            BLK_INTS,
        ),
        (
            "flash_attn_prefill_mask_fill_bf16",
            flash_attn_prefill_mask::FLASH_ATTN_PREFILL_MASK_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_prefill_mask_fill_f16",
            flash_attn_prefill_mask::FLASH_ATTN_PREFILL_MASK_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_prefill_mask_fill_blockdiag_bf16",
            flash_attn_prefill_mask::FLASH_ATTN_PREFILL_MASK_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_train_bwd_compute_d_bf16",
            flash_attn_train::FLASH_ATTN_TRAIN_BWD_COMPUTE_D_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_vec_tq_dk256",
            flash_attn_vec_tq::FLASH_ATTN_VEC_TQ_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "flash_attn_vec_tq_dk512",
            flash_attn_vec_tq::FLASH_ATTN_VEC_TQ_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "fused_residual_norm_bf16",
            fused_residual_norm::FUSED_RESIDUAL_NORM_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "gather_bench_nibble",
            gather_bench::GATHER_BENCH_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "gather_bench_f16_seq",
            gather_bench::GATHER_BENCH_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "hadamard_transform",
            hadamard::HADAMARD_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "hadamard_quantize_kv",
            hadamard_quantize_kv::HADAMARD_QUANTIZE_KV_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "im2col_2d_3ch_f32",
            im2col_2d_3ch::IM2COL_2D_3CH_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "ssm_conv_capture_forward_f32",
            ssm_conv::SSM_CONV_CAPTURE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "vision_2d_rope_f32",
            vision_2d_rope::VISION_2D_ROPE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "vision_2d_rope_bf16",
            vision_2d_rope::VISION_2D_ROPE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "kernel_mul_mv_iq4_xs_f32",
            include_str!("../src/shaders/quantized_matmul_ggml.metal"),
            NO_BOOLS,
            QMATMUL_MV_INTS,
        ),
        (
            "kernel_mul_mv_id_iq4_xs_f32",
            include_str!("../src/shaders/quantized_matmul_id_ggml.metal"),
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "kernel_mul_mm_id_iq4_xs_f32",
            include_str!("../src/shaders/quantized_matmul_id_mm.metal"),
            NO_BOOLS,
            NO_INTS,
        ),
        (
            "kernel_mul_mm_id_iq4_xs_tensor_f32",
            include_str!("../src/shaders/quantized_matmul_id_mm_tensor.metal"),
            NO_BOOLS,
            NO_INTS,
        ),
    ];
    let mut cases: Vec<Case> = fixed
        .into_iter()
        .map(|(name, source, bools, ints)| (name.to_owned(), source, bools, ints))
        .collect();
    for dtype in ["bf16", "f16"] {
        for d in ["d64", "d256", "d512"] {
            for suffix in ["", "_boolmask"] {
                let name = format!("flash_attn_prefill_{dtype}_{d}{suffix}");
                cases.push((
                    name,
                    flash_attn_prefill::FLASH_ATTN_PREFILL_SHADER_SOURCE,
                    PREFILL_BOOLS,
                    NO_INTS,
                ));
            }
        }
        for suffix in ["", "_boolmask"] {
            let name = format!("flash_attn_prefill_d512_{dtype}{suffix}");
            cases.push((
                name,
                flash_attn_prefill_d512::FLASH_ATTN_PREFILL_D512_SHADER_SOURCE,
                D512_BOOLS,
                D512_INTS,
            ));
        }
    }
    for name in [
        "flash_attn_train_fwd_bf16_d64",
        "flash_attn_train_fwd_bf16_d64_boolmask",
        "flash_attn_train_fwd_bf16_d256",
        "flash_attn_train_fwd_bf16_d256_boolmask",
    ] {
        cases.push((
            name.to_owned(),
            flash_attn_train::FLASH_ATTN_TRAIN_FWD_SHADER_SOURCE,
            TRAIN_BOOLS,
            NO_INTS,
        ));
    }
    for name in [
        "flash_attn_train_bwd_bf16_d64",
        "flash_attn_train_bwd_bf16_d256",
        "f32_to_bf16_cast",
    ] {
        cases.push((
            name.to_owned(),
            flash_attn_train::FLASH_ATTN_TRAIN_BWD_SHADER_SOURCE,
            TRAIN_BOOLS,
            NO_INTS,
        ));
    }
    for name in [
        "gated_delta_net_decode_capture_f32_1",
        "gated_delta_net_decode_capture_f32_2",
        "gated_delta_net_decode_capture_f32_4",
    ] {
        cases.push((
            name.to_owned(),
            gated_delta_net_decode::GATED_DELTA_NET_DECODE_CAPTURE_SHADER_SOURCE,
            NO_BOOLS,
            NO_INTS,
        ));
    }
    cases
}

#[test]
fn registration_fallbacks_build_from_runtime_source() {
    std::env::set_var("MLX_PRECOMPILED_METALLIB", "0");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let cases = cases();
    let unique: std::collections::BTreeSet<&str> =
        cases.iter().map(|case| case.0.as_str()).collect();
    // 54 constructor fallbacks (unit test in kernel_registry.rs) + 4 IQ4_XS.
    assert_eq!(unique.len(), 58);
    assert_eq!(cases.len(), 58);
    for (name, source, bools, ints) in cases {
        let pipeline = if bools.is_empty() && ints.is_empty() {
            registry.get_pipeline(&name, device.metal_device())
        } else {
            registry.get_pipeline_with_constants(&name, device.metal_device(), bools, ints)
        };
        let label = pipeline
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .label()
            .to_owned();
        let identity = registry.pipeline_identity(&label).expect("identity");
        assert_eq!(
            identity.origin,
            KernelPipelineOrigin::RuntimeSource,
            "{name}"
        );
        let expected = format!("{:x}", Sha256::digest(source.as_bytes()));
        assert_eq!(
            identity.runtime_source_sha256.as_deref(),
            Some(expected.as_str()),
            "{name}"
        );
        assert!(identity.embedded_metallib_sha256.is_none(), "{name}");
    }
}
