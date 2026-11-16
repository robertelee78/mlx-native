//! Logical activation precision, independent of the output buffer storage.

/// Rounding at complete producer boundaries; accumulation remains F32.
///
/// `Bf16` rounds to nearest-even BF16, then widens exactly when the output
/// uses F32 storage. Fused operations document any intermediate producer
/// boundaries they round as well. Neither mode allocates activation storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationPrecision {
    F32,
    Bf16,
}

pub(crate) fn register(registry: &mut crate::KernelRegistry) {
    for (name, source) in [
        ("rms_norm_f32_bf16_boundary", super::rms_norm::RMS_NORM_SHADER_SOURCE),
        ("rms_norm_f32_v2_bf16_boundary", super::rms_norm::RMS_NORM_SHADER_SOURCE),
        ("dflash2_dynamic_conv_f32_bf16_boundary", super::dflash2_dynamic_conv::DFLASH2_DYNAMIC_CONV_SOURCE),
        ("scalar_mul_f32_bf16_boundary", include_str!("../shaders/elementwise.metal")),
        ("embedding_gather_bf16_f32_bf16_boundary", super::embedding_dense::EMBEDDING_DENSE_SHADER_SOURCE),
        ("embedding_gather_f16_f32_bf16_boundary", super::embedding_dense::EMBEDDING_DENSE_SHADER_SOURCE),
        ("embedding_gather_f32_f32_bf16_boundary", super::embedding_dense::EMBEDDING_DENSE_SHADER_SOURCE),
        ("embedding_gather_q4_0_f32_bf16_boundary", super::embedding_q4_0::EMBEDDING_Q4_0_SHADER_SOURCE),
        ("embedding_gather_q5_0_f32_bf16_boundary", super::embedding_q5_0::EMBEDDING_Q5_0_SHADER_SOURCE),
        ("embedding_gather_q2_k_f32_bf16_boundary", super::embedding_q2_k::EMBEDDING_Q2_K_SHADER_SOURCE),
        ("embedding_gather_q4_k_f32_bf16_boundary", super::embedding_q4_k::EMBEDDING_Q4_K_SHADER_SOURCE),
        ("embedding_gather_q5_k_f32_bf16_boundary", super::embedding_kquant::EMBEDDING_KQUANT_SHADER_SOURCE),
        ("embedding_gather_q6_k_f32_bf16_boundary", super::embedding_kquant::EMBEDDING_KQUANT_SHADER_SOURCE),
        ("embedding_gather_q8_0_f32_bf16_boundary", super::embedding_q8_0::EMBEDDING_Q8_0_SHADER_SOURCE),
        ("qmm_affine_t_packed_simd4_b4_gs64_bf16_precision_bf16", super::qmm_affine::QMM_AFFINE_T_PACKED_SIMD4_B4_GS64_BF16_SHADER_SOURCE),
        ("dflash2_af4g64_staged_m8_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_m16_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_m32_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_reduce_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_gate_up_m8_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_gate_up_m16_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_gate_up_m32_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
        ("dflash2_af4g64_staged_gate_up_reduce_precision_bf16", super::dflash2_af4g64_staged::SOURCE),
    ] {
        registry.register_source(name, source);
    }
}
