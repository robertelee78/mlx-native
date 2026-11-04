//! ADR-066 N-IQ2XXS: every IQ2_XXS kernel name builds from its registered
//! source with the precompiled metallib disabled, so no name resolves only
//! through the embedded library (the IQ4_XS gap, ADR-066 Context item 2).
//! The pipeline identity must bind the sha256 of the exact shader file.
//!
//! This binary holds one test: `MLX_PRECOMPILED_METALLIB` is read once per
//! process, before the first pipeline lookup.

use mlx_native::{KernelPipelineOrigin, KernelRegistry, MlxDevice};
use sha2::{Digest, Sha256};

const IQ2_XXS_KERNELS: [(&str, &str); 3] = [
    (
        "kernel_mul_mv_id_iq2_xxs_f32",
        include_str!("../src/shaders/quantized_matmul_id_ggml.metal"),
    ),
    (
        "kernel_mul_mm_id_iq2_xxs_f32",
        include_str!("../src/shaders/quantized_matmul_id_mm.metal"),
    ),
    (
        "kernel_mul_mm_id_iq2_xxs_tensor_f32",
        include_str!("../src/shaders/quantized_matmul_id_mm_tensor.metal"),
    ),
];
const TENSOR_REFERENCE: &str = "kernel_mul_mm_id_q4_0_tensor_f32";

#[test]
fn iq2_kernels_resolve_from_source() {
    std::env::set_var("MLX_PRECOMPILED_METALLIB", "0");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let tensor_available = registry
        .get_pipeline(TENSOR_REFERENCE, device.metal_device())
        .is_ok();
    for (name, source) in IQ2_XXS_KERNELS {
        let is_tensor = name.contains("_tensor_");
        let result = registry.get_pipeline(name, device.metal_device());
        if is_tensor && !tensor_available {
            assert!(result.is_err(), "{name} built without tensor-API support");
            continue;
        }
        result.unwrap_or_else(|error| panic!("{name} failed to build from source: {error}"));
        let identity = registry.pipeline_identity(name).expect("pipeline identity");
        assert_eq!(
            identity.origin,
            KernelPipelineOrigin::RuntimeSource,
            "{name} must come from its registered source"
        );
        let expected_sha256 = format!("{:x}", Sha256::digest(source.as_bytes()));
        assert_eq!(
            identity.runtime_source_sha256.as_deref(),
            Some(expected_sha256.as_str()),
            "{name} source digest"
        );
        assert!(identity.embedded_metallib_sha256.is_none(), "{name}");
    }
}
