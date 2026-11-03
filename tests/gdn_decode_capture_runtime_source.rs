//! The gated delta net decode capture kernels must build from runtime source
//! when the precompiled metallib is disabled. `KernelRegistry::new()` alone
//! (no `ops::gated_delta_net_decode::register` call) must resolve them.
//!
//! One test per binary: `MLX_PRECOMPILED_METALLIB` is latched on first use.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used)]

use mlx_native::{KernelPipelineOrigin, KernelRegistry, MlxDevice};

#[test]
fn gdn_decode_capture_pipelines_build_from_runtime_source() {
    std::env::set_var("MLX_PRECOMPILED_METALLIB", "0");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for name in [
        "gated_delta_net_decode_capture_f32_1",
        "gated_delta_net_decode_capture_f32_2",
        "gated_delta_net_decode_capture_f32_4",
    ] {
        let label = registry
            .get_pipeline(name, device.metal_device())
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .label()
            .to_owned();
        let identity = registry.pipeline_identity(&label).expect("identity");
        assert_eq!(identity.origin, KernelPipelineOrigin::RuntimeSource, "{name}");
        assert!(identity.runtime_source_sha256.is_some(), "{name}");
    }
}
