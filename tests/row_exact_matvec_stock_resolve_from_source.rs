//! Every stock-order row-exact mat-vec kernel builds from the source the
//! `KernelRegistry` constructor carries when the precompiled metallib is
//! disabled (no `row_exact_matvec_stock::register` call needed), and the pipeline
//! identity binds the sha256 of `row_exact_matvec_stock.metal`.
//!
//! This binary holds one test: `MLX_PRECOMPILED_METALLIB` is read once per
//! process, before the first pipeline lookup.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used)]

use mlx_native::ops::row_exact_matvec_stock::{
    row_exact_matvec_stock_kernel_names, ROW_EXACT_MATVEC_STOCK_SHADER_SOURCE,
};
use mlx_native::{KernelPipelineOrigin, KernelRegistry, MlxDevice};
use sha2::{Digest, Sha256};

#[test]
fn row_exact_matvec_stock_kernels_resolve_from_source() {
    std::env::set_var("MLX_PRECOMPILED_METALLIB", "0");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let expected_sha256 = format!(
        "{:x}",
        Sha256::digest(ROW_EXACT_MATVEC_STOCK_SHADER_SOURCE.as_bytes())
    );
    let mut count = 0;
    for name in row_exact_matvec_stock_kernel_names() {
        registry
            .get_pipeline(name, device.metal_device())
            .unwrap_or_else(|error| panic!("{name} failed to build from source: {error}"));
        let identity = registry.pipeline_identity(name).expect("pipeline identity");
        assert_eq!(
            identity.origin,
            KernelPipelineOrigin::RuntimeSource,
            "{name} must come from its registered source"
        );
        assert_eq!(
            identity.runtime_source_sha256.as_deref(),
            Some(expected_sha256.as_str()),
            "{name} source digest"
        );
        count += 1;
    }
    assert_eq!(count, 80);
}
