// Test-only source override: same shader plus a comment forces runtime path.
fn selected_source(source: &'static str) -> (&'static str, String) {
    use sha2::{Digest, Sha256};
    let mut changed = String::with_capacity(source.len() + 64);
    changed.push_str(source);
    changed.push_str("\n// selected-shader mandatory runtime-source control\n");
    assert!(changed.capacity() <= 128 * 1024);
    let hash = format!("{:x}", Sha256::digest(changed.as_bytes()));
    (Box::leak(changed.into_boxed_str()), hash)
}
fn selected_origin(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    name: &str,
    flags: &[(usize, bool)],
    expected_hash: &str,
) {
    let label = registry
        .get_pipeline_with_bool_constants(name, device.metal_device(), flags)
        .expect("mandatory selected runtime-source pipeline")
        .label()
        .to_owned();
    let identity = registry
        .pipeline_identity(&label)
        .expect("pipeline identity");
    assert_eq!(
        identity.origin,
        mlx_native::KernelPipelineOrigin::RuntimeSource
    );
    assert_eq!(
        identity.runtime_source_sha256.as_deref(),
        Some(expected_hash)
    );
    assert!(identity.embedded_metallib_sha256.is_none());
    eprintln!(
        "SELECTED_SHADER_RUNTIME kernel={name} origin=RuntimeSource source_sha256={expected_hash}"
    );
}
fn selected_environment() {
    for name in [
        "MLX_UNRETAINED_REFS",
        "MLX_PROFILE_DISPATCH",
        "MLX_PROFILE_CB",
        "MLX_ENCODE_TRACE",
        "MLX_PROFILE_BARRIERS",
        "HF2Q_AUTO_BARRIER",
        "HF2Q_FORCE_SERIAL_DISPATCH",
        "HF2Q_PIPELINE_TG_MULT_HINT",
        "MLX_METAL_CAPTURE",
        "METAL_CAPTURE_ENABLED",
    ] {
        assert!(std::env::var_os(name).is_none(), "requires {name} absent");
    }
}
