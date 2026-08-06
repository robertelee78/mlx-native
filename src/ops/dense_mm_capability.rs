use std::sync::OnceLock;

use crate::device::MlxDevice;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenseMmBackend {
    Auto,
    #[cfg(test)]
    TensorRequired,
    #[cfg(test)]
    FallbackRequired,
}

fn force_fallback_from_env() -> bool {
    static FORCE_FALLBACK: OnceLock<bool> = OnceLock::new();
    *FORCE_FALLBACK
        .get_or_init(|| std::env::var("MLX_NATIVE_DISABLE_METAL_TENSOR").as_deref() == Ok("1"))
}

pub(super) fn is_unavailable_tensor_header(error: &MlxError) -> bool {
    matches!(
        error,
        MlxError::ShaderCompilationError { message, .. }
            if message.contains("fatal error")
                && message.contains("'metal_tensor' file not found")
    )
}

/// Resolve a dense matrix-multiply backend without caching transient or
/// unexpected failures.  Automatic capability state belongs to the
/// device-scoped `KernelRegistry`; only the exact missing tensor header is a
/// supported fallback condition.
pub(super) fn tensor_pipeline_available(
    backend: DenseMmBackend,
    pipeline_name: &str,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<bool> {
    let supported = match backend {
        #[cfg(test)]
        DenseMmBackend::FallbackRequired => false,
        #[cfg(test)]
        DenseMmBackend::TensorRequired => {
            registry.get_pipeline(pipeline_name, device.metal_device())?;
            true
        }
        DenseMmBackend::Auto if force_fallback_from_env() => false,
        DenseMmBackend::Auto => registry.probe_optional_pipeline(
            pipeline_name,
            device.metal_device(),
            is_unavailable_tensor_header,
        )?,
    };

    if std::env::var("MLX_LOG_TENSOR_PROBE").is_ok() {
        eprintln!(
            "[mlx-native] {pipeline_name}: {}",
            if supported {
                "using tensor API"
            } else {
                "using tiled simdgroup fallback"
            }
        );
    }
    Ok(supported)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_tensor_header_classifier_is_exact() {
        let missing = MlxError::ShaderCompilationError {
            name: "tensor".into(),
            message: "fatal error: 'metal_tensor' file not found".into(),
        };
        let unrelated = MlxError::ShaderCompilationError {
            name: "tensor".into(),
            message: "fatal error: use of undeclared identifier 'broken'".into(),
        };
        let misleading = MlxError::ShaderCompilationError {
            name: "tensor".into(),
            message: "note: 'metal_tensor' file not found in cached diagnostics".into(),
        };

        assert!(is_unavailable_tensor_header(&missing));
        assert!(!is_unavailable_tensor_header(&unrelated));
        assert!(!is_unavailable_tensor_header(&misleading));
        assert!(!is_unavailable_tensor_header(&MlxError::KernelNotFound(
            "tensor".into()
        )));
    }
}
