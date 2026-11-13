use std::sync::OnceLock;

use crate::device::MlxDevice;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseMmBackend {
    Auto,
    TensorRequired,
    FallbackRequired,
}

pub(super) fn tensor_disabled_from_env() -> bool {
    static FORCE_FALLBACK: OnceLock<bool> = OnceLock::new();
    *FORCE_FALLBACK
        .get_or_init(|| std::env::var("MLX_NATIVE_DISABLE_METAL_TENSOR").as_deref() == Ok("1"))
}

/// Whether a failed tensor pipeline build means the Metal tensor API is
/// unavailable here, as opposed to a shader or toolchain regression.
///
/// Two conditions qualify, one per half of "toolchain or device":
///
/// * the runtime compiler has no `<metal_tensor>` header (macOS before 26);
/// * the device does not support the Metal 4 GPU family, which the tensor
///   API belongs to. Such a device can have the header and still be unable
///   to build a pipeline over it: the GitHub-hosted `macos-26` GPU ("Apple
///   Paravirtual device") compiles the source and specialises the function,
///   then fails pipeline creation with `unsupported
///   deferred-static-alloca-size function body in ...cooperative_tensor...`.
///
/// On a device that supports Metal 4 every other failure stays an error.
pub(super) fn is_unavailable_tensor_api(device: &MlxDevice, error: &MlxError) -> bool {
    tensor_api_unavailable(
        device
            .metal_device()
            .supports_family(metal::MTLGPUFamily::Metal4),
        error,
    )
}

fn tensor_api_unavailable(device_supports_metal4: bool, error: &MlxError) -> bool {
    matches!(
        error,
        MlxError::ShaderCompilationError { message, .. }
            if !device_supports_metal4
                || (message.contains("fatal error")
                    && message.contains("'metal_tensor' file not found"))
    )
}

/// Resolve a dense matrix-multiply backend without caching transient or
/// unexpected failures.  Automatic capability state belongs to the
/// device-scoped `KernelRegistry`; only an unavailable tensor API
/// ([`is_unavailable_tensor_api`]) is a supported fallback condition.
pub(super) fn tensor_pipeline_available(
    backend: DenseMmBackend,
    pipeline_name: &str,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<bool> {
    let supported = match backend {
        DenseMmBackend::FallbackRequired => false,
        // Exact callers perform the single authoritative pipeline lookup in
        // the dispatcher below; do not duplicate that hot-path lookup here.
        DenseMmBackend::TensorRequired => true,
        DenseMmBackend::Auto if tensor_disabled_from_env() => false,
        DenseMmBackend::Auto => {
            registry
                .probe_optional_pipeline(
                    pipeline_name,
                    device.metal_device(),
                    device.registry_id(),
                    |error| is_unavailable_tensor_api(device, error),
                )?
                .available
        }
    };

    if backend == DenseMmBackend::Auto && std::env::var("MLX_LOG_TENSOR_PROBE").is_ok() {
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

    /// The diagnostic of the GitHub-hosted `macos-26` GPU (macOS 26.7.2,
    /// Xcode 26.6) for `kernel_mul_mm_q4_0_tensor_f32`, from pipeline creation.
    const HOSTED_MACOS_26_DIAGNOSTIC: &str = "error: unsupported deferred-static-alloca-size \
        function body in _ZN5metal18cooperative_tensorIfNS_7extentsIiJLm18446744073709551615E\
        Lm18446744073709551615EEEEN3mpp10tensor_ops17__mutmul2d_detail16__operand_layoutIXtl\
        NS4_19matmul2d_descriptorELi32ELi64ELi32ELb0ELb1ELb0ELNS7_4modeE1EEELNS5_36__matmul2d_\
        cooperative_operand_indexE2ENS_20execution_simdgroupsILm4EEEDhDhfiJEEEEE.MTL_SIZEAS";

    fn compile_error(message: &str) -> MlxError {
        MlxError::ShaderCompilationError {
            name: "tensor".into(),
            message: message.into(),
        }
    }

    #[test]
    fn missing_tensor_header_is_unavailable_on_every_device() {
        let missing = compile_error("fatal error: 'metal_tensor' file not found");
        assert!(tensor_api_unavailable(true, &missing));
        assert!(tensor_api_unavailable(false, &missing));
    }

    #[test]
    fn a_device_with_metal4_fails_closed_on_every_other_diagnostic() {
        for message in [
            "fatal error: use of undeclared identifier 'broken'",
            "note: 'metal_tensor' file not found in cached diagnostics",
            HOSTED_MACOS_26_DIAGNOSTIC,
        ] {
            assert!(
                !tensor_api_unavailable(true, &compile_error(message)),
                "{message}"
            );
        }
    }

    #[test]
    fn a_device_without_metal4_has_no_tensor_api() {
        assert!(tensor_api_unavailable(
            false,
            &compile_error(HOSTED_MACOS_26_DIAGNOSTIC)
        ));
    }

    #[test]
    fn only_a_pipeline_build_failure_is_classified() {
        for device_supports_metal4 in [true, false] {
            assert!(!tensor_api_unavailable(
                device_supports_metal4,
                &MlxError::KernelNotFound("tensor".into())
            ));
            assert!(!tensor_api_unavailable(
                device_supports_metal4,
                &MlxError::InvalidArgument("tensor".into())
            ));
        }
    }
}
