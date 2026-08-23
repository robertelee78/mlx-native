pub const Q4_TENSOR_SHADER_STEM: &str = "quantized_matmul_mm_tensor";
const MISSING_METAL_TENSOR_DIAGNOSTIC: &str = "fatal error: 'metal_tensor' file not found";

/// The only compiler failure that represents an unavailable optional SDK
/// capability. Every other failure in the Q4 tensor shader is a source or
/// toolchain regression and must remain fatal.
pub fn is_exact_missing_metal_tensor_capability(shader_stem: &str, stderr: &str) -> bool {
    if shader_stem != Q4_TENSOR_SHADER_STEM {
        return false;
    }
    let mut error_lines = stderr.lines().filter(|line| line.contains("error:"));
    error_lines
        .next()
        .is_some_and(|line| line.contains(MISSING_METAL_TENSOR_DIAGNOSTIC))
        && error_lines.next().is_none()
}
