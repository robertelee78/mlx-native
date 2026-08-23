#[allow(dead_code)]
#[path = "../benches/bench_q4_mm_tensor_64x32.rs"]
mod candidate_bench;

#[path = "../build_support/metal_tensor.rs"]
mod metal_tensor_build;

#[test]
fn shader_candidate_changes_only_compile_time_token_tile() {
    let shader = include_str!("../src/shaders/quantized_matmul_mm_tensor.metal");
    assert!(shader.contains("short n_mm_block_x"));
    assert!(shader.contains("hf2q_mul_mm_tensor_v2_impl<block_q4_0, 2, 4, dequantize_q4_0_t>"));
    assert!(shader.contains("hf2q_mul_mm_tensor_v2_impl<block_q4_0, 2, 1, dequantize_q4_0_t>"));
    assert_eq!(
        shader
            .matches("kernel_mul_mm_q4_0_tensor_64x32_f32")
            .count(),
        1
    );
}

#[test]
fn missing_tensor_capability_classifier_is_exact_and_mutation_sensitive() {
    let exact = "shader.metal:1:1: fatal error: 'metal_tensor' file not found";
    assert!(
        metal_tensor_build::is_exact_missing_metal_tensor_capability(
            metal_tensor_build::Q4_TENSOR_SHADER_STEM,
            exact,
        )
    );

    for (stem, stderr) in [
        (
            metal_tensor_build::Q4_TENSOR_SHADER_STEM,
            "shader.metal:1:1: error: use of undeclared identifier 'bad_symbol'",
        ),
        ("quantized_matmul", exact),
        (
            metal_tensor_build::Q4_TENSOR_SHADER_STEM,
            "shader.metal:1:1: fatal error: 'metal_tensor_extra' file not found",
        ),
        (
            metal_tensor_build::Q4_TENSOR_SHADER_STEM,
            "shader.metal:1:1: error: metal_tensor overload resolution failed",
        ),
        (
            metal_tensor_build::Q4_TENSOR_SHADER_STEM,
            "shader.metal:1:1: fatal error: 'metal_tensor' file not found\n\
             shader.metal:2:1: error: malformed tensor kernel",
        ),
    ] {
        assert!(
            !metal_tensor_build::is_exact_missing_metal_tensor_capability(stem, stderr),
            "unexpected downgrade for stem={stem} stderr={stderr}",
        );
    }
}

#[test]
fn unmatched_shape_filter_is_a_hard_error() {
    let error = candidate_bench::selected_shape_count(Some("definitely-not-a-real-shape"))
        .expect_err("an unmatched filter must not produce a successful zero-work benchmark");
    assert!(error.contains("matched no shapes"));
}
