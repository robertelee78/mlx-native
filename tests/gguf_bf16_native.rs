use half::bf16;
use mlx_native::gguf::{
    test_only_compute_byte_len, test_only_dequantize, test_only_ggml_type_from_u32,
    test_only_raw_tensor_dtype,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::DType;

#[test]
fn gguf_bf16_type_30_is_a_native_two_byte_scalar() {
    let ggml_type = test_only_ggml_type_from_u32(30).expect("GGML BF16 type 30");
    assert_eq!(ggml_type, GgmlType::BF16);
    assert_eq!(ggml_type.block_values(), 1);
    assert_eq!(ggml_type.block_bytes(), 2);
    assert_eq!(test_only_compute_byte_len(&[3, 5], ggml_type).unwrap(), 30);
    assert_eq!(test_only_raw_tensor_dtype(ggml_type), DType::BF16);
}

#[test]
fn gguf_bf16_diagnostic_conversion_preserves_values() {
    let expected = [0.0_f32, 1.0, -2.5, 3.25, f32::INFINITY];
    let bytes: Vec<u8> = expected
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_le_bytes())
        .collect();
    let mut actual = [f32::NAN; 5];

    test_only_dequantize(&bytes, GgmlType::BF16, &mut actual).unwrap();

    for (expected, actual) in expected.into_iter().zip(actual) {
        assert_eq!(bf16::from_f32(expected).to_f32(), actual);
    }
}
