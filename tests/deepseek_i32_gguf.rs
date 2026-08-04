//! DeepSeek-V4 hash-router GGUF I32 parsing and raw-load proof.

#![cfg(target_vendor = "apple")]

use mlx_native::{DType, GgmlType, GgufFile, MlxDevice};

fn fixture(values: &[i32]) -> Vec<u8> {
    let name = "blk.0.exp_probs_b.bias";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&(values.len() as u64).to_le_bytes());
    file.extend_from_slice(&26u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    for value in values {
        file.extend_from_slice(&value.to_le_bytes());
    }
    file
}

#[test]
fn type_26_parses_and_loads_exact_raw_i32() {
    let values = [0, 7, -3, i32::MAX, i32::MIN + 1];
    let path = std::env::temp_dir().join(format!("mlx_dsv4_i32_{}.gguf", std::process::id()));
    std::fs::write(&path, fixture(&values)).expect("write I32 fixture");

    let gguf = GgufFile::open(&path).expect("parse I32 GGUF");
    let info = gguf
        .tensor_info("blk.0.exp_probs_b.bias")
        .expect("I32 tensor info");
    assert_eq!(info.ggml_type, GgmlType::I32);
    assert_eq!(info.ggml_type.block_values(), 1);
    assert_eq!(info.ggml_type.block_bytes(), 4);
    assert_eq!(info.byte_len, values.len() * 4);

    let device = MlxDevice::new().expect("Metal device");
    let raw = gguf
        .load_tensor("blk.0.exp_probs_b.bias", &device)
        .expect("load raw I32");
    assert_eq!(raw.dtype(), DType::I32);
    assert_eq!(raw.shape(), &[values.len()]);
    assert_eq!(raw.as_slice::<i32>().expect("read I32"), values);

    let f32_view = gguf
        .load_tensor_f32("blk.0.exp_probs_b.bias", &device)
        .expect("load I32 as f32");
    let expected: Vec<f32> = values.iter().map(|value| *value as f32).collect();
    assert_eq!(f32_view.as_slice::<f32>().expect("read f32"), expected);
    std::fs::remove_file(path).expect("remove I32 fixture");
}
