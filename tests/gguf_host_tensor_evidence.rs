use std::io::Write;

use half::f16;
use mlx_native::GgufFile;

fn write_minimal_gguf(
    path: &std::path::Path,
    tensor_name: &str,
    shape_innermost_first: &[u64],
    wire_type_id: u32,
    payload: &[u8],
) {
    write_minimal_gguf_at(
        path,
        tensor_name,
        shape_innermost_first,
        wire_type_id,
        0,
        payload,
    );
}

fn write_minimal_gguf_at(
    path: &std::path::Path,
    tensor_name: &str,
    shape_innermost_first: &[u64],
    wire_type_id: u32,
    tensor_offset: u64,
    payload: &[u8],
) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(tensor_name.as_bytes());
    bytes.extend_from_slice(&(shape_innermost_first.len() as u32).to_le_bytes());
    for dimension in shape_innermost_first {
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    bytes.extend_from_slice(&wire_type_id.to_le_bytes());
    bytes.extend_from_slice(&tensor_offset.to_le_bytes());
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.extend_from_slice(payload);
    let mut file = std::fs::File::create(path).expect("create GGUF fixture");
    file.write_all(&bytes).expect("write GGUF fixture");
    file.sync_all().expect("sync GGUF fixture");
}

fn append_tensor_info(
    bytes: &mut Vec<u8>,
    tensor_name: &str,
    shape_innermost_first: &[u64],
    wire_type_id: u32,
) {
    bytes.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(tensor_name.as_bytes());
    bytes.extend_from_slice(&(shape_innermost_first.len() as u32).to_le_bytes());
    for dimension in shape_innermost_first {
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    bytes.extend_from_slice(&wire_type_id.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
}

fn fixture_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mlx_gguf_host_evidence_{}_{}.gguf",
        std::process::id(),
        label
    ))
}

#[test]
fn host_read_returns_exact_packed_q8_bytes_and_logical_values() {
    let path = fixture_path("exact");
    let mut payload = Vec::with_capacity(34);
    payload.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
    let quants: Vec<i8> = (-16_i8..=15).collect();
    payload.extend(quants.iter().map(|value| *value as u8));
    write_minimal_gguf(&path, "weight", &[32, 1], 8, &payload);

    let gguf = GgufFile::open(&path).expect("open GGUF fixture");
    let info = gguf.tensor_info("weight").expect("tensor info");
    assert_eq!(info.shape, vec![1, 32]);
    assert_eq!(gguf.read_tensor_bytes_host("weight").unwrap(), payload);
    let values = gguf.read_tensor_f32_host("weight").unwrap();
    let expected: Vec<f32> = quants.iter().map(|value| f32::from(*value) * 0.5).collect();
    assert_eq!(values, expected);
    std::fs::remove_file(path).expect("remove GGUF fixture");
}

#[test]
fn from_file_preserves_the_already_open_artifact_identity() {
    let path = fixture_path("open_inode");
    let replacement = fixture_path("replacement_inode");
    let mut payload = Vec::with_capacity(34);
    payload.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
    payload.extend(0_u8..32);
    write_minimal_gguf(&path, "weight", &[32], 8, &payload);
    let exact_file = std::fs::File::open(&path).expect("open exact GGUF inode");

    write_minimal_gguf(&replacement, "other", &[32], 8, &[0_u8; 34]);
    std::fs::rename(&replacement, &path).expect("replace pathname after opening exact inode");

    let exact = GgufFile::from_file(exact_file).expect("parse already-open exact inode");
    assert_eq!(exact.read_tensor_bytes_host("weight").unwrap(), payload);
    assert!(exact.tensor_info("other").is_none());
    let reopened = GgufFile::open(&path).expect("parse replacement pathname");
    assert!(reopened.tensor_info("weight").is_none());
    assert!(reopened.tensor_info("other").is_some());
    std::fs::remove_file(path).expect("remove replacement fixture");
}

#[test]
fn host_read_rejects_missing_and_truncated_tensors() {
    let path = fixture_path("missing");
    let payload = vec![0_u8; 34];
    write_minimal_gguf(&path, "weight", &[32], 8, &payload);
    let gguf = GgufFile::open(&path).expect("open GGUF fixture");
    assert!(gguf.read_tensor_bytes_host("missing").is_err());
    assert!(gguf.read_tensor_f32_host("missing").is_err());

    std::fs::remove_file(path).expect("remove GGUF fixture");

    let truncated = fixture_path("truncated");
    write_minimal_gguf(&truncated, "weight", &[32], 8, &payload[..33]);
    let gguf = GgufFile::open(&truncated).expect("parse truncated payload directory");
    assert!(gguf.read_tensor_bytes_host("weight").is_err());
    assert!(gguf.read_tensor_f32_host("weight").is_err());
    std::fs::remove_file(truncated).expect("remove truncated fixture");
}

#[test]
fn host_read_rejects_overflowing_and_out_of_bounds_regions() {
    for (label, offset) in [("offset_overflow", u64::MAX), ("past_eof", 32)] {
        let path = fixture_path(label);
        let payload = vec![0_u8; 34];
        write_minimal_gguf_at(&path, "weight", &[32], 8, offset, &payload);
        let gguf = GgufFile::open(&path).expect("parse tensor directory");
        assert!(gguf.read_tensor_bytes_host("weight").is_err());
        assert!(gguf.read_tensor_f32_host("weight").is_err());
        std::fs::remove_file(path).expect("remove invalid-region fixture");
    }
}

#[test]
fn parser_rejects_row_invalid_and_overflowing_geometry() {
    let row_invalid = fixture_path("row_invalid");
    write_minimal_gguf(&row_invalid, "weight", &[16, 2], 8, &[]);
    assert!(GgufFile::open(&row_invalid).is_err());
    std::fs::remove_file(row_invalid).expect("remove row-invalid fixture");

    let overflow = fixture_path("geometry_overflow");
    write_minimal_gguf(&overflow, "weight", &[1, u64::MAX, 2], 0, &[]);
    assert!(GgufFile::open(&overflow).is_err());
    std::fs::remove_file(overflow).expect("remove overflow fixture");
}

#[test]
fn parser_rejects_duplicate_tensor_names() {
    let path = fixture_path("duplicate_names");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    append_tensor_info(&mut bytes, "weight", &[32], 8);
    append_tensor_info(&mut bytes, "weight", &[32], 8);
    std::fs::write(&path, bytes).expect("write duplicate-name GGUF fixture");
    let error = match GgufFile::open(&path) {
        Ok(_) => panic!("duplicate tensor names must reject"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("duplicate tensor name"));
    std::fs::remove_file(path).expect("remove duplicate-name fixture");
}
