//! File-backed GGUF weight loading and Metal binding proof.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::sqrt_elementwise::dispatch_sqrt_f32;
use mlx_native::{DType, GgufFile, KernelRegistry, MlxBufferPool, MlxDevice};

fn fixture(values: &[f32]) -> Vec<u8> {
    let name = "weight.mapped";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&(values.len() as u64).to_le_bytes());
    file.extend_from_slice(&0u32.to_le_bytes());
    // Put the tensor 32 bytes into the data section so neither its absolute
    // file offset nor the Metal kernel binding is page-aligned.
    file.extend_from_slice(&32u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    file.extend_from_slice(&[0u8; 32]);
    for value in values {
        file.extend_from_slice(&value.to_le_bytes());
    }
    file
}

fn q2_k_fixture() -> Vec<u8> {
    let name = "weight.quantized";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&256u64.to_le_bytes());
    file.extend_from_slice(&10u32.to_le_bytes()); // GGML_TYPE_Q2_K
    file.extend_from_slice(&0u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    file.extend_from_slice(&[0u8; 84]); // one canonical Q2_K block
    file
}

#[test]
fn mapped_tensor_is_read_only_outlives_file_and_binds_with_offset() {
    let values = [1.0f32, 4.0, 9.0, 16.0, 25.0];
    let path = std::env::temp_dir().join(format!("mlx_mapped_gguf_{}.gguf", std::process::id()));
    std::fs::write(&path, fixture(&values)).expect("write mapped GGUF fixture");

    let device = MlxDevice::new().expect("Metal device");
    let gguf = GgufFile::open(&path).expect("open mapped GGUF fixture");
    let mut input = gguf
        .load_tensor_mapped("weight.mapped", &device)
        .expect("map GGUF tensor");

    assert!(input.is_file_backed());
    assert!(!input.is_cpu_writable());
    assert_eq!(input.dtype(), DType::F32);
    assert_eq!(input.shape(), &[values.len()]);
    assert_ne!(
        input.byte_offset(),
        0,
        "fixture must exercise binding offset"
    );
    assert_eq!(
        input.as_slice::<f32>().expect("read mapped tensor data"),
        values
    );
    assert_eq!(
        input
            .as_logical_slice::<f32>()
            .expect("read logical mapped tensor"),
        values
    );
    assert!(input.as_mut_slice::<f32>().is_err());

    // The buffer owns the mapping. Closing and unlinking the GGUF must not
    // invalidate either CPU reads or subsequent GPU reads.
    drop(gguf);
    std::fs::remove_file(&path).expect("unlink mapped GGUF fixture");
    assert_eq!(
        input
            .as_logical_slice::<f32>()
            .expect("mapped tensor survives unlink"),
        values
    );

    let output = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("allocate output");
    let mut params = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("allocate params");
    params.as_mut_slice::<u32>().expect("write params")[0] = values.len() as u32;

    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    dispatch_sqrt_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input,
        &output,
        &params,
    )
    .expect("dispatch mapped sqrt");
    encoder.commit_and_wait().expect("run mapped sqrt");

    assert_eq!(
        output.as_slice::<f32>().expect("read GPU output"),
        &[1.0, 2.0, 3.0, 4.0, 5.0]
    );

    // Tensor views already carry the segment's file offset. A nested slice
    // must add its relative offset instead of replacing the parent offset.
    let middle = input.slice_view(4, 3);
    let nested = middle.slice_view(4, 1);
    assert_eq!(middle.byte_offset(), input.byte_offset() + 4);
    assert_eq!(nested.byte_offset(), input.byte_offset() + 8);
    assert_eq!(nested.data_byte_len(), 4);
    assert_eq!(
        middle.as_slice::<f32>().expect("read middle tensor data"),
        &[4.0, 9.0, 16.0]
    );
    assert_eq!(
        nested.as_slice::<f32>().expect("read nested tensor data"),
        &[9.0]
    );
    assert_eq!(
        nested
            .as_logical_slice::<f32>()
            .expect("read nested mapped tensor"),
        &[9.0]
    );

    let nested_output = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("allocate nested output");
    params.as_mut_slice::<u32>().expect("write nested params")[0] = 1;
    let mut nested_encoder = device.command_encoder().expect("nested command encoder");
    dispatch_sqrt_f32(
        &mut nested_encoder,
        &mut registry,
        device.metal_device(),
        &nested,
        &nested_output,
        &params,
    )
    .expect("dispatch nested mapped sqrt");
    nested_encoder
        .commit_and_wait()
        .expect("run nested mapped sqrt");
    assert_eq!(
        nested_output
            .as_slice::<f32>()
            .expect("read nested GPU output"),
        &[3.0]
    );

    let mut pool = MlxBufferPool::new();
    pool.release(input);
    assert_eq!(
        pool.free_count(),
        0,
        "file-backed resources must not outlive their mmap owner in the pool"
    );
}

#[test]
fn truncated_tensor_payload_is_rejected_before_mmap_access() {
    let values = [1.0f32, 4.0, 9.0, 16.0, 25.0];
    let mut bytes = fixture(&values);
    bytes.truncate(bytes.len() - 2);
    let path = std::env::temp_dir().join(format!(
        "mlx_mapped_gguf_truncated_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write truncated GGUF fixture");

    let device = MlxDevice::new().expect("Metal device");
    let gguf = GgufFile::open(&path).expect("parse truncated GGUF header");
    let error = gguf
        .load_tensor_mapped("weight.mapped", &device)
        .expect_err("truncated mapped tensor must fail before mmap access");
    assert!(error.to_string().contains("exceeds file length"), "{error}");
    assert!(
        gguf.map_tensor_data(&device).is_err(),
        "segment mapping must reject the same truncated payload"
    );

    std::fs::remove_file(path).expect("remove truncated GGUF fixture");
}

#[test]
fn quantized_logical_view_cannot_cross_packed_tensor_bytes() {
    let path = std::env::temp_dir().join(format!(
        "mlx_mapped_gguf_quantized_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, q2_k_fixture()).expect("write Q2_K GGUF fixture");

    let device = MlxDevice::new().expect("Metal device");
    let gguf = GgufFile::open(&path).expect("parse Q2_K GGUF fixture");
    let mapped = gguf
        .load_tensor_mapped("weight.quantized", &device)
        .expect("map Q2_K tensor");
    assert_eq!(mapped.data_byte_len(), 84);
    assert_eq!(mapped.as_slice::<u8>().expect("read packed Q2_K").len(), 84);
    let error = mapped
        .as_logical_slice::<u8>()
        .expect_err("dequantized element shape must not widen a packed data view");
    assert!(error.to_string().contains("exceeds data length"), "{error}");

    std::fs::remove_file(path).expect("remove Q2_K GGUF fixture");
}
