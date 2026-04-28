//! Integration test for `GgufFile::load_tensor_into_pool`.
//!
//! Verifies the W-5b.7 residency-only registration path: a tensor loaded
//! through the pool API ends up registered with the device residency set,
//! the buffer is not bucket-rounded, and it remains valid after the pool
//! is dropped (since the pool only borrows residency membership, not
//! ownership of the underlying Metal buffer).

use std::io::Write;
use std::sync::Mutex;

use mlx_native::{
    gguf::GgufFile, macos_15_or_newer_for_test, reset_residency_env_cache_for_test,
    reset_residency_test_counters, residency_allocation_count_for_test, MlxBufferPool, MlxDevice,
    MlxError,
};

// Shared with tests/test_residency_set.rs; integration tests run in
// separate binaries so only intra-binary tests need synchronization.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Write a minimal GGUF v3 file containing one F32 tensor.
///
/// Layout (modeled on `tests/test_q5_k_dequant.rs::write_minimal_gguf`):
///   u32 magic ("GGUF"), u32 version=3, u64 n_tensors=1, u64 n_kv=0
///   tensor info: name, n_dims=1, dims[0]=n_elem, type=0 (F32), offset=0
///   pad to 32-byte alignment, raw F32 bytes.
fn write_minimal_f32_gguf(path: &std::path::Path, name: &str, data: &[f32]) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv

    // tensor name
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // n_dims = 1
    buf.extend_from_slice(&1u32.to_le_bytes());
    // dims[0] = data.len()
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    // type = 0 (F32)
    buf.extend_from_slice(&0u32.to_le_bytes());
    // offset = 0 (relative to tensor-data block)
    buf.extend_from_slice(&0u64.to_le_bytes());

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }

    let mut f = std::fs::File::create(path).expect("create tmp gguf");
    f.write_all(&buf).expect("write");
    f.flush().expect("flush");
}

fn active_residency_device() -> Option<MlxDevice> {
    reset_residency_env_cache_for_test();
    reset_residency_test_counters();
    std::env::remove_var("HF2Q_NO_RESIDENCY");

    if !macos_15_or_newer_for_test() {
        return None;
    }

    match MlxDevice::new() {
        Ok(device) => {
            assert!(
                device.residency_sets_enabled(),
                "macOS 15+ device boot should enable residency sets",
            );
            Some(device)
        }
        Err(MlxError::DeviceNotFound) => None,
        Err(err) => panic!("MlxDevice::new failed: {err}"),
    }
}

#[test]
fn load_tensor_into_pool_registers_with_residency_set() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    // 256 F32s = 1024 bytes — deliberately not a power of two boundary
    // when combined with non-power-of-two element counts elsewhere; here
    // it happens to be 2^10 but the registration path doesn't care.
    let payload: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    let tmp = std::env::temp_dir().join(format!(
        "mlx_load_into_pool_{}.gguf",
        std::process::id()
    ));
    write_minimal_f32_gguf(&tmp, "weight.test", &payload);

    let gguf = GgufFile::open(&tmp).expect("open mini gguf");
    let mut pool = MlxBufferPool::new();

    let baseline_alloc_count = residency_allocation_count_for_test();

    let buf = gguf
        .load_tensor_into_pool("weight.test", &device, &mut pool)
        .expect("load_tensor_into_pool");

    // Residency allocation count must have grown by exactly one.
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline_alloc_count + 1,
        "load_tensor_into_pool must register exactly one allocation",
    );

    // Pool must NOT have arena-tracked the externally-loaded buffer.
    assert_eq!(pool.in_use_count(), 0);
    assert_eq!(pool.free_count(), 0);

    // Verify the buffer contents round-tripped through the loader.
    let got: &[f32] = buf.as_slice().expect("as_slice");
    assert_eq!(got, payload.as_slice());

    // Verify the buffer was allocated at the EXACT byte length, with no
    // bucket-rounding. F32 tensor of 256 elements = exactly 1024 bytes.
    assert_eq!(
        buf.byte_len(),
        1024,
        "load_tensor_into_pool must not bucket-round the allocation",
    );

    // Drop the pool. The buffer must remain valid (caller owns it).
    drop(pool);
    let still_valid: &[f32] = buf.as_slice().expect("buffer outlives pool");
    assert_eq!(still_valid[0], 0.0);
    assert_eq!(still_valid[255], 127.5);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn load_tensor_into_pool_does_not_bucket_round_irregular_size() {
    // Specifically exercise a size that would balloon under power-of-two
    // bucketing — the whole point of register_existing.
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    // 257 F32 elements = 1028 bytes. bucket_size(1028) = 2048 (bucket-rounded
    // path would 2x). load_tensor_into_pool must allocate at exactly 1028.
    let payload: Vec<f32> = (0..257).map(|i| (i as f32).sin()).collect();
    let tmp = std::env::temp_dir().join(format!(
        "mlx_load_into_pool_irreg_{}.gguf",
        std::process::id()
    ));
    write_minimal_f32_gguf(&tmp, "weight.irreg", &payload);

    let gguf = GgufFile::open(&tmp).expect("open mini gguf");
    let mut pool = MlxBufferPool::new();

    let buf = gguf
        .load_tensor_into_pool("weight.irreg", &device, &mut pool)
        .expect("load_tensor_into_pool");

    assert_eq!(
        buf.byte_len(),
        1028,
        "expected exact-size allocation (no power-of-two rounding)",
    );

    std::fs::remove_file(&tmp).ok();
}
