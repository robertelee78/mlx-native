use std::sync::Mutex;

use mlx_native::{
    macos_15_or_newer_for_test, reset_residency_env_cache_for_test,
    reset_residency_test_counters, residency_allocation_count_for_test,
    residency_commit_call_count_for_test, DType, MlxBufferPool, MlxDevice, MlxError,
};

static TEST_LOCK: Mutex<()> = Mutex::new(());

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
                "macOS 15+ device boot should enable residency sets and log `residency sets = true`",
            );
            Some(device)
        }
        Err(MlxError::DeviceNotFound) => None,
        Err(err) => panic!("MlxDevice::new failed: {err}"),
    }
}

#[test]
fn residency_set_initialized_on_macos_15_plus() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    assert!(device.residency_sets_enabled());
}

#[test]
fn residency_set_buffers_added_on_allocation() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };
    let mut pool = MlxBufferPool::new();

    let _buffers = pool
        .alloc_batch(
            &device,
            (0..4).map(|_| (1024, DType::F32, vec![256])),
        )
        .expect("alloc batch");

    assert_eq!(residency_allocation_count_for_test(), 4);
}

#[test]
fn residency_set_buffers_removed_on_pool_eviction() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };
    let mut pool = MlxBufferPool::new();

    {
        let _buffers = pool
            .alloc_batch(
                &device,
                (0..4).map(|_| (1024, DType::F32, vec![256])),
            )
            .expect("alloc batch");
        assert_eq!(residency_allocation_count_for_test(), 4);
    }

    pool.reset();
    pool.clear();

    assert_eq!(residency_allocation_count_for_test(), 0);
}

#[test]
fn hf2q_no_residency_disables_init() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    reset_residency_env_cache_for_test();
    reset_residency_test_counters();
    std::env::set_var("HF2Q_NO_RESIDENCY", "1");

    let result = MlxDevice::new();

    std::env::remove_var("HF2Q_NO_RESIDENCY");
    reset_residency_env_cache_for_test();

    match result {
        Ok(device) => assert!(!device.residency_sets_enabled()),
        Err(MlxError::DeviceNotFound) => {}
        Err(err) => panic!("MlxDevice::new failed: {err}"),
    }
}

#[test]
fn commit_called_after_alloc_batch() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };
    let mut pool = MlxBufferPool::new();

    let empty = pool
        .alloc_batch(&device, std::iter::empty())
        .expect("empty alloc batch");
    assert!(empty.is_empty());
    assert_eq!(residency_commit_call_count_for_test(), 0);

    let _buffers = pool
        .alloc_batch(
            &device,
            (0..4).map(|_| (1024, DType::F32, vec![256])),
        )
        .expect("alloc batch");

    assert_eq!(residency_allocation_count_for_test(), 4);
    assert_eq!(residency_commit_call_count_for_test(), 1);
}
