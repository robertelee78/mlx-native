use std::sync::Mutex;

use mlx_native::{
    macos_15_or_newer_for_test, reset_residency_env_cache_for_test,
    reset_residency_test_counters, residency_allocation_count_for_test,
    residency_commit_call_count_for_test, DType, MlxBufferPool, MlxDevice, MlxError,
};

// ADR-015 iter8e (Phase 3b — merged tiebreaker): the test cohort below
// asserts the THREE structural ACs from claude's variant (alloc-on-mac15
// auto-registers + drop deregisters; clones share registration; HF2Q_NO_
// RESIDENCY=1 skips registration) PLUS the commit-count regression test
// that fixes both prior variants' per-allocation commit storm.
//
// Design reference: llama.cpp ggml-metal-device.m:1378-1382 (batch
// addAllocation in for-loop, commit ONCE) and lines 1397-1399 (free path
// uses removeAllAllocations + endResidency once for the entire buffer
// set, NOT per-allocation).

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
fn device_alloc_buffers_removed_on_drop() {
    // Phase 3b AC test #1 (from claude): device.alloc_buffer auto-registers
    // 10 buffers; dropping all 10 fires removeAllocation: via the
    // Arc<MlxBufferStorage> RAII path so the residency-set count returns
    // to 0. No clones held — last-Arc-drop semantics.
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    assert_eq!(residency_allocation_count_for_test(), 0);

    let buffers = (0..10)
        .map(|i| {
            device
                .alloc_buffer(1024, DType::F32, vec![256])
                .unwrap_or_else(|e| panic!("alloc_buffer iter {i}: {e}"))
        })
        .collect::<Vec<_>>();

    assert_eq!(
        residency_allocation_count_for_test(),
        10,
        "after 10 alloc_buffer calls, residency count should be 10",
    );

    drop(buffers);

    assert_eq!(
        residency_allocation_count_for_test(),
        0,
        "after dropping all 10 buffers, residency count should be 0",
    );
}

#[test]
fn mlx_buffer_clone_shares_registration() {
    // Phase 3b AC test #2 (from claude): cloning an MlxBuffer must NOT
    // double-register, and dropping a clone while the original is still
    // alive must NOT deregister. The Arc<MlxBufferStorage> is shared via
    // Arc::clone — last-clone-dropped is the only point that triggers
    // removeAllocation:.
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    assert_eq!(residency_allocation_count_for_test(), 0);

    let buf = device
        .alloc_buffer(2048, DType::U8, vec![2048])
        .expect("alloc_buffer");
    assert_eq!(residency_allocation_count_for_test(), 1);

    // Cloning shares the Arc<MlxBufferStorage>, no new addAllocation:.
    let buf_clone1 = buf.clone();
    let buf_clone2 = buf.clone();
    let buf_clone3 = buf.clone();
    let buf_clone4 = buf.clone();
    let buf_clone5 = buf.clone();
    assert_eq!(
        residency_allocation_count_for_test(),
        1,
        "5x MlxBuffer clones must NOT double-register",
    );

    // slice_view also shares the registration.
    let view = buf.slice_view(0, 256);
    assert_eq!(
        residency_allocation_count_for_test(),
        1,
        "slice_view must NOT double-register",
    );

    // Drop clones one by one; original is still alive so count stays 1.
    drop(buf_clone1);
    drop(buf_clone2);
    drop(buf_clone3);
    drop(buf_clone4);
    drop(buf_clone5);
    drop(view);
    assert_eq!(
        residency_allocation_count_for_test(),
        1,
        "dropping clones while original is alive must NOT deregister",
    );

    // Last clone drop triggers deregister via Arc<MlxBufferStorage>::Drop.
    drop(buf);
    assert_eq!(
        residency_allocation_count_for_test(),
        0,
        "dropping the last MlxBuffer clone must deregister",
    );
}

#[test]
fn no_residency_env_skips_registration() {
    // Phase 3b AC test #3 (from claude): with HF2Q_NO_RESIDENCY=1 the
    // device boots without a residency set, so alloc_buffer must produce
    // buffers with NO registration guard — no addAllocation: call, count
    // stays at 0, no panic, no error.
    let _guard = TEST_LOCK.lock().expect("test lock");
    reset_residency_env_cache_for_test();
    reset_residency_test_counters();
    std::env::set_var("HF2Q_NO_RESIDENCY", "1");

    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(MlxError::DeviceNotFound) => {
            std::env::remove_var("HF2Q_NO_RESIDENCY");
            reset_residency_env_cache_for_test();
            return;
        }
        Err(e) => {
            std::env::remove_var("HF2Q_NO_RESIDENCY");
            reset_residency_env_cache_for_test();
            panic!("MlxDevice::new failed: {e}");
        }
    };

    assert!(!device.residency_sets_enabled());

    let _bufs: Vec<_> = (0..5)
        .map(|_| {
            device
                .alloc_buffer(512, DType::F32, vec![128])
                .expect("alloc_buffer")
        })
        .collect();

    assert_eq!(
        residency_allocation_count_for_test(),
        0,
        "HF2Q_NO_RESIDENCY=1 must skip auto-registration entirely",
    );

    std::env::remove_var("HF2Q_NO_RESIDENCY");
    reset_residency_env_cache_for_test();
}

#[test]
fn defer_and_flush_commit_count() {
    // Phase 3b AC test #4 (NEW — commit-count regression):
    //
    // Allocate 100 buffers via device.alloc_buffer, then drop them all,
    // then issue ONE encoder.commit() to flush. This is the canonical
    // production pattern: the decode hot path makes ~440 alloc_buffer
    // calls per token, drops them at scope-exit, and submits one CB per
    // token via encoder.commit().
    //
    // The pre-3b claude+codex variants would issue ~880 [set commit]
    // calls in this scenario (1 per add + 1 per remove). The 3b
    // defer-and-flush design collapses this to AT MOST 1 commit
    // (encoder.commit's flush_pending sees pending=true, issues one
    // [set commit], clears pending). Any further pending changes after
    // that single flush would queue for the next CB.
    //
    // Queen gate (per Phase 3 judgment): ≤ 5 commits per 100-alloc batch.
    // Our actual target: == 1 commit. Anything > 5 fails the AC6 gate.
    let _guard = TEST_LOCK.lock().expect("test lock");
    let Some(device) = active_residency_device() else {
        return;
    };

    let commits_baseline = residency_commit_call_count_for_test();
    assert_eq!(commits_baseline, 0, "active_residency_device resets counters");

    // Allocate 100 buffers — each calls add_allocation (pending=true)
    // but NO [set commit]. Counter must NOT increment per-alloc.
    let buffers = (0..100)
        .map(|i| {
            device
                .alloc_buffer(512, DType::F32, vec![128])
                .unwrap_or_else(|e| panic!("alloc_buffer iter {i}: {e}"))
        })
        .collect::<Vec<_>>();

    let commits_after_allocs = residency_commit_call_count_for_test();
    assert_eq!(
        commits_after_allocs, 0,
        "100 alloc_buffer calls must NOT issue any [set commit] (deferred)",
    );
    assert_eq!(residency_allocation_count_for_test(), 100);

    // Drop all 100 — each storage Drop calls remove_allocation
    // (pending=true), no [set commit].
    drop(buffers);

    let commits_after_drops = residency_commit_call_count_for_test();
    assert_eq!(
        commits_after_drops, 0,
        "100 buffer drops must NOT issue any [set commit] (deferred)",
    );
    assert_eq!(residency_allocation_count_for_test(), 0);

    // Now issue ONE encoder.commit() — flushes pending in a single
    // [set commit] call.
    let mut encoder = device.command_encoder().expect("command_encoder");
    encoder.commit_and_wait().expect("commit_and_wait");

    let commits_after_flush = residency_commit_call_count_for_test();

    // The whole point of the AC: ONE flush after 100 allocs + 100 drops.
    // Queen gate "≤ 5 commits/token"; this is the tightest possible.
    assert!(
        commits_after_flush <= 5,
        "Phase 3b AC4 (queen gate ≤ 5 commits/token): got {} commits for 100-alloc + 100-drop batch (variant baseline = ~200)",
        commits_after_flush,
    );
    // Tighter assertion: defer-and-flush should land at exactly 1 commit
    // (one CB submission flushes the union of pending adds+removes).
    assert_eq!(
        commits_after_flush, 1,
        "defer-and-flush should issue exactly 1 [set commit] for the 100-alloc + 100-drop batch + 1 CB submission",
    );
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
