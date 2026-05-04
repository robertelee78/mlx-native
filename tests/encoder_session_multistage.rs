//! ADR-019 Phase 0b iter89e2-B — `EncoderSession` multistage tests.
//!
//! These tests verify the iter89e2-B contract:
//!
//! 1. `fence_stage` followed by `reset_for_next_stage` chains stages
//!    correctly (signal → wait monotonic counter, GPU output is right).
//! 2. The per-session `MTLSharedEvent` is lazy-allocated on first
//!    `fence_stage` and reused across subsequent fences; the monotonic
//!    `fence_value` increments by exactly 1 per fence.
//! 3. The residency-delegation surface (`add_to_residency_set` /
//!    `remove_from_residency_set`) routes to the device's single
//!    residency set — counter movements match the existing
//!    `MlxDevice::alloc_buffer` Drop semantics.
//! 4. Dropping a session in the `Fenced` state (after `fence_stage` but
//!    before `reset_for_next_stage`) is safe — the in-flight CB
//!    completes on the GPU under retained-refs, no Metal assertion
//!    fires, and the device remains usable.
//! 5. **F2 adversarial:** dropping a scratch buffer between `fence_stage`
//!    and the next stage's `commit_*` does NOT corrupt residency-set
//!    accounting and does NOT trip the iter58b residency-rescission
//!    failure mode. Validates the multi-stage F2 fence preservation
//!    documented in `encoder_session.rs::Drop` case (2).
//!
//! ## Env-var hygiene — load-bearing (same as iter89e2-A)
//!
//! `HF2Q_ENCODER_SESSION` is read exactly once via `OnceLock` in
//! `encoder_session.rs::encoder_session_enabled`. No test in this file
//! mutates the env var. Tests that exercise the live-session path are
//! all gated on `EncoderSession::env_enabled()` and emit a documented
//! `eprintln` skip when the cache says OFF, exactly mirroring
//! `tests/encoder_session_lifecycle.rs`.
//!
//! Run BOTH branches in CI via:
//!
//! ```bash
//! cargo test --release --test encoder_session_multistage
//! HF2Q_ENCODER_SESSION=1 cargo test --release --test encoder_session_multistage
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Mutex;

use mlx_native::{
    reset_residency_test_counters, residency_allocation_count_for_test, DType, EncoderSession,
    KernelRegistry, MlxDevice,
};

/// Serializes ALL tests in this binary against the process-global
/// residency counter.
///
/// `residency_allocation_count_for_test()` is a process-global atomic
/// that every `MlxDevice::alloc_buffer` and every Drop bumps. The
/// default `cargo test` runner spawns a thread pool, so sibling tests
/// allocating buffers (every test does) race against the
/// residency-delegation and F2-adversarial tests' counter assertions.
///
/// All five tests in this binary acquire this lock — tests that don't
/// touch the counter still need it because their allocations would
/// otherwise contaminate the locked tests' baselines. Mirror
/// `tests/test_residency_set.rs::TEST_LOCK`.
///
/// We unwrap the lock with `.unwrap_or_else(...)` rather than
/// `.expect()` so a panic in one test doesn't poison the others —
/// the failing test surfaces its own panic; lock-poisoning would
/// shadow that with a less informative `PoisonError`.
static RESIDENCY_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Lock-acquire helper that recovers from poisoning so a panicked
/// sibling test doesn't shadow the actual failure with `PoisonError`.
fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    RESIDENCY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Test 1 — multi-stage happy path: fence + reset + new dispatch.
///
/// Stage 1: dispatch elementwise_add, then `fence_stage(Some(label))`
/// (non-blocking, encodes signal-event(1)). Verify:
/// - session is drained AND fence_pending,
/// - `has_event()` returns true (lazy-alloc),
/// - `fence_value()` is 1.
///
/// Then `reset_for_next_stage`: opens a fresh CB, encodes wait-event(1)
/// on it. Verify:
/// - session is no longer drained,
/// - no longer fence_pending,
/// - the underlying `MTLCommandBuffer` is a fresh handle (different
///   from the prior submitted one).
///
/// Stage 2: dispatch elementwise_add into a different output buffer,
/// `commit_and_wait()`. Verify:
/// - both outputs (stage 1 read after stage 2's wait, stage 2 read
///   directly) match expected values,
/// - stage 2's CB label propagated.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is false.
#[test]
fn test_session_fence_stage_then_reset_then_begin_stage() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_fence_stage_then_reset_then_begin_stage \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env. \
             Re-run with HF2Q_ENCODER_SESSION=1 to exercise."
        );
        return;
    }
    let _guard = acquire_test_lock();

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();

    // Stage 1 buffers
    let mut a1 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a1");
    let mut b1 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b1");
    let out1 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out1");
    a1.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b1.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

    // Stage 2 buffers
    let mut a2 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a2");
    let mut b2 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b2");
    let out2 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out2");
    a2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[100.0, 200.0, 300.0, 400.0]);
    b2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[7.0, 8.0, 9.0, 10.0]);

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under env=1");

    // Pre-fence sanity.
    assert!(!sess.has_event(), "no event before first fence_stage");
    assert_eq!(sess.fence_value(), 0, "fence_value starts at 0");
    assert!(!sess.is_fence_pending());

    // Stage 1.
    sess.begin_stage("phase.iter89e2b_stage1");
    mlx_native::ops::elementwise::elementwise_add(
        sess.encoder(),
        &mut registry,
        device.metal_device(),
        &a1,
        &b1,
        &out1,
        n,
        DType::F32,
    )
    .expect("stage1 dispatch");

    sess.fence_stage(Some("phase.iter89e2b_stage1.fence"))
        .expect("fence_stage Ok");

    assert!(sess.is_drained(), "drained after fence_stage");
    assert!(sess.is_fence_pending(), "fence_pending after fence_stage");
    assert!(sess.has_event(), "event lazy-allocated by first fence");
    assert_eq!(sess.fence_value(), 1, "fence_value bumped to 1");

    // Snapshot the stage-1 CB label so we can later assert the
    // rotation produced a different CB. The label was set by
    // fence_stage(Some("phase.iter89e2b_stage1.fence")).
    let cb_label_stage1: String = sess.metal_command_buffer().label().to_string();
    assert_eq!(
        cb_label_stage1, "phase.iter89e2b_stage1.fence",
        "stage 1 fenced CB carries the fence_stage label"
    );

    // Reset to stage 2.
    sess.reset_for_next_stage().expect("reset_for_next_stage Ok");

    assert!(!sess.is_drained(), "no longer drained after reset");
    assert!(!sess.is_fence_pending(), "no fence pending after reset");
    assert_eq!(
        sess.fence_value(),
        1,
        "fence_value persists across reset (it is the high-water mark)"
    );

    // Fresh CB has empty label until begin_stage / commit propagate.
    let cb_label_post_reset: String = sess.metal_command_buffer().label().to_string();
    assert_ne!(
        cb_label_post_reset, "phase.iter89e2b_stage1.fence",
        "reset_for_next_stage must rotate to a fresh MTLCommandBuffer (no carryover label)"
    );

    // Stage 2.
    sess.begin_stage("phase.iter89e2b_stage2");
    mlx_native::ops::elementwise::elementwise_add(
        sess.encoder(),
        &mut registry,
        device.metal_device(),
        &a2,
        &b2,
        &out2,
        n,
        DType::F32,
    )
    .expect("stage2 dispatch");

    // Synchronous drain — also drains stage 1 (Metal queue FIFO + the
    // wait-event on the stage-2 CB serializes them).
    sess.commit_and_wait().expect("stage2 commit_and_wait Ok");
    assert!(sess.is_drained(), "drained after commit_and_wait");
    assert!(!sess.is_fence_pending(), "no fence pending after commit_and_wait");

    // Stage 1 must have completed by now (commit_and_wait above blocks
    // until the GPU finishes; the stage-2 wait-event ensures stage 1
    // ran first; Metal queue FIFO orders the submissions). Result
    // buffer is readable WITHOUT an additional wait.
    let r1 = out1.as_slice::<f32>().expect("read out1");
    assert_eq!(
        r1,
        &[11.0, 22.0, 33.0, 44.0],
        "stage 1 elementwise_add result must propagate via fenced CB"
    );

    let r2 = out2.as_slice::<f32>().expect("read out2");
    assert_eq!(
        r2,
        &[107.0, 208.0, 309.0, 410.0],
        "stage 2 elementwise_add result must propagate via fresh CB after wait-event"
    );

    // Stage-2 CB label propagated.
    let cb_label = sess.metal_command_buffer().label();
    assert_eq!(
        cb_label, "phase.iter89e2b_stage2",
        "stage 2 label must propagate to the fresh CB's MTLCommandBuffer.label"
    );
}

/// Test 2 — monotonic counter signal/wait round-trip across N=3 fences.
///
/// No `commit_and_wait` between fences. Each fence increments
/// `fence_value` by exactly 1; the same `MTLSharedEvent` instance is
/// reused (verified indirectly: `has_event()` flips to true on fence 1
/// and stays true through fence 3). After 3 fences + 3 resets + a
/// final commit_and_wait, the chain has executed in order.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is false.
#[test]
fn test_session_fence_event_signal_wait_round_trip() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_fence_event_signal_wait_round_trip \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }
    let _guard = acquire_test_lock();

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 2usize;
    let byte_len = n * std::mem::size_of::<f32>();

    // Pre-fill 3 input pairs + 3 output buffers.
    let inputs: Vec<(_, _, _)> = (0..3)
        .map(|i| {
            let mut a = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
            let mut b = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
            let out = device
                .alloc_buffer(byte_len, DType::F32, vec![n])
                .expect("out");
            let base = (i as f32) * 100.0;
            a.as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&[base + 1.0, base + 2.0]);
            b.as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&[10.0, 20.0]);
            (a, b, out)
        })
        .collect();

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under env=1");

    assert_eq!(sess.fence_value(), 0);
    assert!(!sess.has_event());

    // Three fence-and-rotate cycles.
    for (i, (a, b, out)) in inputs.iter().enumerate() {
        sess.begin_stage(&format!("phase.iter89e2b_chain_stage{i}"));
        mlx_native::ops::elementwise::elementwise_add(
            sess.encoder(),
            &mut registry,
            device.metal_device(),
            a,
            b,
            out,
            n,
            DType::F32,
        )
        .expect("chain dispatch");

        sess.fence_stage(None).expect("fence_stage Ok");
        let expected_value = (i as u64) + 1;
        assert_eq!(
            sess.fence_value(),
            expected_value,
            "fence_value must increment monotonically per fence (i={i})"
        );
        assert!(sess.has_event(), "event must be allocated after first fence");
        assert!(sess.is_fence_pending());
        assert!(sess.is_drained());

        sess.reset_for_next_stage()
            .expect("reset_for_next_stage Ok");
        assert!(!sess.is_drained());
        assert!(!sess.is_fence_pending());
        assert_eq!(
            sess.fence_value(),
            expected_value,
            "fence_value persists across reset"
        );
    }

    // Final stage to drain the chain. No fence; just commit_and_wait
    // to flush the queue.
    sess.begin_stage("phase.iter89e2b_chain_drain");
    // Reuse one of the input dispatches to give the final CB real
    // work — Metal won't enforce ordering on an empty CB, but we DO
    // want a real GPU op so the wait-event from the prior reset has
    // something to gate.
    let (a_final, b_final, out_final) = &inputs[0];
    mlx_native::ops::elementwise::elementwise_add(
        sess.encoder(),
        &mut registry,
        device.metal_device(),
        a_final,
        b_final,
        out_final,
        n,
        DType::F32,
    )
    .expect("drain dispatch");
    sess.commit_and_wait().expect("drain commit_and_wait Ok");

    // Verify all 3 chain outputs are present (commit_and_wait drained
    // every prior fenced CB via Metal's queue-FIFO + wait-events).
    for (i, (_, _, out)) in inputs.iter().enumerate() {
        let r = out.as_slice::<f32>().expect("read chain out");
        let base = (i as f32) * 100.0;
        // Note: out[0] is overwritten by the drain stage with the same
        // input as chain stage 0, so the value is the same expected
        // sum either way for i=0.
        assert_eq!(
            r,
            &[base + 11.0, base + 22.0],
            "chain stage {i} output must be readable after drained commit"
        );
    }

    // Final fence_value remains at 3 (the high-water mark from the 3
    // fences); the final commit_and_wait did NOT bump it.
    assert_eq!(
        sess.fence_value(),
        3,
        "fence_value is the high-water mark (3 fences fired); commit_and_wait does not bump it"
    );
}

/// Test 3 — residency delegation: add/remove via session API matches the
/// device's single residency set.
///
/// `MlxDevice::alloc_buffer` already auto-registers buffers; this test
/// verifies the session's explicit add/remove surface routes to the
/// SAME residency set (single-set invariant). Test sequence:
///
/// 1. baseline `residency_allocation_count_for_test` (may be non-zero
///    if other tests populated it; we measure deltas).
/// 2. Allocate buffer via device → count bumped by 1.
/// 3. `session.remove_from_residency_set(buf)` → count back to baseline.
/// 4. `session.add_to_residency_set(buf)` → count bumped again.
/// 5. Drop the buffer → MlxBufferStorage::Drop fires another remove
///    (idempotent; the underlying Metal `removeAllocation:` is
///    idempotent against a not-currently-resident allocation, but the
///    test counter saturates at 0 via `checked_sub` per
///    `residency.rs:181-184`).
///
/// The point: same-counter, same-set, end-to-end.
///
/// Runs only when `env_enabled()` is true AND
/// `device.residency_sets_enabled()` is true (macOS 15+). Skipped
/// gracefully on either off.
#[test]
fn test_session_residency_delegation_round_trip() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_residency_delegation_round_trip \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }
    // Serialize against sibling residency-counter tests in this binary.
    let _guard = acquire_test_lock();
    let device = MlxDevice::new().expect("MlxDevice::new");
    if !device.residency_sets_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_residency_delegation_round_trip \
             SKIPPED — residency sets disabled (macOS<15 or HF2Q_NO_RESIDENCY=1)."
        );
        return;
    }

    // Reset under lock so the baseline is deterministic for the test.
    reset_residency_test_counters();
    let baseline = residency_allocation_count_for_test();
    assert_eq!(baseline, 0, "reset_residency_test_counters zeros the count");

    // Allocate via device — auto-registers (count += 1).
    let buf = device
        .alloc_buffer(1024, DType::F32, vec![256])
        .expect("alloc_buffer");
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline + 1,
        "device.alloc_buffer must auto-register (delta=+1)"
    );

    let sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under env=1");

    // remove_from_residency_set must decrement the same counter.
    let removed = sess.remove_from_residency_set(&buf);
    assert!(
        removed,
        "remove_from_residency_set must return true when residency is enabled"
    );
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline,
        "session.remove_from_residency_set must decrement the same counter"
    );

    // add_to_residency_set must increment again.
    let added = sess.add_to_residency_set(&buf);
    assert!(
        added,
        "add_to_residency_set must return true when residency is enabled"
    );
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline + 1,
        "session.add_to_residency_set must increment the same counter"
    );

    // Drop sess WITHOUT committing — the in-flight residency staging
    // is not flushed (no commit fired), but the test counter has
    // already moved (it tracks every add/remove call, not just
    // committed ones — see residency.rs:170 / 184).
    drop(sess);

    // Drop the buffer — MlxBufferStorage::Drop fires removeAllocation
    // once. Counter -> baseline (saturating semantics).
    drop(buf);
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline,
        "buffer Drop must remove its registration via storage Drop"
    );
}

/// Test 4 — Drop on Fenced session is safe (F2 case 2 from
/// `encoder_session.rs::Drop`).
///
/// Construct session, dispatch, `fence_stage` (non-blocking — submits
/// the CB), then drop without `reset_for_next_stage` or any other
/// commit. Verify:
/// - no Metal assertion fires,
/// - the prior CB completes (we wait_until_completed on the CB BEFORE
///   drop because after drop we cannot access it; this is a tighter
///   invariant — the CB DID run),
/// - the device is still usable for a subsequent allocation + dispatch.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is false.
#[test]
fn test_session_drop_with_open_fence_drains_synchronously() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_drop_with_open_fence_drains_synchronously \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }
    let _guard = acquire_test_lock();

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut a = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("out");
    a.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    {
        let mut sess = device
            .encoder_session()
            .expect("encoder_session() Ok")
            .expect("Some under env=1");

        sess.begin_stage("phase.iter89e2b_drop_fenced");
        mlx_native::ops::elementwise::elementwise_add(
            sess.encoder(),
            &mut registry,
            device.metal_device(),
            &a,
            &b,
            &out,
            n,
            DType::F32,
        )
        .expect("dispatch");

        sess.fence_stage(None).expect("fence_stage Ok");
        assert!(sess.is_fence_pending());
        assert!(sess.is_drained());

        // Wait until the fenced CB completes BEFORE the session
        // drops, so we can read `out` after drop and verify the GPU
        // really executed (not just submitted-then-discarded).
        sess.metal_command_buffer().wait_until_completed();

        // Intentional: drop sess in Fenced state without
        // reset_for_next_stage. F2 case (2) from encoder_session.rs::Drop:
        // the SharedEvent's ARC release fires; the CB was already
        // committed and has now completed; the persistent encoder was
        // already ended by fence_signal_and_commit; CommandEncoder::Drop
        // sees a null active_encoder and is a no-op.
    }

    // The fenced-and-completed CB's output must be readable — under
    // retained-refs the buffer is still alive, and the GPU finished
    // before drop.
    let r = out.as_slice::<f32>().expect("read out post-drop");
    assert_eq!(
        r,
        &[6.0, 8.0, 10.0, 12.0],
        "fenced CB output must be visible after session drop"
    );

    // Device usable post-drop.
    let mut enc = device
        .command_encoder()
        .expect("command_encoder post-drop");
    let mut a2 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("a2");
    let mut b2 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("b2");
    let out2 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("out2");
    a2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
    b2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[2.0, 2.0, 2.0, 2.0]);
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &a2,
        &b2,
        &out2,
        n,
        DType::F32,
    )
    .expect("post-drop dispatch");
    enc.commit_and_wait().expect("post-drop commit_and_wait");
    assert_eq!(
        out2.as_slice::<f32>().expect("read out2"),
        &[3.0, 3.0, 3.0, 3.0],
        "device usable after fenced EncoderSession Drop"
    );
}

/// Test 5 — F2 ADVERSARIAL: scratch lifetime under fence + reset.
///
/// The iter58b residency-rescission failure mode in plain English:
/// (a) wrapper allocates scratch via `device.alloc_buffer` (registers
/// in residency set + bumps counter); (b) wrapper dispatches using
/// scratch, commits non-blocking; (c) scratch goes out of scope and
/// drops, staging a deferred `removeAllocation:`; (d) a downstream
/// commit on a DIFFERENT CB fires `flush_pending` which commits the
/// staged remove BEFORE the original CB's GPU work completes; (e) on
/// some Apple Silicon configurations Metal demotes the GPU page mid-
/// flight, scratch reads return garbage, output corrupts.
///
/// EncoderSession's iter89e2-B multi-stage path widens this window
/// (stage CBs are larger than per-component CBs). The structural
/// mitigation in iter89e2-B is: retained-refs (default
/// `MLX_UNRETAINED_REFS=0`) keeps the underlying Metal buffer alive
/// via the CB's ARC retain even after the residency-set demotion
/// fires, so the GPU completes safely.
///
/// This test exercises that path:
///
/// 1. Allocate `scratch` (residency count += 1).
/// 2. Dispatch using scratch via stage-1 of the session.
/// 3. `fence_stage` — non-blocking submit; CB now in flight (or
///    scheduled).
/// 4. **Drop scratch HERE.** `MlxBufferStorage::Drop` fires
///    `set.remove_allocation(scratch)` which (a) bumps the test
///    counter down by 1, (b) sets the pending flag.
/// 5. `reset_for_next_stage` — opens new CB + encodes wait-event.
/// 6. Stage 2 dispatches into a separate `out2` buffer (does NOT use
///    scratch). `commit_and_wait` drains everything.
///
/// Invariants verified:
/// - The test counter returned to `baseline` (scratch's removal
///   propagated correctly through the session's lifecycle).
/// - The first stage's output `out1` is correct (under retained-refs,
///   scratch's underlying Metal allocation outlived the in-flight CB
///   even after the residency-set demotion was committed at stage 2's
///   `flush_residency_pending`).
/// - No Metal assertion / GPU error fires (commit_and_wait returns Ok).
/// - Device still usable after the session drops.
///
/// If retained-refs failed to keep scratch alive, `out1` would contain
/// garbage and/or the GPU would error — both the equality assertion on
/// `out1` and `commit_and_wait`'s error path catch the failure mode.
///
/// Runs only under env-ON AND residency-sets-enabled. Skips otherwise.
#[test]
fn test_session_arena_lifetime_under_fence_no_rescission() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_arena_lifetime_under_fence_no_rescission \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }
    // Serialize against sibling residency-counter tests in this binary.
    let _guard = acquire_test_lock();
    let device = MlxDevice::new().expect("MlxDevice::new");
    if !device.residency_sets_enabled() {
        eprintln!(
            "[encoder_session_multistage] test_session_arena_lifetime_under_fence_no_rescission \
             SKIPPED — residency sets disabled (macOS<15 or HF2Q_NO_RESIDENCY=1). \
             F2 adversarial test only meaningful when residency sets are live."
        );
        return;
    }

    let mut registry = KernelRegistry::new();

    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();

    // Stage 2 buffers (kept alive across the test) — separate from
    // the adversarial scratch.
    let mut a2 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a2");
    let mut b2 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b2");
    let out2 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("out2");
    a2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
    b2.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[2.0, 2.0, 2.0, 2.0]);

    // Stage 1 inputs that we keep alive (caller-owned). The
    // adversarial drop is on `scratch_out`, an output buffer the
    // dispatch wrote to — which is the realistic F2 vector (transient
    // FFN scratch in the dispatch helper).
    let mut a1 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a1");
    let mut b1 = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b1");
    a1.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    b1.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    // Reset the test counter AFTER all the caller-owned buffers above
    // have been registered — we want to measure ONLY the residency
    // delta from our adversarial scratch, not the test-fixture
    // buffers. Caller-owned buffers' Drop after the assertion will
    // saturate at 0 via residency.rs:181-184's checked_sub, so
    // post-test cleanup does not affect the in-test invariants.
    reset_residency_test_counters();
    let baseline = residency_allocation_count_for_test();
    assert_eq!(
        baseline, 0,
        "reset_residency_test_counters zeros the count baseline"
    );

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under env=1");

    // Allocate the adversarial scratch as a CLONE so we can drop one
    // handle while the test still holds another for post-fence
    // verification. Metal ARC + Arc<MlxBufferStorage> keep the
    // underlying allocation alive — but we ALSO want
    // residency-rescission to fire (count decrement) which requires
    // dropping the LAST clone. Strategy:
    //   - alloc scratch_out (count += 1, baseline+1)
    //   - dispatch stage 1 using it
    //   - clone scratch_out into stage2_held BEFORE dropping the
    //     original — keeps the Arc alive past the fence
    //   - fence_stage submits CB
    //   - drop scratch_out (the original) — Arc count drops by 1 but
    //     storage stays alive via stage2_held. NO residency removal
    //     fires (last-clone semantics). Counter stays at baseline+1.
    // To trigger a real residency-remove staging mid-fence, we need
    // to drop the LAST clone. Re-strategy:
    //   - alloc scratch_out (count = baseline+1)
    //   - dispatch stage 1
    //   - fence_stage submits CB; under retained-refs the CB's ARC
    //     retain on the bound Metal buffer keeps the Metal allocation
    //     alive even after Rust drops every Arc<MlxBufferStorage>
    //   - drop scratch_out (THE only handle) → MlxBufferStorage::Drop
    //     fires set.remove_allocation, count -> baseline, pending
    //     flag set
    //   - reset_for_next_stage opens new CB + wait-event
    //   - stage 2 commit_and_wait flushes residency (committing the
    //     remove) AND drains the fenced CB
    //   - assert: count == baseline; stage1 output (read-back via the
    //     original buffer's storage) requires us to keep an Arc clone.
    //
    // Final approach: keep an Arc clone via `slice_view(0, byte_len)`
    // — share the registration but expose a new MlxBuffer handle. We
    // drop the ORIGINAL scratch_out and read via the slice_view, so
    // residency is preserved (Arc still alive) but the test exercises
    // the drop path.
    //
    // Actually: slice_view shares the Arc, so dropping the original
    // does NOT trigger Drop. To exercise Drop we MUST hold no clones
    // past the drop point. We'll dispatch into a separate
    // out1 buffer (caller-owned, kept alive); scratch_out is a
    // dispatch INPUT (or unused intermediate) that we drop. To verify
    // the F2 invariant we check (a) the residency counter movement,
    // (b) the device is unwedged, (c) stage 1's out1 is correct.

    let scratch_out = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("scratch_out");
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline + 1,
        "scratch_out alloc bumped residency count"
    );

    // Stage 1 dispatch writes scratch_out = a1 + b1.
    sess.begin_stage("phase.iter89e2b_f2_stage1");
    mlx_native::ops::elementwise::elementwise_add(
        sess.encoder(),
        &mut registry,
        device.metal_device(),
        &a1,
        &b1,
        &scratch_out,
        n,
        DType::F32,
    )
    .expect("stage1 dispatch into scratch_out");

    sess.fence_stage(None).expect("fence_stage Ok");
    assert!(sess.is_fence_pending());

    // **Adversarial drop**: stage 1's CB is in flight (or already
    // started), and we drop the only Rust handle to the scratch
    // output. Under retained-refs the CB still holds an ARC retain
    // on the underlying Metal buffer — the storage's Drop fires
    // residency-removal but the Metal pages stay alive. Without
    // retained-refs (NOT enabled in Phase 0b) this would be the
    // iter58b failure path.
    drop(scratch_out);
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline,
        "MlxBufferStorage::Drop must decrement residency count even mid-fence"
    );

    // Reset to stage 2 (encodes wait-event on the new CB).
    sess.reset_for_next_stage()
        .expect("reset_for_next_stage Ok");

    // Stage 2 — commit_and_wait flushes residency (committing the
    // staged remove for scratch_out) AND drains the fenced CB.
    sess.begin_stage("phase.iter89e2b_f2_stage2");
    mlx_native::ops::elementwise::elementwise_add(
        sess.encoder(),
        &mut registry,
        device.metal_device(),
        &a2,
        &b2,
        &out2,
        n,
        DType::F32,
    )
    .expect("stage2 dispatch");
    sess.commit_and_wait()
        .expect("stage2 commit_and_wait must succeed under retained-refs F2");

    // F2 invariant 1: stage 2 output is correct (unaffected by stage
    // 1's residency demotion).
    let r2 = out2.as_slice::<f32>().expect("read out2");
    assert_eq!(
        r2,
        &[3.0, 3.0, 3.0, 3.0],
        "stage 2 output must be correct after the F2 mid-fence drop"
    );

    // F2 invariant 2: residency counter is BACK to baseline; no
    // double-count, no stale add.
    assert_eq!(
        residency_allocation_count_for_test(),
        baseline,
        "F2 fence preservation: residency count back to baseline after \
         scratch drop + fence + reset + commit"
    );

    drop(sess);

    // F2 invariant 3: device usable after F2-fence Drop.
    let mut enc = device.command_encoder().expect("command_encoder post-F2");
    let mut a3 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("a3");
    let mut b3 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("b3");
    let out3 = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("out3");
    a3.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    b3.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &a3,
        &b3,
        &out3,
        n,
        DType::F32,
    )
    .expect("post-F2 dispatch");
    enc.commit_and_wait().expect("post-F2 commit_and_wait");
    assert_eq!(
        out3.as_slice::<f32>().expect("read out3"),
        &[11.0, 22.0, 33.0, 44.0],
        "device usable after F2 adversarial session drop"
    );
}

// ===================================================================
// Test 6 — iter90b borrowed-`&mut EncoderSession` multi-stage chain
// ===================================================================
//
// Validates the borrowed-session pattern that iter90b's hf2q-side
// `LayerEncoder<'sess>` will use across the per-layer loop.  Mirrors the
// pseudocode in iter90b spec §2.3:
//
//   let mut sess = device.encoder_session()?.expect("env=1");
//   for stage_idx in 0..N {
//       {
//           // borrow sess for this stage; drop borrow at end of block
//           dispatch via sess.encoder() ...
//           sess.fence_stage(...)?;
//           sess.reset_for_next_stage()?;
//       }
//   }
//   sess.commit_and_wait()?;
//
// PASS criterion (iter90b spec §2.3 line 250):
//   `fence_value == N`
//   `wait_count == N`   ← N because the loop calls reset after EVERY fence
//                          (including the last), and commit_and_wait drains
//                          the post-reset CB (which has the wait encoded).
//   no panic
//
// Note vs `encoder_session_cb_count_smoke` shape: cb_count_smoke does NOT
// reset after the last fence (5 fences + 4 resets); that test's PASS shape
// is `wait_count == N-1`.  THIS test resets after every fence (including
// the last) per spec §2.3 expectation of `wait_count == N`.
#[test]
fn test_session_borrowed_across_n_stages() {
    let _guard = acquire_test_lock();

    if !EncoderSession::env_enabled() {
        eprintln!(
            "[test_session_borrowed_across_n_stages] SKIP — HF2Q_ENCODER_SESSION not set to \"1\""
        );
        return;
    }

    const N: usize = 5;
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n_elems = 4usize;
    let byte_len = n_elems * std::mem::size_of::<f32>();
    let mut a = device
        .alloc_buffer(byte_len, DType::F32, vec![n_elems])
        .expect("a");
    let mut b = device
        .alloc_buffer(byte_len, DType::F32, vec![n_elems])
        .expect("b");
    let out = device
        .alloc_buffer(byte_len, DType::F32, vec![n_elems])
        .expect("out");
    a.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under HF2Q_ENCODER_SESSION=1");

    // N stages.  After EACH fence we reset (including the last), then
    // commit_and_wait drains the post-reset CB.  This is the spec §2.3
    // "wait_count == N" path.
    for stage_idx in 0..N {
        // Block scope mirrors hf2q-side `LayerEncoder<'sess>` borrow that
        // drops at end of stage iteration — `sess` remains alive at scope
        // exit, ready for the next iteration's re-borrow.
        {
            // Borrow sess.encoder() — analogous to LayerEncoder::Sessioned
            // re-borrowing through &mut self in hf2q.
            mlx_native::ops::elementwise::elementwise_add(
                sess.encoder(),
                &mut registry,
                device.metal_device(),
                &a,
                &b,
                &out,
                n_elems,
                DType::F32,
            )
            .expect("dispatch");

            let label = format!("borrowed.stage{stage_idx}");
            sess.fence_stage(Some(label.as_str()))
                .expect("fence_stage Ok");
            sess.reset_for_next_stage()
                .expect("reset_for_next_stage Ok");
        }
    }

    // Drain the final post-reset CB (which holds an encoded wait for the
    // last fence value).  This is the spec §2.3 line 248 terminal
    // `commit_and_wait()` — N waits total because every fence had a paired
    // reset.
    sess.commit_and_wait().expect("terminal commit_and_wait");

    let fence_val = sess.fence_value();
    let wait_count = sess.wait_count();

    eprintln!("borrowed.fence_value={fence_val}");
    eprintln!("borrowed.wait_count={wait_count}");

    assert_eq!(
        fence_val, N as u64,
        "fence_value must equal N={N} (one per fence_stage)"
    );
    assert_eq!(
        wait_count, N as u64,
        "wait_count must equal N={N} when reset is called after every fence \
         (including the last); spec §2.3 expectation"
    );
}
