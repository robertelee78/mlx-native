//! ADR-019 Phase 2b iter90b §2 — `EncoderSession` wait-event smoke test.
//!
//! ## Purpose
//!
//! Closes the H1b proof gap from iter90: the multi-stage chain MUST emit
//! `MTLSharedEvent encodeWaitForEvent` on each `reset_for_next_stage`
//! call that follows a `fence_stage`. iter90 only proved the SIGNAL side
//! (via `fence_value`); the WAIT side was unobservable from Rust without
//! xctrace. iter90b §2 adds two pure read-only introspection methods on
//! `EncoderSession` — `wait_value()` and `wait_count()` — that mirror the
//! signal-side `fence_value()` / event-allocation scoreboard. This test
//! asserts both methods report the values expected for a 5-fence chain.
//!
//! ## Wire-up
//!
//! Inside `EncoderSession::reset_for_next_stage` (encoder_session.rs:544),
//! the fence-pending branch calls `inner.encode_wait_for_event(event_ref,
//! value)` — this test exercises that branch 4 times (5 fences chained
//! by 4 resets — the 5th fence is terminal, drained by
//! `wait_until_completed`).
//!
//! ## PASS criterion (worker-α subtask A acceptance gate)
//!
//! All three observables MUST hold AND MUST be printed to stderr:
//!
//! ```text
//! fence_value=5
//! wait_count=4
//! wait_value=4
//! ```
//!
//! - `fence_value == 5`: 5 fence_stage calls were executed (signal side).
//! - `wait_count == 4`: 4 of the 5 reset_for_next_stage calls actually
//!   emitted a wait-event (the 4 between fences 1–4 → 5; the 5th fence
//!   has NO trailing reset — chain terminates with wait_until_completed).
//! - `wait_value == 4`: the most recent wait-event was at value 4 — the
//!   wait emitted on the reset that followed fence #4 (which signaled
//!   value 4). After fence #5 there is NO subsequent reset, so wait_value
//!   stays at 4 (the high-water mark of waits actually emitted).
//!
//! ## Env-var hygiene — load-bearing
//!
//! `HF2Q_ENCODER_SESSION` is cached by `EncoderSession::env_enabled()`
//! via `OnceLock` (`encoder_session.rs:135-143`). Setting the var AFTER
//! the first read is a no-op. This binary is a separate cargo
//! integration-test process, so the OnceLock primes from the OS env at
//! the first `env_enabled()` call.
//!
//! Run with env=1 (exercises both signal + wait paths):
//!
//! ```bash
//! HF2Q_ENCODER_SESSION=1 cargo test --release --test encoder_session_wait_event_smoke -- --nocapture
//! ```
//!
//! Run without env (session path skipped; test still PASS via documented
//! `eprintln` skip):
//!
//! ```bash
//! cargo test --release --test encoder_session_wait_event_smoke -- --nocapture
//! ```
//!
//! ## Counter isolation
//!
//! `wait_count` and `last_wait_value` are PER-SESSION (`u64` fields on
//! `EncoderSession`, NOT process-globals). No cross-test contamination
//! is possible — each test owns its own session. The `TEST_LOCK` below
//! is a defensive copy of the canonical pattern from
//! `tests/encoder_session_multistage.rs::RESIDENCY_TEST_LOCK`; future
//! tests added to this binary that touch process-global counters will
//! inherit the right shape.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Mutex;

use mlx_native::{DType, EncoderSession, KernelRegistry, MlxDevice};

/// Serializes ALL tests in this binary against any process-global
/// counters they may incidentally bump (residency, CMD_BUF_COUNT,
/// SYNC_COUNT). Mirror of `tests/encoder_session_multistage.rs`'s lock.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Lock-acquire helper that recovers from poisoning so a panicked
/// sibling test doesn't shadow the actual failure with `PoisonError`.
fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 5-stage chain wait-event smoke — H1b proof per iter90b §2.
///
/// Layout (mirrors `encoder_session_cb_count_smoke::encoder_session_cb_count_smoke`
/// session path at lines 219–252, with the wait-event scoreboard added):
///
/// ```text
///   encoder_session()             → fresh session; fence_value=0,
///                                   wait_count=0, wait_value=0
///   for i in 0..5 {
///     dispatch elementwise_add
///     fence_stage(Some(label))    → signal value = i+1; SharedEvent
///                                   lazy-alloc on i==0
///     if i < 4 {
///       reset_for_next_stage()    → emit wait at value (i+1); bump
///                                   wait_count by 1, set wait_value
///                                   to (i+1)
///     }
///   }
///   metal_command_buffer().wait_until_completed()  → drain CB #5
/// ```
///
/// After the loop:
/// - `fence_value() == 5`     (one per fence_stage)
/// - `wait_count() == 4`      (one per non-terminal reset)
/// - `wait_value() == 4`      (the wait following fence #4 at value 4)
///
/// Skipped (documented `eprintln`) when `HF2Q_ENCODER_SESSION` is not
/// `"1"` in the process env. The skip prints the three expected fields
/// with sentinel values so log-grep tooling can distinguish "skipped"
/// from "failed".
#[test]
fn encoder_session_wait_event_smoke() {
    let _guard = acquire_test_lock();

    // ------------------------------------------------------------------
    // Environment check.
    // ------------------------------------------------------------------
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_wait_event_smoke] SKIP — HF2Q_ENCODER_SESSION not set to \"1\" \
             in process env. Re-run with HF2Q_ENCODER_SESSION=1 to exercise the wait-event path.\n\
             fence_value=skipped\n\
             wait_count=skipped\n\
             wait_value=skipped"
        );
        return;
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    // Small buffer dimensions used for all dispatches. 4 f32 elements is
    // the minimum meaningful dispatch for elementwise_add.
    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();

    let mut a = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("a");
    let mut b = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("b");
    let out = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("out");
    a.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

    // ------------------------------------------------------------------
    // Build session and verify fresh-session invariants.
    // ------------------------------------------------------------------
    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under HF2Q_ENCODER_SESSION=1");

    assert_eq!(
        sess.fence_value(),
        0,
        "fresh session: fence_value starts at 0"
    );
    assert_eq!(
        sess.wait_count(),
        0,
        "fresh session: wait_count starts at 0"
    );
    assert_eq!(
        sess.wait_value(),
        0,
        "fresh session: wait_value starts at 0 (no wait emitted yet)"
    );
    assert!(
        !sess.has_event(),
        "fresh session: no SharedEvent allocated until first fence_stage"
    );

    // ------------------------------------------------------------------
    // 5-stage chain. Stages 0..4 fence + reset; stage 4 fences only
    // (terminal — drained below by wait_until_completed).
    // ------------------------------------------------------------------
    for i in 0..5usize {
        let label = format!("session.wait_smoke.stage{i}");

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
        .expect("elementwise_add session");

        sess.fence_stage(Some(label.as_str()))
            .expect("fence_stage Ok");

        // After fence #(i+1):
        //   fence_value == i+1
        //   is_fence_pending == true
        //   is_drained == true
        //   has_event == true (lazy-alloc on i==0; persists thereafter)
        let expected_signal = (i as u64) + 1;
        assert_eq!(
            sess.fence_value(),
            expected_signal,
            "after fence_stage #{i}: fence_value must be {expected_signal}"
        );
        assert!(
            sess.is_fence_pending(),
            "after fence_stage #{i}: is_fence_pending must be true"
        );
        assert!(
            sess.is_drained(),
            "after fence_stage #{i}: is_drained must be true"
        );
        assert!(
            sess.has_event(),
            "after fence_stage #{i}: has_event must be true (lazy-alloc on i==0)"
        );

        if i < 4 {
            // Non-terminal reset: emit wait-event at the current
            // fence_value (i+1), bump wait_count, set wait_value.
            sess.reset_for_next_stage()
                .expect("reset_for_next_stage Ok");

            // Per-iteration scoreboard check after the reset:
            //   wait_count == i+1 (one wait per non-terminal reset so far)
            //   wait_value == i+1 (most recent wait was at fence_value i+1)
            //   is_fence_pending == false (cleared by reset)
            //   is_drained == false (cleared by reset)
            let expected_wait = (i as u64) + 1;
            assert_eq!(
                sess.wait_count(),
                expected_wait,
                "after reset following fence #{i}: wait_count must be {expected_wait}"
            );
            assert_eq!(
                sess.wait_value(),
                expected_wait,
                "after reset following fence #{i}: wait_value must be {expected_wait} \
                 (matches the signal we just signaled)"
            );
            assert!(
                !sess.is_fence_pending(),
                "after reset following fence #{i}: is_fence_pending must be cleared"
            );
            assert!(
                !sess.is_drained(),
                "after reset following fence #{i}: is_drained must be cleared"
            );
        }
        // Stage 4 (i == 4): NO reset — terminal. Session is in
        // Fenced/Drained state with fence #5 in flight on the GPU.
    }

    // Capture final observables BEFORE waiting (the wait does not
    // modify these scoreboards, but we read them here so the printed
    // values reflect the state at the loop's end).
    let fence_val = sess.fence_value();
    let wait_count = sess.wait_count();
    let wait_value = sess.wait_value();

    // Drain the terminal fenced CB so the GPU work completes before
    // session drop. wait_until_completed does NOT bump fence_value /
    // wait_count / wait_value (those are scoreboards over Rust-side API
    // calls, not GPU completion events).
    sess.metal_command_buffer().wait_until_completed();

    // Re-read post-drain to verify the scoreboards are stable across
    // wait_until_completed (proof that the introspection methods are
    // pure-read, not influenced by GPU side effects).
    assert_eq!(
        sess.fence_value(),
        fence_val,
        "fence_value must be stable across wait_until_completed"
    );
    assert_eq!(
        sess.wait_count(),
        wait_count,
        "wait_count must be stable across wait_until_completed"
    );
    assert_eq!(
        sess.wait_value(),
        wait_value,
        "wait_value must be stable across wait_until_completed"
    );

    // ------------------------------------------------------------------
    // Print observables — eprintln so they survive cargo's stdout
    // capture. The judge greps these three lines from the --nocapture
    // run.
    // ------------------------------------------------------------------
    eprintln!("fence_value={fence_val}");
    eprintln!("wait_count={wait_count}");
    eprintln!("wait_value={wait_value}");

    // ------------------------------------------------------------------
    // Final PASS-criterion assertions (worker-α subtask A acceptance).
    // ------------------------------------------------------------------
    assert_eq!(
        fence_val, 5,
        "fence_value must be 5 after exactly 5 fence_stage calls (got {fence_val})"
    );
    assert_eq!(
        wait_count, 4,
        "wait_count must be 4 (4 non-terminal resets between 5 fences; got {wait_count})"
    );
    assert_eq!(
        wait_value, 4,
        "wait_value must be 4 (the wait following fence #4 at value 4; got {wait_value})"
    );
}
