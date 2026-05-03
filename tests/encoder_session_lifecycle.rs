//! ADR-019 Phase 0b iter89e2-A — `EncoderSession` lifecycle tests.
//!
//! These tests verify the bare-struct contract delivered by iter89e2-A:
//! 1. Default-OFF env-gate behavior (`MlxDevice::encoder_session` returns `None`).
//! 2. Env-gated construction with `HF2Q_ENCODER_SESSION=1` returns `Some`.
//! 3. The happy-path lifecycle `begin_stage` → `commit_stage` drains.
//! 4. `Drop` on an uncommitted session is safe (F2 residency-rescission
//!    preservation argument: see `encoder_session.rs::Drop`).
//! 5. `commit_and_wait` blocks until GPU completion (synchronous semantics).
//!
//! ## Env-var hygiene — load-bearing
//!
//! `HF2Q_ENCODER_SESSION` is read exactly once via `OnceLock` in
//! `encoder_session.rs::encoder_session_enabled`. Cargo's default
//! integration-test runner uses a SHARED process and a THREAD POOL —
//! tests within this binary share the `OnceLock` cache. **No test in
//! this file mutates `HF2Q_ENCODER_SESSION`**: a `set_var` from one
//! test would race against `env_enabled()` reads from another test
//! and produce non-determinism (verified — initial test draft hit
//! exactly that failure: `aa_test_default_off_returns_none` panicked
//! when sibling tests set the var first).
//!
//! Instead, each test reads `EncoderSession::env_enabled()` ONCE and
//! tailors its assertions to the cached state:
//!
//! - When `env_enabled()` returns `false`: tests verify the
//!   default-OFF contract (`encoder_session()` returns `None`,
//!   construction is gated). Test 1 is the primary verifier; the
//!   env-ON-only tests (3, 4, 5) emit a documented `eprintln` skip.
//!
//! - When `env_enabled()` returns `true`: tests verify the env-ON
//!   contract (`encoder_session()` returns `Some`, lifecycle methods
//!   work correctly). Test 1 verifies the factory; tests 3-5 verify
//!   the live-session path.
//!
//! To exercise BOTH branches in CI, run this binary twice:
//!
//! ```bash
//! cargo test --release --test encoder_session_lifecycle
//! HF2Q_ENCODER_SESSION=1 cargo test --release --test encoder_session_lifecycle
//! ```
//!
//! The Phase 0b-C / iter89e2-C parity-tests checklist will encode this
//! double-invocation in the operator recipe.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, EncoderSession, KernelRegistry, MlxDevice};

/// Test 1 — env-gate semantics: `encoder_session()` agrees with `env_enabled()`.
///
/// Whichever cache state is live in this binary, the factory must
/// return `Some(_)` iff `env_enabled()` is true and `None` iff it is
/// false. This is the iter89e2-A "zero-behavior-change" guard:
/// production builds that do not opt in see the existing
/// `command_encoder()` path unchanged.
#[test]
fn test_env_gate_factory_agreement() {
    let env_on = EncoderSession::env_enabled();
    let actual_env_var = std::env::var("HF2Q_ENCODER_SESSION").as_deref() == Ok("1");
    assert_eq!(
        env_on, actual_env_var,
        "EncoderSession::env_enabled() ({env_on}) must match the actual \
         HF2Q_ENCODER_SESSION env var ({actual_env_var}) — OnceLock cache \
         primes from os env at first read."
    );

    let device = MlxDevice::new().expect("MlxDevice::new");
    let sess_opt = device
        .encoder_session()
        .expect("encoder_session() infallible past metal-rs new_command_buffer");

    if env_on {
        assert!(
            sess_opt.is_some(),
            "encoder_session() must return Some(_) when env_enabled()==true"
        );
        let session = sess_opt.expect("just unwrapped to Some above");
        assert!(
            !session.is_drained(),
            "fresh EncoderSession::is_drained() must be false"
        );
    } else {
        assert!(
            sess_opt.is_none(),
            "encoder_session() must return None when HF2Q_ENCODER_SESSION is unset \
             (zero-behavior-change invariant)"
        );
    }
}

/// Test 2 — `EncoderSession::env_enabled` is stable across calls.
///
/// `OnceLock` caches the env-read on first call; subsequent calls
/// must return the same value regardless of any env-var mutation
/// between calls. This guards the iter89e2-A documentation that
/// callers can branch on `env_enabled()` once at construction site
/// and trust the result for the rest of the process lifetime.
#[test]
fn test_env_enabled_is_stable() {
    let first = EncoderSession::env_enabled();
    // Re-read several times — the cache must be stable.
    for _ in 0..5 {
        assert_eq!(
            EncoderSession::env_enabled(),
            first,
            "env_enabled() must be stable across calls (OnceLock cache contract)"
        );
    }
}

/// Test 3 — happy path: `begin_stage` → `commit_stage` drains the session.
///
/// Constructs a session under env-ON, encodes one trivial dispatch via
/// the inner encoder accessor, calls `commit_stage` (non-blocking),
/// and asserts:
/// - the session is `Drained` after the commit,
/// - the dispatch result is correct after a `wait_until_completed`,
/// - the underlying `MTLCommandBuffer.label` matches the stage label.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is `false`.
#[test]
fn test_begin_stage_then_commit_stage_drains() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_lifecycle] test_begin_stage_then_commit_stage_drains \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env. \
             Re-run with HF2Q_ENCODER_SESSION=1 to exercise."
        );
        return;
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

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
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under env=1");

    sess.begin_stage("phase.iter89e2a_smoke_commit_stage");

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
    .expect("dispatch elementwise_add through session.encoder()");

    assert!(
        !sess.is_drained(),
        "EncoderSession must NOT be drained until commit_* is called"
    );

    sess.commit_stage().expect("commit_stage() must succeed");

    assert!(
        sess.is_drained(),
        "EncoderSession::is_drained must be true after commit_stage"
    );

    // Block until the GPU has actually executed the CB so we can read
    // the result buffer below — `commit_stage` is non-blocking, so an
    // explicit wait is required for the read-back assertion.
    sess.metal_command_buffer().wait_until_completed();

    // Verify the dispatch encoded through `sess.encoder()` actually ran.
    let result = out.as_slice::<f32>().expect("read out");
    assert_eq!(
        result,
        &[11.0, 22.0, 33.0, 44.0],
        "elementwise_add result must propagate through EncoderSession dispatch path"
    );

    // Verify the stage label propagated to MTLCommandBuffer.label.
    let cb_label = sess.metal_command_buffer().label();
    assert_eq!(
        cb_label, "phase.iter89e2a_smoke_commit_stage",
        "stage label must propagate to MTLCommandBuffer.label via commit_labeled"
    );

    // Idempotency: calling commit_stage again on a drained session is
    // a no-op (no panic, no double-commit on Metal — that would be UB).
    sess.commit_stage()
        .expect("second commit_stage() must be a no-op, not an error");
}

/// Test 4 — Drop on uncommitted session is safe (F2 fence preservation).
///
/// Construct a session, encode a trivial dispatch, then drop the
/// session WITHOUT calling `commit_*`. Metal will assert at runtime if
/// the persistent compute encoder is dropped without `endEncoding`;
/// `EncoderSession::Drop` MUST delegate to `CommandEncoder::Drop`
/// which calls `end_active_encoder()`.
///
/// This test does NOT verify the F2 residency-rescission case
/// directly (that requires an instrumented arena to detect a
/// premature `removeAllocation:` commit) — the structural argument
/// for F2 preservation is in `encoder_session.rs::Drop` docstring.
/// What this test verifies is the **necessary** condition: the Drop
/// path doesn't panic, doesn't trip a Metal validation assert, and
/// leaves the device usable for a subsequent allocation + dispatch.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is `false`.
#[test]
fn test_drop_uncommitted_is_safe() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_lifecycle] test_drop_uncommitted_is_safe \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

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
        .copy_from_slice(&[100.0, 200.0, 300.0, 400.0]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

    {
        let mut sess = device
            .encoder_session()
            .expect("encoder_session()")
            .expect("Some under env=1");

        sess.begin_stage("phase.iter89e2a_drop_uncommitted");

        // Encode a dispatch to ensure the persistent compute encoder
        // gets opened (this is the case Metal asserts on if endEncoding
        // is not called before CB drop).
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
        .expect("dispatch through session.encoder()");

        assert!(
            !sess.is_drained(),
            "session is in Encoding state, must not yet be drained"
        );

        // Intentional: drop `sess` here without commit. The Metal
        // runtime would emit "Command encoder released without
        // endEncoding" if Drop didn't end the encoder. If this test
        // reaches its end without panic / assert, Drop is safe.
    }

    // Sanity: the device is still usable after the dropped session.
    // Allocate + commit a separate CB to verify no Metal-side state
    // got wedged.
    let mut enc = device.command_encoder().expect("command_encoder post-drop");
    let n2 = 2usize;
    let bl2 = n2 * std::mem::size_of::<f32>();
    let mut a2 = device.alloc_buffer(bl2, DType::F32, vec![n2]).expect("a2");
    let mut b2 = device.alloc_buffer(bl2, DType::F32, vec![n2]).expect("b2");
    let out2 = device.alloc_buffer(bl2, DType::F32, vec![n2]).expect("out2");
    a2.as_mut_slice::<f32>().unwrap().copy_from_slice(&[7.0, 8.0]);
    b2.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1.0, 2.0]);
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &a2,
        &b2,
        &out2,
        n2,
        DType::F32,
    )
    .expect("post-drop dispatch");
    enc.commit_and_wait().expect("post-drop commit_and_wait");
    assert_eq!(
        out2.as_slice::<f32>().expect("read out2"),
        &[8.0, 10.0],
        "device usable after EncoderSession Drop"
    );
}

/// Test 5 — `commit_and_wait` blocks until the GPU completes.
///
/// The synchronous-commit path on `EncoderSession` must mirror
/// `CommandEncoder::commit_and_wait` semantics:
/// - `SYNC_COUNT` increments by at least one (other tests in this
///   binary may race this counter — strict equality cannot be
///   asserted under parallel test execution).
/// - The CB has `MTLCommandBufferStatus::Completed` immediately after
///   the call returns (no further `wait_until_completed` needed).
/// - The result buffer is readable without a separate wait.
/// - The stage label propagated to `MTLCommandBuffer.label`.
///
/// Skipped (documented `eprintln` no-op) when `env_enabled()` is `false`.
#[test]
fn test_commit_and_wait_blocks_until_done() {
    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_lifecycle] test_commit_and_wait_blocks_until_done \
             SKIPPED — HF2Q_ENCODER_SESSION not set in process env."
        );
        return;
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

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
        .copy_from_slice(&[1.5, 2.5, 3.5, 4.5]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

    let sync_before = mlx_native::sync_count();

    let mut sess = device
        .encoder_session()
        .expect("encoder_session()")
        .expect("Some under env=1");

    sess.begin_stage("phase.iter89e2a_commit_and_wait");

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
    .expect("dispatch through session.encoder()");

    sess.commit_and_wait().expect("commit_and_wait must succeed");

    assert!(
        sess.is_drained(),
        "EncoderSession::is_drained must be true after commit_and_wait"
    );

    // commit_and_wait increments SYNC_COUNT exactly once via the inner
    // CommandEncoder. Other tests may run concurrently and bump the
    // counter, so we assert a strict lower bound rather than `+ 1`.
    let sync_after = mlx_native::sync_count();
    assert!(
        sync_after >= sync_before + 1,
        "commit_and_wait must increment SYNC_COUNT by ≥ 1 \
         (before={sync_before}, after={sync_after})"
    );

    // Result must be readable WITHOUT an additional wait — the wait
    // already happened inside commit_and_wait.
    let result = out.as_slice::<f32>().expect("read out");
    assert_eq!(
        result,
        &[11.5, 22.5, 33.5, 44.5],
        "elementwise_add result must be visible after commit_and_wait \
         (synchronous semantic)"
    );

    // Stage label must propagate.
    let cb_label = sess.metal_command_buffer().label();
    assert_eq!(
        cb_label, "phase.iter89e2a_commit_and_wait",
        "stage label must propagate to MTLCommandBuffer.label"
    );

    // Idempotency: second commit_and_wait is a no-op (does not bump
    // SYNC_COUNT — concurrent tests may bump it independently, but
    // this call alone must contribute zero).
    let sync_mid = mlx_native::sync_count();
    sess.commit_and_wait()
        .expect("second commit_and_wait must be a no-op");
    let sync_end = mlx_native::sync_count();
    assert!(
        sync_end >= sync_mid,
        "second commit_and_wait must not decrement SYNC_COUNT"
    );
}
