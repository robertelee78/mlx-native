//! ADR-019 Phase 2 iter90b S3 — CB-count REDUCTION smoke test for EncoderSession.
//!
//! ## Purpose
//!
//! Verifies the structural CB-count reduction promised by iter90b's
//! borrowed-`&mut EncoderSession` multi-stage chain.  Per iter90b spec §3
//! (`/opt/hf2q/.cfa-archive/iter90b/spec.md`), the H2b reduction is achieved
//! by replacing the per-FFN-layer `fence_stage` between layer N's FFN and
//! layer N+1's attention with `enc.memory_barrier()` — the next layer's
//! pre-attention norm encodes into the SAME persistent compute encoder.
//!
//! This test mimics that pattern at toy scale: 5 "layer pairs" of
//! attention-then-FFN dispatches.  In the plain path each stage opens a
//! fresh `device.command_encoder()` (10 CBs total).  In the sessioned path
//! the FFN→next-attention boundary is a `memory_barrier()` (no fence, no
//! commit), so 5 attention CBs cover BOTH the attention dispatch AND the
//! preceding layer's FFN.
//!
//! ## iter90b H2b PASS criterion (replaces iter90 OQ1 weak `>=` check)
//!
//! PASS iff:
//!   1. `fence_value == 5` (one fence per attention boundary).
//!   2. `cb_count_session * 2 <= cb_count_plain` — strict factor-2x reduction.
//!      Specifically `cb_count_session == 5` and `cb_count_plain == 10`.
//!   3. `wait_count == 4` (one wait per non-terminal `reset_for_next_stage`).
//!   4. No panic during the run.
//!
//! Observable output (eprintln so it survives cargo's stdout capture):
//!   `fence_value=<N>`        (must be 5)
//!   `cb_count_plain=<N>`     (must be 10)
//!   `cb_count_session=<N>`   (must be 5; <= cb_count_plain / 2)
//!   `wait_count=<N>`         (must be 4)
//!
//! ## Why a STRICT inequality and not the iter90 weak `>=`
//!
//! The iter90 test asserted `cb_count_session >= cb_count_plain` — a
//! forward-compatible no-regression guard, but it could not catch a bug
//! where `carry_into_next_stage` was implemented as `fence_or_commit`
//! instead of `memory_barrier`.  iter90b's `<=` half-of-plain assertion
//! IS the H2b structural proof: it FAILS if the FFN→attention boundary
//! emits a fence_stage instead of a memory_barrier.
//!
//! ## Env-var hygiene (unchanged from iter90)
//!
//! `HF2Q_ENCODER_SESSION` cached via `OnceLock` at first
//! `EncoderSession::env_enabled()` read in this process.  No `set_var`.
//! Run with env=1 to exercise the sessioned path.
//!
//! ## Counter isolation
//!
//! `CMD_BUF_COUNT` is process-global; `TEST_LOCK` serializes within this
//! binary (one test).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Mutex;

use mlx_native::{
    cmd_buf_count, reset_counters, DType, EncoderSession, KernelRegistry, MlxDevice,
};

/// Serializes ALL tests in this binary against the process-global
/// CMD_BUF_COUNT and residency counters.  Copied from
/// `tests/encoder_session_multistage.rs::RESIDENCY_TEST_LOCK`.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Lock-acquire helper that recovers from poisoning.
fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// CB-count REDUCTION smoke test for iter90b's borrowed-session multi-stage
/// chain.
///
/// Path A (plain — legacy 10-CB pattern):
///   For each of 5 layer pairs:
///     attention CB:  open command_encoder() + dispatch + commit_labeled
///     FFN CB:        open command_encoder() + dispatch + commit_labeled
///   Total: 5 + 5 = 10 CBs.
///
/// Path B (sessioned — iter90b 5-CB pattern):
///   One `encoder_session()` covering the entire chain.
///   For each of 5 layer pairs:
///     attention dispatch (encodes into the persistent compute encoder)
///     intra-CB memory_barrier()  ← RAW dep between attention and FFN
///     FFN dispatch (encodes into the SAME persistent compute encoder)
///     fence_stage("attn.{i}")    ← terminates the layer's CB
///     [if i<4] reset_for_next_stage()  ← rotates to fresh CB w/ wait
///   Total: 1 (initial) + 4 (resets) = 5 CBs.
///
/// Assertions per iter90b spec §3.4:
///   `fence_value == 5`
///   `cb_count_session == 5`
///   `cb_count_plain == 10`
///   `cb_count_session * 2 <= cb_count_plain`  ← H2b structural proof
///   `wait_count == 4`                          ← H1b sanity
///
/// Skipped (documented eprintln) when `EncoderSession::env_enabled() == false`.
#[test]
fn encoder_session_cb_count_smoke() {
    let _guard = acquire_test_lock();

    if !EncoderSession::env_enabled() {
        eprintln!(
            "[encoder_session_cb_count_smoke] SKIP — HF2Q_ENCODER_SESSION not set to \"1\" \
             in process env.  Re-run with HF2Q_ENCODER_SESSION=1 to exercise the H2b path.\n\
             fence_value=skipped\n\
             cb_count_plain=skipped\n\
             cb_count_session=skipped\n\
             wait_count=skipped"
        );
        return;
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    // Two scratch buffers shared across both paths.  4 f32 elements is the
    // minimum meaningful dispatch for elementwise_add.
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
    // Path A — plain: 5 attention-FFN layer pairs, 2 CBs per pair.
    //
    // Each pair:
    //   attention:  command_encoder() + dispatch + commit_labeled
    //   FFN:        command_encoder() + dispatch + commit_labeled
    //
    // Expected delta: 10 CBs.
    // ------------------------------------------------------------------
    reset_counters();
    let cb_before_plain = cmd_buf_count();

    for i in 0..5usize {
        // Attention CB.
        {
            let mut enc = device
                .command_encoder()
                .expect("command_encoder plain attn");
            mlx_native::ops::elementwise::elementwise_add(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &a,
                &b,
                &out,
                n,
                DType::F32,
            )
            .expect("elementwise_add plain attn");
            let label = format!("plain.attn.layer{i}");
            enc.commit_labeled(&label);
        }
        // FFN CB.
        {
            let mut enc = device
                .command_encoder()
                .expect("command_encoder plain ffn");
            mlx_native::ops::elementwise::elementwise_add(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &a,
                &b,
                &out,
                n,
                DType::F32,
            )
            .expect("elementwise_add plain ffn");
            let label = format!("plain.ffn.layer{i}");
            enc.commit_labeled(&label);
        }
    }

    // Drain to ensure plain CBs complete before resetting counters for
    // path B.
    {
        let mut drain_enc = device.command_encoder().expect("drain encoder");
        mlx_native::ops::elementwise::elementwise_add(
            &mut drain_enc,
            &mut registry,
            device.metal_device(),
            &a,
            &b,
            &out,
            n,
            DType::F32,
        )
        .expect("drain dispatch");
        drain_enc.commit_and_wait().expect("drain commit_and_wait");
    }

    let cb_after_plain = cmd_buf_count();
    // 10 plain CBs + 1 drain CB; subtract drain.
    let cb_count_plain = cb_after_plain - cb_before_plain - 1;

    // ------------------------------------------------------------------
    // Path B — sessioned: 5 layer pairs, 1 CB per pair via in-CB chaining.
    //
    // Layout per iter90b spec §3.1, §3.3:
    //   encoder_session()             → CB count + 1 (initial CB)
    //   [for i in 0..5]
    //     attention dispatch  (intra-CB)
    //     memory_barrier()    (intra-CB RAW for FFN reads attn output)
    //     FFN dispatch        (intra-CB)
    //     fence_stage(label)  (no CB count change)
    //     [if i < 4] reset_for_next_stage()  → CB count + 1, wait_count + 1
    //
    // Expected delta: 1 (initial) + 4 (resets) = 5.
    // Expected fence_value: 5.
    // Expected wait_count:  4.
    // ------------------------------------------------------------------
    reset_counters();
    let cb_before_session = cmd_buf_count();

    let mut sess = device
        .encoder_session()
        .expect("encoder_session() Ok")
        .expect("Some under HF2Q_ENCODER_SESSION=1");

    for i in 0..5usize {
        // "Attention" dispatch — encodes into the persistent compute encoder.
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
        .expect("elementwise_add session attn");

        // Intra-CB RAW barrier — FFN reads attention output.  This is
        // exactly what `LayerEncoder::carry_into_next_stage` does on the
        // Sessioned variant in iter90b's hf2q wire-up: NO commit, NO
        // fence — the FFN dispatch encodes into the SAME persistent
        // compute encoder as the attention dispatch.
        sess.encoder().memory_barrier();

        // "FFN" dispatch — same persistent compute encoder as attention.
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
        .expect("elementwise_add session ffn");

        // Layer boundary: fence the CB.  Stages 1–4 also reset; stage 5
        // does NOT reset (terminal — drained by wait_until_completed).
        let label = format!("session.attn.layer{i}");
        sess.fence_stage(Some(label.as_str()))
            .expect("fence_stage Ok");

        if i < 4 {
            sess.reset_for_next_stage().expect("reset_for_next_stage Ok");
        }
    }

    // Snapshot scoreboards BEFORE drain (introspection is pure-read; the
    // post-drain re-read should match).
    let fence_val = sess.fence_value();
    let wait_count = sess.wait_count();

    // Drain the last fenced CB.  fence_stage submitted it non-blocking;
    // metal_command_buffer() returns that CB.
    sess.metal_command_buffer().wait_until_completed();

    let cb_after_session = cmd_buf_count();
    let cb_count_session = cb_after_session - cb_before_session;

    // ------------------------------------------------------------------
    // Print observables.
    // ------------------------------------------------------------------
    eprintln!("fence_value={fence_val}");
    eprintln!("cb_count_plain={cb_count_plain}");
    eprintln!("cb_count_session={cb_count_session}");
    eprintln!("wait_count={wait_count}");

    // ------------------------------------------------------------------
    // Assertions per iter90b spec §3.4 PASS criterion (AC-2b).
    // ------------------------------------------------------------------
    assert_eq!(
        fence_val, 5,
        "fence_value must be 5 after exactly 5 fence_stage calls (got {fence_val})"
    );

    assert_eq!(
        cb_count_plain, 10,
        "cb_count_plain must be 10 (5 attention + 5 FFN CBs); got {cb_count_plain}"
    );

    assert_eq!(
        cb_count_session, 5,
        "cb_count_session must be 5 (1 initial + 4 resets, with FFN folded \
         into each attention CB via memory_barrier); got {cb_count_session}"
    );

    assert!(
        cb_count_session * 2 <= cb_count_plain,
        "iter90b H2b structural proof: cb_count_session ({cb_count_session}) must be \
         at most half of cb_count_plain ({cb_count_plain}).  FAILURE means the \
         FFN→next-attention boundary emitted fence_stage instead of memory_barrier — \
         i.e. carry_into_next_stage on the Sessioned variant did NOT keep the \
         persistent compute encoder open."
    );

    assert_eq!(
        wait_count, 4,
        "wait_count must be 4 (one wait per non-terminal reset_for_next_stage); \
         got {wait_count}"
    );
}
