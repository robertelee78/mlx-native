//! ADR-040 §0.21c-track2 — regression guard for the concurrent-encoder
//! ordering ROOT FIX (encoder retain, mlx-native 80a58de).
//!
//! ## The bug this guards against
//!
//! mlx-native drives a SINGLE persistent `MTLDispatchTypeConcurrent` compute
//! encoder (`encoder.rs::get_or_create_encoder`). metal-rs returns that encoder
//! as a +0 autoreleased object (gfx-rs/metal-rs#128). Before the fix, the
//! engine held it through a borrowed `*const ComputeCommandEncoderRef`
//! (`encoder.rs:757 active_encoder`) WITHOUT taking ownership. An autorelease-
//! pool drain between two dependent dispatches could drop the ownership state
//! Metal's barrier ordering keys on, so `memoryBarrierWithScope:Buffers` no
//! longer reliably ordered a slow producer (e.g. the Q6_K mvN lm_head matmul)
//! before its in-place consumer (softcap). The faster nr2 producer won the race
//! by timing; mvN lost it → ~1/3 of runtime-compile runs diverged. No crash
//! because the command buffer also references the encoder (no UAF), so the
//! symptom was silent data-race non-determinism, NOT a panic.
//!
//! llama.cpp uses the IDENTICAL primitive (concurrent encoder +
//! `[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]`,
//! ggml-metal-device.m:512) and works — because it RETAINS the encoder
//! (`[res->obj retain]`, released at end). The fix matches that: retain in
//! `get_or_create_encoder`, balanced release in `end_active_encoder`, and a
//! leak-safe `reset_command_buffer` that routes through `end_active_encoder`
//! (review Finder B, mlx-native 3e7b030).
//!
//! ## Why this is a STRUCTURAL guard + a functional round-trip, not a
//!    timing-race reproduction
//!
//! A faithful timing-race reproduction in a unit test is not reliable: the
//! race window depends on a real multi-stage decode workload (the slow Q6_K
//! mvN lm_head over vocab=262144 vs an in-place softcap) plus an autorelease
//! drain landing in exactly the wrong place. An isolated single-session writer
//! →barrier→reader with a per-iteration full sync does NOT open that window —
//! an earlier attempt at a pure timing test passed WITH and WITHOUT the fix and
//! was therefore deleted (it gave false confidence). The codebase convention
//! for "the fix is real but the failure can't be force-injected from a unit
//! test" is a source-structural regression guard plus a functional GPU round-
//! trip of the fixed path (see `encoder_session_lifecycle.rs` Test 6). This
//! file follows that convention:
//!
//! 1. `retain_root_fix_source_invariants` — reads `encoder.rs` and asserts the
//!    three load-bearing edits are present (retain, balanced release, reset
//!    routed through end_active_encoder). A revert of any one fails CI. THIS is
//!    the durable regression catch.
//! 2. `writer_barrier_reader_orders_through_retained_encoder` — a cheap
//!    functional SMOKE of the writer→barrier→reader path on the concurrent
//!    encoder. HONEST LIMIT: it does NOT reproduce the race (verified — it
//!    passes even with the retain reverted; an isolated add→mul edge with a
//!    per-iteration full-sync does not open the window, same as the earlier
//!    deleted timing test). It guards the happy path only. The real
//!    fail-direction discriminator is (1) above plus the production model-
//!    parity oracle `slot_aware_n8_per_slot_parity_vs_serial` under
//!    `MLX_PRECOMPILED_METALLIB=0` (which ran the actual slow Q6_K mvN lm_head
//!    vs in-place softcap and reproduced ~1/3 without the fix).
//!
//! ## CI recommendation
//!
//! The durable regression catch for the encoder-ordering fix is the structural
//! guard here PLUS running the hf2q model-parity test on the runtime-source-
//! compile path: `MLX_PRECOMPILED_METALLIB=0 ... slot_aware_n8_per_slot_parity\
//! _vs_serial`. That path is the only one that exercised the real race; the
//! -O3 metallib masks it. (Note: that parity test is currently confounded by a
//! separate, pre-existing batched-flash-attention non-determinism, §0.19 /
//! task #19 — kernel-independent — so it is not yet a clean 0-flake gate.)

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxDevice};

/// Structural regression guard — the THREE load-bearing edits of the retain
/// root fix (80a58de) + the leak-safe reset (3e7b030) must remain in
/// `encoder.rs`. This runs without a GPU and is the durable CI catch: any
/// revert that drops the retain, the balancing release, or the reset routing
/// fails here.
#[test]
fn retain_root_fix_source_invariants() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_path = std::path::Path::new(manifest_dir)
        .join("src")
        .join("encoder.rs");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_path.display()));

    // Match against ACTIVE CODE ONLY: drop any line whose first non-whitespace
    // characters are `//`. A text-only `contains` would be fooled by a
    // commented-out `// ... msg_send![encoder, retain]` revert (verified: a
    // comment-out evades a raw contains check). Stripping line comments makes a
    // comment-out revert fail this guard, not just a deletion.
    let active: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let active_norm: String = active.split_whitespace().collect::<Vec<_>>().join(" ");

    // (1) The concurrent compute encoder is RETAINED when created, as ACTIVE
    // code. This is the root fix: metal-rs returns it +0 autoreleased; we take
    // ownership so an autorelease drain cannot drop the state Metal's barrier
    // ordering keys on. Catches both deletion AND comment-out reverts.
    assert!(
        active_norm.contains("msg_send![encoder, retain]"),
        "encoder.rs::get_or_create_encoder MUST retain the concurrent compute \
         encoder as ACTIVE code (ADR-040 §0.21c-track2 root fix 80a58de) — not \
         deleted, not commented out. Without it, an autorelease-pool drain drops \
         the encoder ownership state and memoryBarrierWithScope no longer \
         reliably orders dependent dispatches."
    );

    // (2) The retain is BALANCED by a release in end_active_encoder (no leak),
    // also as active code.
    assert!(
        active_norm.contains("msg_send![enc, release]"),
        "encoder.rs::end_active_encoder MUST release the encoder it retained \
         in get_or_create_encoder as ACTIVE code (balance the +1; ADR-040 \
         §0.21c-track2). A missing/commented release leaks one \
         MTLComputeCommandEncoder per encoder cycle."
    );

    // (3) reset_command_buffer is leak-safe: it routes through
    // end_active_encoder (which ends + releases if non-null, no-op if null)
    // instead of a bare `active_encoder = null` that would leak the +1 retain
    // and leave a zombie encoder (review Finder B, 3e7b030). Checked on active
    // lines of the fn body.
    let reset_body = active
        .split("fn reset_command_buffer")
        .nth(1)
        .expect("encoder.rs must define reset_command_buffer (active code)");
    let reset_head: String = reset_body.lines().take(40).collect::<Vec<_>>().join("\n");
    assert!(
        reset_head.contains("self.end_active_encoder()"),
        "encoder.rs::reset_command_buffer MUST route through \
         self.end_active_encoder() as ACTIVE code (review Finder B, 3e7b030) so \
         the retained encoder is released on reset. A bare `active_encoder = \
         null` guarded only by debug_assert leaks the +1 retain in release builds."
    );
}

/// Functional SMOKE test of the fixed writer→barrier→reader path.
///
/// Writer (`tmp = a + b`) → `barrier()` → reader (`out = tmp * c`) on the
/// persistent concurrent compute encoder, ×16, asserting `out == (a+b)*c`.
///
/// HONEST SCOPE — this is NOT a fail-direction repro of the ownership bug.
/// Verified: with the retain temporarily reverted (commented out), this test
/// still PASSES. An isolated elementwise add→mul edge with a per-iteration
/// `finish()` full-sync does not open the race window — the fast producer wins
/// by timing regardless of retain, exactly as the earlier deleted timing test
/// did. So this test guards only that the happy path stays correct; it does
/// NOT discriminate the fix. The DISCRIMINATING guard is
/// `retain_root_fix_source_invariants` above (structural, catches a
/// deleted-or-commented retain) plus the production model-parity oracle
/// (`hf2q slot_aware_n8_per_slot_parity_vs_serial` under
/// `MLX_PRECOMPILED_METALLIB=0`, which exercised the real slow-producer race).
/// Kept as a cheap smoke + documentation of the path, not as the regression
/// gate — see the module docstring.
#[test]
fn writer_barrier_reader_orders_through_retained_encoder() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4096usize;
    let byte_len = n * std::mem::size_of::<f32>();

    // Deterministic, non-degenerate inputs (no zeros so a stale tmp read can't
    // accidentally match the correct product).
    let a_data: Vec<f32> = (0..n).map(|i| 1.0 + (i % 97) as f32 * 0.5).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 2.0 + (i % 53) as f32 * 0.25).collect();
    let c_data: Vec<f32> = (0..n).map(|i| 0.5 + (i % 31) as f32 * 0.125).collect();
    let expected: Vec<f32> = (0..n).map(|i| (a_data[i] + b_data[i]) * c_data[i]).collect();

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let mut c_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("c");
    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);
    c_buf.as_mut_slice::<f32>().expect("c").copy_from_slice(&c_data);

    let tmp = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("tmp");
    let out = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    let executor = GraphExecutor::new(MlxDevice::new().expect("device2"));

    // Repeat the writer→barrier→reader edge several times. Each iteration opens
    // the persistent concurrent encoder, encodes the dependent pair across a
    // drained autorelease pool, and full-syncs. Under the pre-fix unretained
    // encoder this is the shape that lost barrier ordering; under the fix every
    // iteration must produce the byte-correct product.
    for iter in 0..16 {
        {
            let mut session = executor.begin().expect("begin");

            // Writer: tmp = a + b
            session
                .elementwise_add(
                    &mut registry,
                    device.metal_device(),
                    &a_buf,
                    &b_buf,
                    &tmp,
                    n,
                    DType::F32,
                )
                .expect("graph add (writer)");

            // RAW dependency: tmp written by add, read by mul. The barrier must
            // order the reader after the writer on the concurrent encoder. This
            // is the same primitive (`memoryBarrierWithScope:Buffers` on a
            // concurrent encoder) that lost ordering pre-fix when the encoder
            // was held unretained across the per-`begin()` command-buffer cycle.
            session.barrier();

            // Reader: out = tmp * c
            session
                .elementwise_mul(
                    &mut registry,
                    device.metal_device(),
                    &tmp,
                    &c_buf,
                    &out,
                    n,
                    DType::F32,
                )
                .expect("graph mul (reader)");

            session.finish().expect("finish");
        }

        let result: Vec<f32> = out.as_slice::<f32>().expect("read out").to_vec();
        for i in 0..n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "iter {iter}, element {i}: reader observed a stale `tmp` — the \
                 barrier did not order the writer before the reader on the \
                 concurrent encoder (ordering regression; ADR-040 §0.21c-track2 \
                 retain root fix). expected={}, got={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }
}
