//! ADR-019 Phase 0a.3 — MTLEvent cost calibration microbench.
//!
//! Measures three primitives on M5 Max so AC-P4 (sync_count ≤ 12 chunk-engaged
//! prefill) can be validated against actual per-CB event/commit costs rather
//! than the iter89b "100-500 ns / commit" estimate, which was never directly
//! measured on this hardware.
//!
//! ## Three hypotheses
//!
//! - **H1** — `MTLSharedEvent` `encodeSignalEvent` + `encodeWaitForEvent`
//!   roundtrip cost per CB-pair (one signal CB + one wait CB), measured as
//!   wall delta vs. an "two empty CBs" baseline of the same shape so the
//!   subtraction isolates the event encode + cross-CB sync cost from the
//!   per-CB driver-commit floor (H3).
//!
//! - **H2** — Residency add/remove cost per-CB. Cycle a fresh `MlxBuffer`
//!   in/out of the residency set (`alloc` → `flush_pending` → `drop` →
//!   `flush_pending`), measured as wall delta vs. "two empty CBs" baseline
//!   so the result isolates the `[set addAllocation:] / [set commit] /
//!   [set removeAllocation:] / [set commit]` cost from driver-commit floor.
//!
//! - **H3** — Driver commit overhead per-CB (empty-CB
//!   `commit + wait_until_completed`). Cleanest signal — no compute encoder,
//!   no kernel dispatch, no event, no residency staging. Measured at the
//!   raw `metal-rs` surface (bypassing `MlxDevice`'s
//!   `flush_residency_pending` hook in `commit_and_wait`) so H2 is a clean
//!   superset of H3.
//!
//! ## Peer reference (semantic only, no code copied)
//!
//! `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:935-985`:
//!   `newSharedEvent` (961), `encodeSignalEvent:value:` with
//!   `atomic_fetch_add_explicit + 1` monotonic counter (945-949),
//!   `encodeWaitForEvent:value:` (952-957). One event per stage boundary,
//!   monotonic per-event counter, signal `N+1` then wait `N`.
//!
//! ## Methodology — "measure 3x, cut once"
//!
//! - Warmup: 100 iters per bench before the first measurement.
//! - Inner loop: N=200 paired ops per iter (averaged inside the iter so
//!   per-iter wall is well above `Instant::now()` resolution on Apple Silicon
//!   ~50-100 ns).
//! - Criterion-driven outer sample count (default 100 samples per bench);
//!   results.md averages mean ± std across two end-to-end bench runs to
//!   flag drift > 5 %.
//! - Trim-median (drop high+low, mean ± std of remaining 3) is applied
//!   off-line in the report; this file emits criterion's own
//!   median/mean/std, which is what the operator quotes in §0a.3.
//!
//! ## Constraints honored
//!
//! - No stub. metal-rs 0.33 fully exposes `Device::new_shared_event`,
//!   `CommandBufferRef::encode_signal_event`, `encode_wait_for_event` —
//!   verified at `/Users/robert/.cargo/registry/src/index.crates.io-…/
//!   metal-0.33.0/src/{device.rs:2059-2065, sync.rs:33-83,
//!   commandbuffer.rs:194-210}`.
//! - `SharedEventRef: Deref<Target = EventRef>` per the macro at
//!   `metal-0.33.0/src/lib.rs:148-167`, so passing `&shared_event`
//!   to the `&EventRef` parameter is a borrow-deref, not a cast.
//! - No `cd /opt/mlx-native` anywhere — all paths absolute.

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use metal::{CommandBufferRef, CommandQueue, Device, SharedEvent};

use mlx_native::{DType, MlxDevice};

/// Inner-loop pair count — every iter executes this many CB pairs (or H3
/// empty CBs). Chosen so per-iter wall on M5 Max sits in the 50 µs-2 ms
/// band, comfortably above `Instant::now()` floor (~80 ns observed).
const N_PAIRS: usize = 200;

/// Number of warmup iters before any measured iter. Lets the Metal driver,
/// command-queue ringbuffer, and any first-touch caches reach steady state
/// before criterion starts sampling.
const WARMUP_ITERS: usize = 100;

// ---------------------------------------------------------------------------
// Helpers — raw metal-rs surface, bypassing MlxDevice's wrapper logic for
// H1 and H3 so the measurements isolate the primitive being measured.
// ---------------------------------------------------------------------------

/// Submit one empty `MTLCommandBuffer` (no compute encoder, no dispatch)
/// and block until completion. This is the pure driver-commit cost — H3.
#[inline(never)]
fn submit_empty_cb(queue: &CommandQueue) {
    let cb = queue.new_command_buffer();
    cb.commit();
    cb.wait_until_completed();
}

/// Encode a signal of `value` on the given CB, but do not commit yet.
#[inline(never)]
fn encode_signal(cb: &CommandBufferRef, event: &SharedEvent, value: u64) {
    cb.encode_signal_event(event, value);
}

/// Encode a wait for `value` on the given CB, but do not commit yet.
#[inline(never)]
fn encode_wait(cb: &CommandBufferRef, event: &SharedEvent, value: u64) {
    cb.encode_wait_for_event(event, value);
}

// ---------------------------------------------------------------------------
// H1 — MTLSharedEvent signal+wait roundtrip per CB-pair.
//
// Pattern (mirrors llama.cpp ggml-metal-device.m:945-957 monotonic-counter
// semantic): for each pair i in 0..N_PAIRS,
//   * cb_a := queue.new_command_buffer
//   * cb_a.encode_signal_event(ev, i+1)
//   * cb_a.commit()                       (non-blocking)
//   * cb_b := queue.new_command_buffer
//   * cb_b.encode_wait_for_event(ev, i+1)
//   * cb_b.commit()
//   * cb_b.wait_until_completed()         (blocks until both have run;
//                                          cb_a guaranteed completed before
//                                          cb_b's wait satisfied)
//
// Per-pair wall = (signal-CB submit) + (wait-CB submit) + (cross-CB
// signal→wait sync). The H3 baseline measures (empty-CB submit) × 2 with
// the same shape (two CBs per pair, no kernels, no events). Subtracting:
//
//   per-pair-event-cost = wall(H1) − wall(2 × H3)
//
// is the deliverable H1 number.
// ---------------------------------------------------------------------------

fn bench_mtl_event_signal_wait_roundtrip(c: &mut Criterion) {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device — skipping H1");
            return;
        }
    };
    let queue = device.new_command_queue();
    let event: SharedEvent = device.new_shared_event();

    // Track the monotonic event-value counter exactly as ggml does
    // (`atomic_fetch_add_explicit + 1`). Carried across iters so each
    // signal value is fresh and `setSignaledValue` race conditions are
    // impossible.
    let mut counter: u64 = 0;

    // Warmup — drive the queue/event pair to steady state before criterion
    // starts sampling so the first measured iter is not a cold-cache outlier.
    for _ in 0..WARMUP_ITERS {
        for _ in 0..N_PAIRS {
            counter += 1;
            let cb_a = queue.new_command_buffer();
            encode_signal(cb_a, &event, counter);
            cb_a.commit();
            let cb_b = queue.new_command_buffer();
            encode_wait(cb_b, &event, counter);
            cb_b.commit();
            cb_b.wait_until_completed();
        }
    }

    c.bench_function("h1_mtl_event_signal_wait_roundtrip_per_pair", |b| {
        b.iter_custom(|n_iters| {
            let start = Instant::now();
            for _ in 0..n_iters {
                for _ in 0..N_PAIRS {
                    counter += 1;
                    let cb_a = queue.new_command_buffer();
                    encode_signal(cb_a, &event, counter);
                    cb_a.commit();
                    let cb_b = queue.new_command_buffer();
                    encode_wait(cb_b, &event, counter);
                    cb_b.commit();
                    cb_b.wait_until_completed();
                }
            }
            // Total wall divided by (n_iters × N_PAIRS) gives per-pair wall.
            // criterion treats the returned Duration as covering n_iters
            // iterations, so we report (total / N_PAIRS) — criterion's own
            // /n_iters division then yields per-pair time.
            let elapsed = start.elapsed();
            elapsed / N_PAIRS as u32
        });
    });

    black_box(counter);
}

// ---------------------------------------------------------------------------
// H2 — Residency add/remove per-CB.
//
// Pattern: alloc a fresh MlxBuffer (auto-stages add via residency_set),
// commit_and_wait an empty MlxNative encoder (which calls flush_pending →
// `[set commit]`), then drop the buffer (Drop calls remove_allocation,
// staging remove), then commit_and_wait again to flush the remove.
//
// Per-cycle wall = (alloc + add-stage + commit-encoder + flush-add-commit)
//                + (drop + remove-stage + commit-encoder + flush-remove-commit)
// vs. baseline of (commit-encoder × 2) — i.e. two MlxNative empty
// commit_and_wait calls with no buffer in flight. Subtracting:
//
//   per-cycle-residency-cost = wall(H2) − wall(2 × empty-MlxNative-CB)
//
// Buffer size is tiny (8 B = one f32) so allocation cost itself is at the
// `[device newBufferWithLength:]` floor; the residency-set staging is the
// dominant variable.
//
// REQUIRES: residency_sets_enabled() — if HF2Q_NO_RESIDENCY=1 or macOS<15,
// this bench reports the cost as "n/a (residency disabled)" via stderr and
// skips the criterion call. Phase 0a.3 acceptance documents the gap.
// ---------------------------------------------------------------------------

fn bench_residency_add_remove(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device — skipping H2");
            return;
        }
    };

    if !device.residency_sets_enabled() {
        eprintln!(
            "[bench_mtl_event_cost] H2 skipped: residency sets disabled \
             (HF2Q_NO_RESIDENCY=1 or macOS<15). \
             Document as 'deferred — residency surface inactive on this run'."
        );
        return;
    }

    // Warmup: cycle alloc/commit/drop/commit to burn in the residency set's
    // internal NSObject pools and the device's buffer-allocator freelist so
    // the first measured iter is not a cold-cache outlier. The buffer
    // pointer is allowed to escape via black_box to suppress dead-code
    // elimination of the alloc + drop pair.
    for _ in 0..WARMUP_ITERS {
        for _ in 0..N_PAIRS {
            let buf = device
                .alloc_buffer(8, DType::F32, vec![1])
                .expect("alloc 8 B should succeed");
            let mut enc1 = device.command_encoder().expect("command_encoder");
            enc1.commit_and_wait().expect("commit_and_wait #1");
            drop(buf);
            let mut enc2 = device.command_encoder().expect("command_encoder");
            enc2.commit_and_wait().expect("commit_and_wait #2");
            black_box(&enc1);
            black_box(&enc2);
        }
    }

    c.bench_function("h2_residency_add_remove_per_cycle", |b| {
        b.iter_custom(|n_iters| {
            let start = Instant::now();
            for _ in 0..n_iters {
                for _ in 0..N_PAIRS {
                    let buf = device
                        .alloc_buffer(8, DType::F32, vec![1])
                        .expect("alloc 8 B should succeed");
                    // First commit flushes the staged addAllocation.
                    let mut enc1 = device.command_encoder().expect("command_encoder");
                    enc1.commit_and_wait().expect("commit_and_wait #1");
                    // Drop stages removeAllocation.
                    drop(buf);
                    // Second commit flushes the staged removeAllocation.
                    let mut enc2 = device.command_encoder().expect("command_encoder");
                    enc2.commit_and_wait().expect("commit_and_wait #2");
                }
            }
            start.elapsed() / N_PAIRS as u32
        });
    });

    // Companion baseline so the report can subtract two MlxNative empty
    // commit_and_wait calls from H2's per-cycle wall, isolating the
    // residency-set staging cost from the H3 floor (H3 is at the raw
    // metal-rs surface; this baseline is at the MlxNative wrapper surface
    // and therefore includes one no-op flush_pending per CB).
    c.bench_function("h2_baseline_two_mlx_native_empty_cb_per_cycle", |b| {
        b.iter_custom(|n_iters| {
            let start = Instant::now();
            for _ in 0..n_iters {
                for _ in 0..N_PAIRS {
                    let mut enc1 = device.command_encoder().expect("command_encoder");
                    enc1.commit_and_wait().expect("commit_and_wait #1");
                    let mut enc2 = device.command_encoder().expect("command_encoder");
                    enc2.commit_and_wait().expect("commit_and_wait #2");
                }
            }
            start.elapsed() / N_PAIRS as u32
        });
    });
}

// ---------------------------------------------------------------------------
// H3 — Empty-CB driver commit overhead. Raw metal-rs surface; no MlxNative
// wrapper — this is the cleanest possible "per-CB driver cost" number,
// the floor below which D3 cannot push wall-time per CB.
// ---------------------------------------------------------------------------

fn bench_empty_cb_commit_overhead(c: &mut Criterion) {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device — skipping H3");
            return;
        }
    };
    let queue = device.new_command_queue();

    for _ in 0..WARMUP_ITERS {
        for _ in 0..N_PAIRS {
            submit_empty_cb(&queue);
        }
    }

    c.bench_function("h3_empty_cb_commit_per_cb", |b| {
        b.iter_custom(|n_iters| {
            let start = Instant::now();
            for _ in 0..n_iters {
                for _ in 0..N_PAIRS {
                    submit_empty_cb(&queue);
                }
            }
            start.elapsed() / N_PAIRS as u32
        });
    });

    // Companion: two empty CBs back-to-back, single wait at the end. This
    // is the H1 baseline — H1 subtracts this from its per-pair wall.
    c.bench_function("h3_two_empty_cb_pipelined_per_pair", |b| {
        b.iter_custom(|n_iters| {
            let start = Instant::now();
            for _ in 0..n_iters {
                for _ in 0..N_PAIRS {
                    let cb_a = queue.new_command_buffer();
                    cb_a.commit();
                    let cb_b = queue.new_command_buffer();
                    cb_b.commit();
                    cb_b.wait_until_completed();
                }
            }
            start.elapsed() / N_PAIRS as u32
        });
    });

    // Suppress an unused-import warning if BatchSize is not referenced
    // elsewhere — kept available so future iters can switch to
    // `iter_batched_ref` if iter_custom is too noisy on a given chip.
    let _ = BatchSize::SmallInput;
}

criterion_group!(
    benches,
    bench_empty_cb_commit_overhead,
    bench_mtl_event_signal_wait_roundtrip,
    bench_residency_add_remove,
);
criterion_main!(benches);
