//! [`EncoderSession`] — D3 Per-Stage Fence encoder abstraction (ADR-019 Phase 0b).
//!
//! `EncoderSession` lifts [`CommandEncoder`] into a session-aware shell that
//! carries semantic *stage* metadata across the lifetime of one logical
//! transformer stage (e.g. `"layer.full_attn.stage1"`). Phase 0b-A delivers
//! the bare struct + lifecycle methods; Phase 0b-B will add `MTLSharedEvent`
//! stage-fence semantics; Phase 0b-C will broaden label propagation for
//! xctrace MST attribution. Production callers MUST stay on
//! [`crate::CommandEncoder`] in iter89e2-A — this struct is feature-flagged
//! behind `HF2Q_ENCODER_SESSION=1` (default OFF) and is constructed only via
//! [`MlxDevice::encoder_session`](crate::MlxDevice::encoder_session).
//!
//! # Lifecycle (iter89e2-A — single-stage)
//!
//! ```text
//!                  MlxDevice::encoder_session()
//!                                |
//!                                v
//!                          +-----------+
//!                          | Empty     |  no CB or encoder open yet
//!                          +-----------+
//!                                |
//!                          first dispatch
//!                          (via inner CommandEncoder)
//!                                |
//!                                v
//!                          +-----------+
//!                          | Encoding  |  CB open, persistent compute encoder open
//!                          +-----------+
//!                                |
//!                       commit_stage() (non-blocking)
//!                       commit_and_wait() (blocking)
//!                                |
//!                                v
//!                          +-----------+
//!                          | Drained   |  CB submitted; session is one-shot
//!                          +-----------+
//!                                |
//!                              Drop
//! ```
//!
//! Multi-stage chaining (`reset_for_next_stage` after a non-blocking
//! `commit_stage()` opens a fresh CB on the same queue) requires
//! `MTLSharedEvent` ordering between stages and is therefore Phase 0b-B
//! scope. In iter89e2-A each `EncoderSession` represents exactly one
//! command buffer.
//!
//! # Risk register fence preservation (F1-F12 from ADR-019)
//!
//! - **F1 — persistent compute encoder per CB**: ADOPTED unchanged.
//!   `EncoderSession` borrows `&mut CommandEncoder` via [`Self::encoder`];
//!   every dispatch reuses the same lazy-opened encoder.
//! - **F2 — iter58b residency-rescission**: PRESERVED. `commit_stage` and
//!   `commit_and_wait` delegate to the inner encoder, which calls
//!   `flush_residency_pending()` at every commit boundary
//!   (`encoder.rs:1842, 2004`). `EncoderSession` does NOT widen the F2
//!   exposure window in iter89e2-A: a session holds exactly one CB, and
//!   `Drop` on a non-committed session ends the encoder cleanly
//!   (`CommandEncoder::Drop` at `encoder.rs:2057-2063`); no residency-remove
//!   is staged for buffers that were never bound. Phase 0b-B's longer-window
//!   stage CBs are the F2 expansion vector — in iter89e2-A, F2 is invariant.
//! - **F11 — zero-init alloc_buffer**: INVARIANT. `EncoderSession` does
//!   not allocate buffers.
//! - **F12 — `HF2Q_FORCE_SERIAL_DISPATCH` falsification probe**: PRESERVED.
//!   The `MTLDispatchType::Serial` selection lives in
//!   `CommandEncoder::get_or_create_encoder` (`encoder.rs:840-851`); since
//!   `EncoderSession` delegates dispatch through that same encoder, the
//!   probe still fires.
//! - F3, F4, F5, F6, F7, F8, F9, F10 are out of scope for iter89e2-A
//!   (forward-path phases 1-4 territory) — `EncoderSession` is purely
//!   structural and does not touch any forward path.
//!
//! # Feature gate
//!
//! [`MlxDevice::encoder_session`] returns `Ok(None)` when the
//! `HF2Q_ENCODER_SESSION` env var is unset (default). When set to `"1"`
//! it returns `Ok(Some(EncoderSession))`. Production code paths in hf2q
//! consume `device.command_encoder()` (returns plain `CommandEncoder`) so
//! the gate is a no-op in default builds — zero behavior change.

use crate::encoder::CommandEncoder;
use crate::error::Result;
use crate::residency::ResidencySet;

/// Cached `HF2Q_ENCODER_SESSION` decision.
///
/// Identical pattern to `auto_barrier_enabled` / `unretained_refs_enabled`
/// in `encoder.rs` — `OnceLock` so the env-read happens exactly once per
/// process, and the per-call cost is a single atomic load. Declared at
/// module scope so the gate is observable from both
/// [`EncoderSession::env_enabled`] (the public introspection helper) and
/// [`MlxDevice::encoder_session`] (the factory site).
fn encoder_session_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("HF2Q_ENCODER_SESSION")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Session-level wrapper around a [`CommandEncoder`] for one transformer stage.
///
/// See module docs for lifecycle and fence preservation. Iter89e2-A scope:
/// bare struct + `new`, `begin_stage`, `commit_stage`, `commit_and_wait`.
/// `MTLSharedEvent` fence + `reset_for_next_stage` are Phase 0b-B.
///
/// # Thread safety
///
/// `EncoderSession` is `Send` because [`CommandEncoder`] is `Send` (the
/// existing unsafe impl at `encoder.rs:613-619`) and `String` is `Send`.
/// It is NOT `Sync` — exclusive ownership during dispatch encoding is
/// the same contract as the inner [`CommandEncoder`].
pub struct EncoderSession {
    /// Inner command encoder. Carries `cmd_buf`, the persistent
    /// `active_encoder`, the residency-set flush hook, capture-mode IR,
    /// the auto-barrier `MemRanges` tracker, the iter63 sample buffer,
    /// and the iter16 `last_label` history. All dispatch operations
    /// flow through here.
    ///
    /// INVARIANT: `inner` is in a consistent state at every public API
    /// boundary. Drops cleanly via `CommandEncoder::Drop` which calls
    /// `end_active_encoder()` (Metal-asserts on a CB dropped with an
    /// unended encoder).
    inner: CommandEncoder,

    /// Human-readable stage label for xctrace MST attribution.
    ///
    /// Set by [`Self::begin_stage`]. Empty by default. When non-empty,
    /// [`Self::commit_stage`] and [`Self::commit_and_wait`] delegate
    /// to the inner encoder's `commit_labeled` / `commit_and_wait_labeled`
    /// path, which propagates the label to `MTLCommandBuffer.label` and
    /// `MTLComputeCommandEncoder.label` via `apply_labels` at
    /// `encoder.rs:1968-1986`.
    ///
    /// Phase 0b-C will broaden this surface (per-substage labels,
    /// MST-friendly hierarchical naming). Iter89e2-A keeps it as a single
    /// per-session string.
    stage_label: String,

    /// Latch flipped to `true` after the first `commit_stage` /
    /// `commit_and_wait` call.
    ///
    /// Used to enforce the iter89e2-A one-shot invariant: a single
    /// `EncoderSession` represents exactly one CB. Calling `commit_*`
    /// twice without an intervening `reset_for_next_stage` (which is
    /// Phase 0b-B) is a logic error — we surface it as a no-op rather
    /// than a panic so the session remains drop-safe.
    drained: bool,
}

// SAFETY: `EncoderSession` is `Send` provided that:
// 1. `CommandEncoder` is `Send` (verified via the existing unsafe impl
//    at encoder.rs:613-619 — Apple documents that command buffers and
//    encoders may be encoded from any thread provided exclusive
//    ownership).
// 2. `String` is `Send`.
// 3. `bool` is `Send`.
// All three hold. `EncoderSession` does NOT add any non-Send fields in
// iter89e2-A. Phase 0b-B will add `Option<metal::SharedEvent>`; that
// addition will need to re-validate this invariant against
// `metal::SharedEvent`'s `Send`/`Sync` story.
unsafe impl Send for EncoderSession {}

impl EncoderSession {
    /// Construct a new session over a fresh `CommandEncoder`.
    ///
    /// Returns `Err` if the underlying `CommandEncoder::new_with_residency`
    /// fails (currently impossible — that constructor is infallible after
    /// metal-rs 0.33's `new_command_buffer` returned a valid handle, but
    /// the `Result` is preserved for future-proofing against driver-side
    /// allocation failures).
    ///
    /// # Crate-internal
    ///
    /// `pub(crate)` because the public construction surface is
    /// [`MlxDevice::encoder_session`](crate::MlxDevice::encoder_session),
    /// which threads the env-gate. Direct construction from outside
    /// `mlx-native` would bypass the `HF2Q_ENCODER_SESSION` flag, which
    /// is the wrong layering.
    pub(crate) fn new(
        queue: &metal::CommandQueue,
        residency_set: Option<ResidencySet>,
    ) -> Result<Self> {
        Ok(Self {
            inner: CommandEncoder::new_with_residency(queue, residency_set)?,
            stage_label: String::new(),
            drained: false,
        })
    }

    /// Whether `HF2Q_ENCODER_SESSION=1` is set in the process environment.
    ///
    /// Public introspection helper for hf2q-side dispatch wrappers that
    /// need to choose between the legacy `command_encoder()` path and the
    /// new `encoder_session()` path. Cached on first read via `OnceLock`
    /// so the per-call cost is a single atomic load.
    #[inline]
    pub fn env_enabled() -> bool {
        encoder_session_enabled()
    }

    /// Set the semantic stage label.
    ///
    /// The label propagates to `MTLCommandBuffer.label` and (when an
    /// encoder is active) `MTLComputeCommandEncoder.label` at the next
    /// `commit_stage` or `commit_and_wait` call, enabling xctrace MST
    /// attribution per ADR-015 iter16. Calling `begin_stage` does NOT
    /// itself touch any Metal object — it only stores the string.
    ///
    /// Idempotent: calling `begin_stage` multiple times before commit
    /// overwrites the previous label with the latest value, matching
    /// the existing `apply_labels` semantic at `encoder.rs:1980-1985`
    /// (the `last_label` field is overwritten on every labeled commit).
    pub fn begin_stage(&mut self, label: &str) {
        self.stage_label.clear();
        self.stage_label.push_str(label);
    }

    /// Borrow the inner [`CommandEncoder`] for dispatch encoding.
    ///
    /// All dispatch APIs (`encode`, `encode_threadgroups`,
    /// `encode_with_args`, `dispatch_tracked_*`, `memory_barrier`,
    /// `start_capture` / `take_capture`, etc.) live on
    /// [`CommandEncoder`]; `EncoderSession` adds a stage-aware commit
    /// surface on top of them. Use this accessor inside the dispatch
    /// loop, then call `EncoderSession::commit_stage` /
    /// `commit_and_wait` at the stage boundary.
    ///
    /// # Caller contract
    ///
    /// Do NOT call `inner.commit*` methods directly through this
    /// borrow. Use [`Self::commit_stage`] / [`Self::commit_and_wait`]
    /// so the stage label propagates correctly. Calling the inner
    /// commit bypasses the session's drained-latch and label, which
    /// is a logic error — but is not unsafe (no UB risk).
    #[inline]
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        &mut self.inner
    }

    /// Commit the stage's command buffer non-blocking.
    ///
    /// Delegates to `CommandEncoder::commit_labeled` (when a label is
    /// set) or `CommandEncoder::commit` (when not). Both end the
    /// persistent compute encoder, flush the residency-set pending
    /// staging (`flush_residency_pending` at `encoder.rs:2004`), and
    /// hand the CB to the GPU without blocking the CPU.
    ///
    /// The session enters the `Drained` state. Calling `commit_stage`
    /// or `commit_and_wait` again is a no-op in iter89e2-A (one-shot
    /// session); Phase 0b-B will replace this latch with a
    /// `reset_for_next_stage` call that opens a fresh CB on the same
    /// queue.
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` unconditionally — `CommandEncoder::commit` and
    /// `CommandEncoder::commit_labeled` are infallible (they hand the
    /// CB to Metal without waiting for completion; errors surface only
    /// at `wait_until_completed`). The `Result` is preserved for
    /// symmetry with [`Self::commit_and_wait`] and for future-proofing.
    pub fn commit_stage(&mut self) -> Result<()> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        if self.stage_label.is_empty() {
            self.inner.commit();
        } else {
            // Take a snapshot of the label so we don't borrow `self`
            // both immutably (for the label) and mutably (for inner)
            // — clones a small String, fine for stage-boundary cost.
            let label = self.stage_label.clone();
            self.inner.commit_labeled(&label);
        }
        Ok(())
    }

    /// Commit the stage's command buffer and block until GPU completion.
    ///
    /// Delegates to `CommandEncoder::commit_and_wait_labeled` (when a
    /// label is set) or `CommandEncoder::commit_and_wait` (when not).
    /// Required at K-batch boundaries (F7) and at output-head CPU reads
    /// (F6). Increments `SYNC_COUNT` exactly once per call (matches
    /// `encoder.rs:1845`).
    ///
    /// The session enters the `Drained` state. See [`Self::commit_stage`]
    /// for the one-shot rationale.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an
    /// error after wait — propagated from `CommandEncoder`.
    pub fn commit_and_wait(&mut self) -> Result<()> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        if self.stage_label.is_empty() {
            self.inner.commit_and_wait()
        } else {
            let label = self.stage_label.clone();
            self.inner.commit_and_wait_labeled(&label)
        }
    }

    /// Whether the session has been committed (either path).
    ///
    /// Test-and-introspection helper. Production code should not need
    /// this — call `commit_*` exactly once per session.
    #[inline]
    pub fn is_drained(&self) -> bool {
        self.drained
    }

    /// Borrow the underlying Metal command buffer.
    ///
    /// Mirrors [`CommandEncoder::metal_command_buffer`]. Used by the
    /// label-propagation test to read back `MTLCommandBuffer.label()`.
    #[inline]
    pub fn metal_command_buffer(&self) -> &metal::CommandBuffer {
        self.inner.metal_command_buffer()
    }
}

impl Drop for EncoderSession {
    /// Drain the inner [`CommandEncoder`] safely on drop.
    ///
    /// # F2 residency-rescission preservation (load-bearing)
    ///
    /// In iter89e2-A a session represents exactly one CB. Three Drop
    /// scenarios:
    ///
    /// 1. **Drained** — `commit_stage` / `commit_and_wait` already ran.
    ///    `inner.flush_residency_pending()` was already called; the GPU
    ///    has the CB (and may already have completed it under
    ///    `commit_and_wait`). `CommandEncoder::Drop` runs and calls
    ///    `end_active_encoder()`, which is a no-op because `commit*`
    ///    already ended the encoder. Safe.
    ///
    /// 2. **Encoding (uncommitted)** — caller created the session,
    ///    optionally encoded dispatches, then dropped without calling
    ///    `commit_*`. `CommandEncoder::Drop` ends the active compute
    ///    encoder cleanly (`encoder.rs:2057-2063`). The `cmd_buf` is
    ///    dropped without ever being committed — Metal discards the
    ///    encoded work. **No residency-remove is staged** because no
    ///    buffers were registered as freed during this session (the
    ///    F2 race requires a buffer drop staging a remove that a later
    ///    `flush_pending` commits before the in-flight CB finishes; here
    ///    no commit ever happens). The residency-set's pending state
    ///    persists into the next encoder; correct.
    ///
    /// 3. **Empty** — no dispatches encoded. `active_encoder` is null;
    ///    `CommandEncoder::Drop`'s `end_active_encoder` is a no-op.
    ///    Safe.
    ///
    /// We deliberately do NOT call `wait_until_completed` here for the
    /// committed-but-not-waited case (scenario 1 with `commit_stage`
    /// rather than `commit_and_wait`). Under retained-refs mode (default
    /// — `MLX_UNRETAINED_REFS=0`), the in-flight CB holds ARC retains on
    /// every bound buffer, so the GPU completes safely after the session
    /// drops. Under `MLX_UNRETAINED_REFS=1` (NOT enabled in Phase 0b),
    /// the caller-owned-arena contract is the only structural mitigation
    /// — same as the existing async-`commit()` path at
    /// `encoder.rs:2014-2022`.
    ///
    /// In short: `Drop` does no extra work; the inner `CommandEncoder`'s
    /// own Drop is the entire safety story.
    fn drop(&mut self) {
        // The actual end-encoder call lives in `CommandEncoder::Drop`,
        // which fires automatically when `self.inner` goes out of scope
        // here.  No additional work needed — see this docstring's case
        // analysis above for the F2 fence preservation argument.
    }
}
