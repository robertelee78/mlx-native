//! [`EncoderSession`] — D3 Per-Stage Fence encoder abstraction (ADR-019 Phase 0b).
//!
//! `EncoderSession` lifts [`CommandEncoder`] into a session-aware shell that
//! carries semantic *stage* metadata across the lifetime of one or more
//! logical transformer stages (e.g. `"layer.full_attn.stage1"` →
//! `"layer.full_attn.stage2"`). Phase 0b-A delivered the bare struct +
//! single-stage lifecycle methods. Phase 0b-B (this file) adds the
//! [`MTLSharedEvent`](metal::SharedEvent) inter-CB ordering primitives D3
//! needs:
//!
//! - [`Self::fence_stage`] — encode signal-event(N+1) on the current CB,
//!   commit non-blocking, increment the per-session monotonic counter.
//! - [`Self::reset_for_next_stage`] — open a fresh CB on the same queue
//!   and (when a fence is active) encode wait-event(N) on the new CB so
//!   its GPU work blocks until the prior fenced CB completes.
//! - [`Self::add_to_residency_set`] / [`Self::remove_from_residency_set`]
//!   — public delegation surface for the single residency set owned by
//!   [`MlxDevice`](crate::MlxDevice). EncoderSession does NOT own a
//!   separate set; it routes calls through the Arc clone the inner
//!   [`CommandEncoder`] already holds.
//!
//! Phase 0b-C will broaden label propagation (per-substage labels +
//! xctrace MST round-trip). iter89e2-B leaves Phase 0b-A's existing
//! per-session label semantics intact.
//!
//! Production callers MUST stay on [`crate::CommandEncoder`] until Phase
//! 2 (FA-path D3 stage migration) wires `forward_gpu.rs` to consume
//! `EncoderSession`. The struct is feature-flagged behind
//! `HF2Q_ENCODER_SESSION=1` (default OFF) and is constructed only via
//! [`MlxDevice::encoder_session`](crate::MlxDevice::encoder_session).
//!
//! # Lifecycle (iter89e2-B — multi-stage chaining)
//!
//! ```text
//!                   MlxDevice::encoder_session()
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
//!                            |       |
//!                            |       +---fence_stage(label)----+
//!                            |                                 |
//!                            |                                 v
//!                            |                          +-----------+
//!                            |                          | Fenced    |  signal encoded; CB submitted
//!                            |                          +-----------+
//!                            |                                 |
//!                  commit_stage()                  reset_for_next_stage()
//!                  commit_and_wait()                            |
//!                            |                                 v
//!                            v                          (loop back to Encoding
//!                      +-----------+                    on next dispatch — wait
//!                      | Drained   |                    is encoded automatically
//!                      +-----------+                    on the new CB)
//!                            |
//!                          Drop
//! ```
//!
//! `fence_stage` collapses the design-doc's separate Encoding→Fenced→
//! Committed transitions into a single "submit-with-fence" call: the
//! signal is encoded on the current CB at `event_value+1`, the encoder
//! is ended, the CB is committed non-blocking, and the per-session
//! monotonic counter is incremented. The session is `drained` until
//! [`Self::reset_for_next_stage`] rotates the inner [`CommandEncoder`]'s
//! command buffer to a fresh CB on the same queue and (when the event is
//! present) encodes the matching wait at `event_value` on the new CB.
//!
//! # Risk register fence preservation (F1-F12 from ADR-019)
//!
//! - **F1 — persistent compute encoder per CB**: ADOPTED unchanged.
//!   `EncoderSession` borrows `&mut CommandEncoder` via [`Self::encoder`];
//!   every dispatch reuses the same lazy-opened encoder per CB. Each
//!   stage CB still has exactly one persistent compute encoder.
//! - **F2 — iter58b residency-rescission**: PRESERVED. `commit_stage`,
//!   `commit_and_wait`, and `fence_stage` all delegate to the inner
//!   encoder, which calls `flush_residency_pending()` at every commit
//!   boundary (`encoder.rs:1842, 2004`). `reset_for_next_stage` does NOT
//!   re-flush — staged add/remove operations between stages flush at
//!   the next commit on the new CB. The single residency set is owned
//!   by [`MlxDevice`](crate::MlxDevice) (single-set invariant per
//!   ADR-019:467). Multi-stage chaining DOES widen the in-flight CB
//!   window — dropping a buffer between stage 1's `fence_stage` and
//!   stage 2's `commit_*` stages a remove-allocation that flushes at
//!   stage 2's commit, while stage 1's CB may still be GPU-pipelined.
//!   Under retained-refs (default), the prior CB's ARC retains keep the
//!   underlying Metal buffer alive across the residency-set demotion;
//!   the GPU completes safely. Under `MLX_UNRETAINED_REFS=1` (NOT
//!   enabled in Phase 0b), caller-owned arenas remain the only
//!   structural mitigation — same contract as the existing async-commit
//!   path. The adversarial F2 test (see
//!   `/opt/mlx-native/tests/encoder_session_multistage.rs`) explicitly
//!   exercises this window.
//! - **F11 — zero-init alloc_buffer**: INVARIANT. `EncoderSession` does
//!   not allocate buffers; the zero-init contract on
//!   `MlxDevice::alloc_buffer` is unchanged.
//! - **F12 — `HF2Q_FORCE_SERIAL_DISPATCH` falsification probe**: PRESERVED.
//!   The probe lives in `CommandEncoder::get_or_create_encoder` and is
//!   re-read every time a fresh CB lazily opens its compute encoder
//!   (every `reset_for_next_stage` rotation). Both pre- and post-fence
//!   CBs honor the env var.
//! - F3, F4, F5, F6, F7, F8, F9, F10 are out of scope for iter89e2-B
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

use crate::buffer::MlxBuffer;
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

/// Session-level wrapper around a [`CommandEncoder`] for one or more
/// logical transformer stages.
///
/// See module docs for lifecycle and fence preservation. iter89e2-B scope:
/// multi-stage chaining via [`MTLSharedEvent`](metal::SharedEvent), residency
/// delegation surface, and the matching test cohort. Phase 0b-C will
/// broaden label propagation; Phase 2+ will wire this struct into the
/// production forward path.
///
/// # Thread safety
///
/// `EncoderSession` is `Send` because [`CommandEncoder`] is `Send` (the
/// existing unsafe impl at `encoder.rs:613-619`), `String`/`u64`/`bool`
/// are `Send`, [`metal::Device`] is `Send + Sync` (foreign_obj_type! at
/// metal-rs 0.33 lib.rs:179), and [`metal::SharedEvent`] is `Send + Sync`
/// for the same reason. It is NOT `Sync` — exclusive ownership during
/// dispatch encoding is the same contract as the inner [`CommandEncoder`].
pub struct EncoderSession {
    /// Inner command encoder. Carries `cmd_buf`, the persistent
    /// `active_encoder`, the `queue` clone (read by
    /// [`CommandEncoder::reset_command_buffer`]), the residency-set
    /// flush hook, capture-mode IR, the auto-barrier `MemRanges`
    /// tracker, the iter63 sample buffer, and the iter16 `last_label`
    /// history. All dispatch operations flow through here.
    ///
    /// INVARIANT: `inner` is in a consistent state at every public API
    /// boundary. Drops cleanly via `CommandEncoder::Drop` which calls
    /// `end_active_encoder()` (Metal-asserts on a CB dropped with an
    /// unended encoder).
    inner: CommandEncoder,

    /// Owned clone of the originating [`metal::Device`].
    ///
    /// iter89e2-B: held so [`Self::fence_stage`] can lazily allocate an
    /// [`metal::SharedEvent`] on first call without threading a
    /// `&MlxDevice` through every call site. metal-rs 0.33's `Device`
    /// is `Send + Sync` (foreign_obj_type! lib.rs:179), so adding this
    /// field preserves the existing unsafe `Send` impl on
    /// [`EncoderSession`] declared below.
    device: metal::Device,

    /// Lazily-allocated [`MTLSharedEvent`](metal::SharedEvent) backing
    /// the per-session monotonic stage fence.
    ///
    /// `None` until the first [`Self::fence_stage`] call. Once
    /// allocated, the same event is reused across every fence in this
    /// session — the value half of the (event, value) pair carries the
    /// monotonic identity. Cost is one ObjC alloc + autorelease per
    /// session lifetime; subsequent fences reuse the same event.
    event: Option<metal::SharedEvent>,

    /// Per-session monotonic fence counter.
    ///
    /// Mirrors `ggml_metal_event::value` at
    /// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:941`.
    /// [`Self::fence_stage`] post-increments (signal = current+1, then
    /// store current+1); [`Self::reset_for_next_stage`] reads (wait =
    /// current). Starts at 0; bumps to 1 on first fence; CB N waits on
    /// value N to gate after CB N's signal lands.
    event_value: u64,

    /// Human-readable stage label for xctrace MST attribution.
    ///
    /// Set by [`Self::begin_stage`] and by the `Some` arm of
    /// [`Self::fence_stage`]'s `label` parameter. Empty by default.
    /// When non-empty, [`Self::commit_stage`], [`Self::commit_and_wait`],
    /// and [`Self::fence_stage`] all delegate to the inner encoder's
    /// `commit_labeled` / `commit_and_wait_labeled` path, which
    /// propagates the label to `MTLCommandBuffer.label` and
    /// `MTLComputeCommandEncoder.label` via `apply_labels` at
    /// `encoder.rs:1968-1986`.
    ///
    /// Cleared by [`Self::reset_for_next_stage`] so each chained stage
    /// starts with a fresh label slot — the caller calls `begin_stage`
    /// (or passes `Some(label)` to the next `fence_stage`) per stage.
    stage_label: String,

    /// Latch flipped to `true` after a `commit_stage` / `commit_and_wait`
    /// / `fence_stage` call.
    ///
    /// Used to enforce the one-CB-per-state contract: a `EncoderSession`
    /// in the `Drained` (or `Fenced`) state must call
    /// [`Self::reset_for_next_stage`] before further dispatches encode
    /// onto a new CB. Calling `commit_*` twice without an intervening
    /// reset is a logic error — we surface it as a no-op rather than a
    /// panic so the session remains drop-safe.
    drained: bool,

    /// Whether the most recent commit was a [`Self::fence_stage`] call.
    ///
    /// When `true`, [`Self::reset_for_next_stage`] encodes an
    /// `encodeWaitForEvent` on the new CB at `event_value`. Cleared by
    /// `reset_for_next_stage` so a subsequent `commit_stage` (no fence)
    /// does not spuriously emit a wait on the next reset.
    fence_pending: bool,

    /// Per-session count of `encodeWaitForEvent` calls actually emitted
    /// inside [`Self::reset_for_next_stage`].
    ///
    /// Symmetric counterpart to `event_value` (the signal-side high-water
    /// mark) — `wait_count` is the wait-side scoreboard. Bumped exactly
    /// once each time `reset_for_next_stage` finds `fence_pending == true`
    /// and routes through `inner.encode_wait_for_event`. Read-only via
    /// [`Self::wait_count`]; never mutated by control flow (introspection
    /// only — does NOT widen F1/F2/F11/F12 windows).
    ///
    /// iter90b §2 H1b proof: the multi-stage chain test asserts this
    /// equals `(num_stages - 1)` for an N-stage chain (one wait per
    /// reset; the first stage's CB never had a prior signal to wait on).
    wait_count: u64,

    /// Value of the most recent `encodeWaitForEvent` actually emitted
    /// inside [`Self::reset_for_next_stage`].
    ///
    /// Mirrors the relationship between `event_value` (signal-side) and
    /// the value passed to `inner.encode_wait_for_event`. Starts at 0
    /// (no wait yet emitted); each successful wait sets this to the
    /// `value` argument. Read-only via [`Self::wait_value`]; pure
    /// introspection (does NOT widen F1/F2/F11/F12 windows).
    ///
    /// iter90b §2 H1b proof: after a `fence_stage(N)` followed by
    /// `reset_for_next_stage()`, this MUST equal N (the wait-side
    /// matches the signal we just signaled).
    last_wait_value: u64,
}

// SAFETY: `EncoderSession` is `Send` provided that:
// 1. `CommandEncoder` is `Send` (existing unsafe impl at encoder.rs:606,
//    Apple documents that command buffers / encoders may be encoded
//    from any thread provided exclusive ownership).
// 2. `metal::Device` is `Send + Sync` via foreign_obj_type!
//    (metal-0.33.0/src/lib.rs:179).
// 3. `metal::SharedEvent` is `Send + Sync` via foreign_obj_type!
//    (same site — the macro emits `unsafe type ...: Sync + Send`
//    for every type, including SharedEvent in sync.rs:36-40).
// 4. `String`, `u64`, `bool` are `Send`.
// All five hold. `EncoderSession` does NOT add any non-Send fields in
// iter89e2-B beyond `metal::Device` + `Option<metal::SharedEvent>` +
// `u64` + `bool`, all already validated.
unsafe impl Send for EncoderSession {}

impl EncoderSession {
    /// Construct a new session over a fresh `CommandEncoder`.
    ///
    /// Returns `Err` if the underlying `CommandEncoder::new_with_residency`
    /// fails (currently impossible past metal-rs 0.33's
    /// `new_command_buffer`, but the `Result` is preserved for
    /// future-proofing against driver-side allocation failures).
    ///
    /// # Crate-internal
    ///
    /// `pub(crate)` because the public construction surface is
    /// [`MlxDevice::encoder_session`](crate::MlxDevice::encoder_session),
    /// which threads the env-gate. Direct construction from outside
    /// `mlx-native` would bypass the `HF2Q_ENCODER_SESSION` flag, which
    /// is the wrong layering.
    pub(crate) fn new(
        device: &metal::DeviceRef,
        queue: &metal::CommandQueue,
        residency_set: Option<ResidencySet>,
    ) -> Result<Self> {
        Ok(Self {
            inner: CommandEncoder::new_with_residency(queue, residency_set)?,
            device: device.to_owned(),
            event: None,
            event_value: 0,
            stage_label: String::new(),
            drained: false,
            fence_pending: false,
            wait_count: 0,
            last_wait_value: 0,
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
    /// `commit_stage` / `commit_and_wait` / `fence_stage` call, enabling
    /// xctrace MST attribution per ADR-015 iter16. Calling `begin_stage`
    /// does NOT itself touch any Metal object — it only stores the
    /// string.
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
    /// loop, then call one of [`Self::commit_stage`] /
    /// [`Self::commit_and_wait`] / [`Self::fence_stage`] at the stage
    /// boundary.
    ///
    /// # Caller contract
    ///
    /// Do NOT call `inner.commit*` methods directly through this
    /// borrow. Use the session's commit surface so the stage label
    /// propagates and the drained-latch / fence state stay consistent.
    /// Calling the inner commit bypasses these — it is not unsafe (no
    /// UB risk) but it makes the session state inconsistent with what
    /// it has actually committed.
    #[inline]
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        &mut self.inner
    }

    /// Commit the stage's command buffer non-blocking (no fence).
    ///
    /// Delegates to `CommandEncoder::commit_labeled` (when a label is
    /// set) or `CommandEncoder::commit` (when not). Both end the
    /// persistent compute encoder, flush the residency-set pending
    /// staging (`flush_residency_pending` at `encoder.rs:2004`), and
    /// hand the CB to the GPU without blocking the CPU.
    ///
    /// The session enters the `Drained` state. To chain into another
    /// stage on the same session, call [`Self::reset_for_next_stage`]
    /// — that opens a fresh CB and (if a fence was pending from a prior
    /// `fence_stage`) encodes the matching wait. After
    /// `commit_stage` (no fence), `reset_for_next_stage` does NOT emit
    /// a wait — the CBs are merely sequenced by the Metal queue's FIFO
    /// dispatch order.
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
        self.fence_pending = false;
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
    /// The session enters the `Drained` state with NO fence pending —
    /// blocking commit fully drains the GPU, so the next stage (after
    /// [`Self::reset_for_next_stage`]) needs no wait-event.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an
    /// error after wait — propagated from `CommandEncoder`.
    ///
    /// ADR-015 iter94 Task #2 — fail-loud contract.  iter93 final-report
    /// §"Root-cause hypothesis" point 5 noted that under
    /// `MLX_UNRETAINED_REFS=1` + `HF2Q_ENCODER_SESSION=1` + `K>1`, the
    /// session appeared to silently absorb a `MTLCommandBufferStatus::
    /// Error` and produce deterministic-but-wrong tokens.  By code
    /// reading, the tail-expression `self.inner.commit_and_wait()`
    /// already returns the inner error (commit_and_wait at
    /// encoder.rs:1852 explicitly matches on `cmd_buf.status()`).  This
    /// re-shape converts the implicit propagation into an explicit `?`
    /// chain so future maintainers cannot accidentally swallow the
    /// error by inserting a `let _ = inner.commit_and_wait();` or
    /// adding fall-through logic between the inner call and the
    /// function return.  Latched `drained = true` happens BEFORE the
    /// inner call so a panicking unwind through Drop sees the same
    /// drained-state contract.
    pub fn commit_and_wait(&mut self) -> Result<()> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        self.fence_pending = false;
        let result = if self.stage_label.is_empty() {
            self.inner.commit_and_wait()
        } else {
            let label = self.stage_label.clone();
            self.inner.commit_and_wait_labeled(&label)
        };
        // Explicit `?`-style propagation: any `Err` from the inner
        // commit_and_wait MUST surface to the caller.  This is the
        // iter94 Task #2 fail-loud guarantee — silent absorption here
        // would replicate the iter93 §"Root-cause hypothesis" point 5
        // failure mode (deterministic-but-wrong outputs at the triple
        // combo).  The extra `?` is a no-op codegen-wise vs the prior
        // tail-expression form but documents intent and is unit-tested
        // by `test_commit_and_wait_propagates_inner_cb_error`.
        result?;
        Ok(())
    }

    /// Encode a stage-fence signal on the current CB and commit non-blocking.
    ///
    /// This is the D3 multi-stage building block: the prior stage's
    /// final CB-level op is `encodeSignalEvent:value:value+1`, where
    /// `value+1` is then both stored in `event_value` (so the next
    /// stage's `encodeWaitForEvent:value:` blocks on it) and committed.
    /// The session enters the `Fenced` (drained-with-fence-pending)
    /// state; [`Self::reset_for_next_stage`] rotates the inner CB and
    /// emits the matching wait.
    ///
    /// # Lazy event allocation
    ///
    /// On the first call, allocates the per-session
    /// [`MTLSharedEvent`](metal::SharedEvent) via
    /// [`metal::DeviceRef::new_shared_event`]
    /// (`/Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/metal-0.33.0/src/device.rs:2063`).
    /// Subsequent calls reuse the same event — the monotonic
    /// `event_value` carries the per-fence identity. This matches the
    /// llama.cpp pattern at
    /// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:944-958`.
    ///
    /// # Label
    ///
    /// `label`'s `Some(value)` arm overwrites `stage_label` and
    /// propagates via `commit_labeled`'s `apply_labels` chain — same as
    /// calling [`Self::begin_stage`] before this. `None` keeps any
    /// previously-set `begin_stage` label intact.
    ///
    /// # Counter semantics
    ///
    /// Bumps `SYNC_COUNT` zero times (non-blocking). Bumps
    /// `CMD_BUF_COUNT` zero times (no new CB allocated here —
    /// `reset_for_next_stage` does that). Increments `event_value` by
    /// exactly 1.
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` unconditionally for the same reason
    /// [`Self::commit_stage`] does.
    pub fn fence_stage(&mut self, label: Option<&str>) -> Result<()> {
        if self.drained {
            return Ok(());
        }
        // Apply the label argument before committing so commit_labeled
        // (called below) propagates the latest value to the CB. Note
        // that the encoder.rs:1968 apply_labels writes to the active
        // compute encoder iff one is open — at this point one IS open
        // (we have not yet ended it), so the encoder picks up the label
        // before end_encoding fires. After end_encoding the CB still
        // has its label set (set on the CB itself, not the encoder).
        if let Some(l) = label {
            self.stage_label.clear();
            self.stage_label.push_str(l);
        }

        // Lazy-alloc the SharedEvent on first fence in this session.
        // metal::DeviceRef::new_shared_event lives at
        // /Users/robert/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/metal-0.33.0/src/device.rs:2063.
        if self.event.is_none() {
            self.event = Some(self.device.new_shared_event());
        }

        // Sequence: end-active-encoder + encodeSignalEvent (CB-level) +
        // residency-flush + cmd_buf.commit, all inside the inner
        // helper. This preserves F1 (encoder is ended exactly once per
        // CB), F2 (residency-flush still fires at the commit boundary),
        // and matches llama.cpp's pattern at
        // `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:944-950`.
        let new_value = self.event_value + 1;
        let event_ref: &metal::SharedEventRef = self
            .event
            .as_ref()
            .expect("event allocated immediately above this borrow")
            .as_ref();
        // Deref-coerce SharedEventRef -> EventRef via the
        // ParentType = Event chain in metal-0.33.0/src/sync.rs:36-40.
        let label_opt: Option<&str> = if self.stage_label.is_empty() {
            None
        } else {
            Some(self.stage_label.as_str())
        };
        self.inner
            .fence_signal_and_commit(event_ref, new_value, label_opt);

        self.event_value = new_value;
        self.drained = true;
        self.fence_pending = true;
        Ok(())
    }

    /// Open a fresh command buffer on the same queue and (when a fence
    /// is pending) encode the matching wait on the new CB.
    ///
    /// This is the second half of the multi-stage chaining primitive.
    /// After [`Self::fence_stage`] (or [`Self::commit_stage`] /
    /// [`Self::commit_and_wait`]) has put the session in the `Drained`
    /// state, callers invoke this to start the next stage's CB. The
    /// session transitions back to `Encoding` (no CB or compute encoder
    /// open until the next dispatch lazy-opens them).
    ///
    /// # Wait-event encoding
    ///
    /// If [`Self::fence_stage`] was the most recent commit, this
    /// method encodes `encodeWaitForEvent:value:event_value` on the
    /// freshly-allocated CB before returning. The new CB's GPU work
    /// blocks until the prior CB's signal lands at the same value.
    /// After [`Self::commit_stage`] / [`Self::commit_and_wait`] (no
    /// fence), no wait is encoded — Metal's queue-FIFO sequencing is
    /// the implicit ordering primitive.
    ///
    /// # State machine
    ///
    /// | Before | After |
    /// |---|---|
    /// | Drained (no fence) | Encoding (new CB, no wait) |
    /// | Fenced (fence pending) | Encoding (new CB, wait encoded) |
    /// | Encoding (not drained) | no-op (returns Ok) |
    ///
    /// The not-drained case is intentionally a no-op rather than a
    /// panic: it keeps the session drop-safe under unusual call
    /// sequences (e.g. test scaffolding that calls reset speculatively).
    ///
    /// # Counter semantics
    ///
    /// Bumps `CMD_BUF_COUNT` by exactly 1 (the new CB). Does NOT bump
    /// `SYNC_COUNT` (no commit/wait happens here).
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` unconditionally. Future error paths (e.g.
    /// queue-side allocation failure on `new_command_buffer`) would
    /// surface here.
    pub fn reset_for_next_stage(&mut self) -> Result<()> {
        if !self.drained {
            return Ok(());
        }

        // Snapshot the wait-event metadata BEFORE rotating cmd_buf so
        // we encode the wait on the NEW CB.
        let wait_metadata = if self.fence_pending {
            self.event
                .as_ref()
                .map(|ev| (ev.clone(), self.event_value))
        } else {
            None
        };

        self.inner.reset_command_buffer();

        if let Some((event, value)) = wait_metadata {
            // Deref-coerce SharedEventRef → EventRef via the
            // ParentType = Event chain in metal-0.33.0/src/sync.rs:36-40.
            let event_ref: &metal::EventRef = event.as_ref();
            self.inner.encode_wait_for_event(event_ref, value);
            // iter90b §2 H1b — track the wait-event for introspection.
            // Bump scoreboard ONLY after the wait actually encoded
            // (mirrors the signal-side discipline: `event_value` is
            // updated AFTER `fence_signal_and_commit` returns). These
            // fields are pure read-only observability — they do NOT
            // alter F1 (encoder lazy-open), F2 (residency-flush), F11
            // (alloc_buffer zero-init), or F12 (force-serial-dispatch).
            self.wait_count += 1;
            self.last_wait_value = value;
        }

        self.drained = false;
        self.fence_pending = false;
        self.stage_label.clear();
        Ok(())
    }

    /// Add a buffer to the device-level residency set.
    ///
    /// Delegates to the inner encoder's [`ResidencySet::add_allocation`]
    /// (the same Arc clone the device, the encoder, and every other
    /// concurrent encoder shares — single-set invariant per ADR-019:467).
    /// The actual `[set commit]` is deferred until the next
    /// `commit_stage` / `commit_and_wait` / `fence_stage`, which all
    /// route through `flush_residency_pending`.
    ///
    /// Returns `false` and is a no-op when the device booted without
    /// a residency set (HF2Q_NO_RESIDENCY=1, macOS<15, or
    /// `MlxError::DeviceNotFound` test paths).
    ///
    /// # Use case
    ///
    /// Caller holds an [`MlxBuffer`] not previously registered (e.g.
    /// from a pool, slice_view, or external interop) and wants the GPU
    /// pages hinted as resident before the stage's first dispatch.
    /// `MlxDevice::alloc_buffer` already auto-registers — this method
    /// is the explicit hook for the residual cases.
    pub fn add_to_residency_set(&self, buffer: &MlxBuffer) -> bool {
        match self.inner.residency_set() {
            Some(set) => {
                set.add_allocation(buffer.metal_buffer());
                true
            }
            None => false,
        }
    }

    /// Remove a buffer from the device-level residency set.
    ///
    /// Mirror of [`Self::add_to_residency_set`]. Stages a deferred
    /// `removeAllocation:` that flushes at the next commit boundary.
    /// Returns `false` and no-ops when no residency set is active.
    ///
    /// # F2 caveat
    ///
    /// Removing a buffer that the in-flight CB still references is the
    /// iter58b residency-rescission class. Under retained-refs
    /// (default), the CB's ARC retain keeps the underlying Metal page
    /// alive; the residency-set demotion only affects the resident-hint
    /// (a perf knob, not a safety knob). Under `MLX_UNRETAINED_REFS=1`
    /// (NOT enabled in Phase 0b), the caller-owned arena contract is
    /// the only structural mitigation.
    pub fn remove_from_residency_set(&self, buffer: &MlxBuffer) -> bool {
        match self.inner.residency_set() {
            Some(set) => {
                set.remove_allocation(buffer.metal_buffer());
                true
            }
            None => false,
        }
    }

    /// Whether the session has been committed (any commit path).
    ///
    /// Test-and-introspection helper. Production code should use the
    /// explicit `reset_for_next_stage` cycle to chain stages rather
    /// than polling this field.
    #[inline]
    pub fn is_drained(&self) -> bool {
        self.drained
    }

    /// Whether a fence is pending (most recent commit was `fence_stage`).
    ///
    /// Test-and-introspection helper for verifying the multi-stage
    /// state machine. Cleared by the next `reset_for_next_stage` /
    /// `commit_stage` / `commit_and_wait`.
    #[inline]
    pub fn is_fence_pending(&self) -> bool {
        self.fence_pending
    }

    /// The current monotonic fence value.
    ///
    /// Returns 0 before the first `fence_stage`; otherwise returns the
    /// most recently signaled value. Mirrors the semantics of
    /// `ggml_metal_event::value` — a fence at value `N` means signal
    /// `N` is in flight (or completed) and any subsequent waiters at
    /// `N` will be unblocked.
    #[inline]
    pub fn fence_value(&self) -> u64 {
        self.event_value
    }

    /// Whether a [`MTLSharedEvent`](metal::SharedEvent) has been allocated
    /// in this session.
    ///
    /// Returns `false` until the first `fence_stage`; `true` afterwards.
    /// Test helper for verifying lazy-allocation behavior.
    #[inline]
    pub fn has_event(&self) -> bool {
        self.event.is_some()
    }

    /// The most recent value passed to `encode_wait_for_event` inside
    /// [`Self::reset_for_next_stage`].
    ///
    /// Returns 0 until the first `reset_for_next_stage` actually emits
    /// a wait (i.e. the prior commit was [`Self::fence_stage`], not
    /// [`Self::commit_stage`] / [`Self::commit_and_wait`]). After a
    /// `fence_stage(N)` followed by `reset_for_next_stage()`, this MUST
    /// equal `N` — the wait-side scoreboard mirrors the signal-side
    /// [`Self::fence_value`].
    ///
    /// iter90b §2 H1b proof helper: makes the wait-event encoding
    /// observable from a Rust test without xctrace.
    ///
    /// # Risk register
    ///
    /// Pure read-only introspection. Reads a `u64` field updated under
    /// `&mut self` exclusively (no concurrent mutation possible —
    /// `EncoderSession` is `!Sync`). Does NOT widen F1/F2/F11/F12.
    #[inline]
    pub fn wait_value(&self) -> u64 {
        self.last_wait_value
    }

    /// Cumulative count of `encode_wait_for_event` calls actually
    /// emitted inside [`Self::reset_for_next_stage`] in this session.
    ///
    /// Bumped exactly once per `reset_for_next_stage` call that finds
    /// `fence_pending == true` — i.e. once per "fence + reset" pair.
    /// `commit_stage` / `commit_and_wait` followed by
    /// `reset_for_next_stage` does NOT bump this (no wait emitted —
    /// Metal queue FIFO is the implicit ordering primitive in that
    /// case).
    ///
    /// For an N-stage chain (N fences + (N-1) resets), this returns
    /// `N - 1` after the last reset. The Nth (terminal) fence is
    /// drained by the caller via `metal_command_buffer().wait_until_completed()`
    /// or by a subsequent `commit_and_wait`, neither of which emits an
    /// additional wait.
    ///
    /// iter90b §2 H1b proof helper: paired with [`Self::wait_value`] to
    /// make the wait-event side of the multi-stage chain observable.
    ///
    /// # Risk register
    ///
    /// Same as [`Self::wait_value`] — pure read-only introspection over
    /// a `u64` field updated under `&mut self` exclusively. Does NOT
    /// widen F1/F2/F11/F12.
    #[inline]
    pub fn wait_count(&self) -> u64 {
        self.wait_count
    }

    /// Borrow the underlying Metal command buffer.
    ///
    /// Mirrors [`CommandEncoder::metal_command_buffer`]. Used by
    /// label-propagation tests and by callers that need to call
    /// `wait_until_completed` after a non-blocking `commit_stage` /
    /// `fence_stage`.
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
    /// Drop scenarios across the multi-stage state machine:
    ///
    /// 1. **Drained (no fence)** — `commit_stage` / `commit_and_wait`
    ///    already ran. `inner.flush_residency_pending()` was already
    ///    called; the GPU has the CB (and may already have completed
    ///    it under `commit_and_wait`). `CommandEncoder::Drop` runs and
    ///    calls `end_active_encoder()`, which is a no-op because
    ///    `commit*` already ended the encoder. Safe.
    ///
    /// 2. **Fenced (fence pending)** — `fence_stage` already ran. The
    ///    signal-event has been encoded onto the prior CB and the CB
    ///    has been submitted non-blocking. The session never opened a
    ///    new CB (no `reset_for_next_stage` call), so `cmd_buf` still
    ///    points at the FENCED CB. `CommandEncoder::Drop` runs and
    ///    end_active_encoder is a no-op (encoder was ended inside
    ///    `fence_signal_and_commit`). The submitted CB executes on the
    ///    GPU normally — the signal lands, the value is observable to
    ///    any external `waitUntilSignaledValue:` consumer (none in
    ///    iter89e2-B), and the next allocation/CB on the same residency
    ///    set will see the bumped pending flag flushed at its commit
    ///    boundary. The fence event itself is dropped with `event` (an
    ///    Option<SharedEvent>); ARC drop releases it.
    ///
    /// 3. **Encoding (uncommitted)** — caller created the session,
    ///    optionally encoded dispatches, then dropped without calling
    ///    any `commit_*`. `CommandEncoder::Drop` ends the active
    ///    compute encoder cleanly (`encoder.rs:2057-2063`). The
    ///    `cmd_buf` is dropped without ever being committed — Metal
    ///    discards the encoded work. **No residency-remove is staged**
    ///    because no buffers were registered as freed during this
    ///    session (the F2 race requires a buffer drop staging a remove
    ///    that a later `flush_pending` commits before the in-flight CB
    ///    finishes; here no commit ever happens). The residency-set's
    ///    pending state persists into the next encoder; correct.
    ///
    /// 4. **Empty** — no dispatches encoded. `active_encoder` is null;
    ///    `CommandEncoder::Drop`'s `end_active_encoder` is a no-op.
    ///    Safe.
    ///
    /// We deliberately do NOT call `wait_until_completed` here for the
    /// committed-but-not-waited case (scenarios 1 with `commit_stage`
    /// or 2 with `fence_stage`). Under retained-refs mode (default —
    /// `MLX_UNRETAINED_REFS=0`), the in-flight CB holds ARC retains on
    /// every bound buffer, so the GPU completes safely after the
    /// session drops. Under `MLX_UNRETAINED_REFS=1` (NOT enabled in
    /// Phase 0b), the caller-owned-arena contract is the only
    /// structural mitigation — same as the existing async-`commit()`
    /// path at `encoder.rs:2014-2022`.
    ///
    /// In short: `Drop` does no extra work; the inner `CommandEncoder`'s
    /// own Drop is the entire safety story. `metal::SharedEvent` drops
    /// via its foreign_obj_type! ARC release.
    fn drop(&mut self) {
        // The actual end-encoder call lives in `CommandEncoder::Drop`,
        // which fires automatically when `self.inner` goes out of scope
        // here. The `event` field's ObjC release fires via
        // foreign_obj_type! ARC. No additional work needed — see this
        // docstring's case analysis above for the F2 fence preservation
        // argument.
    }
}
