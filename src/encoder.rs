//! [`CommandEncoder`] — batched GPU command submission.
//!
//! Wraps a Metal command buffer.  Encode one or more compute kernel dispatches,
//! then call [`commit_and_wait`](CommandEncoder::commit_and_wait) to submit the
//! entire batch and block until the GPU finishes.
//!
//! # Persistent compute encoder
//!
//! A single Metal `ComputeCommandEncoder` is kept alive across multiple
//! dispatches within the same command buffer.  This avoids the overhead of
//! creating and ending a new compute encoder per dispatch — the same pattern
//! candle uses (`compute_per_buffer`).  On a forward pass with ~800 dispatches
//! this saves ~800 encoder create/end cycles.
//!
//! # Capture mode (Phase 4e.1)
//!
//! When `start_capture()` is called, subsequent dispatches are recorded into a
//! `Vec<CapturedNode>` instead of being encoded into Metal.  `memory_barrier()`
//! records a barrier sentinel.  Call `take_capture()` to extract the recorded
//! graph for later replay via `ComputeGraph::encode_sequential()`.

use std::sync::atomic::{AtomicU64, Ordering};

use metal::{
    CommandBuffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState,
    ComputePipelineStateRef, MTLCommandBufferStatus, MTLDispatchType, MTLSize,
};
#[allow(unused_imports)]
use objc::{msg_send, sel, sel_impl};

use crate::buffer::MlxBuffer;
use crate::error::{MlxError, Result};
use crate::mem_ranges::MemRanges;
use crate::residency::ResidencySet;

/// A buffer or inline-bytes binding for a compute kernel argument slot.
pub enum KernelArg<'a> {
    /// Bind an existing Metal buffer at the given index.
    Buffer(&'a MlxBuffer),
    /// Bind an existing Metal buffer at the given index with a byte offset.
    BufferWithOffset(&'a MlxBuffer, u64),
    /// Bind inline bytes (small constant data) at the given index.
    /// The data must be `Pod` and is copied into the command encoder.
    Bytes(&'a [u8]),
}

/// Convert a `Pod` value to a byte slice suitable for `KernelArg::Bytes`.
///
/// # Safety
///
/// The caller must ensure `T` has the same layout as the corresponding
/// MSL struct in the shader (matching field order, sizes, and alignment).
pub fn as_bytes<T: bytemuck::Pod>(val: &T) -> &[u8] {
    bytemuck::bytes_of(val)
}

// ---------------------------------------------------------------------------
// Capture-mode types (Phase 4e.1 — Graph IR)
// ---------------------------------------------------------------------------

/// A recorded kernel argument binding.
///
/// When the encoder is in capture mode, each `set_buffer` / `set_bytes` call
/// is stored as a `RecordedBinding` instead of being applied to Metal.
#[derive(Clone)]
pub enum RecordedBinding {
    /// A Metal buffer at the given offset.
    Buffer {
        metal_buffer: metal::Buffer,
        offset: u64,
    },
    /// Inline bytes (small constant data, copied).
    Bytes(Vec<u8>),
}

/// How to dispatch the recorded kernel.
#[derive(Clone, Copy, Debug)]
pub enum DispatchKind {
    /// `dispatch_threads(grid_size, threadgroup_size)` — Metal picks threadgroup count.
    Threads,
    /// `dispatch_thread_groups(threadgroups, threadgroup_size)` — caller specifies threadgroup count.
    ThreadGroups,
}

/// Operation kind tag for captured nodes, used by the fusion pass (4e.2).
///
/// When the encoder is in capture mode, each dispatch can be tagged with an
/// `OpKind` so the fusion pass can identify fuseable sequences without
/// inspecting pipeline names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturedOpKind {
    /// RMS normalization (with learned scale).
    RmsNorm,
    /// Elementwise multiply.
    ElemMul,
    /// Elementwise add.
    ElemAdd,
    /// Scaled dot-product attention (NOT reorderable — breaks lookahead).
    Sdpa,
    /// Softmax (NOT reorderable — breaks lookahead).
    Softmax,
    /// Any other operation — treated as reorderable by the graph optimizer.
    Other,
}

impl CapturedOpKind {
    /// Whether this captured op kind is safe to reorder past in the graph
    /// optimizer (Phase 4e.3).
    ///
    /// Mirrors the `h_safe` whitelist from llama.cpp's
    /// `ggml_metal_graph_optimize_reorder`.  Non-safe ops break the 64-node
    /// lookahead — the reorder pass cannot look past them.
    pub fn is_reorderable(&self) -> bool {
        match self {
            Self::Sdpa | Self::Softmax => false,
            Self::RmsNorm | Self::ElemMul | Self::ElemAdd | Self::Other => true,
        }
    }
}

/// A memory range annotation: (start_address, end_address).
///
/// Represents a contiguous GPU buffer region for conflict detection in the
/// reorder pass (Phase 4e.3).  Addresses are CPU-visible `contents_ptr()`
/// values, which on Apple Silicon unified memory equal the GPU addresses.
pub type MemRange = (usize, usize);

/// A single captured compute dispatch or barrier sentinel.
///
/// Created when the encoder is in capture mode.  Replayed later by
/// `ComputeGraph::encode_sequential()`.
#[derive(Clone)]
pub enum CapturedNode {
    /// A compute dispatch to replay.
    Dispatch {
        /// Pipeline state object to bind.
        pipeline: ComputePipelineState,
        /// Kernel argument bindings: (slot_index, binding).
        bindings: Vec<(u64, RecordedBinding)>,
        /// Grid or threadgroup count (interpretation depends on `dispatch_kind`).
        threads_per_grid: MTLSize,
        /// Threads per threadgroup.
        threads_per_threadgroup: MTLSize,
        /// Optional threadgroup memory allocations: (index, byte_length).
        threadgroup_memory: Vec<(u64, u64)>,
        /// Whether this is a dispatch_threads or dispatch_thread_groups call.
        dispatch_kind: DispatchKind,
        /// Operation kind tag for the fusion pass (4e.2).
        /// Defaults to `Other` if not explicitly set via `set_op_kind()`.
        op_kind: CapturedOpKind,
        /// Read buffer ranges for reorder conflict detection (4e.3).
        /// Populated from `barrier_between` calls in capture mode.
        reads: Vec<MemRange>,
        /// Write buffer ranges for reorder conflict detection (4e.3).
        /// Populated from `barrier_between` calls in capture mode.
        writes: Vec<MemRange>,
    },
    /// A memory barrier sentinel — forces a barrier at replay time.
    Barrier,
}

/// Convert a slice of buffer references into capture-mode
/// [`MemRange`] tuples.  Used by the [`CommandEncoder::dispatch_tracked*`]
/// family in capture mode — equivalent to the conversion
/// `GraphSession::barrier_between` does at `graph.rs:1452-1465`.
///
/// `(start, end)` uses `contents_ptr() + byte_offset` as the start
/// and `contents_ptr() + byte_offset + slice_extent` as the end.
fn ranges_from_buffers(bufs: &[&MlxBuffer]) -> Vec<MemRange> {
    bufs.iter()
        .map(|b| {
            let base = b.contents_ptr() as usize + b.byte_offset() as usize;
            let extent = (b.byte_len()).saturating_sub(b.byte_offset() as usize);
            (base, base + extent)
        })
        .collect()
}

/// Apply a slice of `KernelArg` bindings to a compute encoder.
///
/// `KernelArg::Buffer(buf)` propagates the `MlxBuffer::byte_offset()` so
/// `slice_view`-derived sub-buffers are honored automatically — the
/// kernel sees memory starting at the slice's offset. This matches the
/// documented contract of `slice_view` and the offset-handling in the
/// other binding paths in this file (`encode`, `encode_threadgroups`,
/// `encode_threadgroups_with_shared`, replay). Without it, every
/// `slice_view`-derived buffer bound via `KernelArg::Buffer` silently
/// exposes the entire underlying allocation — surfaced by hf2q's
/// nomic-bert iter-79 cosine parity bisection (cosine 0.098 → 0.999962
/// after fix).
///
/// `KernelArg::BufferWithOffset(buf, offset)` continues to use the
/// explicit `offset` argument verbatim (callers asking for an explicit
/// offset get exactly that, even on sliced buffers). The two API
/// surfaces are intentional: implicit (sliced views auto-propagate) vs.
/// explicit (caller-controlled).
#[inline]
fn apply_bindings(encoder: &ComputeCommandEncoderRef, bindings: &[(u64, KernelArg<'_>)]) {
    for &(index, ref arg) in bindings {
        match arg {
            KernelArg::Buffer(buf) => {
                encoder.set_buffer(index, Some(buf.metal_buffer()), buf.byte_offset());
            }
            KernelArg::BufferWithOffset(buf, offset) => {
                encoder.set_buffer(index, Some(buf.metal_buffer()), *offset);
            }
            KernelArg::Bytes(bytes) => {
                encoder.set_bytes(index, bytes.len() as u64, bytes.as_ptr() as *const _);
            }
        }
    }
}

/// Number of times `commit_and_wait()` has been called (CPU sync points).
static SYNC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of times an encode method has been called (GPU dispatches).
static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of `MTLCommandBuffer` instances created via `CommandEncoder::new`.
/// Increments once per `device.command_encoder()` call.  Used by hf2q's
/// `HF2Q_DECODE_PROFILE` instrumentation to measure command-buffer
/// overhead per decode token (ADR-012 §Optimize / Task #15 follow-up).
static CMD_BUF_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of `memory_barrier()` calls that reached the
/// `objc::msg_send![encoder, memoryBarrierWithScope:]` site.  Capture-mode
/// no-ops and pre-encoder no-ops are excluded so the count reflects
/// actual MTL barriers issued.
///
/// Always tracked — the increment is one atomic op, ~5 ns.  ADR-015 H4
/// (Wave 2b hard gate #2) requires per-barrier counter resolution to
/// confirm-or-falsify the barrier-coalescing lever; xctrace TimeProfiler
/// at 1 ms sampling cannot resolve `memory_barrier` even though it fires
/// ~440×/token (`docs/ADR-015-mlx-native-single-cb-decode.md` §"P3a' live
/// profile pass" hypothesis register row H4).
static BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total nanoseconds spent inside the `objc::msg_send!` barrier site,
/// summed across all calls.  ONLY updated when the env var
/// `MLX_PROFILE_BARRIERS=1` is set on the process (cached on first
/// `memory_barrier` call).  When disabled the timing path is a single
/// branch + the unconditional barrier dispatch — same hot-path cost as
/// before this counter was added.
///
/// Why env-gated: timing adds 2 × `Instant::now()` (~50–100 ns each via
/// `mach_absolute_time`) per barrier.  At ~440 barriers/token that is
/// ~22–44 µs/token of measurement overhead — comparable to what we are
/// trying to measure.  Production must keep this off; profiling runs
/// opt-in.
static BARRIER_NS: AtomicU64 = AtomicU64::new(0);

/// Reset all counters to zero.
pub fn reset_counters() {
    SYNC_COUNT.store(0, Ordering::Relaxed);
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
    CMD_BUF_COUNT.store(0, Ordering::Relaxed);
    BARRIER_COUNT.store(0, Ordering::Relaxed);
    BARRIER_NS.store(0, Ordering::Relaxed);
    AUTO_BARRIER_COUNT.store(0, Ordering::Relaxed);
    AUTO_BARRIER_CONCURRENT.store(0, Ordering::Relaxed);
}

/// Read the current value of `SYNC_COUNT`.
///
/// Each call to `commit_and_wait()` increments this counter.
pub fn sync_count() -> u64 {
    SYNC_COUNT.load(Ordering::Relaxed)
}

/// Read the current value of `DISPATCH_COUNT`.
///
/// Each call to `encode()`, `encode_threadgroups()`, or
/// `encode_threadgroups_with_shared()` increments this counter.
pub fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Read the current value of `CMD_BUF_COUNT`.
///
/// Each `CommandEncoder::new` (i.e. each `MlxDevice::command_encoder()`)
/// increments this counter.  Useful for diagnosing per-dispatch Metal
/// command-buffer overhead in inner loops.
pub fn cmd_buf_count() -> u64 {
    CMD_BUF_COUNT.load(Ordering::Relaxed)
}

/// Read the current value of `BARRIER_COUNT`.
///
/// Each `memory_barrier()` call that reaches the underlying
/// `objc::msg_send![encoder, memoryBarrierWithScope:]` site increments this
/// counter.  Capture-mode no-ops and pre-encoder no-ops are excluded.
/// ADR-015 H4 hypothesis: ~440 barriers/token on the qwen35 decode hot
/// path (verify against this counter).
pub fn barrier_count() -> u64 {
    BARRIER_COUNT.load(Ordering::Relaxed)
}

/// Read the total nanoseconds spent in the `memoryBarrierWithScope:`
/// `objc::msg_send!` site.  Only non-zero when `MLX_PROFILE_BARRIERS=1`
/// was in the environment at the time of the first `memory_barrier()`
/// call (the env check is cached on first use).
///
/// Combined with [`barrier_count`] this gives µs/barrier =
/// `barrier_total_ns() / 1000 / barrier_count()`.
pub fn barrier_total_ns() -> u64 {
    BARRIER_NS.load(Ordering::Relaxed)
}

/// Whether barrier timing is enabled (env-gated, cached on first check).
///
/// Reading the env var via `std::env::var` is itself non-trivial; using
/// `OnceLock` caches the decision so the per-barrier branch is a single
/// atomic-load + compare.
fn barrier_profile_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("MLX_PROFILE_BARRIERS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Whether `MLX_UNRETAINED_REFS=1` is set in the process environment.
///
/// ADR-015 iter13 — when true, `CommandEncoder::new_with_residency` opens
/// each `MTLCommandBuffer` via
/// [`CommandQueueRef::new_command_buffer_with_unretained_references`]
/// instead of the default `commandBuffer`.  llama.cpp's per-token decode
/// CBs use this same call (`/opt/llama.cpp/ggml/src/ggml-metal/`
/// `ggml-metal-context.m:512` `[queue commandBufferWithUnretainedReferences]`)
/// and gain ~3-5% wall on M-series GPUs by skipping per-buffer-binding ARC
/// retains on submit.
///
/// **Caller-side prerequisite.**  Every Metal buffer bound to a dispatch
/// must outlive the CB — see the docstring on
/// [`CommandEncoder::new_with_residency`] for the full caller contract.
/// In hf2q, the per-decode-token `MlxBufferPool` (`buffer_pool.rs`)
/// already keeps ARC clones alive in its `in_use` list across the entire
/// decode token; routing transient scratches through that pool is the
/// canonical way to satisfy the contract.
///
/// Cached on first read via `OnceLock` to keep the per-CB-construction
/// branch single-atomic-load fast.  Default OFF so any production decode
/// run that does NOT explicitly set the var preserves retained-refs
/// behavior verbatim.
fn unretained_refs_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("MLX_UNRETAINED_REFS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Whether `HF2Q_AUTO_BARRIER=1` is set in the process environment.
///
/// ADR-015 iter37 — when true, every [`CommandEncoder::dispatch_tracked`]
/// call consults a [`MemRanges`](crate::mem_ranges::MemRanges) tracker
/// and auto-emits a `memoryBarrierWithScope:` exactly when the new
/// dispatch's read/write ranges conflict with previously-recorded
/// ranges (mirrors llama.cpp's `ggml_metal_op_concurrency_check` at
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:147-225`).
/// When false, `dispatch_tracked` collapses to the same code path as
/// `encode*` — no tracking, no auto-barriers — preserving sourdough
/// behavior for any caller that opts into the tracked API but runs
/// without the env gate.
///
/// Cached on first read via `OnceLock`.  Default OFF — production
/// decode/prefill keeps its hand-placed `enc.memory_barrier()` calls
/// until the migration in iter38+.
fn auto_barrier_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("HF2Q_AUTO_BARRIER")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Number of `memory_barrier()` calls auto-emitted by
/// [`CommandEncoder::dispatch_tracked`] under
/// `HF2Q_AUTO_BARRIER=1`.  Disjoint from [`BARRIER_COUNT`] —
/// auto-barriers also bump `BARRIER_COUNT` since they go through
/// `memory_barrier()`, so this counter measures only the
/// auto-emitted subset.
static AUTO_BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of `dispatch_tracked` calls whose mem-ranges check returned
/// "concurrent" (no barrier needed).  Together with
/// [`AUTO_BARRIER_COUNT`] this measures the elision rate of the
/// dataflow barrier: `concurrent / (concurrent + barriers)` is the
/// fraction of dispatches that ran inside the previous concurrent
/// group rather than starting a new one.
static AUTO_BARRIER_CONCURRENT: AtomicU64 = AtomicU64::new(0);

/// Read the cumulative number of auto-emitted barriers across all
/// encoders since process start (or last [`reset_counters`]).
pub fn auto_barrier_count() -> u64 {
    AUTO_BARRIER_COUNT.load(Ordering::Relaxed)
}

/// Read the cumulative number of `dispatch_tracked` calls that did NOT
/// emit a barrier (ran concurrent with the previous group).
pub fn auto_barrier_concurrent_count() -> u64 {
    AUTO_BARRIER_CONCURRENT.load(Ordering::Relaxed)
}

/// Issue the underlying Metal `memoryBarrierWithScope:` ObjC msg_send.
///
/// Held in its own `#[inline(never)]` function so xctrace / Instruments
/// has a stable Rust frame to attribute barrier time against, separate
/// from the surrounding encoder accounting.  Per ADR-015 §P3a' Codex
/// review Q2: TimeProfiler at 1 ms sampling cannot see this site when
/// inlined; an explicit non-inline frame plus the [`BARRIER_NS`] counter
/// closes the H4 hard gate.
#[inline(never)]
fn issue_metal_buffer_barrier(encoder: &ComputeCommandEncoderRef) {
    // MTLBarrierScopeBuffers = 1 << 0 = 1.
    const MTL_BARRIER_SCOPE_BUFFERS: u64 = 1;
    unsafe {
        let _: () =
            objc::msg_send![encoder, memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS];
    }
}

/// A batched compute command encoder.
///
/// Keeps a single Metal `ComputeCommandEncoder` alive across multiple
/// dispatches.  The encoder is created on the first dispatch and ended
/// only when the command buffer is committed.  This mirrors candle's
/// `compute_per_buffer` pattern and avoids per-dispatch encoder overhead.
///
/// # Typical usage
///
/// ```ignore
/// let mut enc = device.command_encoder()?;
/// // Multiple dispatches share the same compute encoder:
/// enc.encode_threadgroups(pipeline1, &buffers1, tg1, tg_size1);
/// enc.encode_threadgroups(pipeline2, &buffers2, tg2, tg_size2);
/// enc.commit_and_wait()?;
/// ```
pub struct CommandEncoder {
    cmd_buf: CommandBuffer,
    // SAFETY marker: see unsafe Send impl below.
    /// Raw pointer to the persistent compute encoder.
    /// Non-null when a compute pass is active.
    /// The encoder borrows from `cmd_buf` but we cannot express this
    /// lifetime in safe Rust, so we use a raw pointer.
    /// SAFETY: the pointer is valid as long as `cmd_buf` is alive and
    /// `end_encoding()` has not been called on it.
    active_encoder: *const ComputeCommandEncoderRef,
    /// When `Some`, dispatches are recorded here instead of being encoded
    /// into Metal.  Set via `start_capture()`, extracted via `take_capture()`.
    capture: Option<Vec<CapturedNode>>,
    /// Op kind tag for the NEXT captured dispatch.  Set via `set_op_kind()`,
    /// consumed (reset to `Other`) when a dispatch is captured.
    pending_op_kind: CapturedOpKind,
    /// Pending read buffer ranges for the NEXT captured dispatch.
    /// Set via `set_pending_buffer_ranges()`, consumed when the next dispatch
    /// is captured.  Used by the reorder pass (Phase 4e.3).
    pending_reads: Vec<MemRange>,
    /// Pending write buffer ranges for the NEXT captured dispatch.
    pending_writes: Vec<MemRange>,
    /// ADR-015 iter8e (Phase 3b): residency set whose pending add/remove
    /// staging is flushed at every `commit*` boundary.
    ///
    /// Cloned from the device at `device.command_encoder()` time. `None`
    /// when residency sets are disabled (HF2Q_NO_RESIDENCY=1, macOS<15,
    /// or test-only `CommandEncoder::new` from a residency-less queue).
    residency_set: Option<ResidencySet>,
    /// ADR-015 iter37: dataflow barrier inference state.
    ///
    /// Populated only when `HF2Q_AUTO_BARRIER=1` is set at process
    /// start (cached via [`auto_barrier_enabled`]).  Each
    /// [`Self::dispatch_tracked`] call consults this state to decide
    /// whether a Metal memory barrier is required; on conflict the
    /// barrier is emitted, the state is reset, and the new dispatch's
    /// ranges seed the next concurrent group.  When the env gate is
    /// off, `dispatch_tracked` collapses to its untracked equivalent
    /// and this field is left empty for the encoder's lifetime.
    ///
    /// The field is always present (zero-sized when empty) so the
    /// gate-off branch is a single bool-load + early return rather
    /// than an allocation/Option indirection.
    mem_ranges: MemRanges,
}

/// SAFETY: CommandEncoder is safe to Send across threads provided that:
/// 1. Only one thread accesses the encoder at a time (exclusive ownership).
/// 2. The encoder is not used concurrently from multiple threads.
///
/// Metal command buffers and compute encoders are thread-safe for exclusive
/// access (Apple documentation: "You can create command buffers, encode
/// commands, and submit them from any thread"). The raw pointer
/// `active_encoder` borrows from `cmd_buf` and is valid as long as
/// `cmd_buf` is alive — this invariant holds across thread boundaries
/// because both fields move together.
///
/// This matches llama.cpp's pattern of encoding command buffers on GCD
/// worker threads via `dispatch_apply`, and is used for the dual-buffer
/// pipeline where buf1 is encoded on a worker thread while buf0 executes.
unsafe impl Send for CommandEncoder {}

impl CommandEncoder {
    /// Create a new command encoder from the given command queue.
    ///
    /// This immediately creates a Metal command buffer.
    ///
    /// # Why retained references
    ///
    /// We use the regular `commandBuffer` (Metal retains every bound
    /// resource for the lifetime of the buffer) rather than
    /// `commandBufferWithUnretainedReferences`.  llama.cpp uses unretained
    /// refs for an additional perf bump (~3-5% on M-series GPUs), but the
    /// hf2q dispatch pattern allocates many transient scratch buffers
    /// inside helper functions (`apply_proj` → `weight_bf16_owned`,
    /// `apply_pre_norm` → `params`, etc.) that go out of scope at the
    /// helper's return.  With unretained refs the metal::Buffer's ARC
    /// drops to zero, freeing the underlying GPU memory before the
    /// dispatch executes.  Verified 2026-04-26: switching to unretained
    /// hits "Command buffer error: GPU command buffer completed with
    /// error status" on the first MoE FFN dispatch.
    ///
    /// To enable unretained refs in the future, every helper that
    /// allocates and dispatches must thread its scratch buffers up to a
    /// caller scope that outlives the eventual commit, OR all such
    /// scratch must come from the per-decode-token pool (which already
    /// ARC-retains in its in_use list).  Today the lm_head + router-
    /// download paths are still unpooled.
    #[allow(dead_code)]
    pub(crate) fn new(queue: &CommandQueue) -> Result<Self> {
        Self::new_with_residency(queue, None)
    }

    /// Create a new command encoder, optionally bound to a residency set so
    /// `commit*` boundaries can flush deferred add/remove staging.
    ///
    /// ADR-015 iter8e (Phase 3b): the encoder's `commit_and_wait`,
    /// `commit_and_wait_labeled`, `commit`, `commit_labeled`,
    /// `commit_wait_with_gpu_time` all call
    /// [`ResidencySet::flush_pending`](ResidencySet::flush_pending) before
    /// submitting the Metal command buffer. This converts the
    /// per-allocation `[set commit]` storm
    /// (~880 commits/decode-token in iter8d/8e claude+codex variants) into
    /// at most one commit per CB submission — mirrors llama.cpp's
    /// `ggml-metal-device.m:1378-1382` pattern (batch addAllocation in
    /// loop, commit ONCE).
    ///
    /// ADR-015 iter13: when the `MLX_UNRETAINED_REFS=1` env var is set at
    /// process start, this constructor uses
    /// [`CommandQueueRef::new_command_buffer_with_unretained_references`]
    /// instead of `new_command_buffer`.  llama.cpp's per-token decode CBs
    /// use `commandBufferWithUnretainedReferences` (see
    /// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:512`) which
    /// skips Metal's per-buffer-binding ARC-retain on submit and saves
    /// ~3-5% on M-series GPUs (per the docstring above).
    ///
    /// **Caller contract under unretained refs.**  Every Metal buffer bound
    /// to a dispatch in this CB MUST outlive the CB's GPU completion.  In
    /// the hf2q decode path, that means every transient scratch must be
    /// either (a) backed by the per-decode-token arena pool
    /// (`MlxBufferPool` keeps an ARC clone in `in_use` until the next
    /// `reset` — see `buffer_pool.rs:60`) or (b) hoisted to a caller scope
    /// that lives across the terminal `commit_and_wait_labeled`.  Helpers
    /// in `apply_proj` / `apply_pre_norm` / lm_head cast / router-download
    /// that allocated transients via `device.alloc_buffer` and dropped
    /// them at function return MUST be lifted to `pooled_alloc_buffer`
    /// before `MLX_UNRETAINED_REFS=1` is enabled, or the first MoE FFN
    /// dispatch will crash with "Command buffer error: GPU command buffer
    /// completed with error status" (verified 2026-04-26).
    ///
    /// The default (`MLX_UNRETAINED_REFS` unset) preserves retained-refs
    /// behavior verbatim — this is the sourdough-safe path.
    pub(crate) fn new_with_residency(
        queue: &CommandQueue,
        residency_set: Option<ResidencySet>,
    ) -> Result<Self> {
        let cmd_buf = if unretained_refs_enabled() {
            queue.new_command_buffer_with_unretained_references().to_owned()
        } else {
            queue.new_command_buffer().to_owned()
        };
        CMD_BUF_COUNT.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            cmd_buf,
            active_encoder: std::ptr::null(),
            capture: None,
            pending_op_kind: CapturedOpKind::Other,
            pending_reads: Vec::new(),
            pending_writes: Vec::new(),
            residency_set,
            mem_ranges: MemRanges::new(),
        })
    }

    /// Enable capture mode.
    ///
    /// All subsequent dispatch and barrier calls will be recorded into a
    /// `Vec<CapturedNode>` instead of being encoded into Metal.
    /// Call `take_capture()` to extract the recorded nodes.
    pub fn start_capture(&mut self) {
        self.capture = Some(Vec::with_capacity(128));
    }

    /// Whether the encoder is currently in capture mode.
    pub fn is_capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Extract the captured nodes, ending capture mode.
    ///
    /// Returns `None` if capture mode was not active.
    pub fn take_capture(&mut self) -> Option<Vec<CapturedNode>> {
        self.capture.take()
    }

    /// Tag the NEXT captured dispatch with the given operation kind.
    ///
    /// The tag is consumed (reset to `Other`) after the next dispatch is
    /// captured.  Only meaningful in capture mode — has no effect on
    /// direct-dispatch encoding.
    ///
    /// Used by op dispatch functions to annotate captures for the fusion
    /// pass (Phase 4e.2).
    pub fn set_op_kind(&mut self, kind: CapturedOpKind) {
        self.pending_op_kind = kind;
    }

    /// Consume and return the pending op kind, resetting it to `Other`.
    fn take_pending_op_kind(&mut self) -> CapturedOpKind {
        let kind = self.pending_op_kind;
        self.pending_op_kind = CapturedOpKind::Other;
        kind
    }

    /// Stash buffer range annotations for the NEXT captured dispatch.
    ///
    /// Called by `GraphSession::barrier_between()` in capture mode to record
    /// which buffers the next dispatch reads from and writes to.  The ranges
    /// are consumed by the next `encode_*` call and attached to the captured
    /// `CapturedNode::Dispatch`.
    ///
    /// Only meaningful in capture mode — has no effect on direct-dispatch.
    pub fn set_pending_buffer_ranges(&mut self, reads: Vec<MemRange>, writes: Vec<MemRange>) {
        self.pending_reads = reads;
        self.pending_writes = writes;
    }

    /// Patch the last captured dispatch node's empty reads/writes with the
    /// given ranges. No-op if not capturing, or if the last node isn't a
    /// Dispatch, or if its ranges are already populated.
    ///
    /// Used by `GraphSession::track_dispatch` in recording mode to annotate
    /// dispatches that were called without a preceding `barrier_between`.
    pub fn annotate_last_dispatch_if_missing(&mut self, reads: Vec<MemRange>, writes: Vec<MemRange>) {
        if let Some(ref mut nodes) = self.capture {
            if let Some(CapturedNode::Dispatch { reads: r, writes: w, .. }) = nodes.last_mut() {
                if r.is_empty() && !reads.is_empty() {
                    *r = reads;
                }
                if w.is_empty() && !writes.is_empty() {
                    *w = writes;
                }
            }
        }
    }

    /// Consume and return the pending buffer range annotations.
    fn take_pending_buffer_ranges(&mut self) -> (Vec<MemRange>, Vec<MemRange>) {
        let reads = std::mem::take(&mut self.pending_reads);
        let writes = std::mem::take(&mut self.pending_writes);
        (reads, writes)
    }

    /// Record buffer bindings into `RecordedBinding` form.
    fn record_buffer_bindings(buffers: &[(u64, &MlxBuffer)]) -> Vec<(u64, RecordedBinding)> {
        buffers
            .iter()
            .map(|&(index, buf)| {
                (
                    index,
                    RecordedBinding::Buffer {
                        metal_buffer: buf.metal_buffer().clone(),
                        offset: buf.byte_offset(),
                    },
                )
            })
            .collect()
    }

    /// Record `KernelArg` bindings into `RecordedBinding` form.
    ///
    /// `KernelArg::Buffer(buf)` records `buf.byte_offset()` so capture →
    /// replay round-trips of `slice_view`-derived buffers preserve their
    /// offsets, matching `record_buffer_bindings`'s behavior at line 382.
    fn record_arg_bindings(bindings: &[(u64, KernelArg<'_>)]) -> Vec<(u64, RecordedBinding)> {
        bindings
            .iter()
            .map(|(index, arg)| {
                let recorded = match arg {
                    KernelArg::Buffer(buf) => RecordedBinding::Buffer {
                        metal_buffer: buf.metal_buffer().clone(),
                        offset: buf.byte_offset(),
                    },
                    KernelArg::BufferWithOffset(buf, offset) => RecordedBinding::Buffer {
                        metal_buffer: buf.metal_buffer().clone(),
                        offset: *offset,
                    },
                    KernelArg::Bytes(bytes) => RecordedBinding::Bytes(bytes.to_vec()),
                };
                (*index, recorded)
            })
            .collect()
    }

    /// Get or create the persistent compute encoder.
    ///
    /// On the first call, creates a new compute encoder from the command
    /// buffer.  On subsequent calls, returns the existing one.
    ///
    /// SAFETY: The returned reference borrows from `self.cmd_buf` which is
    /// alive for the lifetime of this `CommandEncoder`.  The raw pointer is
    /// valid until `end_active_encoder()` is called.
    #[inline]
    fn get_or_create_encoder(&mut self) -> &ComputeCommandEncoderRef {
        if self.active_encoder.is_null() {
            // Use MTLDispatchTypeConcurrent to allow independent dispatches
            // to overlap on the GPU.  Memory barriers are inserted between
            // dependent dispatches via `memory_barrier()`.
            let encoder = self
                .cmd_buf
                .compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            self.active_encoder = encoder as *const ComputeCommandEncoderRef;
        }
        // SAFETY: active_encoder is non-null and points to a valid encoder
        // owned by cmd_buf.
        unsafe { &*self.active_encoder }
    }

    /// End the active compute encoder if one exists.
    #[inline]
    fn end_active_encoder(&mut self) {
        if !self.active_encoder.is_null() {
            // SAFETY: the pointer was obtained from cmd_buf.new_compute_command_encoder()
            // and has not been ended yet.
            unsafe { &*self.active_encoder }.end_encoding();
            self.active_encoder = std::ptr::null();
        }
    }

    /// Insert a memory barrier with scope `MTLBarrierScopeBuffers`.
    ///
    /// When the encoder uses `MTLDispatchTypeConcurrent`, all dispatches can
    /// execute concurrently unless separated by a barrier.  Call this between
    /// dispatches where the later dispatch reads a buffer written by an
    /// earlier one.
    ///
    /// This is the same pattern llama.cpp uses:
    /// `[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]`
    #[allow(unexpected_cfgs)]
    pub fn memory_barrier(&mut self) {
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Barrier);
            return;
        }
        if self.active_encoder.is_null() {
            return;
        }
        BARRIER_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: active_encoder is non-null and valid.
        let encoder = unsafe { &*self.active_encoder };
        if barrier_profile_enabled() {
            // mach_absolute_time path — only on when MLX_PROFILE_BARRIERS=1.
            let start = std::time::Instant::now();
            issue_metal_buffer_barrier(encoder);
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            BARRIER_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        } else {
            issue_metal_buffer_barrier(encoder);
        }
    }

    /// Set the compute pipeline state for subsequent dispatches.
    ///
    /// This begins a new compute pass if one is not already active.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipelineStateRef) {
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
    }

    /// Bind a buffer to a compute kernel argument slot.
    ///
    /// The `index` corresponds to the `[[buffer(N)]]` attribute in the MSL shader.
    pub fn set_buffer(&self, index: u64, buffer: &MlxBuffer) {
        let _ = (index, buffer);
    }

    /// Dispatch threads on the GPU.
    pub fn dispatch_threads(&self, grid_size: MTLSize, threadgroup_size: MTLSize) {
        let _ = (grid_size, threadgroup_size);
    }

    /// Encode a complete compute pass: set pipeline, bind buffers, dispatch.
    ///
    /// Reuses the persistent compute encoder — no per-dispatch encoder
    /// creation overhead.
    ///
    /// # Arguments
    ///
    /// * `pipeline`         — The compiled compute pipeline to execute.
    /// * `buffers`          — Slice of `(index, &MlxBuffer)` pairs for buffer bindings.
    /// * `grid_size`        — Total number of threads to launch.
    /// * `threadgroup_size` — Threads per threadgroup.
    pub fn encode(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_buffer_bindings(buffers),
                threads_per_grid: grid_size,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: Vec::new(),
                dispatch_kind: DispatchKind::Threads,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), buf.byte_offset());
        }
        encoder.dispatch_threads(grid_size, threadgroup_size);
    }

    /// Encode a compute pass using threadgroups instead of raw thread counts.
    ///
    /// Reuses the persistent compute encoder — no per-dispatch encoder
    /// creation overhead.
    pub fn encode_threadgroups(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_buffer_bindings(buffers),
                threads_per_grid: threadgroups,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: Vec::new(),
                dispatch_kind: DispatchKind::ThreadGroups,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), buf.byte_offset());
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a compute pass using threadgroups with shared threadgroup memory.
    ///
    /// Like [`encode_threadgroups`](Self::encode_threadgroups), but additionally
    /// allocates threadgroup memory at the specified indices.  This is required
    /// for kernels that use `threadgroup` memory (e.g. reductions in rms_norm
    /// and softmax).
    ///
    /// # Arguments
    ///
    /// * `pipeline`         — The compiled compute pipeline to execute.
    /// * `buffers`          — Slice of `(index, &MlxBuffer)` pairs for buffer bindings.
    /// * `threadgroup_mem`  — Slice of `(index, byte_length)` pairs for threadgroup memory.
    /// * `threadgroups`     — Number of threadgroups to dispatch.
    /// * `threadgroup_size` — Threads per threadgroup.
    pub fn encode_threadgroups_with_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        threadgroup_mem: &[(u64, u64)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_buffer_bindings(buffers),
                threads_per_grid: threadgroups,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: threadgroup_mem.to_vec(),
                dispatch_kind: DispatchKind::ThreadGroups,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), buf.byte_offset());
        }
        for &(index, byte_length) in threadgroup_mem {
            encoder.set_threadgroup_memory_length(index, byte_length);
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings (dispatch_threads).
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_arg_bindings(bindings),
                threads_per_grid: grid_size,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: Vec::new(),
                dispatch_kind: DispatchKind::Threads,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        encoder.dispatch_threads(grid_size, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings (dispatch_thread_groups).
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_threadgroups_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_arg_bindings(bindings),
                threads_per_grid: threadgroups,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: Vec::new(),
                dispatch_kind: DispatchKind::ThreadGroups,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings and shared memory.
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_threadgroups_with_args_and_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        threadgroup_mem: &[(u64, u64)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let op_kind = self.take_pending_op_kind();
        let (pending_reads, pending_writes) = self.take_pending_buffer_ranges();
        if let Some(ref mut nodes) = self.capture {
            nodes.push(CapturedNode::Dispatch {
                pipeline: pipeline.to_owned(),
                bindings: Self::record_arg_bindings(bindings),
                threads_per_grid: threadgroups,
                threads_per_threadgroup: threadgroup_size,
                threadgroup_memory: threadgroup_mem.to_vec(),
                dispatch_kind: DispatchKind::ThreadGroups,
                op_kind,
                reads: pending_reads,
                writes: pending_writes,
            });
            return;
        }
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        for &(index, byte_length) in threadgroup_mem {
            encoder.set_threadgroup_memory_length(index, byte_length);
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    // -----------------------------------------------------------------
    // ADR-015 iter37 — dataflow-driven auto-barrier dispatch family.
    //
    // These mirrors of `encode_threadgroups*_with_args*` take explicit
    // `reads: &[&MlxBuffer]` and `writes: &[&MlxBuffer]` slices.  When
    // the process started with `HF2Q_AUTO_BARRIER=1`, the encoder's
    // [`MemRanges`] tracker checks the new ranges against the
    // cumulative state since the last barrier; on conflict it emits
    // `memory_barrier()` and resets the state before recording the
    // new ranges.  When the env gate is unset, the check is skipped
    // entirely and the dispatch is applied identically to the
    // matching `encode_*` method — sourdough-safe by construction.
    //
    // Capture mode: the `reads`/`writes` ranges are recorded onto the
    // captured node via the existing `pending_reads`/`pending_writes`
    // mechanism, so a `dispatch_tracked` call inside capture mode is
    // equivalent to `set_pending_buffer_ranges + encode_*`.
    //
    // No production callsite migrates in iter37 — this is the API
    // surface the qwen35 forward path will adopt incrementally in
    // iter38+.  Today, every call to `dispatch_tracked` from a
    // production code path lives behind an explicit caller decision
    // to opt in.
    // -----------------------------------------------------------------

    /// Auto-barrier-aware dispatch with [`KernelArg`] bindings (uses
    /// `dispatch_thread_groups`).
    ///
    /// Behaves identically to
    /// [`encode_threadgroups_with_args`](Self::encode_threadgroups_with_args)
    /// when `HF2Q_AUTO_BARRIER` is unset.  When set, consults the
    /// per-encoder [`MemRanges`] tracker:
    ///
    /// * Conflict (RAW/WAR/WAW on a same-buffer range) → emit
    ///   `memory_barrier()`, increment [`AUTO_BARRIER_COUNT`], reset
    ///   the tracker, then dispatch and seed the new concurrent group
    ///   with this dispatch's ranges.
    /// * No conflict → increment [`AUTO_BARRIER_CONCURRENT`], record
    ///   the ranges into the cumulative state, dispatch.
    pub fn dispatch_tracked_threadgroups_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        // Capture mode: stash ranges + delegate to the standard encode.
        // The ranges flow through `pending_reads`/`pending_writes` and
        // attach to the captured `Dispatch` node — identical to what
        // `GraphSession::barrier_between` already does in capture mode.
        if self.is_capturing() {
            let read_ranges = ranges_from_buffers(reads);
            let write_ranges = ranges_from_buffers(writes);
            self.set_pending_buffer_ranges(read_ranges, write_ranges);
            self.encode_threadgroups_with_args(pipeline, bindings, threadgroups, threadgroup_size);
            return;
        }

        if auto_barrier_enabled() {
            self.maybe_auto_barrier(reads, writes);
        }

        self.encode_threadgroups_with_args(pipeline, bindings, threadgroups, threadgroup_size);
    }

    /// Auto-barrier-aware dispatch with [`KernelArg`] bindings + shared
    /// threadgroup memory.
    ///
    /// See [`dispatch_tracked_threadgroups_with_args`](Self::dispatch_tracked_threadgroups_with_args)
    /// for the behavioral contract; this variant additionally takes a
    /// `threadgroup_mem` slice that is forwarded to
    /// [`encode_threadgroups_with_args_and_shared`](Self::encode_threadgroups_with_args_and_shared).
    ///
    /// The 8-argument signature mirrors the existing
    /// `encode_threadgroups_with_args_and_shared` plus the two
    /// dataflow slices; `clippy::too_many_arguments` is allowed
    /// because each parameter is load-bearing for either the dispatch
    /// (pipeline/bindings/threadgroups/threadgroup_size/shared_mem)
    /// or the auto-barrier (reads/writes).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_tracked_threadgroups_with_args_and_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        threadgroup_mem: &[(u64, u64)],
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        if self.is_capturing() {
            let read_ranges = ranges_from_buffers(reads);
            let write_ranges = ranges_from_buffers(writes);
            self.set_pending_buffer_ranges(read_ranges, write_ranges);
            self.encode_threadgroups_with_args_and_shared(
                pipeline,
                bindings,
                threadgroup_mem,
                threadgroups,
                threadgroup_size,
            );
            return;
        }

        if auto_barrier_enabled() {
            self.maybe_auto_barrier(reads, writes);
        }

        self.encode_threadgroups_with_args_and_shared(
            pipeline,
            bindings,
            threadgroup_mem,
            threadgroups,
            threadgroup_size,
        );
    }

    /// Auto-barrier-aware dispatch using `(slot, &MlxBuffer)` bindings
    /// (uses `dispatch_thread_groups`).
    ///
    /// Convenience wrapper for callers that don't need
    /// [`KernelArg::Bytes`] inline-byte arguments.  See
    /// [`dispatch_tracked_threadgroups_with_args`](Self::dispatch_tracked_threadgroups_with_args)
    /// for behavioral contract.
    pub fn dispatch_tracked_threadgroups(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        if self.is_capturing() {
            let read_ranges = ranges_from_buffers(reads);
            let write_ranges = ranges_from_buffers(writes);
            self.set_pending_buffer_ranges(read_ranges, write_ranges);
            self.encode_threadgroups(pipeline, buffers, threadgroups, threadgroup_size);
            return;
        }

        if auto_barrier_enabled() {
            self.maybe_auto_barrier(reads, writes);
        }

        self.encode_threadgroups(pipeline, buffers, threadgroups, threadgroup_size);
    }

    /// Auto-barrier-aware dispatch using `(slot, &MlxBuffer)` bindings
    /// **plus shared threadgroup memory** (uses `dispatch_thread_groups`).
    ///
    /// Mirrors [`encode_threadgroups_with_shared`](Self::encode_threadgroups_with_shared)
    /// — convenience variant for kernels that allocate threadgroup
    /// memory (reductions in `rms_norm`, `softmax`, etc.) but don't
    /// need [`KernelArg::Bytes`] inline-byte arguments.  See
    /// [`dispatch_tracked_threadgroups_with_args`](Self::dispatch_tracked_threadgroups_with_args)
    /// for the behavioral contract; the only addition here is the
    /// `threadgroup_mem` slice forwarded to the underlying encode.
    ///
    /// Closes the iter38-audit coverage gap: the 5 `rms_norm.rs`
    /// callsites (`/opt/mlx-native/src/ops/rms_norm.rs:124,236,443,
    /// 516,589`) all use `encode_threadgroups_with_shared` and need
    /// dataflow tracking when migrated to auto-barrier in iter40+.
    ///
    /// 7-argument signature; `clippy::too_many_arguments` is allowed
    /// because each parameter is load-bearing for either the dispatch
    /// (pipeline/buffers/threadgroups/threadgroup_size/shared_mem) or
    /// the auto-barrier (reads/writes).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_tracked_threadgroups_with_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        threadgroup_mem: &[(u64, u64)],
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        if self.is_capturing() {
            let read_ranges = ranges_from_buffers(reads);
            let write_ranges = ranges_from_buffers(writes);
            self.set_pending_buffer_ranges(read_ranges, write_ranges);
            self.encode_threadgroups_with_shared(
                pipeline,
                buffers,
                threadgroup_mem,
                threadgroups,
                threadgroup_size,
            );
            return;
        }

        if auto_barrier_enabled() {
            self.maybe_auto_barrier(reads, writes);
        }

        self.encode_threadgroups_with_shared(
            pipeline,
            buffers,
            threadgroup_mem,
            threadgroups,
            threadgroup_size,
        );
    }

    /// Auto-barrier-aware `dispatch_threads` variant with
    /// [`KernelArg`] bindings.
    ///
    /// Mirrors [`encode_with_args`](Self::encode_with_args) — the
    /// `dispatch_threads` (per-thread grid) flavor, as opposed to the
    /// `dispatch_thread_groups` flavor of
    /// [`dispatch_tracked_threadgroups_with_args`](Self::dispatch_tracked_threadgroups_with_args).
    /// See that method for the behavioral contract.
    ///
    /// Closes the iter38-audit coverage gap: callers that use
    /// per-thread grids — `rope.rs:108` (IMROPE), `sigmoid_mul.rs:76`
    /// (sigmoid-mul), and `encode_helpers.rs:41` (kv_cache_copy) —
    /// need a `dispatch_threads` flavor of the tracked dispatch
    /// because their grid sizes are expressed in threads, not
    /// threadgroups.
    ///
    /// Note: the simpler `(slot, &MlxBuffer)` form (from
    /// [`encode`](Self::encode)) is a special case of this method —
    /// callers can wrap each binding as `KernelArg::Buffer(buf)` to
    /// reuse this single tracked variant rather than introducing a
    /// fifth one.
    pub fn dispatch_tracked_threads_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        if self.is_capturing() {
            let read_ranges = ranges_from_buffers(reads);
            let write_ranges = ranges_from_buffers(writes);
            self.set_pending_buffer_ranges(read_ranges, write_ranges);
            self.encode_with_args(pipeline, bindings, grid_size, threadgroup_size);
            return;
        }

        if auto_barrier_enabled() {
            self.maybe_auto_barrier(reads, writes);
        }

        self.encode_with_args(pipeline, bindings, grid_size, threadgroup_size);
    }

    /// Run the dataflow check, emit a barrier on conflict, and record
    /// the dispatch's ranges into the cumulative state.
    ///
    /// Always called *before* the underlying `encode_*` method
    /// applies the dispatch.  Mirrors lines 220-225 of
    /// `ggml-metal-ops.cpp` (`concurrency_check + concurrency_reset +
    /// concurrency_add` around each node).
    fn maybe_auto_barrier(
        &mut self,
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
    ) {
        if self.mem_ranges.check_dispatch(reads, writes) {
            // Concurrent — no barrier needed; just record the new ranges.
            self.mem_ranges.add_dispatch(reads, writes);
            AUTO_BARRIER_CONCURRENT.fetch_add(1, Ordering::Relaxed);
        } else {
            // Conflict — emit barrier, reset state, seed new group.
            //
            // `memory_barrier()` itself increments `BARRIER_COUNT` and,
            // when `MLX_PROFILE_BARRIERS=1`, accumulates `BARRIER_NS`.
            // We additionally bump `AUTO_BARRIER_COUNT` so the
            // "auto-emitted vs hand-placed" subset is queryable.
            self.memory_barrier();
            self.mem_ranges.reset();
            self.mem_ranges.add_dispatch(reads, writes);
            AUTO_BARRIER_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Force a barrier and reset the auto-barrier tracker.
    ///
    /// Use at boundaries where the caller knows a barrier is required
    /// regardless of dataflow — typically before reading data back to
    /// CPU, or at the end of an op group whose internal dependencies
    /// the tracker can't see (e.g. host-driven memcpy).
    ///
    /// Equivalent to `memory_barrier()` plus a `MemRanges::reset()`
    /// when `HF2Q_AUTO_BARRIER=1`; equivalent to plain
    /// `memory_barrier()` otherwise.
    pub fn force_barrier_and_reset_tracker(&mut self) {
        self.memory_barrier();
        if auto_barrier_enabled() {
            self.mem_ranges.reset();
        }
    }

    /// Diagnostic accessor — number of ranges currently recorded in
    /// this encoder's [`MemRanges`] tracker.  Always zero unless
    /// `HF2Q_AUTO_BARRIER=1` and at least one `dispatch_tracked` call
    /// has fired since the last conflict.
    #[inline]
    pub fn mem_ranges_len(&self) -> usize {
        self.mem_ranges.len()
    }

    /// Replay a single captured dispatch node into this encoder.
    ///
    /// This is the inverse of capture: it takes a previously recorded
    /// `CapturedNode::Dispatch` and encodes it into the live Metal encoder.
    /// Barrier nodes are handled by the caller (ComputeGraph::encode_sequential).
    ///
    /// Does NOT increment `DISPATCH_COUNT` — that was already counted at
    /// capture time.
    pub fn replay_dispatch(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, RecordedBinding)],
        threadgroup_memory: &[(u64, u64)],
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
        dispatch_kind: DispatchKind,
    ) {
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, binding) in bindings {
            match binding {
                RecordedBinding::Buffer { metal_buffer, offset } => {
                    encoder.set_buffer(*index, Some(metal_buffer), *offset);
                }
                RecordedBinding::Bytes(bytes) => {
                    encoder.set_bytes(
                        *index,
                        bytes.len() as u64,
                        bytes.as_ptr() as *const _,
                    );
                }
            }
        }
        for &(index, byte_length) in threadgroup_memory {
            encoder.set_threadgroup_memory_length(index, byte_length);
        }
        match dispatch_kind {
            DispatchKind::Threads => {
                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            }
            DispatchKind::ThreadGroups => {
                encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
            }
        }
    }

    /// Flush any pending residency-set add/remove staging.
    ///
    /// Hooked at every commit boundary so per-allocation
    /// [`ResidencySet::add_allocation`](ResidencySet::add_allocation) and
    /// [`ResidencySet::remove_allocation`](ResidencySet::remove_allocation)
    /// calls (as fired by `MlxDevice::alloc_buffer` and
    /// `MlxBufferStorage::Drop`) collapse into at most ONE `[set commit]`
    /// per CB submission. Mirrors llama.cpp's
    /// `ggml-metal-device.m:1378-1382` (batch addAllocation in loop,
    /// commit ONCE).
    #[inline]
    fn flush_residency_pending(&self) {
        if let Some(set) = self.residency_set.as_ref() {
            set.flush_pending();
        }
    }

    /// Commit the command buffer and block until the GPU finishes execution.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn commit_and_wait(&mut self) -> Result<()> {
        SYNC_COUNT.fetch_add(1, Ordering::Relaxed);

        // End the persistent compute encoder before committing.
        self.end_active_encoder();

        // ADR-015 iter8e (Phase 3b): flush deferred residency-set
        // add/remove staging so the residency hint covers any buffers
        // referenced by this CB. Single commit per CB boundary; no-op
        // when no residency set or no staged changes.
        self.flush_residency_pending();

        self.cmd_buf.commit();
        self.cmd_buf.wait_until_completed();

        match self.cmd_buf.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            MTLCommandBufferStatus::Error => {
                Err(MlxError::CommandBufferError(
                    "GPU command buffer completed with error status".into(),
                ))
            }
            status => Err(MlxError::CommandBufferError(format!(
                "Unexpected command buffer status after wait: {:?}",
                status
            ))),
        }
    }

    /// Commit + wait, accumulating GPU wall-clock time under `label` into
    /// the [`crate::kernel_profile`] global table when `MLX_PROFILE_CB=1`
    /// is set.  When the env var is unset, this is identical to
    /// [`commit_and_wait`](Self::commit_and_wait) — zero overhead.
    ///
    /// Used by hf2q's decode hot path to attribute per-cb GPU time to
    /// labeled phases (per-layer attn, per-layer ffn, output_head, etc.)
    /// without manually wiring `commit_wait_with_gpu_time` everywhere.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn commit_and_wait_labeled(&mut self, label: &str) -> Result<()> {
        // ADR-015 iter16 — propagate `label` to MTLCommandBuffer.setLabel and
        // (if a compute encoder is active) MTLComputeCommandEncoder.setLabel
        // BEFORE end_encoding/commit so xctrace's
        // `metal-application-encoders-list` table populates `cmdbuffer-label`
        // and `encoder-label` columns with the semantic phase name (e.g.
        // `layer.attn_moe_ffn`, `output_head.fused_norm_lm_argmax`,
        // `layer.delta_net.ops1-9`).  Joined to per-CB GPU duration via
        // `metal-gpu-submission-to-command-buffer-id` (sub_id ↔ encoder_id) →
        // `metal-gpu-execution-points` (per-dispatch start/end), this enables
        // per-phase µs/token attribution comparing hf2q vs llama side-by-side
        // (iter15 §E "iter16 ATTRIBUTION PATH").  Cost is a single ObjC
        // msg_send per CB submission — sub-µs on M5 Max — and a no-op when
        // xctrace isn't recording, so this is unconditionally safe to call on
        // the production decode hot path.
        self.apply_labels(label);
        if crate::kernel_profile::is_enabled() {
            let (start_s, end_s) = self.commit_wait_with_gpu_time()?;
            let ns = ((end_s - start_s).max(0.0) * 1_000_000_000.0) as u64;
            crate::kernel_profile::record(label, ns);
            Ok(())
        } else {
            self.commit_and_wait()
        }
    }

    /// Async commit, but with profiling label.  When `MLX_PROFILE_CB=1`
    /// is set, redirects to a synchronous [`commit_and_wait_labeled`]
    /// call to capture per-cb GPU time (this defeats async pipelining
    /// while profiling, which is the whole point — profile-mode is slow
    /// but informative).  When unset, identical to [`commit`](Self::commit).
    pub fn commit_labeled(&mut self, label: &str) {
        // ADR-015 iter16 — see `commit_and_wait_labeled` for rationale.
        if crate::kernel_profile::is_enabled() {
            // Profile mode: force sync to capture GPU time.  apply_labels is
            // called inside commit_and_wait_labeled — do NOT call it twice
            // here (would double the ObjC msg_send under MLX_PROFILE_CB=1).
            // Errors are logged via stderr because the void return matches
            // commit().
            if let Err(e) = self.commit_and_wait_labeled(label) {
                eprintln!("[mlx-native] commit_labeled({}) failed: {}", label, e);
            }
        } else {
            // Async path: apply labels here so xctrace MST traces capture
            // per-CB phase attribution under default decode (no
            // `MLX_PROFILE_CB`).
            self.apply_labels(label);
            self.commit();
        }
    }

    /// Apply `label` to the underlying `MTLCommandBuffer` and, if a compute
    /// encoder is currently active, to the `MTLComputeCommandEncoder`.
    ///
    /// Called from [`commit_labeled`] and [`commit_and_wait_labeled`] BEFORE
    /// the encoder is ended / the CB is committed so xctrace's
    /// `metal-application-encoders-list` table picks up the label on the
    /// row emitted at the encoder's `endEncoding` / CB submission boundary.
    /// Single ObjC `msg_send` per call (two if an encoder is active); sub-µs
    /// on M5 Max; no-op when xctrace isn't recording.
    ///
    /// Skipped (debug-only assert) if `label` is empty — empty labels would
    /// produce an indistinguishable trace row from the metal-rs default
    /// `Command Buffer 0` placeholder.
    #[inline]
    fn apply_labels(&self, label: &str) {
        debug_assert!(!label.is_empty(), "commit_*_labeled called with empty label");
        if label.is_empty() {
            return;
        }
        self.cmd_buf.set_label(label);
        if !self.active_encoder.is_null() {
            // SAFETY: active_encoder is non-null and points to a live encoder
            // owned by cmd_buf — same invariant as get_or_create_encoder /
            // memory_barrier.  set_label is a single property write on the
            // ObjC object; safe before endEncoding.
            unsafe { &*self.active_encoder }.set_label(label);
        }
    }

    /// Commit + wait, returning `(gpu_start_s, gpu_end_s)` CFTimeInterval
    /// timestamps from `MTLCommandBuffer`'s `GPUStartTime`/`GPUEndTime`
    /// properties.  Both are mach-absolute CFTimeInterval seconds (double).
    ///
    /// Intended for `HF2Q_PROFILE_GPU_TS=1` per-bucket GPU wall-clock
    /// attribution.  Adds exactly two ObjC property reads per call on top
    /// of the regular `commit_and_wait` — measured well under 1 μs on
    /// M5 Max.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn commit_wait_with_gpu_time(&mut self) -> Result<(f64, f64)> {
        self.commit_and_wait()?;
        // SAFETY: cmd_buf is a valid MTLCommandBuffer that has been
        // committed and awaited.  GPUStartTime / GPUEndTime return
        // CFTimeInterval (double precision seconds).  See
        // https://developer.apple.com/documentation/metal/mtlcommandbuffer/1639925-gpustarttime
        let (gpu_start, gpu_end): (f64, f64) = unsafe {
            let cb = &*self.cmd_buf;
            let s: f64 = msg_send![cb, GPUStartTime];
            let e: f64 = msg_send![cb, GPUEndTime];
            (s, e)
        };
        Ok((gpu_start, gpu_end))
    }

    /// Commit the command buffer WITHOUT blocking.
    ///
    /// The GPU begins executing the encoded commands immediately.  Call
    /// [`wait_until_completed`](Self::wait_until_completed) later to block
    /// the CPU and check for errors.  This allows the CPU to continue doing
    /// other work (e.g. preparing the next batch) while the GPU runs.
    pub fn commit(&mut self) {
        self.end_active_encoder();
        // ADR-015 iter8e (Phase 3b): same flush hook as commit_and_wait —
        // this is the async-pipeline path that production decode uses.
        self.flush_residency_pending();
        self.cmd_buf.commit();
    }

    /// Block until a previously committed command buffer completes.
    ///
    /// Must be called after [`commit`](Self::commit).  Do not call after
    /// [`commit_and_wait`](Self::commit_and_wait) — that method already waits.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn wait_until_completed(&self) -> Result<()> {
        self.cmd_buf.wait_until_completed();
        match self.cmd_buf.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            MTLCommandBufferStatus::Error => Err(MlxError::CommandBufferError(
                "GPU command buffer completed with error status".into(),
            )),
            status => Err(MlxError::CommandBufferError(format!(
                "Unexpected command buffer status after wait: {:?}",
                status
            ))),
        }
    }

    /// Borrow the underlying Metal command buffer.
    #[inline]
    pub fn metal_command_buffer(&self) -> &CommandBuffer {
        &self.cmd_buf
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        // End the persistent compute encoder before the command buffer
        // is dropped, otherwise Metal will assert:
        // "Command encoder released without endEncoding"
        self.end_active_encoder();
    }
}
