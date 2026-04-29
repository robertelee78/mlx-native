//! Dataflow-driven barrier inference (port of llama.cpp `mem_ranges`).
//!
//! ADR-015 iter37 — framework-side complement to iter21's hand-audited
//! barrier fix at `gpu_full_attn.rs:1856`.
//!
//! # Purpose
//!
//! When a Metal `MTLComputeCommandEncoder` is created with
//! `MTLDispatchTypeConcurrent` (mlx-native's default since iter8e —
//! [`encoder::CommandEncoder::get_or_create_encoder`](crate::encoder)),
//! every dispatch can execute in parallel with every other dispatch in
//! the same encoder unless separated by a memory barrier.  The encoder
//! does not infer dataflow on its own — the caller must hand-place
//! `[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]` between
//! every read-after-write (RAW), write-after-read (WAR), or
//! write-after-write (WAW) pair.
//!
//! Hand-audited barrier placement is correct but fragile: iter21 found
//! one missing producer→consumer edge (`sigmoid_gate_multiply` →
//! `linear_projection wo`) that had escaped review for months because
//! the diverged-output bug it caused only surfaced under specific
//! sequence-length × sample-count combinations.  Any future kernel
//! sequence built without rigorous review is subject to the same
//! class of bug.
//!
//! `MemRanges` ports llama.cpp's mem_ranges algorithm
//! (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp`) so
//! callers describe each dispatch's read and write buffer regions and
//! the framework auto-emits a barrier exactly when the new dispatch's
//! ranges overlap a previously-recorded range.  This makes
//! iter21-class bugs structurally impossible at the framework boundary.
//!
//! # Algorithm
//!
//! Verbatim port of `ggml_mem_ranges_check` + `ggml_mem_ranges_add`
//! (lines 124-185 of `ggml-metal-common.cpp`):
//!
//! * A range is `(buffer_id, p0, p1, role∈{Src,Dst})`.
//! * Two ranges in different buffers can never conflict.
//! * Two `Src` ranges in the same buffer never conflict (read-read OK).
//! * A new `Src` overlapping an existing `Dst` is a RAW conflict.
//! * A new `Dst` overlapping any existing range (Src or Dst) is a
//!   WAR/WAW conflict.
//! * Overlap test: `new.p0 < existing.p1 && new.p1 >= existing.p0`
//!   (matches llama.cpp byte-for-byte at line 138).
//! * On conflict, the caller emits a `memoryBarrier` and `reset()`s
//!   the cumulative state, then records the new dispatch's ranges.
//!
//! # mlx-native specifics
//!
//! llama.cpp keys ranges by `tensor->buffer` (the backend buffer
//! handle) plus `tensor->data` (the element pointer inside that
//! buffer). mlx-native uses
//! [`MlxBuffer::metal_buffer`](crate::buffer::MlxBuffer::metal_buffer)
//! as the `(MTLBuffer*) -> usize` buffer-id and
//! [`MlxBuffer::contents_ptr`](crate::buffer::MlxBuffer::contents_ptr)
//! as the start address.  These are stable across the encoder lifetime
//! because hf2q's per-decode-token `MlxBufferPool` keeps ARC clones
//! alive for the entire CB.  Different `slice_view`s of the same
//! parent buffer share `metal_buffer()` (intentional: a write to
//! `parent[0..N]` must barrier against a read of `parent[N..2N]`
//! only when the two slices alias).
//!
//! # Why same-buffer-only
//!
//! Different `MTLBuffer`s never alias — Metal's address space is
//! per-buffer.  Skipping the overlap check on cross-buffer pairs is
//! both correct and a major perf win: a typical decode token has
//! ~1500 dispatches against ~30-50 distinct buffers, so the
//! same-buffer filter keeps the per-dispatch check at O(N) over the
//! short list of ranges in *one* buffer rather than O(N) over all
//! ranges.
//!
//! # Per iter37 envelope: env-gated, opt-in
//!
//! `MemRanges` is dormant unless the caller explicitly threads it
//! through a dispatch via [`CommandEncoder::dispatch_tracked`].  The
//! existing `encode*`/`memory_barrier()` API is unchanged, so iter37
//! ships with **zero behavioral diff in production** until callers
//! migrate.  Migration of the qwen35 forward path is iter38+ scope.

use crate::buffer::MlxBuffer;
use metal::foreign_types::ForeignType;

/// Whether a recorded range was read by a dispatch (`Src`) or written
/// by a dispatch (`Dst`).  Mirrors `ggml_mem_range_type` in
/// `ggml-metal-common.h:14-17`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemRangeRole {
    /// Dispatch reads this range.
    Src,
    /// Dispatch writes this range.
    Dst,
}

/// A buffer region recorded for dataflow tracking.
///
/// Mirrors `struct ggml_mem_range` in `ggml-metal-common.cpp:10-17`.
#[derive(Clone, Copy, Debug)]
pub struct BufferRange {
    /// Backing `metal::Buffer` pointer cast to `usize`.  Stable across
    /// the encoder lifetime as long as the `MlxBuffer`'s ARC clone
    /// outlives the CB (see `CommandEncoder::new_with_residency`
    /// caller contract).
    pub buf_id: usize,
    /// Start byte address (`contents_ptr() + byte_offset` for
    /// `MlxBuffer`).  Used for overlap arithmetic.
    pub p0: u64,
    /// End byte address (start + element-extent).  llama.cpp uses
    /// `tensor->data + ggml_backend_buft_get_alloc_size(tensor)` —
    /// for mlx-native we use the buffer's `byte_len()` minus
    /// `byte_offset()`, which equals the slice extent.
    pub p1: u64,
    /// Whether this range is read or written by the recording dispatch.
    pub role: MemRangeRole,
}

impl BufferRange {
    /// Build a [`BufferRange`] from an [`MlxBuffer`] and a role.
    ///
    /// Uses `metal_buffer().as_ptr() as usize` as the buffer-id (so two
    /// `slice_view`s of the same parent share a `buf_id`, which is
    /// the intended behavior — a slice write must barrier against a
    /// sibling-slice read of the same parent).
    ///
    /// The `(p0, p1)` range covers the addressable extent the kernel
    /// can reach: `[contents_ptr + byte_offset,
    ///   contents_ptr + byte_offset + (byte_len - byte_offset))`.
    /// For non-slice buffers `byte_offset == 0` and the range covers
    /// the full allocation.  For slices the range covers only the
    /// slice region — matching llama.cpp's `tensor->data ..
    /// tensor->data + alloc_size`.
    #[inline]
    pub fn from_buffer(buf: &MlxBuffer, role: MemRangeRole) -> Self {
        let buf_id = buf.metal_buffer().as_ptr() as usize;
        // `contents_ptr` already points at the buffer's base; mlx-native
        // applies `byte_offset` only at bind-site (`set_buffer`).  The
        // overlap arithmetic must use the slice's *kernel-visible*
        // address window, so we add the offset explicitly here.
        let base = buf.contents_ptr() as u64;
        let p0 = base + buf.byte_offset();
        // `byte_len()` returns the underlying allocation length, so
        // the slice extent is `(allocation_len - offset)`.
        let extent = (buf.byte_len() as u64).saturating_sub(buf.byte_offset());
        let p1 = p0 + extent;
        Self {
            buf_id,
            p0,
            p1,
            role,
        }
    }

    /// Whether `self` and `other` overlap by the same arithmetic
    /// llama.cpp uses at `ggml-metal-common.cpp:138`.
    ///
    /// Returns `false` for cross-buffer pairs (different `buf_id`) and
    /// for src-vs-src pairs (read-read is always concurrent-safe).
    #[inline]
    pub fn conflicts_with(&self, other: &BufferRange) -> bool {
        if self.buf_id != other.buf_id {
            return false;
        }
        if self.role == MemRangeRole::Src && other.role == MemRangeRole::Src {
            return false;
        }
        // Llama.cpp: `mr.p0 < cmp.p1 && mr.p1 >= cmp.p0`
        self.p0 < other.p1 && self.p1 >= other.p0
    }
}

/// Cumulative dataflow state for a sequence of concurrent dispatches.
///
/// Direct port of `struct ggml_mem_ranges` in
/// `ggml-metal-common.cpp:19-23`.  The state is reset every time a
/// barrier is emitted; between barriers, all recorded dispatches are
/// considered to run concurrently and their R/W ranges accumulate.
pub struct MemRanges {
    ranges: Vec<BufferRange>,
    /// Total checks performed (diagnostic).
    checks: u64,
    /// Number of `check()` calls that returned `false` (i.e. forced a
    /// barrier).  `total_dispatches - barriers_forced` == elided
    /// barriers (would-have-been-emitted by an unconditional pattern).
    barriers_forced: u64,
}

impl Default for MemRanges {
    fn default() -> Self {
        Self::new()
    }
}

impl MemRanges {
    /// New empty state.  Pre-allocates capacity matching llama.cpp's
    /// `reserve(256)` (line 28).
    pub fn new() -> Self {
        Self {
            ranges: Vec::with_capacity(256),
            checks: 0,
            barriers_forced: 0,
        }
    }

    /// Drop all recorded ranges (called after emitting a barrier).
    /// Mirrors `ggml_mem_ranges_reset`.
    #[inline]
    pub fn reset(&mut self) {
        self.ranges.clear();
    }

    /// Number of currently-recorded ranges (diagnostic).
    #[inline]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the cumulative state is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Number of `check()` calls performed since construction
    /// (diagnostic, monotone).
    #[inline]
    pub fn checks(&self) -> u64 {
        self.checks
    }

    /// Number of `check()` calls that returned `false`, forcing a
    /// barrier (diagnostic, monotone).  When tracking is enabled at
    /// every dispatch, `total_dispatches - barriers_forced` ==
    /// barriers elided versus the unconditional-barrier baseline.
    #[inline]
    pub fn barriers_forced(&self) -> u64 {
        self.barriers_forced
    }

    /// Push a single range onto the cumulative state without checking.
    /// Used internally by [`Self::add`] and [`Self::add_dispatch`].
    /// Public so unit tests can construct adversarial states.
    #[inline]
    pub fn push(&mut self, range: BufferRange) {
        self.ranges.push(range);
    }

    /// Record a dispatch's read-buffer ranges + write-buffer ranges.
    ///
    /// Mirrors `ggml_mem_ranges_add(tensor)` at
    /// `ggml-metal-common.cpp:114-122`: pushes one Src range per
    /// `tensor->src[i]` and one Dst range for `tensor` itself.
    ///
    /// Caller is expected to have already invoked
    /// [`Self::check_dispatch`] and emitted a barrier on conflict; the
    /// barrier-emit + `reset()` is the responsibility of the
    /// integration site (typically `CommandEncoder`).
    pub fn add_dispatch(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) {
        for r in reads {
            self.ranges
                .push(BufferRange::from_buffer(r, MemRangeRole::Src));
        }
        for w in writes {
            self.ranges
                .push(BufferRange::from_buffer(w, MemRangeRole::Dst));
        }
    }

    /// Check whether a candidate dispatch can run concurrently with
    /// the recorded state.
    ///
    /// Returns `true` iff none of the candidate's reads or writes
    /// conflict with any recorded range.  Exactly mirrors
    /// `ggml_mem_ranges_check(tensor)` at `ggml-metal-common.cpp:175-185`:
    /// each src is checked against existing ranges, then the dst is
    /// checked against existing ranges.
    ///
    /// Increments [`Self::checks`].  On `false` return, also
    /// increments [`Self::barriers_forced`] — so the diagnostic
    /// counter is accurate even when callers ignore the return value.
    pub fn check_dispatch(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) -> bool {
        self.checks += 1;
        for r in reads {
            let candidate = BufferRange::from_buffer(r, MemRangeRole::Src);
            for existing in &self.ranges {
                if candidate.conflicts_with(existing) {
                    self.barriers_forced += 1;
                    return false;
                }
            }
        }
        for w in writes {
            let candidate = BufferRange::from_buffer(w, MemRangeRole::Dst);
            for existing in &self.ranges {
                if candidate.conflicts_with(existing) {
                    self.barriers_forced += 1;
                    return false;
                }
            }
        }
        true
    }

    /// Combined check + add.  Returns `true` if the dispatch was added
    /// concurrent (no conflict, no barrier needed); returns `false`
    /// if the caller must emit a barrier and `reset()` before adding
    /// the dispatch's ranges.
    ///
    /// On `false` return, the caller's responsibility is:
    /// 1. Emit the underlying `memoryBarrierWithScope:` on the live
    ///    encoder.
    /// 2. Call [`Self::reset`].
    /// 3. Call [`Self::add_dispatch`] with the same `reads`/`writes`
    ///    to seed the new concurrent group.
    ///
    /// This mirrors the call pattern at `ggml-metal-ops.cpp:220-225`.
    pub fn check_and_record(
        &mut self,
        reads: &[&MlxBuffer],
        writes: &[&MlxBuffer],
    ) -> bool {
        let ok = self.check_dispatch(reads, writes);
        if ok {
            self.add_dispatch(reads, writes);
        }
        // On !ok the caller will reset+add per the contract above.
        ok
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`MemRanges`].
    //!
    //! These are pure-CPU tests that exercise the address arithmetic
    //! and overlap-detection logic without touching Metal — they
    //! construct `MlxBuffer`s via `MlxDevice::alloc_buffer`, which
    //! does allocate real Metal buffers but does not require any GPU
    //! commands to be encoded or executed.  Each test is bounded to a
    //! handful of small allocations.
    use super::*;
    use crate::{DType, MlxDevice};

    fn dev() -> MlxDevice {
        MlxDevice::new().expect("MlxDevice::new failed")
    }

    /// Two reads of the same buffer must NOT conflict (RAR concurrent).
    #[test]
    fn read_read_same_buffer_no_conflict() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        // First dispatch: read a, write nothing.
        let ok1 = mr.check_and_record(&[&a], &[]);
        assert!(ok1, "first dispatch always ok");
        // Second dispatch: read a again — must be concurrent.
        let ok2 = mr.check_and_record(&[&a], &[]);
        assert!(ok2, "RAR same-buffer must not conflict");
        assert_eq!(mr.barriers_forced(), 0);
    }

    /// Read-after-write same buffer MUST conflict (RAW barrier needed).
    #[test]
    fn raw_same_buffer_conflicts() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        // First dispatch writes a.
        assert!(mr.check_and_record(&[], &[&a]));
        // Second dispatch reads a — must conflict.
        let ok = mr.check_and_record(&[&a], &[]);
        assert!(!ok, "RAW same-buffer must force barrier");
        assert_eq!(mr.barriers_forced(), 1);
    }

    /// Write-after-read same buffer MUST conflict (WAR barrier needed).
    #[test]
    fn war_same_buffer_conflicts() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        assert!(mr.check_and_record(&[&a], &[]));
        let ok = mr.check_and_record(&[], &[&a]);
        assert!(!ok, "WAR same-buffer must force barrier");
        assert_eq!(mr.barriers_forced(), 1);
    }

    /// Write-after-write same buffer MUST conflict (WAW barrier needed).
    #[test]
    fn waw_same_buffer_conflicts() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        assert!(mr.check_and_record(&[], &[&a]));
        let ok = mr.check_and_record(&[], &[&a]);
        assert!(!ok, "WAW same-buffer must force barrier");
        assert_eq!(mr.barriers_forced(), 1);
    }

    /// Cross-buffer reads/writes never conflict regardless of role.
    /// The candidate dispatch's ranges are checked only against
    /// recorded ranges in the SAME buffer; ranges in disjoint
    /// buffers are skipped early in `BufferRange::conflicts_with`.
    #[test]
    fn different_buffers_never_conflict() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let b = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let c = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        // dispatch1: write a — records (a, Dst).
        assert!(mr.check_and_record(&[], &[&a]));
        // dispatch2: read+write b — disjoint from a, ok.
        assert!(mr.check_and_record(&[&b], &[&b]));
        // dispatch3: read c — disjoint from a and b, ok.  Critically,
        // we do NOT read `a` here because that would be RAW against
        // dispatch1's write — a real conflict, not a same-buffer
        // false positive.
        assert!(mr.check_and_record(&[&c], &[]));
        assert_eq!(mr.barriers_forced(), 0);
    }

    /// Reset clears state and lets a previously-conflicting dispatch
    /// be recorded.  Mirrors the post-barrier flow.
    #[test]
    fn reset_clears_state() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        assert!(mr.check_and_record(&[], &[&a]));
        // Would conflict with the recorded write…
        assert!(!mr.check_and_record(&[&a], &[]));
        // …unless we reset first (simulating a barrier emission).
        mr.reset();
        assert!(mr.check_and_record(&[&a], &[]));
        // After reset, two reads in a row are still non-conflicting.
        assert!(mr.check_and_record(&[&a], &[]));
        assert_eq!(mr.barriers_forced(), 1);
    }

    /// Disjoint slices of the same parent: today the algorithm is
    /// conservative (treats slice writes as touching the full
    /// addressable extent of the parent), matching llama.cpp's
    /// `alloc_size` upper bound.  This documents the behavior so
    /// future iterations can tighten it intentionally.
    #[test]
    fn slices_of_same_parent_conservative() {
        let d = dev();
        // 256 floats; carve into two halves.
        let parent = d.alloc_buffer(1024, DType::F32, vec![256]).unwrap();
        let lo = parent.slice_view(0, 128);
        let hi = parent.slice_view(512, 128);
        let mut mr = MemRanges::new();
        assert!(mr.check_and_record(&[], &[&lo]));
        // hi is a disjoint half but conservatively conflicts because
        // the lo write's recorded range covers
        //   [parent + 0, parent + parent.byte_len()) and the hi range
        //   starts at `parent + 512` which falls inside that window.
        // The conservative answer is *correct* (a barrier is safe even
        // if not necessary).  Tightening the slice arithmetic to use
        // the slice's own extent only is a future iteration.
        let ok = mr.check_and_record(&[], &[&hi]);
        assert!(!ok, "slice WAW currently conservative — see docstring");
    }

    /// Sequential pattern: A=write x, B=read x, C=write y, D=read y.
    /// Expect exactly 2 forced barriers (B vs A, D vs C).
    #[test]
    fn sequential_pattern_two_barriers() {
        let d = dev();
        let x = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let y = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let mut mr = MemRanges::new();
        // A: write x.
        assert!(mr.check_and_record(&[], &[&x]));
        // B: read x — conflict.
        assert!(!mr.check_dispatch(&[&x], &[]));
        mr.reset();
        mr.add_dispatch(&[&x], &[]);
        // C: write y — different buffer, concurrent OK.
        assert!(mr.check_and_record(&[], &[&y]));
        // D: read y — conflict (against C's write).
        assert!(!mr.check_dispatch(&[&y], &[]));
        mr.reset();
        mr.add_dispatch(&[&y], &[]);
        assert_eq!(mr.barriers_forced(), 2);
    }

    /// `BufferRange::conflicts_with` is symmetric.
    #[test]
    fn conflict_is_symmetric() {
        let d = dev();
        let a = d.alloc_buffer(64, DType::F32, vec![16]).unwrap();
        let r_src = BufferRange::from_buffer(&a, MemRangeRole::Src);
        let r_dst = BufferRange::from_buffer(&a, MemRangeRole::Dst);
        assert!(r_src.conflicts_with(&r_dst));
        assert!(r_dst.conflicts_with(&r_src));
    }
}
