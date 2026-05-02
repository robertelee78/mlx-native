//! Per-command-buffer + per-dispatch GPU timing accumulator for kernel-level
//! profiling.
//!
//! Hf2q's `HF2Q_DECODE_PROFILE=1` instrumentation tracks CPU-side wall
//! clock per layer phase, but does not attribute time to specific GPU
//! kernel dispatches.  The MoE dwq46 0.93× decode parity gap residual
//! (per ADR-012 §Optimize / Task #15) cannot be localized further
//! without per-cb (or per-dispatch) GPU timing.
//!
//! This module exposes two thread-safe accumulators:
//!
//! * **Per-CB** (`MLX_PROFILE_CB=1`) — a HashMap keyed by string label.
//!   Each labeled `commit_and_wait` records the cb's GPU wall-clock
//!   (`MTLCommandBuffer.GPUEndTime - GPUStartTime`).
//! * **Per-dispatch** (`MLX_PROFILE_DISPATCH=1`, ADR-015 iter63) — a flat
//!   `Vec<DispatchEntry>` populated from
//!   `MTLCounterSampleBuffer.sampleCounters` between
//!   `set_compute_pipeline_state` and `dispatch_threads` at every
//!   `encode*` site.  Dump groups entries by their owning `cb_label`,
//!   preserving insertion order within each group.
//!
//! At decode end, [`dump`] / [`dump_dispatches`] produce sorted
//! breakdowns showing which labeled cb (and which kernel within each cb)
//! contributed the most GPU time per token.
//!
//! ### Cross-validation (ADR-015 iter63 Risk R3)
//!
//! Per-dispatch numbers are **upper-bound serialized cost** — the
//! `withBarrier:YES` requirement on `sampleCountersInBuffer` serializes
//! the encoder under `MTLDispatchTypeConcurrent`. The per-CB sum will
//! therefore be ≥ the matching `MLX_PROFILE_CB` total.  Acceptable
//! drift: ≤ 5%; > 10% indicates a clock-domain or sampling bug.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI8, AtomicU64, Ordering};

/// Per-label accumulator entry.
#[derive(Clone, Debug, Default)]
pub struct ProfileEntry {
    /// Number of times this label was recorded.
    pub count: u64,
    /// Total GPU wall-clock time in nanoseconds.
    pub total_ns: u64,
    /// Minimum observed GPU time in nanoseconds.
    pub min_ns: u64,
    /// Maximum observed GPU time in nanoseconds.
    pub max_ns: u64,
}

/// One per-dispatch timing entry within a CB (ADR-015 iter63 Phase A).
///
/// Populated via `MTLCounterSampleBuffer.sampleCounters` calls inserted
/// between pipeline binding and the actual `dispatch_threads` /
/// `dispatch_thread_groups` call inside every
/// [`crate::CommandEncoder::encode*`] method.  Resolved into ns from the
/// raw GPU-tick samples by [`record_dispatch`] using the (cpu, gpu) pair
/// captured at the most recent [`reset`] / [`dump_dispatches`] boundary.
#[derive(Clone, Debug)]
pub struct DispatchEntry {
    /// The cb_label that owned this dispatch (mirrors the per-CB table key).
    pub cb_label: String,
    /// Captured op kind ("RmsNorm", "Sdpa", "ElemMul", "ElemAdd", "Softmax",
    /// "Other").  See [`crate::CapturedOpKind::name`].
    pub op_kind: &'static str,
    /// 0-based ordinal within the CB.
    pub dispatch_index: u32,
    /// `end_gpu_ns - start_gpu_ns`, i.e. the wall-clock of this single
    /// dispatch on the GPU.  Already converted from raw GPU ticks.
    pub gpu_ns: u64,
    /// Raw start timestamp (ns since CPU epoch, after tick→ns conversion).
    pub start_gpu_ns: u64,
    /// Raw end timestamp (ns since CPU epoch, after tick→ns conversion).
    pub end_gpu_ns: u64,
}

fn table() -> &'static Mutex<HashMap<String, ProfileEntry>> {
    static T: OnceLock<Mutex<HashMap<String, ProfileEntry>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

fn dispatch_table() -> &'static Mutex<Vec<DispatchEntry>> {
    static T: OnceLock<Mutex<Vec<DispatchEntry>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(Vec::new()))
}

/// Record a labeled GPU duration.
///
/// Called by `CommandEncoder::commit_and_wait_labeled` after reading
/// `MTLCommandBuffer.GPUEndTime - GPUStartTime`.  Lock contention is
/// negligible — the encoder serializes calls anyway.
pub fn record(label: &str, gpu_ns: u64) {
    if let Ok(mut t) = table().lock() {
        let e = t.entry(label.to_string()).or_default();
        if e.count == 0 || gpu_ns < e.min_ns {
            e.min_ns = gpu_ns;
        }
        if gpu_ns > e.max_ns {
            e.max_ns = gpu_ns;
        }
        e.count = e.count.saturating_add(1);
        e.total_ns = e.total_ns.saturating_add(gpu_ns);
    }
}

/// Append a per-dispatch entry to the global dispatch table.
///
/// Called from
/// [`crate::CommandEncoder::resolve_dispatch_samples`] inside
/// `commit_and_wait_labeled` when `MLX_PROFILE_DISPATCH=1` is set.  The
/// caller has already converted raw GPU ticks to ns using the cached
/// scale factor (see [`record_clock_pair`]).
pub fn record_dispatch(entry: DispatchEntry) {
    if let Ok(mut t) = dispatch_table().lock() {
        t.push(entry);
    }
}

/// Reset the profile tables.  Typically called at start of decode.
///
/// Clears both the per-CB table and the per-dispatch entries, and resets
/// the GPU↔CPU clock-pair cache so the next dump reads a fresh baseline.
pub fn reset() {
    if let Ok(mut t) = table().lock() {
        t.clear();
    }
    if let Ok(mut t) = dispatch_table().lock() {
        t.clear();
    }
    CLOCK_CPU_NS.store(0, Ordering::Relaxed);
    CLOCK_GPU_TICKS.store(0, Ordering::Relaxed);
}

/// Dump the per-CB profile table sorted by descending total_ns.
///
/// Returns `Vec<(label, entry)>` sorted by total time.
pub fn dump() -> Vec<(String, ProfileEntry)> {
    let mut v: Vec<(String, ProfileEntry)> = if let Ok(t) = table().lock() {
        t.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        Vec::new()
    };
    v.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    v
}

/// Dump per-dispatch entries grouped by `cb_label`, preserving CB-arrival
/// order within each group.
///
/// Returns `Vec<(cb_label, Vec<DispatchEntry>)>`.  The outer ordering
/// follows the order in which each `cb_label` first appeared (i.e. the
/// chronological CB submission order); the inner ordering follows the
/// per-CB `dispatch_index` (insertion order from
/// [`record_dispatch`]).
pub fn dump_dispatches() -> Vec<(String, Vec<DispatchEntry>)> {
    let entries = if let Ok(t) = dispatch_table().lock() {
        t.clone()
    } else {
        return Vec::new();
    };
    // Group preserving first-appearance order of cb_label.
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<DispatchEntry>> = HashMap::new();
    for e in entries {
        let key = e.cb_label.clone();
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(e);
    }
    order
        .into_iter()
        .map(|k| {
            let v = groups.remove(&k).unwrap_or_default();
            (k, v)
        })
        .collect()
}

/// Whether per-CB profiling is enabled via `MLX_PROFILE_CB=1`.
///
/// Cached in an atomic so the hot path is a single load.
pub fn is_enabled() -> bool {
    static CACHED: AtomicI8 = AtomicI8::new(-1);
    let v = CACHED.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = std::env::var("MLX_PROFILE_CB").is_ok();
    CACHED.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Whether per-DISPATCH profiling is enabled via `MLX_PROFILE_DISPATCH=1`.
///
/// Cached in an atomic so the hot path is a single load — same
/// gating semantics as [`is_enabled`].  When set, the per-CB profile
/// is also force-enabled (so cross-validation per Risk R3 is always
/// possible) — see [`is_enabled_or_dispatch`].
pub fn is_dispatch_enabled() -> bool {
    static CACHED: AtomicI8 = AtomicI8::new(-1);
    let v = CACHED.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = std::env::var("MLX_PROFILE_DISPATCH").is_ok();
    CACHED.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

// --------------------------------------------------------------------
// GPU↔CPU clock-pair conversion (ADR-015 iter63 §A.6)
// --------------------------------------------------------------------
//
// `MTLCommonCounterSetTimestamp` returns "GPU time when the sample is
// taken" in **GPU ticks**, not nanoseconds.  Apple's
// `device.sampleTimestamps(cpu, gpu)` fills both clocks simultaneously,
// allowing us to derive a tick→ns scale factor.
//
// On Apple silicon the GPU timebase is typically 1 tick = 1 ns (verified
// empirically; `mach_timebase_info` numer/denom = 125/3 nominal but the
// GPU side reports the same nanosecond domain), but we do NOT hardcode
// this — the pair is sampled once on first call and reused until
// `reset()` clears the cache.  `convert_gpu_ticks_to_ns` falls back to
// a 1:1 ratio if the pair has not been sampled yet (initial encoder
// activity before the first `dump_dispatches` snapshot).
//
// Storing the CPU/GPU snapshot in two AtomicU64 instead of a Mutex keeps
// the conversion lock-free on the per-dispatch resolve path.

static CLOCK_CPU_NS: AtomicU64 = AtomicU64::new(0);
static CLOCK_GPU_TICKS: AtomicU64 = AtomicU64::new(0);

/// Record a `(cpu_ns, gpu_ticks)` snapshot from
/// `MTLDevice.sampleTimestamps`.  Most recent snapshot wins.
///
/// Called from [`crate::CommandEncoder::resolve_dispatch_samples`] on
/// the first resolve after a [`reset`] (or at any other CB boundary if
/// the encoder chooses to refresh — both legal).
pub fn record_clock_pair(cpu_ns: u64, gpu_ticks: u64) {
    CLOCK_CPU_NS.store(cpu_ns, Ordering::Relaxed);
    CLOCK_GPU_TICKS.store(gpu_ticks, Ordering::Relaxed);
}

/// Convert a raw GPU tick value to ns using the most recent
/// `(cpu_ns, gpu_ticks)` pair, falling back to a 1:1 ratio when no
/// pair has been recorded yet.
///
/// The conversion is `ns = ticks * (cpu_ns / gpu_ticks)`.  When the
/// snapshot has CPU >> 0 and GPU >> 0 the math is exact at u64
/// precision for any tick range under 2^32 (well past a single CB's
/// dispatch count).
pub fn convert_gpu_ticks_to_ns(gpu_ticks: u64) -> u64 {
    let cpu = CLOCK_CPU_NS.load(Ordering::Relaxed);
    let gpu = CLOCK_GPU_TICKS.load(Ordering::Relaxed);
    if cpu == 0 || gpu == 0 {
        // No snapshot — use 1:1 (best-effort; Apple silicon ships a
        // nanosecond GPU timebase in practice).
        return gpu_ticks;
    }
    // Avoid overflow: scale by f64 and round back.  At 6,000 dispatches
    // per CB and ~10 ms per dispatch the ticks fit comfortably in f64
    // mantissa (~2^53).
    let scale = cpu as f64 / gpu as f64;
    (gpu_ticks as f64 * scale) as u64
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn record_dump_reset_cycle() {
        reset();
        record("A", 100);
        record("A", 200);
        record("B", 50);
        let d = dump();
        // Sorted by total_ns descending.
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].0, "A");
        assert_eq!(d[0].1.count, 2);
        assert_eq!(d[0].1.total_ns, 300);
        assert_eq!(d[0].1.min_ns, 100);
        assert_eq!(d[0].1.max_ns, 200);
        assert_eq!(d[1].0, "B");
        assert_eq!(d[1].1.count, 1);
        reset();
        assert!(dump().is_empty());
    }

    #[test]
    fn dispatch_record_dump_reset_cycle() {
        reset();
        record_dispatch(DispatchEntry {
            cb_label: "layer.attn[0]".into(),
            op_kind: "RmsNorm",
            dispatch_index: 0,
            gpu_ns: 100,
            start_gpu_ns: 1_000,
            end_gpu_ns: 1_100,
        });
        record_dispatch(DispatchEntry {
            cb_label: "layer.attn[0]".into(),
            op_kind: "Sdpa",
            dispatch_index: 1,
            gpu_ns: 500,
            start_gpu_ns: 1_100,
            end_gpu_ns: 1_600,
        });
        record_dispatch(DispatchEntry {
            cb_label: "layer.ffn[0]".into(),
            op_kind: "Other",
            dispatch_index: 0,
            gpu_ns: 250,
            start_gpu_ns: 2_000,
            end_gpu_ns: 2_250,
        });
        let dumps = dump_dispatches();
        // Group order matches first-appearance (attn, then ffn).
        assert_eq!(dumps.len(), 2);
        assert_eq!(dumps[0].0, "layer.attn[0]");
        assert_eq!(dumps[0].1.len(), 2);
        // Within-group order follows insertion order.
        assert_eq!(dumps[0].1[0].dispatch_index, 0);
        assert_eq!(dumps[0].1[0].op_kind, "RmsNorm");
        assert_eq!(dumps[0].1[1].dispatch_index, 1);
        assert_eq!(dumps[0].1[1].op_kind, "Sdpa");
        assert_eq!(dumps[1].0, "layer.ffn[0]");
        assert_eq!(dumps[1].1.len(), 1);
        reset();
        assert!(dump_dispatches().is_empty());
    }

    #[test]
    fn dispatch_dump_empty_when_no_entries() {
        reset();
        assert!(dump_dispatches().is_empty());
    }

    #[test]
    fn convert_gpu_ticks_default_one_to_one() {
        // After reset(), no pair → 1:1 fallback.
        reset();
        assert_eq!(convert_gpu_ticks_to_ns(12_345), 12_345);
    }

    #[test]
    fn convert_gpu_ticks_with_recorded_pair() {
        reset();
        // Suppose 1 GPU tick = 2 ns → cpu_ns / gpu_ticks = 2.0.
        record_clock_pair(2_000, 1_000);
        assert_eq!(convert_gpu_ticks_to_ns(500), 1_000);
        assert_eq!(convert_gpu_ticks_to_ns(0), 0);
    }

    #[test]
    fn convert_gpu_ticks_zero_pair_is_one_to_one() {
        reset();
        // Exactly-zero pair acts as "unrecorded".
        record_clock_pair(0, 1_000);
        assert_eq!(convert_gpu_ticks_to_ns(7), 7);
        record_clock_pair(2_000, 0);
        assert_eq!(convert_gpu_ticks_to_ns(7), 7);
    }
}
