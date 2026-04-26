//! Per-command-buffer GPU timing accumulator for kernel-level profiling.
//!
//! Hf2q's `HF2Q_DECODE_PROFILE=1` instrumentation tracks CPU-side wall
//! clock per layer phase, but does not attribute time to specific GPU
//! kernel dispatches.  The MoE dwq46 0.93× decode parity gap residual
//! (per ADR-012 §Optimize / Task #15) cannot be localized further
//! without per-cb (or per-dispatch) GPU timing.
//!
//! This module exposes a thread-safe accumulator keyed by string label.
//! Each labeled `commit_and_wait` records the cb's GPU wall-clock
//! (`MTLCommandBuffer.GPUEndTime - GPUStartTime`).  At decode end,
//! `dump()` produces a sorted breakdown showing which labeled cb
//! contributed the most GPU time per token.
//!
//! Per-DISPATCH timing (using `MTLCounterSampleBuffer.sampleCounters`)
//! is a separate Metal API surface deferred to a future ADR; this
//! module establishes the per-CB ground truth first.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

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

fn table() -> &'static Mutex<HashMap<String, ProfileEntry>> {
    static T: OnceLock<Mutex<HashMap<String, ProfileEntry>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
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

/// Reset the profile table.  Typically called at start of decode.
pub fn reset() {
    if let Ok(mut t) = table().lock() {
        t.clear();
    }
}

/// Dump the profile table sorted by descending total_ns.
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

/// Whether per-CB profiling is enabled via `MLX_PROFILE_CB=1`.
///
/// Cached in an atomic so the hot path is a single load.
pub fn is_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static CACHED: AtomicI8 = AtomicI8::new(-1);
    let v = CACHED.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = std::env::var("MLX_PROFILE_CB").is_ok();
    CACHED.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

#[cfg(test)]
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
}
