//! Raw completed-CB envelopes, not shader busy time or a CPU-clock mapping.
use serde::{Deserialize, Serialize};
use objc::{msg_send, sel, sel_impl};

/// Apple documents these host seconds as relative to system mach time.
/// No conversion/affine join to CLOCK_MONOTONIC or POSIX is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuTimestampClock {
    MetalCommandBufferHostSecondsSystemMachTime,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuTimestampUnknown {
    NotHealthyCompleted,
    EmptyCommand,
    NonFinite,
    NonPositive,
    NotStrictlyOrdered,
    DroppedWithoutCheckedTiming,
    TaintedObservation,
}
/// Version-one timing payload. Unknown preserves invalid raw property bits
/// where actually read; it must not be silently interpreted as zero duration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CompletedGpuTimestamps {
    Available { clock: GpuTimestampClock, start_seconds: f64, end_seconds: f64 },
    Unknown { reason: GpuTimestampUnknown, checked_status: Option<u64>, raw_bits: Option<[u64; 2]> },
}
impl CompletedGpuTimestamps {
    pub(crate) fn unknown(reason: GpuTimestampUnknown, status: Option<u64>) -> Self {
        Self::Unknown { reason, checked_status: status, raw_bits: None }
    }
}
pub(crate) fn validate(status: u64, start: f64, end: f64) -> CompletedGpuTimestamps {
    let reason = if status != 4 { Some(GpuTimestampUnknown::NotHealthyCompleted) }
        else if !start.is_finite() || !end.is_finite() { Some(GpuTimestampUnknown::NonFinite) }
        else if start <= 0.0 || end <= 0.0 { Some(GpuTimestampUnknown::NonPositive) }
        else if end <= start { Some(GpuTimestampUnknown::NotStrictlyOrdered) } else { None };
    if let Some(reason) = reason {
        CompletedGpuTimestamps::Unknown { reason, checked_status: Some(status), raw_bits: Some([start.to_bits(), end.to_bits()]) }
    } else {
        CompletedGpuTimestamps::Available { clock: GpuTimestampClock::MetalCommandBufferHostSecondsSystemMachTime,
            start_seconds: start, end_seconds: end }
    }
}
pub(crate) fn read(command: &metal::CommandBufferRef) -> CompletedGpuTimestamps {
    let status = command.status() as u64;
    if status != 4 { return CompletedGpuTimestamps::unknown(GpuTimestampUnknown::NotHealthyCompleted, Some(status)); }
    // Read-only properties on this actual healthy completed CB, no wait/commit.
    let (start, end): (f64, f64) = unsafe {
        (msg_send![command, GPUStartTime], msg_send![command, GPUEndTime])
    };
    validate(status, start, end)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_timestamp_validator_preserves_available_and_unknown_bits() {
        assert!(matches!(validate(4, 1.25, 1.5), CompletedGpuTimestamps::Available {start_seconds: 1.25, end_seconds:1.5, ..}));
        for (status,s,e,expected) in [(5,1.,2.,GpuTimestampUnknown::NotHealthyCompleted),
            (3,1.,2.,GpuTimestampUnknown::NotHealthyCompleted), (4,0.,2.,GpuTimestampUnknown::NonPositive),
            (4,2.,1.,GpuTimestampUnknown::NotStrictlyOrdered), (4,1.,1.,GpuTimestampUnknown::NotStrictlyOrdered),
            (4,f64::NAN,2.,GpuTimestampUnknown::NonFinite), (4,1.,f64::INFINITY,GpuTimestampUnknown::NonFinite)] {
            match validate(status,s,e) { CompletedGpuTimestamps::Unknown {reason,raw_bits,..} => {
                assert_eq!(reason,expected); assert_eq!(raw_bits,Some([s.to_bits(),e.to_bits()])); }, _ => panic!("invalid timestamps accepted") }
        }
    }
}
