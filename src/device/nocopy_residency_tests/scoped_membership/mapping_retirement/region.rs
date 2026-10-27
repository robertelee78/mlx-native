//! Bounded metadata lookup, never a read from a retired mapping. SDK Basic64
//! flavour9 is packed4/36bytes/9int words; no packed Rust field access needed.
use serde::Serialize;
use std::time::Instant;
extern "C" {
    static mach_task_self_: u32;
    fn mach_vm_region(
        task: u32,
        address: *mut u64,
        size: *mut u64,
        flavor: i32,
        info: *mut i32,
        count: *mut u32,
        object: *mut u32,
    ) -> i32;
    fn mach_port_deallocate(task: u32, name: u32) -> i32;
}
const BASIC64: i32 = 9;
const WORDS: usize = 9;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) enum Kind {
    LiveComplete,
    Absent,
    OverlapOrSplit,
    Unavailable,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct Record {
    pub label: &'static str,
    pub begin_ns: u128,
    pub end_ns: u128,
    pub requested_base: u64,
    pub requested_bytes: u64,
    pub kr: i32,
    pub returned_address: u64,
    pub returned_size: u64,
    pub returned_count: u32,
    pub returned_object_port: u32,
    pub port_cleanup_kr: i32,
    pub kind: Kind,
    pub protocol_valid: bool,
}
fn interpret(
    base: u64,
    bytes: u64,
    kr: i32,
    address: u64,
    size: u64,
    count: u32,
    port_cleanup: i32,
) -> (Kind, bool) {
    let Some(end) = base.checked_add(bytes) else {
        return (Kind::Unavailable, false);
    };
    if base == 0 || bytes == 0 || count != WORDS as u32 || port_cleanup != 0 {
        return (Kind::Unavailable, false);
    }
    if kr == 1 {
        return (Kind::Absent, true);
    } // KERN_INVALID_ADDRESS: no containing/next entry.
    if kr != 0 {
        return (Kind::Unavailable, false);
    }
    let Some(region_end) = address.checked_add(size) else {
        return (Kind::Unavailable, false);
    };
    if size == 0 {
        return (Kind::Unavailable, false);
    }
    if address >= end {
        return (Kind::Absent, true);
    }
    if address <= base && region_end >= end {
        return (Kind::LiveComplete, true);
    }
    (Kind::OverlapOrSplit, true)
}
pub(super) fn observe(label: &'static str, base: usize, bytes: usize, origin: &Instant) -> Record {
    let begin_ns = origin.elapsed().as_nanos();
    assert_ne!(base, 0);
    assert_eq!(base % super::PAGE, 0);
    assert_eq!(bytes, super::FILE_BYTES);
    let mut address = base as u64;
    let mut size = 0u64;
    let mut count = WORDS as u32;
    let mut info = [0i32; WORDS];
    let mut object = 0u32;
    // SAFETY: current task; exact SDK Basic64 ABI and bounded initialized9word
    // output. Saved address is metadata input only, never dereferenced.
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut address,
            &mut size,
            BASIC64,
            info.as_mut_ptr(),
            &mut count,
            &mut object,
        )
    };
    // Attempt cleanup BEFORE interpreting ANY output, including error/count
    // contradictions. The port is only a returned owned right, never a task.
    let cleanup = if object == 0 {
        0
    } else {
        unsafe { mach_port_deallocate(mach_task_self_, object) }
    };
    let (kind, protocol_valid) =
        interpret(base as u64, bytes as u64, kr, address, size, count, cleanup);
    let end_ns = origin.elapsed().as_nanos();
    Record {
        label,
        begin_ns,
        end_ns,
        requested_base: base as u64,
        requested_bytes: bytes as u64,
        kr,
        returned_address: address,
        returned_size: size,
        returned_count: count,
        returned_object_port: object,
        port_cleanup_kr: cleanup,
        kind,
        protocol_valid,
    }
}
#[test]
fn mapping_region_interpreter_never_calls_split_or_bad_output_absent() {
    assert_eq!(
        interpret(4096, 4096, 0, 4096, 4096, 9, 0),
        (Kind::LiveComplete, true)
    );
    assert_eq!(
        interpret(4096, 4096, 0, 0, 12288, 9, 0),
        (Kind::LiveComplete, true)
    );
    assert_eq!(
        interpret(4096, 4096, 0, 8192, 4096, 9, 0),
        (Kind::Absent, true)
    );
    assert_eq!(
        interpret(4096, 4096, 1, 4096, 0, 9, 0),
        (Kind::Absent, true)
    );
    assert_eq!(
        interpret(4096, 4096, 0, 4096, 2048, 9, 0),
        (Kind::OverlapOrSplit, true)
    );
    assert_eq!(
        interpret(4096, 4096, 0, 6144, 4096, 9, 0),
        (Kind::OverlapOrSplit, true)
    );
    for (kr, count, port) in [(5, 9, 0), (0, 8, 0), (1, 8, 0), (0, 9, 5), (1, 9, 5)] {
        assert_eq!(
            interpret(4096, 4096, kr, 8192, 4096, count, port),
            (Kind::Unavailable, false)
        );
    }
    assert_eq!(
        interpret(u64::MAX - 1, 4096, 1, 0, 0, 9, 0),
        (Kind::Unavailable, false)
    );
}
