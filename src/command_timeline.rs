//! Opt-in command-buffer timeline (diagnostic, default off).
//!
//! While enabled, every command buffer handed to Metal is retained until
//! [`drain`], together with the host times of its encode start, its commit
//! and any host wait on it. [`drain`] adds the four times Metal reports for
//! a completed command buffer: `kernelStartTime`/`kernelEndTime` (the
//! driver scheduling it) and `GPUStartTime`/`GPUEndTime` (the GPU executing
//! it). All times are seconds on one clock, mach absolute time, which is the
//! clock Metal documents for those properties, so host and GPU intervals
//! can be laid on one axis without a conversion.
//!
//! Recording changes no scheduling decision: no commit becomes a wait, no
//! command buffer is split and nothing is added to a command. The cost is
//! two clock reads and one short mutex section per commit. Retaining a
//! committed command buffer keeps the resources it references alive until
//! [`drain`], so a caller must drain at a bounded cadence.
//!
//! Disabled, each hook is one relaxed atomic load.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use metal::{CommandBuffer, CommandBufferRef, MTLCommandBufferStatus};
use objc::{msg_send, sel, sel_impl};

static ENABLED: AtomicBool = AtomicBool::new(false);

struct Pending {
    command: CommandBuffer,
    record: CommandTimeline,
}

// SAFETY: `MTLCommandBuffer` is documented as safe to retain, release and
// query for status and timing properties from any thread; `Pending` is only
// moved between threads, never encoded into.
unsafe impl Send for Pending {}

static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());
static MARKS: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

/// One committed command buffer. Times are seconds of mach absolute time;
/// `0.0` means "did not happen" (no compute encoder, no host wait) or, for
/// the four Metal-reported times, "not completed without error".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CommandTimeline {
    /// Label applied by a labeled commit; empty for an unlabeled one.
    pub label: String,
    /// Host: the command's first compute encoder was opened (first dispatch).
    pub encode_begin_s: f64,
    /// Host: immediately before and after `-[MTLCommandBuffer commit]`.
    pub commit_begin_s: f64,
    pub commit_end_s: f64,
    /// Host: around `waitUntilCompleted`, when the host waited on this
    /// command before the drain.
    pub wait_begin_s: f64,
    pub wait_end_s: f64,
    /// Metal: driver scheduling interval.
    pub kernel_start_s: f64,
    pub kernel_end_s: f64,
    /// Metal: GPU execution interval.
    pub gpu_start_s: f64,
    pub gpu_end_s: f64,
    /// Process-wide dispatch and barrier counters when the command was
    /// committed. With one encoding thread and one open encoder at a time,
    /// the difference between consecutive records is the command's own
    /// dispatch and barrier count.
    pub dispatches_at_commit: u64,
    pub barriers_at_commit: u64,
    /// The command completed without error.
    pub completed: bool,
    /// [`drain`] had to wait for this command (it was still executing).
    pub waited_in_drain: bool,
}

/// Turn recording on or off. Turning it off does not discard pending
/// records; [`drain`] them.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether recording is on.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Record a named host instant on the timeline's clock. One relaxed atomic
/// load when recording is off.
#[inline]
pub fn mark(name: &'static str) {
    if enabled() {
        record_mark(name);
    }
}

#[cold]
fn record_mark(name: &'static str) {
    let now = now_seconds();
    if let Ok(mut marks) = MARKS.lock() {
        if marks.len() >= MAX_PENDING {
            DROPPED.fetch_add(marks.len() as u64, Ordering::Relaxed);
            marks.clear();
        }
        marks.push((name, now));
    }
}

/// Take every recorded host mark, in recording order.
pub fn drain_marks() -> Vec<(&'static str, f64)> {
    MARKS
        .lock()
        .map(|mut marks| std::mem::take(&mut *marks))
        .unwrap_or_default()
}

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

/// Seconds of mach absolute time: the clock of `GPUStartTime`,
/// `GPUEndTime`, `kernelStartTime` and `kernelEndTime`.
pub fn now_seconds() -> f64 {
    static SECONDS_PER_TICK: LazyLock<f64> = LazyLock::new(|| {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        // SAFETY: `info` is a valid, exclusively borrowed out-parameter.
        let status = unsafe { mach_timebase_info(&mut info) };
        if status != 0 || info.denom == 0 {
            return 0.0;
        }
        f64::from(info.numer) / f64::from(info.denom) * 1e-9
    });
    // SAFETY: no arguments, no preconditions.
    unsafe { mach_absolute_time() as f64 * *SECONDS_PER_TICK }
}

/// Commands held between drains; reaching it discards the held records.
const MAX_PENDING: usize = 4096;

static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Records discarded because [`drain`] was not called for [`MAX_PENDING`]
/// commits. Nonzero means a drained sequence has a gap.
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Record a commit. `encode_begin_s` is `0.0` when the command opened no
/// compute encoder.
pub(crate) fn committed(
    command: &CommandBufferRef,
    label: &str,
    encode_begin_s: f64,
    commit_begin_s: f64,
    commit_end_s: f64,
    dispatches_at_commit: u64,
    barriers_at_commit: u64,
) {
    let pending = Pending {
        command: command.to_owned(),
        record: CommandTimeline {
            label: label.to_owned(),
            encode_begin_s,
            commit_begin_s,
            commit_end_s,
            dispatches_at_commit,
            barriers_at_commit,
            ..CommandTimeline::default()
        },
    };
    if let Ok(mut list) = PENDING.lock() {
        if list.len() >= MAX_PENDING {
            // Nobody is draining. Release what is held rather than keep
            // every completed command's resources alive without bound.
            DROPPED.fetch_add(list.len() as u64, Ordering::Relaxed);
            objc::rc::autoreleasepool(|| list.clear());
        }
        list.push(pending);
    }
}

/// Record a host wait on a command already recorded by [`committed`].
pub(crate) fn waited(command: &CommandBufferRef, wait_begin_s: f64, wait_end_s: f64) {
    let Ok(mut list) = PENDING.lock() else {
        return;
    };
    let target = command as *const CommandBufferRef;
    if let Some(pending) = list
        .iter_mut()
        .rev()
        .find(|pending| std::ptr::eq(&*pending.command as *const CommandBufferRef, target))
    {
        // A command can be waited on more than once; keep the first wait,
        // the one that could have blocked.
        if pending.record.wait_begin_s == 0.0 {
            pending.record.wait_begin_s = wait_begin_s;
            pending.record.wait_end_s = wait_end_s;
        }
    }
}

/// Take every recorded command in commit order, completing each with the
/// times Metal reports. A command that is still executing is waited for
/// (and marked `waited_in_drain`), so call this where a wait is harmless.
pub fn drain() -> Vec<CommandTimeline> {
    let pending = match PENDING.lock() {
        Ok(mut list) => std::mem::take(&mut *list),
        Err(_) => return Vec::new(),
    };
    let mut records = Vec::with_capacity(pending.len());
    // The retained command buffers are released here; scope any autorelease
    // work to this call for pool-less worker threads.
    objc::rc::autoreleasepool(|| {
        for Pending {
            command,
            mut record,
        } in pending
        {
            let mut status = command.status();
            if !matches!(
                status,
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            ) {
                command.wait_until_completed();
                record.waited_in_drain = true;
                status = command.status();
            }
            if matches!(status, MTLCommandBufferStatus::Completed) {
                record.completed = true;
                // SAFETY: read-only CFTimeInterval properties of a completed
                // command buffer.
                unsafe {
                    let command: &CommandBufferRef = &command;
                    record.kernel_start_s = msg_send![command, kernelStartTime];
                    record.kernel_end_s = msg_send![command, kernelEndTime];
                    record.gpu_start_s = msg_send![command, GPUStartTime];
                    record.gpu_end_s = msg_send![command, GPUEndTime];
                }
            }
            records.push(record);
        }
    });
    records
}
