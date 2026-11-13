//! Command-buffer timeline contract (`mlx_native::command_timeline`).
//!
//! Off by default and inert; when on, every committed command is recorded
//! once, in commit order, with host times and Metal's own times on one
//! clock. One test function: the recorder is process-wide state.

use mlx_native::command_timeline::{self, CommandTimeline};
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn copies(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    source: &MlxBuffer,
    sink: &MlxBuffer,
    count: usize,
) -> mlx_native::CommandEncoder {
    let mut encoder = device.command_encoder().expect("create encoder");
    for _ in 0..count {
        dispatch_copy_f32(
            &mut encoder,
            registry,
            device.metal_device(),
            source,
            sink,
            0,
            0,
            source.element_count(),
        )
        .expect("encode copy");
        encoder.memory_barrier();
    }
    encoder
}

fn assert_ordered(record: &CommandTimeline, label: &str) {
    assert_eq!(record.label, label);
    assert!(record.completed, "{label}: not completed");
    assert!(record.encode_begin_s > 0.0, "{label}: no encode start");
    assert!(
        record.encode_begin_s <= record.commit_begin_s
            && record.commit_begin_s <= record.commit_end_s,
        "{label}: host times out of order: {record:?}"
    );
    // Metal's times are on the host marks' clock: the driver cannot have
    // scheduled, nor the GPU started, a command before its commit began.
    assert!(
        record.commit_begin_s <= record.kernel_start_s
            && record.kernel_start_s <= record.kernel_end_s,
        "{label}: scheduling interval out of order: {record:?}"
    );
    assert!(
        record.commit_begin_s <= record.gpu_start_s && record.gpu_start_s <= record.gpu_end_s,
        "{label}: GPU interval out of order: {record:?}"
    );
}

#[test]
fn records_committed_commands_on_one_clock_only_when_enabled() {
    let device = MlxDevice::new().expect("create Metal device");
    let mut registry = KernelRegistry::new();
    let mut source = device
        .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
        .expect("allocate source");
    source
        .as_mut_slice::<f32>()
        .expect("write source")
        .fill(2.5);
    let sink = device
        .alloc_buffer(4_096 * 4, DType::F32, vec![4_096])
        .expect("allocate sink");

    // Off (the default): nothing is recorded or retained.
    assert!(!command_timeline::enabled());
    command_timeline::mark("ignored");
    copies(&device, &mut registry, &source, &sink, 2)
        .commit_and_wait()
        .expect("run while off");
    assert!(command_timeline::drain().is_empty());
    assert!(command_timeline::drain_marks().is_empty());

    command_timeline::set_enabled(true);
    let before = command_timeline::now_seconds();
    let mut first = copies(&device, &mut registry, &source, &sink, 3);
    first.commit_labeled("first");
    command_timeline::mark("between");
    let mut second = copies(&device, &mut registry, &source, &sink, 5);
    second
        .commit_and_wait_labeled("second")
        .expect("run second");
    // A command that encodes nothing and is never waited for by its owner.
    let mut empty = device.command_encoder().expect("create empty encoder");
    empty.commit();
    let after_commits = command_timeline::now_seconds();
    command_timeline::set_enabled(false);

    // Committed after recording stopped: not recorded.
    copies(&device, &mut registry, &source, &sink, 1)
        .commit_and_wait()
        .expect("run after stop");

    let records = command_timeline::drain();
    assert_eq!(records.len(), 3, "{records:?}");
    let [first, second, empty] = &records[..] else {
        unreachable!()
    };
    assert_ordered(first, "first");
    assert_ordered(second, "second");
    assert!(before <= first.encode_begin_s && empty.commit_end_s <= after_commits);

    // Commit order, and one queue runs its commands in that order.
    assert!(first.commit_end_s <= second.commit_begin_s);
    assert!(first.gpu_start_s <= second.gpu_start_s);
    // Only the second was waited for by the host, and the wait returned
    // after the GPU finished it; the first needed no wait at drain.
    assert_eq!(first.wait_begin_s, 0.0);
    assert!(!first.waited_in_drain && !second.waited_in_drain);
    assert!(
        second.commit_end_s <= second.wait_begin_s
            && second.wait_begin_s <= second.wait_end_s
            && second.gpu_end_s <= second.wait_end_s,
        "{second:?}"
    );
    // The counters at commit give each command's own dispatches and barriers.
    assert_eq!(second.dispatches_at_commit - first.dispatches_at_commit, 5);
    assert_eq!(second.barriers_at_commit - first.barriers_at_commit, 5);
    assert_eq!(empty.dispatches_at_commit, second.dispatches_at_commit);

    assert_eq!(empty.label, "");
    assert_eq!(empty.encode_begin_s, 0.0, "nothing was encoded");
    assert!(empty.completed, "drain waits for a command still running");

    let marks = command_timeline::drain_marks();
    assert_eq!(marks.len(), 1);
    assert_eq!(marks[0].0, "between");
    assert!(first.commit_end_s <= marks[0].1 && marks[0].1 <= second.commit_begin_s);

    // Drained once.
    assert!(command_timeline::drain().is_empty());
    assert_eq!(command_timeline::dropped(), 0);

    // The copies ran: recording changed no command.
    assert!(sink
        .as_slice::<f32>()
        .expect("read sink")
        .iter()
        .all(|value| *value == 2.5));
}
