#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use std::sync::{Mutex, Weak};

#[derive(Default)]
struct Collect(Mutex<Vec<SubmissionEvent>>);
impl SubmissionObserver for Collect {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        let mut events = self.0.try_lock().map_err(|_| ObserverSinkError)?;
        if events.len() >= 64 {
            return Err(ObserverSinkError);
        }
        events.push(*event);
        Ok(())
    }
}
fn tick() -> Option<u64> {
    static T: AtomicU64 = AtomicU64::new(100);
    Some(T.fetch_add(1, Ordering::Relaxed))
}
fn local(limit: u64) -> (Arc<Core>, Arc<Collect>) {
    let sink = Arc::new(Collect::default());
    let mut core = Core::new(sink.clone(), limit, 16 * 1024 * 1024);
    core.clock = tick;
    (Arc::new(core), sink)
}
fn scope(epoch: u64, id: u64) -> ObservationScope {
    ObservationScope { epoch, id }
}
fn geometry() -> DispatchGeometry {
    DispatchGeometry {
        pipeline_identity: 123,
        dispatch_kind: crate::DispatchKind::Threads,
        grid: [4, 1, 1],
        threads_per_threadgroup: [4, 1, 1],
    }
}

#[test]
fn install_state_refuses_late_duplicate_and_racing_install() {
    let state = AtomicU8::new(0);
    begin_install(&state).unwrap();
    finish_install(&state).unwrap();
    mark_started(&state);
    assert_eq!(state.load(Ordering::Acquire), 4);
    assert!(begin_install(&state).is_err());
    let late = AtomicU8::new(0);
    mark_started(&late);
    assert!(begin_install(&late).is_err());
    let raced = AtomicU8::new(0);
    begin_install(&raced).unwrap();
    mark_started(&raced);
    assert!(finish_install(&raced).is_err());
}

#[test]
fn actual_command_state_emits_one_submit_and_idempotent_checked_drain() {
    let (core, sink) = local(64);
    let mut c = CommandObservation::with_core(core.clone(), 3, 4, 11).unwrap();
    c.set_scope(Some(scope(11, 2)));
    c.dispatch(geometry());
    c.submitted(tick(), tick());
    c.drained(tick(), tick(), 4);
    c.drained(tick(), tick(), 4);
    drop(c);
    let events = sink.0.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == ObservationKind::Submitted)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == ObservationKind::CheckedDrain)
            .count(),
        1
    );
    assert!(events
        .iter()
        .all(|e| e.command_id == 1 && e.queue_identity == 3 && e.device_registry_id == 4));
    assert_eq!(events[1].scope, Some(scope(11, 2)));
    assert!(events
        .iter()
        .filter(|e| e.kind != ObservationKind::DispatchEncoded)
        .all(|e| e.scope.is_none()));
    assert!(!core.invalid.load(Ordering::Acquire));
    assert_eq!(core.unknown.load(Ordering::Acquire), 0);
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
}

#[test]
fn actual_event_keeps_numeric_pipeline_and_dispatch_geometry_without_inferred_rows() {
    let (core, sink) = local(64);
    let mut c = CommandObservation::with_core(core, 1, 2, 3).unwrap();
    c.set_scope(Some(scope(3, 1)));
    c.dispatch(geometry());
    let mut changed = geometry();
    changed.dispatch_kind = crate::DispatchKind::ThreadGroups;
    changed.grid = [8, 2, 1];
    changed.threads_per_threadgroup = [32, 1, 1];
    c.dispatch(changed);
    let events = sink.0.lock().unwrap();
    assert_eq!(events[1].geometry, Some(geometry()));
    assert_eq!(events[2].geometry, Some(changed));
    assert_eq!(events[1].dispatch_ordinal, 1);
    assert_eq!(events[2].dispatch_ordinal, 2);
    assert!(events[0].geometry.is_none());
    drop(events);
    drop(c);
}

#[test]
fn duplicate_submit_or_changed_wait_status_is_sticky_invalid() {
    let (core, _) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
    c.submitted(tick(), tick());
    c.submitted(tick(), tick());
    assert!(core.invalid.load(Ordering::Acquire));
    let (core, _) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
    c.submitted(tick(), tick());
    c.drained(tick(), tick(), 4);
    c.drained(tick(), tick(), 5);
    assert!(core.invalid.load(Ordering::Acquire));
}

#[test]
fn unsubmitted_drop_is_not_gpu_progress_and_unchecked_submit_stays_unknown() {
    let (core, sink) = local(64);
    drop(CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap());
    assert_eq!(core.unknown.load(Ordering::Acquire), 0);
    assert!(!sink
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == ObservationKind::Submitted));
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
    c.set_scope(Some(scope(3, 1)));
    c.dispatch(geometry());
    c.submitted(tick(), tick());
    drop(c);
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
    assert!(!sink
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == ObservationKind::CheckedDrain));
}

#[test]
fn fresh_epoch_replay_requires_explicit_rebind_and_unknown_never_heals() {
    let (core, _) = local(64);
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 20).unwrap();
    c.bind_replay(Some(scope(10, 7)), None);
    c.dispatch(geometry());
    assert_eq!(c.scope, None);
    c.replay_map(&[(scope(10, 7), scope(20, 9))]).unwrap();
    c.bind_replay(Some(scope(10, 7)), None);
    c.dispatch(geometry());
    assert_eq!(c.scope, None); // replay never replaces the explicit direct scope
    assert!(core.unknown.load(Ordering::Acquire) > 0);
    assert!(c.replay_map(&[(scope(10, 7), scope(10, 8))]).is_err());
    assert!(c
        .replay_map(&[(scope(10, 7), scope(20, 8)), (scope(10, 7), scope(20, 9))])
        .is_err());
}

#[test]
fn private_capture_tags_permute_by_index_and_mixed_fusion_is_unknown() {
    let (core, _) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    let mut tags = c.capture_scopes().unwrap();
    tags.push(Some(scope(10, 1)));
    tags.push(Some(scope(10, 2)));
    tags.push(Some(scope(10, 3)));
    tags.permute(&[2, 0]);
    assert_eq!(tags.get(0), Some(scope(10, 3)));
    assert_eq!(tags.get(1), Some(scope(10, 1)));
    tags.transformed(1);
    assert_eq!(tags.get(0), None);
    assert!(core.metadata.load(Ordering::Acquire) > 0);
    drop(c);
    assert!(core.metadata.load(Ordering::Acquire) > 0); // graph owns its own lease
    drop(tags);
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
}

#[test]
fn metadata_and_event_caps_refuse_without_favorable_partial_receipts() {
    let sink = Arc::new(Collect::default());
    let core = Arc::new(Core::new(sink.clone(), 64, 1));
    assert!(CommandObservation::with_core(core.clone(), 1, 2, 3).is_err());
    assert!(sink.0.lock().unwrap().is_empty());
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
    let (core, sink) = local(1);
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
    c.set_scope(Some(scope(3, 1)));
    c.dispatch(geometry());
    assert!(core.invalid.load(Ordering::Acquire));
    assert_eq!(sink.0.lock().unwrap().len(), 1);
}

#[test]
fn command_rotation_uses_new_ids_and_does_not_infer_old_completion() {
    let (core, sink) = local(64);
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
    c.set_scope(Some(scope(3, 1)));
    c.dispatch(geometry());
    c.submitted(tick(), tick());
    let next = c.rotate().unwrap();
    drop(c);
    drop(next);
    let events = sink.0.lock().unwrap();
    let ids: Vec<_> = events
        .iter()
        .filter(|e| e.kind == ObservationKind::Created)
        .map(|e| e.command_id)
        .collect();
    assert_eq!(ids, vec![1, 2]);
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
}

struct BadSink(bool);
impl SubmissionObserver for BadSink {
    fn record(&self, _: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        if self.0 {
            panic!("injected observer panic, never through GPU ownership");
        }
        Err(ObserverSinkError)
    }
}
#[test]
fn sink_error_and_panic_do_not_unwind_the_command_state() {
    for panic in [false, true] {
        let mut core = Core::new(Arc::new(BadSink(panic)), 64, 16 * 1024 * 1024);
        core.clock = tick;
        let core = Arc::new(core);
        let c = CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap();
        c.submitted(tick(), tick());
        c.drained(tick(), tick(), 4);
        drop(c);
        assert!(core.invalid.load(Ordering::Acquire));
    }
}

#[derive(Default)]
struct Reenter(Mutex<Weak<Core>>);
impl SubmissionObserver for Reenter {
    fn record(&self, e: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        let core = self.0.try_lock().map_err(|_| ObserverSinkError)?.upgrade();
        if let Some(core) = core {
            core.emit(*e);
        }
        Ok(())
    }
}
#[test]
fn reentrant_sink_is_sticky_invalid_without_deadlock() {
    let sink = Arc::new(Reenter::default());
    let mut core = Core::new(sink.clone(), 64, 16 * 1024 * 1024);
    core.clock = tick;
    let core = Arc::new(core);
    *sink.0.lock().unwrap() = Arc::downgrade(&core);
    drop(CommandObservation::with_core(core.clone(), 1, 2, 3).unwrap());
    assert!(core.invalid.load(Ordering::Acquire));
}

#[test]
fn same_epoch_second_replay_requires_remap_and_never_leaks_into_direct_dispatch() {
    let (core, sink) = local(64);
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    let mut tags = c.capture_scopes().unwrap();
    tags.push(Some(scope(10, 1)));
    c.set_scope(Some(scope(10, 99)));
    let first = tags.claim(Some(c.id()), 1);
    c.bind_replay(tags.get(0), Some(first));
    c.dispatch(geometry());
    let second = tags.claim(Some(c.id()), 1);
    c.bind_replay(tags.get(0), Some(second));
    c.dispatch(geometry());
    // An explicit current-epoch mapping must be consulted, never ignored.
    c.replay_map(&[(scope(10, 1), scope(10, 2))]).unwrap();
    c.bind_replay(tags.get(0), Some(tags.claim(Some(c.id()), 1)));
    c.dispatch(geometry());
    c.dispatch(geometry());
    let events = sink.0.lock().unwrap();
    assert_eq!(events[1].scope, Some(scope(10, 1)));
    assert_eq!(events[1].replay_binding, ReplayBinding::ImmediateCapture);
    assert_eq!(events[1].capture_command_id, Some(c.id()));
    assert!(events[1].capture_generation.is_some());
    assert_eq!(events[2].scope, None);
    assert_eq!(events[2].replay_binding, ReplayBinding::Unknown);
    assert_eq!(events[3].scope, Some(scope(10, 2)));
    assert_eq!(events[3].replay_binding, ReplayBinding::ExplicitRemap);
    assert_eq!(events[4].scope, Some(scope(10, 99)));
    assert_eq!(events[4].captured_scope, None);
    assert_eq!(events[4].replay_binding, ReplayBinding::Direct);
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
}

#[test]
fn capture_replay_into_another_command_is_not_automatic_even_same_epoch() {
    let (core, _) = local(64);
    let first = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    let mut tags = first.capture_scopes().unwrap();
    tags.push(Some(scope(10, 1)));
    let next = CommandObservation::with_core(core, 1, 2, 10).unwrap();
    assert!(!tags.claim(Some(next.id()), 1).automatic);
}

#[test]
fn tag_misalignment_and_scope_overflow_never_leave_shifted_known_tags() {
    let (core, _) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    let mut tags = c.capture_scopes().unwrap();
    tags.push(Some(scope(10, 1)));
    assert!(!tags.aligned(2));
    assert_eq!(tags.get(0), None);
    assert!(core.invalid.load(Ordering::Acquire));
    let (core, _) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    let mut tags = c.capture_scopes().unwrap();
    for _ in 0..=MAX_SCOPES {
        tags.push(Some(scope(10, 1)));
    }
    assert!(!tags.aligned(MAX_SCOPES + 1));
    assert_eq!(tags.get(0), None);
    assert!(core.invalid.load(Ordering::Acquire));
}

#[test]
fn raw_escape_taints_submit_drain_and_drop_even_when_scope_and_status_look_known() {
    let (core, sink) = local(64);
    let mut c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    c.set_scope(Some(scope(10, 1)));
    c.raw_escape();
    c.dispatch(geometry());
    c.submitted(tick(), tick());
    c.drained(tick(), tick(), 4);
    drop(c);
    assert!(sink
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|e| matches!(
            e.kind,
            ObservationKind::Submitted | ObservationKind::CheckedDrain | ObservationKind::Dropped
        ))
        .all(|e| e.tainted));
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
}

#[test]
fn unobserved_commits_are_counted_without_manufacturing_scoped_gpu_events() {
    let (core, sink) = local(64);
    core.note_unobserved();
    core.note_unobserved();
    assert_eq!(core.unobserved.load(Ordering::Acquire), 2);
    assert!(sink.0.lock().unwrap().is_empty());
}

#[test]
fn empty_committed_cb_is_not_forward_progress_but_unchecked_noncompute_is_unknown() {
    let (core, sink) = local(64);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    c.submitted(tick(), tick());
    drop(c);
    assert_eq!(core.unknown.load(Ordering::Acquire), 0);
    let submitted = *sink
        .0
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.kind == ObservationKind::Submitted)
        .unwrap();
    assert_eq!(submitted.dispatch_ordinal, 0);
    assert_eq!(submitted.noncompute_operations, 0);
    assert_eq!(submitted.scope, None);
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    c.noncompute();
    c.submitted(tick(), tick());
    drop(c);
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
}

#[test]
fn sink_contention_is_error_not_a_blocking_native_call() {
    let (core, sink) = local(64);
    let held = sink.0.lock().unwrap();
    let c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
    assert!(core.invalid.load(Ordering::Acquire));
    assert!(core.dropped.load(Ordering::Acquire) > 0);
    drop(held);
    drop(c);
}

#[test]
fn destructor_during_unwind_skips_sink_and_records_dropped_metadata() {
    let (core, sink) = local(64);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _c = CommandObservation::with_core(core.clone(), 1, 2, 10).unwrap();
        std::panic::resume_unwind(Box::new("injected preexisting unwind"));
    }));
    assert!(result.is_err());
    assert!(core.invalid.load(Ordering::Acquire));
    assert!(core.dropped.load(Ordering::Acquire) > 0);
    assert!(!sink
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == ObservationKind::Dropped));
}

#[test]
fn bounded_id_exhaustion_refuses_without_retaining_metadata_reservation() {
    let (core, _) = local(64);
    core.next_id.store(u64::MAX, Ordering::Release);
    assert!(CommandObservation::with_core(core.clone(), 1, 2, 10).is_err());
    assert!(core.invalid.load(Ordering::Acquire));
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
}

#[test]
#[ignore = "tiny two-operation actual recorded GPU observer; fresh supervised process required"]
fn submission_observer_recorded_two_operation_actual() {
    use crate::{DType, GraphExecutor, KernelRegistry, MlxDevice};
    // Known payload/metadata bound, not driver/bootstrap or physical admission.
    const USERDATA: usize = 4 * 1024 * 1024;
    assert!(
        128 * std::mem::size_of::<crate::CapturedNode>()
            + 64 * std::mem::size_of::<SubmissionEvent>()
            + 1024 * 1024
            + 64
            < USERDATA
    );
    let sink = Arc::new(Collect::default());
    install_submission_observer(sink.clone(), 64, 1024 * 1024).unwrap();
    let device = MlxDevice::new().expect("actual Metal required; no skip");
    let mut a = device.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut b = device.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut tmp = device.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut out = device.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    a.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1., 2., 3., 4.]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10., 20., 30., 40.]);
    tmp.as_mut_slice::<f32>().unwrap().fill(-999.);
    out.as_mut_slice::<f32>().unwrap().fill(-999.);
    let executor = GraphExecutor::new(device.clone());
    let mut registry = KernelRegistry::new();
    let mut graph = executor.begin_recorded().unwrap();
    graph
        .encoder_mut()
        .enable_submission_observation(11)
        .unwrap();
    graph
        .encoder_mut()
        .set_observation_scope(Some(scope(11, 1)));
    graph.barrier_between(&[&a, &b], &[&tmp]);
    graph
        .elementwise_add(
            &mut registry,
            device.metal_device(),
            &a,
            &b,
            &tmp,
            4,
            DType::F32,
        )
        .unwrap();
    graph
        .encoder_mut()
        .set_observation_scope(Some(scope(11, 2)));
    graph.barrier_between(&[&tmp, &b], &[&out]);
    graph
        .elementwise_add(
            &mut registry,
            device.metal_device(),
            &tmp,
            &b,
            &out,
            4,
            DType::F32,
        )
        .unwrap();
    assert!(!sink.0.lock().unwrap().iter().any(|e| matches!(
        e.kind,
        ObservationKind::DispatchEncoded | ObservationKind::Submitted
    )));
    graph
        .finish_with_reorder()
        .expect("actual annotated compute and checked drain");
    assert_eq!(
        tmp.as_logical_slice::<f32>().unwrap(),
        &[11., 22., 33., 44.]
    );
    assert_eq!(
        out.as_logical_slice::<f32>().unwrap(),
        &[21., 42., 63., 84.]
    );
    let status = submission_observation_status();
    assert!(!status.invalid);
    assert!(status.capture_valid);
    assert_eq!(status.dropped_events, 0);
    assert_eq!(status.unobserved_commits, 0);
    assert_eq!(status.unknown_scopes, 0);
    assert_eq!(status.metadata_reserved_bytes, 0);
    let events = sink.0.lock().unwrap();
    let dispatches: Vec<_> = events
        .iter()
        .filter(|e| e.kind == ObservationKind::DispatchEncoded)
        .collect();
    assert_eq!(dispatches.len(), 2);
    assert_eq!(dispatches[0].scope, Some(scope(11, 1)));
    assert_eq!(dispatches[1].scope, Some(scope(11, 2)));
    for event in &dispatches {
        let g = event.geometry.expect("actual encoded geometry required");
        assert!(g.pipeline_identity > 0);
        assert_eq!(g.dispatch_kind, crate::DispatchKind::Threads);
        assert_eq!(g.grid, [4, 1, 1]);
        assert_eq!(g.threads_per_threadgroup, [4, 1, 1]);
    }
    assert_eq!(
        dispatches[0].geometry.unwrap().pipeline_identity,
        dispatches[1].geometry.unwrap().pipeline_identity
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == ObservationKind::Submitted)
            .count(),
        1
    );
    let drains: Vec<_> = events
        .iter()
        .filter(|e| e.kind == ObservationKind::CheckedDrain)
        .collect();
    assert_eq!(drains.len(), 1);
    assert_eq!(
        drains[0].checked_status,
        Some(metal::MTLCommandBufferStatus::Completed as u64)
    );
    assert!(events
        .iter()
        .all(|e| e.command_id == dispatches[0].command_id
            && e.begin_monotonic_ns <= e.end_monotonic_ns));
    println!(
        "F2_NATIVE_OBSERVER_TWO_OP_COMPLETE {}",
        serde_json::json!({
        "protocol":"PASS","dispatches":2,"submits":1,"checked_drains":1,
        "userdata_upper_bytes":USERDATA,"driver_bootstrap_bytes":"Unknown",
        "clock_id":"CLOCK_MONOTONIC","status":status,"events":*events,
        "physical_row_membership":"opaque HF join not supplied by native",
        "model_admission":false,"performance_claim":false})
    );
}

#[path = "submission_observer/async_tests.rs"]
mod async_tests;

#[path = "submission_observer/rotation_controls.rs"]
mod rotation_controls;
#[path = "submission_observer/rotation_actual.rs"]
mod rotation_actual;
