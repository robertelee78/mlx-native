//! Fresh-process actual production async API boundary; root alone executes.
use super::super::*;
use std::sync::{Arc, Mutex};
#[derive(Default)]
struct Collect(Mutex<Vec<SubmissionEvent>>);
impl SubmissionObserver for Collect {
    fn record(&self, e: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        let mut v = self.0.try_lock().map_err(|_| ObserverSinkError)?;
        if v.len() >= 64 {
            return Err(ObserverSinkError);
        }
        v.push(*e);
        Ok(())
    }
}
#[test]
#[ignore = "actual tiny async submitted-CB ownership; fresh supervised process required"]
fn submission_observer_async_drop_then_actual_consumer_checked_wait() {
    use crate::{DType, KernelRegistry, MlxDevice};
    const USERDATA: usize = 4 * 1024 * 1024;
    assert!(64 + 1024 * 1024 + 64 * std::mem::size_of::<SubmissionEvent>() < USERDATA);
    assert!(
        std::env::var_os("MLX_PROFILE_CB").is_none(),
        "must exercise true async commit_labeled"
    );
    let sink = Arc::new(Collect::default());
    install_submission_observer(sink.clone(), 64, 1024 * 1024).unwrap();
    let d = MlxDevice::new().expect("actual Metal required; no skip");
    let mut registry = KernelRegistry::new();
    let mut a = d.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut b = d.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut tmp = d.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    let mut out = d.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    a.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1., 2., 3., 4.]);
    b.as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[10., 20., 30., 40.]);
    tmp.as_mut_slice::<f32>().unwrap().fill(-999.);
    out.as_mut_slice::<f32>().unwrap().fill(-999.);
    let mut producer = d.command_encoder().unwrap();
    producer.enable_submission_observation(11).unwrap();
    producer.set_observation_scope(Some(ObservationScope { epoch: 11, id: 1 }));
    crate::ops::elementwise::elementwise_add(
        &mut producer,
        &mut registry,
        d.metal_device(),
        &a,
        &b,
        &tmp,
        4,
        DType::F32,
    )
    .unwrap();
    producer.commit_labeled("observer.async.producer");
    drop(producer);
    let before = submission_observation_status();
    assert_eq!(before.pending_commands, 1);
    assert_eq!(before.unknown_scopes, 0);
    assert!(!before.invalid);
    assert!(!sink
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == ObservationKind::Dropped));
    println!(
        "NATIVE_ASYNC_OWNER_PHASE pending=1 encoder_dropped=true completion_not_inferred=true"
    );
    let mut consumer = d.command_encoder().unwrap();
    consumer.enable_submission_observation(11).unwrap();
    consumer.set_observation_scope(Some(ObservationScope { epoch: 11, id: 2 }));
    crate::ops::elementwise::elementwise_add(
        &mut consumer,
        &mut registry,
        d.metal_device(),
        &tmp,
        &b,
        &out,
        4,
        DType::F32,
    )
    .unwrap();
    consumer
        .commit_and_wait_labeled("observer.async.consumer")
        .expect("real consumer checked wait");
    drop(consumer);
    assert_eq!(
        tmp.as_logical_slice::<f32>().unwrap(),
        &[11., 22., 33., 44.]
    );
    assert_eq!(
        out.as_logical_slice::<f32>().unwrap(),
        &[21., 42., 63., 84.]
    );
    let status = submission_observation_status();
    assert_eq!(status.pending_commands, 0);
    assert_eq!(status.unknown_scopes, 0);
    assert_eq!(status.dropped_events, 0);
    assert_eq!(status.unobserved_commits, 0);
    assert!(!status.invalid);
    let events = sink.0.lock().unwrap();
    let queues = events
        .iter()
        .map(|e| e.queue_identity)
        .collect::<std::collections::BTreeSet<_>>();
    let devices = events
        .iter()
        .map(|e| e.device_registry_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(queues.len(), 1);
    assert!(!queues.contains(&0));
    assert_eq!(devices.len(), 1);
    assert!(!devices.contains(&0));
    let dispatches = events
        .iter()
        .filter(|e| e.kind == ObservationKind::DispatchEncoded)
        .collect::<Vec<_>>();
    assert_eq!(dispatches.len(), 2);
    assert!(dispatches
        .iter()
        .all(|e| e.geometry.is_some_and(|g| g.pipeline_identity != 0
            && g.grid.iter().all(|v| *v > 0)
            && g.threads_per_threadgroup.iter().all(|v| *v > 0))));
    let ids = events
        .iter()
        .filter(|e| e.kind == ObservationKind::Submitted)
        .map(|e| e.command_id)
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    for id in &ids {
        assert_eq!(
            events
                .iter()
                .filter(|e| e.command_id == *id
                    && e.kind == ObservationKind::CheckedDrain
                    && e.checked_status == Some(4))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e.command_id == *id
                    && e.kind == ObservationKind::Dropped
                    && e.checked_status == Some(4))
                .count(),
            1
        );
    }
    println!("NATIVE_ASYNC_OWNER_COMPLETE userdata_limit={USERDATA} payload=64 dispatches=2 submitted=2 checked=2 dropped_checked=2 pending=0 unknown=0 exact_values=8 no_model=true");
}
