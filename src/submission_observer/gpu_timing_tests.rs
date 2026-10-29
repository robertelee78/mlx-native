use super::*;
use crate::{CompletedGpuTimestamps as T, GpuTimestampUnknown as U};
use std::sync::Mutex;
#[derive(Default)]
struct Sink(Mutex<Vec<CompletedGpuTimingEvent>>);
impl SubmissionObserver for Sink {
    fn record(&self, _: &SubmissionEvent) -> Result<(), ObserverSinkError> { Ok(()) }
    fn record_gpu_timing(&self, e: &CompletedGpuTimingEvent) -> Result<(), ObserverSinkError> {
        let mut v = self.0.try_lock().map_err(|_| ObserverSinkError)?;
        if v.len() >= 16 { return Err(ObserverSinkError); }
        v.push(*e); Ok(())
    }
}
fn make(enabled: bool, limit: u64) -> (Arc<Core>, Arc<Sink>) {
    let sink = Arc::new(Sink::default());
    let mut c = Core::new(sink.clone(), limit, 65536);
    c.gpu_timestamps = enabled;
    c.clock = || Some(10);
    (Arc::new(c), sink)
}
#[test]
fn actual_terminal_hook_once_default_off_empty_error_and_drop_unknown() {
    for enabled in [false, true] {
        let (c,s) = make(enabled, 100);
        let o = CommandObservation::with_core(c.clone(),1,2,3).unwrap();
        o.noncompute(); o.submitted(Some(1),Some(2)); o.drained(Some(3),Some(4),4);
        o.completed_gpu_timing(|| crate::gpu_timestamps::validate(4,1.,2.));
        o.completed_gpu_timing(|| panic!("duplicate timing read")); drop(o);
        assert_eq!(s.0.lock().unwrap().len(), usize::from(enabled));
    }
    let (c,s) = make(true,100);
    let empty = CommandObservation::with_core(c.clone(),1,2,3).unwrap();
    empty.submitted(Some(1),Some(2)); empty.drained(Some(3),Some(4),4);
    empty.completed_gpu_timing(|| panic!("empty must not read GPU properties")); drop(empty);
    let error = CommandObservation::with_core(c.clone(),1,2,3).unwrap();
    error.noncompute(); error.submitted(Some(1),Some(2)); error.drained(Some(3),Some(4),5);
    error.completed_gpu_timing(|| panic!("error must not read GPU properties")); drop(error);
    drop(CommandObservation::with_core(c,1,2,3).unwrap());
    let e = s.0.lock().unwrap();
    assert_eq!(e.len(),3);
    for (e,expected) in e.iter().zip([U::EmptyCommand,U::NotHealthyCompleted,U::DroppedWithoutCheckedTiming]) {
        assert!(matches!(e.timestamps,T::Unknown{reason,..} if reason==expected));
    }
}
#[test]
fn timing_hook_event_limit_is_fail_closed_and_unknown_not_silently_zero() {
    let (c,s) = make(true,1);
    let o = CommandObservation::with_core(c.clone(),1,2,3).unwrap(); // Created consumes sole slot.
    drop(o);
    assert!(s.0.lock().unwrap().is_empty());
    assert!(c.invalid.load(Ordering::Acquire));
    assert!(c.dropped.load(Ordering::Acquire)>0);
}
#[test]
fn retained_registry_reads_same_terminal_resource_without_extra_retention() {
    struct ActualStatus;
    impl pending::TerminalResource for ActualStatus {
        fn status(&self)->u64 { 4 }
        fn gpu_timestamps(&self)->T { crate::gpu_timestamps::validate(4,3.,4.) }
    }
    let (c,s)=make(true,100);
    let o=CommandObservation::with_core(c.clone(),7,2,3).unwrap();
    o.noncompute(); o.submitted(Some(1),Some(2));
    let registry=pending::Registry::new(2);
    registry.retain(o,||ActualStatus);
    assert_eq!(registry.outstanding(),1);
    registry.inspect_after_wait(&c,8); // Wrong queue cannot infer completion.
    assert_eq!(registry.outstanding(),1);
    registry.inspect_after_wait(&c,7);
    assert_eq!(registry.outstanding(),0);
    let events=s.0.lock().unwrap(); assert_eq!(events.len(),1);
    assert!(matches!(events[0].timestamps,T::Available{start_seconds:3.,end_seconds:4.,..}));
}
#[test]
fn default_sink_refuses_optin_and_unknown_survives_prior_invalidity() {
    struct OldSink;
    impl SubmissionObserver for OldSink {
        fn record(&self,_:&SubmissionEvent)->Result<(),ObserverSinkError>{Ok(())}
    }
    let mut core=Core::new(Arc::new(OldSink),100,65536);
    core.gpu_timestamps=true; core.clock=||Some(10);
    let core=Arc::new(core);
    drop(CommandObservation::with_core(core.clone(),1,2,3).unwrap());
    assert!(core.invalid.load(Ordering::Acquire));
    assert!(core.dropped.load(Ordering::Acquire)>0);
    let (core,sink)=make(true,100);
    let o=CommandObservation::with_core(core.clone(),1,2,3).unwrap();
    core.fail(); drop(o);
    assert_eq!(sink.0.lock().unwrap().len(),1);
    assert!(matches!(sink.0.lock().unwrap()[0].timestamps,T::Unknown{reason:U::DroppedWithoutCheckedTiming,..}));
}
#[test]
#[ignore = "fresh-process tiny actual CB timing/drop/empty control; root supervised only"]
fn actual_completed_gpu_timing_direct_and_retired_async_no_extra_wait() {
    use crate::{DType,KernelRegistry,MlxDevice};
    assert!(64 + 1024*1024 + 16*std::mem::size_of::<CompletedGpuTimingEvent>() < 4*1024*1024);
    let sink = Arc::new(Sink::default());
    install_submission_observer_with_gpu_timestamps(sink.clone(),128,1024*1024).unwrap();
    let d=MlxDevice::new().expect("actual Metal required; no skip");
    let mut registry=KernelRegistry::new();
    let mut a=d.alloc_buffer(16,DType::F32,vec![4]).unwrap();
    let mut b=d.alloc_buffer(16,DType::F32,vec![4]).unwrap();
    let tmp=d.alloc_buffer(16,DType::F32,vec![4]).unwrap();
    let out=d.alloc_buffer(16,DType::F32,vec![4]).unwrap();
    a.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1.,2.,3.,4.]);
    b.as_mut_slice::<f32>().unwrap().copy_from_slice(&[10.,20.,30.,40.]);
    assert!(std::env::var_os("MLX_PROFILE_CB").is_none());
    let mut producer=d.command_encoder().unwrap();
    producer.enable_submission_observation(11).unwrap();
    producer.set_observation_scope(Some(ObservationScope{epoch:11,id:1}));
    assert!(matches!(producer.completed_gpu_timestamps(),T::Unknown{..}));
    crate::ops::elementwise::elementwise_add(&mut producer,&mut registry,d.metal_device(),&a,&b,&tmp,4,DType::F32).unwrap();
    producer.commit_labeled("gpu.timing.async.producer"); drop(producer);
    assert_eq!(submission_observation_status().pending_commands,1);
    let mut consumer=d.command_encoder().unwrap();
    consumer.enable_submission_observation(11).unwrap();
    consumer.set_observation_scope(Some(ObservationScope{epoch:11,id:2}));
    crate::ops::elementwise::elementwise_add(&mut consumer,&mut registry,d.metal_device(),&tmp,&b,&out,4,DType::F32).unwrap();
    consumer.commit_and_wait_labeled("gpu.timing.consumer").unwrap();
    assert!(matches!(consumer.completed_gpu_timestamps(),T::Available{..}));
    assert_eq!(consumer.completed_gpu_timestamps(),consumer.completed_gpu_timestamps());
    drop(consumer);
    assert_eq!(tmp.as_logical_slice::<f32>().unwrap(),&[11.,22.,33.,44.]);
    assert_eq!(out.as_logical_slice::<f32>().unwrap(),&[21.,42.,63.,84.]);
    let mut empty=d.command_encoder().unwrap(); empty.enable_submission_observation(11).unwrap();
    empty.commit_and_wait_labeled("gpu.timing.empty").unwrap(); drop(empty);
    let mut abandoned=d.command_encoder().unwrap(); abandoned.enable_submission_observation(11).unwrap(); drop(abandoned);
    let status=submission_observation_status();
    assert_eq!(status.pending_commands,0); assert_eq!(status.dropped_events,0);
    assert_eq!(status.unknown_scopes,0); assert!(!status.invalid);
    let e=sink.0.lock().unwrap(); assert_eq!(e.len(),4);
    assert_eq!(e.iter().filter(|e|matches!(e.timestamps,T::Available{..})).count(),2);
    assert!(matches!(e[2].timestamps,T::Unknown{reason:U::EmptyCommand,..}));
    assert!(matches!(e[3].timestamps,T::Unknown{reason:U::DroppedWithoutCheckedTiming,..}));
    let mut ids=e.iter().map(|e|e.command_id).collect::<Vec<_>>(); ids.sort_unstable(); ids.dedup(); assert_eq!(ids.len(),4);
    for event in e.iter() { println!("GPU_TIMING_TERMINAL {}",serde_json::to_string(event).unwrap()); }
    println!("GPU_TIMING_CONTROL completed=2 empty_unknown=1 unsubmitted_unknown=1 output_exact=true pending=0 extra_wait=false physical_qualified=false");
}
