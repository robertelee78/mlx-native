//! Actual production EncoderSession reset; separate fresh process for each mode.
use super::*;
fn run(explicit: bool) {
    use crate::{DType, KernelRegistry, MlxDevice};
    assert_eq!(std::env::var("HF2Q_ENCODER_SESSION").as_deref(), Ok("1"));
    assert!(std::env::var_os("MLX_PROFILE_CB").is_none());
    const USERDATA: usize = 4 * 1024 * 1024;
    assert!(64 + 1024 * 1024 + 64 * std::mem::size_of::<SubmissionEvent>() < USERDATA);
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
    let mut session = d
        .encoder_session()
        .unwrap()
        .expect("HF2Q_ENCODER_SESSION=1 actual session required");
    session.encoder().enable_submission_observation(11).unwrap();
    session.encoder().set_observation_scope(Some(scope(11, 1)));
    if explicit {
        session
            .encoder()
            .set_observation_rotation_scope(Some(scope(11, 2)))
            .unwrap();
    }
    crate::ops::elementwise::elementwise_add(
        session.encoder(),
        &mut registry,
        d.metal_device(),
        &a,
        &b,
        &tmp,
        4,
        DType::F32,
    )
    .unwrap();
    session.commit_stage().unwrap();
    session.reset_for_next_stage().unwrap();
    assert_eq!(submission_observation_status().pending_commands, 1);
    // Do NOT reset CURRENT here: the production rotation must select only fallback.
    crate::ops::elementwise::elementwise_add(
        session.encoder(),
        &mut registry,
        d.metal_device(),
        &tmp,
        &b,
        &out,
        4,
        DType::F32,
    )
    .unwrap();
    session
        .commit_and_wait()
        .expect("actual consumer checked wait");
    drop(session);
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
    assert_eq!(status.dropped_events, 0);
    assert_eq!(status.unobserved_commits, 0);
    assert!(!status.invalid);
    assert_eq!(status.unknown_scopes, if explicit { 0 } else { 1 });
    let events = sink.0.lock().unwrap();
    let dispatches = events
        .iter()
        .filter(|e| e.kind == ObservationKind::DispatchEncoded)
        .collect::<Vec<_>>();
    assert_eq!(dispatches.len(), 2);
    assert_eq!(dispatches[0].scope, Some(scope(11, 1)));
    assert_eq!(
        dispatches[1].scope,
        if explicit { Some(scope(11, 2)) } else { None }
    );
    assert_ne!(dispatches[0].command_id, dispatches[1].command_id);
    for dispatch in dispatches {
        assert_eq!(
            events
                .iter()
                .filter(|e| e.command_id == dispatch.command_id
                    && e.kind == ObservationKind::CheckedDrain
                    && e.checked_status == Some(4))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e.command_id == dispatch.command_id
                    && e.kind == ObservationKind::Dropped
                    && e.checked_status == Some(4))
                .count(),
            1
        );
    }
    println!("NATIVE_ROTATION_SCOPE_COMPLETE explicit_fallback={explicit} payload=64 userdata_limit={USERDATA} dispatches=2 checked=2 pending=0 unknown={} exact_values=8 native_capture_qualified={} no_model=true",status.unknown_scopes,explicit);
}
#[test]
#[ignore = "tiny actual production session reset, explicit fallback; fresh supervised process"]
fn submission_observer_explicit_rotation_fallback_actual() {
    run(true);
}
#[test]
#[ignore = "tiny actual missing-fallback negative, expected Unknown; fresh supervised process"]
fn submission_observer_missing_rotation_fallback_unknown_actual() {
    run(false);
}
