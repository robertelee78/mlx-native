//! CPU controls of the actual rotation/setter, not a copied scope emulator.
use super::*;
#[test]
fn rotation_explicit_fallback_never_inherits_current_known_or_replay_state() {
    let (core, _) = local(64);
    let mut current = CommandObservation::with_core(core, 3, 4, 11).unwrap();
    current.set_scope(Some(scope(11, 1)));
    current.set_rotation_scope(Some(scope(11, 2))).unwrap();
    current.captured_scope = Some(scope(11, 1));
    current.pending_replay = Some(Some(scope(11, 1)));
    current.capture_command_id = Some(current.id);
    current.capture_generation = Some(7);
    current.replay_binding = ReplayBinding::ExplicitRemap;
    let next = current.rotate().unwrap();
    assert_eq!(current.scope, Some(scope(11, 1)));
    assert_eq!(next.scope, Some(scope(11, 2)));
    assert_eq!(next.rotation_scope, Some(scope(11, 2)));
    assert!(
        next.captured_scope.is_none()
            && next.pending_replay.is_none()
            && next.capture_command_id.is_none()
            && next.capture_generation.is_none()
    );
    assert_eq!(next.replay_binding, ReplayBinding::Direct);
    assert_eq!(next.ordinal, 0);
}
#[test]
fn rotation_missing_or_cleared_fallback_remains_unknown_not_current_rows() {
    let (core, _) = local(64);
    let mut current = CommandObservation::with_core(core.clone(), 3, 4, 11).unwrap();
    current.set_scope(Some(scope(11, 1)));
    let mut next = current.rotate().unwrap();
    assert!(next.scope.is_none());
    next.dispatch(geometry());
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
    current.set_rotation_scope(Some(scope(11, 2))).unwrap();
    current.set_rotation_scope(None).unwrap();
    assert!(current.rotate().unwrap().scope.is_none());
}
#[test]
fn rotation_stale_fallback_refuses_and_does_not_keep_old_valid_fallback() {
    let (core, _) = local(64);
    let mut current = CommandObservation::with_core(core.clone(), 3, 4, 11).unwrap();
    current.set_rotation_scope(Some(scope(11, 2))).unwrap();
    assert!(current.set_rotation_scope(Some(scope(12, 2))).is_err());
    assert!(current.rotation_scope.is_none());
    assert!(current.rotate().unwrap().scope.is_none());
    assert_eq!(core.unknown.load(Ordering::Acquire), 1);
}
