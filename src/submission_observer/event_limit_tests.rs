use super::*;

struct Count(AtomicU64);
impl SubmissionObserver for Count {
    fn record(&self, _: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn total_event_limit_validation_accepts_exact_ceiling_and_refuses_one_past() {
    for limit in [1, 1_000_000, MAX_TOTAL_EVENTS] {
        assert!(validate_observer_limits(limit, 16 * 1024 * 1024).is_ok());
    }
    for limit in [0, MAX_TOTAL_EVENTS + 1, u64::MAX] {
        assert!(validate_observer_limits(limit, 16 * 1024 * 1024).is_err());
    }
}

#[test]
fn total_event_limit_does_not_relax_metadata_or_pending_bounds() {
    assert!(validate_observer_limits(MAX_TOTAL_EVENTS, 0).is_err());
    assert!(validate_observer_limits(MAX_TOTAL_EVENTS, 16 * 1024 * 1024 + 1).is_err());
    assert!(validate_observer_limits(MAX_TOTAL_EVENTS, 1).is_ok());
    assert_eq!(pending::LIMIT, 1024);
}

fn last_event_then_refusal(limit: u64) {
    let sink = Arc::new(Count(AtomicU64::new(0)));
    let core = Arc::new(Core::new(sink.clone(), limit, 16 * 1024 * 1024));
    assert_eq!(core.events.load(Ordering::Acquire), 0);
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
    assert_eq!(core.pending.outstanding(), 0);
    // Seed only the historical counter: exercise actual command emission at
    // the boundary without allocating/storing/emitting sixteen million events.
    core.events.store(limit - 1, Ordering::Release);
    let command = CommandObservation::with_core(core.clone(), 3, 4, 11).unwrap();
    assert_eq!(core.events.load(Ordering::Acquire), limit);
    assert_eq!(sink.0.load(Ordering::Acquire), 1);
    assert!(!core.invalid.load(Ordering::Acquire));
    drop(command); // Real Dropped event is the first refused event.
    assert!(core.invalid.load(Ordering::Acquire));
    assert_eq!(core.events.load(Ordering::Acquire), limit);
    assert_eq!(core.dropped.load(Ordering::Acquire), 1);
    assert_eq!(sink.0.load(Ordering::Acquire), 1);
    assert_eq!(core.metadata.load(Ordering::Acquire), 0);
    assert_eq!(core.pending.outstanding(), 0);
}

#[test]
fn total_event_limit_actual_emit_refuses_at_new_ceiling_without_linear_storage() {
    last_event_then_refusal(MAX_TOTAL_EVENTS);
}

#[test]
fn total_event_limit_explicit_legacy_million_remains_unchanged() {
    last_event_then_refusal(1_000_000);
}
