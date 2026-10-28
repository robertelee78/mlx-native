use super::*;
#[derive(Default)]
struct Sink(Mutex<Vec<SubmissionEvent>>);
impl SubmissionObserver for Sink {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError> {
        self.0
            .try_lock()
            .map_err(|_| ObserverSinkError)?
            .push(*event);
        Ok(())
    }
}
fn tick() -> Option<u64> {
    static T: AtomicU64 = AtomicU64::new(100);
    Some(T.fetch_add(1, Ordering::Relaxed))
}
fn core(metadata: u64) -> (Arc<Core>, Arc<Sink>) {
    let sink = Arc::new(Sink::default());
    let mut c = Core::new(sink.clone(), 256, metadata);
    c.clock = tick;
    (Arc::new(c), sink)
}
fn submitted(c: &Arc<Core>, queue: u64) -> CommandObservation {
    let mut o = CommandObservation::with_core(c.clone(), queue, 7, 11).unwrap();
    o.noncompute();
    o.submitted(tick(), tick());
    o
}
struct Fake {
    status: Arc<AtomicU64>,
    released: Arc<AtomicU64>,
    count: Arc<AtomicU64>,
}
impl TerminalResource for Fake {
    fn status(&self) -> u64 {
        self.status.load(Ordering::Acquire)
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        assert!(
            self.count.load(Ordering::Acquire) > 0,
            "pending charge survives payload destruction"
        );
        self.released.fetch_add(1, Ordering::Relaxed);
    }
}
fn retain(
    r: &Registry<Fake>,
    c: &Arc<Core>,
    queue: u64,
    status: Arc<AtomicU64>,
    released: Arc<AtomicU64>,
) {
    r.retain(submitted(c, queue), || Fake {
        status,
        released,
        count: r.outstanding.clone(),
    });
}
#[test]
fn pending_actual_registry_retains_nonterminal_wrong_queue_and_checks_each_status() {
    let (c, s) = core(1024 * 1024);
    let r = Registry::new(4);
    let status = Arc::new(AtomicU64::new(2));
    let released = Arc::new(AtomicU64::new(0));
    retain(&r, &c, 3, status.clone(), released.clone());
    assert_eq!(r.outstanding(), 1);
    r.inspect_after_wait(&c, 4);
    assert_eq!(r.outstanding(), 1);
    r.inspect_after_wait(&c, 3);
    assert_eq!(r.outstanding(), 1);
    assert_eq!(released.load(Ordering::Relaxed), 0);
    status.store(4, Ordering::Release);
    r.inspect_after_wait(&c, 3);
    assert_eq!(r.outstanding(), 0);
    assert_eq!(released.load(Ordering::Relaxed), 1);
    assert_eq!(c.unknown.load(Ordering::Acquire), 0);
    assert_eq!(c.metadata.load(Ordering::Acquire), 0);
    let e = s.0.lock().unwrap();
    assert_eq!(
        e.iter()
            .filter(|e| e.kind == ObservationKind::CheckedDrain && e.checked_status == Some(4))
            .count(),
        1
    );
    assert_eq!(
        e.iter()
            .filter(|e| e.kind == ObservationKind::Dropped)
            .count(),
        1
    );
}
#[test]
fn pending_actual_error_is_recorded_invalid_never_anchor_completed() {
    let (c, s) = core(1024 * 1024);
    let r = Registry::new(1);
    retain(
        &r,
        &c,
        3,
        Arc::new(AtomicU64::new(5)),
        Arc::new(AtomicU64::new(0)),
    );
    r.inspect_after_wait(&c, 3);
    assert_eq!(r.outstanding(), 0);
    assert!(c.invalid.load(Ordering::Acquire));
    assert!(s
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == ObservationKind::CheckedDrain && e.checked_status == Some(5)));
}
#[test]
fn pending_actual_limit_and_metadata_refuse_before_resource_factory() {
    let (c, _) = core(1024 * 1024);
    let r = Registry::new(1);
    let status = Arc::new(AtomicU64::new(2));
    retain(&r, &c, 3, status.clone(), Arc::new(AtomicU64::new(0)));
    r.retain(submitted(&c, 3), || {
        panic!("count denial must precede factory")
    });
    assert_eq!(r.outstanding(), 1);
    assert!(c.invalid.load(Ordering::Acquire));
    assert!(c.unknown.load(Ordering::Acquire) > 0);
    status.store(4, Ordering::Release);
    r.inspect_after_wait(&c, 3);
    assert_eq!(r.outstanding(), 0);
    let (c, _) = core(std::mem::size_of::<CommandObservation>() as u64 + 1);
    let r: Registry<Fake> = Registry::new(1);
    r.retain(submitted(&c, 3), || {
        panic!("metadata denial must precede factory")
    });
    assert_eq!(r.outstanding(), 0);
    assert!(c.invalid.load(Ordering::Acquire));
    assert_eq!(c.metadata.load(Ordering::Acquire), 0);
}
#[test]
fn pending_actual_registry_poison_fails_closed_without_sink_under_lock() {
    let (c, _) = core(1024 * 1024);
    let r: Registry<Fake> = Registry::new(1);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = r.items.lock().unwrap();
        panic!("injected poison");
    }));
    r.retain(submitted(&c, 3), || Fake {
        status: Arc::new(AtomicU64::new(4)),
        released: Arc::new(AtomicU64::new(0)),
        count: r.outstanding.clone(),
    });
    assert!(c.invalid.load(Ordering::Acquire));
    assert!(c.unknown.load(Ordering::Acquire) > 0);
    assert_eq!(r.outstanding(), 0);
}

#[test]
fn pending_actual_unwind_never_defers_invalid_owner_or_calls_factory() {
    let (c, _) = core(1024 * 1024);
    let r: Registry<Fake> = Registry::new(1);
    let called = Arc::new(AtomicBool::new(false));
    struct OnDrop<'a> {
        r: &'a Registry<Fake>,
        o: Option<CommandObservation>,
        called: Arc<AtomicBool>,
    }
    impl Drop for OnDrop<'_> {
        fn drop(&mut self) {
            self.r.retain(self.o.take().unwrap(), || {
                self.called.store(true, Ordering::Release);
                Fake {
                    status: Arc::new(AtomicU64::new(2)),
                    released: Arc::new(AtomicU64::new(0)),
                    count: self.r.outstanding.clone(),
                }
            });
        }
    }
    let o = submitted(&c, 3);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = OnDrop {
            r: &r,
            o: Some(o),
            called: called.clone(),
        };
        panic!("actual unwind seam");
    }));
    assert!(!called.load(Ordering::Acquire));
    assert_eq!(r.outstanding(), 0);
    assert!(c.invalid.load(Ordering::Acquire));
    assert_eq!(c.metadata.load(Ordering::Acquire), 0);
}
