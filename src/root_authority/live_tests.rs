//! Actual pure authority/ticket boundary; no Metal, globals or class probes.
use super::*;
use std::sync::{atomic::AtomicUsize, Barrier};
fn key() -> DeviceKey {
    DeviceKey {
        object: 1,
        registry: 1,
    }
}
fn core(bytes: usize, roots: usize) -> Arc<Core> {
    let c = Arc::new(Core::default());
    let mut s = c.lock().unwrap();
    s.install(RootCreationLimits {
        max_attempted_requested_bytes: 1024,
        max_root_attempts: 32,
        max_raw_constructor_attempts: 32,
        max_registered_devices: 1,
        max_rope_cache_entries: 1,
    })
    .unwrap();
    s.live.limits = Some(LiveRootLimits {
        max_live_and_pending_requested_bytes: bytes,
        max_live_and_pending_roots: roots,
    });
    s.register(key()).unwrap();
    drop(s);
    c
}
fn ticket(o: &Origin) -> Arc<live::Ticket> {
    match o {
        Origin::Known(c) => c.live.as_ref().unwrap().clone(),
        _ => panic!("known required"),
    }
}
#[test]
fn live_root_final_ticket_waits_for_origin_and_resource_owner() {
    let c = core(64, 1);
    let (resource, origin) = c
        .create_with_origin(key(), 64, false, |o| Ok(ticket(o)))
        .unwrap();
    assert_eq!(c.live_snapshot().unwrap().live_requested_bytes, 64);
    drop(origin);
    assert!(c.create(key(), 1, false, || Ok(())).is_err());
    assert_eq!(c.live_snapshot().unwrap().live_requested_bytes, 64);
    drop(resource);
    let s = c.live_snapshot().unwrap();
    assert_eq!(s.live_requested_bytes, 0);
    assert_eq!(s.final_releases, 1);
    assert_eq!(s.high_water_requested_bytes, 64);
    assert_eq!(c.lock().unwrap().bytes, 64, "history never refunded");
    c.create(key(), 64, false, || Ok(())).unwrap();
}
#[test]
fn live_root_nil_rolls_pending_back_but_keeps_history() {
    let c = core(64, 1);
    let err: Result<((), Origin)> = c.create(key(), 64, false, || Err(state::fail("injected nil")));
    assert!(err.is_err());
    let s = c.live_snapshot().unwrap();
    assert_eq!(s.pending_requested_bytes, 0);
    assert_eq!(s.live_requested_bytes, 0);
    assert_eq!(c.lock().unwrap().bytes, 64);
    c.create(key(), 64, false, || Ok(())).unwrap();
}
#[test]
fn live_root_denial_precedes_actual_factory_and_checks_overflow() {
    let c = core(64, 1);
    let called = AtomicUsize::new(0);
    let held = c.create(key(), 64, false, || Ok(())).unwrap();
    assert!(c
        .create(key(), 1, false, || {
            called.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .is_err());
    assert_eq!(called.load(Ordering::SeqCst), 0);
    drop(held);
    let mut counters = live::Counters::default();
    counters.limits = Some(LiveRootLimits {
        max_live_and_pending_requested_bytes: usize::MAX,
        max_live_and_pending_roots: usize::MAX,
    });
    counters.reserve(usize::MAX).unwrap();
    assert!(counters.reserve(1).is_err());
}
#[test]
fn live_root_pending_blocks_competing_constructor() {
    let c = core(64, 1);
    let entered = Arc::new(Barrier::new(2));
    let finish = Arc::new(Barrier::new(2));
    let worker = {
        let c = c.clone();
        let entered = entered.clone();
        let finish = finish.clone();
        std::thread::spawn(move || {
            c.create(key(), 64, false, || {
                entered.wait();
                finish.wait();
                Ok(())
            })
        })
    };
    entered.wait();
    let snap = c.live_snapshot().unwrap();
    assert_eq!(snap.pending_requested_bytes, 64);
    let denied = c
        .create::<()>(key(), 1, false, || panic!("denied factory invoked"))
        .is_err();
    finish.wait();
    let result = worker.join().unwrap();
    assert!(denied);
    drop(result);
    assert_eq!(c.live_snapshot().unwrap().pending_roots, 0);
}
#[test]
fn live_root_postraw_uncertain_keeps_ok_and_charge() {
    let c = core(64, 1);
    let result = c.create_with_origin(key(), 64, false, |o| {
        ticket(o).uncertain();
        Ok(())
    });
    assert!(result.is_ok(), "nonnil cannot become Native55 no-root Err");
    drop(result);
    assert!(c.live_snapshot().is_err());
    let s = c.state.lock().unwrap();
    assert_eq!(s.live.pending_bytes, 64);
    assert_eq!(s.live.live_bytes, 0);
}
#[test]
fn live_root_unwind_never_refunds_uncertain_pending() {
    let c = core(64, 1);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = c.create::<()>(key(), 64, false, || panic!("injected raw unwind"));
    }));
    assert!(failed.is_err());
    assert!(c.health().is_err());
    assert_eq!(c.state.lock().unwrap().live.pending_bytes, 64);
}
#[test]
fn live_root_postraw_mutex_poison_preserves_resource_success() {
    let c = core(64, 1);
    let result = c.create(key(), 64, false, || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = c.state.lock().unwrap();
            panic!("postraw mutex fault");
        }));
        Ok(())
    });
    assert!(result.is_ok());
    drop(result);
    assert!(c.health().is_err());
    assert_eq!(
        c.state.lock().unwrap_err().into_inner().live.pending_bytes,
        64
    );
}
#[test]
fn live_root_legacy_and_history_only_do_not_create_live_tickets() {
    let c = Core::default();
    let (_, origin) = c.create(key(), 64, false, || Ok(())).unwrap();
    assert!(matches!(origin, Origin::Legacy));
    let c = core(64, 1);
    c.lock().unwrap().live.limits = None;
    let (_, origin) = c.create(key(), 64, false, || Ok(())).unwrap();
    assert!(matches!(origin,Origin::Known(c) if c.live.is_none()));
}

#[test]
fn live_root_unknown_class_never_invokes_attachment_and_readback_failure_poisoned() {
    assert!(live::eligible(
        "AGXG17XFamilyBuffer",
        "Apple M5 Max",
        b"27.2\0",
        b"26B5112a\0"
    ));
    assert!(!live::eligible(
        "OtherBuffer",
        "Apple M5 Max",
        b"27.2\0",
        b"26B5112a\0"
    ));
    assert!(!live::eligible(
        "AGXG17XFamilyBuffer",
        "Apple M5 Max",
        b"27.3\0",
        b"26B5112a\0"
    ));
    for allowed in [false, true] {
        let c = core(64, 1);
        let called = AtomicUsize::new(0);
        let result = c.create_with_origin(key(), 64, false, |o| {
            live::attach_checked(&ticket(o), allowed, || {
                called.fetch_add(1, Ordering::SeqCst);
                false
            });
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(called.load(Ordering::SeqCst), usize::from(allowed));
        drop(result);
        assert!(c.health().is_err());
        assert_eq!(c.state.lock().unwrap().live.pending_bytes, 64);
    }
}

#[test]
fn live_root_blocked_waiters_cannot_allocate_or_publish_covered_after_poison() {
    use std::sync::mpsc;
    for snapshot in [false, true] {
        let c = core(64, 2);
        let held = c.create(key(), 32, false, || Ok(())).unwrap();
        let t = ticket(&held.1);
        let guard = c.state.lock().unwrap();
        let (tx, rx) = mpsc::channel();
        *c.after_lock_precheck.lock().unwrap() = Some(Box::new(move || {
            tx.send(()).unwrap();
        }));
        let calls = Arc::new(AtomicUsize::new(0));
        let worker = {
            let c = c.clone();
            let calls = calls.clone();
            std::thread::spawn(move || {
                if snapshot {
                    c.live_snapshot().is_err()
                } else {
                    c.create(key(), 1, false, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .is_err()
                }
            })
        };
        // Worker passed the ACTUAL lock fast precheck and is blocked on guard.
        rx.recv().unwrap();
        live::attach_checked(&t, false, || panic!("unknown class must not attach"));
        drop(guard);
        let refused = worker.join().unwrap(); // always unblock/join before assertions
        assert!(refused);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(c.state.lock().unwrap().live.pending_bytes, 0);
        drop(held);
        drop(t); // poisoned state keeps existing live charge conservative
        assert_eq!(c.state.lock().unwrap().live.live_bytes, 32);
    }
}

#[test]
fn live_root_same_product_version_different_driver_build_is_unobserved() {
    assert!(!live::eligible(
        "AGXG17XFamilyBuffer",
        "Apple M5 Max",
        b"27.2\0",
        b"26B5112b\0"
    ));
    assert!(!live::eligible(
        "AGXG17XFamilyBuffer",
        "Apple M5 Max",
        b"27.2\0",
        b""
    ));
}

#[test]
fn live_root_host_identity_cache_initializes_once_under_contention() {
    use std::sync::OnceLock;
    let cache = Arc::new(OnceLock::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(8));
    let mut threads = Vec::new();
    for _ in 0..8 {
        let cache = cache.clone();
        let calls = calls.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            *live::cached_host(&cache, || {
                calls.fetch_add(1, Ordering::SeqCst);
                live::HostBytes::from_slices(b"27.2\0", b"26B5112a\0")
            })
            .unwrap()
        }));
    }
    let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(values.iter().all(|v| *v == values[0]));
    assert!(values[0].allows("AGXG17XFamilyBuffer", "Apple M5 Max"));
    assert!(!values[0].allows("OtherBuffer", "Apple M5 Max"));
    assert!(!values[0].allows("AGXG17XFamilyBuffer", "OtherDevice"));
}
#[test]
fn live_root_host_identity_failure_and_wrong_build_cache_never_heal() {
    use std::sync::OnceLock;
    let cache = OnceLock::new();
    assert!(live::cached_host(&cache, || None).is_none());
    assert!(live::cached_host(&cache, || panic!("cached error must not retry")).is_none());
    let wrong = OnceLock::new();
    let host = live::cached_host(&wrong, || {
        live::HostBytes::from_slices(b"27.2\0", b"26B5112b\0")
    })
    .unwrap();
    assert!(!host.allows("AGXG17XFamilyBuffer", "Apple M5 Max"));
    assert_eq!(
        Some(host),
        live::cached_host(&wrong, || panic!("wrong build must remain cached"))
    );
    assert!(live::HostBytes::from_slices(&[1; 65], b"x").is_none());
}
