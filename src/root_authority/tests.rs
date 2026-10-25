//! Pure production-core controls; no Metal devices, globals or resource allocation.
use super::*;
use std::sync::{atomic::AtomicUsize, Barrier};
fn limits(bytes: usize, roots: usize) -> RootCreationLimits {
    RootCreationLimits {
        max_attempted_requested_bytes: bytes,
        max_root_attempts: roots,
        max_raw_constructor_attempts: roots,
        max_registered_devices: 2,
        max_rope_cache_entries: 2,
    }
}
fn key(n: usize) -> DeviceKey {
    DeviceKey {
        object: n,
        registry: n as u64,
    }
}
fn installed(bytes: usize, roots: usize) -> Core {
    let core = Core::default();
    {
        let mut s = core.lock().unwrap();
        s.install(limits(bytes, roots)).unwrap();
        s.register(key(1)).unwrap();
    }
    core
}
#[test]
fn root_history_factory_callback_observes_precharge_without_lock() {
    let c = installed(128, 2);
    let result = c
        .create(key(1), 64, false, || {
            let s = c.lock().unwrap();
            assert_eq!((s.bytes, s.roots, s.inflight), (64, 1, 1));
            Ok(17)
        })
        .unwrap();
    assert_eq!(result.0, 17);
    assert!(matches!(result.1, Origin::Known(_)));
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.successful, s.inflight), (64, 1, 0));
}
#[test]
fn root_history_denials_call_no_factory_and_change_no_charge() {
    let c = installed(64, 1);
    c.create(key(1), 64, false, || Ok(())).unwrap();
    let calls = AtomicUsize::new(0);
    assert!(c
        .create(key(1), 1, false, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .is_err());
    assert!(c
        .create(key(2), 1, false, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.roots, s.raw, s.inflight), (64, 1, 1, 0));
}
#[test]
fn root_history_nil_keeps_all_history_and_combined_cache_slot() {
    let c = installed(128, 2);
    assert!(c
        .create::<()>(key(1), 64, true, || Err(fail("nil")))
        .is_err());
    {
        let s = c.lock().unwrap();
        assert_eq!(
            (
                s.bytes,
                s.roots,
                s.raw,
                s.cache_slots,
                s.published_cache,
                s.inflight,
                s.successful
            ),
            (64, 1, 1, 1, 0, 0, 0)
        );
    }
    c.create(key(1), 64, true, || Ok(())).unwrap();
    assert!(c.create(key(1), 1, true, || Ok(())).is_err());
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.cache_slots), (128, 2));
}
#[test]
fn root_history_staging_failure_spends_reservation_without_insert() {
    let c = installed(128, 2);
    let calls = AtomicUsize::new(0);
    let result = c.create::<()>(key(1), 32, true, || {
        // Same preconstructor-failure boundary as a failed RoPE reserve/staging.
        assert_eq!(c.lock().unwrap().cache_slots, 1);
        let staging: Result<()> = Err(fail("host staging failure before raw callback"));
        staging?;
        // The raw-constructor seam is reached only after staging succeeds.
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.cache_slots, s.published_cache), (32, 1, 0));
}
#[test]
fn root_history_postcreate_unknown_preserves_nonnil_ok() {
    let c = installed(64, 1);
    let result = c.create(key(1), 64, false, || {
        c.unknown("postcreation bookkeeping failure");
        Ok(42)
    });
    assert!(result.is_ok());
    assert!(c.health().is_err());
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.successful), (64, 1));
}
#[test]
fn root_history_unwind_is_sticky_no_refund() {
    let c = installed(64, 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        c.create::<()>(key(1), 64, false, || panic!("injected unwind"))
    }));
    assert!(result.is_err());
    assert!(c.health().is_err());
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.roots, s.inflight), (64, 1, 0));
}
#[test]
fn root_history_legacy_creation_and_import_forbid_late_install() {
    for import in [false, true] {
        let c = Core::default();
        if import {
            c.lock().unwrap().import();
        } else {
            assert!(matches!(
                c.create(key(1), 64, false, || Ok(())).unwrap().1,
                Origin::Legacy
            ));
        }
        assert!(c.lock().unwrap().install(limits(128, 2)).is_err());
    }
}
#[test]
fn root_history_install_creation_race_has_only_serialized_outcomes() {
    for _ in 0..32 {
        let c = Arc::new(Core::default());
        let barrier = Arc::new(Barrier::new(3));
        let a = {
            let c = c.clone();
            let b = barrier.clone();
            std::thread::spawn(move || {
                b.wait();
                let mut s = c.lock().unwrap();
                let ok = s.install(limits(64, 1)).is_ok();
                if ok {
                    s.register(key(1)).unwrap();
                }
                ok
            })
        };
        let b = {
            let c = c.clone();
            let b = barrier.clone();
            std::thread::spawn(move || {
                b.wait();
                c.create(key(1), 64, false, || Ok(())).is_ok()
            })
        };
        barrier.wait();
        let installed = a.join().unwrap();
        let created = b.join().unwrap();
        let s = c.lock().unwrap();
        if installed {
            assert!(!s.legacy);
            if created {
                assert_eq!(s.bytes, 64);
            } else {
                assert_eq!(s.bytes, 0);
            }
        } else {
            assert!(created);
            assert!(s.legacy);
            assert!(s.limits.is_none());
        }
    }
}
#[test]
fn root_history_global_parallel_devices_share_one_quota() {
    let c = Arc::new(installed(128, 2));
    c.lock().unwrap().register(key(2)).unwrap();
    let b = Arc::new(Barrier::new(5));
    let mut workers = Vec::new();
    for i in 0..4 {
        let c = c.clone();
        let b = b.clone();
        workers.push(std::thread::spawn(move || {
            b.wait();
            c.create(key(i % 2 + 1), 64, false, || Ok(())).is_ok()
        }));
    }
    b.wait();
    assert_eq!(
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .filter(|ok| *ok)
            .count(),
        2
    );
    let s = c.lock().unwrap();
    assert_eq!((s.bytes, s.roots), (128, 2));
}
#[test]
fn root_history_device_registration_is_bounded_and_deduplicated() {
    let c = installed(128, 2);
    let mut s = c.lock().unwrap();
    s.register(key(1)).unwrap();
    s.register(key(2)).unwrap();
    assert!(s.register(key(3)).is_err());
    assert_eq!(s.devices.len(), 2);
}
#[test]
fn root_history_unknown_import_heap_and_reinstall_fail_closed() {
    let c = installed(64, 1);
    assert!(c.lock().unwrap().heap().is_err());
    assert_eq!(c.lock().unwrap().roots, 0);
    c.lock().unwrap().import();
    assert!(c.create(key(1), 1, false, || Ok(())).is_err());
    assert!(!c.lock().unwrap().snapshot().covered);
    assert!(c.lock().unwrap().install(limits(64, 1)).is_err());
    let legacy = Core::default();
    legacy.lock().unwrap().heap().unwrap();
    assert!(legacy.lock().unwrap().install(limits(64, 1)).is_err());
}
#[test]
fn root_history_zero_overflow_and_poison_refuse_before_callback() {
    let c = installed(usize::MAX, 2);
    assert!(c.create(key(1), 0, false, || Ok(())).is_err());
    c.create(key(1), usize::MAX, false, || Ok(())).unwrap();
    assert!(c.create(key(1), 1, false, || Ok(())).is_err());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lock = c.state.lock().unwrap();
        panic!("mutex poison");
    }));
    assert!(result.is_err());
    assert!(c.health().is_err());
    assert!(c.create(key(1), 1, false, || Ok(())).is_err());
}

#[test]
fn root_history_postconstructor_publication_guard_retains_charge_on_failure() {
    for unwind in [false, true] {
        let c = installed(64, 1);
        c.create(key(1), 64, false, || Ok(())).unwrap();
        if unwind {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _publication = PublicationGuard::new(&c);
                panic!("postconstructor failure");
            }));
        } else {
            drop(PublicationGuard::new(&c));
        }
        assert!(c.health().is_err());
        let s = c.lock().unwrap();
        assert_eq!((s.bytes, s.roots, s.successful), (64, 1, 1));
    }
    let c = installed(64, 1);
    c.create(key(1), 64, false, || Ok(())).unwrap();
    PublicationGuard::new(&c).complete();
    assert!(c.health().is_ok());
}

#[test]
fn root_history_distinct_byte_root_raw_and_cache_caps_are_atomic() {
    for field in 0..4 {
        let c = Core::default();
        let mut limit = limits(128, 4);
        match field {
            0 => limit.max_attempted_requested_bytes = 0,
            1 => limit.max_root_attempts = 0,
            2 => limit.max_raw_constructor_attempts = 0,
            _ => limit.max_rope_cache_entries = 0,
        }
        {
            let mut s = c.lock().unwrap();
            s.install(limit).unwrap();
            s.register(key(1)).unwrap();
        }
        let calls = AtomicUsize::new(0);
        assert!(c
            .create(key(1), 32, true, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let s = c.lock().unwrap();
        assert_eq!(
            (s.bytes, s.roots, s.raw, s.cache_slots, s.inflight),
            (0, 0, 0, 0, 0)
        );
    }
}
#[test]
fn root_history_nonnil_with_completion_mutex_poison_still_returns_ok() {
    let c = installed(64, 1);
    let result = c.create(key(1), 64, false, || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lock = c.state.lock().unwrap();
            panic!("completion bookkeeping poison");
        }));
        Ok(())
    });
    assert!(result.is_ok());
    assert!(c.health().is_err());
    let state = c.state.lock().unwrap_err().into_inner();
    assert_eq!((state.bytes, state.roots, state.inflight), (64, 1, 1));
}
#[test]
fn root_history_legacy_fast_path_only_after_permanent_state_transition() {
    let c = Core::default();
    assert!(!c.legacy_fast.load(Ordering::Acquire));
    c.create(key(1), 32, false, || {
        assert!(c.legacy_fast.load(Ordering::Acquire));
        assert!(c.lock().unwrap().install(limits(64, 1)).is_err());
        Ok(())
    })
    .unwrap();
    assert!(matches!(
        c.create(key(2), 64, false, || Ok(())).unwrap().1,
        Origin::Legacy
    ));
    assert!(c.lock().unwrap().legacy);
}
