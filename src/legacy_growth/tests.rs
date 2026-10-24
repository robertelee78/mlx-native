use super::{ledger::Ledger, *};
use std::sync::{Arc, Barrier, Mutex};
fn quota(bytes: usize, roots: usize, attempts: usize) -> GrowthQuota {
    GrowthQuota {
        bytes,
        roots,
        attempts,
    }
}
fn limits() -> LegacyGrowthLimits {
    LegacyGrowthLimits {
        quota: quota(256, 4, 6),
        max_pools: 4,
        max_buckets_per_pool: 2,
        max_rank: 4,
        max_batch: 4,
    }
}
fn key(kind: u64) -> GrowthPoolKey {
    GrowthPoolKey {
        kind,
        worker_generation: 7,
    }
}
fn buckets() -> [GrowthBucket; 2] {
    [
        GrowthBucket {
            bytes: 64,
            quota: quota(256, 4, 6),
        },
        GrowthBucket {
            bytes: 128,
            quota: quota(256, 2, 6),
        },
    ]
}
fn registered() -> Ledger {
    let mut l = Ledger::prepare(limits()).unwrap();
    l.register(key(0), limits().quota, &buckets()).unwrap();
    l
}
#[test]
fn shared_cross_pool_parallel_precharge_is_atomic() {
    let mut l = registered();
    l.register(key(1), limits().quota, &buckets()).unwrap();
    let shared = Arc::new(Mutex::new(l));
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|pool| {
            let shared = shared.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                shared
                    .lock()
                    .unwrap()
                    .construct(pool, 128, || Ok(()))
                    .is_ok()
            })
        })
        .collect();
    barrier.wait();
    let outcomes: Vec<_> = workers.into_iter().map(|w| w.join()).collect();
    assert!(outcomes.into_iter().all(|result| result.unwrap()));
    let mut l = shared.lock().unwrap();
    let before = l.snapshot(0);
    let mut called = false;
    assert!(l
        .construct(0, 64, || {
            called = true;
            Ok(())
        })
        .is_err());
    assert!(!called);
    assert_eq!(l.snapshot(0), before);
    assert_eq!(
        before.process,
        GrowthCounts {
            bytes: 256,
            roots: 2,
            attempts: 2
        }
    );
}
#[test]
fn nil_keeps_attempt_and_success_before_later_failure_keeps_history() {
    let mut l = registered();
    assert!(l.construct::<()>(0, 64, || Err(fail("nil"))).is_err());
    assert_eq!(
        l.snapshot(0).pool,
        GrowthCounts {
            bytes: 0,
            roots: 0,
            attempts: 1
        }
    );
    let postcreation: Result<()> = l.construct(0, 64, || Ok(())).and_then(|_| {
        Err(fail(
            "fallible consumer after successful constructor boundary",
        ))
    });
    assert!(postcreation.is_err());
    assert_eq!(
        l.snapshot(0).pool,
        GrowthCounts {
            bytes: 64,
            roots: 1,
            attempts: 2
        }
    );
}
#[test]
fn attempts_bucket_roots_and_bytes_each_refuse_without_constructor() {
    for q in [quota(63, 4, 6), quota(256, 0, 6), quota(256, 4, 0)] {
        let mut l = Ledger::prepare(limits()).unwrap();
        let bs = [GrowthBucket {
            bytes: 64,
            quota: q,
        }];
        l.register(key(0), limits().quota, &bs).unwrap();
        let before = l.snapshot(0);
        let mut called = false;
        assert!(l
            .construct(0, 64, || {
                called = true;
                Ok(())
            })
            .is_err());
        assert!(!called);
        assert_eq!(l.snapshot(0), before);
    }
    let mut l = registered();
    for _ in 0..6 {
        assert!(l.construct::<()>(0, 64, || Err(fail("nil"))).is_err());
    }
    let before = l.snapshot(0);
    let mut called = false;
    assert!(l
        .construct(0, 64, || {
            called = true;
            Ok(())
        })
        .is_err());
    assert!(!called);
    assert_eq!(l.snapshot(0), before);
}
#[test]
fn identity_registration_quarantine_and_overflow_are_fail_closed() {
    let mut l = registered();
    assert!(l.register(key(0), limits().quota, &buckets()).is_err());
    assert!(l.construct(0, 32, || Ok(())).is_err());
    l.quarantine(7).unwrap();
    assert!(l.validate(0, 64).is_err());
    assert!(l.register(key(2), limits().quota, &buckets()).is_err());
    assert!(l.snapshot(0).quarantined);
    assert!(checked_bucket(usize::MAX).is_err());
    let mut l = registered();
    l.counts.attempts = usize::MAX;
    let before = l.snapshot(0);
    assert!(l.construct(0, 64, || Ok(())).is_err());
    assert_eq!(l.snapshot(0), before);
}
#[test]
fn fixed_registration_metadata_and_no_creation_refund() {
    let mut l = registered();
    l.construct(0, 64, || Ok(())).unwrap();
    // No clear/reset/drop/restart credit operation exists in the ledger.
    for n in 1..4 {
        l.register(key(n), limits().quota, &buckets()).unwrap();
    }
    assert!(l.register(key(5), limits().quota, &buckets()).is_err());
    assert_eq!(l.snapshot(0).process.bytes, 64);
    let mut duplicate = buckets();
    duplicate[1] = duplicate[0];
    assert!(registered()
        .configuration(key(1), limits().quota, &duplicate)
        .is_err());
}
#[test]
fn constructor_panic_preserves_precharge_and_mutex_poison_refuses() {
    let lock = Mutex::new(registered());
    let result = std::panic::catch_unwind(|| {
        lock.lock()
            .unwrap()
            .construct::<()>(0, 64, || panic!("constructor interrupted"))
            .unwrap();
    });
    assert!(result.is_err());
    assert!(lock.is_poisoned());
    let poisoned = match lock.lock() {
        Err(error) => error.into_inner(),
        Ok(_) => panic!("expected poison"),
    };
    assert_eq!(
        poisoned.snapshot(0).pool,
        GrowthCounts {
            bytes: 64,
            roots: 1,
            attempts: 1
        }
    );
}

#[test]
fn process_pool_and_bucket_each_enforce_all_three_ceilings() {
    for scope in 0..3 {
        for ceiling in 0..3 {
            let mut cap = quota(256, 4, 6);
            match ceiling {
                0 => cap.bytes = 63,
                1 => cap.roots = 0,
                _ => cap.attempts = 0,
            }
            let mut config = limits();
            if scope == 0 {
                config.quota = cap;
            }
            let pool = if scope == 1 { cap } else { config.quota };
            let bucket = if scope == 2 { cap } else { pool };
            let bs = [GrowthBucket {
                bytes: 64,
                quota: bucket,
            }];
            let mut l = Ledger::prepare(config).unwrap();
            l.register(key(0), pool, &bs).unwrap();
            let before = l.snapshot(0);
            let mut called = false;
            assert!(l
                .construct(0, 64, || {
                    called = true;
                    Ok(())
                })
                .is_err());
            assert!(!called, "scope={scope} ceiling={ceiling}");
            assert_eq!(l.snapshot(0), before);
        }
    }
}

#[test]
fn quarantine_registry_exhaustion_refuses_every_pool_but_keeps_receipts() {
    let mut l = registered();
    for generation in 100..104 {
        l.quarantine(generation).unwrap();
    }
    assert!(l.quarantine(104).is_err());
    assert!(l.snapshot(0).quarantined);
    assert!(l.validate(0, 64).is_err());
    assert!(l.register(key(1), limits().quota, &buckets()).is_err());
}
