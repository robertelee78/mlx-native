//! Injected failures use the production reservation/constructor decision path.
//! Private leaked authorities isolate tiny domains; public singleton is exercised
//! separately in a fresh subprocess. No production independent constructor exists.
use super::resource::faults::{self, Faults};
use super::*;
use crate::{DType, MlxDevice};
use metal::foreign_types::ForeignType;

fn limits() -> ProcessPoolLimits {
    ProcessPoolLimits {
        max_domains: 2,
        max_reserved_heap_bytes: 131072,
        max_child_creations: 16,
        max_constructor_attempts: 8,
        max_rank: 4,
        max_batch: 4,
    }
}
fn authority() -> &'static ProcessPoolAuthority {
    Box::leak(Box::new(ProcessPoolAuthority::test_authority(limits())))
}
fn config() -> BoundedPoolConfig {
    BoundedPoolConfig {
        requested_heap_bytes: 65536,
        reserved_heap_bytes: 65536,
        max_child_creations: 8,
    }
}
fn pool(
    authority: &'static ProcessPoolAuthority,
    device: &MlxDevice,
) -> Option<BoundedMlxBufferPool> {
    match authority.create_pool(device, config()) {
        Ok(pool) => Some(pool),
        Err(error @ (BoundedPoolError::Unsupported | BoundedPoolError::AllocationFailed)) => {
            eprintln!("BOUNDED_POOL_FAILURE_CONTROL unavailable_or_unknown {error}");
            None
        }
        Err(error) => panic!("unexpected {error}"),
    }
}
#[test]
#[ignore = "tiny Metal injected failure/identity/ownership controls; no GPU/model"]
fn tiny_constructor_identity_and_retention_controls() {
    objc::rc::autoreleasepool(|| {
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("BOUNDED_POOL_FAILURE_CONTROL unavailable_or_unknown {error}");
                return;
            }
        };
        // Repeated heap nil consumes attempts, rolls back only unpublished reservation.
        let nil_authority = authority();
        faults::set(Faults {
            heap_nil: true,
            ..Default::default()
        });
        for _ in 0..8 {
            assert!(matches!(
                nil_authority.create_pool(&device, config()),
                Err(BoundedPoolError::AllocationFailed)
            ));
        }
        assert!(matches!(
            nil_authority.create_pool(&device, config()),
            Err(BoundedPoolError::Quota)
        ));
        let nil = nil_authority.snapshot().unwrap();
        assert_eq!(
            (
                nil.domains,
                nil.reserved_heap_bytes,
                nil.reserved_child_slots
            ),
            (0, 0, 0)
        );
        assert_eq!(nil.constructor_attempts, 8);
        assert_eq!(faults::get().heap_calls, 8);
        assert_eq!(faults::get().child_calls, 0);

        // Nonnil tiny heap, forced predicate mismatch: retain/quarantine, never
        // enlarge the reservation or allocate another object to "recover".
        faults::set(Faults {
            invalid_heap: true,
            ..Default::default()
        });
        let breach_authority = authority();
        match breach_authority.create_pool(&device, config()) {
            Err(BoundedPoolError::EnvelopeBreached) => (),
            Err(BoundedPoolError::Unsupported | BoundedPoolError::AllocationFailed) => {
                eprintln!("BOUNDED_POOL_FAILURE_CONTROL unavailable_or_unknown breach_heap");
                return;
            }
            _ => panic!("expected quarantine"),
        }
        let quarantined = breach_authority.snapshot().unwrap();
        assert_eq!(quarantined.domains, 1);
        assert_eq!(quarantined.retired_domains, 1);
        assert!(quarantined.observed_heap_bytes > 0);
        assert!(breach_authority.ledger.lock().unwrap().domains[0]
            .heap
            .is_some());
        assert!(matches!(
            breach_authority.create_pool(&device, config()),
            Err(BoundedPoolError::Poisoned)
        ));
        assert_eq!(faults::get().heap_calls, 1);

        faults::set(Faults {
            dirty_child: true,
            ..Default::default()
        });
        let authority = authority();
        let Some(mut first) = pool(authority, &device) else {
            return;
        };
        let Some(mut second) = pool(authority, &device) else {
            return;
        };
        let before = authority.snapshot().unwrap();
        // Invalid pure inputs cannot reach either Metal constructor seam.
        for (bytes, shape) in [
            (0, &[0][..]),
            (usize::MAX, &[usize::MAX][..]),
            (7, &[8][..]),
            (4, &[1, 1, 1, 1, 1][..]),
        ] {
            assert!(first.alloc(&device, bytes, DType::U8, shape).is_err());
        }
        assert_eq!(authority.snapshot().unwrap(), before);
        assert_eq!(faults::get().child_calls, 0);
        let mut lease = first.alloc(&device, 37, DType::U8, &[37]).unwrap();
        assert_eq!(lease.buffer().byte_len(), 64);
        assert_eq!(lease.buffer().data_byte_len(), 37);
        // Seam dirtied full capacity, so this proves actual fresh zeroing,
        // not a lucky zero-filled Metal allocation. All 64 bytes initialized.
        let full =
            unsafe { std::slice::from_raw_parts(lease.buffer().contents_ptr().cast::<u8>(), 64) };
        assert!(full.iter().all(|byte| *byte == 0));
        lease.buffer_mut().as_mut_slice::<u8>().unwrap().fill(0x33);
        let identity = lease.buffer().metal_buffer().as_ptr();
        let first_before = first.snapshot();
        let wrong = unsafe { second.try_release(&device, lease) }.unwrap_err();
        assert_eq!(wrong.error, BoundedPoolError::InvalidLease);
        assert_eq!(first.snapshot(), first_before);
        assert_eq!(wrong.lease.buffer().metal_buffer().as_ptr(), identity);
        let before_wrong_queue = first.snapshot();
        let wrong_queue = device.bounded_pool_test_other_queue();
        let failed = unsafe { first.try_release(&wrong_queue, wrong.lease) }.unwrap_err();
        assert_eq!(failed.error, BoundedPoolError::IdentityMismatch);
        assert_eq!(first.snapshot(), before_wrong_queue);
        // Public mutable access can replace the wrapped object without changing
        // the private issue token. Reject that identity and return it intact.
        let foreign = device.alloc_buffer(64, DType::U8, vec![64]).unwrap();
        let foreign_identity = foreign.metal_buffer().as_ptr();
        let mut replaced_lease = failed.lease;
        let original = std::mem::replace(replaced_lease.buffer_mut(), foreign);
        let before_replaced = first.snapshot();
        let rejected = unsafe { first.try_release(&device, replaced_lease) }.unwrap_err();
        assert_eq!(rejected.error, BoundedPoolError::InvalidLease);
        assert_eq!(first.snapshot(), before_replaced);
        assert_eq!(
            rejected.lease.buffer().metal_buffer().as_ptr(),
            foreign_identity
        );
        let mut restored_lease = rejected.lease;
        let foreign = std::mem::replace(restored_lease.buffer_mut(), original);
        assert_eq!(restored_lease.buffer().metal_buffer().as_ptr(), identity);
        drop(foreign);
        unsafe { first.try_release(&device, restored_lease) }.unwrap();
        unsafe { first.try_reset(&device) }.unwrap();
        let free_before = first.snapshot();
        let different_queue = device.bounded_pool_test_other_queue();
        assert!(matches!(
            first.alloc(&different_queue, 37, DType::U8, &[37]),
            Err(BoundedPoolError::IdentityMismatch)
        ));
        assert_eq!(
            unsafe { first.try_reset(&different_queue) },
            Err(BoundedPoolError::IdentityMismatch)
        );
        assert_eq!(
            first.clear_free(&different_queue),
            Err(BoundedPoolError::IdentityMismatch)
        );
        if device.residency_sets_enabled() {
            let none = device.bounded_pool_test_without_residency();
            assert!(matches!(
                first.alloc(&none, 37, DType::U8, &[37]),
                Err(BoundedPoolError::IdentityMismatch)
            ));
        }
        assert_eq!(first.snapshot(), free_before);
        assert_eq!(faults::get().child_calls, 1);
        // Simulate observable property drift without mutating a live resource.
        faults::set(Faults {
            invalid_resource: true,
            ..faults::get()
        });
        assert!(matches!(
            first.alloc(&device, 37, DType::U8, &[37]),
            Err(BoundedPoolError::ResourceMismatch)
        ));
        assert_eq!(first.snapshot(), free_before);
        faults::set(Faults {
            invalid_resource: false,
            ..faults::get()
        });
        let reused = first.alloc(&device.clone(), 37, DType::U8, &[37]).unwrap();
        assert_eq!(reused.buffer().metal_buffer().as_ptr(), identity);
        assert!(reused
            .buffer()
            .as_slice::<u8>()
            .unwrap()
            .iter()
            .all(|byte| *byte == 0x33));
        // A simultaneous lease cannot receive the same object (release+reset
        // has not duplicated free membership).
        let other = first.alloc(&device, 37, DType::U8, &[37]).unwrap();
        assert_ne!(other.buffer().metal_buffer().as_ptr(), identity);
        assert_eq!(
            unsafe { first.try_reset(&device) },
            Err(BoundedPoolError::LiveLeases)
        );
        drop(other);
        // One remaining live lease still blocks whole-domain reset unchanged.
        let live_before = first.snapshot();
        assert_eq!(
            unsafe { first.try_reset(&device) },
            Err(BoundedPoolError::LiveLeases)
        );
        assert_eq!(first.snapshot(), live_before);
        let raw = reused.buffer().metal_buffer().clone();
        let recorded = crate::RecordedBinding::Buffer {
            metal_buffer: raw.clone(),
            offset: 0,
        };
        drop(reused);
        // No GPU work or later alias accesses; raw/recorded are lifetime-only
        // survivors. Reset/clear cannot refund or remove authority membership.
        unsafe { first.try_reset(&device) }.unwrap();
        let membership_before_clear = crate::residency::residency_allocation_count_for_test();
        first.clear_free(&device).unwrap();
        assert_eq!(
            crate::residency::residency_allocation_count_for_test(),
            membership_before_clear
        );
        let membership_before = crate::residency::residency_allocation_count_for_test();
        let charge = authority.snapshot().unwrap();
        drop(first);
        assert_eq!(
            crate::residency::residency_allocation_count_for_test(),
            membership_before
        );
        assert_eq!(
            authority.snapshot().unwrap().reserved_heap_bytes,
            charge.reserved_heap_bytes
        );
        let ledger = authority.ledger.lock().unwrap();
        assert_eq!(ledger.domains[0].children.len(), 2);
        assert_eq!(raw.length(), 64); // still live; no freed-pointer query
        assert_eq!(ledger.domains[0].children[0].as_ptr(), identity);
        drop(ledger);
        drop(raw);
        drop(recorded);

        // Late batch nil uses production call seam, publishes no prefix, and
        // retains successful roots/history. No hidden global/second-heap retry.
        let calls = faults::get().child_calls;
        faults::set(Faults {
            child_nil_at: calls + 2,
            ..faults::get()
        });
        let requests = [
            PoolRequest {
                bytes: 128,
                dtype: DType::U8,
                shape: &[128],
            },
            PoolRequest {
                bytes: 256,
                dtype: DType::U8,
                shape: &[256],
            },
        ];
        assert!(matches!(
            second.alloc_batch(&device, &requests),
            Err(BoundedPoolError::AllocationFailed)
        ));
        assert_eq!(second.snapshot().live_leases, 0);
        assert_eq!(second.snapshot().free_roots, 1);
        assert_eq!(second.snapshot().successful_child_creations, 1);
        assert_eq!(faults::get().child_calls, calls + 2);
        assert_eq!(faults::get().heap_calls, 2);
        // Actual fixed-heap exhaustion, not injected nil: the retained 128-byte
        // child prevents a full 64KiB child from fitting. Request remains tiny.
        assert!(matches!(
            second.alloc(&device, 65536, DType::U8, &[65536]),
            Err(BoundedPoolError::AllocationFailed)
        ));
        assert_eq!(faults::get().child_calls, calls + 3);
        assert_eq!(faults::get().heap_calls, 2);
        drop(second);
        faults::set(Faults::default());
        let none_device = device.bounded_pool_test_without_residency();
        let none_authority = self::authority();
        let Some(mut none_pool) = pool(none_authority, &none_device) else {
            return;
        };
        let membership_before_none = crate::residency::residency_allocation_count_for_test();
        let none_lease = none_pool.alloc(&none_device, 64, DType::U8, &[64]).unwrap();
        drop(none_lease);
        unsafe { none_pool.try_reset(&none_device) }.unwrap();
        none_pool.clear_free(&none_device).unwrap();
        drop(none_pool);
        assert_eq!(
            crate::residency::residency_allocation_count_for_test(),
            membership_before_none
        );
        eprintln!("BOUNDED_POOL_RESIDENCY active_owner={} explicit_none=true membership_retained=true os_acknowledgement=false", device.residency_sets_enabled());
        eprintln!("BOUNDED_POOL_FAILURE_CONTROL completed nil=true quarantine=true prevalidation=true identity=true unchanged_failure=true metadata_leases=true batch=true retained_children=true runtime_admission=false");
        faults::set(Faults::default());
    });
}
