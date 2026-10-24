use super::state::{bucket, reset_allowed, RootState, Status};
use super::*;

#[test]
fn checked_bucket_rejects_zero_and_overflow() {
    assert_eq!(bucket(0), Err(BoundedPoolError::InvalidRequest));
    assert_eq!(bucket(usize::MAX), Err(BoundedPoolError::Overflow));
    assert_eq!(bucket(1025), Ok(2048));
}
#[test]
fn release_then_reset_has_single_free_membership() {
    let mut state = RootState::fresh();
    state.release(1).unwrap();
    state.reset();
    assert_eq!(state.status, Status::Free);
    assert_eq!(state.issue(), Ok(2));
    assert_eq!(state.issue(), Err(BoundedPoolError::InvalidLease));
}
#[test]
fn stale_and_duplicate_release_do_not_change_issue() {
    let mut state = RootState::fresh();
    assert_eq!(state.release(2), Err(BoundedPoolError::InvalidLease));
    assert_eq!(state.status, Status::Issued(1));
    state.release(1).unwrap();
    assert_eq!(state.release(1), Err(BoundedPoolError::InvalidLease));
    state.issue().unwrap();
    assert_eq!(state.release(1), Err(BoundedPoolError::InvalidLease));
    assert_eq!(state.status, Status::Issued(2));
}
#[test]
fn live_lease_blocks_reset_not_just_root_return() {
    assert_eq!(reset_allowed(1), Err(BoundedPoolError::LiveLeases));
    assert_eq!(reset_allowed(usize::MAX), Err(BoundedPoolError::LiveLeases));
    assert_eq!(reset_allowed(0), Ok(()));
}
#[test]
fn retired_root_never_reissues() {
    let mut state = RootState::fresh();
    state.status = Status::Retired;
    state.reset();
    assert_eq!(state.issue(), Err(BoundedPoolError::InvalidLease));
    assert_eq!(state.status, Status::Retired);
}
fn limits() -> ProcessPoolLimits {
    ProcessPoolLimits {
        max_domains: 2,
        max_reserved_heap_bytes: 131072,
        max_child_creations: 8,
        max_constructor_attempts: 12,
        max_rank: 4,
        max_batch: 4,
    }
}
#[test]
fn constructor_attempt_quota_is_monotonic() {
    let mut config = limits();
    config.max_constructor_attempts = 1;
    let authority = ProcessPoolAuthority::test_authority(config);
    let mut ledger = authority.ledger.lock().unwrap();
    authority.attempt(&mut ledger).unwrap();
    assert_eq!(authority.attempt(&mut ledger), Err(BoundedPoolError::Quota));
    assert_eq!(ledger.snapshot.constructor_attempts, 1);
    ledger.poisoned = true;
    assert_eq!(
        authority.attempt(&mut ledger),
        Err(BoundedPoolError::Poisoned)
    );
}

/// Fresh process only: successful execution installs the one global authority.
#[test]
#[ignore = "bounded 64 KiB optional pool CPU/GPU control; external process watchdog required"]
fn tiny_tracked_pool_transitions_and_gpu_completion() {
    // Exercise public singleton installation in a fresh exec, never reset it.
    const CHILD: &str = "MLX_BOUNDED_POOL_CONTROL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "bounded_pool::tests::tiny_tracked_pool_transitions_and_gpu_completion",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        print!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success(), "fresh-process control failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stdout.contains("running 1 test"),
            "exact child filter matched no test"
        );
        assert!(
            stderr.contains("BOUNDED_POOL_CONTROL completed")
                || stderr.contains("BOUNDED_POOL_CONTROL unavailable_or_unknown"),
            "missing explicit child capability outcome"
        );
        return;
    }
    use crate::ops::elementwise::{elementwise_add, elementwise_mul};
    use crate::{DType, KernelRegistry, MlxDevice};
    for name in [
        "MLX_UNRETAINED_REFS",
        "MLX_PROFILE_DISPATCH",
        "MLX_PROFILE_CB",
        "MLX_ENCODE_TRACE",
        "MLX_PROFILE_BARRIERS",
        "HF2Q_AUTO_BARRIER",
        "HF2Q_FORCE_SERIAL_DISPATCH",
        "HF2Q_PIPELINE_TG_MULT_HINT",
    ] {
        assert!(std::env::var_os(name).is_none(), "requires {name} absent");
    }
    objc::rc::autoreleasepool(|| {
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("BOUNDED_POOL_CONTROL unavailable_or_unknown device={error}");
                return;
            }
        };
        let authority = ProcessPoolAuthority::install(limits()).unwrap();
        assert!(matches!(
            ProcessPoolAuthority::install(limits()),
            Err(BoundedPoolError::AlreadyInstalled)
        ));
        let config = BoundedPoolConfig {
            requested_heap_bytes: 65536,
            reserved_heap_bytes: 65536,
            max_child_creations: 4,
        };
        let mut pool = match authority.create_pool(&device, config) {
            Ok(pool) => pool,
            Err(error @ (BoundedPoolError::Unsupported | BoundedPoolError::AllocationFailed)) => {
                eprintln!("BOUNDED_POOL_CONTROL unavailable_or_unknown heap={error}");
                return;
            }
            Err(error) => panic!("unexpected constructor refusal: {error}"),
        };
        assert_eq!(pool.snapshot().metadata_upper_bound_known, false);
        assert!(std::ptr::eq(
            ProcessPoolAuthority::get().unwrap(),
            authority
        ));
        eprintln!("BOUNDED_POOL_IDENTITY actual_heap_bytes={} actual_hazard_mode=Tracked requested_heap_bytes=65536 runtime_admission=false", pool.snapshot().observed_heap_bytes);
        let mut leases = Vec::new();
        for index in 0..4 {
            let mut lease = pool.alloc(&device, 1024, DType::F32, &[256]).unwrap();
            assert!(lease
                .buffer()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.to_bits() == 0));
            for (i, value) in lease
                .buffer_mut()
                .as_mut_slice::<f32>()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *value = match index {
                    0 => ((i % 17) as i32 - 8) as f32,
                    1 => 2.,
                    2 => -99.,
                    _ => -77.,
                };
            }
            leases.push(lease);
        }
        // No GPU work has started; a live token alone must prevent reset.
        assert_eq!(
            unsafe { pool.try_reset(&device) },
            Err(BoundedPoolError::LiveLeases)
        );
        assert!(matches!(
            pool.alloc(&device, 1024, DType::F32, &[256]),
            Err(BoundedPoolError::Quota)
        ));
        assert_eq!(
            pool.register_existing(&device, leases[0].buffer()),
            Err(BoundedPoolError::ImportUnsupported)
        );
        let mut registry = KernelRegistry::new();
        let mut first = device.command_encoder().unwrap();
        let mut second = device.command_encoder().unwrap();
        elementwise_add(
            &mut first,
            &mut registry,
            device.metal_device(),
            leases[0].buffer(),
            leases[1].buffer(),
            leases[2].buffer(),
            256,
            DType::F32,
        )
        .unwrap();
        elementwise_mul(
            &mut second,
            &mut registry,
            device.metal_device(),
            leases[2].buffer(),
            leases[1].buffer(),
            leases[3].buffer(),
            256,
            DType::F32,
        )
        .unwrap();
        // All fallible encoding precedes either submission. Owners stay live.
        first.commit();
        second.commit();
        let second_result = second.wait_until_completed();
        let first_result = first.wait_until_completed();
        eprintln!(
            "BOUNDED_POOL_COMMANDS cb1_ok={} cb2_ok={}",
            first_result.is_ok(),
            second_result.is_ok()
        );
        second_result.unwrap();
        first_result.unwrap();
        for i in 0..256 {
            let a = ((i % 17) as i32 - 8) as f32;
            for (index, expected) in [a, 2., a + 2., 2. * (a + 2.)].iter().enumerate() {
                assert_eq!(
                    leases[index].buffer().as_slice::<f32>().unwrap()[i].to_bits(),
                    expected.to_bits()
                );
            }
        }
        drop(second);
        drop(first);
        // After waits and readback, no escaped wrappers/raw/recorded handles exist.
        let survivor = leases.pop().unwrap();
        unsafe { pool.try_release(&device, survivor) }.unwrap();
        assert_eq!(pool.snapshot().free_roots, 1);
        assert_eq!(
            unsafe { pool.try_reset(&device) },
            Err(BoundedPoolError::LiveLeases)
        );
        drop(leases);
        unsafe { pool.try_reset(&device) }.unwrap();
        assert_eq!(pool.snapshot().free_roots, 4);
        let reused = pool.alloc(&device, 1024, DType::F32, &[256]).unwrap();
        assert_eq!(pool.snapshot().successful_child_creations, 4);
        drop(reused);
        unsafe { pool.try_reset(&device) }.unwrap();
        pool.clear_free(&device).unwrap();
        assert_eq!(pool.snapshot().retired_roots, 4);
        assert!(matches!(
            pool.alloc(&device, 1024, DType::F32, &[256]),
            Err(BoundedPoolError::Quota)
        ));
        let charge = authority.snapshot().unwrap();
        drop(pool);
        assert_eq!(
            authority.snapshot().unwrap().reserved_heap_bytes,
            charge.reserved_heap_bytes
        );
        assert_eq!(
            authority.snapshot().unwrap().reserved_child_slots,
            charge.reserved_child_slots
        );
        assert_eq!(authority.snapshot().unwrap().retired_domains, 1);
        // A second domain consumes remaining aggregate capacity; dropping it
        // cannot permit a third domain or refund its heap reservation.
        let second = authority.create_pool(&device, config).unwrap();
        drop(second);
        assert!(matches!(
            authority.create_pool(&device, config),
            Err(BoundedPoolError::Quota)
        ));
        eprintln!("BOUNDED_POOL_CONTROL completed exact_cpu_parity=true transitions=true runtime_admission=false metadata_upper_bound_known=false");
    });
}
