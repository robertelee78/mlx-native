use super::*;
fn quota(bytes: usize, roots: usize, attempts: usize) -> GrowthQuota {
    GrowthQuota {
        bytes,
        roots,
        attempts,
    }
}
fn key(kind: u64) -> GrowthPoolKey {
    GrowthPoolKey {
        kind,
        worker_generation: 7,
    }
}
/// Fresh subprocess only: installs the one real process authority. CPU Metal
/// allocation/readback only, no encoders/dispatch/models. Requested new payload
/// is at most 192 bytes (64+128), excluding driver/allocator metadata.
#[test]
#[ignore = "tiny Metal allocation control; run exact test in isolated process"]
fn actual_legacy_aababa_growth_gate() {
    use crate::{DType, MlxDevice};
    objc::rc::autoreleasepool(|| {
        let device = MlxDevice::new().expect("Metal required; absence is not a completed control");
        let q = quota(192, 2, 3);
        let limits = LegacyGrowthLimits {
            quota: q,
            max_pools: 2,
            max_buckets_per_pool: 2,
            max_rank: 2,
            max_batch: 2,
        };
        let authority = LegacyPoolGrowthAuthority::install(limits).unwrap();
        assert!(LegacyPoolGrowthAuthority::install(limits).is_err());
        let bs = [
            GrowthBucket {
                bytes: 64,
                quota: quota(64, 1, 2),
            },
            GrowthBucket {
                bytes: 128,
                quota: quota(128, 1, 2),
            },
        ];
        let mut pool = authority.bind(&device, key(0), q, &bs).unwrap();
        let mut identities = [0usize; 2];
        for (round, class) in [0usize, 0, 1, 0, 1, 0].into_iter().enumerate() {
            let bytes = [64, 128][class];
            let buffer = if round == 5 {
                pool.alloc_uninitialized(&device, bytes, DType::U8, vec![bytes])
                    .unwrap()
            } else {
                pool.alloc(&device, bytes, DType::U8, vec![bytes]).unwrap()
            };
            let raw = buffer.metal_buffer();
            assert_eq!(raw.storage_mode(), metal::MTLStorageMode::Shared);
            assert_eq!(raw.length() as usize, bytes);
            assert!(!raw.contents().is_null());
            let identity = &**raw as *const metal::BufferRef as usize;
            // Entire payload was initialized by fresh alloc or this test's prior write.
            let contents =
                unsafe { std::slice::from_raw_parts_mut(raw.contents() as *mut u8, bytes) };
            if identities[class] == 0 {
                assert!(contents.iter().all(|v| *v == 0));
                identities[class] = identity;
                contents.fill((class + 17) as u8);
            } else {
                assert_eq!(identity, identities[class]);
                assert!(contents.iter().all(|v| *v == (class + 17) as u8));
            }
            drop(buffer);
            // CPU-only test; all returned owners dropped and no recorded/GPU uses.
            unsafe {
                pool.reset(&device).unwrap();
            }
            let s = pool.snapshot().unwrap();
            assert_eq!(s.pool.roots, if round < 2 { 1 } else { 2 });
        }
        assert_ne!(identities[0], identities[1]);
        let before = pool.snapshot().unwrap();
        let other_queue = device.bounded_pool_test_other_queue();
        assert!(pool.alloc(&other_queue, 64, DType::U8, vec![64]).is_err());
        let other_residency = device.bounded_pool_test_without_residency();
        if device.residency_sets_enabled() {
            assert!(pool
                .alloc(&other_residency, 64, DType::U8, vec![64])
                .is_err());
        }
        assert!(pool.alloc(&device, 64, DType::U8, vec![1, 1, 64]).is_err());
        assert!(pool.alloc(&device, usize::MAX, DType::U8, vec![1]).is_err());
        assert!(pool.alloc(&device, 256, DType::U8, vec![256]).is_err());
        assert!(pool
            .alloc_batch(&device, &vec![(64, DType::U8, vec![64]); 3])
            .is_err());
        assert_eq!(pool.snapshot().unwrap(), before);
        // Wrong-thread warm reuse must fail before touching a free bin.
        let device_copy = device.clone();
        pool = std::thread::spawn(move || {
            assert!(pool.alloc(&device_copy, 64, DType::U8, vec![64]).is_err());
            pool
        })
        .join()
        .unwrap();
        assert_eq!(pool.snapshot().unwrap(), before);
        let pair = pool
            .alloc_batch(
                &device,
                &[(64, DType::U8, vec![64]), (128, DType::U8, vec![128])],
            )
            .unwrap();
        for (buffer, expected) in pair.iter().zip(identities) {
            assert_eq!(
                &**buffer.metal_buffer() as *const metal::BufferRef as usize,
                expected
            );
        }
        assert!(pool.clear(&device).is_err());
        drop(pair);
        unsafe {
            pool.reset(&device).unwrap();
        }
        // First warm handout succeeds; second same-bucket miss is refused. Its
        // earlier handout is dropped by alloc_batch but still arena-tracked.
        assert!(pool
            .alloc_batch(
                &device,
                &[(64, DType::U8, vec![64]), (64, DType::U8, vec![64])]
            )
            .is_err());
        assert!(pool.clear(&device).is_err());
        unsafe {
            pool.reset(&device).unwrap();
        }
        assert_eq!(pool.snapshot().unwrap(), before);
        pool.clear(&device).unwrap();
        assert_eq!(pool.snapshot().unwrap(), before);
        // Both buckets now cold but historical process bytes/roots are exhausted.
        assert!(pool.alloc(&device, 64, DType::U8, vec![64]).is_err());
        assert_eq!(pool.snapshot().unwrap(), before);
        drop(pool);
        assert_eq!(authority.snapshot(key(0)).unwrap(), before);
        assert!(authority.bind(&device, key(0), q, &bs).is_err());
        let mut next = authority.bind(&device, key(1), q, &bs).unwrap();
        assert!(next.alloc(&device, 64, DType::U8, vec![64]).is_err());
        authority.quarantine_generation(7).unwrap();
        assert!(next.alloc(&device, 64, DType::U8, vec![64]).is_err());
        assert!(next.snapshot().unwrap().quarantined);
        println!("LEGACY_GROWTH_AABABA_COMPLETE roots=2 historical_bytes=192 attempts=2 gpu_dispatch=false");
    });
}
