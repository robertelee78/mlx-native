//! Ignored, separately supervised actual CPU-only controls. Fresh process per test.
use super::*;
use crate::encoder::RecordedBinding;
use crate::{
    GrowthBucket, GrowthPoolKey, GrowthQuota, LegacyGrowthLimits, LegacyPoolGrowthAuthority,
    MlxBufferPool,
};
use metal::foreign_types::ForeignType;

fn install(bytes: usize, roots: usize, cache: usize) -> RootAuthorityHandle {
    NativeRootAuthority::install_early(RootCreationLimits {
        max_attempted_requested_bytes: bytes,
        max_root_attempts: roots,
        max_raw_constructor_attempts: roots,
        max_registered_devices: 1,
        max_rope_cache_entries: cache,
    })
    .expect("fresh process authority")
}
fn history(a: &RootAuthorityHandle, bytes: usize, roots: usize) -> RootCreationSnapshot {
    let s = a.snapshot().unwrap();
    assert!(s.covered && s.unknown.is_none());
    assert_eq!(
        (
            s.attempted_requested_bytes,
            s.root_attempts,
            s.raw_attempt_reservations,
            s.successful_roots,
            s.inflight
        ),
        (bytes, roots, roots, roots, 0)
    );
    assert_eq!(s.registered_devices, 1);
    s
}
fn known(b: &MlxBuffer, bytes: usize) -> usize {
    match b.root_origin_snapshot() {
        RootOriginSnapshot::KnownNative {
            id,
            requested_bytes,
        } => {
            assert_eq!(requested_bytes, bytes);
            assert_ne!(id, 0);
            id
        }
        other => panic!("expected known root, got {other:?}"),
    }
}
#[test]
#[ignore = "separate supervised CPU-only 6 roots/416 requested bytes"]
fn actual_shared_factory_composition() {
    let a = install(416, 6, 0);
    let d = MlxDevice::new().unwrap();
    a.register_device(&d).unwrap();
    a.register_device(&d.clone()).unwrap();
    let source = [0x5au8; 96];
    let first = d.alloc_buffer(64, DType::U8, vec![64]).unwrap();
    assert!(first.as_slice::<u8>().unwrap().iter().all(|x| *x == 0));
    let mut ids = [0usize; 6];
    ids[0] = known(&first, 64);
    history(&a, 64, 1);
    // SAFETY: CPU initializes every logical byte before any read; no GPU access.
    let mut second = unsafe { d.alloc_buffer_for_overwrite(64, DType::U8, vec![64]) }.unwrap();
    second.as_mut_slice::<u8>().unwrap().fill(0x5a);
    assert_eq!(second.as_slice::<u8>().unwrap(), &source[..64]);
    ids[1] = known(&second, 64);
    history(&a, 128, 2);
    let third =
        crate::weight::safetensors_to_metal_buffer(&d, &source[..64], DType::U8, vec![64]).unwrap();
    assert_eq!(third.as_slice::<u8>().unwrap(), &source[..64]);
    ids[2] = known(&third, 64);
    history(&a, 192, 3);
    let mut pool = MlxBufferPool::new();
    let fourth = pool.alloc(&d, 64, DType::U8, vec![64]).unwrap();
    assert!(fourth.as_slice::<u8>().unwrap().iter().all(|x| *x == 0));
    ids[3] = known(&fourth, 64);
    history(&a, 256, 4);
    let quota = GrowthQuota {
        bytes: 64,
        roots: 1,
        attempts: 1,
    };
    let growth = LegacyPoolGrowthAuthority::install(LegacyGrowthLimits {
        quota,
        max_pools: 1,
        max_buckets_per_pool: 1,
        max_rank: 1,
        max_batch: 1,
    })
    .unwrap();
    let key = GrowthPoolKey {
        kind: 57,
        worker_generation: 1,
    };
    let mut limited = growth
        .bind(&d, key, quota, &[GrowthBucket { bytes: 64, quota }])
        .unwrap();
    let fifth = limited.alloc(&d, 64, DType::U8, vec![64]).unwrap();
    assert!(fifth.as_slice::<u8>().unwrap().iter().all(|x| *x == 0));
    ids[4] = known(&fifth, 64);
    history(&a, 320, 5);
    let local = limited.snapshot().unwrap();
    assert_eq!(
        (local.pool.bytes, local.pool.roots, local.pool.attempts),
        (64, 1, 1)
    );
    let sixth = d.upload_shared_bytes(&source, DType::U8, vec![96]).unwrap();
    assert_eq!(sixth.as_slice::<u8>().unwrap(), &source);
    ids[5] = known(&sixth, 96);
    history(&a, 416, 6);
    for i in 0..ids.len() {
        for j in 0..i {
            assert_ne!(ids[i], ids[j]);
        }
    }
    let view = first.slice_view(0, 16);
    assert_eq!(known(&view, 64), ids[0]);
    drop(view);
    drop(first);
    drop(second);
    drop(third);
    drop(sixth);
    pool.release(fourth);
    drop(pool);
    drop(fifth);
    // SAFETY: all issued values dropped, no encoded/recorded/CPU future use.
    unsafe { limited.reset(&d) }.unwrap();
    limited.clear(&d).unwrap();
    let before = history(&a, 416, 6);
    assert_eq!(before.cache_slot_history, 0);
    assert!(d.alloc_buffer(64, DType::U8, vec![64]).is_err());
    let after = history(&a, 416, 6);
    assert_eq!(after.denied, before.denied + 1);
    println!("NATIVE57_ACTUAL_SHARED_COMPLETE roots=6 bytes=416 gpu=false");
}
#[test]
#[ignore = "separate supervised CPU-only one-page mapped lifetime control"]
fn actual_mapped_root_lineage_and_backing() {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
    extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    // SAFETY: argument-free OS page-size query; no allocation or payload access.
    let page = unsafe { getpagesize() };
    assert_eq!(page, 16384);
    let page = page as usize;
    let a = install(page, 1, 0);
    let dir = std::env::temp_dir().join(format!("native57-mapped-{}", std::process::id()));
    std::fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    let directory = Directory(dir.clone());
    let path = dir.join("fixture");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .unwrap();
    // Install cleanup only AFTER exclusive creation succeeds: collision is unowned.
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let cleanup = Cleanup(path.clone());
    let block = [0x39u8; 256];
    for _ in 0..page / 256 {
        file.write_all(&block).unwrap();
    }
    file.sync_all().unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    let before = file.metadata().unwrap();
    assert_eq!(before.len(), page as u64);
    drop(file);
    let file = std::fs::File::open(&path).unwrap();
    let after = file.metadata().unwrap();
    assert_eq!(
        (before.dev(), before.ino(), before.len()),
        (after.dev(), after.ino(), after.len())
    );
    // SAFETY: uniquely owned private file, fixed initialized size, never mutated.
    let escrow = Arc::new(unsafe { memmap2::MmapOptions::new().len(page).map(&file) }.unwrap());
    let weak = Arc::downgrade(&escrow);
    objc::rc::autoreleasepool(|| {
        let d = MlxDevice::new().unwrap();
        a.register_device(&d).unwrap();
        let b = d
            .map_file_buffer(escrow.clone(), 4096, 4096, DType::U8, vec![4096])
            .unwrap();
        assert_eq!(
            (b.byte_offset(), b.data_byte_len(), b.byte_len()),
            (4096, 4096, page)
        );
        known(&b, page);
        assert!(b.as_slice::<u8>().unwrap().iter().all(|x| *x == 0x39));
        let raw = b.metal_buffer().clone();
        let recorded = RecordedBinding::Buffer {
            metal_buffer: raw.clone(),
            offset: b.byte_offset(),
        };
        drop(b);
        if let Some(set) = d.residency_set() {
            set.flush_pending();
        }
        assert!(Arc::strong_count(&escrow) > 1);
        drop(raw);
        assert!(Arc::strong_count(&escrow) > 1);
        drop(recorded);
        if let Some(set) = d.residency_set() {
            set.flush_pending();
        }
        history(&a, page, 1);
    });
    assert_eq!(Arc::strong_count(&escrow), 1);
    drop(escrow);
    assert!(weak.upgrade().is_none());
    drop(file);
    std::fs::remove_file(&path).unwrap();
    drop(cleanup);
    std::fs::remove_dir(&dir).unwrap();
    drop(directory);
    println!(
        "NATIVE57_ACTUAL_MAPPED_COMPLETE roots=1 bytes=16384 logical=4096 offset=4096 gpu=false"
    );
}
#[test]
#[ignore = "separate supervised CPU-only already-charged raw import control"]
fn actual_raw_clone_import_is_unknown() {
    let a = install(128, 2, 0);
    let d = MlxDevice::new().unwrap();
    a.register_device(&d).unwrap();
    let source = [0x29u8; 64];
    let b = d.upload_shared_bytes(&source, DType::U8, vec![64]).unwrap();
    assert_eq!(b.as_slice::<u8>().unwrap(), &source);
    known(&b, 64);
    history(&a, 64, 1);
    let raw = b.metal_buffer().clone();
    let identity = raw.as_ptr();
    let imported = MlxBuffer::from_raw(raw, DType::U8, vec![64]);
    assert_eq!(imported.metal_buffer().as_ptr(), identity);
    assert_eq!(
        imported.root_origin_snapshot(),
        RootOriginSnapshot::UnknownImport
    );
    let s = a.snapshot().unwrap();
    assert!(!s.covered && s.unknown.is_some());
    assert_eq!(
        (
            s.attempted_requested_bytes,
            s.root_attempts,
            s.raw_attempt_reservations,
            s.successful_roots,
            s.inflight
        ),
        (64, 1, 1, 1, 0)
    );
    assert!(d.alloc_buffer(64, DType::U8, vec![64]).is_err());
    let t = a.snapshot().unwrap();
    assert!(!t.covered);
    assert_eq!(t.unknown, s.unknown);
    assert_eq!(
        (t.raw_attempt_reservations, t.inflight),
        (s.raw_attempt_reservations, 0)
    );
    assert_eq!(
        (
            t.attempted_requested_bytes,
            t.root_attempts,
            t.successful_roots
        ),
        (64, 1, 1)
    );
    drop(imported);
    drop(b);
    println!("NATIVE57_ACTUAL_IMPORT_COMPLETE roots=1 bytes=64 covered=false gpu=false");
}
#[test]
#[ignore = "separate supervised zero-payload strict heap refusal control"]
fn actual_strict_heap_refuses_before_creation() {
    let a = install(0, 0, 0);
    let d = MlxDevice::new().unwrap();
    a.register_device(&d).unwrap();
    let authority = crate::ProcessPoolAuthority::install(crate::ProcessPoolLimits {
        max_domains: 1,
        max_reserved_heap_bytes: 65536,
        max_child_creations: 1,
        max_constructor_attempts: 1,
        max_rank: 1,
        max_batch: 1,
    })
    .unwrap();
    let before = history(&a, 0, 0);
    let result = authority.create_pool(
        &d,
        crate::BoundedPoolConfig {
            requested_heap_bytes: 65536,
            reserved_heap_bytes: 65536,
            max_child_creations: 1,
        },
    );
    assert!(matches!(result, Err(crate::BoundedPoolError::Unsupported)));
    let after = history(&a, 0, 0);
    assert_eq!(after.denied, before.denied + 1);
    assert_eq!(authority.snapshot().unwrap().domains, 0);
    println!("NATIVE57_ACTUAL_HEAP_COMPLETE roots=0 bytes=0 unsupported=true gpu=false");
}
#[test]
#[ignore = "separate supervised CPU-only 32-byte RoPE cold/hit/cache-limit control"]
fn actual_rope_cache_combined_reservation() {
    let a = install(64, 2, 1);
    let d = MlxDevice::new().unwrap();
    a.register_device(&d).unwrap();
    let inspect = |b: &MlxBuffer| {
        let certificate = known(b, 32);
        let values = b.as_slice::<f32>().unwrap();
        assert_eq!(values.len(), 8);
        for (pair, value) in values.iter().enumerate() {
            // Independent fixed identity: 10000^(-2p/16) = 10^(-p/2).
            let expected = 10.0f64.powf(-(pair as f64) / 2.0);
            assert!(value.is_finite());
            assert!((f64::from(*value) - expected).abs() <= 1e-6 * expected.max(1e-12));
        }
        (b.metal_buffer().as_ptr() as usize, certificate)
    };
    let identity =
        crate::ops::native57_control_with_inv_freqs(d.metal_device(), 10000.0, 16, 8, inspect)
            .unwrap();
    let cold = history(&a, 32, 1);
    assert_eq!(
        (cold.cache_slot_history, cold.published_cache_entries),
        (1, 1)
    );
    let hit =
        crate::ops::native57_control_with_inv_freqs(d.metal_device(), 10000.0, 16, 8, inspect)
            .unwrap();
    assert_eq!(identity, hit);
    let hit_state = history(&a, 32, 1);
    assert_eq!(
        (
            hit_state.cache_slot_history,
            hit_state.published_cache_entries,
            hit_state.denied
        ),
        (
            cold.cache_slot_history,
            cold.published_cache_entries,
            cold.denied
        )
    );
    assert!(
        crate::ops::native57_control_with_inv_freqs(d.metal_device(), 20000.0, 16, 8, |_| ())
            .is_err()
    );
    let denied = history(&a, 32, 1);
    assert_eq!(
        (denied.cache_slot_history, denied.published_cache_entries),
        (1, 1)
    );
    assert_eq!(denied.denied, cold.denied + 1);
    let old =
        crate::ops::native57_control_with_inv_freqs(d.metal_device(), 10000.0, 16, 8, inspect)
            .unwrap();
    assert_eq!(old, identity);
    let final_hit = history(&a, 32, 1);
    assert_eq!(
        (
            final_hit.cache_slot_history,
            final_hit.published_cache_entries,
            final_hit.denied
        ),
        (
            denied.cache_slot_history,
            denied.published_cache_entries,
            denied.denied
        )
    );
    println!("NATIVE57_ACTUAL_ROPE_COMPLETE roots=1 bytes=32 cache=1 gpu=false");
}
