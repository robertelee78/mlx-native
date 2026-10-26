//! Each ignored control requires its OWN fresh process; no GPU or model work.
use super::*;
use crate::encoder::RecordedBinding;
fn install() -> RootAuthorityHandle {
    NativeRootAuthority::install_early_with_live(
        RootCreationLimits {
            max_attempted_requested_bytes: 65536,
            max_root_attempts: 8,
            max_raw_constructor_attempts: 8,
            max_registered_devices: 1,
            max_rope_cache_entries: 1,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: 4096,
            max_live_and_pending_roots: 1,
        },
    )
    .unwrap()
}
#[test]
#[ignore = "fresh-process 4096B CPU Shared live-cap control; root supervised only"]
fn live_root_shared_actual_raw_recorded_control() {
    println!("LIVE_ROOT_SHARED_BEGIN requested_bytes=4096 gpu_work=false");
    let authority = install();
    let device = MlxDevice::new()
        .unwrap()
        .bounded_pool_test_without_residency();
    authority.register_device(&device).unwrap();
    objc::rc::autoreleasepool(|| {
        let buffer = device.alloc_buffer(4096, DType::U8, vec![4096]).unwrap();
        let raw = buffer.metal_buffer().clone();
        let binding = RecordedBinding::Buffer {
            metal_buffer: raw.clone(),
            offset: 0,
        };
        let snapshot = authority.live_snapshot().unwrap();
        assert_eq!(snapshot.live_requested_bytes, 4096);
        let history = authority.snapshot().unwrap();
        assert!(device.alloc_buffer(1, DType::U8, vec![1]).is_err());
        assert_eq!(
            authority.snapshot().unwrap().raw_attempt_reservations,
            history.raw_attempt_reservations
        );
        drop(buffer);
        assert_eq!(
            authority.live_snapshot().unwrap().live_requested_bytes,
            4096
        );
        drop(raw);
        assert_eq!(
            authority.live_snapshot().unwrap().live_requested_bytes,
            4096
        );
        drop(binding);
    });
    let s = authority.live_snapshot().unwrap();
    assert_eq!(s.pending_requested_bytes, 0);
    assert_eq!(s.live_requested_bytes, 0);
    assert_eq!(s.final_releases, 1);
    println!("LIVE_ROOT_SHARED_COMPLETED final_releases=1 live_bytes=0 gpu_work=false");
}
#[test]
#[ignore = "fresh-process <=64KiB readonly NoCopy live control; root supervised only"]
fn live_root_nocopy_actual_extent_control() {
    use memmap2::MmapOptions;
    use std::fs::OpenOptions;
    extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    let raw_page = unsafe { getpagesize() };
    assert!(raw_page > 0);
    let page = raw_page as usize;
    assert!((4160..=65536).contains(&page));
    let authority = NativeRootAuthority::install_early_with_live(
        RootCreationLimits {
            max_attempted_requested_bytes: 65536,
            max_root_attempts: 4,
            max_raw_constructor_attempts: 4,
            max_registered_devices: 1,
            max_rope_cache_entries: 1,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: page,
            max_live_and_pending_roots: 1,
        },
    )
    .unwrap();
    println!("LIVE_ROOT_NOCOPY_BEGIN requested_bytes={page} gpu_work=false");
    let path = std::env::temp_dir().join(format!("mlx-live-root-{}.bin", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path);
    file.set_len(page as u64).unwrap();
    // Escrow outlives every Metal owner, including unwind; readonly owned file.
    let escrow = Arc::new(unsafe { MmapOptions::new().len(page).map(&file) }.unwrap());
    let device = MlxDevice::new()
        .unwrap()
        .bounded_pool_test_without_residency();
    authority.register_device(&device).unwrap();
    objc::rc::autoreleasepool(|| {
        let buffer = device
            .map_file_buffer(escrow.clone(), 64, 4096, DType::U8, vec![4096])
            .unwrap();
        let raw = buffer.metal_buffer().clone();
        drop(buffer);
        assert_eq!(
            authority.live_snapshot().unwrap().live_requested_bytes,
            page
        );
        assert!(Arc::strong_count(&escrow) > 1);
        drop(raw);
    });
    let s = authority.live_snapshot().unwrap();
    assert_eq!(s.live_requested_bytes, 0);
    assert_eq!(s.final_releases, 1);
    assert_eq!(Arc::strong_count(&escrow), 1);
    println!("LIVE_ROOT_NOCOPY_COMPLETED final_releases=1 live_bytes=0 cpu_escrow_still_owned=true gpu_work=false");
}
