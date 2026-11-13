//! Escrow-backed lifetime witnesses over at most two host pages of file-backed
//! Metal storage. Raw-owner tests do not submit work; commit/reclaim tests use
//! the public submission path and a bounded GPU blit on pool-less Rust threads.
//! Escrow keeps no-copy addresses valid even during assertion unwinding.

use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use super::{host_page_size, MlxDevice};
use crate::dtypes::DType;
use crate::encoder::RecordedBinding;

const LOGICAL_BYTES: usize = 4096;
const OFFSET: usize = 64;
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    file: File,
    path: PathBuf,
    physical_bytes: usize,
}

impl Fixture {
    fn new() -> Self {
        let page = host_page_size().expect("query actual host page size");
        let physical_bytes = (OFFSET + LOGICAL_BYTES).div_ceil(page) * page;
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlx-mapped-backing-{}-{sequence}.bin",
            std::process::id()
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create private regular mapping fixture");
        file.set_len(physical_bytes as u64)
            .expect("size fixture before mapping");
        Self {
            file,
            path,
            physical_bytes,
        }
    }

    fn mapping(&self) -> Arc<Mmap> {
        // SAFETY: this private, uniquely created regular file is sized before
        // mapping and never modified/truncated by these tests. Fixture remains
        // alive until every mapping/resource owner and the escrow have dropped.
        Arc::new(
            unsafe { MmapOptions::new().len(self.physical_bytes).map(&self.file) }
                .expect("map bounded read-only fixture"),
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Clone, Copy, Debug)]
enum ResidencyMode {
    Default,
    Disabled,
}

fn device(mode: ResidencyMode) -> MlxDevice {
    match mode {
        ResidencyMode::Default => MlxDevice::new().expect("Metal device for bounded lifetime test"),
        // Explicit per-device configuration: do not mutate process-global env
        // flags (which are cached and could race other native tests).
        ResidencyMode::Disabled => {
            let device = metal::Device::system_default().expect("Metal device");
            let queue = device.new_command_queue();
            MlxDevice {
                device,
                queue,
                residency_set: None,
            }
        }
    }
}

/// Declared before every resource owner, so unwind drops those owners before
/// flushing their staged residency removals. No command encoder is created.
struct FlushResidencyOnDrop<'a>(&'a MlxDevice);

impl Drop for FlushResidencyOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(set) = self.0.residency_set() {
            set.flush_pending();
        }
    }
}

fn mapped(device: &MlxDevice, escrow: &Arc<Mmap>) -> crate::MlxBuffer {
    device
        .map_file_buffer(
            Arc::clone(escrow),
            OFFSET,
            LOGICAL_BYTES,
            DType::U8,
            vec![LOGICAL_BYTES],
        )
        .expect("bounded no-copy mapping")
}

fn raw_owner_witness(mode: ResidencyMode, recorded: bool) {
    let fixture = Fixture::new();
    // Escrow is outside every autorelease scope/device/Metal owner. It remains
    // strong even if the required >1 predicate fails on the unfixed source.
    let escrow = fixture.mapping();
    let weak = Arc::downgrade(&escrow);
    assert_eq!(Arc::strong_count(&escrow), 1);
    objc::rc::autoreleasepool(|| {
        let device = device(mode);
        let _flush_on_unwind = FlushResidencyOnDrop(&device);
        eprintln!(
            "mapped backing witness mode={mode:?} actual_residency={} recorded={recorded}",
            device.residency_sets_enabled()
        );
        if recorded {
            let retained = objc::rc::autoreleasepool(|| {
                let buffer = mapped(&device, &escrow);
                let view = buffer.slice_view(32, 64);
                let binding = RecordedBinding::Buffer {
                    metal_buffer: view.metal_buffer().clone(),
                    offset: view.byte_offset(),
                };
                let retained = binding.clone();
                drop(binding);
                drop(view);
                drop(buffer);
                retained
            });
            if let Some(set) = device.residency_set() {
                set.flush_pending();
            }
            let count = Arc::strong_count(&escrow);
            eprintln!("mapped backing recorded-only strong_count={count}");
            assert!(
                count > 1,
                "RecordedBinding must retain mapping independently of escrow; observed {count}"
            );
            drop(retained);
        } else {
            let retained = objc::rc::autoreleasepool(|| {
                let buffer = mapped(&device, &escrow);
                let cloned = buffer.clone();
                let view = cloned.slice_view(32, 64);
                let retained = view.metal_buffer().clone();
                drop(view);
                drop(cloned);
                drop(buffer);
                retained
            });
            if let Some(set) = device.residency_set() {
                set.flush_pending();
            }
            let count = Arc::strong_count(&escrow);
            eprintln!("mapped backing raw-only strong_count={count}");
            assert!(
                count > 1,
                "raw Metal owner must retain mapping independently of escrow; observed {count}"
            );
            drop(retained);
        }
        // The guard flushes removal and device/queue ownership ends before
        // this autorelease pool drains; escrow remains outside the whole scope.
    });
    assert_eq!(
        Arc::strong_count(&escrow),
        1,
        "all resource/registration/autorelease owners must release backing, not leak it"
    );
    drop(escrow);
    assert!(
        weak.upgrade().is_none(),
        "final mapping teardown must occur"
    );
}

#[test]
fn raw_metal_owner_retains_mapping_disabled_residency() {
    raw_owner_witness(ResidencyMode::Disabled, false);
}

#[test]
fn raw_metal_owner_retains_mapping_default_residency() {
    raw_owner_witness(ResidencyMode::Default, false);
}

#[test]
fn recorded_binding_retains_mapping_disabled_residency() {
    raw_owner_witness(ResidencyMode::Disabled, true);
}

#[test]
fn recorded_binding_retains_mapping_default_residency() {
    raw_owner_witness(ResidencyMode::Default, true);
}

#[test]
fn ordinary_clones_and_views_retain_mapping_and_release_it() {
    for mode in [ResidencyMode::Disabled, ResidencyMode::Default] {
        let fixture = Fixture::new();
        let escrow = fixture.mapping();
        let weak = Arc::downgrade(&escrow);
        objc::rc::autoreleasepool(|| {
            let device = device(mode);
            let _flush_on_unwind = FlushResidencyOnDrop(&device);
            let buffer = mapped(&device, &escrow);
            assert_eq!(buffer.data_byte_len(), LOGICAL_BYTES);
            assert_eq!(buffer.byte_len(), fixture.physical_bytes);
            assert_eq!(buffer.byte_offset(), OFFSET as u64);
            assert!(!buffer.is_cpu_writable());
            let cloned = buffer.clone();
            let view = cloned.slice_view(32, 64);
            drop(buffer);
            assert!(Arc::strong_count(&escrow) > 1);
            drop(cloned);
            assert!(Arc::strong_count(&escrow) > 1);
            assert_eq!(view.data_byte_len(), 64);
            assert_eq!(view.byte_len(), fixture.physical_bytes);
            assert_eq!(view.byte_offset(), (OFFSET + 32) as u64);
            drop(view);
        });
        assert_eq!(Arc::strong_count(&escrow), 1);
        drop(escrow);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn rejected_ranges_do_not_retain_mapping() {
    for mode in [ResidencyMode::Disabled, ResidencyMode::Default] {
        let fixture = Fixture::new();
        let escrow = fixture.mapping();
        let weak = Arc::downgrade(&escrow);
        objc::rc::autoreleasepool(|| {
            let device = device(mode);
            for (offset, length) in [(0, 0), (fixture.physical_bytes, 1), (usize::MAX, 2)] {
                let result = device.map_file_buffer(
                    Arc::clone(&escrow),
                    offset,
                    length,
                    DType::U8,
                    vec![length],
                );
                assert!(result.is_err());
                drop(result);
                assert_eq!(Arc::strong_count(&escrow), 1);
            }
        });
        drop(escrow);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn nil_constructor_disposes_the_same_backing_capture_without_callback_dependency() {
    let fixture = Fixture::new();
    let escrow = fixture.mapping();
    let weak = Arc::downgrade(&escrow);
    let mut calls = 0;
    objc::rc::autoreleasepool(|| {
        let result = super::mapped_resource_with(
            Arc::clone(&escrow),
            fixture.physical_bytes,
            |_deallocator| {
                calls += 1;
                assert!(
                    Arc::strong_count(&escrow) > 1,
                    "actual production block capture must exist across constructor"
                );
                // Nil creation may never invoke the callback. The very same
                // helper must dispose its local copied block/capture anyway.
                std::ptr::null_mut()
            },
        );
        assert!(matches!(
            result,
            Err(crate::MlxError::BufferAllocationError { .. })
        ));
        drop(result);
    });
    assert_eq!(calls, 1);
    assert_eq!(Arc::strong_count(&escrow), 1);
    drop(escrow);
    assert!(weak.upgrade().is_none());
}

#[test]
fn constructor_unwind_disposes_capture_without_metal_or_payload_access() {
    let fixture = Fixture::new();
    let escrow = fixture.mapping();
    let weak = Arc::downgrade(&escrow);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::mapped_resource_with(Arc::clone(&escrow), fixture.physical_bytes, |_| {
            assert!(Arc::strong_count(&escrow) > 1);
            panic!("bounded injected constructor unwind before resource creation");
        })
    }));
    assert!(result.is_err());
    assert_eq!(Arc::strong_count(&escrow), 1);
    drop(escrow);
    assert!(weak.upgrade().is_none());
}

fn commit_drop_reclaim_witness(mode: ResidencyMode, gpu_read: bool) {
    // Deliberately no outer autoreleasepool: a consumer's Rust loading/worker
    // thread is not required to install one around native API calls.
    std::thread::spawn(move || {
        let fixture = Fixture::new();
        let escrow = fixture.mapping();
        let weak = Arc::downgrade(&escrow);
        let loading_device = device(mode);
        let set = loading_device.residency_set().cloned();
        let buffer = mapped(&loading_device, &escrow);
        let binding = RecordedBinding::Buffer {
            metal_buffer: buffer.metal_buffer().clone(),
            offset: buffer.byte_offset(),
        };

        // This device has never encoded a resource. Exactly one empty public
        // commit publishes the mapped allocation's pending residency membership.
        let mut loading_encoder = loading_device.command_encoder().unwrap();
        loading_encoder.commit_and_wait().unwrap();
        drop(loading_encoder);
        drop(loading_device);

        if gpu_read {
            let consumer = device(mode);
            let output = consumer
                .alloc_buffer(LOGICAL_BYTES, DType::U8, vec![LOGICAL_BYTES])
                .unwrap();
            let mut encoder = consumer.command_encoder().unwrap();
            encoder
                .blit_copy_bytes(&buffer, 0, &output, 0, LOGICAL_BYTES as u64)
                .unwrap();
            // Metal's retained command owns the mapping independently of its
            // logical wrapper; RecordedBinding is another independent owner.
            drop(buffer);
            if let Some(set) = &set {
                set.flush_pending();
            }
            assert!(Arc::strong_count(&escrow) > 1);
            encoder.commit_and_wait().unwrap();
            assert!(
                output.as_logical_slice::<u8>().unwrap().iter().all(|&b| b == 0),
                "the private, zero-initialized file must survive the GPU read"
            );
            drop(encoder);
            drop(output);
            drop(consumer);
        } else {
            drop(buffer);
        }

        // Keep the set alive for a deterministic removal boundary even if its
        // heartbeat is concurrently borrowing it. This retires committed
        // membership, not the independent raw Metal owner below.
        if let Some(set) = &set {
            set.flush_pending();
        }
        assert!(
            Arc::strong_count(&escrow) > 1,
            "recorded Metal owner must still retain its no-copy backing"
        );
        drop(binding);
        assert_eq!(
            Arc::strong_count(&escrow),
            1,
            "completed commits and retired residency must not keep mapped backing"
        );
        drop(set);
        drop(escrow);
        assert!(weak.upgrade().is_none());
    })
    .join()
    .expect("pool-less consumer commit/drop/reclaim witness");
}

#[test]
fn empty_loading_commit_then_drop_reclaims_mapping_without_ambient_pool() {
    for mode in [ResidencyMode::Disabled, ResidencyMode::Default] {
        commit_drop_reclaim_witness(mode, false);
    }
}

#[test]
fn loading_commit_then_gpu_read_and_drop_reclaims_mapping_without_ambient_pool() {
    assert!(
        !crate::command_buffer_uses_unretained_references(),
        "this lifetime witness requires Metal's default retained command owners"
    );
    for mode in [ResidencyMode::Disabled, ResidencyMode::Default] {
        commit_drop_reclaim_witness(mode, true);
    }
}
