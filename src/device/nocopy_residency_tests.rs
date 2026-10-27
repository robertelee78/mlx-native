//! Finite owned-file NoCopy observations, not model residency/admission proof.
//! Run one ignored test under an external process-group watchdog; never in parallel.
use super::{host_page_size, MlxDevice};
use crate::DType;
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde_json::json;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const PAGE: usize = 16384;
const FILE_BYTES: usize = 64 * 1024 * 1024;
const PAGES: usize = FILE_BYTES / PAGE;
const SENTINELS: [usize; 4] = [0, 1024, 2048, PAGES - 1];
const WIRED: i32 = 0x1000;
// Installed SDK mach/mach_vm.h: mach_vm_size_t dispositions_count is u64,
// NOT mach_msg_type_number_t/u32. vm_page_disposition_t is signed int32.
extern "C" {
    static mach_task_self_: u32;
    fn mach_vm_page_range_query(
        task: u32,
        address: u64,
        size: u64,
        dispositions: u64,
        dispositions_count: *mut u64,
    ) -> i32;
    fn mlock(address: *const std::ffi::c_void, size: usize) -> i32;
    fn munlock(address: *const std::ffi::c_void, size: usize) -> i32;
}

fn query(ptr: *const u8, bytes: usize) -> Vec<i32> {
    assert_eq!(ptr as usize % PAGE, 0);
    assert_eq!(bytes % PAGE, 0);
    let mut bits = vec![i32::MIN; bytes / PAGE];
    let mut count = bits.len() as u64;
    // SAFETY: current task, live owned mappings and exact bounded output array.
    // This query inspects metadata; it never dereferences/touches mapped bytes.
    let kr = unsafe {
        mach_vm_page_range_query(
            mach_task_self_,
            ptr as u64,
            bytes as u64,
            bits.as_mut_ptr() as u64,
            &mut count,
        )
    };
    assert_eq!(kr, 0, "page query unavailable: no capability fallback");
    assert_eq!(count, bits.len() as u64, "short disposition response");
    assert!(bits.iter().all(|b| *b >= 0), "invalid disposition output");
    bits
}

struct Lock {
    ptr: *const u8,
    locked: bool,
}
impl Lock {
    fn release(&mut self) {
        if self.locked {
            // SAFETY: witness mapping outlives this guard on all unwind paths.
            assert_eq!(
                unsafe { munlock(self.ptr.cast(), PAGE) },
                0,
                "munlock failed"
            );
            self.locked = false;
        }
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        if self.locked {
            // Never panic twice while unwinding; cleanup failure is explicit.
            let rc = unsafe { munlock(self.ptr.cast(), PAGE) };
            eprintln!("NOCOPY_WITNESS_UNWIND_UNLOCK rc={rc}");
        }
    }
}

fn cpu_positive() {
    let mut map = MmapMut::map_anon(PAGE).expect("one anonymous witness page");
    map[0] = 7;
    let ptr = map.as_ptr();
    let before = query(ptr, PAGE);
    assert_eq!(before[0] & WIRED, 0, "baseline witness already wired");
    // SAFETY: this owned anonymous page remains mapped through Lock drop.
    let rc = unsafe { mlock(ptr.cast(), PAGE) };
    assert_eq!(rc, 0, "mlock capability unavailable; no skip/fallback");
    let mut lock = Lock { ptr, locked: true };
    let locked = query(ptr, PAGE);
    lock.release();
    let after = query(ptr, PAGE);
    assert_ne!(locked[0] & WIRED, 0, "missing WIRED positive capability");
    assert_eq!(after[0] & WIRED, 0, "unlock did not restore baseline bit");
    println!(
        "NOCOPY_WITNESS_CPU {}",
        json!({"before":before,"locked":locked,"after":after,"requested_lock_bytes":PAGE,"positive":true})
    );
}

struct Fixture {
    dir: PathBuf,
    file: Option<File>,
    dir_id: (u64, u64),
    file_id: Option<(u64, u64)>,
    cleaned: bool,
}
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("mlx-nocopy-witness-{}", std::process::id()));
        DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("exclusive owned fixture directory");
        let m = fs::symlink_metadata(&dir).unwrap();
        let mut this = Self {
            dir,
            file: None,
            dir_id: (m.dev(), m.ino()),
            file_id: None,
            cleaned: false,
        };
        let p = this.dir.join("owned.bin");
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&p)
            .unwrap();
        let m = f.metadata().unwrap();
        this.file_id = Some((m.dev(), m.ino()));
        // One page of bounded stack scratch, NOT a 64MiB host duplicate.
        let mut chunk = [0u8; PAGE];
        for page in 0..PAGES {
            for (j, b) in chunk.iter_mut().enumerate() {
                *b = ((page + j) % 251) as u8;
            }
            f.write_all(&chunk).expect("stream initialize private file");
        }
        drop(f); // no flush/fsync/global sync; no writes after immutable mapping.
        let f = OpenOptions::new().read(true).open(&p).unwrap();
        let m = f.metadata().unwrap();
        assert_eq!(Some((m.dev(), m.ino())), this.file_id);
        assert_eq!(m.len(), FILE_BYTES as u64);
        this.file = Some(f);
        this
    }
    fn mapping(&self) -> Arc<Mmap> {
        // SAFETY: private file is fully initialized, sized and never modified.
        Arc::new(
            unsafe {
                MmapOptions::new()
                    .len(FILE_BYTES)
                    .map(self.file.as_ref().unwrap())
            }
            .unwrap(),
        )
    }
    fn cleanup(&mut self) -> bool {
        if self.cleaned {
            return true;
        }
        let Ok(d) = fs::symlink_metadata(&self.dir) else {
            return false;
        };
        if !d.is_dir() || d.file_type().is_symlink() || (d.dev(), d.ino()) != self.dir_id {
            return false;
        }
        let p = self.dir.join("owned.bin");
        let Ok(m) = fs::symlink_metadata(&p) else {
            return false;
        };
        if !m.is_file() || m.file_type().is_symlink() || Some((m.dev(), m.ino())) != self.file_id {
            return false;
        }
        let Ok(items) = fs::read_dir(&self.dir) else {
            return false;
        };
        if items.count() != 1 {
            return false;
        }
        self.file.take();
        if fs::remove_file(p).is_err() || fs::remove_dir(&self.dir).is_err() {
            return false;
        }
        self.cleaned = true;
        true
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if !self.cleaned {
            eprintln!("NOCOPY_WITNESS_FIXTURE_CLEANUP ok={}", self.cleanup());
        }
    }
}

fn observe(phase: &str, a: &Mmap, b: &Mmap, device: Option<&MlxDevice>) {
    let start = Instant::now();
    let x = query(a.as_ptr(), FILE_BYTES);
    let y = query(b.as_ptr(), FILE_BYTES);
    let wired_a = x.iter().filter(|n| **n & WIRED != 0).count();
    let wired_b = y.iter().filter(|n| **n & WIRED != 0).count();
    let current = device.map(|d| d.metal_device().current_allocated_size());
    println!(
        "NOCOPY_WITNESS_PHASE {}",
        json!({"phase":phase,"pages":PAGES,"wired_a":wired_a,"wired_b":wired_b,"present_a":x.iter().filter(|n|**n&1!=0).count(),"present_b":y.iter().filter(|n|**n&1!=0).count(),"alias_dispositions_equal":x==y,"a":x,"b":y,"query_elapsed_ns":start.elapsed().as_nanos(),"metal_current_allocated_size_raw":current,"physical_admission":false})
    );
}

#[test]
#[ignore = "one separately supervised64MiB owned-file Metal residency observation, no model"]
fn nocopy_default_residency_owned_file_observation() {
    assert_eq!(host_page_size().unwrap(), PAGE, "fixed16KiB SDK/packet");
    assert!(
        !std::env::vars().any(|(k, _)| k.starts_with("HF2Q_")
            || k.starts_with("MLX_")
            || k.starts_with("MTL_")
            || k.starts_with("METAL_")),
        "ambient runtime override"
    );
    cpu_positive();
    let mut fixture = Fixture::new();
    let a = fixture.mapping();
    let b = fixture.mapping();
    assert_ne!(a.as_ptr(), b.as_ptr());
    println!(
        "NOCOPY_WITNESS_PACKET {}",
        json!({"file_bytes":FILE_BYTES,"file_alias_virtual_bytes":FILE_BYTES*2,"anonymous_witness_bytes":PAGE,"page_query_output_bytes_max":PAGES*4*2,"GPU_output_bytes":64,"model_loaded":false,"file":fixture.dir.join("owned.bin")})
    );
    observe("plain_alias_control_pre_device", &a, &b, None);
    let device = MlxDevice::new().expect("actual Metal device required, never skip");
    assert!(
        device.residency_sets_enabled(),
        "default residency capability required"
    );
    let set = device.residency_set().unwrap();
    observe("precreate_device_initialized", &a, &b, Some(&device));
    let mapped = device
        .map_file_buffer(
            Arc::clone(&a),
            0,
            FILE_BYTES,
            DType::F32,
            vec![FILE_BYTES / 4],
        )
        .expect("actual native NoCopy mapping");
    assert!(mapped.is_file_backed());
    assert!(!mapped.is_cpu_writable());
    observe("after_create_before_explicit_flush", &a, &b, Some(&device));
    set.flush_pending();
    // Actual queue-registered default set, not a fake requestResidency ack.
    // Its private heartbeat may request residency concurrently; timing is not forced.
    observe("after_default_residency_flush", &a, &b, Some(&device));
    let output = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    let mut encoder = device.command_encoder().unwrap();
    for (i, page) in SENTINELS.iter().enumerate() {
        encoder
            .blit_copy_bytes(&mapped, (page * PAGE) as u64, &output, (i * 16) as u64, 16)
            .expect("encode existing native GPU blit");
    }
    encoder
        .commit_and_wait()
        .expect("checked tiny GPU completion");
    let actual = output.as_logical_slice::<u8>().unwrap();
    for (i, page) in SENTINELS.iter().enumerate() {
        for j in 0..16 {
            assert_eq!(
                actual[i * 16 + j],
                ((page + j) % 251) as u8,
                "sentinel bytes"
            );
        }
    }
    observe(
        "after_four_sentinel_GPU_read_checked_drain",
        &a,
        &b,
        Some(&device),
    );
    drop(encoder);
    drop(output);
    drop(mapped);
    set.flush_pending();
    observe(
        "after_explicit_resource_owners_drop_flush_aliases_live",
        &a,
        &b,
        Some(&device),
    );
    // CPU aliases remain alive through all Metal owners and observations.
    drop(b);
    drop(a);
    drop(device);
    assert!(fixture.cleanup(), "owned identity-checked cleanup failed");
    println!("NOCOPY_WITNESS_COMPLETE phases=6 GPU_sentinels=4 checked_drain=true cleanup=true model_loaded=false physical_admission=false model_scale_qualified=false");
}
