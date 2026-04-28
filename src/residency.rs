//! Raw Objective-C wrapper for Metal residency sets.
//!
//! `metal = 0.33` does not expose `MTLResidencySet`, so this module uses
//! `objc = 0.2` directly. The wrapper is no-op on macOS versions before 15.0.

use std::ffi::CStr;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use metal::{BufferRef as MTLBufferRef, CommandQueueRef, DeviceRef};
use objc::runtime::{Class, Object, BOOL, YES};
use objc::{msg_send, sel, sel_impl};

use crate::error::MlxError;

static RESIDENCY_DISABLED_FLAG: AtomicU8 = AtomicU8::new(0);
static TEST_ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static TEST_COMMIT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

#[repr(C)]
#[allow(non_snake_case)]
struct NSOperatingSystemVersion {
    majorVersion: isize,
    minorVersion: isize,
    patchVersion: isize,
}

struct ObjcResidencySet {
    ptr: *mut Object,
}

unsafe impl Send for ObjcResidencySet {}
unsafe impl Sync for ObjcResidencySet {}

enum ResidencySetInner {
    Active {
        object: ObjcResidencySet,
        lock: Mutex<()>,
        /// ADR-015 iter8e (Phase 3b): defer-and-flush pending flag.
        ///
        /// `add_allocation` / `remove_allocation` set this to `true` instead
        /// of calling `[set commit]` per-call. `flush_pending` issues a
        /// single `[set commit]` iff this flag was set, then clears it.
        ///
        /// This converts the per-allocation commit storm
        /// (~880 commits/token in iter8d/8e claude+codex variants) into one
        /// commit per CB-submission boundary — mirrors llama.cpp's
        /// `ggml-metal-device.m:1378-1382` (batch addAllocation in loop,
        /// commit ONCE at the end of the batch).
        pending: AtomicBool,
    },
    Noop,
}

impl Drop for ResidencySetInner {
    fn drop(&mut self) {
        let Self::Active { object, lock, .. } = self else {
            return;
        };

        if let Ok(_guard) = lock.lock() {
            unsafe {
                let _: () = msg_send![object.ptr, removeAllAllocations];
                let _: () = msg_send![object.ptr, commit];
                let _: () = msg_send![object.ptr, release];
            }
        }
    }
}

/// Owns a Metal `MTLResidencySet` NSObject pointer, or represents a no-op set
/// when the OS does not support residency sets.
#[derive(Clone)]
pub(crate) struct ResidencySet {
    inner: Arc<ResidencySetInner>,
}

impl ResidencySet {
    /// Create a new `MTLResidencySet`.
    ///
    /// On macOS versions before 15.0 this returns a true no-op wrapper without
    /// touching any residency-set selectors.
    pub(crate) fn new(device: &DeviceRef) -> Result<Self, MlxError> {
        if !macos_15_or_newer() {
            return Ok(Self {
                inner: Arc::new(ResidencySetInner::Noop),
            });
        }

        let Some(descriptor_class) = Class::get("MTLResidencySetDescriptor") else {
            return Err(MlxError::ResidencySetError(
                "MTLResidencySetDescriptor is unavailable".into(),
            ));
        };

        unsafe {
            let descriptor: *mut Object = msg_send![descriptor_class, alloc];
            let descriptor: *mut Object = msg_send![descriptor, init];
            if descriptor.is_null() {
                return Err(MlxError::ResidencySetError(
                    "failed to allocate MTLResidencySetDescriptor".into(),
                ));
            }

            let _: () = msg_send![descriptor, setInitialCapacity: 256usize];

            // Set a stable label so the residency set is identifiable in
            // Instruments / Metal-system-trace.  Literal is statically
            // NUL-terminated; NSString reads up to the NUL.
            let label_ptr = b"mlx_native_default\0".as_ptr() as *const i8;
            if let Some(nsstring_class) = Class::get("NSString") {
                let label_ns: *mut Object = msg_send![nsstring_class, stringWithUTF8String: label_ptr];
                if !label_ns.is_null() {
                    let _: () = msg_send![descriptor, setLabel: label_ns];
                }
            }

            let mut error: *mut Object = ptr::null_mut();
            let set: *mut Object =
                msg_send![device, newResidencySetWithDescriptor: descriptor error: &mut error];

            let _: () = msg_send![descriptor, release];

            if !error.is_null() {
                return Err(MlxError::ResidencySetError(ns_error_message(error)));
            }

            if set.is_null() {
                return Err(MlxError::ResidencySetError(
                    "newResidencySetWithDescriptor:error: returned nil".into(),
                ));
            }

            Ok(Self {
                inner: Arc::new(ResidencySetInner::Active {
                    object: ObjcResidencySet { ptr: set },
                    lock: Mutex::new(()),
                    pending: AtomicBool::new(false),
                }),
            })
        }
    }

    /// Return `true` when this wrapper is the macOS<15 no-op variant.
    #[inline]
    pub(crate) fn is_noop(&self) -> bool {
        matches!(&*self.inner, ResidencySetInner::Noop)
    }

    /// Return `true` if both handles point at the same residency set owner.
    #[inline]
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Stage a buffer allocation to join the residency set (deferred).
    ///
    /// Calls Metal's `addAllocation:` but does NOT commit. The caller is
    /// expected to invoke [`flush_pending`](Self::flush_pending) at the next
    /// CB-submission boundary (or [`commit`](Self::commit) explicitly for a
    /// batched-add path like `MlxBufferPool::alloc_batch`). This matches
    /// llama.cpp's `ggml-metal-device.m:1378-1382` pattern: addAllocation
    /// in a loop, commit ONCE.
    pub(crate) fn add_allocation(&self, buffer: &MTLBufferRef) {
        self.with_active_set(|set, pending| unsafe {
            let _: () = msg_send![set, addAllocation: buffer];
            TEST_ALLOCATION_COUNT.fetch_add(1, Ordering::AcqRel);
            pending.store(true, Ordering::Release);
        });
    }

    /// Stage a buffer allocation to leave the residency set (deferred).
    ///
    /// Calls Metal's `removeAllocation:` but does NOT commit; same
    /// defer-and-flush contract as [`add_allocation`](Self::add_allocation).
    pub(crate) fn remove_allocation(&self, buffer: &MTLBufferRef) {
        self.with_active_set(|set, pending| unsafe {
            let _: () = msg_send![set, removeAllocation: buffer];
            let _ = TEST_ALLOCATION_COUNT.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |count| count.checked_sub(1),
            );
            pending.store(true, Ordering::Release);
        });
    }

    /// Stage all allocations to leave the residency set.
    #[allow(dead_code)]
    pub(crate) fn remove_all_allocations(&self) {
        self.with_active_set(|set, pending| unsafe {
            let _: () = msg_send![set, removeAllAllocations];
            TEST_ALLOCATION_COUNT.store(0, Ordering::Release);
            pending.store(true, Ordering::Release);
        });
    }

    /// Apply staged residency-set changes immediately.
    ///
    /// Always issues a `[set commit]` call. Used by batched paths that have
    /// gathered N add/remove calls and want to flush as a single op (e.g.
    /// `MlxBufferPool::alloc_batch`, `MlxBufferPool::clear`). Also clears
    /// the pending flag so a subsequent `flush_pending` is a no-op.
    pub(crate) fn commit(&self) {
        self.with_active_set(|set, pending| unsafe {
            let _: () = msg_send![set, commit];
            TEST_COMMIT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
            pending.store(false, Ordering::Release);
        });
    }

    /// Defer-and-flush: commit only if a pending add/remove has been
    /// recorded since the last commit, then clear the pending flag.
    ///
    /// Hooked at every `CommandEncoder::commit*` boundary so the
    /// per-allocation commit storm collapses to at most one
    /// `[set commit]` per CB submission. Mirrors the lifetime of
    /// llama.cpp's `ggml_metal_buffer_rset_init` which batches addAllocation
    /// in `ggml-metal-device.m:1378-1382` and commits exactly once.
    ///
    /// Returns whether a commit was actually issued (useful for tests).
    pub(crate) fn flush_pending(&self) -> bool {
        let mut committed = false;
        self.with_active_set(|set, pending| {
            // swap-to-false: only the first concurrent flusher actually
            // issues `[set commit]`; subsequent racers see `false` and skip.
            // The Mutex inside `with_active_set` already serializes against
            // `add_allocation` / `remove_allocation`, so this is belt-and-
            // braces against future lock-free callers.
            if pending.swap(false, Ordering::AcqRel) {
                unsafe {
                    let _: () = msg_send![set, commit];
                }
                TEST_COMMIT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
                committed = true;
            }
        });
        committed
    }

    /// Register this residency set with a command queue.
    pub(crate) fn register_with_queue(&self, queue: &CommandQueueRef) {
        self.with_active_set(|set, _pending| unsafe {
            let _: () = msg_send![queue, addResidencySet: set];
        });
    }

    fn with_active_set(&self, f: impl FnOnce(*mut Object, &AtomicBool)) {
        let ResidencySetInner::Active { object, lock, pending } = &*self.inner else {
            return;
        };

        if let Ok(_guard) = lock.lock() {
            f(object.ptr, pending);
        }
    }
}

/// Whether the process opted out of residency sets via `HF2Q_NO_RESIDENCY=1`.
///
/// The environment check is cached after the first call, which occurs during
/// `MlxDevice` boot.
pub(crate) fn residency_disabled_by_env() -> bool {
    match RESIDENCY_DISABLED_FLAG.load(Ordering::Acquire) {
        1 => false,
        2 => true,
        _ => {
            let disabled = std::env::var("HF2Q_NO_RESIDENCY")
                .map(|value| value == "1")
                .unwrap_or(false);
            RESIDENCY_DISABLED_FLAG.store(if disabled { 2 } else { 1 }, Ordering::Release);
            disabled
        }
    }
}

#[doc(hidden)]
pub fn residency_allocation_count_for_test() -> usize {
    TEST_ALLOCATION_COUNT.load(Ordering::Acquire)
}

#[doc(hidden)]
pub fn residency_commit_call_count_for_test() -> usize {
    TEST_COMMIT_CALL_COUNT.load(Ordering::Acquire)
}

#[doc(hidden)]
pub fn reset_residency_test_counters() {
    TEST_ALLOCATION_COUNT.store(0, Ordering::Release);
    TEST_COMMIT_CALL_COUNT.store(0, Ordering::Release);
}

#[doc(hidden)]
pub fn reset_residency_env_cache_for_test() {
    RESIDENCY_DISABLED_FLAG.store(0, Ordering::Release);
}

#[doc(hidden)]
pub fn macos_15_or_newer_for_test() -> bool {
    macos_15_or_newer()
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_15_or_newer() -> bool {
    let Some(process_info_class) = Class::get("NSProcessInfo") else {
        return false;
    };

    let version = NSOperatingSystemVersion {
        majorVersion: 15,
        minorVersion: 0,
        patchVersion: 0,
    };

    unsafe {
        let process_info: *mut Object = msg_send![process_info_class, processInfo];
        if process_info.is_null() {
            return false;
        }

        let ok: BOOL = msg_send![process_info, isOperatingSystemAtLeastVersion: version];
        ok == YES
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn macos_15_or_newer() -> bool {
    false
}

unsafe fn ns_error_message(error: *mut Object) -> String {
    let desc: *mut Object = msg_send![error, localizedDescription];
    if desc.is_null() {
        return "MTLResidencySet creation failed".into();
    }

    let text: *const std::os::raw::c_char = msg_send![desc, UTF8String];
    if text.is_null() {
        return "MTLResidencySet creation failed".into();
    }

    CStr::from_ptr(text).to_string_lossy().into_owned()
}
