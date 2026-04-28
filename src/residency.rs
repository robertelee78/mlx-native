//! Raw Objective-C wrapper for Metal residency sets.
//!
//! `metal = 0.33` does not expose `MTLResidencySet`, so this module uses
//! `objc = 0.2` directly. The wrapper is no-op on macOS versions before 15.0.

use std::ffi::CStr;
use std::ptr;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
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
    },
    Noop,
}

impl Drop for ResidencySetInner {
    fn drop(&mut self) {
        let Self::Active { object, lock } = self else {
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

    /// Stage a buffer allocation to join the residency set.
    pub(crate) fn add_allocation(&self, buffer: &MTLBufferRef) {
        self.with_active_set(|set| unsafe {
            let _: () = msg_send![set, addAllocation: buffer];
            TEST_ALLOCATION_COUNT.fetch_add(1, Ordering::AcqRel);
        });
    }

    /// Stage a buffer allocation to leave the residency set.
    pub(crate) fn remove_allocation(&self, buffer: &MTLBufferRef) {
        self.with_active_set(|set| unsafe {
            let _: () = msg_send![set, removeAllocation: buffer];
            let _ = TEST_ALLOCATION_COUNT.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |count| count.checked_sub(1),
            );
        });
    }

    /// Stage all allocations to leave the residency set.
    #[allow(dead_code)]
    pub(crate) fn remove_all_allocations(&self) {
        self.with_active_set(|set| unsafe {
            let _: () = msg_send![set, removeAllAllocations];
            TEST_ALLOCATION_COUNT.store(0, Ordering::Release);
        });
    }

    /// Apply staged residency-set changes.
    pub(crate) fn commit(&self) {
        self.with_active_set(|set| unsafe {
            let _: () = msg_send![set, commit];
            TEST_COMMIT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
        });
    }

    /// Register this residency set with a command queue.
    pub(crate) fn register_with_queue(&self, queue: &CommandQueueRef) {
        self.with_active_set(|set| unsafe {
            let _: () = msg_send![queue, addResidencySet: set];
        });
    }

    fn with_active_set(&self, f: impl FnOnce(*mut Object)) {
        let ResidencySetInner::Active { object, lock } = &*self.inner else {
            return;
        };

        if let Ok(_guard) = lock.lock() {
            f(object.ptr);
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
