use super::{BoundedPoolError as E, Result};
use crate::MlxDevice;
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use objc::{sel, sel_impl};

pub(super) fn options() -> metal::MTLResourceOptions {
    metal::MTLResourceOptions::StorageModeShared
        | metal::MTLResourceOptions::HazardTrackingModeTracked
}
pub(super) fn heap(device: &MlxDevice, bytes: u64) -> Result<metal::Heap> {
    #[cfg(test)]
    if faults::heap_call() {
        return Err(E::AllocationFailed);
    }
    let descriptor = metal::HeapDescriptor::new();
    let supported: bool = unsafe {
        objc::msg_send![device.metal_device(), respondsToSelector: sel!(newHeapWithDescriptor:)]
    };
    let tracked: bool =
        unsafe { objc::msg_send![&*descriptor, respondsToSelector: sel!(setHazardTrackingMode:)] };
    let typed: bool = unsafe { objc::msg_send![&*descriptor, respondsToSelector: sel!(setType:)] };
    if !supported || !tracked || !typed {
        return Err(E::Unsupported);
    }
    descriptor.set_size(bytes);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_heap_type(metal::MTLHeapType::Automatic);
    descriptor.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
    let raw: *mut metal::MTLHeap =
        unsafe { objc::msg_send![device.metal_device(), newHeapWithDescriptor: &*descriptor] };
    if raw.is_null() {
        return Err(E::AllocationFailed);
    }
    // SAFETY: nullable +1 constructor checked before adopting its ownership.
    Ok(unsafe { metal::Heap::from_ptr(raw) })
}
pub(super) fn heap_valid(heap: &metal::HeapRef) -> bool {
    #[cfg(test)]
    if faults::invalid_heap() {
        return false;
    }
    heap.storage_mode() == metal::MTLStorageMode::Shared
        && heap.heap_type() == metal::MTLHeapType::Automatic
        && heap.hazard_tracking_mode() == metal::MTLHazardTrackingMode::Tracked
}
pub(super) fn validate(
    buffer: &metal::BufferRef,
    heap: &metal::HeapRef,
    bytes: usize,
) -> Result<()> {
    #[cfg(test)]
    if faults::get().invalid_resource {
        return Err(E::ResourceMismatch);
    }
    let parent: *mut metal::MTLHeap = unsafe { objc::msg_send![buffer, heap] };
    if parent != heap.as_ptr()
        || buffer.length() != bytes as u64
        || buffer.contents().is_null()
        || buffer.is_aliasable()
        || buffer.storage_mode() != metal::MTLStorageMode::Shared
        || buffer.hazard_tracking_mode() != metal::MTLHazardTrackingMode::Tracked
        || buffer.set_purgeable_state(metal::MTLPurgeableState::KeepCurrent)
            != metal::MTLPurgeableState::NonVolatile
    {
        return Err(E::ResourceMismatch);
    }
    Ok(())
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Identity {
    device: usize,
    registry: u64,
    queue: usize,
    residency: Option<usize>,
}
impl Identity {
    pub fn of(device: &MlxDevice) -> Self {
        Self {
            device: device.metal_device().as_ptr() as usize,
            registry: device.registry_id(),
            queue: device.metal_queue().as_ptr() as usize,
            residency: device.residency_set().map(|set| set.identity()),
        }
    }
    pub fn check(self, device: &MlxDevice) -> Result<()> {
        if self == Self::of(device) {
            Ok(())
        } else {
            Err(E::IdentityMismatch)
        }
    }
}

pub(super) fn child(heap: &metal::HeapRef, bytes: u64) -> Option<metal::Buffer> {
    #[cfg(test)]
    if faults::child_call() {
        return None;
    }
    let result = heap.new_buffer(bytes, options());
    #[cfg(test)]
    if faults::get().dirty_child {
        if let Some(buffer) = &result {
            // Check the actual returned allocation before touching CPU memory.
            // A mismatched driver resource must fail without any unsafe access.
            assert_eq!(buffer.storage_mode(), metal::MTLStorageMode::Shared);
            assert_eq!(buffer.length(), bytes);
            assert!(!buffer.contents().is_null());
            // Exclusive unpublished test root: dirty all bytes before real zeroing.
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0x5a, bytes as usize);
            }
        }
    }
    result
}

#[cfg(test)]
pub(super) mod faults {
    use std::cell::RefCell;
    #[derive(Clone, Copy, Default)]
    pub struct Faults {
        pub heap_nil: bool,
        pub child_nil_at: usize,
        pub invalid_heap: bool,
        pub invalid_resource: bool,
        pub dirty_child: bool,
        pub heap_calls: usize,
        pub child_calls: usize,
    }
    thread_local! { static FAULTS: RefCell<Faults> = RefCell::new(Faults::default()); }
    pub fn set(value: Faults) {
        FAULTS.with(|state| *state.borrow_mut() = value);
    }
    pub fn get() -> Faults {
        FAULTS.with(|state| *state.borrow())
    }
    pub fn heap_call() -> bool {
        FAULTS.with(|state| {
            let mut state = state.borrow_mut();
            state.heap_calls += 1;
            state.heap_nil
        })
    }
    pub fn child_call() -> bool {
        FAULTS.with(|state| {
            let mut state = state.borrow_mut();
            state.child_calls += 1;
            state.child_nil_at == state.child_calls
        })
    }
    pub fn invalid_heap() -> bool {
        get().invalid_heap
    }
}
