//! Ignored Native47 falsification probe, not a pool implementation or admission.
//! A fixed heap backing domain is distinct from host/driver object metadata.
//! Requested size may round; no pre-allocation physical bound is inferred from
//! post-allocation observations. No GPU commands, models or implicit zeroing.

use metal::foreign_types::ForeignType;
use objc::{sel, sel_impl};

use crate::encoder::RecordedBinding;
use crate::{DType, MlxBuffer};

const HEAP_REQUEST: u64 = 64 * 1024;
const PROBE_LIMIT: u64 = 1024 * 1024;
const BUFFER_REQUEST: u64 = 4096;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RequestedHazard {
    Default,
    Tracked,
}

impl RequestedHazard {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Tracked => "tracked",
        }
    }
}

fn unavailable(requested: RequestedHazard, reason: &'static str) {
    eprintln!(
        concat!(
            "Q5_SHARED_HEAP_CAPABILITY {{\"status\":\"unavailable_or_unknown\",",
            "\"requested_hazard_mode\":\"{}\",\"reason\":\"{}\",\"runtime_admission\":false}}"
        ),
        requested.name(),
        reason
    );
}

fn optional(value: Option<u64>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn sample(
    requested: RequestedHazard,
    device: &metal::DeviceRef,
    heap: Option<&metal::HeapRef>,
    resource: Option<(&metal::BufferRef, usize, u64)>,
    alignment: u64,
    phase: &'static str,
) {
    let (heap_size, heap_used, heap_current, heap_available, heap_hazard) = match heap {
        Some(heap) => (
            Some(heap.size()),
            Some(heap.used_size()),
            Some(heap.current_allocated_size()),
            Some(heap.max_available_size_with_alignment(alignment)),
            Some(heap.hazard_tracking_mode() as u64),
        ),
        None => (None, None, None, None, None),
    };
    let (length, logical, view_offset, allocated, same_heap, heap_offset, resource_hazard) =
        match resource {
            Some((buffer, logical, view_offset)) => {
                // SAFETY: buffer has a live owning wrapper/raw/recorded reference.
                // Unlike metal-rs Resource::heap(), this query preserves null.
                let parent: *mut metal::MTLHeap = unsafe { objc::msg_send![buffer, heap] };
                let same_heap = heap.map(|expected| {
                    parent == expected as *const metal::HeapRef as *mut metal::MTLHeap
                });
                assert_eq!(
                    same_heap,
                    Some(true),
                    "heap resource must name its live parent"
                );
                (
                    Some(buffer.length()),
                    Some(logical as u64),
                    Some(view_offset),
                    Some(buffer.allocated_size()),
                    same_heap,
                    Some(buffer.heap_offset()),
                    Some(buffer.hazard_tracking_mode() as u64),
                )
            }
            None => (None, None, None, None, None, None, None),
        };
    let same_heap = same_heap.map_or_else(|| "null".into(), |value| value.to_string());
    eprintln!(
        concat!(
            "Q5_SHARED_HEAP_OBSERVATION {{\"schema\":\"native.shared-heap-observation.v1\",",
            "\"requested_hazard_mode\":\"{}\",\"phase\":\"{}\",\"heap_requested_bytes\":{},\"buffer_requested_bytes\":{},",
            "\"heap_size\":{},\"heap_used_size\":{},\"heap_current_allocated_size\":{},",
            "\"alignment\":{},\"heap_max_available_size\":{},",
            "\"device_current_allocated_size\":{},\"resource_length\":{},",
            "\"view_logical_bytes\":{},\"view_byte_offset\":{},",
            "\"resource_allocated_size\":{},\"resource_heap_matches\":{},",
            "\"heap_offset\":{},\"heap_offset_meaning\":\"automatic_nonplacement_zero_sentinel\",",
            "\"heap_hazard_tracking_mode\":{},\"resource_hazard_tracking_mode\":{},",
            "\"hazard_mode_encoding\":\"0_default_1_untracked_2_tracked\",",
            "\"runtime_admission\":false}}"
        ),
        requested.name(), phase,
        HEAP_REQUEST,
        BUFFER_REQUEST,
        optional(heap_size),
        optional(heap_used),
        optional(heap_current),
        alignment,
        optional(heap_available),
        device.current_allocated_size(),
        optional(length),
        optional(logical),
        optional(view_offset),
        optional(allocated),
        same_heap,
        optional(heap_offset),
        optional(heap_hazard),
        optional(resource_hazard)
    );
}

fn probe(device: &metal::DeviceRef, requested: RequestedHazard) {
    // Check APIs before setting a descriptor field or sending a heap message.
    // All unavailable paths retain the explicit capability status above.
    let supported: bool =
        unsafe { objc::msg_send![device, respondsToSelector: sel!(newHeapWithDescriptor:)] };
    if !supported {
        unavailable(requested, "newHeapWithDescriptor selector unavailable");
        return;
    }
    let descriptor = metal::HeapDescriptor::new();
    let descriptor_type_available: bool =
        unsafe { objc::msg_send![&*descriptor, respondsToSelector: sel!(setType:)] };
    if !descriptor_type_available {
        unavailable(
            requested,
            "Automatic heap descriptor type selector unavailable",
        );
        return;
    }
    descriptor.set_size(HEAP_REQUEST);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_heap_type(metal::MTLHeapType::Automatic);
    let mut options = metal::MTLResourceOptions::StorageModeShared;
    if requested == RequestedHazard::Tracked {
        // SDK MTLHeap.h documents explicit Tracked and matching child mode.
        // The CPU-only probe cannot establish GPU ordering or compatibility.
        let available: bool = unsafe {
            objc::msg_send![&*descriptor, respondsToSelector: sel!(setHazardTrackingMode:)]
        };
        if !available {
            unavailable(requested, "Tracked heap descriptor selector unavailable");
            return;
        }
        descriptor.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
        options |= metal::MTLResourceOptions::HazardTrackingModeTracked;
    }
    let layout = device.heap_buffer_size_and_align(BUFFER_REQUEST, options);
    eprintln!("Q5_SHARED_HEAP_LAYOUT {{\"requested_hazard_mode\":\"{}\",\"required_size\":{},\"required_alignment\":{},\"runtime_admission\":false}}",
              requested.name(), layout.size, layout.align);
    if layout.size == 0
        || layout.size > HEAP_REQUEST
        || layout.align == 0
        || !layout.align.is_power_of_two()
    {
        unavailable(
            requested,
            "unsupported or out-of-envelope Shared buffer layout",
        );
        return;
    }
    sample(
        requested,
        device,
        None,
        None,
        layout.align,
        "before_heap_creation",
    );
    // SAFETY: descriptor is live and requests Shared/Automatic, not Private.
    // metal-rs Device::new_heap assumes a nonnull return; keep the SDK's nil.
    let raw: *mut metal::MTLHeap =
        unsafe { objc::msg_send![device, newHeapWithDescriptor: &*descriptor] };
    if raw.is_null() {
        unavailable(requested, "Shared Automatic heap creation returned nil");
        return;
    }
    // SAFETY: newHeap returned nonnull owned (+1) reference.
    let heap = unsafe { metal::Heap::from_ptr(raw) };
    if requested == RequestedHazard::Tracked
        && heap.hazard_tracking_mode() != metal::MTLHazardTrackingMode::Tracked
    {
        unavailable(requested, "heap did not resolve requested Tracked mode");
        return;
    }
    if heap.storage_mode() != metal::MTLStorageMode::Shared
        || heap.heap_type() != metal::MTLHeapType::Automatic
    {
        unavailable(
            requested,
            "created heap does not match requested Shared Automatic mode",
        );
        return;
    }
    sample(
        requested,
        device,
        Some(&heap),
        None,
        layout.align,
        "empty_heap",
    );
    if heap.size() == 0 || heap.size() > PROBE_LIMIT {
        unavailable(
            requested,
            "rounded heap size exceeds bounded probe envelope",
        );
        return;
    }

    // Heap is a strong outer owner until all resource owners and this inner
    // autorelease scope end, including on assertion unwinding. No aliasable
    // heap-region reuse is requested: aliases below are wrapper/view owners.
    let completed = objc::rc::autoreleasepool(|| {
        let Some(buffer) = heap.new_buffer(BUFFER_REQUEST, options) else {
            unavailable(requested, "Shared heap buffer creation returned nil");
            return false;
        };
        if requested == RequestedHazard::Tracked
            && buffer.hazard_tracking_mode() != metal::MTLHazardTrackingMode::Tracked
        {
            unavailable(
                requested,
                "child resource did not resolve requested Tracked mode",
            );
            return false;
        }
        if buffer.storage_mode() != metal::MTLStorageMode::Shared || buffer.contents().is_null() {
            unavailable(requested, "Shared heap buffer lacks writable CPU contents");
            return false;
        }
        assert_eq!(buffer.length(), BUFFER_REQUEST);
        // SAFETY: this sole Shared root owns BUFFER_REQUEST writable bytes;
        // no GPU work or other CPU references exist. Initialize every byte
        // before any CPU read; no zero-initialized-heap promise is assumed.
        let contents = unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), BUFFER_REQUEST as usize)
        };
        for (index, byte) in contents.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        assert!(contents
            .iter()
            .enumerate()
            .all(|(index, byte)| *byte == (index % 251) as u8));

        // SDK MTLHeap.h: newBuffer returns nil if the heap is full. A length
        // greater than this heap's complete size cannot fit in it. This asks
        // only the fixed heap, never the global allocator or a Private fallback.
        let oversized = heap.size().checked_add(1).expect("bounded heap size");
        if oversized <= PROBE_LIMIT && oversized <= device.max_buffer_length() {
            let impossible = heap.new_buffer(oversized, options);
            let refused = impossible.is_none();
            drop(impossible);
            eprintln!("Q5_SHARED_HEAP_CAPACITY_CONTROL {{\"requested_hazard_mode\":\"{}\",\"requested_bytes\":{oversized},\"nil_observed\":{refused},\"runtime_admission\":false}}", requested.name());
            assert!(
                refused,
                "fixed heap must refuse a resource larger than itself"
            );
        } else {
            eprintln!("Q5_SHARED_HEAP_CAPACITY_CONTROL {{\"requested_hazard_mode\":\"{}\",\"status\":\"not_run_request_envelope\",\"runtime_admission\":false}}", requested.name());
        }

        let wrapper = MlxBuffer::from_raw(buffer, DType::U8, vec![BUFFER_REQUEST as usize]);
        sample(
            requested,
            device,
            Some(&heap),
            Some((wrapper.metal_buffer(), wrapper.data_byte_len(), 0)),
            layout.align,
            "root_initialized",
        );
        let cloned = wrapper.clone();
        let view = cloned.slice_view(64, 128);
        sample(
            requested,
            device,
            Some(&heap),
            Some((
                view.metal_buffer(),
                view.data_byte_len(),
                view.byte_offset(),
            )),
            layout.align,
            "clone_and_view",
        );
        let raw = view.metal_buffer().clone();
        let binding = RecordedBinding::Buffer {
            metal_buffer: raw.clone(),
            offset: view.byte_offset(),
        };
        drop(view);
        drop(cloned);
        drop(wrapper);
        sample(
            requested,
            device,
            Some(&heap),
            Some((&raw, 128, 64)),
            layout.align,
            "raw_and_recorded",
        );
        drop(raw);
        if let RecordedBinding::Buffer {
            metal_buffer,
            offset,
        } = &binding
        {
            sample(
                requested,
                device,
                Some(&heap),
                Some((metal_buffer, 128, *offset)),
                layout.align,
                "recorded_final_owner",
            );
        }
        drop(binding);
        // No freed resource query: heap and device are the only live subjects.
        sample(
            requested,
            device,
            Some(&heap),
            None,
            layout.align,
            "final_explicit_resource_owner_released",
        );
        true
    });
    sample(
        requested,
        device,
        Some(&heap),
        None,
        layout.align,
        "after_resource_autorelease_scope",
    );
    drop(heap);
    // No heap/resource query after their final explicit owner drops either.
    sample(
        requested,
        device,
        None,
        None,
        layout.align,
        "heap_explicit_owner_released",
    );
    if completed {
        eprintln!("Q5_SHARED_HEAP_CAPABILITY {{\"requested_hazard_mode\":\"{}\",\"status\":\"bounded_probe_completed\",\"runtime_admission\":false}}", requested.name());
    }
}

#[test]
#[ignore = "experimental Shared Automatic heap observations, no admission or allocator change"]
fn bounded_shared_automatic_heap_observations() {
    run(RequestedHazard::Default);
}

#[test]
#[ignore = "experimental TRACKED CPU-only heap observations, not GPU hazard-ordering proof"]
fn bounded_shared_automatic_tracked_heap_observations() {
    run(RequestedHazard::Tracked);
}

fn run(requested: RequestedHazard) {
    objc::rc::autoreleasepool(|| {
        let Some(device) = metal::Device::system_default() else {
            unavailable(requested, "Metal device unavailable");
            return;
        };
        probe(&device, requested);
    });
}
