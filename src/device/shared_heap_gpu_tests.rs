//! Tiny ignored same-queue Tracked control, not pool/admission/performance proof.
//! Root must supervise the test process externally: synchronous GPU waits are
//! not a watchdog. CPU heap probe remains a separately bound historical result.

use metal::foreign_types::ForeignType;
use objc::{sel, sel_impl};

use super::MlxDevice;
use crate::ops::elementwise::{elementwise_add, elementwise_mul};
use crate::{DType, KernelRegistry, MlxBuffer};

const N: usize = 256;
const BYTES: u64 = (N * 4) as u64;
const HEAP_BYTES: u64 = 64 * 1024;
const MAX_HEAP: u64 = 1024 * 1024;

fn unavailable(reason: &'static str) {
    eprintln!("Q5_HEAP_GPU_CAPABILITY {{\"status\":\"unavailable_or_unknown\",\"reason\":\"{reason}\",\"runtime_admission\":false}}");
}

fn options() -> metal::MTLResourceOptions {
    metal::MTLResourceOptions::StorageModeShared
        | metal::MTLResourceOptions::HazardTrackingModeTracked
}

fn create_heap(device: &metal::DeviceRef) -> Option<metal::Heap> {
    let descriptor = metal::HeapDescriptor::new();
    // All mode properties are macOS10.15+; refuse unavailable APIs rather
    // than silently run a Private/Untracked or changed control.
    let supported: bool =
        unsafe { objc::msg_send![device, respondsToSelector: sel!(newHeapWithDescriptor:)] };
    let tracked: bool =
        unsafe { objc::msg_send![&*descriptor, respondsToSelector: sel!(setHazardTrackingMode:)] };
    let typed: bool = unsafe { objc::msg_send![&*descriptor, respondsToSelector: sel!(setType:)] };
    if !supported || !tracked || !typed {
        unavailable("required Shared Automatic Tracked API unavailable");
        return None;
    }
    let layout = device.heap_buffer_size_and_align(BYTES, options());
    let packing = layout
        .size
        .checked_add(layout.align.saturating_sub(1))
        .map(|value| value / layout.align.max(1) * layout.align)
        .and_then(|value| value.checked_mul(4));
    if layout.size == 0
        || layout.align == 0
        || !layout.align.is_power_of_two()
        || !packing.is_some_and(|required| required <= HEAP_BYTES)
    {
        unavailable("four-child heap layout cannot fit fixed 64 KiB request");
        return None;
    }
    eprintln!("Q5_HEAP_GPU_LAYOUT {{\"requested_heap_bytes\":{HEAP_BYTES},\"child_bytes\":{BYTES},\"child_required_size\":{},\"alignment\":{},\"four_child_packing\":{},\"runtime_admission\":false}}",
              layout.size, layout.align, packing.unwrap());
    descriptor.set_size(HEAP_BYTES);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_heap_type(metal::MTLHeapType::Automatic);
    descriptor.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Tracked);
    // SAFETY: descriptor is live. Preserve nullable SDK return; the binding's
    // Device::new_heap wrapper assumes nonnull and is unsuitable for a probe.
    let raw: *mut metal::MTLHeap =
        unsafe { objc::msg_send![device, newHeapWithDescriptor: &*descriptor] };
    if raw.is_null() {
        unavailable("Shared Automatic Tracked heap creation returned nil");
        return None;
    }
    // SAFETY: newHeap returned a nonnull owned (+1) object.
    let heap = unsafe { metal::Heap::from_ptr(raw) };
    if heap.size() == 0
        || heap.size() > MAX_HEAP
        || heap.storage_mode() != metal::MTLStorageMode::Shared
        || heap.heap_type() != metal::MTLHeapType::Automatic
        || heap.hazard_tracking_mode() != metal::MTLHazardTrackingMode::Tracked
    {
        unavailable("created heap outside size/mode envelope");
        return None;
    }
    eprintln!("Q5_HEAP_GPU_HEAP {{\"size\":{},\"current_allocated_size\":{},\"hazard_mode\":2,\"runtime_admission\":false}}", heap.size(), heap.current_allocated_size());
    Some(heap)
}

fn child(device: &MlxDevice, heap: Option<&metal::HeapRef>, index: usize) -> Option<MlxBuffer> {
    let buffer = match heap {
        Some(heap) => heap.new_buffer(BYTES, options()),
        None => {
            // SAFETY: bounded direct request; preserve raw nullable return.
            let raw: *mut metal::MTLBuffer = unsafe {
                objc::msg_send![device.metal_device(), newBufferWithLength: BYTES options: options()]
            };
            if raw.is_null() {
                None
            } else {
                // SAFETY: newBuffer returned a nonnull owned (+1) reference.
                Some(unsafe { metal::Buffer::from_ptr(raw) })
            }
        }
    };
    let Some(buffer) = buffer else {
        unavailable("bounded child allocation returned nil");
        return None;
    };
    if buffer.length() != BYTES
        || buffer.contents().is_null()
        || buffer.storage_mode() != metal::MTLStorageMode::Shared
        || buffer.hazard_tracking_mode() != metal::MTLHazardTrackingMode::Tracked
    {
        unavailable("child lacks required length/CPU Shared/Tracked contract");
        return None;
    }
    // SAFETY: buffer is live. Preserve nullable heap query (standalone=None).
    let parent: *mut metal::MTLHeap = unsafe { objc::msg_send![&*buffer, heap] };
    let expected = heap.map_or(std::ptr::null_mut(), |heap| {
        heap as *const metal::HeapRef as *mut metal::MTLHeap
    });
    assert_eq!(parent, expected, "child parent identity mismatch");
    // Four simultaneously live nonaliasable resources are the contract.
    // Automatic heapOffset0 is NOT a placement offset/nonoverlap witness.
    assert!(!buffer.is_aliasable());
    eprintln!("Q5_HEAP_GPU_RESOURCE {{\"arm\":\"{}\",\"index\":{index},\"length\":{},\"allocated_size\":{},\"hazard_mode\":2,\"heap_matches\":true,\"heap_offset\":{},\"heap_offset_meaning\":\"automatic_nonplacement_zero_sentinel\",\"runtime_admission\":false}}",
              if heap.is_some() { "heap" } else { "standalone" }, buffer.length(), buffer.allocated_size(), buffer.heap_offset());
    Some(MlxBuffer::from_raw(buffer, DType::F32, vec![N]))
}

fn initial(index: usize, buffer: usize) -> f32 {
    match buffer {
        0 => ((index % 17) as i32 - 8) as f32,
        1 => 2.0,
        2 => -99.0,
        3 => -77.0,
        _ => unreachable!(),
    }
}

fn arm(device: &MlxDevice, heap: Option<&metal::HeapRef>) -> Option<[Vec<u32>; 4]> {
    let mut buffers = Vec::with_capacity(4);
    for index in 0..4 {
        buffers.push(child(device, heap, index)?);
    }
    for left in 0..4 {
        for right in left + 1..4 {
            assert!(
                buffers[left].metal_buffer().as_ptr() != buffers[right].metal_buffer().as_ptr(),
                "simultaneously live children must be distinct resource objects"
            );
        }
    }
    for (index, buffer) in buffers.iter_mut().enumerate() {
        // Native typed Shared slice checks dtype/size. Fully write each own
        // logical byte before any read; never access CPU data during GPU work.
        for (element, value) in buffer.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
            *value = initial(element, index);
        }
    }
    let [a, b, t, o] = buffers.as_slice() else {
        unreachable!()
    };
    let mut registry = KernelRegistry::new();
    let mut first = device.command_encoder().expect("first native encoder");
    let mut second = device
        .command_encoder()
        .expect("second native encoder, same queue");
    elementwise_add(
        &mut first,
        &mut registry,
        device.metal_device(),
        a,
        b,
        t,
        N,
        DType::F32,
    )
    .expect("encode native add before first commit");
    elementwise_mul(
        &mut second,
        &mut registry,
        device.metal_device(),
        t,
        b,
        o,
        N,
        DType::F32,
    )
    .expect("encode native mul before either command is submitted");
    // All fallible preparation finishes before submission. Consecutive commits
    // retain the GPU dependency under test without a CPU wait between them.
    first.commit();
    second.commit();
    // Deliberately no CPU wait between the commits. Both owners remain live
    // through both waits AND CPU verification; do not commit a second time.
    let second_result = second.wait_until_completed();
    let first_result = first.wait_until_completed();
    eprintln!("Q5_HEAP_GPU_COMMANDS {{\"arm\":\"{}\",\"same_queue\":true,\"queue_identity\":\"single_test_queue\",\"cb1_completed_ok\":{},\"cb2_completed_ok\":{},\"wait_between_commits\":false,\"runtime_admission\":false}}",
              if heap.is_some() { "heap" } else { "standalone" }, first_result.is_ok(), second_result.is_ok());
    second_result.expect("consumer command completed successfully");
    first_result.expect("producer command completed successfully");
    let actual: [Vec<u32>; 4] = std::array::from_fn(|buffer| {
        buffers[buffer]
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    });
    for element in 0..N {
        let input = initial(element, 0);
        let expected = [input, 2.0, input + 2.0, 2.0 * (input + 2.0)];
        for buffer in 0..4 {
            assert_eq!(
                actual[buffer][element],
                expected[buffer].to_bits(),
                "oracle mismatch arm={} buffer={buffer} element={element}",
                if heap.is_some() { "heap" } else { "standalone" }
            );
        }
    }
    // Explicitly retain both command-buffer owners through oracle readback.
    drop(second);
    drop(first);
    eprintln!("Q5_HEAP_GPU_ORACLE {{\"arm\":\"{}\",\"elements_per_buffer\":{N},\"buffers_checked\":4,\"exact_bits_equal\":true,\"runtime_admission\":false}}", if heap.is_some() { "heap" } else { "standalone" });
    Some(actual)
}

#[test]
#[ignore = "tiny GPU Tracked Shared heap parity; root must externally supervise process timeout"]
fn tracked_shared_heap_two_command_buffers_match_standalone_and_cpu() {
    // These controls change native command submission/ownership or profiling.
    // Require a fresh default retained-reference process, never mutate env.
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
        assert!(
            std::env::var_os(name).is_none(),
            "control requires {name} absent"
        );
    }
    objc::rc::autoreleasepool(|| {
        let Some(metal_device) = metal::Device::system_default() else {
            unavailable("Metal device unavailable");
            return;
        };
        if BYTES > metal_device.max_buffer_length() {
            unavailable("tiny buffer exceeds device limit");
            return;
        }
        let queue = metal_device.new_command_queue();
        // Explicitly no residency set; this tests direct bindings, not a pool
        // or useHeap/argument-buffer side path. The device/queue are shared by
        // both arms and all four deciding command buffers.
        let device = MlxDevice {
            device: metal_device,
            queue,
            residency_set: None,
        };
        let Some(heap) = create_heap(device.metal_device()) else {
            return;
        };
        eprintln!("Q5_HEAP_GPU_IDENTITY {{\"device_registry_id\":{},\"arm_order\":\"standalone_then_heap\",\"queue_identity\":\"single_test_queue\",\"unretained_refs\":false,\"indirect_bindings\":false,\"runtime_admission\":false}}", device.registry_id());
        let standalone = objc::rc::autoreleasepool(|| arm(&device, None));
        let Some(standalone) = standalone else {
            return;
        };
        // Heap strong owner encloses every child and encoder/autorelease owner.
        let mapped = objc::rc::autoreleasepool(|| arm(&device, Some(&heap)));
        let Some(mapped) = mapped else {
            return;
        };
        assert_eq!(standalone, mapped, "heap versus standalone exact parity");
        // After arm scopes, query only still-live heap/device, no freed child.
        eprintln!("Q5_HEAP_GPU_CAPABILITY {{\"status\":\"bounded_gpu_probe_completed\",\"exact_cpu_and_standalone_parity\":true,\"heap_size\":{},\"heap_used_size\":{},\"heap_current_allocated_size\":{},\"runtime_admission\":false,\"production_pool_qualified\":false}}", heap.size(), heap.used_size(), heap.current_allocated_size());
    });
}
