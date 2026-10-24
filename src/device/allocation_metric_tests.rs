//! Ignored raw observations, not allocation upper bounds or admission proof.
//! No command encoder, kernel, model, GPU dispatch or payload read.

use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use objc::{sel, sel_impl};

use super::{host_page_size, MlxDevice};
use crate::encoder::RecordedBinding;
use crate::{DType, MlxBuffer};

const OFFSET: usize = 64;
const MAPPED_LOGICAL: usize = 4096;
const REQUEST_LIMIT: usize = 1024 * 1024;
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
enum Mode {
    Default,
    Disabled,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Disabled => "disabled",
        }
    }

    fn device(self) -> MlxDevice {
        match self {
            Self::Default => MlxDevice::new().expect("bounded probe device"),
            Self::Disabled => {
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
}

struct Fixture {
    file: File,
    path: PathBuf,
    len: usize,
}

impl Fixture {
    fn new(page: usize) -> Self {
        let len = (OFFSET + MAPPED_LOGICAL).div_ceil(page) * page;
        assert!(len <= 2 * page && len < REQUEST_LIMIT);
        let serial = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlx-q5-metrics-{}-{serial}.bin",
            std::process::id()
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("private bounded file");
        file.set_len(len as u64)
            .expect("size file before immutable mapping");
        Self { file, path, len }
    }

    fn mapping(&self) -> Arc<Mmap> {
        // SAFETY: this private regular file is sized before mapping, never
        // written/truncated afterward and retained beyond all mapping owners.
        Arc::new(
            unsafe { MmapOptions::new().len(self.len).map(&self.file) }
                .expect("tiny read-only mapping"),
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct FlushOnDrop<'a>(&'a MlxDevice);

impl Drop for FlushOnDrop<'_> {
    fn drop(&mut self) {
        flush(self.0);
    }
}

fn flush(device: &MlxDevice) {
    if let Some(set) = device.residency_set() {
        set.flush_pending();
    }
}

fn current(device: &MlxDevice) -> Option<u64> {
    // SAFETY: read-only selector availability query on the live device.
    let available: bool =
        unsafe { objc::msg_send![&*device.device, respondsToSelector: sel!(currentAllocatedSize)] };
    available.then(|| device.device.current_allocated_size() as u64)
}

fn number(value: Option<u64>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

struct Observation<'a> {
    mode: Mode,
    kind: &'static str,
    requested: usize,
    page: usize,
    baseline: Option<u64>,
    escrow: Option<&'a Arc<Mmap>>,
}

impl Observation<'_> {
    fn sample(
        &self,
        device: &MlxDevice,
        phase: &'static str,
        // This borrow must come from a live owning wrapper/raw/recorded owner.
        resource: Option<(&metal::BufferRef, usize, u64)>,
    ) {
        let (length, logical, offset, available, allocated) = match resource {
            Some((buffer, logical, offset)) => {
                // SAFETY: buffer is borrowed from a still-live strong owner.
                let available: bool =
                    unsafe { objc::msg_send![buffer, respondsToSelector: sel!(allocatedSize)] };
                (
                    Some(buffer.length() as u64),
                    Some(logical as u64),
                    Some(offset),
                    Some(available),
                    available.then(|| buffer.allocated_size() as u64),
                )
            }
            None => (None, None, None, None, None),
        };
        let current = current(device);
        let delta = current
            .zip(self.baseline)
            .map(|(now, before)| i128::from(now) - i128::from(before));
        let delta = delta.map_or_else(|| "null".into(), |value| value.to_string());
        let available = available.map_or_else(|| "null".into(), |value| value.to_string());
        let escrow_count = self.escrow.map(|map| Arc::strong_count(map) as u64);
        // All string fields are closed literals, not unescaped external input.
        eprintln!(
            concat!(
                "Q5_ALLOCATION_OBSERVATION {{\"schema\":\"native.allocation-observation.v1\",",
                "\"mode\":\"{}\",\"kind\":\"{}\",\"phase\":\"{}\",",
                "\"host_page_bytes\":{},\"requested_logical_bytes\":{},",
                "\"actual_residency\":{},\"data_byte_len\":{},\"byte_offset\":{},",
                "\"resource_length\":{},\"resource_allocated_size_available\":{},",
                "\"resource_allocated_size\":{},\"device_current_allocated_available\":{},",
                "\"device_current_allocated_size\":{},\"device_delta_from_baseline\":{},",
                "\"escrow_strong_count\":{},\"runtime_admission\":false}}"
            ),
            self.mode.name(),
            self.kind,
            phase,
            self.page,
            self.requested,
            device.residency_sets_enabled(),
            number(logical),
            number(offset),
            number(length),
            available,
            number(allocated),
            current.is_some(),
            number(current),
            delta,
            number(escrow_count)
        );
    }
}

fn owners(device: &MlxDevice, buffer: MlxBuffer, observation: &Observation<'_>) {
    observation.sample(
        device,
        "root",
        Some((
            buffer.metal_buffer(),
            buffer.data_byte_len(),
            buffer.byte_offset(),
        )),
    );
    let cloned = buffer.clone();
    let view = cloned.slice_view(0, 1);
    observation.sample(
        device,
        "clone_and_one_byte_view",
        Some((
            view.metal_buffer(),
            view.data_byte_len(),
            view.byte_offset(),
        )),
    );
    let raw = view.metal_buffer().clone();
    let retained_offset = view.byte_offset();
    let binding = RecordedBinding::Buffer {
        metal_buffer: raw.clone(),
        offset: view.byte_offset(),
    };
    assert_eq!(
        view.byte_len(),
        buffer.byte_len(),
        "view retains whole root"
    );
    drop(view);
    drop(cloned);
    drop(buffer);
    flush(device);
    if let Some(escrow) = observation.escrow {
        assert!(Arc::strong_count(escrow) > 1);
    }
    observation.sample(
        device,
        "raw_and_recorded_only",
        Some((&raw, 1, retained_offset)),
    );
    drop(raw);
    if let RecordedBinding::Buffer {
        metal_buffer,
        offset,
    } = &binding
    {
        observation.sample(
            device,
            "recorded_final_owner",
            Some((metal_buffer, 1, *offset)),
        );
    } else {
        unreachable!("constructed Buffer binding");
    }
    drop(binding);
    flush(device);
    // Never query the released resource via a saved pointer. Only the live
    // device and escrow are observed after the final explicit owner is gone.
    observation.sample(device, "final_owner_released_before_pool_drain", None);
}

fn anonymous(mode: Mode, logical: usize, page: usize) {
    assert!(logical.div_ceil(page) <= 2);
    assert!(logical < REQUEST_LIMIT);
    objc::rc::autoreleasepool(|| {
        let device = mode.device();
        let _flush_on_unwind = FlushOnDrop(&device);
        let observation = Observation {
            mode,
            kind: "anonymous",
            requested: logical,
            page,
            baseline: current(&device),
            escrow: None,
        };
        observation.sample(&device, "baseline", None);
        objc::rc::autoreleasepool(|| {
            let buffer = device
                .alloc_buffer(logical, DType::U8, vec![logical])
                .expect("tiny anonymous root");
            owners(&device, buffer, &observation);
        });
        flush(&device);
        observation.sample(&device, "after_resource_autorelease_scope", None);
    });
}

fn mapped(mode: Mode, page: usize) {
    let fixture = Fixture::new(page);
    // Before all resource/device/autorelease owners: panic cleanup cannot
    // invalidate a retained no-copy address. No payload is ever read.
    let escrow = fixture.mapping();
    let weak = Arc::downgrade(&escrow);
    objc::rc::autoreleasepool(|| {
        let device = mode.device();
        let _flush_on_unwind = FlushOnDrop(&device);
        let observation = Observation {
            mode,
            kind: "mapped",
            requested: MAPPED_LOGICAL,
            page,
            baseline: current(&device),
            escrow: Some(&escrow),
        };
        observation.sample(&device, "baseline", None);
        objc::rc::autoreleasepool(|| {
            let buffer = device
                .map_file_buffer(
                    Arc::clone(&escrow),
                    OFFSET,
                    MAPPED_LOGICAL,
                    DType::U8,
                    vec![MAPPED_LOGICAL],
                )
                .expect("tiny mapped root");
            owners(&device, buffer, &observation);
        });
        flush(&device);
        observation.sample(&device, "after_resource_autorelease_scope", None);
    });
    assert_eq!(
        Arc::strong_count(&escrow),
        1,
        "bounded explicit cleanup must release mapping owner"
    );
    drop(escrow);
    assert!(weak.upgrade().is_none());
    eprintln!("Q5_ALLOCATION_MAPPING_TEARDOWN {{\"mode\":\"{}\",\"weak_expired\":true,\"runtime_admission\":false}}", mode.name());
}

#[test]
#[ignore = "experimental raw metrics; explicitly run serially; not a memory upper-bound proof"]
fn bounded_raw_allocation_observations() {
    let page = host_page_size().expect("host page size");
    // This bounds requested logical/page-rounded roots, not opaque actual
    // allocator/driver memory. One explicit test root is retained per case;
    // opaque driver retention may overlap. The three anonymous requests plus
    // one mapping per mode sum to <=6 host pages;
    // across both modes the requested rounded total remains <1 MiB too.
    assert!(page <= 64 * 1024 && 12 * page < REQUEST_LIMIT);
    for mode in [Mode::Disabled, Mode::Default] {
        for logical in [1, 4096, page + 1] {
            anonymous(mode, logical, page);
        }
        mapped(mode, page);
    }
}
