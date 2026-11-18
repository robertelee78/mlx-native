//! Private no-copy source fixture for the consumer-facing completed-owner tests.
//!
//! The native mapping constructor is crate-private. Import through the public
//! MlxBuffer::from_raw boundary, with the same Arc<Mmap>-capturing Metal
//! deallocator ownership pattern as device::mapped_backing_tests. No source
//! text, retainCount, allocator counters, or model files serve as witnesses.

use memmap2::{Mmap, MmapOptions};
use mlx_native::{DType, GgmlQuantizedMatmulIdParams, GgmlType, MlxBuffer, MlxBufferPool, MlxDevice};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

const OFFSET: usize = 64;
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
pub struct Spec {
    pub ty: GgmlType,
    pub n: u32,
    pub k: u32,
    pub seed: u32,
}

impl Spec {
    pub fn params(self) -> GgmlQuantizedMatmulIdParams {
        GgmlQuantizedMatmulIdParams {
            n_tokens: 128,
            top_k: 8,
            n: self.n,
            k: self.k,
            n_experts: 8,
            // Exercise real inter-expert padding, not just packed stacks.
            expert_stride: self.matrix_bytes() as u64 + 32,
            ggml_type: self.ty,
        }
    }

    fn matrix_bytes(self) -> usize {
        self.n as usize * (self.k / 256) as usize * self.ty.block_bytes() as usize
    }

    pub fn weight(self, expert: u32, row: u32) -> f32 {
        (1 + (self.seed * 3 + expert * 5 + row) % 13) as f32
    }
}

struct PrivateFile {
    file: File,
    path: PathBuf,
}

impl Drop for PrivateFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// This arena owns ONLY these tests' source registrations and source queue.
/// Its registration pool drops first, removing and committing its memberships;
/// then its originating device/queue ends. It never flushes another device's
/// pending additions. Completion of the preparation CB is not this retirement.
pub struct SourceArena {
    registrations: MlxBufferPool,
    device: MlxDevice,
}

impl SourceArena {
    pub fn new(consumer: &MlxDevice) -> Self {
        let device = MlxDevice::new().expect("dedicated source device/queue");
        assert_eq!(device.registry_id(), consumer.registry_id());
        eprintln!("source residency enabled={}", device.residency_sets_enabled());
        Self { registrations: MlxBufferPool::new(), device }
    }

    pub fn source(&mut self, spec: Spec) -> (MlxBuffer, Weak<Mmap>) {
        let params = spec.params();
        let logical_bytes = (params.n_experts - 1) as usize * params.expert_stride as usize
            + spec.matrix_bytes();
        extern "C" {
            fn getpagesize() -> std::ffi::c_int;
        }
        // SAFETY: getpagesize is a no-argument macOS query, as in device.rs.
        let page = unsafe { getpagesize() };
        assert!(page > 0);
        let physical_bytes = (OFFSET + logical_bytes).div_ceil(page as usize) * page as usize;
        let path = std::env::temp_dir().join(format!(
            "mlx-completed-expert-{}-{}.bin",
            std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let fixture = PrivateFile {
            file: OpenOptions::new().read(true).write(true).create_new(true)
                .open(&path).expect("private mapped expert fixture"),
            path,
        };
        fixture.file.set_len(physical_bytes as u64).expect("size mapped fixture");
        // SAFETY: this uniquely created file is already sized. No other code
        // opens, modifies or truncates it; mutation ends before Metal sees it.
        let mut mapping = unsafe { MmapOptions::new().len(physical_bytes).map_mut(&fixture.file) }
            .expect("page-aligned expert mapping");
        mapping.fill(0xa7);
        let block_bytes = spec.ty.block_bytes() as usize;
        for expert in 0..params.n_experts {
            for row in 0..spec.n {
                let q = spec.weight(expert, row) as u8;
                for block in 0..spec.k / 256 {
                    let start = OFFSET + expert as usize * params.expert_stride as usize
                        + (row as usize * (spec.k / 256) as usize + block as usize) * block_bytes;
                    let bytes = &mut mapping[start..start + block_bytes];
                    bytes.fill(0);
                    match spec.ty {
                        GgmlType::Q4_K => {
                            // d=1, dmin=0, all eight scales=1, all quants=q.
                            bytes[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                            bytes[4..16].fill(1);
                            bytes[16..144].fill(q | (q << 4));
                        }
                        GgmlType::Q6_K => {
                            // Signed six-bit q: low=q, high=2 => (q+32)-32.
                            bytes[..128].fill(q | (q << 4));
                            bytes[128..192].fill(0xaa);
                            bytes[192..208].fill(1);
                            bytes[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
                        }
                        _ => unreachable!("fixture only constructs Q4_K/Q6_K"),
                    }
                }
            }
        }
        let backing = Arc::new(mapping.make_read_only().expect("freeze expert fixture"));
        let weak = Arc::downgrade(&backing);
        let pointer = backing.as_ptr().cast();
        // The block capture, not a hidden fixture escrow, owns the mapping.
        // Disposal follows the last raw Metal owner, including retained CBs.
        let callback = move |_pointer: *const std::ffi::c_void, _length: metal::NSUInteger| {
            let _keep_backing_alive = &backing;
        };
        fn require_send_sync<T: Send + Sync>(_: &T) {}
        require_send_sync(&callback);
        let deallocator = block::ConcreteBlock::new(callback).copy();
        let metal = self.device.metal_device().new_buffer_with_bytes_no_copy(
            pointer, physical_bytes as metal::NSUInteger,
            metal::MTLResourceOptions::StorageModeShared, Some(&deallocator),
        );
        let root = MlxBuffer::from_raw(metal, DType::U8, vec![physical_bytes]);
        let source = root.slice_view(OFFSET as u64, logical_bytes);
        self.registrations.register_existing(&self.device, &source)
            .expect("register only the dedicated source arena's mapped resource");
        (source, weak)
    }
}

pub fn assert_released(witnesses: &[Weak<Mmap>]) {
    for (index, witness) in witnesses.iter().enumerate() {
        assert!(witness.upgrade().is_none(), "mapped source {index} still owned after source retirement");
    }
}
