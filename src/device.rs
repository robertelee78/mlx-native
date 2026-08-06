//! [`MlxDevice`] — Metal device and command queue wrapper.
//!
//! This is the entry-point for all GPU work.  Create one with
//! [`MlxDevice::new()`] and use it to allocate buffers and create
//! command encoders.

use std::sync::Arc;

use memmap2::Mmap;
use metal::foreign_types::ForeignType;
use metal::{CommandQueue, Device, MTLResourceOptions};
use objc::{sel, sel_impl};

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::encoder_session::EncoderSession;
use crate::error::{MlxError, Result};
use crate::residency::{macos_15_or_newer, residency_disabled_by_env, ResidencySet};

/// Wraps a Metal device and its command queue.
///
/// # Thread Safety
///
/// `MlxDevice` is `Send + Sync` — you can share it across threads. The
/// underlying Metal device and command queue are thread-safe on Apple Silicon.
///
/// `Clone` is a cheap Arc-bump on the underlying handles: `metal::Device`
/// + `metal::CommandQueue` wrap NSObject Arc-pointers internally, and
/// [`ResidencySet`] is `#[derive(Clone)]` over an `Arc<ResidencySetInner>`.
/// Cloning yields a SECOND handle pointing at the SAME GPU device, command
/// queue, and residency-set NSObject — multiple owners (e.g. an
/// `AdamOptimizer` + a per-step `GpuTape`) can register allocations
/// against the same residency set without double-create.  ADR-020
/// iter-13b dependency.
#[derive(Clone)]
pub struct MlxDevice {
    device: Device,
    queue: CommandQueue,
    residency_set: Option<ResidencySet>,
}

// metal::Device and metal::CommandQueue are both Send + Sync.
crate::static_assertions_send_sync!(MlxDevice);

impl MlxDevice {
    /// Allocate an owned shared-memory Metal buffer without letting
    /// `metal-rs` construct a `Buffer` from a null Objective-C pointer.
    ///
    /// `DeviceRef::new_buffer` returns `Buffer` directly, so an allocation
    /// failure reaches `ForeignType::from_ptr` first and aborts on its
    /// non-null assertion. Query the hard per-buffer limit up front, then
    /// request the raw Objective-C object and classify `nil` as the typed
    /// allocation error promised by mlx-native's public API.
    pub(crate) fn new_shared_buffer(&self, byte_len: usize) -> Result<metal::Buffer> {
        if byte_len == 0 || byte_len as u64 > self.device.max_buffer_length() as u64 {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }

        let raw: *mut metal::MTLBuffer = unsafe {
            objc::msg_send![
                &*self.device,
                newBufferWithLength: byte_len as u64
                options: MTLResourceOptions::StorageModeShared
            ]
        };
        if raw.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }

        // SAFETY: `newBufferWithLength:options:` returned a non-null owned
        // (+1 retain-count) MTLBuffer. The wrapper assumes that ownership and
        // releases it exactly once on drop.
        Ok(unsafe { metal::Buffer::from_ptr(raw) })
    }

    /// Create a read-only Metal view over bytes in a file mapping.
    ///
    /// Metal requires the host pointer and resource length to be page-aligned.
    /// GGUF tensor offsets are normally only 32-byte aligned, so the resource
    /// starts at the preceding page and the returned [`MlxBuffer`] carries the
    /// remaining offset for kernel bindings.
    pub(crate) fn map_file_buffer(
        &self,
        file_backing: Arc<Mmap>,
        file_offset: usize,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        if byte_len == 0 {
            return Err(MlxError::InvalidArgument(
                "File-backed buffer byte length must be > 0".into(),
            ));
        }
        let logical_end = file_offset
            .checked_add(byte_len)
            .ok_or_else(|| MlxError::InvalidArgument("File-backed buffer range overflow".into()))?;
        if logical_end > file_backing.len() {
            return Err(MlxError::InvalidArgument(format!(
                "File-backed buffer range [{file_offset}, {logical_end}) exceeds mapping length {}",
                file_backing.len()
            )));
        }

        let page_size = host_page_size()?;
        let aligned_start = file_offset / page_size * page_size;
        let aligned_end = logical_end
            .checked_add(page_size - 1)
            .ok_or_else(|| MlxError::InvalidArgument("Mapped range alignment overflow".into()))?
            / page_size
            * page_size;
        let mapped_len = aligned_end - aligned_start;
        if mapped_len as u64 > self.device.max_buffer_length() as u64 {
            return Err(MlxError::BufferAllocationError { bytes: mapped_len });
        }

        // SAFETY: `aligned_start` is within the mapping and page-aligned. The
        // mmap owns the final partial page even when its public slice length
        // ends before `aligned_end`; kernels are constrained to `byte_len`.
        let mapped_ptr = unsafe { file_backing.as_ptr().add(aligned_start) };
        if (mapped_ptr as usize) % page_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Mapped file pointer is not aligned to the host page size {page_size}"
            )));
        }

        let deallocator: *mut objc::runtime::Object = std::ptr::null_mut();
        let raw: *mut metal::MTLBuffer = unsafe {
            objc::msg_send![
                &*self.device,
                newBufferWithBytesNoCopy: mapped_ptr as *const std::ffi::c_void
                length: mapped_len as u64
                options: MTLResourceOptions::StorageModeShared
                deallocator: deallocator
            ]
        };
        if raw.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: mapped_len });
        }

        // SAFETY: the Objective-C initializer returned a non-null owned Metal
        // buffer. `MlxBufferStorage` releases it before dropping `file_backing`.
        let metal_buf = unsafe { metal::Buffer::from_ptr(raw) };
        Ok(MlxBuffer::from_file_mapping(
            metal_buf,
            dtype,
            shape,
            (file_offset - aligned_start) as u64,
            byte_len,
            file_backing,
            self.residency_set.clone(),
        ))
    }

    /// Initialize the Metal GPU device and create a command queue.
    ///
    /// Returns `Err(MlxError::DeviceNotFound)` if no Metal device is available
    /// (e.g. running on a non-Apple-Silicon machine or in a headless Linux VM).
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MlxError::DeviceNotFound)?;
        let queue = device.new_command_queue();
        let log_init = std::env::var("MLX_NATIVE_LOG_INIT").as_deref() == Ok("1");

        let residency_set = if residency_disabled_by_env() {
            if log_init {
                eprintln!("[mlx-native] residency sets = false (reason: HF2Q_NO_RESIDENCY=1)");
            }
            None
        } else if !macos_15_or_newer() {
            if log_init {
                eprintln!("[mlx-native] residency sets = false (reason: macOS < 15.0)");
            }
            None
        } else {
            match ResidencySet::new(&device) {
                Ok(set) if !set.is_noop() => {
                    set.register_with_queue(&queue);
                    if log_init {
                        eprintln!("[mlx-native] residency sets = true");
                    }
                    Some(set)
                }
                Ok(_) => {
                    if log_init {
                        eprintln!("[mlx-native] residency sets = false (reason: unsupported)");
                    }
                    None
                }
                Err(error) => {
                    // macOS reports the API as available on some virtualized
                    // Apple-Silicon hosts but rejects residency-set creation.
                    // Residency is an optimization, not a correctness
                    // requirement, so retain the ordinary Metal path.
                    if log_init {
                        eprintln!("[mlx-native] residency sets = false (reason: {error})");
                    }
                    None
                }
            }
        };

        Ok(Self {
            device,
            queue,
            residency_set,
        })
    }

    /// Create a [`CommandEncoder`] for batching GPU dispatches.
    ///
    /// The encoder wraps a fresh Metal command buffer from the device's command
    /// queue.  Encode one or more kernel dispatches, then call
    /// [`CommandEncoder::commit_and_wait`] to submit and block until completion.
    ///
    /// ADR-015 iter8e (Phase 3b): the encoder is bound to the device's
    /// residency set so every `commit*` boundary flushes deferred
    /// add/remove staging (one `[set commit]` per CB submission instead
    /// of per-allocation). When residency sets are disabled
    /// (HF2Q_NO_RESIDENCY=1, macOS<15) the binding is `None` and the
    /// flush is a no-op.
    pub fn command_encoder(&self) -> Result<CommandEncoder> {
        CommandEncoder::new_with_residency(&self.queue, self.residency_set.clone())
    }

    /// Create an [`EncoderSession`] (ADR-019 Phase 0b iter89e2-A — bare
    /// struct) for one transformer stage's worth of GPU work.
    ///
    /// Gated on `HF2Q_ENCODER_SESSION=1` (default OFF). When the gate is
    /// unset, returns `Ok(None)` so callers can fall back to
    /// [`Self::command_encoder`] without an extra conditional. When set,
    /// returns `Ok(Some(EncoderSession))` carrying a fresh
    /// [`CommandEncoder`] — same construction path as `command_encoder()`,
    /// just wrapped in the session shell.
    ///
    /// In iter89e2-A no production code path consumes this method; it
    /// exists so the env-gate has a callable factory and the lifecycle
    /// tests have a public entry point. Phase 1+ migrations
    /// (`forward_gpu.rs`, `gpu_full_attn.rs`, `gpu_delta_net.rs`) opt in
    /// per-call site.
    ///
    /// # Errors
    ///
    /// Surfaces any error from the underlying `EncoderSession::new`
    /// — currently infallible past metal-rs's `new_command_buffer`,
    /// preserved for future-proofing.
    pub fn encoder_session(&self) -> Result<Option<EncoderSession>> {
        if !EncoderSession::env_enabled() {
            return Ok(None);
        }
        EncoderSession::new(&self.device, &self.queue, self.residency_set.clone()).map(Some)
    }

    /// Allocate a new GPU buffer with `StorageModeShared`.
    ///
    /// # Arguments
    ///
    /// * `byte_len` — Size of the buffer in bytes.  Must be > 0.
    /// * `dtype`    — Element data type for metadata tracking.
    /// * `shape`    — Tensor dimensions for metadata tracking.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::InvalidArgument` if `byte_len` is zero.
    /// Returns `MlxError::BufferAllocationError` if Metal cannot allocate.
    pub fn alloc_buffer(
        &self,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        if byte_len == 0 {
            return Err(MlxError::InvalidArgument(
                "Buffer byte length must be > 0".into(),
            ));
        }
        let metal_buf = self.new_shared_buffer(byte_len)?;
        // ADR-015 iter61a (broken-window B-W-1 fix): explicitly zero every
        // newly-allocated GPU buffer. `MTLResourceOptions::StorageModeShared`
        // does NOT guarantee zeroed pages on Apple Silicon — Metal's allocator
        // recycles pages from recently-freed allocations within the device's
        // private heap before the OS sees the free, so a fresh buffer can
        // contain residual bytes from prior allocations in the same process.
        // In a cold process this surfaces as run-to-run non-determinism: the
        // heap state at the moment Metal services `newBufferWithLength`
        // differs across cold invocations, and any kernel that reads a buffer
        // before fully populating it (e.g. DeltaNet's `ssm_conv` reads
        // conv_state, MoE expert routing reads scratch, attn-output buffers
        // before the final write barrier) propagates that garbage into
        // logits → argmax → divergent generations across cold runs.
        // The cost is one memset per allocation; on workloads dominated by
        // weight-load (one-time) and kvcache (one-time), this is negligible.
        // Safety: `metal_buf.contents()` is non-null (verified above), points
        // to exactly `byte_len` bytes of `StorageModeShared` memory we just
        // allocated and have exclusive access to (no other thread or GPU
        // dispatch references it yet — we haven't returned the MlxBuffer
        // wrapper yet, and the underlying CB queue is not in flight on this
        // allocation). Writing zero bytes is well-defined for any DType.
        unsafe {
            std::ptr::write_bytes(metal_buf.contents() as *mut u8, 0, byte_len);
        }
        // ADR-015 iter8e (Phase 3b): auto-register the new allocation with the
        // device's residency set so it gets the MTLResidencySet hint on the
        // next dispatch. The `with_residency` path stages the addAllocation
        // but DEFERS the `[set commit]` to the next CommandEncoder::commit*
        // boundary via flush_pending — mirrors llama.cpp's batch-add /
        // single-commit pattern in ggml-metal-device.m:1378-1382.
        //
        // No-op when residency_set is None (HF2Q_NO_RESIDENCY=1, macOS<15,
        // or no Metal device).
        match self.residency_set.as_ref() {
            Some(set) => Ok(MlxBuffer::with_residency(
                metal_buf,
                dtype,
                shape,
                set.clone(),
            )),
            None => Ok(MlxBuffer::from_raw(metal_buf, dtype, shape)),
        }
    }

    /// Borrow the underlying `metal::Device` for direct Metal API calls
    /// (e.g. kernel compilation in [`KernelRegistry`](crate::KernelRegistry)).
    #[inline]
    pub fn metal_device(&self) -> &metal::DeviceRef {
        &self.device
    }

    /// Borrow the underlying `metal::CommandQueue`.
    #[inline]
    pub fn metal_queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Borrow the device-level residency set, if residency support is enabled.
    #[inline]
    pub(crate) fn residency_set(&self) -> Option<&ResidencySet> {
        self.residency_set.as_ref()
    }

    /// Return whether this device has an active Metal residency set.
    #[inline]
    pub fn residency_sets_enabled(&self) -> bool {
        self.residency_set.is_some()
    }

    /// Human-readable name of the GPU (e.g. "Apple M2 Max").
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }
}

fn host_page_size() -> Result<usize> {
    extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }

    // SAFETY: `getpagesize` has no arguments or side effects and is available
    // on every macOS version supported by mlx-native.
    let size = unsafe { getpagesize() };
    if size <= 0 {
        return Err(MlxError::InvalidArgument(
            "Operating system reported an invalid page size".into(),
        ));
    }
    Ok(size as usize)
}

impl std::fmt::Debug for MlxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxDevice")
            .field("name", &self.device.name())
            .finish()
    }
}
