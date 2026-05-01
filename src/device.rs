//! [`MlxDevice`] — Metal device and command queue wrapper.
//!
//! This is the entry-point for all GPU work.  Create one with
//! [`MlxDevice::new()`] and use it to allocate buffers and create
//! command encoders.

use metal::{CommandQueue, Device, MTLResourceOptions};

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::residency::{macos_15_or_newer, residency_disabled_by_env, ResidencySet};

/// Wraps a Metal device and its command queue.
///
/// # Thread Safety
///
/// `MlxDevice` is `Send + Sync` — you can share it across threads. The
/// underlying Metal device and command queue are thread-safe on Apple Silicon.
pub struct MlxDevice {
    device: Device,
    queue: CommandQueue,
    residency_set: Option<ResidencySet>,
}

// metal::Device and metal::CommandQueue are both Send + Sync.
crate::static_assertions_send_sync!(MlxDevice);

impl MlxDevice {
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
            let set = ResidencySet::new(&device)?;
            if set.is_noop() {
                if log_init {
                    eprintln!("[mlx-native] residency sets = false (reason: macOS < 15.0)");
                }
                None
            } else {
                set.register_with_queue(&queue);
                if log_init {
                    eprintln!("[mlx-native] residency sets = true");
                }
                Some(set)
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
        let metal_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        // Metal returns a non-null buffer on success; a null pointer indicates
        // failure (typically out-of-memory).
        if metal_buf.contents().is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }
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
        // Per `feedback_no_broken_windows` + mantra "No fallback. No stub.
        // Just pure excellence." — fix at the source.
        //
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

impl std::fmt::Debug for MlxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxDevice")
            .field("name", &self.device.name())
            .finish()
    }
}
