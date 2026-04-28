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
    pub fn command_encoder(&self) -> Result<CommandEncoder> {
        CommandEncoder::new(&self.queue)
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
        Ok(MlxBuffer::from_raw(metal_buf, dtype, shape))
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
