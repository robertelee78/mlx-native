//! [`MlxBuffer`] — typed wrapper around a Metal GPU buffer.
//!
//! Buffers are allocated with `StorageModeShared` so that CPU and GPU share
//! the same physical memory on Apple Silicon (zero-copy access via
//! [`as_slice`](MlxBuffer::as_slice) / [`as_mut_slice`](MlxBuffer::as_mut_slice)).

use std::fmt;

use metal::Buffer as MetalBuffer;

use crate::dtypes::DType;
use crate::error::{MlxError, Result};

/// A Metal GPU buffer annotated with element dtype and tensor shape.
///
/// On Apple Silicon the underlying memory is unified — `contents_ptr()` gives
/// direct CPU access without any copy or transfer.
///
/// # Thread Safety
///
/// `MlxBuffer` is `Send + Sync` because the inner `metal::Buffer` is.
pub struct MlxBuffer {
    /// The underlying Metal buffer (StorageModeShared).
    inner: MetalBuffer,
    /// Element data type.
    dtype: DType,
    /// Tensor shape (e.g. `[2, 3, 4]` for a rank-3 tensor).
    shape: Vec<usize>,
}

// metal::Buffer is Send + Sync; our extra fields (DType, Vec<usize>) are too.
crate::static_assertions_send_sync!(MlxBuffer);

impl MlxBuffer {
    /// Create a new `MlxBuffer` wrapping an already-allocated Metal buffer.
    ///
    /// # When to use
    ///
    /// Use this to wrap Metal buffers obtained from external frameworks (e.g.
    /// candle's `MetalStorage::buffer()`) for zero-copy interop on Apple
    /// Silicon unified memory.  Both frameworks see the same physical memory.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure that `inner` remains valid for the lifetime of
    /// the returned `MlxBuffer`.  If the buffer was obtained from another
    /// framework, the caller must ensure that framework does not deallocate
    /// the buffer while this `MlxBuffer` exists.
    pub fn from_raw(inner: MetalBuffer, dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            inner,
            dtype,
            shape,
        }
    }

    // ---- accessors ----

    /// Element data type.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Tensor shape (dimensions).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total byte length of the Metal buffer.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.inner.length() as usize
    }

    /// Number of elements (product of shape dimensions, or `byte_len / dtype.size_of()`).
    #[inline]
    pub fn element_count(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Raw pointer to the buffer contents (CPU-accessible on Apple Silicon).
    ///
    /// # Safety
    ///
    /// The caller must ensure proper synchronization — do not read while a GPU
    /// command buffer that writes this buffer is in flight.
    #[inline]
    pub fn contents_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.contents()
    }

    /// Reference to the underlying `metal::Buffer` for passing to the encoder.
    #[inline]
    pub fn metal_buffer(&self) -> &MetalBuffer {
        &self.inner
    }

    /// Consume self and return the inner `metal::Buffer` (used by buffer pool).
    #[inline]
    pub(crate) fn into_inner(self) -> MetalBuffer {
        self.inner
    }

    // ---- typed CPU access (zero-copy on unified memory) ----

    /// View the buffer contents as a typed slice.
    ///
    /// Returns an error if the buffer byte length is not an exact multiple of
    /// `size_of::<T>()`.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure:
    /// 1. `T` matches the actual element type stored in the buffer.
    /// 2. No GPU command buffer that writes this buffer is currently in flight.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(MlxError::InvalidArgument(
                "Cannot view buffer as zero-sized type".into(),
            ));
        }
        let byte_len = self.byte_len();
        if byte_len % elem_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer byte length {byte_len} is not a multiple of element size {elem_size}"
            )));
        }
        let ptr = self.contents_ptr();
        if ptr.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }
        let count = byte_len / elem_size;
        // SAFETY: Metal guarantees the pointer is valid for `byte_len` bytes and
        // properly aligned for any type on Apple Silicon shared memory.  The
        // caller upholds the type-match and no-concurrent-GPU-write contract.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, count) };
        Ok(slice)
    }

    /// View the buffer contents as a mutable typed slice.
    ///
    /// Same safety contract as [`as_slice`](Self::as_slice), plus: the caller
    /// must ensure exclusive access (no other references to this buffer's memory
    /// exist).
    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> Result<&mut [T]> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(MlxError::InvalidArgument(
                "Cannot view buffer as zero-sized type".into(),
            ));
        }
        let byte_len = self.byte_len();
        if byte_len % elem_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer byte length {byte_len} is not a multiple of element size {elem_size}"
            )));
        }
        let ptr = self.contents_ptr();
        if ptr.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }
        let count = byte_len / elem_size;
        // SAFETY: same as as_slice, plus caller ensures exclusive mutable access.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) };
        Ok(slice)
    }

    /// Overwrite the dtype and shape metadata.
    ///
    /// This does **not** re-allocate the Metal buffer — it only changes the
    /// logical interpretation.  The caller must ensure the new shape is
    /// consistent with the buffer's byte length.
    #[allow(dead_code)]
    pub(crate) fn reshape(&mut self, dtype: DType, shape: Vec<usize>) {
        self.dtype = dtype;
        self.shape = shape;
    }
}

impl fmt::Debug for MlxBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxBuffer")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("byte_len", &self.byte_len())
            .finish()
    }
}
