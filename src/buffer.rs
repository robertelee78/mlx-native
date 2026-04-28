//! [`MlxBuffer`] — typed wrapper around a Metal GPU buffer.
//!
//! Buffers are allocated with `StorageModeShared` so that CPU and GPU share
//! the same physical memory on Apple Silicon (zero-copy access via
//! [`as_slice`](MlxBuffer::as_slice) / [`as_mut_slice`](MlxBuffer::as_mut_slice)).

use std::fmt;
use std::sync::Arc;

use metal::Buffer as MetalBuffer;

use crate::dtypes::DType;
use crate::error::{MlxError, Result};
use crate::residency::ResidencySet;

/// A Metal GPU buffer annotated with element dtype and tensor shape.
///
/// On Apple Silicon the underlying memory is unified — `contents_ptr()` gives
/// direct CPU access without any copy or transfer.
///
/// # Thread Safety
///
/// `MlxBuffer` is `Send + Sync` because the inner `metal::Buffer` is.
///
/// # Residency-set lifecycle
///
/// Buffers produced by [`MlxDevice::alloc_buffer`](crate::MlxDevice::alloc_buffer)
/// on a residency-enabled device carry a shared
/// [`Arc<MlxBufferStorage>`](MlxBufferStorage) that owns the residency-set
/// reference and runs `removeAllocation:` (deferred — flushed at the next
/// `CommandEncoder::commit*` boundary) when the last clone is dropped.
/// Mirrors llama.cpp's `ggml-metal-device.m:1378-1382` pattern: batch
/// `addAllocation:` calls in a loop, commit ONCE.
pub struct MlxBuffer {
    /// The underlying Metal buffer (StorageModeShared) plus optional
    /// residency-set membership guard.
    storage: Arc<MlxBufferStorage>,
    /// Element data type.
    dtype: DType,
    /// Tensor shape (e.g. `[2, 3, 4]` for a rank-3 tensor).
    shape: Vec<usize>,
    /// Byte offset into the underlying Metal buffer (for slice views).
    /// Zero for normally-allocated buffers.
    byte_offset: u64,
}

/// Owns a single Metal buffer allocation plus an optional residency-set
/// membership guard.
///
/// Wrapped in [`Arc`] inside [`MlxBuffer`] so that [`Clone`] / [`slice_view`]
/// share both the underlying Metal allocation and the residency-set
/// registration. The Drop fires `removeAllocation:` only when the LAST clone
/// goes out of scope — matching llama.cpp's `addAllocation:` /
/// `removeAllocation:` lifecycle in `ggml-metal-device.m:1378-1382` and
/// `ggml-metal-device.m:1397-1399`.
///
/// Drop is **deferred**: it calls `set.remove_allocation(buffer)` which marks
/// the residency set's pending flag but does NOT call `[set commit]`. The
/// commit is flushed at the next [`CommandEncoder::commit*`] boundary via
/// [`ResidencySet::flush_pending`]. This collapses the per-allocation commit
/// storm (~880 commits/decode-token in iter8d/8e claude+codex variants) into
/// at most one commit per CB submission.
pub(crate) struct MlxBufferStorage {
    inner: MetalBuffer,
    residency_set: Option<ResidencySet>,
}

impl Drop for MlxBufferStorage {
    fn drop(&mut self) {
        if let Some(set) = self.residency_set.as_ref() {
            // Mirror ggml-metal-device.m:1397-1399 free-path semantics, but
            // deferred — the actual `[set commit]` is issued at the next
            // CommandEncoder::commit* boundary by flush_pending().
            set.remove_allocation(&self.inner);
        }
    }
}

// metal::Buffer is Send + Sync; our extra fields (DType, Vec<usize>) are too.
crate::static_assertions_send_sync!(MlxBuffer);

impl Clone for MlxBuffer {
    /// Increment the storage's `Arc` ref-count and wrap it in a new
    /// `MlxBuffer`. Both the original and the clone refer to the same
    /// underlying GPU allocation AND share the residency-set membership
    /// guard — no data is copied, no double-registration occurs.
    ///
    /// This is safe because `metal::Buffer` wraps an `MTLBuffer` Objective-C
    /// object whose lifetime is managed by ARC; `Arc::clone` increments the
    /// Rust-side refcount, and the inner `MlxBufferStorage` Drop runs once
    /// when the last clone is released.
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            byte_offset: self.byte_offset,
        }
    }
}

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
    ///
    /// The returned buffer carries no residency-set guard — pool / external
    /// callers that want residency tracking should go through
    /// [`MlxDevice::alloc_buffer`](crate::MlxDevice::alloc_buffer) or
    /// [`MlxBufferPool::register_existing`](crate::MlxBufferPool::register_existing).
    pub fn from_raw(inner: MetalBuffer, dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            storage: Arc::new(MlxBufferStorage {
                inner,
                residency_set: None,
            }),
            dtype,
            shape,
            byte_offset: 0,
        }
    }

    /// Create a new buffer and stage its Metal allocation for inclusion in
    /// the given residency set.
    ///
    /// Calls `set.add_allocation(buffer)` (deferred — no `[set commit]` until
    /// the next [`flush_pending`](ResidencySet::flush_pending) at a
    /// `CommandEncoder::commit*` boundary). The buffer's residency-set guard
    /// is dropped when the last clone of the returned `MlxBuffer` (and any
    /// slice views) goes out of scope, which fires the matching
    /// `removeAllocation:` (also deferred).
    ///
    /// Crate-private — external callers should go through
    /// [`MlxDevice::alloc_buffer`](crate::MlxDevice::alloc_buffer).
    pub(crate) fn with_residency(
        inner: MetalBuffer,
        dtype: DType,
        shape: Vec<usize>,
        residency_set: ResidencySet,
    ) -> Self {
        // Stage the addAllocation; the actual `[set commit]` is deferred to
        // the next encoder.commit* boundary via flush_pending. This is the
        // structural fix for the per-allocation commit storm; mirrors
        // llama.cpp's ggml-metal-device.m:1378-1382 pattern.
        residency_set.add_allocation(&inner);

        Self {
            storage: Arc::new(MlxBufferStorage {
                inner,
                residency_set: Some(residency_set),
            }),
            dtype,
            shape,
            byte_offset: 0,
        }
    }

    /// Create a zero-copy slice view of this buffer.
    ///
    /// Returns a new `MlxBuffer` that shares the same underlying Metal buffer
    /// but starts at `byte_offset` bytes from the beginning and contains
    /// `n_elements` elements of type `dtype`. No data is copied.
    ///
    /// The slice view shares the parent's residency-set guard via the
    /// `Arc<MlxBufferStorage>`, so it does NOT trigger a second
    /// `addAllocation:` and does NOT deregister the parent on drop.
    ///
    /// When this view is bound to a kernel, the encoder passes the byte offset
    /// to Metal's `setBuffer:offset:atIndex:`, so the kernel sees only the
    /// slice region.
    ///
    /// # Panics
    ///
    /// Panics if `byte_offset + n_elements * dtype.size_of() > self.inner.length()`.
    #[inline]
    pub fn slice_view(&self, byte_offset: u64, n_elements: usize) -> Self {
        let end = byte_offset as usize + n_elements * self.dtype.size_of();
        assert!(
            end <= self.storage.inner.length() as usize,
            "slice_view: out of bounds (byte_offset={}, n_elements={}, dtype_size={}, buf_len={})",
            byte_offset,
            n_elements,
            self.dtype.size_of(),
            self.storage.inner.length()
        );
        Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape: vec![n_elements],
            byte_offset,
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
        self.storage.inner.length() as usize
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
        self.storage.inner.contents()
    }

    /// Reference to the underlying `metal::Buffer` for passing to the encoder.
    #[inline]
    pub fn metal_buffer(&self) -> &MetalBuffer {
        &self.storage.inner
    }

    /// Byte offset into the underlying Metal buffer (zero for non-slice buffers).
    ///
    /// When passing this buffer to a Metal kernel via `setBuffer:offset:atIndex:`,
    /// use this offset so the kernel sees only the intended sub-region.
    #[inline]
    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Consume self and return the inner `metal::Buffer` (used by buffer pool).
    ///
    /// If this is the last clone of the underlying `Arc<MlxBufferStorage>`,
    /// the storage Drop fires after this returns — staging a deferred
    /// `removeAllocation:` if the buffer carried a residency-set guard.
    /// Pool-internal buffers do not carry guards, so this is a no-op for
    /// the pool's `release` path.
    #[inline]
    pub(crate) fn into_inner(self) -> MetalBuffer {
        self.storage.inner.clone()
    }

    /// Borrow the residency set that this buffer was registered with, if any.
    ///
    /// Used by [`MlxBufferPool::register_existing`](crate::MlxBufferPool::register_existing)
    /// to short-circuit re-registration: a buffer created via
    /// [`MlxDevice::alloc_buffer`](crate::MlxDevice::alloc_buffer) on a
    /// residency-enabled device already owns its registration via the
    /// `Arc<MlxBufferStorage>`, so the pool path is a no-op (modulo
    /// validation that the device matches).
    #[inline]
    pub(crate) fn residency_set(&self) -> Option<&ResidencySet> {
        self.storage.residency_set.as_ref()
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
