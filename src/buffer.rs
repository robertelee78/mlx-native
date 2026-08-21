//! [`MlxBuffer`] — typed wrapper around a Metal GPU buffer.
//!
//! Buffers are allocated with `StorageModeShared` so that CPU and GPU share
//! the same physical memory on Apple Silicon (zero-copy access via
//! [`as_slice`](MlxBuffer::as_slice) / [`as_mut_slice`](MlxBuffer::as_mut_slice)).

use std::fmt;
use std::sync::Arc;

use memmap2::Mmap;
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
/// Pattern: batch
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
    /// Number of bytes belonging to this logical buffer view. This differs
    /// from the page-rounded Metal allocation length for file-backed tensors.
    data_byte_len: usize,
}

/// Owns a single Metal buffer allocation plus an optional residency-set
/// membership guard.
///
/// Wrapped in [`Arc`] inside [`MlxBuffer`] so that [`Clone`] / [`slice_view`]
/// share both the underlying Metal allocation and the residency-set
/// registration. The Drop fires `removeAllocation:` only when the LAST clone
/// goes out of scope — matching the reference `addAllocation:` /
/// `removeAllocation:` lifecycle.
///
/// Drop is **deferred**: it calls `set.remove_allocation(buffer)` which marks
/// the residency set's pending flag but does NOT call `[set commit]`. The
/// commit is flushed at the next [`CommandEncoder::commit*`] boundary via
/// [`ResidencySet::flush_pending`]. This collapses the per-allocation commit
/// storm (~880 commits/decode-token in iter8d/8e) into
/// at most one commit per CB submission.
pub(crate) struct MlxBufferStorage {
    // Keep the Metal object before `file_backing`: Rust drops fields in
    // declaration order after `Drop::drop`, so Metal releases its no-copy
    // view before the mmap that supplies the bytes is unmapped.
    inner: MetalBuffer,
    residency_set: Option<ResidencySet>,
    file_backing: Option<Arc<Mmap>>,
    cpu_writable: bool,
}

impl Drop for MlxBufferStorage {
    fn drop(&mut self) {
        if let Some(set) = self.residency_set.as_ref() {
            // Mirror the reference free-path semantics, but
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
            data_byte_len: self.data_byte_len,
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
        let data_byte_len = inner.length() as usize;
        Self {
            storage: Arc::new(MlxBufferStorage {
                inner,
                residency_set: None,
                file_backing: None,
                cpu_writable: true,
            }),
            dtype,
            shape,
            byte_offset: 0,
            data_byte_len,
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
        // structural fix for the per-allocation commit storm; the
        // batch-add / single-commit pattern.
        residency_set.add_allocation(&inner);

        let data_byte_len = inner.length() as usize;
        Self {
            storage: Arc::new(MlxBufferStorage {
                inner,
                residency_set: Some(residency_set),
                file_backing: None,
                cpu_writable: true,
            }),
            dtype,
            shape,
            byte_offset: 0,
            data_byte_len,
        }
    }

    /// Wrap a read-only file mapping in a no-copy Metal buffer.
    ///
    /// `byte_offset` identifies the tensor's first byte inside the page-aligned
    /// Metal view. The mapping is retained by the shared storage so clones and
    /// slice views cannot outlive the file-backed bytes.
    pub(crate) fn from_file_mapping(
        inner: MetalBuffer,
        dtype: DType,
        shape: Vec<usize>,
        byte_offset: u64,
        data_byte_len: usize,
        file_backing: Arc<Mmap>,
        residency_set: Option<ResidencySet>,
    ) -> Self {
        if let Some(set) = residency_set.as_ref() {
            set.add_allocation(&inner);
        }

        Self {
            storage: Arc::new(MlxBufferStorage {
                inner,
                residency_set,
                file_backing: Some(file_backing),
                cpu_writable: false,
            }),
            dtype,
            shape,
            byte_offset,
            data_byte_len,
        }
    }

    /// Create a typed byte-range view into this buffer's logical data region.
    ///
    /// This is used by GGUF segment mappings: a small number of large Metal
    /// resources own the file mappings, while each tensor carries its own
    /// dtype, shape, offset, and packed byte length.
    pub(crate) fn data_view(
        &self,
        relative_byte_offset: usize,
        data_byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let relative_end = relative_byte_offset
            .checked_add(data_byte_len)
            .ok_or_else(|| MlxError::InvalidArgument("Buffer data view range overflow".into()))?;
        if relative_end > self.data_byte_len {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer data view [{relative_byte_offset}, {relative_end}) exceeds logical data length {}",
                self.data_byte_len
            )));
        }
        let byte_offset = self
            .byte_offset
            .checked_add(relative_byte_offset as u64)
            .ok_or_else(|| MlxError::InvalidArgument("Buffer data view offset overflow".into()))?;
        let physical_end = usize::try_from(byte_offset)
            .ok()
            .and_then(|offset| offset.checked_add(data_byte_len))
            .ok_or_else(|| MlxError::InvalidArgument("Buffer data view range overflow".into()))?;
        if physical_end > self.byte_len() {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer data view ends at {physical_end}, beyond Metal length {}",
                self.byte_len()
            )));
        }

        Ok(Self {
            storage: self.storage.clone(),
            dtype,
            shape,
            byte_offset,
            data_byte_len,
        })
    }

    /// Create a zero-copy slice view of this buffer.
    ///
    /// Returns a new `MlxBuffer` that shares the same underlying Metal buffer
    /// but starts at `byte_offset` bytes from this handle's logical data start
    /// and contains `n_elements` elements of type `dtype`. Nested views
    /// therefore compose their offsets. No data is copied.
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
    /// Panics if `byte_offset + n_elements * dtype.size_of()` exceeds this
    /// handle's logical data region.
    #[inline]
    pub fn slice_view(&self, byte_offset: u64, n_elements: usize) -> Self {
        let view_byte_len = n_elements
            .checked_mul(self.dtype.size_of())
            .expect("slice_view: byte length overflow");
        let relative_end = usize::try_from(byte_offset)
            .ok()
            .and_then(|offset| offset.checked_add(view_byte_len))
            .expect("slice_view: range overflow");
        assert!(
            relative_end <= self.data_byte_len,
            "slice_view: out of logical bounds (byte_offset={}, n_elements={}, dtype_size={}, data_len={})",
            byte_offset,
            n_elements,
            self.dtype.size_of(),
            self.data_byte_len
        );
        let absolute_byte_offset = self
            .byte_offset
            .checked_add(byte_offset)
            .expect("slice_view: absolute offset overflow");
        Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape: vec![n_elements],
            byte_offset: absolute_byte_offset,
            data_byte_len: view_byte_len,
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

    /// Byte length of the logical data region represented by this handle.
    ///
    /// Owned allocations normally match [`byte_len`](Self::byte_len).
    /// File-backed buffers exclude page-alignment prefix and suffix bytes.
    #[inline]
    pub fn data_byte_len(&self) -> usize {
        self.data_byte_len
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

    /// Whether the Metal buffer reads directly from a file mapping.
    #[inline]
    pub fn is_file_backed(&self) -> bool {
        self.storage.file_backing.is_some()
    }

    /// Whether typed CPU mutation is supported for this allocation.
    #[inline]
    pub fn is_cpu_writable(&self) -> bool {
        self.storage.cpu_writable
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

    /// View this handle's logical data region as a typed slice.
    ///
    /// This honors [`byte_offset`](Self::byte_offset) and
    /// [`data_byte_len`](Self::data_byte_len), so tensor and nested slice
    /// views never expose page-alignment bytes or adjacent tensors. Returns an
    /// error if the logical data length is not an exact multiple of
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
        let byte_len = self.data_byte_len;
        if byte_len % elem_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer byte length {byte_len} is not a multiple of element size {elem_size}"
            )));
        }
        let base = self.contents_ptr();
        if base.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }
        let ptr = unsafe { (base as *const u8).add(self.byte_offset as usize) };
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer data offset {} is not aligned for {}",
                self.byte_offset,
                std::any::type_name::<T>()
            )));
        }
        let count = byte_len / elem_size;
        // SAFETY: the handle's construction validates that byte_offset plus
        // data_byte_len remains within the Metal allocation. The alignment
        // check above and caller's type/synchronization contract cover T.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, count) };
        Ok(slice)
    }

    /// View the buffer contents as a mutable typed slice.
    ///
    /// Same safety contract as [`as_slice`](Self::as_slice), plus: the caller
    /// must ensure exclusive access (no other references to this buffer's memory
    /// exist).
    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> Result<&mut [T]> {
        if !self.storage.cpu_writable {
            return Err(MlxError::InvalidArgument(
                "Cannot mutate a read-only file-backed buffer".into(),
            ));
        }
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(MlxError::InvalidArgument(
                "Cannot view buffer as zero-sized type".into(),
            ));
        }
        let byte_len = self.data_byte_len;
        if byte_len % elem_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer byte length {byte_len} is not a multiple of element size {elem_size}"
            )));
        }
        let base = self.contents_ptr();
        if base.is_null() {
            return Err(MlxError::BufferAllocationError { bytes: byte_len });
        }
        let ptr = unsafe { (base as *mut u8).add(self.byte_offset as usize) };
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer data offset {} is not aligned for {}",
                self.byte_offset,
                std::any::type_name::<T>()
            )));
        }
        let count = byte_len / elem_size;
        // SAFETY: same as as_slice, plus caller ensures exclusive mutable access.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) };
        Ok(slice)
    }

    /// View the logical tensor region as a typed slice.
    ///
    /// Unlike [`as_slice`](Self::as_slice), this honors `shape` and
    /// `byte_offset`. It is intended for pooled tensors whose physical Metal
    /// allocation may be larger than the logical tensor. Quantized buffers
    /// whose shape describes dequantized dimensions should continue to use
    /// `as_slice` for physical-byte access.
    pub fn as_logical_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        let (ptr, count) = self.logical_cpu_view::<T>()?;
        // SAFETY: logical_cpu_view validates the complete range and alignment;
        // the caller upholds the documented type and synchronization contract.
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, count) })
    }

    /// Mutable counterpart to [`as_logical_slice`](Self::as_logical_slice).
    pub fn as_logical_mut_slice<T: bytemuck::Pod>(&mut self) -> Result<&mut [T]> {
        if !self.storage.cpu_writable {
            return Err(MlxError::InvalidArgument(
                "Cannot mutate a read-only file-backed buffer".into(),
            ));
        }
        let (ptr, count) = self.logical_cpu_view::<T>()?;
        // SAFETY: same as as_logical_slice, plus `&mut self` provides the
        // exclusive handle required by the mutable CPU-access contract.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) })
    }

    fn logical_cpu_view<T: bytemuck::Pod>(&self) -> Result<(*mut u8, usize)> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(MlxError::InvalidArgument(
                "Cannot view buffer as zero-sized type".into(),
            ));
        }
        let logical_bytes = self
            .element_count()
            .checked_mul(self.dtype.size_of())
            .ok_or_else(|| {
                MlxError::InvalidArgument("Buffer logical byte length overflow".into())
            })?;
        if logical_bytes % elem_size != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer logical byte length {logical_bytes} is not a multiple of element size {elem_size}"
            )));
        }
        if logical_bytes > self.data_byte_len {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer logical byte length {logical_bytes} exceeds data length {}",
                self.data_byte_len
            )));
        }
        let offset = self.byte_offset as usize;
        let end = offset.checked_add(logical_bytes).ok_or_else(|| {
            MlxError::InvalidArgument("Buffer logical CPU view range overflow".into())
        })?;
        let allocation_len = self.byte_len();
        if end > allocation_len {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer logical CPU view [{offset}, {end}) exceeds allocation length {allocation_len}"
            )));
        }
        let base = self.contents_ptr();
        if base.is_null() {
            return Err(MlxError::BufferAllocationError {
                bytes: logical_bytes,
            });
        }
        let ptr = unsafe { (base as *mut u8).add(offset) };
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "Buffer logical CPU view offset {offset} is not aligned for {}",
                std::any::type_name::<T>()
            )));
        }
        Ok((ptr, logical_bytes / elem_size))
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

    /// Produce a zero-copy clone of this buffer with a new logical shape.
    ///
    /// The cloned buffer shares the same underlying Metal allocation
    /// and residency-set guard via `Arc::clone` on the storage; only
    /// the per-handle `shape` metadata is replaced.  Both handles
    /// continue to alias the same GPU memory — writes through one are
    /// observed by the other.
    ///
    /// Validates `shape.iter().product() == self.element_count()`
    /// and `self.dtype == dtype` (dtype unchanged).  Useful for
    /// implementing zero-copy `view`/`reshape` ops in autograd tapes.
    ///
    /// ADR-020 iter-13c: tape `view` op dependency.
    pub fn with_shape(&self, shape: Vec<usize>) -> std::result::Result<Self, MlxError> {
        let numel: usize = shape.iter().product();
        if numel != self.element_count() {
            return Err(MlxError::InvalidArgument(format!(
                "with_shape: numel({:?}) = {numel} != element_count = {}",
                shape,
                self.element_count(),
            )));
        }
        Ok(Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape,
            byte_offset: self.byte_offset,
            data_byte_len: self.data_byte_len,
        })
    }
}

impl fmt::Debug for MlxBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxBuffer")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("byte_len", &self.byte_len())
            .field("data_byte_len", &self.data_byte_len())
            .field("byte_offset", &self.byte_offset)
            .field("file_backed", &self.is_file_backed())
            .finish()
    }
}
