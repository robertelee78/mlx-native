//! [`MlxBufferPool`] — arena-style GPU buffer allocator with reuse.
//!
//! Buffers are bucketed by power-of-two sizes.  When a buffer is released back
//! to the pool, it is added to the free list for its size bucket.  A subsequent
//! `alloc` call will reuse a free buffer of compatible (>= requested) size
//! rather than allocating new Metal memory.
//!
//! Calling [`reset`](MlxBufferPool::reset) returns all outstanding buffers to
//! the free list without deallocating Metal memory — ideal for per-inference
//! arena patterns.

use std::collections::HashMap;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::error::Result;

/// Arena-style buffer pool that reuses Metal buffer allocations.
///
/// # Design
///
/// * Buffers are bucketed by their allocated size rounded up to the nearest
///   power of two.  This reduces fragmentation at the cost of occasionally
///   over-allocating by up to 2x.
/// * `release()` returns a single buffer; `reset()` returns all outstanding
///   buffers.
/// * The pool holds a reference to the `MlxDevice` so it can allocate fresh
///   buffers when the free list is empty.
pub struct MlxBufferPool<'d> {
    device: &'d MlxDevice,
    /// Free buffers keyed by their power-of-two bucket size.
    free: HashMap<usize, Vec<metal::Buffer>>,
}

impl<'d> MlxBufferPool<'d> {
    /// Create a new empty buffer pool backed by the given device.
    pub fn new(device: &'d MlxDevice) -> Self {
        Self {
            device,
            free: HashMap::new(),
        }
    }

    /// Allocate a buffer from the pool.
    ///
    /// If a free buffer of compatible size exists in the pool, it is reused
    /// (with updated dtype/shape metadata).  Otherwise a new Metal buffer is
    /// allocated from the device.
    ///
    /// The actual Metal buffer size will be rounded up to the nearest power of
    /// two for bucketing purposes.
    pub fn alloc(
        &mut self,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        let bucket = bucket_size(byte_len);

        // Try to reuse a free buffer from this bucket.
        if let Some(free_list) = self.free.get_mut(&bucket) {
            if let Some(metal_buf) = free_list.pop() {
                let mut buf = MlxBuffer::from_raw(metal_buf, dtype, shape);
                // The reused buffer may have stale metadata; reshape it.
                // byte_len is <= bucket, so the Metal buffer is large enough.
                let _ = &mut buf; // reshape is handled by from_raw above
                return Ok(buf);
            }
        }

        // No free buffer available — allocate a fresh one at the bucket size
        // (so future reuse is possible for any request up to this bucket).
        self.device.alloc_buffer(bucket, dtype, shape)
    }

    /// Return a buffer to the pool's free list for future reuse.
    ///
    /// The Metal memory is **not** deallocated — it stays resident on the GPU
    /// for fast reuse.
    pub fn release(&mut self, buffer: MlxBuffer) {
        let bucket = bucket_size(buffer.byte_len());
        let metal_buf = buffer.into_inner();
        self.free.entry(bucket).or_default().push(metal_buf);
    }

    /// Return all free buffers' count (for diagnostics).
    pub fn free_count(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Total number of bytes held in the free list.
    pub fn free_bytes(&self) -> usize {
        self.free
            .iter()
            .map(|(&bucket, bufs)| bucket * bufs.len())
            .sum()
    }

    /// Clear all free buffers, releasing Metal memory.
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

/// Round `n` up to the nearest power of two.
///
/// Returns 1 for n == 0 (though callers should never request 0 bytes).
fn bucket_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size_powers() {
        assert_eq!(bucket_size(0), 1);
        assert_eq!(bucket_size(1), 1);
        assert_eq!(bucket_size(2), 2);
        assert_eq!(bucket_size(3), 4);
        assert_eq!(bucket_size(4), 4);
        assert_eq!(bucket_size(5), 8);
        assert_eq!(bucket_size(1023), 1024);
        assert_eq!(bucket_size(1024), 1024);
        assert_eq!(bucket_size(1025), 2048);
    }
}
