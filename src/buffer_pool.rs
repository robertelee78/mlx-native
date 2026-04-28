//! [`MlxBufferPool`] — arena-style GPU buffer allocator with reuse.
//!
//! Buffers are bucketed by power-of-two sizes.  When a buffer is released back
//! to the pool, it is added to the free list for its size bucket.  A subsequent
//! `alloc` call will reuse a free buffer of compatible (>= requested) size
//! rather than allocating new Metal memory.
//!
//! Two return-path patterns are supported and **must not be mixed within a
//! single arena cycle**:
//!
//! * **Per-buffer** via [`release`](MlxBufferPool::release) — explicit return
//!   of a single buffer to the free list, suitable for ad-hoc patterns where
//!   the caller knows the precise lifetime of each buffer.
//! * **Arena bulk** via [`reset`](MlxBufferPool::reset) — bulk-return of every
//!   buffer handed out by [`alloc`](MlxBufferPool::alloc) since the previous
//!   reset.  Suitable for per-inference / per-decode-token arena patterns
//!   where no individual buffer's lifetime crosses the reset boundary.
//!
//! Internally, every `alloc` records an ARC-cloned `metal::Buffer` handle so
//! that `reset` can bulk-recycle without requiring callers to enumerate every
//! buffer individually.  ARC retain on `metal::Buffer` is cheap (refcount inc).

use std::collections::HashMap;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::error::{MlxError, Result};

/// Arena-style buffer pool that reuses Metal buffer allocations.
///
/// # Design
///
/// * Buffers are bucketed by their allocated size rounded up to the nearest
///   power of two.  This reduces fragmentation at the cost of occasionally
///   over-allocating by up to 2x.
/// * `release()` returns a single buffer; `reset()` returns all outstanding
///   buffers handed out since the last reset.
/// * The `MlxDevice` is passed in at every [`alloc`] call (rather than stored
///   in the pool).  This keeps the pool free of lifetime parameters so it
///   can be embedded in any owner struct (e.g. the per-decode-token
///   `DecodeBuffers` cache in hf2q's qwen35 forward path).
///
/// # Why an arena reset matters
///
/// In the per-decode-token hot path, each token allocates ~1750 Metal buffers
/// for scratch / intermediate / parameter storage across attention, FFN, and
/// linear-attention layers.  Direct `MlxDevice::alloc_buffer()` calls hit
/// Metal's allocator each time (5-30 µs each); pooling reuses the underlying
/// `metal::Buffer` objects across token boundaries so steady-state allocation
/// cost amortizes to near zero.  See ADR-012 §Optimize / Task #15 for the
/// MoE dwq46 0.90× parity gap that motivated this work.
pub struct MlxBufferPool {
    /// Free buffers keyed by their power-of-two bucket size.
    free: HashMap<usize, Vec<metal::Buffer>>,
    /// Buffers handed out by [`alloc`] since the last [`reset`].  Each entry
    /// holds an ARC-cloned `metal::Buffer` so the pool's reference keeps the
    /// underlying GPU allocation alive even after the caller's `MlxBuffer`
    /// goes out of scope.  [`reset`] drains this into [`free`].
    in_use: Vec<(usize, metal::Buffer)>,
    /// Residency set that owns the allocations registered by this pool.
    residency_set: Option<crate::residency::ResidencySet>,
    /// Unique Metal buffers this pool added to the residency set, keyed by
    /// their stable contents pointer. This avoids double-removing buffers if
    /// callers mix release/reset despite that pattern being unsupported.
    resident_buffers: HashMap<usize, metal::Buffer>,
}

impl Default for MlxBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MlxBufferPool {
    /// Create a new empty buffer pool.  The Metal device is passed to
    /// [`alloc`] at every call site, so the pool itself is lifetime-free.
    pub fn new() -> Self {
        Self {
            free: HashMap::new(),
            in_use: Vec::new(),
            residency_set: None,
            resident_buffers: HashMap::new(),
        }
    }

    /// Allocate a buffer from the pool.
    ///
    /// If a free buffer of compatible size exists in the pool, it is reused
    /// (with updated dtype/shape metadata).  Otherwise a new Metal buffer is
    /// allocated from `device` at the bucket size so future reuse is
    /// possible for any request up to that bucket.
    ///
    /// Each successful `alloc` registers the buffer in the pool's in-use
    /// list (ARC clone — cheap), so a subsequent [`reset`] returns it to
    /// the free list automatically.
    pub fn alloc(
        &mut self,
        device: &MlxDevice,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        let (buffer, added_residency) = self.alloc_inner(device, byte_len, dtype, shape)?;
        if added_residency {
            if let Some(set) = self.residency_set.as_ref() {
                set.commit();
            }
        }
        Ok(buffer)
    }

    /// Allocate several buffers and commit residency-set updates once.
    pub fn alloc_batch<I>(&mut self, device: &MlxDevice, requests: I) -> Result<Vec<MlxBuffer>>
    where
        I: IntoIterator<Item = (usize, DType, Vec<usize>)>,
    {
        let mut buffers = Vec::new();
        let mut added_residency = false;

        for (byte_len, dtype, shape) in requests {
            let (buffer, added) = self.alloc_inner(device, byte_len, dtype, shape)?;
            added_residency |= added;
            buffers.push(buffer);
        }

        if added_residency {
            if let Some(set) = self.residency_set.as_ref() {
                set.commit();
            }
        }

        Ok(buffers)
    }

    fn alloc_inner(
        &mut self,
        device: &MlxDevice,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<(MlxBuffer, bool)> {
        let bucket = bucket_size(byte_len);
        let mut added_residency = false;

        // Try to reuse a free buffer from this bucket.
        let metal_buf = self
            .free
            .get_mut(&bucket)
            .and_then(|free_list| free_list.pop());

        let metal_buf = match metal_buf {
            Some(b) => b,
            None => {
                // Fresh allocation at bucket size.
                let raw = device
                    .metal_device()
                    .new_buffer(bucket as u64, metal::MTLResourceOptions::StorageModeShared);
                if raw.contents().is_null() {
                    return Err(MlxError::BufferAllocationError { bytes: bucket });
                }
                added_residency = self.register_residency_allocation(device, &raw)?;
                raw
            }
        };

        // Track the handout so reset() can recycle it.  ARC clone is cheap.
        self.in_use.push((bucket, metal_buf.clone()));

        Ok((MlxBuffer::from_raw(metal_buf, dtype, shape), added_residency))
    }

    /// Return a single buffer to the pool's free list for future reuse.
    ///
    /// The Metal memory is **not** deallocated — it stays resident on the GPU
    /// for fast reuse.  `release` is the per-buffer alternative to [`reset`];
    /// see the module docs for guidance on which to use.
    ///
    /// **Mixing `release` and `reset` within the same arena cycle is not
    /// supported** — the pool's in-use list does not deduplicate, so a buffer
    /// returned via `release` and then bulk-returned via `reset` would land in
    /// the free list twice (each entry holds an ARC clone of the same Metal
    /// buffer; the duplication wastes a free-list slot but is not a memory
    /// leak — both clones drop together once popped).  Pick one pattern per
    /// arena cycle.
    pub fn release(&mut self, buffer: MlxBuffer) {
        let bucket = bucket_size(buffer.byte_len());
        let metal_buf = buffer.into_inner();
        self.free.entry(bucket).or_default().push(metal_buf);
    }

    /// Bulk-return every buffer handed out by [`alloc`] since the last reset
    /// to the pool's free list.
    ///
    /// # Caller contract
    ///
    /// All `MlxBuffer` values returned by `alloc` since the last reset must be
    /// out-of-scope (dropped) at the time `reset` is called.  Reset transfers
    /// the pool's ARC clones to the free list, where they become available to
    /// subsequent [`alloc`] calls.  If a caller is still holding an `MlxBuffer`
    /// and a later `alloc` re-issues the underlying buffer, the two callers
    /// will share GPU memory (aliasing).  The Metal ARC keeps the storage
    /// alive in either case, but writes from the new caller will be visible
    /// to the stale caller — a correctness bug, not a memory error.
    ///
    /// In Rust's ownership model, locally-bound `MlxBuffer` values fall out of
    /// scope at the end of their lexical block, making the per-decode-token
    /// arena pattern safe by construction:
    ///
    /// ```ignore
    /// loop {
    ///     pool.reset();          // start of token — recycle previous token's buffers
    ///     forward_pass(&pool);   // many alloc(), no explicit release
    /// }                          // forward_pass returns; locals dropped
    /// ```
    pub fn reset(&mut self) {
        for (bucket, metal_buf) in self.in_use.drain(..) {
            self.free.entry(bucket).or_default().push(metal_buf);
        }
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

    /// Number of buffers currently in-use (alloc'd but not yet reset).
    pub fn in_use_count(&self) -> usize {
        self.in_use.len()
    }

    /// Clear all free buffers, releasing Metal memory.  Does not affect
    /// in-use tracking.
    pub fn clear(&mut self) {
        let mut removed_any = false;

        if let Some(set) = self.residency_set.as_ref() {
            for metal_buf in self.free.values().flatten() {
                let key = buffer_key(metal_buf);
                if let Some(resident_buf) = self.resident_buffers.remove(&key) {
                    set.remove_allocation(&resident_buf);
                    removed_any = true;
                }
            }

            if removed_any {
                set.commit();
            }
        }

        self.free.clear();
    }

    fn register_residency_allocation(
        &mut self,
        device: &MlxDevice,
        buffer: &metal::Buffer,
    ) -> Result<bool> {
        let Some(device_set) = device.residency_set() else {
            return Ok(false);
        };

        match self.residency_set.as_ref() {
            Some(pool_set) if !pool_set.same_owner(device_set) => {
                return Err(MlxError::InvalidArgument(
                    "MlxBufferPool cannot mix residency-enabled devices".into(),
                ));
            }
            Some(_) => {}
            None => {
                self.residency_set = Some(device_set.clone());
            }
        }

        let key = buffer_key(buffer);
        if !self.resident_buffers.contains_key(&key) {
            device_set.add_allocation(buffer);
            self.resident_buffers.insert(key, buffer.clone());
            return Ok(true);
        }

        Ok(false)
    }

    fn remove_all_residency_allocations(&mut self) {
        let Some(set) = self.residency_set.as_ref() else {
            return;
        };

        if self.resident_buffers.is_empty() {
            return;
        }

        for buffer in self.resident_buffers.values() {
            set.remove_allocation(buffer);
        }
        set.commit();
        self.resident_buffers.clear();
    }
}

impl Drop for MlxBufferPool {
    fn drop(&mut self) {
        self.remove_all_residency_allocations();
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

#[inline]
fn buffer_key(buffer: &metal::Buffer) -> usize {
    buffer.contents() as usize
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

    #[test]
    fn test_pool_arena_reset_recycles_in_use() {
        // Per-decode-token arena pattern: alloc many, drop locals, reset, alloc again.
        // Subsequent allocs must reuse the same Metal buffers (verified by ARC-cloned
        // contents pointer).
        let device = MlxDevice::new().expect("device");
        let mut pool = MlxBufferPool::new();

        // Cycle 1: allocate three buffers in different buckets, then drop them
        // (locals fall out of scope at the end of the block).
        let (ptr_a, ptr_b, ptr_c) = {
            let buf_a = pool.alloc(&device, 1024, DType::F32, vec![256]).expect("alloc a");
            let buf_b = pool.alloc(&device, 2048, DType::F32, vec![512]).expect("alloc b");
            let buf_c = pool.alloc(&device, 1024, DType::F32, vec![256]).expect("alloc c");
            (buf_a.contents_ptr(), buf_b.contents_ptr(), buf_c.contents_ptr())
        };
        assert_eq!(pool.in_use_count(), 3);
        assert_eq!(pool.free_count(), 0);

        // Reset returns all three to free.
        pool.reset();
        assert_eq!(pool.in_use_count(), 0);
        assert_eq!(pool.free_count(), 3);

        // Cycle 2: allocate compatible-bucket buffers, must reuse the same
        // underlying Metal buffers (contents_ptr equal).
        let buf_d = pool.alloc(&device, 1024, DType::F32, vec![256]).expect("alloc d");
        let buf_e = pool.alloc(&device, 2048, DType::F32, vec![512]).expect("alloc e");
        let ptr_d = buf_d.contents_ptr();
        let ptr_e = buf_e.contents_ptr();

        // Pointers must come from {a, b, c} — bucket 1024 reuse for d (matches a or c),
        // bucket 2048 reuse for e (matches b).
        assert!(
            ptr_d == ptr_a || ptr_d == ptr_c,
            "buf_d {:?} must reuse one of a {:?} / c {:?}",
            ptr_d, ptr_a, ptr_c,
        );
        assert_eq!(ptr_e, ptr_b, "buf_e must reuse b (only 2048-bucket buffer)");

        // After cycle-2 alloc, free has 1 (the unused 1024-bucket buffer) + in_use 2.
        assert_eq!(pool.in_use_count(), 2);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_pool_reset_with_no_alloc_is_idempotent() {
        // Empty reset must be a no-op.  No MlxDevice required — pool
        // operations on an empty pool don't touch the device; the
        // smoke check used to live here was incidental and triggered
        // the unused-variable warning since `device` was bound but
        // never consumed.
        let mut pool = MlxBufferPool::new();
        pool.reset();
        assert_eq!(pool.in_use_count(), 0);
        assert_eq!(pool.free_count(), 0);
        // Multiple resets without intervening alloc — still no-op.
        pool.reset();
        pool.reset();
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_pool_release_remains_supported_for_compat() {
        // The existing per-buffer release() pattern still works.  Mixing
        // release+reset within the same arena cycle is documented as
        // unsupported but technically lands a duplicate clone in free —
        // verify the duplicate is harmless (alloc still picks up a buffer).
        let device = MlxDevice::new().expect("device");
        let mut pool = MlxBufferPool::new();

        let buf = pool.alloc(&device, 1024, DType::F32, vec![256]).expect("alloc");
        assert_eq!(pool.in_use_count(), 1);
        pool.release(buf);
        // release() does NOT remove from in_use; that's acceptable per the
        // documented contract (don't mix patterns).  Free has the released one.
        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Allocating again pulls from free first.
        let _buf2 = pool.alloc(&device, 1024, DType::F32, vec![256]).expect("alloc 2");
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.in_use_count(), 2);
    }
}
