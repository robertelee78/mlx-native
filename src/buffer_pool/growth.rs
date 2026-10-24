use super::*;
use crate::legacy_growth::{fail, Binding};
use crate::{GrowthBucket, GrowthQuota, LegacyGrowthSnapshot};

/// Optional checked facade over the same legacy shared-buffer allocator.
/// Fresh buckets are zeroed by alloc, reused contents are preserved. No import,
/// per-buffer release, inner-pool escape, fallback or creation-credit refund.
/// Reset is still an unsafe caller lifetime assertion, not a replay/GPU guard.
/// Requested bucket history and authority/pool-owned record cardinalities are
/// bounded. Repeated handouts still create MlxBuffer/view/shape metadata:
/// externally retained handouts, snapshots and completed-command history are
/// NOT bounded by root quotas. HF must separately bound those lifetimes.
/// Opaque driver/host allocator bytes remain unknown.
pub struct GrowthLimitedMlxBufferPool {
    inner: MlxBufferPool,
}
impl GrowthLimitedMlxBufferPool {
    pub(crate) fn prepare(quota: GrowthQuota, buckets: &[GrowthBucket]) -> Result<Self> {
        let mut inner = MlxBufferPool::new();
        inner
            .free
            .try_reserve(buckets.len())
            .map_err(|_| fail("free-map capacity"))?;
        inner
            .in_use
            .try_reserve_exact(quota.roots)
            .map_err(|_| fail("issued capacity"))?;
        inner
            .resident_buffers
            .try_reserve(quota.roots)
            .map_err(|_| fail("residency capacity"))?;
        for b in buckets {
            if inner.free.contains_key(&b.bytes) {
                return Err(fail("duplicate bucket"));
            }
            let mut list = Vec::new();
            list.try_reserve_exact(b.quota.roots)
                .map_err(|_| fail("free-list capacity"))?;
            inner.free.insert(b.bytes, list);
        }
        Ok(Self { inner })
    }
    pub(crate) fn attach(&mut self, binding: Binding) {
        self.inner.growth = Some(binding);
    }
    fn binding(&self) -> &Binding {
        self.inner
            .growth
            .as_ref()
            .expect("registered facade invariant")
    }
    pub fn alloc(
        &mut self,
        device: &MlxDevice,
        bytes: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        self.inner.alloc(device, bytes, dtype, shape)
    }
    /// Every logical byte must be initialized before CPU/GPU reads.
    pub fn alloc_uninitialized(
        &mut self,
        device: &MlxDevice,
        bytes: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<MlxBuffer> {
        self.inner.alloc_uninitialized(device, bytes, dtype, shape)
    }
    /// Bounded slice, not the legacy unbounded iterator. A late error retains
    /// earlier successful creations and in-use entries; no transactional refund.
    pub fn alloc_batch(
        &mut self,
        device: &MlxDevice,
        requests: &[(usize, DType, Vec<usize>)],
    ) -> Result<Vec<MlxBuffer>> {
        self.binding().check(device)?;
        if requests.len() > self.binding().batch_limit()? {
            return Err(fail("batch limit"));
        }
        for (bytes, _, shape) in requests {
            self.binding().request(device, *bytes, shape.len())?;
        }
        self.inner.alloc_batch(device, requests.iter().cloned())
    }
    /// Recycles issued roots; no quota refund or dynamic completion proof.
    /// # Safety
    /// All issued buffers/views/raw/recorded owners must have no future accesses,
    /// and submitted GPU uses must be quiescent, as for the legacy arena contract.
    pub unsafe fn reset(&mut self, device: &MlxDevice) -> Result<()> {
        self.binding().check(device)?;
        self.inner.reset();
        Ok(())
    }
    /// Recycles GPU-only scratch inside ONE unsubmitted, one-shot GraphSession
    /// on the registered queue. No waits, barriers or quota refunds are added.
    /// This is a caller assertion, not a dynamic proof of ordering or replay safety.
    ///
    /// # Safety
    /// All outgoing accesses must already be encoded, with outgoing Rust buffer
    /// handles dropped and no future old-epoch append/raw/callback access. All
    /// outgoing reads AND writes must precede incoming overwrites through actual
    /// buffer barriers and complete physical RAW/WAR/WAW alias dependencies.
    /// Same queue or a literal barrier alone is insufficient for reordered graphs.
    /// Retained capture bindings may be consumed ONCE in this same session via
    /// finish/finish_with_reorder; independent old-segment replay is forbidden.
    /// No CPU read/write/refill/initialization may touch recycled roots until GPU
    /// completion. Incoming GPU producers must fully initialize all bytes read.
    /// Keep every native resource alive through completion; uncertain encoding,
    /// submit or drain failure requires generation quarantine before further reuse.
    /// CPU submission inputs and already-submitted/multi-command lineages are
    /// excluded: use quiescent reset or a separately proven ownership contract.
    pub unsafe fn reset_for_ordered_gpu_reuse(&mut self, device: &MlxDevice) -> Result<()> {
        self.binding().check(device)?;
        self.inner.reset();
        Ok(())
    }
    /// Retires free roots only, preserving preallocated tracking capacity.
    /// In-use roots must first be safely reset. Never refunds historical quota.
    pub fn clear(&mut self, device: &MlxDevice) -> Result<()> {
        self.binding().check(device)?;
        if !self.inner.in_use.is_empty() {
            return Err(fail("clear with issued roots"));
        }
        let mut removed = false;
        for list in self.inner.free.values_mut() {
            for buffer in list.drain(..) {
                if let Some(root) = self.inner.resident_buffers.remove(&buffer_key(&buffer)) {
                    if let Some(set) = &self.inner.residency_set {
                        set.remove_allocation(&root);
                        removed = true;
                    }
                }
            }
        }
        if removed {
            self.inner.residency_set.as_ref().unwrap().commit();
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Result<LegacyGrowthSnapshot> {
        self.binding().snapshot()
    }
}
