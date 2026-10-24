use super::{
    authority::{Domain, ProcessPoolAuthority},
    resource::{self, Identity},
    state::{self, RootState, Status},
    BoundedPoolConfig, BoundedPoolError as E, BoundedPoolSnapshot, Result,
};
use crate::{DType, MlxBuffer, MlxDevice};
use metal::foreign_types::ForeignType;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Borrowed batch input; no unbounded iterator is accepted.
pub struct PoolRequest<'a> {
    pub bytes: usize,
    pub dtype: DType,
    pub shape: &'a [usize],
}
/// Non-Clone issue token. Drop destroys its payload before releasing live charge.
/// Escaping cloned/raw buffers remain subject to the unsafe recycling contract.
pub struct PoolLease {
    buffer: Option<MlxBuffer>,
    domain: usize,
    root: usize,
    epoch: u64,
    live: Arc<AtomicUsize>,
}
impl PoolLease {
    pub fn buffer(&self) -> &MlxBuffer {
        self.buffer.as_ref().expect("live lease payload")
    }
    /// Mutable access does not permit recycling while escaped references live.
    pub fn buffer_mut(&mut self) -> &mut MlxBuffer {
        self.buffer.as_mut().expect("live lease payload")
    }
}
impl Drop for PoolLease {
    fn drop(&mut self) {
        super::drop_order::drop_payload_then_release(self.buffer.take(), &self.live);
    }
}
pub struct ReturnFailure {
    pub error: E,
    pub lease: PoolLease,
}
impl std::fmt::Debug for ReturnFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReturnFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
struct Root {
    buffer: Option<metal::Buffer>,
    bytes: usize,
    logical_bytes: usize,
    state: RootState,
}
/// Fixed optional heap pool, explicitly separate from legacy allocation paths.
///
/// The process authority retains its heap, every created child and residency
/// ownership until exit. Clear/Drop refund nothing; lease Drop does not recycle.
/// This mechanism does not establish a whole-OS memory bound.
///
/// Raw makeAliasable/purgeable/mode mutation is forbidden for its entire domain.
/// Tracked hazards do not establish CPU quiescence or eliminate caller aliases.
pub struct BoundedMlxBufferPool {
    authority: &'static ProcessPoolAuthority,
    domain: usize,
    device: MlxDevice,
    identity: Identity,
    heap: metal::Heap,
    config: BoundedPoolConfig,
    roots: Vec<Root>,
    live: Arc<AtomicUsize>,
}
impl BoundedMlxBufferPool {
    pub(super) fn create(
        authority: &'static ProcessPoolAuthority,
        device: &MlxDevice,
        config: BoundedPoolConfig,
    ) -> Result<Self> {
        if config.requested_heap_bytes == 0
            || config.reserved_heap_bytes < config.requested_heap_bytes
            || config.max_child_creations == 0
        {
            return Err(E::InvalidRequest);
        }
        if config.max_child_creations > authority.limits.max_child_creations
            || config.reserved_heap_bytes > authority.limits.max_reserved_heap_bytes
        {
            return Err(E::Quota);
        }
        let mut ledger = authority.ledger.lock().map_err(|_| E::Poisoned)?;
        if ledger.poisoned {
            return Err(E::Poisoned);
        }
        let domains = ledger.snapshot.domains.checked_add(1).ok_or(E::Overflow)?;
        let bytes = ledger
            .snapshot
            .reserved_heap_bytes
            .checked_add(config.reserved_heap_bytes)
            .ok_or(E::Overflow)?;
        let slots = ledger
            .snapshot
            .reserved_child_slots
            .checked_add(config.max_child_creations)
            .ok_or(E::Overflow)?;
        if domains > authority.limits.max_domains
            || bytes > authority.limits.max_reserved_heap_bytes
            || slots > authority.limits.max_child_creations
        {
            return Err(E::Quota);
        }
        // Serialize metadata preallocation too: refused concurrent constructors
        // cannot create unbounded temporary per-domain descriptor arrays.
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(config.max_child_creations)
            .map_err(|_| E::Quota)?;
        let live = Arc::new(AtomicUsize::new(0));
        let mut children = Vec::new();
        children
            .try_reserve_exact(config.max_child_creations)
            .map_err(|_| E::Quota)?;
        // Precharge before Metal call; lock serializes all optional constructors.
        authority.attempt(&mut ledger)?;
        let previous = ledger.snapshot;
        ledger.snapshot.domains = domains;
        ledger.snapshot.reserved_heap_bytes = bytes;
        ledger.snapshot.reserved_child_slots = slots;
        let heap = match resource::heap(device, config.requested_heap_bytes) {
            Ok(heap) => heap,
            Err(error) => {
                let attempts = ledger.snapshot.constructor_attempts;
                ledger.snapshot = previous;
                ledger.snapshot.constructor_attempts = attempts;
                return Err(error);
            }
        };
        let observed = ledger.snapshot.observed_heap_bytes.checked_add(heap.size());
        let valid = resource::heap_valid(&heap)
            && heap.size() > 0
            && heap.size() <= config.reserved_heap_bytes
            && observed.is_some();
        let domain = ledger.domains.len();
        ledger.domains.push(Domain {
            heap: Some(heap.clone()),
            owner: device.clone(),
            children,
            retired: !valid,
        });
        ledger.snapshot.observed_heap_bytes = observed.unwrap_or(u64::MAX);
        if !valid {
            ledger.poisoned = true;
            ledger.snapshot.retired_domains += 1;
            return Err(E::EnvelopeBreached);
        }
        drop(ledger);
        Ok(Self {
            authority,
            domain,
            device: device.clone(),
            identity: Identity::of(device),
            heap,
            config,
            roots,
            live,
        })
    }
    fn check(&self, device: &MlxDevice) -> Result<()> {
        self.identity.check(device)?;
        if self
            .authority
            .ledger
            .lock()
            .map_err(|_| E::Poisoned)?
            .poisoned
        {
            return Err(E::Poisoned);
        }
        Ok(())
    }
    fn request(&self, request: &PoolRequest<'_>) -> Result<usize> {
        if request.shape.len() > self.authority.limits.max_rank {
            return Err(E::Quota);
        }
        let elements = request
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(E::Overflow))?;
        let expected = elements
            .checked_mul(request.dtype.size_of())
            .ok_or(E::Overflow)?;
        if expected != request.bytes {
            return Err(E::InvalidRequest);
        }
        let bucket = state::bucket(request.bytes)?;
        if bucket as u64 > self.device.metal_device().max_buffer_length() {
            return Err(E::InvalidRequest);
        }
        let layout = self
            .device
            .metal_device()
            .heap_buffer_size_and_align(bucket as u64, resource::options());
        if layout.size == 0
            || layout.align == 0
            || !layout.align.is_power_of_two()
            || (layout
                .size
                .checked_add(layout.align - 1)
                .ok_or(E::Overflow)?
                / layout.align)
                .checked_mul(layout.align)
                .ok_or(E::Overflow)?
                > self.config.reserved_heap_bytes
        {
            return Err(E::Quota);
        }
        Ok(bucket)
    }
    /// Allocate a logical view; zero the entire bucket only on fresh creation.
    /// Reused roots preserve prior contents. Lease Drop does not recycle or
    /// refund anything; release/reset require their unsafe quiescence contract.
    pub fn alloc(
        &mut self,
        device: &MlxDevice,
        bytes: usize,
        dtype: DType,
        shape: &[usize],
    ) -> Result<PoolLease> {
        self.allocate(
            device,
            PoolRequest {
                bytes,
                dtype,
                shape,
            },
            true,
        )
    }
    /// # Safety
    /// Initialize every byte a CPU/GPU consumer can read before observation.
    pub unsafe fn alloc_uninitialized(
        &mut self,
        device: &MlxDevice,
        bytes: usize,
        dtype: DType,
        shape: &[usize],
    ) -> Result<PoolLease> {
        self.allocate(
            device,
            PoolRequest {
                bytes,
                dtype,
                shape,
            },
            false,
        )
    }
    fn allocate(
        &mut self,
        device: &MlxDevice,
        request: PoolRequest<'_>,
        zero: bool,
    ) -> Result<PoolLease> {
        self.check(device)?;
        let bucket = self.request(&request)?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(request.shape.len())
            .map_err(|_| E::Quota)?;
        shape.extend_from_slice(request.shape);
        let free = self
            .roots
            .iter()
            .position(|root| root.bytes == bucket && root.state.status == Status::Free);
        let (index, epoch) = if let Some(index) = free {
            let root = &mut self.roots[index];
            resource::validate(
                root.buffer.as_ref().ok_or(E::ResourceMismatch)?,
                &self.heap,
                bucket,
            )?;
            let epoch = root.state.issue()?;
            root.logical_bytes = request.bytes;
            (index, epoch)
        } else {
            if self.roots.len() >= self.config.max_child_creations {
                return Err(E::Quota);
            }
            let mut ledger = self.authority.ledger.lock().map_err(|_| E::Poisoned)?;
            self.authority.attempt(&mut ledger)?;
            // Nil is an exhausted/fragmented heap, never a global retry.
            let buffer = resource::child(&self.heap, bucket as u64).ok_or(E::AllocationFailed)?;
            ledger.domains[self.domain].children.push(buffer.clone());
            ledger.snapshot.successful_child_creations += 1; // bounded by precharged slots
            drop(ledger);
            let index = self.roots.len();
            let valid = resource::validate(&buffer, &self.heap, bucket);
            self.roots.push(Root {
                buffer: Some(buffer),
                bytes: bucket,
                logical_bytes: request.bytes,
                state: RootState::fresh(),
            });
            if let Err(error) = valid {
                self.roots[index].state.status = Status::Retired;
                self.roots[index].buffer = None;
                return Err(error);
            }
            let buffer = self.roots[index].buffer.as_ref().unwrap();
            if zero {
                // Exclusive unpublished fresh root; initialize the entire bucket.
                unsafe {
                    std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bucket);
                }
            }
            if let Some(set) = self.device.residency_set() {
                set.add_allocation(buffer);
                set.commit();
            }
            (index, 1)
        };
        let raw = self.roots[index].buffer.as_ref().unwrap().clone();
        // Validated dense shape; this logical view may be smaller than the bucket.
        let buffer = MlxBuffer::from_raw(raw, request.dtype, Vec::new())
            .data_view(0, request.bytes, request.dtype, shape)
            .map_err(|_| E::InvalidRequest)?;
        self.live.fetch_add(1, Ordering::AcqRel); // at most one live lease per issued root
        Ok(PoolLease {
            buffer: Some(buffer),
            domain: self.domain,
            root: index,
            epoch,
            live: self.live.clone(),
        })
    }
    /// No external/device-direct/mapped resource enters this domain.
    pub fn register_existing(&mut self, _device: &MlxDevice, _buffer: &MlxBuffer) -> Result<()> {
        Err(E::ImportUnsupported)
    }
    /// # Safety
    /// All GPU uses and escaped view/raw/RecordedBinding accesses must be quiescent.
    /// No raw aliasable/purgeable/mode mutation is permitted during domain ownership.
    pub unsafe fn try_release(
        &mut self,
        device: &MlxDevice,
        lease: PoolLease,
    ) -> std::result::Result<(), ReturnFailure> {
        let result = (|| {
            self.check(device)?;
            if lease.domain != self.domain || !Arc::ptr_eq(&lease.live, &self.live) {
                return Err(E::InvalidLease);
            }
            let root = self.roots.get_mut(lease.root).ok_or(E::InvalidLease)?;
            if lease.buffer().metal_buffer().as_ptr()
                != root.buffer.as_ref().ok_or(E::InvalidLease)?.as_ptr()
            {
                return Err(E::InvalidLease);
            }
            resource::validate(
                root.buffer.as_ref().ok_or(E::ResourceMismatch)?,
                &self.heap,
                root.bytes,
            )?;
            root.state.release(lease.epoch)?;
            root.logical_bytes = 0;
            Ok(())
        })();
        match result {
            Ok(()) => {
                drop(lease);
                Ok(())
            }
            Err(error) => Err(ReturnFailure { error, lease }),
        }
    }
    /// # Safety
    /// Zero live leases is necessary, not sufficient: all escaped aliases and
    /// submitted GPU uses of issued roots must also be quiescent.
    pub unsafe fn try_reset(&mut self, device: &MlxDevice) -> Result<()> {
        self.check(device)?;
        state::reset_allowed(self.live.load(Ordering::Acquire))?;
        // Preflight all affected roots before any transition.
        for root in &self.roots {
            if matches!(root.state.status, Status::Issued(_)) {
                resource::validate(
                    root.buffer.as_ref().ok_or(E::ResourceMismatch)?,
                    &self.heap,
                    root.bytes,
                )?;
            }
        }
        for root in &mut self.roots {
            root.state.reset();
            root.logical_bytes = 0;
        }
        Ok(())
    }
    /// Retire Free roots from reuse without reclaiming physical children.
    /// The authority retains children, heap and residency ownership until exit;
    /// no backing reservation or historical creation quota is refunded.
    pub fn clear_free(&mut self, device: &MlxDevice) -> Result<()> {
        self.check(device)?;
        for root in &mut self.roots {
            if root.state.status == Status::Free {
                root.buffer = None;
                root.state.status = Status::Retired;
            }
        }
        // Authority retains children and residency memberships until exit.
        Ok(())
    }
    pub fn alloc_batch(
        &mut self,
        device: &MlxDevice,
        requests: &[PoolRequest<'_>],
    ) -> Result<Vec<PoolLease>> {
        self.check(device)?;
        if requests.len() > self.authority.limits.max_batch {
            return Err(E::Quota);
        }
        for request in requests {
            self.request(request)?;
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(requests.len())
            .map_err(|_| E::Quota)?;
        for request in requests {
            match self.alloc(device, request.bytes, request.dtype, request.shape) {
                Ok(lease) => result.push(lease),
                Err(error) => {
                    // Only unpublished staged leases are recycled; no caller/GPU saw them.
                    for lease in result {
                        let root = &mut self.roots[lease.root];
                        root.state.release(lease.epoch)?;
                        root.logical_bytes = 0;
                        drop(lease);
                    }
                    return Err(error);
                }
            }
        }
        Ok(result)
    }
    pub fn snapshot(&self) -> BoundedPoolSnapshot {
        let mut result = BoundedPoolSnapshot {
            reserved_heap_bytes: self.config.reserved_heap_bytes,
            observed_heap_bytes: self.heap.size(),
            domains: 1,
            reserved_child_slots: self.config.max_child_creations,
            successful_child_creations: self.roots.len(),
            live_leases: self.live.load(Ordering::Acquire),
            root_state_counts_known: true,
            ..Default::default()
        };
        for root in &self.roots {
            match root.state.status {
                Status::Issued(_) => {
                    result.issued_roots += 1;
                    result.issued_logical_bytes += root.logical_bytes as u64;
                }
                Status::Free => result.free_roots += 1,
                Status::Retired => result.retired_roots += 1,
            }
        }
        result
    }
}
impl Drop for BoundedMlxBufferPool {
    fn drop(&mut self) {
        if let Ok(mut ledger) = self.authority.ledger.lock() {
            if !ledger.domains[self.domain].retired {
                ledger.domains[self.domain].retired = true;
                ledger.snapshot.retired_domains += 1;
            }
        }
    }
}
