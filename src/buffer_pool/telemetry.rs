//! Opt-in, fallible census of existing pool owners. No allocation-path hooks.
use std::collections::HashMap;

use crate::error::{MlxError, Result};

/// Root count and sum of complete Metal buffer lengths, not logical view bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolRootCapacity {
    pub roots: usize,
    pub bytes: usize,
}

/// Metrics this pool census cannot measure. Unknown is never represented by zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoolMemoryUnavailable {
    /// Metal residency-set registration is a hint, not observed physical pages.
    PhysicalResidency,
    /// Raw, command, view or caller owners may survive removal from the pool.
    OwnersOutsidePool,
    /// An on-demand census cannot see peaks between observations.
    LifetimeHighWater,
}

/// Current capacities retained by this pool's existing strong root containers.
///
/// Labels overlap: do not add `free`, `issued`, or `residency_hint` to `retained`.
/// Roots are deduplicated by the live Metal OBJECT identity. Views/clones of the
/// same object count once. Distinct objects sharing mapped pages or heap backing
/// are not deduplicated; these are resource capacities, never physical memory.
/// This is one pool's scope. Summing pools requires a separate no-shared-roots
/// proof. The census neither flushes residency hints nor waits for GPU work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolMemorySnapshot {
    pub retained: PoolRootCapacity,
    pub free: PoolRootCapacity,
    pub issued: PoolRootCapacity,
    pub residency_hint: PoolRootCapacity,
    /// At least one Metal root appears under more than one ownership label.
    pub labels_overlap: bool,
    pub physical_residency: PoolMemoryUnavailable,
    pub owners_outside_pool: PoolMemoryUnavailable,
    pub lifetime_high_water: PoolMemoryUnavailable,
}

#[derive(Clone, Copy)]
pub(super) enum Label {
    Free = 1,
    Issued = 2,
    Residency = 4,
}

fn error(message: &str) -> MlxError {
    MlxError::InvalidArgument(format!("pool memory census: {message}"))
}

fn add(total: &mut PoolRootCapacity, bytes: usize) -> Result<()> {
    total.roots = total
        .roots
        .checked_add(1)
        .ok_or_else(|| error("root count overflow"))?;
    total.bytes = total
        .bytes
        .checked_add(bytes)
        .ok_or_else(|| error("capacity overflow"))?;
    Ok(())
}

/// The production census and pure controls share this exact union computation.
pub(super) fn census(
    records: impl IntoIterator<Item = (usize, usize, Label)>,
    record_limit: usize,
) -> Result<PoolMemorySnapshot> {
    let mut roots = HashMap::<usize, (usize, u8)>::new();
    roots
        .try_reserve(record_limit)
        .map_err(|_| error("metadata reservation failed"))?;
    let mut seen = 0usize;
    for (identity, bytes, label) in records {
        seen = seen
            .checked_add(1)
            .ok_or_else(|| error("record count overflow"))?;
        if seen > record_limit {
            return Err(error("record limit exceeded"));
        }
        match roots.get_mut(&identity) {
            Some((length, labels)) => {
                if *length != bytes {
                    return Err(error("one Metal object has inconsistent lengths"));
                }
                *labels |= label as u8;
            }
            None => {
                roots.insert(identity, (bytes, label as u8));
            }
        }
    }
    let mut out = PoolMemorySnapshot {
        retained: PoolRootCapacity::default(),
        free: PoolRootCapacity::default(),
        issued: PoolRootCapacity::default(),
        residency_hint: PoolRootCapacity::default(),
        labels_overlap: false,
        physical_residency: PoolMemoryUnavailable::PhysicalResidency,
        owners_outside_pool: PoolMemoryUnavailable::OwnersOutsidePool,
        lifetime_high_water: PoolMemoryUnavailable::LifetimeHighWater,
    };
    for (bytes, labels) in roots.into_values() {
        add(&mut out.retained, bytes)?;
        if labels & Label::Free as u8 != 0 {
            add(&mut out.free, bytes)?;
        }
        if labels & Label::Issued as u8 != 0 {
            add(&mut out.issued, bytes)?;
        }
        if labels & Label::Residency as u8 != 0 {
            add(&mut out.residency_hint, bytes)?;
        }
        out.labels_overlap |= labels.count_ones() > 1;
    }
    Ok(out)
}

impl super::MlxBufferPool {
    /// Opt-in read-only census; metadata allocation failure/overflow returns Err.
    /// Does not allocate Metal roots, clone resource owners, reset or synchronize.
    /// Caller must treat failure as unavailable, never a zero-byte observation.
    pub fn memory_snapshot(&self) -> Result<PoolMemorySnapshot> {
        let free_records = self
            .free
            .values()
            .try_fold(0usize, |n, v| n.checked_add(v.len()))
            .ok_or_else(|| error("free record count overflow"))?;
        let limit = free_records
            .checked_add(self.in_use.len())
            .and_then(|n| n.checked_add(self.resident_buffers.len()))
            .ok_or_else(|| error("record count overflow"))?;
        let records = self
            .free
            .values()
            .flatten()
            .map(|r| (r, Label::Free))
            .chain(self.in_use.iter().map(|(_, r)| (r, Label::Issued)))
            .chain(
                self.resident_buffers
                    .values()
                    .map(|r| (r, Label::Residency)),
            )
            .map(|(r, label)| {
                let identity = (&**r.metal() as *const metal::BufferRef) as usize;
                (identity, r.metal().length() as usize, label)
            });
        census(records, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_memory_census_deduplicates_aliases_and_overlapping_labels() {
        let s = census(
            [
                (1, 128, Label::Issued),
                (1, 128, Label::Issued),
                (1, 128, Label::Residency),
                (2, 64, Label::Free),
            ],
            4,
        )
        .unwrap();
        assert_eq!(
            s.retained,
            PoolRootCapacity {
                roots: 2,
                bytes: 192
            }
        );
        assert_eq!(s.issued.bytes, 128);
        assert_eq!(s.free.bytes, 64);
        assert_eq!(s.residency_hint.bytes, 128);
        assert!(s.labels_overlap);
    }
    #[test]
    fn pool_memory_census_counts_residency_only_and_distinct_objects() {
        let s = census([(1, 64, Label::Residency), (2, 64, Label::Issued)], 2).unwrap();
        assert_eq!(s.retained.bytes, 128); // equal size/backing is not object identity
        assert_eq!(s.residency_hint.roots, 1);
        assert!(!s.labels_overlap);
    }
    #[test]
    fn pool_memory_census_empty_is_not_physical_or_retirement_proof() {
        let s = census([], 0).unwrap();
        assert_eq!(s.retained.bytes, 0);
        assert_eq!(
            s.physical_residency,
            PoolMemoryUnavailable::PhysicalResidency
        );
        assert_eq!(
            s.owners_outside_pool,
            PoolMemoryUnavailable::OwnersOutsidePool
        );
        assert_eq!(
            s.lifetime_high_water,
            PoolMemoryUnavailable::LifetimeHighWater
        );
    }
    #[test]
    fn pool_memory_census_rejects_inconsistent_identity_overflow_and_bad_limit() {
        assert!(census([(1, 64, Label::Free), (1, 128, Label::Issued)], 2).is_err());
        assert!(census([(1, usize::MAX, Label::Free), (2, 1, Label::Issued)], 2).is_err());
        assert!(census([(1, 64, Label::Free)], 0).is_err());
    }
    #[test]
    fn pool_memory_census_reset_moves_labels_without_changing_union() {
        let before = census([(1, 64, Label::Issued), (1, 64, Label::Residency)], 2).unwrap();
        let after = census([(1, 64, Label::Free), (1, 64, Label::Residency)], 2).unwrap();
        assert_eq!(before.retained, after.retained);
        assert_eq!(after.issued.bytes, 0);
        assert_eq!(after.free.bytes, before.issued.bytes);
        // A later empty census cannot claim escaped owners have been destroyed.
        assert_eq!(
            census([], 0).unwrap().owners_outside_pool,
            PoolMemoryUnavailable::OwnersOutsidePool
        );
    }

    /// Two standalone Shared roots,24B requested payload; no GPU commands.
    #[test]
    #[ignore = "opt-in tiny Metal allocation/residency census; no GPU submission"]
    fn pool_memory_actual_roots_views_reset_and_clear() {
        use crate::{DType, MlxBufferPool, MlxDevice};
        let device = MlxDevice::new().expect("Metal device");
        assert!(
            device.residency_set().is_some(),
            "control requires active residency hints"
        );
        let mut pool = MlxBufferPool::new();
        let a = pool.alloc(&device, 8, DType::F32, vec![2]).unwrap();
        let b = pool.alloc(&device, 16, DType::F32, vec![4]).unwrap();
        let external = a.clone();
        let view = a.slice_view(4, 1);
        let identity = |buffer: &crate::MlxBuffer| {
            (&**buffer.metal_buffer() as *const metal::BufferRef) as usize
        };
        assert_eq!(identity(&a), identity(&external));
        assert_eq!(identity(&a), identity(&view));
        assert_ne!(identity(&a), identity(&b));
        assert_eq!(view.data_byte_len(), 4);
        assert_eq!(view.byte_len(), 8);
        // Re-registering an aliased view must not manufacture a third root.
        pool.register_existing(&device, &view).unwrap();
        let issued = pool.memory_snapshot().unwrap();
        assert_eq!(
            issued.retained,
            PoolRootCapacity {
                roots: 2,
                bytes: 24
            }
        );
        assert_eq!(issued.issued, issued.retained);
        assert_eq!(issued.free, PoolRootCapacity::default());
        assert_eq!(issued.residency_hint, issued.retained);
        assert!(issued.labels_overlap);
        drop(a);
        drop(b);
        // No GPU work exists and no subsequent allocation/reuse occurs. External
        // aliases remain alive solely to falsify a final-owner interpretation.
        pool.reset();
        let free = pool.memory_snapshot().unwrap();
        assert_eq!(free.retained, issued.retained);
        assert_eq!(free.free, free.retained);
        assert_eq!(free.issued, PoolRootCapacity::default());
        assert_eq!(free.residency_hint, free.retained);
        pool.clear();
        let cleared = pool.memory_snapshot().unwrap();
        assert_eq!(cleared.retained, PoolRootCapacity::default());
        assert_eq!(cleared.residency_hint, PoolRootCapacity::default());
        assert_eq!(external.byte_len(), 8);
        assert_eq!(identity(&external), identity(&view));
        assert_eq!(
            cleared.owners_outside_pool,
            PoolMemoryUnavailable::OwnersOutsidePool
        );
        assert_eq!(
            cleared.physical_residency,
            PoolMemoryUnavailable::PhysicalResidency
        );
        assert_eq!(
            cleared.lifetime_high_water,
            PoolMemoryUnavailable::LifetimeHighWater
        );
        drop(view);
        drop(external);
        println!("POOL_MEMORY_ACTUAL_COMPLETE roots=2 requested_bytes=24 gpu_submissions=0 outside_owners=Unavailable");
    }
}
