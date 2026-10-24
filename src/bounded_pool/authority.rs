use super::pool::BoundedMlxBufferPool;
use super::{BoundedPoolConfig, BoundedPoolError as E, BoundedPoolSnapshot, Result};
use crate::MlxDevice;
use std::sync::{Mutex, OnceLock};

/// Fixed process-aggregate limits for this optional allocation domain only.
#[derive(Clone, Copy, Debug)]
pub struct ProcessPoolLimits {
    pub max_domains: usize,
    pub max_reserved_heap_bytes: u64,
    pub max_child_creations: usize,
    pub max_constructor_attempts: usize,
    pub max_rank: usize,
    pub max_batch: usize,
}
pub(super) struct Domain {
    // Heap is deliberately retained by the process authority, not the pool.
    pub heap: Option<metal::Heap>,
    // Strong retention keeps device, queue and residency owner alive until exit.
    #[allow(dead_code)]
    pub owner: MlxDevice,
    pub children: Vec<metal::Buffer>,
    pub retired: bool,
}
pub(super) struct Ledger {
    pub domains: Vec<Domain>,
    pub snapshot: BoundedPoolSnapshot,
    pub poisoned: bool,
}
/// One process-installed authority for this optional allocation mechanism.
///
/// Retains created heaps, children and residency ownership until process exit.
/// Pool clear/Drop and lease Drop never refund reservations or creation quotas;
/// lease Drop does not recycle roots. This is not a whole-OS memory bound.
pub struct ProcessPoolAuthority {
    pub(super) limits: ProcessPoolLimits,
    pub(super) ledger: Mutex<Ledger>,
}
static AUTHORITY: OnceLock<ProcessPoolAuthority> = OnceLock::new();
static INSTALL_LOCK: Mutex<()> = Mutex::new(());
impl ProcessPoolAuthority {
    /// Installs exactly once. No public independent authority or budget reset.
    /// Other allocators/crate copies are outside this authority's coverage.
    pub fn install(limits: ProcessPoolLimits) -> Result<&'static Self> {
        let _install = INSTALL_LOCK.lock().map_err(|_| E::Poisoned)?;
        if AUTHORITY.get().is_some() {
            return Err(E::AlreadyInstalled);
        }
        let authority = Self::prepare(limits)?;
        AUTHORITY.set(authority).map_err(|_| E::AlreadyInstalled)?;
        Ok(AUTHORITY.get().unwrap())
    }
    pub fn get() -> Option<&'static Self> {
        AUTHORITY.get()
    }
    fn prepare(limits: ProcessPoolLimits) -> Result<Self> {
        if limits.max_domains == 0
            || limits.max_reserved_heap_bytes == 0
            || limits.max_child_creations == 0
            || limits.max_constructor_attempts == 0
            || limits.max_rank == 0
            || limits.max_batch == 0
        {
            return Err(E::InvalidRequest);
        }
        let mut domains = Vec::new();
        domains
            .try_reserve_exact(limits.max_domains)
            .map_err(|_| E::Quota)?;
        Ok(Self {
            limits,
            ledger: Mutex::new(Ledger {
                domains,
                snapshot: BoundedPoolSnapshot::default(),
                poisoned: false,
            }),
        })
    }
    pub fn snapshot(&self) -> Result<BoundedPoolSnapshot> {
        Ok(self.ledger.lock().map_err(|_| E::Poisoned)?.snapshot)
    }
    pub fn create_pool(
        &'static self,
        device: &MlxDevice,
        config: BoundedPoolConfig,
    ) -> Result<BoundedMlxBufferPool> {
        BoundedMlxBufferPool::create(self, device, config)
    }
    pub(super) fn attempt(&self, ledger: &mut Ledger) -> Result<()> {
        if ledger.poisoned {
            return Err(E::Poisoned);
        }
        let next = ledger
            .snapshot
            .constructor_attempts
            .checked_add(1)
            .ok_or(E::Overflow)?;
        if next > self.limits.max_constructor_attempts {
            return Err(E::Quota);
        }
        ledger.snapshot.constructor_attempts = next;
        Ok(())
    }
    // Test-only local authority is never exported or available in production.
    #[cfg(test)]
    pub(super) fn test_authority(limits: ProcessPoolLimits) -> Self {
        Self::prepare(limits).unwrap()
    }
}
