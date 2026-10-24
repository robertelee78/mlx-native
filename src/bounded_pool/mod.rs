//! Optional fixed Shared/Tracked heap mechanism. Not a process-footprint bound.
//! Legacy `MlxBufferPool` and all default allocation paths are unchanged.
mod authority;
mod drop_order;
#[cfg(test)]
mod failure_tests;
mod pool;
mod resource;
mod state;
#[cfg(test)]
mod tests;

pub use authority::{ProcessPoolAuthority, ProcessPoolLimits};
pub use pool::{BoundedMlxBufferPool, PoolLease, PoolRequest, ReturnFailure};

/// Per-domain fixed capacity and historical creation allowance.
#[derive(Clone, Copy, Debug)]
pub struct BoundedPoolConfig {
    pub requested_heap_bytes: u64,
    pub reserved_heap_bytes: u64,
    pub max_child_creations: usize,
}

/// Additive errors: no changes to the legacy exhaustive error enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedPoolError {
    AlreadyInstalled,
    InvalidRequest,
    Overflow,
    Quota,
    Unsupported,
    AllocationFailed,
    EnvelopeBreached,
    Poisoned,
    IdentityMismatch,
    ResourceMismatch,
    LiveLeases,
    InvalidLease,
    ImportUnsupported,
}
impl std::fmt::Display for BoundedPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bounded pool: {self:?}")
    }
}
impl std::error::Error for BoundedPoolError {}
pub(crate) type Result<T> = std::result::Result<T, BoundedPoolError>;

/// Scopes overlap. Neither metadata nor total OS/driver footprint is bounded here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundedPoolSnapshot {
    pub reserved_heap_bytes: u64,
    pub observed_heap_bytes: u64,
    pub domains: usize,
    pub retired_domains: usize,
    pub reserved_child_slots: usize,
    pub successful_child_creations: usize,
    pub constructor_attempts: usize,
    pub issued_roots: usize,
    pub free_roots: usize,
    pub retired_roots: usize,
    pub live_leases: usize,
    pub issued_logical_bytes: u64,
    /// False on aggregate authority snapshots: per-pool states are not aggregated.
    pub root_state_counts_known: bool,
    pub metadata_upper_bound_known: bool,
}
