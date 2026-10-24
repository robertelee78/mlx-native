//! Optional historical cold-growth accounting for the legacy shared allocator.
//! Requested bucket payload and finite gate records are covered, not driver or
//! allocator metadata, external buffer clones, other allocators or crate copies.
#[cfg(test)]
mod allocation_tests;
mod authority;
mod ledger;
#[cfg(test)]
mod ordered_gpu_tests;
#[cfg(test)]
mod tests;
use crate::error::{MlxError, Result};
pub(crate) use authority::Binding;
pub use authority::LegacyPoolGrowthAuthority;

/// Immutable requested-payload, successful-root and constructor-attempt ceilings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthQuota {
    pub bytes: usize,
    pub roots: usize,
    pub attempts: usize,
}
/// Process limits. No selected-model defaults or implicit authority installation.
#[derive(Clone, Copy, Debug)]
pub struct LegacyGrowthLimits {
    pub quota: GrowthQuota,
    pub max_pools: usize,
    pub max_buckets_per_pool: usize,
    pub max_rank: usize,
    pub max_batch: usize,
}
/// An allowed exact power-of-two bucket and its historical ceilings.
#[derive(Clone, Copy, Debug)]
pub struct GrowthBucket {
    pub bytes: usize,
    pub quota: GrowthQuota,
}
/// Caller identity is never recycled, including after pool Drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthPoolKey {
    pub kind: u64,
    pub worker_generation: u64,
}
/// Historical counts; clear/reset/Drop never refund successful creations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrowthCounts {
    pub bytes: usize,
    pub roots: usize,
    pub attempts: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyGrowthSnapshot {
    pub process: GrowthCounts,
    pub registered_pools: usize,
    pub pool: GrowthCounts,
    pub buckets: Vec<(usize, GrowthCounts)>,
    pub quarantined: bool,
}
pub(crate) fn fail(message: &str) -> MlxError {
    MlxError::InvalidArgument(format!("legacy growth gate: {message}"))
}
pub(crate) fn checked_bucket(bytes: usize) -> Result<usize> {
    bytes
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| fail("bucket overflow"))
}
