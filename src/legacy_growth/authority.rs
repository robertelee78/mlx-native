use super::{ledger::Ledger, *};
use crate::MlxDevice;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

struct Owner {
    device: MlxDevice,
    thread: ThreadId,
}
struct State {
    ledger: Ledger,
    owners: Vec<Owner>,
}
/// Process-once authority for optional legacy pools, not a whole-process memory cap.
/// No successful creation credit is refunded on reset, clear, failure or Drop.
/// Retains registered device/queue/residency identities, not every created root.
/// Other allocators, escaped clone metadata and opaque driver bytes are uncovered.
pub struct LegacyPoolGrowthAuthority {
    state: Mutex<State>,
}
static AUTHORITY: OnceLock<LegacyPoolGrowthAuthority> = OnceLock::new();
static INSTALL: Mutex<()> = Mutex::new(());
impl LegacyPoolGrowthAuthority {
    pub fn install(limits: LegacyGrowthLimits) -> Result<&'static Self> {
        let _guard = INSTALL.lock().map_err(|_| fail("installation poisoned"))?;
        if AUTHORITY.get().is_some() {
            return Err(fail("already installed"));
        }
        let ledger = Ledger::prepare(limits)?;
        let mut owners = Vec::new();
        owners
            .try_reserve_exact(limits.max_pools)
            .map_err(|_| fail("owner metadata capacity"))?;
        AUTHORITY
            .set(Self {
                state: Mutex::new(State { ledger, owners }),
            })
            .map_err(|_| fail("already installed"))?;
        Ok(AUTHORITY.get().unwrap())
    }
    pub fn get() -> Option<&'static Self> {
        AUTHORITY.get()
    }
    /// Registers a new empty pool on this thread. Keys and slots are never recycled.
    pub fn bind(
        &'static self,
        device: &MlxDevice,
        key: GrowthPoolKey,
        quota: GrowthQuota,
        buckets: &[GrowthBucket],
    ) -> Result<crate::GrowthLimitedMlxBufferPool> {
        // Prepare all pool-side container capacity before publishing a registration.
        let mut state = self.state.lock().map_err(|_| fail("authority poisoned"))?;
        state.ledger.configuration(key, quota, buckets)?;
        let mut pool = crate::GrowthLimitedMlxBufferPool::prepare(quota, buckets)?;
        let index = state.ledger.register(key, quota, buckets)?;
        state.owners.push(Owner {
            device: device.clone(),
            thread: std::thread::current().id(),
        });
        pool.attach(Binding {
            authority: self,
            index,
        });
        Ok(pool)
    }
    pub fn snapshot(&self, key: GrowthPoolKey) -> Result<LegacyGrowthSnapshot> {
        let state = self.state.lock().map_err(|_| fail("authority poisoned"))?;
        let index = state
            .ledger
            .pools
            .iter()
            .position(|p| p.key == key)
            .ok_or_else(|| fail("unknown pool"))?;
        Ok(state.ledger.snapshot(index))
    }
    /// Invalidates all pools of this generation, including warm reuse; no reset.
    pub fn quarantine_generation(&self, generation: u64) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| fail("authority poisoned"))?;
        state.ledger.quarantine(generation)
    }
}
pub(crate) struct Binding {
    authority: &'static LegacyPoolGrowthAuthority,
    index: usize,
}
fn same_device(a: &MlxDevice, b: &MlxDevice) -> bool {
    std::ptr::eq(a.metal_device(), b.metal_device())
        && a.registry_id() == b.registry_id()
        && std::ptr::eq(&**a.metal_queue(), &**b.metal_queue())
        && a.residency_set().map(|s| s.identity()) == b.residency_set().map(|s| s.identity())
}
impl Binding {
    fn identity(state: &State, index: usize, device: &MlxDevice) -> Result<()> {
        let owner = &state.owners[index];
        if owner.thread != std::thread::current().id() || !same_device(&owner.device, device) {
            return Err(fail("thread/device/queue/residency identity mismatch"));
        }
        if !state.ledger.usable(index) {
            return Err(fail("quarantined generation"));
        }
        Ok(())
    }
    pub fn check(&self, device: &MlxDevice) -> Result<()> {
        let state = self
            .authority
            .state
            .lock()
            .map_err(|_| fail("authority poisoned"))?;
        Self::identity(&state, self.index, device)
    }
    fn request_locked(
        state: &State,
        index: usize,
        device: &MlxDevice,
        bytes: usize,
        rank: usize,
    ) -> Result<usize> {
        Self::identity(state, index, device)?;
        if rank > state.ledger.limits.max_rank {
            return Err(fail("rank limit"));
        }
        let bucket = checked_bucket(bytes)?;
        if bucket as u64 > device.metal_device().max_buffer_length() as u64 {
            return Err(fail("Metal maximum buffer length"));
        }
        state.ledger.validate(index, bucket)?;
        Ok(bucket)
    }
    pub fn request(&self, device: &MlxDevice, bytes: usize, rank: usize) -> Result<usize> {
        let state = self
            .authority
            .state
            .lock()
            .map_err(|_| fail("authority poisoned"))?;
        Self::request_locked(&state, self.index, device, bytes, rank)
    }
    /// Validation/free-pop/cold-construction linearize under the same lock as
    /// quarantine. An already selected operation may finish after quarantine.
    pub fn select(
        &self,
        device: &MlxDevice,
        bytes: usize,
        rank: usize,
        free: &mut std::collections::HashMap<usize, Vec<crate::root_authority::KnownRoot>>,
    ) -> Result<(usize, crate::root_authority::KnownRoot, bool)> {
        let mut state = self
            .authority
            .state
            .lock()
            .map_err(|_| fail("authority poisoned"))?;
        let bucket = Self::request_locked(&state, self.index, device, bytes, rank)?;
        if let Some(root) = free.get(&bucket).and_then(|list| list.last()) {
            root.check_device(device.metal_device())?;
        }
        if let Some(buffer) = free.get_mut(&bucket).and_then(|list| list.pop()) {
            return Ok((bucket, buffer, false));
        }
        // device.rs new_shared_buffer Err is prevalidation or nil only; nonnil
        // immediately returns an owned Buffer. Commit precedes all later work.
        let buffer = state
            .ledger
            .construct(self.index, bucket, || device.new_shared_buffer(bucket))?;
        // The local ledger has committed nonnil creation BEFORE checked use.
        buffer.check_device(device.metal_device())?;
        Ok((bucket, buffer, true))
    }
    pub fn batch_limit(&self) -> Result<usize> {
        let state = self
            .authority
            .state
            .lock()
            .map_err(|_| fail("authority poisoned"))?;
        Ok(state.ledger.limits.max_batch)
    }
    pub fn snapshot(&self) -> Result<LegacyGrowthSnapshot> {
        let state = self
            .authority
            .state
            .lock()
            .map_err(|_| fail("authority poisoned"))?;
        Ok(state.ledger.snapshot(self.index))
    }
}

#[cfg(test)]
mod root_history_composition_tests {
    use super::super::*;
    use crate::root_authority::{state::DeviceKey, Core, RootCreationLimits};
    #[test]
    fn root_history_native55_nonnil_health_error_keeps_local_commit_but_nil_refunds_local() {
        for nil in [false, true] {
            let quota = GrowthQuota {
                bytes: 64,
                roots: 1,
                attempts: 1,
            };
            let limits = LegacyGrowthLimits {
                quota,
                max_pools: 1,
                max_buckets_per_pool: 1,
                max_rank: 1,
                max_batch: 1,
            };
            let mut local = ledger::Ledger::prepare(limits).unwrap();
            local
                .register(
                    GrowthPoolKey {
                        kind: 1,
                        worker_generation: 1,
                    },
                    quota,
                    &[GrowthBucket { bytes: 64, quota }],
                )
                .unwrap();
            let global = Core::default();
            let key = DeviceKey {
                object: 1,
                registry: 1,
            };
            {
                let mut s = global.lock().unwrap();
                s.install(RootCreationLimits {
                    max_attempted_requested_bytes: 64,
                    max_root_attempts: 1,
                    max_raw_constructor_attempts: 1,
                    max_registered_devices: 1,
                    max_rope_cache_entries: 0,
                })
                .unwrap();
                s.register(key).unwrap();
            }
            let result = local.construct(0, 64, || {
                global.create(key, 64, false, || {
                    if nil {
                        Err(fail("nil"))
                    } else {
                        global.unknown("postcreate bookkeeping failure");
                        Ok(())
                    }
                })
            });
            assert_eq!(result.is_err(), nil);
            let snapshot = local.snapshot(0);
            assert_eq!(snapshot.pool.bytes, if nil { 0 } else { 64 });
            assert_eq!(snapshot.pool.roots, if nil { 0 } else { 1 });
            assert_eq!(snapshot.pool.attempts, 1);
            let state = global.lock().unwrap();
            assert_eq!((state.bytes, state.roots), (64, 1));
            if !nil {
                assert!(state.health().is_err());
            }
        }
    }
}
