use super::*;

pub(super) struct Bucket {
    pub config: GrowthBucket,
    pub counts: GrowthCounts,
}
pub(super) struct Pool {
    pub key: GrowthPoolKey,
    pub quota: GrowthQuota,
    pub counts: GrowthCounts,
    pub buckets: Vec<Bucket>,
    pub quarantined: bool,
}
pub(super) struct Ledger {
    pub limits: LegacyGrowthLimits,
    pub counts: GrowthCounts,
    pub pools: Vec<Pool>,
    quarantined: Vec<u64>,
    poisoned: bool,
}
fn next(c: GrowthCounts, q: GrowthQuota, bytes: usize) -> Result<GrowthCounts> {
    let n = GrowthCounts {
        bytes: c
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| fail("byte overflow"))?,
        roots: c
            .roots
            .checked_add(1)
            .ok_or_else(|| fail("root overflow"))?,
        attempts: c
            .attempts
            .checked_add(1)
            .ok_or_else(|| fail("attempt overflow"))?,
    };
    if n.bytes > q.bytes || n.roots > q.roots || n.attempts > q.attempts {
        return Err(fail("cold-growth quota exhausted"));
    }
    Ok(n)
}
impl Ledger {
    pub fn prepare(limits: LegacyGrowthLimits) -> Result<Self> {
        if limits.max_pools == 0
            || limits.max_buckets_per_pool == 0
            || limits.max_buckets_per_pool > usize::BITS as usize
            || limits.max_rank == 0
            || limits.max_batch == 0
        {
            return Err(fail("invalid finite metadata limits"));
        }
        let mut pools = Vec::new();
        pools
            .try_reserve_exact(limits.max_pools)
            .map_err(|_| fail("pool metadata capacity"))?;
        let mut quarantined = Vec::new();
        quarantined
            .try_reserve_exact(limits.max_pools)
            .map_err(|_| fail("quarantine metadata capacity"))?;
        Ok(Self {
            limits,
            counts: GrowthCounts::default(),
            pools,
            quarantined,
            poisoned: false,
        })
    }
    pub fn configuration(
        &self,
        key: GrowthPoolKey,
        quota: GrowthQuota,
        configs: &[GrowthBucket],
    ) -> Result<()> {
        if self.poisoned || self.quarantined.contains(&key.worker_generation) {
            return Err(fail("quarantined generation"));
        }
        if self.pools.len() == self.limits.max_pools || self.pools.iter().any(|p| p.key == key) {
            return Err(fail("pool slot exhausted or identity already registered"));
        }
        if configs.is_empty() || configs.len() > self.limits.max_buckets_per_pool {
            return Err(fail("bucket metadata limit"));
        }
        for (index, config) in configs.iter().enumerate() {
            if !config.bytes.is_power_of_two()
                || configs[..index].iter().any(|b| b.bytes == config.bytes)
            {
                return Err(fail("invalid or duplicate bucket"));
            }
            // Root cardinality also bounds the pool's tracking containers.
            if config.quota.roots > quota.roots || quota.roots > self.limits.quota.roots {
                return Err(fail("root metadata exceeds parent ceiling"));
            }
        }
        Ok(())
    }
    pub fn register(
        &mut self,
        key: GrowthPoolKey,
        quota: GrowthQuota,
        configs: &[GrowthBucket],
    ) -> Result<usize> {
        self.configuration(key, quota, configs)?;
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(configs.len())
            .map_err(|_| fail("bucket metadata capacity"))?;
        buckets.extend(configs.iter().map(|config| Bucket {
            config: *config,
            counts: GrowthCounts::default(),
        }));
        let index = self.pools.len();
        self.pools.push(Pool {
            key,
            quota,
            counts: GrowthCounts::default(),
            buckets,
            quarantined: false,
        });
        Ok(index)
    }
    pub fn validate(&self, pool: usize, bytes: usize) -> Result<usize> {
        let p = self.pools.get(pool).ok_or_else(|| fail("unknown pool"))?;
        if self.poisoned || p.quarantined {
            return Err(fail("quarantined generation"));
        }
        p.buckets
            .iter()
            .position(|b| b.config.bytes == bytes)
            .ok_or_else(|| fail("unregistered bucket"))
    }
    /// One lock protects all preflight and mutation; no partial charge on refusal.
    pub fn reserve(&mut self, pool: usize, bytes: usize) -> Result<usize> {
        let bucket = self.validate(pool, bytes)?;
        let p = &self.pools[pool];
        let process = next(self.counts, self.limits.quota, bytes)?;
        let pc = next(p.counts, p.quota, bytes)?;
        let bc = next(
            p.buckets[bucket].counts,
            p.buckets[bucket].config.quota,
            bytes,
        )?;
        self.counts = process;
        self.pools[pool].counts = pc;
        self.pools[pool].buckets[bucket].counts = bc;
        Ok(bucket)
    }
    /// Called only under the same lock for a constructor that returned no root.
    pub fn nil(&mut self, pool: usize, bucket: usize, bytes: usize) {
        for c in [&mut self.counts, &mut self.pools[pool].counts] {
            c.bytes -= bytes;
            c.roots -= 1;
        }
        let c = &mut self.pools[pool].buckets[bucket].counts;
        c.bytes -= bytes;
        c.roots -= 1;
    }
    pub fn quarantine(&mut self, generation: u64) -> Result<()> {
        if !self.quarantined.contains(&generation) {
            if self.quarantined.len() == self.limits.max_pools {
                self.poisoned = true;
                return Err(fail("quarantine registry exhausted; all pools refused"));
            }
            self.quarantined.push(generation);
        }
        for p in &mut self.pools {
            if p.key.worker_generation == generation {
                p.quarantined = true;
            }
        }
        Ok(())
    }
    pub fn usable(&self, index: usize) -> bool {
        !self.poisoned && !self.pools[index].quarantined
    }
    /// Same decision boundary used by production. Constructor Err must mean
    /// no root was created; postcreation work MUST occur after this returns.
    pub fn construct<T>(
        &mut self,
        pool: usize,
        bytes: usize,
        constructor: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let bucket = self.reserve(pool, bytes)?;
        match constructor() {
            Ok(value) => Ok(value),
            Err(error) => {
                self.nil(pool, bucket, bytes);
                Err(error)
            }
        }
    }
    pub fn snapshot(&self, index: usize) -> LegacyGrowthSnapshot {
        let p = &self.pools[index];
        LegacyGrowthSnapshot {
            process: self.counts,
            registered_pools: self.pools.len(),
            pool: p.counts,
            buckets: p
                .buckets
                .iter()
                .map(|b| (b.config.bytes, b.counts))
                .collect(),
            quarantined: !self.usable(index),
        }
    }
}
