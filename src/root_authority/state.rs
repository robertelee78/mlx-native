use super::{RootCreationLimits, RootCreationSnapshot};
use crate::{MlxError, Result};
pub(super) fn fail(reason: &str) -> MlxError {
    MlxError::InvalidArgument(format!("native root authority: {reason}"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceKey {
    pub object: usize,
    pub registry: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Charge {
    pub id: usize,
    pub device: DeviceKey,
    pub bytes: usize,
}
#[derive(Default, Debug)]
pub(crate) struct State {
    pub limits: Option<RootCreationLimits>,
    pub legacy: bool,
    pub unknown: Option<&'static str>,
    pub devices: Vec<DeviceKey>,
    pub bytes: usize,
    pub roots: usize,
    pub raw: usize,
    pub inflight: usize,
    pub successful: usize,
    pub cache_slots: usize,
    pub published_cache: usize,
    pub denied: usize,
}
impl State {
    pub fn install(&mut self, limits: RootCreationLimits) -> Result<()> {
        if self.limits.is_some() || self.legacy || self.unknown.is_some() {
            return Err(fail(
                "installation requires fresh native creation/import history",
            ));
        }
        if limits.max_registered_devices == 0 {
            return Err(fail("device limit must be nonzero"));
        }
        self.devices
            .try_reserve_exact(limits.max_registered_devices)
            .map_err(|_| fail("device metadata reservation failed"))?;
        self.limits = Some(limits);
        Ok(())
    }
    pub fn health(&self) -> Result<()> {
        match self.unknown {
            Some(reason) => Err(fail(reason)),
            None => Ok(()),
        }
    }
    pub fn register(&mut self, device: DeviceKey) -> Result<()> {
        self.health()?;
        let limits = self.limits.ok_or_else(|| fail("not installed"))?;
        if self.devices.contains(&device) {
            return Ok(());
        }
        if self.devices.len() >= limits.max_registered_devices {
            return Err(fail("device registration exhausted"));
        }
        self.devices.push(device);
        Ok(())
    }
    pub fn import(&mut self) {
        if self.limits.is_some() {
            self.unknown.get_or_insert("unknown public raw import");
        } else {
            self.legacy = true;
        }
    }
    pub fn heap(&mut self) -> Result<()> {
        self.health()?;
        if self.limits.is_some() {
            self.denied = self.denied.saturating_add(1);
            Err(fail("heap allocator unsupported in covered mode"))
        } else {
            self.legacy = true;
            Ok(())
        }
    }
    pub fn reserve(
        &mut self,
        device: DeviceKey,
        bytes: usize,
        cache: bool,
    ) -> Result<Option<Charge>> {
        self.health()?;
        if bytes == 0 {
            return Err(fail("zero creation request"));
        }
        let Some(limits) = self.limits else {
            self.legacy = true;
            return Ok(None);
        };
        let next =
            |value: usize, add: usize, max: usize| value.checked_add(add).filter(|n| *n <= max);
        let values = (
            next(self.bytes, bytes, limits.max_attempted_requested_bytes),
            next(self.roots, 1, limits.max_root_attempts),
            next(self.raw, 1, limits.max_raw_constructor_attempts),
            next(self.inflight, 1, usize::MAX),
            next(
                self.cache_slots,
                usize::from(cache),
                limits.max_rope_cache_entries,
            ),
        );
        if !self.devices.contains(&device) {
            return Err(fail("unregistered actual Metal device"));
        }
        let (Some(b), Some(r), Some(a), Some(i), Some(c)) = values else {
            self.denied = self.denied.saturating_add(1);
            return Err(fail("historical creation quota exhausted or overflow"));
        };
        self.bytes = b;
        self.roots = r;
        self.raw = a;
        self.inflight = i;
        self.cache_slots = c;
        Ok(Some(Charge {
            id: r,
            device,
            bytes,
        }))
    }
    pub fn finish(&mut self, success: bool, abandoned: bool) {
        self.inflight = self.inflight.saturating_sub(1);
        if success {
            self.successful = self.successful.saturating_add(1);
        }
        if abandoned {
            self.unknown
                .get_or_insert("creation attempt abandoned during unwind");
        }
    }
    pub fn snapshot(&self) -> RootCreationSnapshot {
        RootCreationSnapshot {
            scope: "native_mediated_requested_root_creation_history",
            limits: self.limits,
            covered: self.limits.is_some() && self.unknown.is_none(),
            legacy_used: self.legacy,
            unknown: self.unknown,
            attempted_requested_bytes: self.bytes,
            root_attempts: self.roots,
            raw_attempt_reservations: self.raw,
            inflight: self.inflight,
            successful_roots: self.successful,
            cache_slot_history: self.cache_slots,
            published_cache_entries: self.published_cache,
            registered_devices: self.devices.len(),
            denied: self.denied,
        }
    }
}
