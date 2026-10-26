//! Optional immutable native-mediated requested-root history. Not physical admission.
mod live;
pub(crate) mod state;
pub use live::{LiveRootLimits, LiveRootSnapshot};
#[cfg(test)]
mod control_tests;
#[cfg(test)]
mod live_control_tests;
#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod tests;
use crate::{DType, MlxBuffer, MlxDevice, Result};
use state::{fail, Charge, DeviceKey, State};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Caller-selected historical ceilings. No resource drop refunds these limits.
#[derive(Clone, Copy, Debug)]
pub struct RootCreationLimits {
    pub max_attempted_requested_bytes: usize,
    pub max_root_attempts: usize,
    pub max_raw_constructor_attempts: usize,
    pub max_registered_devices: usize,
    pub max_rope_cache_entries: usize,
}
/// Requested history and qualification state; never a physical-memory promise.
#[derive(Clone, Debug)]
pub struct RootCreationSnapshot {
    pub scope: &'static str,
    pub limits: Option<RootCreationLimits>,
    pub covered: bool,
    pub legacy_used: bool,
    pub unknown: Option<&'static str>,
    pub attempted_requested_bytes: usize,
    pub root_attempts: usize,
    pub raw_attempt_reservations: usize,
    pub inflight: usize,
    pub successful_roots: usize,
    pub cache_slot_history: usize,
    pub published_cache_entries: usize,
    pub registered_devices: usize,
    pub denied: usize,
}
/// Informational lineage only; snapshot data cannot mint a creation certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootOriginSnapshot {
    Legacy,
    UnknownImport,
    KnownNative { id: usize, requested_bytes: usize },
}
#[derive(Clone)]
pub(crate) enum Origin {
    Legacy,
    Unknown,
    Known(Arc<Charge>),
}
#[derive(Clone)]
pub(crate) struct KnownRoot {
    metal: metal::Buffer,
    origin: Origin,
}
impl KnownRoot {
    pub(crate) fn metal(&self) -> &metal::Buffer {
        &self.metal
    }
    pub(crate) fn into_parts(self) -> (metal::Buffer, Origin) {
        (self.metal, self.origin)
    }
    pub(crate) fn from_parts(metal: metal::Buffer, origin: Origin) -> Self {
        Self { metal, origin }
    }
    pub(crate) fn check(&self) -> Result<()> {
        global().core.health()?;
        if global().core.installed()? {
            match &self.origin {
                Origin::Known(charge)
                    if self.metal.length() as usize == charge.bytes
                        && !self.metal.contents().is_null()
                        && self.metal.storage_mode() == metal::MTLStorageMode::Shared =>
                {
                    Ok(())
                }
                _ => {
                    global()
                        .core
                        .unknown("unknown origin or invalid returned resource");
                    Err(fail("known native resource required"))
                }
            }
        } else {
            Ok(())
        }
    }
    pub(crate) fn check_device(&self, device: &metal::DeviceRef) -> Result<()> {
        self.check()?;
        if let Origin::Known(charge) = &self.origin {
            if charge.device != device_key(device) {
                return Err(fail("known root device mismatch"));
            }
        }
        Ok(())
    }
    pub(crate) fn snapshot(&self) -> RootOriginSnapshot {
        match &self.origin {
            Origin::Legacy => RootOriginSnapshot::Legacy,
            Origin::Unknown => RootOriginSnapshot::UnknownImport,
            Origin::Known(c) => RootOriginSnapshot::KnownNative {
                id: c.id,
                requested_bytes: c.bytes,
            },
        }
    }
}
pub(crate) struct Core {
    state: Arc<Mutex<State>>,
    poisoned: Arc<AtomicBool>,
    // Published only AFTER Fresh->permanently legacy under the state mutex.
    legacy_fast: AtomicBool,
    #[cfg(test)]
    after_lock_precheck: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}
impl Default for Core {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(State::default())),
            poisoned: Arc::new(AtomicBool::new(false)),
            legacy_fast: AtomicBool::new(false),
            #[cfg(test)]
            after_lock_precheck: Mutex::new(None),
        }
    }
}
impl Core {
    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(fail("state poisoned"));
        }
        #[cfg(test)]
        if let Some(hook) = self.after_lock_precheck.lock().unwrap().take() {
            hook();
        }
        let guard = self.state.lock().map_err(|_| {
            self.poisoned.store(true, Ordering::Release);
            fail("state poisoned")
        })?;
        // A waiter can pass the fast precheck before another thread marks an
        // uncertain resource. Never publish its now-stale healthy guard.
        if self.poisoned.load(Ordering::Acquire) {
            return Err(fail("state poisoned"));
        }
        Ok(guard)
    }
    pub(crate) fn health(&self) -> Result<()> {
        if self.legacy_fast.load(Ordering::Acquire) {
            return Ok(());
        }
        self.lock()?.health()
    }
    fn installed(&self) -> Result<bool> {
        if self.legacy_fast.load(Ordering::Acquire) {
            return Ok(false);
        }
        Ok(self.lock()?.limits.is_some())
    }
    pub(crate) fn unknown(&self, reason: &'static str) {
        match self.lock() {
            Ok(mut s) => {
                s.unknown.get_or_insert(reason);
            }
            Err(_) => {
                self.poisoned.store(true, Ordering::Release);
            }
        }
    }
    #[cfg(test)]
    pub(crate) fn create<T>(
        &self,
        key: DeviceKey,
        bytes: usize,
        cache: bool,
        raw: impl FnOnce() -> Result<T>,
    ) -> Result<(T, Origin)> {
        self.create_with_origin(key, bytes, cache, |_| raw())
    }
    fn create_with_origin<T>(
        &self,
        key: DeviceKey,
        bytes: usize,
        cache: bool,
        raw: impl FnOnce(&Origin) -> Result<T>,
    ) -> Result<(T, Origin)> {
        if bytes == 0 {
            return Err(fail("zero creation request"));
        }
        if self.legacy_fast.load(Ordering::Acquire) {
            return raw(&Origin::Legacy).map(|value| (value, Origin::Legacy));
        }
        let (charge, live_enabled) = {
            let mut state = self.lock()?;
            let charge = state.reserve(key, bytes, cache)?;
            (charge, state.live.limits.is_some())
        };
        if charge.is_none() {
            self.legacy_fast.store(true, Ordering::Release);
        }
        // Allocate all certificate metadata BEFORE invoking the raw constructor.
        let origin = charge
            .map(|mut c| {
                if live_enabled {
                    c.live = Some(Arc::new(live::Ticket::new(self, bytes)));
                }
                Origin::Known(Arc::new(c))
            })
            .unwrap_or(Origin::Legacy);
        let mut attempt = Attempt {
            core: self,
            tracked: matches!(origin, Origin::Known(_)),
            closed: false,
            live: match &origin {
                Origin::Known(c) => c.live.clone(),
                _ => None,
            },
        };
        let result = raw(&origin);
        // Nonnil/raw success MUST remain Ok, even if diagnostic completion fails:
        // Native55 construct interprets every Err as no resource created.
        attempt.finish(result.is_ok(), false);
        result.map(|value| (value, origin))
    }
}
struct Attempt<'a> {
    core: &'a Core,
    tracked: bool,
    closed: bool,
    live: Option<Arc<live::Ticket>>,
}
impl Attempt<'_> {
    fn finish(&mut self, success: bool, abandoned: bool) {
        if self.tracked {
            match self.core.lock() {
                Ok(mut s) => s.finish(success, abandoned),
                Err(_) => self.core.poisoned.store(true, Ordering::Release),
            }
        }
        if let Some(ticket) = &self.live {
            if abandoned || self.core.poisoned.load(Ordering::Acquire) {
                ticket.uncertain();
            } else {
                ticket.finish(success);
            }
        }
        self.closed = true;
    }
}
impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        if !self.closed {
            self.finish(false, true);
        }
    }
}
// A successful raw creation has already committed Native55's local charge.
// Postcreation publication failure poisons coverage without a constructor Err.
pub(crate) struct PublicationGuard<'a> {
    core: &'a Core,
    tracked: bool,
    done: bool,
}
impl<'a> PublicationGuard<'a> {
    fn new(core: &'a Core) -> Self {
        Self {
            core,
            tracked: core.installed().unwrap_or(true),
            done: false,
        }
    }
    pub(crate) fn complete(mut self) {
        self.done = true;
    }
}
impl Drop for PublicationGuard<'_> {
    fn drop(&mut self) {
        if self.tracked && !self.done {
            self.core
                .unknown("postcreation publication failed or unwound");
        }
    }
}
pub(crate) fn publication_guard() -> PublicationGuard<'static> {
    PublicationGuard::new(&global().core)
}

struct Global {
    core: Core,
    devices: Mutex<Vec<metal::Device>>,
}
fn global() -> &'static Global {
    static G: OnceLock<Global> = OnceLock::new();
    G.get_or_init(|| Global {
        core: Core::default(),
        devices: Mutex::new(Vec::new()),
    })
}
fn device_key(device: &metal::DeviceRef) -> DeviceKey {
    DeviceKey {
        object: device as *const metal::DeviceRef as usize,
        registry: crate::device::metal_device_registry_id(device),
    }
}
/// Installs only before any native root/heap creation or unknown import.
pub struct NativeRootAuthority;
#[derive(Clone, Copy)]
pub struct RootAuthorityHandle {
    _private: (),
}
impl NativeRootAuthority {
    pub fn install_early(limits: RootCreationLimits) -> Result<RootAuthorityHandle> {
        Self::install(limits, None)
    }
    /// Optional immutable live+pending requested extents. Not physical admission.
    /// Association capability is restricted to observed Apple M5 Max/macOS27.2
    /// AGXG17XFamilyBuffer/build26B5112a; unknown actual classes poison coverage after raw success.
    pub fn install_early_with_live(
        limits: RootCreationLimits,
        live: LiveRootLimits,
    ) -> Result<RootAuthorityHandle> {
        if live.max_live_and_pending_requested_bytes == 0 || live.max_live_and_pending_roots == 0 {
            return Err(fail("live limits must be nonzero"));
        }
        Self::install(limits, Some(live))
    }
    fn install(
        limits: RootCreationLimits,
        live: Option<LiveRootLimits>,
    ) -> Result<RootAuthorityHandle> {
        let g = global();
        let mut devices = g
            .devices
            .lock()
            .map_err(|_| fail("device registry poisoned"))?;
        let mut state = g.core.lock()?;
        if state.limits.is_some() || state.legacy || state.unknown.is_some() {
            return Err(fail(
                "installation requires fresh native creation/import history",
            ));
        }
        if limits.max_registered_devices == 0 {
            return Err(fail("device limit must be nonzero"));
        }
        devices
            .try_reserve_exact(limits.max_registered_devices)
            .map_err(|_| fail("device handle reservation failed"))?;
        state.install(limits)?;
        state.live.limits = live;
        Ok(RootAuthorityHandle { _private: () })
    }
}
impl RootAuthorityHandle {
    /// Current component accounting only; unavailable on poison. No OS/physical proof.
    pub fn live_snapshot(&self) -> Result<LiveRootSnapshot> {
        global().core.live_snapshot()
    }

    /// Retains the actual device. Different queues share this process history.
    pub fn register_device(&self, device: &MlxDevice) -> Result<()> {
        let g = global();
        let mut owners = g
            .devices
            .lock()
            .map_err(|_| fail("device registry poisoned"))?;
        let key = device_key(device.metal_device());
        // Retain outside the global state lock; publication below is allocation-free.
        let retained = device.metal_device().to_owned();
        let mut s = g.core.lock()?;
        if s.devices.contains(&key) {
            return s.health();
        }
        // Registry capacity was preallocated before installation; no buffer creation.
        s.register(key)?;
        owners.push(retained);
        Ok(())
    }
    pub fn snapshot(&self) -> Result<RootCreationSnapshot> {
        Ok(global().core.lock()?.snapshot())
    }
}
pub(crate) fn import_origin() -> Origin {
    let core = &global().core;
    if core.legacy_fast.load(Ordering::Acquire) {
        return Origin::Legacy;
    }
    match core.lock() {
        Ok(mut s) => {
            s.import();
            if s.limits.is_some() {
                Origin::Unknown
            } else {
                core.legacy_fast.store(true, Ordering::Release);
                Origin::Legacy
            }
        }
        Err(_) => Origin::Unknown,
    }
}
pub(crate) fn require_no_heap() -> Result<()> {
    let core = &global().core;
    if core.legacy_fast.load(Ordering::Acquire) {
        return Ok(());
    }
    let mut state = core.lock()?;
    let result = state.heap();
    if state.limits.is_none() && state.legacy {
        core.legacy_fast.store(true, Ordering::Release);
    }
    result
}

pub(crate) fn shared_with(
    device: &metal::DeviceRef,
    bytes: usize,
    cache: bool,
    raw: impl FnOnce() -> Result<metal::Buffer>,
) -> Result<KnownRoot> {
    global()
        .core
        .create_with_origin(device_key(device), bytes, cache, |origin| {
            let ticket = match origin {
                Origin::Known(c) => c.live.as_ref(),
                _ => None,
            };
            live::create_attached(device, ticket, raw)
        })
        .map(|(metal, origin)| KnownRoot { metal, origin })
}
pub(crate) fn healthy() -> Result<()> {
    global().core.health()
}
pub(crate) fn cache_published() {
    if global().core.legacy_fast.load(Ordering::Acquire) {
        return;
    }
    match global().core.lock() {
        Ok(mut s) => {
            if s.limits.is_some() {
                s.published_cache = s.published_cache.saturating_add(1);
            }
        }
        Err(_) => global().core.poisoned.store(true, Ordering::Release),
    }
}
/// Exact initialized Shared upload. No kernel, queue creation or conversion.
pub(crate) fn upload_on_device(
    device: &metal::DeviceRef,
    data: &[u8],
    dtype: DType,
    shape: Vec<usize>,
    cache: bool,
) -> Result<MlxBuffer> {
    let expected = shape
        .iter()
        .try_fold(dtype.size_of(), |n, d| n.checked_mul(*d))
        .ok_or_else(|| fail("upload shape byte overflow"))?;
    if expected != data.len() {
        return Err(fail("upload shape/byte mismatch"));
    }
    if data.is_empty() {
        return Err(fail("empty upload"));
    }
    let root = crate::device::new_shared_on_device(device, data.len(), cache)?;
    let publication = publication_guard();
    root.check_device(device)?;
    // Exclusive unpublished root, checked extent and nonnull before CPU write.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            root.metal().contents().cast::<u8>(),
            data.len(),
        );
    }
    let buffer = MlxBuffer::from_known(root, dtype, shape)?;
    publication.complete();
    Ok(buffer)
}
