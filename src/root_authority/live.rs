//! Optional requested extents only; actual resource lifetime, never physical RAM.
use super::{
    state::{fail, State},
    Core,
};
use crate::Result;
use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

#[derive(Clone, Copy, Debug)]
pub struct LiveRootLimits {
    pub max_live_and_pending_requested_bytes: usize,
    pub max_live_and_pending_roots: usize,
}
#[derive(Clone, Debug)]
pub struct LiveRootSnapshot {
    pub scope: &'static str,
    pub limits: Option<LiveRootLimits>,
    pub covered: bool,
    pub unknown: Option<&'static str>,
    pub pending_requested_bytes: usize,
    pub live_requested_bytes: usize,
    pub pending_roots: usize,
    pub live_roots: usize,
    pub high_water_requested_bytes: usize,
    pub denied: usize,
    pub final_releases: usize,
}
#[derive(Default, Debug)]
pub(crate) struct Counters {
    pub limits: Option<LiveRootLimits>,
    pub pending_bytes: usize,
    pub live_bytes: usize,
    pub pending_roots: usize,
    pub live_roots: usize,
    pub high_water: usize,
    pub denied: usize,
    pub releases: usize,
}
impl Counters {
    pub fn reserve(&mut self, bytes: usize) -> Result<()> {
        let Some(l) = self.limits else {
            return Ok(());
        };
        let total = self
            .pending_bytes
            .checked_add(self.live_bytes)
            .and_then(|n| n.checked_add(bytes));
        let roots = self
            .pending_roots
            .checked_add(self.live_roots)
            .and_then(|n| n.checked_add(1));
        if total.is_none_or(|n| n > l.max_live_and_pending_requested_bytes)
            || roots.is_none_or(|n| n > l.max_live_and_pending_roots)
        {
            self.denied = self.denied.saturating_add(1);
            return Err(fail("live requested quota exhausted or overflow"));
        }
        self.pending_bytes += bytes;
        self.pending_roots += 1;
        self.high_water = self.high_water.max(total.unwrap());
        Ok(())
    }
    pub fn finish(&mut self, bytes: usize, success: bool) -> bool {
        let (Some(b), Some(r)) = (
            self.pending_bytes.checked_sub(bytes),
            self.pending_roots.checked_sub(1),
        ) else {
            return false;
        };
        if success {
            let (Some(lb), Some(lr)) = (
                self.live_bytes.checked_add(bytes),
                self.live_roots.checked_add(1),
            ) else {
                return false;
            };
            self.live_bytes = lb;
            self.live_roots = lr;
        }
        self.pending_bytes = b;
        self.pending_roots = r;
        true
    }
    pub fn release(&mut self, bytes: usize) -> bool {
        let (Some(b), Some(r)) = (
            self.live_bytes.checked_sub(bytes),
            self.live_roots.checked_sub(1),
        ) else {
            return false;
        };
        self.live_bytes = b;
        self.live_roots = r;
        self.releases = self.releases.saturating_add(1);
        true
    }
}
const PENDING: u8 = 0;
const LIVE: u8 = 1;
const CANCELLED: u8 = 2;
const UNCERTAIN: u8 = 3;
#[derive(Debug)]
pub(crate) struct Ticket {
    state: Weak<Mutex<State>>,
    poison: Weak<AtomicBool>,
    bytes: usize,
    phase: AtomicU8,
}
impl Ticket {
    pub(super) fn new(core: &Core, bytes: usize) -> Self {
        Self {
            state: Arc::downgrade(&core.state),
            poison: Arc::downgrade(&core.poisoned),
            bytes,
            phase: AtomicU8::new(PENDING),
        }
    }
    fn poison(&self) {
        if let Some(p) = self.poison.upgrade() {
            p.store(true, Ordering::Release);
        }
    }
    pub(super) fn uncertain(&self) {
        self.phase.store(UNCERTAIN, Ordering::Release);
        self.poison();
    }
    pub(super) fn finish(&self, success: bool) {
        if self.phase.load(Ordering::Acquire) != PENDING {
            return;
        }
        let Some(state) = self.state.upgrade() else {
            self.uncertain();
            return;
        };
        let Ok(mut s) = state.lock() else {
            self.uncertain();
            return;
        };
        if s.unknown.is_some()
            || self.phase.load(Ordering::Acquire) != PENDING
            || self
                .poison
                .upgrade()
                .is_some_and(|p| p.load(Ordering::Acquire))
        {
            self.uncertain();
            return;
        }
        if !s.live.finish(self.bytes, success) {
            self.uncertain();
            return;
        }
        if self
            .phase
            .compare_exchange(
                PENDING,
                if success { LIVE } else { CANCELLED },
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            self.uncertain();
        }
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        match self.phase.load(Ordering::Acquire) {
            LIVE => {
                if self
                    .poison
                    .upgrade()
                    .is_some_and(|p| p.load(Ordering::Acquire))
                {
                    return;
                }
                let Some(state) = self.state.upgrade() else {
                    self.poison();
                    return;
                };
                match state.lock() {
                    Ok(mut s) => {
                        if s.unknown.is_some() {
                            return;
                        }
                        if !s.live.release(self.bytes) {
                            self.poison();
                        }
                    }
                    Err(_) => self.poison(),
                };
            }
            PENDING => self.poison(), // unwind/uncertain raw outcome: retain charge
            _ => {}
        }
    }
}

static KEY: u8 = 0;
#[link(name = "objc")]
extern "C" {
    fn objc_setAssociatedObject(
        object: *mut objc::runtime::Object,
        key: *const std::ffi::c_void,
        value: *mut objc::runtime::Object,
        policy: usize,
    );
    fn objc_getAssociatedObject(
        object: *mut objc::runtime::Object,
        key: *const std::ffi::c_void,
    ) -> *mut objc::runtime::Object;
    fn sysctlbyname(
        name: *const std::os::raw::c_char,
        old: *mut std::ffi::c_void,
        len: *mut usize,
        new: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::os::raw::c_int;
}
pub(super) fn eligible(class: &str, device: &str, os: &[u8], build: &[u8]) -> bool {
    class == "AGXG17XFamilyBuffer"
        && device == "Apple M5 Max"
        && os == b"27.2\0"
        && build == b"26B5112a\0"
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostBytes {
    product: [u8; 64],
    build: [u8; 64],
    product_len: usize,
    build_len: usize,
}
impl HostBytes {
    pub(super) fn from_slices(product: &[u8], build: &[u8]) -> Option<Self> {
        if product.is_empty() || product.len() > 64 || build.is_empty() || build.len() > 64 {
            return None;
        }
        let mut out = Self {
            product: [0; 64],
            build: [0; 64],
            product_len: product.len(),
            build_len: build.len(),
        };
        out.product[..product.len()].copy_from_slice(product);
        out.build[..build.len()].copy_from_slice(build);
        Some(out)
    }
    pub(super) fn allows(&self, class: &str, device: &str) -> bool {
        eligible(
            class,
            device,
            &self.product[..self.product_len],
            &self.build[..self.build_len],
        )
    }
}
// Same helper used by production and local-cache pure controls. Cache failures
// too: an unavailable initial query can never heal into eligibility midprocess.
pub(super) fn cached_host(
    cache: &OnceLock<Option<HostBytes>>,
    query: impl FnOnce() -> Option<HostBytes>,
) -> Option<&HostBytes> {
    cache.get_or_init(query).as_ref()
}
fn query_host() -> Option<HostBytes> {
    let query = |name: &[u8], out: &mut [u8; 64]| -> Option<usize> {
        let mut len = out.len();
        let status = unsafe {
            sysctlbyname(
                name.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && len <= out.len()).then_some(len)
    };
    let mut os = [0u8; 64];
    let mut build = [0u8; 64];
    let (Some(n), Some(b)) = (
        query(b"kern.osproductversion\0", &mut os),
        query(b"kern.osversion\0", &mut build),
    ) else {
        return None;
    };
    HostBytes::from_slices(&os[..n], &build[..b])
}
fn known_platform(resource: &metal::BufferRef, device: &metal::DeviceRef) -> bool {
    use metal::foreign_types::ForeignTypeRef;
    // This is outside StateMutex, after raw success; class/device are checked
    // on EVERY actual resource. Only immutable boot OS identity is cached.
    static HOST: OnceLock<Option<HostBytes>> = OnceLock::new();
    let class = unsafe {
        (&*resource.as_ptr().cast::<objc::runtime::Object>())
            .class()
            .name()
    };
    cached_host(&HOST, query_host).is_some_and(|host| host.allows(class, device.name()))
}

pub(super) fn attach_checked(ticket: &Ticket, eligible: bool, attach: impl FnOnce() -> bool) {
    if !eligible || !attach() {
        ticket.uncertain();
    }
}
/// Owner metadata is copied before raw construction. No Metal object captured.
pub(super) fn create_attached(
    device: &metal::DeviceRef,
    ticket: Option<&Arc<Ticket>>,
    raw: impl FnOnce() -> Result<metal::Buffer>,
) -> Result<metal::Buffer> {
    let Some(ticket) = ticket else {
        return raw();
    };
    let capture = Arc::clone(ticket);
    let owner = block::ConcreteBlock::new(move || {
        let _ = Arc::strong_count(&capture);
    })
    .copy();
    let result = raw();
    if let Ok(resource) = &result {
        use metal::foreign_types::ForeignType;
        let allowed = known_platform(resource, device);
        attach_checked(ticket, allowed, || {
            let object = resource.as_ptr().cast();
            let key = std::ptr::addr_of!(KEY).cast();
            let ptr = &*owner as *const _ as *mut objc::runtime::Object;
            // ObjC exceptions/fatal unsupported classes are not Rust errors.
            unsafe {
                if !objc_getAssociatedObject(object, key).is_null() {
                    return false;
                }
                objc_setAssociatedObject(object, key, ptr, 1);
                objc_getAssociatedObject(object, key) == ptr
            }
        });
    }
    result // NEVER turn nonnil creation into Err after association failure
}

impl Core {
    pub(super) fn live_snapshot(&self) -> Result<LiveRootSnapshot> {
        let s = self.lock()?;
        let l = &s.live;
        Ok(LiveRootSnapshot {
            scope: "native_live_and_pending_requested_resource_extents",
            limits: l.limits,
            covered: l.limits.is_some() && s.unknown.is_none(),
            unknown: s.unknown,
            pending_requested_bytes: l.pending_bytes,
            live_requested_bytes: l.live_bytes,
            pending_roots: l.pending_roots,
            live_roots: l.live_roots,
            high_water_requested_bytes: l.high_water,
            denied: l.denied,
            final_releases: l.releases,
        })
    }
}
