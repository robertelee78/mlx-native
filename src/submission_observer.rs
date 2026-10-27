//! Optional, bounded command observation. No scheduling or model interpretation.
//! Install before the first native device/encoder, then explicitly enable each
//! observed encoder with a fresh invocation epoch. Unobserved work is unchanged.
//! Installing is NOT a completeness certificate: the caller must freeze an
//! unobserved_commits baseline/end snapshot for its measured window and refuse
//! nonzero deltas, dropped/invalid/tainted metadata or required Unknown scopes.
//! Dual-buffer child encoders are not automatically opted in. Such a route is
//! explicitly unqualified unless the caller binds every child; never disable a
//! selected runtime route merely to manufacture a successful observation.
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationScope { pub epoch: u64, pub id: u64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ObservationKind { Created, DispatchEncoded, NonComputeEncoded, Submitted, CheckedDrain, Dropped }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationClock { #[serde(rename="CLOCK_MONOTONIC")] Monotonic }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchGeometry {
    /// Process-local identity only, not shader source identity or logical rows.
    pub pipeline_identity: u64,
    pub dispatch_kind: crate::DispatchKind,
    pub grid: [u64; 3],
    pub threads_per_threadgroup: [u64; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SubmissionEvent {
    pub clock_id: ObservationClock,
    pub command_id: u64,
    pub invocation_epoch: u64,
    pub queue_identity: u64,
    pub device_registry_id: u64,
    pub kind: ObservationKind,
    pub begin_monotonic_ns: u64,
    pub end_monotonic_ns: u64,
    pub dispatch_ordinal: u64,
    pub noncompute_operations: u64,
    pub scope: Option<ObservationScope>,
    pub captured_scope: Option<ObservationScope>,
    pub capture_command_id: Option<u64>,
    pub capture_generation: Option<u64>,
    pub replay_binding: ReplayBinding,
    pub tainted: bool,
    pub geometry: Option<DispatchGeometry>,
    pub submitted: bool,
    pub checked_status: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayBinding { Direct, ImmediateCapture, ExplicitRemap, Unknown }

#[derive(Clone, Copy, Debug)]
pub struct ObserverSinkError;

/// Called outside native resource locks. Implementations must be bounded and
/// must not reenter native observation or acquire a blocking lock: use try_lock
/// and return an error on contention. Error/panic invalidates observation,
/// never changes the GPU dispatch or its returned completion status.
pub trait SubmissionObserver: Send + Sync + 'static {
    fn record(&self, event: &SubmissionEvent) -> Result<(), ObserverSinkError>;
}

#[derive(Clone, Copy, Debug, Serialize)]
#[non_exhaustive]
pub struct ObservationStatus {
    pub installed: bool,
    pub invalid: bool,
    pub events: u64,
    pub unknown_scopes: u64,
    pub event_limit: u64,
    pub metadata_reserved_bytes: u64,
    pub metadata_limit_bytes: u64,
    pub dropped_events: u64,
    pub unobserved_commits: u64,
    pub capture_valid: bool,
    /// Always false: native metadata alone cannot certify HF rows or quality.
    pub comparison_qualified: bool,
}

struct Core {
    sink: Arc<dyn SubmissionObserver>,
    limit: u64,
    events: AtomicU64,
    next_id: AtomicU64,
    invalid: AtomicBool,
    unknown: AtomicU64,
    metadata: AtomicU64,
    metadata_limit: u64,
    clock: fn() -> Option<u64>,
    dropped: AtomicU64,
    unobserved: AtomicU64,
    next_capture: AtomicU64,
}
// 0 fresh,1 installing,2 installed,3 started without observer,4 started with observer.
static INSTALL: AtomicU8 = AtomicU8::new(0);
static CORE: OnceLock<Arc<Core>> = OnceLock::new();
thread_local! { static IN_SINK: Cell<bool> = const { Cell::new(false) }; }

pub fn install_submission_observer(sink: Arc<dyn SubmissionObserver>, event_limit: u64,
    metadata_limit_bytes: u64)
    -> Result<(), &'static str>
{
    if cfg!(panic="abort") {return Err("observer panic containment requires panic=unwind");}
    if event_limit == 0 || event_limit > 1_000_000 { return Err("observer event bound must be 1..=1000000"); }
    if metadata_limit_bytes == 0 || metadata_limit_bytes > 16 * 1024 * 1024 {
        return Err("observer metadata quota must be 1..=16MiB");
    }
    begin_install(&INSTALL)?;
    let core = Arc::new(Core::new(sink, event_limit, metadata_limit_bytes));
    if CORE.set(core).is_err() { INSTALL.store(3, Ordering::Release); return Err("observer already installed"); }
    finish_install(&INSTALL)?;
    Ok(())
}

pub fn submission_observation_status() -> ObservationStatus {
    match CORE.get() {
        Some(c) => ObservationStatus { installed:matches!(INSTALL.load(Ordering::Acquire),2|4),
            invalid:c.invalid.load(Ordering::Acquire), events:c.events.load(Ordering::Acquire),
            unknown_scopes:c.unknown.load(Ordering::Acquire), event_limit:c.limit,
            metadata_reserved_bytes:c.metadata.load(Ordering::Acquire),metadata_limit_bytes:c.metadata_limit,
            dropped_events:c.dropped.load(Ordering::Acquire),unobserved_commits:c.unobserved.load(Ordering::Acquire),
            capture_valid:matches!(INSTALL.load(Ordering::Acquire),2|4)&&!c.invalid.load(Ordering::Acquire),comparison_qualified:false },
        None => ObservationStatus { installed:false,invalid:false,events:0,unknown_scopes:0,event_limit:0,
            metadata_reserved_bytes:0,metadata_limit_bytes:0,dropped_events:0,unobserved_commits:0,
            capture_valid:false,comparison_qualified:false },
    }
}

/// Includes EVERY primitive commit while installed, even an encoder not opted
/// into scoped observation. HF must bind before/after counters around its
/// measured window; a nonzero delta is Unknown, never inferred "no work".
pub(crate) fn unobserved_commit() {
    if !matches!(INSTALL.load(Ordering::Acquire),2|4) {return;}
    if let Some(core)=CORE.get() {
        core.note_unobserved();
    }
}

pub(crate) fn native_creation() {
    mark_started(&INSTALL);
}
fn begin_install(state:&AtomicU8)->Result<(),&'static str> {
    state.compare_exchange(0,1,Ordering::AcqRel,Ordering::Acquire)
        .map(|_|()).map_err(|_|"observer installation must precede native device/encoder creation")
}
fn finish_install(state:&AtomicU8)->Result<(),&'static str> {
    state.compare_exchange(1,2,Ordering::AcqRel,Ordering::Acquire)
        .map(|_|()).map_err(|_|"native creation raced observer installation; observation unavailable")
}
fn mark_started(state:&AtomicU8) {
    loop {
        let old = state.load(Ordering::Acquire);
        let new = match old { 0|1 => 3, 2 => 4, _ => return };
        if state.compare_exchange(old,new,Ordering::AcqRel,Ordering::Acquire).is_ok() { return; }
    }
}

impl Core {
    fn new(sink: Arc<dyn SubmissionObserver>, limit:u64, metadata_limit:u64) -> Self {
        Self { sink,limit,events:AtomicU64::new(0),next_id:AtomicU64::new(1),
            invalid:AtomicBool::new(false),unknown:AtomicU64::new(0),
            metadata:AtomicU64::new(0),metadata_limit,clock:clock_ns,dropped:AtomicU64::new(0),
            unobserved:AtomicU64::new(0),next_capture:AtomicU64::new(1) }
    }
    fn fail(&self) { self.invalid.store(true,Ordering::Release); }
    fn note_unobserved(&self) {
        if self.unobserved.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_add(1)).is_err() {self.fail();}
    }
    fn drop_event(&self) {
        if self.dropped.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_add(1)).is_err() {
            self.fail();
        }
    }
    fn emit(&self, event:SubmissionEvent) {
        if self.invalid.load(Ordering::Acquire) {self.drop_event();return;}
        if std::thread::panicking() {self.fail();self.drop_event();return;}
        if event.end_monotonic_ns < event.begin_monotonic_ns {
            self.fail();self.drop_event();return;
        }
        if self.events.fetch_update(Ordering::AcqRel,Ordering::Acquire,
            |n| n.checked_add(1).filter(|n| *n<=self.limit)).is_err() {self.fail();self.drop_event();return;}
        if IN_SINK.try_with(|flag| {
            if flag.replace(true) {self.fail();self.drop_event();return;}
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.record(&event)));
            flag.set(false);
            if !matches!(result,Ok(Ok(()))) {self.fail();self.drop_event();}
        }).is_err() {self.fail();self.drop_event();}
    }
    fn unknown(&self) {
        if self.unknown.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n| n.checked_add(1)).is_err() {
            self.fail();
        }
    }
}

const MAX_SCOPES: usize = 4096;
struct Reservation { core:Arc<Core>,bytes:u64 }
impl Drop for Reservation {
    fn drop(&mut self) { self.core.metadata.fetch_sub(self.bytes,Ordering::AcqRel); }
}
fn reserve(core:&Arc<Core>,bytes:usize)->Result<Reservation,&'static str> {
    let bytes=u64::try_from(bytes).map_err(|_|"observer metadata overflow")?;
    core.metadata.fetch_update(Ordering::AcqRel,Ordering::Acquire,
        |n|n.checked_add(bytes).filter(|n|*n<=core.metadata_limit))
        .map_err(|_| {core.fail();"observer metadata quota exhausted"})?;
    Ok(Reservation{core:core.clone(),bytes})
}

/// Private side attachment; public CapturedNode construction/layout unchanged.
pub(crate) struct CapturedScopes {
    tags:Vec<Option<ObservationScope>>,
    owner:u64,
    generation:u64,
    used:AtomicBool,
    invalid_alignment:AtomicBool,
    _reservation:Reservation,
}
#[derive(Clone,Copy)]
pub(crate) struct ReplayPermit {pub(crate) automatic:bool,pub(crate) owner:u64,pub(crate) generation:u64}
impl CapturedScopes {
    pub(crate) fn push(&mut self,scope:Option<ObservationScope>) {
        if self.tags.len()==MAX_SCOPES {
            self.invalid_alignment.store(true,Ordering::Release);self._reservation.core.fail();return;
        }
        self.tags.push(scope);
    }
    pub(crate) fn unknown(&self) { self._reservation.core.unknown(); }
    pub(crate) fn require_observed(&self,enabled:bool) { if !enabled {self.unknown();} }
    pub(crate) fn get(&self,index:usize)->Option<ObservationScope> {
        if self.invalid_alignment.load(Ordering::Acquire) {return None;}
        self.tags.get(index).copied().flatten()
    }
    pub(crate) fn aligned(&self,len:usize)->bool {
        if len!=self.tags.len() {
            self.invalid_alignment.store(true,Ordering::Release);self._reservation.core.fail();
        }
        !self.invalid_alignment.load(Ordering::Acquire)
    }
    pub(crate) fn claim(&self,command:Option<u64>,len:usize)->ReplayPermit {
        let first=!self.used.swap(true,Ordering::AcqRel);
        let automatic=self.aligned(len)&&first&&command==Some(self.owner);
        ReplayPermit{automatic,owner:self.owner,generation:self.generation}
    }
    pub(crate) fn permute(&mut self,indices:&[usize]) {
        if indices.len()>MAX_SCOPES || indices.iter().any(|i|*i>=self.tags.len()) {
            self.invalid_alignment.store(true,Ordering::Release);
            self._reservation.core.fail();self.tags.clear();return;
        }
        // Reservation covers both old and new bounded tag vectors at once.
        self.tags=indices.iter().map(|&i|self.get(i)).collect();
    }
    pub(crate) fn transformed(&mut self,new_len:usize) {
        let common=self.tags.first().copied().flatten();
        let same=common.is_some()&&self.tags.iter().all(|s|*s==common);
        if new_len>MAX_SCOPES {self._reservation.core.fail();self.tags.clear();return;}
        // Mixed scopes cannot be guessed through fusion: actual replay is Unknown.
        self.tags=vec![if same {common}else{None};new_len];
    }
}

/// macOS CLOCK_MONOTONIC epoch, shared with the host's clock_gettime consumer.
pub(crate) fn clock_ns() -> Option<u64> {
    #[cfg(target_os="macos")]
    {
        #[repr(C)] struct Timespec { sec:std::os::raw::c_long,nsec:std::os::raw::c_long }
        extern "C" { fn clock_gettime(clock:std::os::raw::c_int,t:*mut Timespec)->std::os::raw::c_int; }
        let mut t = std::mem::MaybeUninit::<Timespec>::uninit();
        // SAFETY: Darwin CLOCK_MONOTONIC=6; correctly aligned writable timespec.
        if unsafe { clock_gettime(6,t.as_mut_ptr()) } != 0 { return None; }
        let t = unsafe { t.assume_init() };
        u64::try_from(t.sec).ok()?.checked_mul(1_000_000_000)?
            .checked_add(u64::try_from(t.nsec).ok().filter(|n| *n<1_000_000_000)?)
    }
    #[cfg(not(target_os="macos"))]
    { None }
}

pub(crate) struct CommandObservation {
    core: Arc<Core>,
    id:u64, queue:u64, device:u64,
    pub(crate) epoch:u64,
    pub(crate) scope:Option<ObservationScope>,
    captured_scope:Option<ObservationScope>,
    geometry:Option<DispatchGeometry>,
    pending_replay:Option<Option<ObservationScope>>,
    replay_binding:ReplayBinding,
    capture_command_id:Option<u64>,
    capture_generation:Option<u64>,
    tainted:AtomicBool,
    pub(crate) replay:Vec<(ObservationScope,ObservationScope)>,
    replay_reservation:Option<Reservation>,
    ordinal:u64,
    noncompute:AtomicU64,
    submitted:AtomicBool,
    checked:AtomicU64,
    _reservation:Reservation,
}

impl CommandObservation {
    pub(crate) fn create(queue:u64,device:u64,epoch:u64)->Result<Self,&'static str> {
        if !matches!(INSTALL.load(Ordering::Acquire),2|4) || epoch==0 {
            return Err("installed observer and nonzero fresh invocation epoch required");
        }
        let core=CORE.get().ok_or("observer unavailable")?.clone();
        Self::with_core(core,queue,device,epoch)
    }
    fn with_core(core:Arc<Core>,queue:u64,device:u64,epoch:u64)->Result<Self,&'static str> {
        let reservation=reserve(&core,std::mem::size_of::<Self>())?;
        let id=core.next_id.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_add(1))
            .map_err(|_| {core.fail();"command ID exhausted"})?;
        let this=Self {core,id,queue,device,epoch,scope:None,captured_scope:None,geometry:None,
            pending_replay:None,replay_binding:ReplayBinding::Direct,capture_command_id:None,
            capture_generation:None,tainted:AtomicBool::new(false),replay:Vec::new(),replay_reservation:None,ordinal:0,noncompute:AtomicU64::new(0),
            submitted:AtomicBool::new(false),checked:AtomicU64::new(u64::MAX),_reservation:reservation};
        this.event(ObservationKind::Created,None,None);Ok(this)
    }
    fn event(&self,kind:ObservationKind,bracket:Option<(u64,u64)>,status:Option<u64>) {
        let Some((begin,end))=bracket.or_else(||(self.core.clock)().map(|t|(t,t))) else {self.core.fail();self.core.drop_event();return;};
        self.core.emit(SubmissionEvent {clock_id:ObservationClock::Monotonic,command_id:self.id,
            invocation_epoch:self.epoch,queue_identity:self.queue,
            device_registry_id:self.device,kind,begin_monotonic_ns:begin,end_monotonic_ns:end,
            dispatch_ordinal:self.ordinal,noncompute_operations:self.noncompute.load(Ordering::Acquire),
            scope:if kind==ObservationKind::DispatchEncoded {self.scope}else{None},
            captured_scope:if kind==ObservationKind::DispatchEncoded {self.captured_scope}else{None},
            capture_command_id:if kind==ObservationKind::DispatchEncoded {self.capture_command_id}else{None},
            capture_generation:if kind==ObservationKind::DispatchEncoded {self.capture_generation}else{None},
            replay_binding:if kind==ObservationKind::DispatchEncoded {self.replay_binding}else{ReplayBinding::Direct},
            tainted:self.tainted.load(Ordering::Acquire),
            geometry:if kind==ObservationKind::DispatchEncoded {self.geometry}else{None},
            submitted:self.submitted.load(Ordering::Acquire),
            checked_status:status});
    }
    pub(crate) fn invalid(&self) { self.core.fail(); }
    pub(crate) fn unknown(&self) { self.core.unknown(); }
    pub(crate) fn raw_escape(&self) {
        if !self.tainted.swap(true,Ordering::AcqRel) {self.core.unknown();}
    }
    pub(crate) fn id(&self)->u64 {self.id}
    pub(crate) fn dispatch(&mut self,geometry:DispatchGeometry) {
        if self.submitted.load(Ordering::Acquire) {self.core.fail();return;}
        let Some(next)=self.ordinal.checked_add(1) else {self.core.fail();return;};
        self.ordinal=next;
        self.geometry=Some(geometry);
        let explicit_scope=self.scope;
        if let Some(replay)=self.pending_replay.take() {self.scope=replay;}
        if self.scope.is_none() {self.core.unknown();}
        self.event(ObservationKind::DispatchEncoded,None,None);
        self.scope=explicit_scope;
        self.captured_scope=None;self.capture_command_id=None;self.capture_generation=None;
        self.replay_binding=ReplayBinding::Direct;
    }
    pub(crate) fn noncompute(&self) {
        if self.noncompute.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_add(1)).is_err() {
            self.core.fail();return;
        }
        self.event(ObservationKind::NonComputeEncoded,None,None);
    }
    pub(crate) fn submitted(&self,begin:Option<u64>,end:Option<u64>) {
        if self.submitted.swap(true,Ordering::AcqRel) {self.core.fail();return;}
        match (begin,end) { (Some(a),Some(b))=>self.event(ObservationKind::Submitted,Some((a,b)),None),
            _=>self.core.fail() }
    }
    pub(crate) fn drained(&self,begin:Option<u64>,end:Option<u64>,status:u64) {
        if !self.submitted.load(Ordering::Acquire) {self.core.fail();return;}
        match self.checked.compare_exchange(u64::MAX,status,Ordering::AcqRel,Ordering::Acquire) {
            Ok(_)=>match (begin,end) { (Some(a),Some(b))=>self.event(ObservationKind::CheckedDrain,Some((a,b)),Some(status)), _=>self.core.fail() },
            Err(previous) if previous!=status=>self.core.fail(), _=>{}
        }
    }
    pub(crate) fn bind_replay(&mut self,old:Option<ObservationScope>,permit:Option<ReplayPermit>) {
        self.captured_scope=old;
        self.capture_command_id=permit.map(|p|p.owner);self.capture_generation=permit.map(|p|p.generation);
        self.replay_binding=ReplayBinding::Unknown;
        let bound=old.and_then(|s| {
            if let Ok(index)=self.replay.binary_search_by_key(&(s.epoch,s.id),|(from,_)|(from.epoch,from.id)) {
                self.replay_binding=ReplayBinding::ExplicitRemap;Some(self.replay[index].1)
            } else if permit.is_some_and(|p|p.automatic)&&s.epoch==self.epoch {
                self.replay_binding=ReplayBinding::ImmediateCapture;Some(s)
            } else {None}
        });
        self.pending_replay=Some(bound);
    }
    pub(crate) fn rotate(&self)->Result<Self,&'static str> {
        Self::with_core(self.core.clone(),self.queue,self.device,self.epoch)
    }
    pub(crate) fn set_scope(&mut self,scope:Option<ObservationScope>) {
        self.captured_scope=None;
        self.pending_replay=None;self.capture_command_id=None;self.capture_generation=None;
        self.replay_binding=ReplayBinding::Direct;
        self.scope=scope.filter(|s|s.epoch==self.epoch);
        if scope.is_some() && self.scope.is_none() { self.core.unknown(); }
    }
    pub(crate) fn capture_scopes(&self)->Result<CapturedScopes,&'static str> {
        let reservation=reserve(&self.core,3*MAX_SCOPES*std::mem::size_of::<Option<ObservationScope>>())?;
        let generation=self.core.next_capture.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_add(1))
            .map_err(|_|{self.core.fail();"capture generation exhausted"})?;
        Ok(CapturedScopes{tags:Vec::with_capacity(MAX_SCOPES),owner:self.id,generation,
            used:AtomicBool::new(false),invalid_alignment:AtomicBool::new(false),_reservation:reservation})
    }
    pub(crate) fn replay_map(&mut self,map:&[(ObservationScope,ObservationScope)])
        ->Result<(),&'static str>
    {
        if map.len()>MAX_SCOPES || map.iter().any(|(_,to)|to.epoch!=self.epoch) {
            self.core.unknown();return Err("bounded current-epoch replay mapping required");
        }
        let reservation=reserve(&self.core,2*map.len()*std::mem::size_of::<(ObservationScope,ObservationScope)>())?;
        let mut sorted=map.to_vec();sorted.sort_unstable_by_key(|(from,_)|(from.epoch,from.id));
        if sorted.windows(2).any(|pair|pair[0].0==pair[1].0) {
            self.core.unknown();return Err("ambiguous replay scope mapping");
        }
        self.replay=sorted;self.replay_reservation=Some(reservation);Ok(())
    }
}

impl Drop for CommandObservation {
    fn drop(&mut self) {
        if self.submitted.load(Ordering::Acquire) && self.checked.load(Ordering::Acquire)==u64::MAX
            && (self.ordinal>0 || self.noncompute.load(Ordering::Acquire)>0 || self.tainted.load(Ordering::Acquire)) {
            self.core.unknown();
        }
        self.event(ObservationKind::Dropped,None,
            (self.checked.load(Ordering::Acquire)!=u64::MAX).then(||self.checked.load(Ordering::Acquire)));
    }
}

#[cfg(test)]
#[path="submission_observer_tests.rs"]
mod tests;
