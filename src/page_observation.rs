//! Bounded, read-only VM metadata. Not physical ownership, residency or release.
use crate::{BufferProvenance,MlxBuffer,RootOriginSnapshot};
use memmap2::Mmap;
use std::sync::OnceLock;
use std::time::Instant;
use thiserror::Error;
/// Per-call output/work ceiling; no caller can enlarge it.
pub const MAX_OBSERVATION_PAGES:u32=4096;
/// The call is not interruptible; a stale result is refused after it returns.
pub const MAX_OBSERVATION_ELAPSED_NS:u64=250_000_000;
/// SDK dispositions through WIRED. This is not a live-kernel equivalence claim.
pub const KNOWN_PAGE_DISPOSITION_BITS:i32=0x1fff;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct PageQueryRange {pub page_offset:u64,pub page_count:u32}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct PageQueryLimits {pub max_elapsed_ns:u64}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
#[non_exhaustive]
pub enum PageObservationSource {
    NoCopyRoot {root_id:usize,requested_bytes:usize,registered_base:usize,registered_bytes:usize},
    /// A readonly mapping is not native proof of its file/FD association.
    ReadonlyMapping {mapping_base:usize,mapping_bytes:usize},
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct PageDispositionCounts {
    pub present:u32,pub dirty:u32,pub wired:u32,pub copied:u32,pub paged_out:u32,
    /// Number of pages with any other recognized SDK bit, nonadditive.
    pub other_known:u32,
}
#[derive(Debug)]
pub struct PageObservation {
    pub source:PageObservationSource,pub range:PageQueryRange,pub page_bytes:usize,
    pub query_address:usize,pub queried_bytes:usize,pub raw_mach_status:i32,pub returned_count:u64,
    pub dispositions:Vec<i32>,pub counts:PageDispositionCounts,
    pub clock:&'static str,pub start_ns:u64,pub end_ns:u64,pub elapsed_ns:u64,
}
#[derive(Debug,Error)]
#[non_exhaustive]
pub enum PageObservationError {
    #[error("page metadata observation is unsupported on this platform")] UnsupportedPlatform,
    #[error("root has no certified readonly NoCopy backing")] NotCertifiedNoCopy,
    #[error("backing descriptor unavailable: {0}")] BackingUnavailable(crate::MlxError),
    #[error("invalid host page size")] InvalidPageSize,
    #[error("invalid, empty or out-of-bounds page range")] InvalidRange,
    #[error("page count exceeds bounded observation cap")] PageCap,
    #[error("address/span arithmetic overflow")] Overflow,
    #[error("invalid elapsed-time budget")] InvalidTimeBudget,
    #[error("bounded observation allocation failed")] Allocation,
    #[error("Mach query failed: status={status}, returned_count={returned_count}")] QueryFailed{status:i32,returned_count:u64},
    #[error("partial/invalid Mach count: expected={expected}, returned={returned}")] Partial{expected:u64,returned:u64},
    #[error("unknown disposition at page {index}: {raw}")] UnknownFlags{index:usize,raw:i32},
    #[error("query exceeded elapsed budget: actual={elapsed_ns}, maximum={maximum_ns}")] Stale{elapsed_ns:u64,maximum_ns:u64},
    #[error("process-local clock overflow")] ClockOverflow,
}
type QueryResult<T>=std::result::Result<T,PageObservationError>;
fn covering_bytes(bytes:usize,page:usize)->QueryResult<usize>{
    if !(4096..=65536).contains(&page)||!page.is_power_of_two(){return Err(PageObservationError::InvalidPageSize);}
    if bytes==0{return Err(PageObservationError::InvalidRange);}
    bytes.checked_add(page-1).map(|b|b/page*page).ok_or(PageObservationError::Overflow)
}
fn checked_range(base:usize,extent:usize,page:usize,r:PageQueryRange)->QueryResult<(usize,usize)>{
    let covering=covering_bytes(extent,page)?;
    if base%page!=0{return Err(PageObservationError::InvalidRange);}
    if r.page_count==0{return Err(PageObservationError::InvalidRange);}
    if r.page_count>MAX_OBSERVATION_PAGES{return Err(PageObservationError::PageCap);}
    let offset=usize::try_from(r.page_offset).ok().and_then(|n|n.checked_mul(page)).ok_or(PageObservationError::Overflow)?;
    let bytes=(r.page_count as usize).checked_mul(page).ok_or(PageObservationError::Overflow)?;
    let end=offset.checked_add(bytes).ok_or(PageObservationError::Overflow)?;
    if end>covering{return Err(PageObservationError::InvalidRange);}
    base.checked_add(covering).ok_or(PageObservationError::Overflow)?;
    Ok((base.checked_add(offset).ok_or(PageObservationError::Overflow)?,bytes))
}
fn validate_limit(limits:PageQueryLimits)->QueryResult<()> {
    if limits.max_elapsed_ns==0||limits.max_elapsed_ns>MAX_OBSERVATION_ELAPSED_NS{return Err(PageObservationError::InvalidTimeBudget);}Ok(())
}
fn decode(status:i32,returned:u64,bits:&[i32],elapsed:u64,limit:PageQueryLimits)->QueryResult<PageDispositionCounts>{
    validate_limit(limit)?;
    if bits.is_empty()||bits.len()>MAX_OBSERVATION_PAGES as usize{return Err(PageObservationError::PageCap);}
    if status!=0{return Err(PageObservationError::QueryFailed{status,returned_count:returned});}
    if returned!=bits.len()as u64{return Err(PageObservationError::Partial{expected:bits.len()as u64,returned});}
    if elapsed>limit.max_elapsed_ns{return Err(PageObservationError::Stale{elapsed_ns:elapsed,maximum_ns:limit.max_elapsed_ns});}
    let mut c=PageDispositionCounts{present:0,dirty:0,wired:0,copied:0,paged_out:0,other_known:0};
    for(index,&raw)in bits.iter().enumerate(){
        if raw<0||raw&!KNOWN_PAGE_DISPOSITION_BITS!=0{return Err(PageObservationError::UnknownFlags{index,raw});}
        c.present+=u32::from(raw&1!=0);c.dirty+=u32::from(raw&8!=0);c.wired+=u32::from(raw&0x1000!=0);
        c.copied+=u32::from(raw&0x20!=0);c.paged_out+=u32::from(raw&0x10!=0);
        c.other_known+=u32::from(raw&(KNOWN_PAGE_DISPOSITION_BITS&!(1|8|0x1000|0x20|0x10))!=0);
    }Ok(c)
}
#[cfg(target_os="macos")]
fn page_size()->QueryResult<usize>{
    extern "C"{fn getpagesize()->std::ffi::c_int;}
    // SAFETY: process page-size query, no pointers or side effects.
    let p=unsafe{getpagesize()};let p=usize::try_from(p).map_err(|_|PageObservationError::InvalidPageSize)?;
    covering_bytes(1,p)?;Ok(p)
}
#[cfg(not(target_os="macos"))]
fn page_size()->QueryResult<usize>{Err(PageObservationError::UnsupportedPlatform)}
fn now_ns()->QueryResult<u64>{static ORIGIN:OnceLock<Instant>=OnceLock::new();
    u64::try_from(ORIGIN.get_or_init(Instant::now).elapsed().as_nanos()).map_err(|_|PageObservationError::ClockOverflow)
}
#[cfg(target_os="macos")]
fn mach_query(address:usize,bytes:usize,bits:&mut[i32])->(i32,u64){
    extern "C" {
        static mach_task_self_:u32;
        fn mach_vm_page_range_query(task:u32,address:u64,size:u64,dispositions:u64,count:*mut u64)->i32;
    }
    let mut count=bits.len()as u64;
    // SAFETY: caller retains the borrowed mapping/root; checked page range is
    // entirely in its OS covering pages. Output is bounded/initialized i32,
    // count is SDK mach_vm_size_t/u64, never the unrelated u32 message count.
    let status=unsafe{mach_vm_page_range_query(mach_task_self_,address as u64,bytes as u64,bits.as_mut_ptr()as u64,&mut count)};
    (status,count)
}
#[cfg(not(target_os="macos"))]
fn mach_query(_:usize,_:usize,_:&mut[i32])->(i32,u64){(-1,0)}
fn observe(source:PageObservationSource,base:usize,extent:usize,page:usize,range:PageQueryRange,limits:PageQueryLimits)->QueryResult<PageObservation>{
    validate_limit(limits)?;let(address,bytes)=checked_range(base,extent,page,range)?;
    let mut dispositions=Vec::new();dispositions.try_reserve_exact(range.page_count as usize).map_err(|_|PageObservationError::Allocation)?;
    dispositions.resize(range.page_count as usize,i32::MIN);
    let start_ns=now_ns()?;let(status,returned_count)=mach_query(address,bytes,&mut dispositions);let end_ns=now_ns()?;
    let elapsed_ns=end_ns.checked_sub(start_ns).ok_or(PageObservationError::ClockOverflow)?;
    let counts=decode(status,returned_count,&dispositions,elapsed_ns,limits)?;
    Ok(PageObservation{source,range,page_bytes:page,query_address:address,queried_bytes:bytes,raw_mach_status:status,returned_count,
        dispositions,counts,clock:"rust.std.time.Instant.process-local.page-observation-v1",start_ns,end_ns,elapsed_ns})
}
impl MlxBuffer {
    /// Query actual registered ROOT covering pages; clones/subviews report the
    /// same root. Borrows retain resource/mmap throughout the call. No command
    /// health, physical release, file identity, or future lifetime is certified.
    pub fn observe_nocopy_root_pages(&self,range:PageQueryRange,limits:PageQueryLimits)->QueryResult<PageObservation>{
        let page=page_size()?;
        let s=self.backing_snapshot().map_err(PageObservationError::BackingUnavailable)?;
        let(root_id,requested_bytes)=match s.origin{RootOriginSnapshot::KnownNative{id,requested_bytes}=>(id,requested_bytes),_=>return Err(PageObservationError::NotCertifiedNoCopy)};
        if s.provenance!=BufferProvenance::NoCopy||!self.is_file_backed()||self.is_cpu_writable(){return Err(PageObservationError::NotCertifiedNoCopy);}
        let n=s.no_copy.ok_or(PageObservationError::NotCertifiedNoCopy)?;
        let map_end=n.mapping_base_address.checked_add(covering_bytes(n.mapping_bytes,page)?).ok_or(PageObservationError::Overflow)?;
        let root_end=n.registered_base_address.checked_add(n.registered_bytes).ok_or(PageObservationError::Overflow)?;
        if n.registered_base_address<n.mapping_base_address||root_end>map_end||n.registered_bytes%page!=0{return Err(PageObservationError::InvalidRange);}
        observe(PageObservationSource::NoCopyRoot{root_id,requested_bytes,registered_base:n.registered_base_address,registered_bytes:n.registered_bytes},n.registered_base_address,n.registered_bytes,page,range,limits)
    }
}
/// Query metadata for a borrowed readonly mapping, NOT a native certificate of
/// file backing/SAMEFD. HF/consumer owns that association. Covering final-page
/// tail is queried as OS metadata only; no byte is read. Returned data retains
/// no mapping and cannot authorize later unmapping, demotion, or physical credit.
pub fn observe_readonly_mapping_pages(mapping:&Mmap,range:PageQueryRange,limits:PageQueryLimits)->QueryResult<PageObservation>{
    let page=page_size()?;
    observe(PageObservationSource::ReadonlyMapping{mapping_base:mapping.as_ptr()as usize,mapping_bytes:mapping.len()},mapping.as_ptr()as usize,mapping.len(),page,range,limits)
}
#[cfg(test)]mod tests;
