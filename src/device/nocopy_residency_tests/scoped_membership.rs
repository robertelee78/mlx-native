//! Two-parent membership observation. Protocol success never grants model credit.
use super::{cpu_positive, host_page_size, query, Fixture, FILE_BYTES, PAGE, PAGES, SENTINELS, WIRED};
use crate::{DType, MlxBuffer, MlxDevice};
use memmap2::{Mmap, MmapOptions};
use serde_json::json;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const LOG_CAP: usize = 16 * 1024 * 1024;
const LIVE_METADATA_CAP: usize = 8 * 1024 * 1024;
// NoCopy roots and file aliases share the same128MiB payload, not another copy.
// Four64MiB alias VA ranges are256MiB VIRTUAL, nonadditive to payload bytes.
const USERDATA: usize = 2 * FILE_BYTES + LOG_CAP + LIVE_METADATA_CAP + 4 * PAGE + 128;
const LIMIT: usize = 256 * 1024 * 1024;
const PHASE_METADATA: usize = 4 * PAGES * (std::mem::size_of::<i32>()
    + std::mem::size_of::<serde_json::Value>()) + 1024 * 1024 + 64 * 1024;

struct SecondFile {
    path: PathBuf,
    parent: PathBuf,
    parent_id: (u64, u64),
    id: Option<(u64, u64)>,
    file: Option<File>,
    cleaned: bool,
}
impl SecondFile {
    fn new(first: &Fixture) -> Self {
        let mut this = Self { path:first.dir.join("second.bin"), parent:first.dir.clone(),
            parent_id:first.dir_id, id:None, file:None, cleaned:false };
        let f=OpenOptions::new().write(true).create_new(true).mode(0o600).open(&this.path).unwrap();
        let m=f.metadata().unwrap();this.id=Some((m.dev(),m.ino()));this.file=Some(f);
        let mut page=[0u8;PAGE];
        for index in 0..PAGES {
            for (j,v) in page.iter_mut().enumerate(){*v=((index+j+97)%251)as u8;}
            this.file.as_mut().unwrap().write_all(&page).unwrap();
        }
        this.file.take();
        let f=OpenOptions::new().read(true).open(&this.path).unwrap();
        let m=f.metadata().unwrap();assert_eq!(Some((m.dev(),m.ino())),this.id);
        assert_eq!(m.len(),FILE_BYTES as u64);this.file=Some(f);this
    }
    fn mapping(&self)->Arc<Mmap>{
        // SAFETY: owned independently initialized file, immutable until cleanup.
        Arc::new(unsafe{MmapOptions::new().len(FILE_BYTES).map(self.file.as_ref().unwrap())}.unwrap())
    }
    fn cleanup(&mut self)->bool{
        if self.cleaned{return true;}
        let Ok(d)=fs::symlink_metadata(&self.parent)else{return false;};
        if !d.is_dir()||d.file_type().is_symlink()||(d.dev(),d.ino())!=self.parent_id{return false;}
        let Ok(m)=fs::symlink_metadata(&self.path)else{return false;};
        if !m.is_file()||m.file_type().is_symlink()||Some((m.dev(),m.ino()))!=self.id{return false;}
        self.file.take();if fs::remove_file(&self.path).is_err(){return false;}
        self.cleaned=true;true
    }
}
impl Drop for SecondFile{
    fn drop(&mut self){if !self.cleaned{eprintln!("SCOPED_SECOND_FILE_UNWIND_CLEANUP ok={}",self.cleanup());}}
}
#[derive(Clone,Copy,Debug)]
struct Flags{zero_both:bool,any_both:bool,aliases_equal:bool}
fn flags(a:&[i32],b:&[i32])->Flags{
    Flags{zero_both:a.iter().chain(b).all(|x|x&WIRED==0),
        any_both:a.iter().any(|x|x&WIRED!=0)&&b.iter().any(|x|x&WIRED!=0),aliases_equal:a==b}
}
fn exclusion(f:Flags,entry_local:Option<bool>)->bool{
    f.zero_both&&f.aliases_equal&&entry_local.unwrap_or(true)
}
fn mechanism(a_was:Flags,b_switch_inactive_a:Flags,b_was:Flags,a_switch_inactive_b:Flags,
    entry_a:Option<bool>,entry_b:Option<bool>)->bool{
    a_was.any_both&&a_was.aliases_equal&&b_was.any_both&&b_was.aliases_equal
        &&exclusion(b_switch_inactive_a,entry_a)
        &&exclusion(a_switch_inactive_b,entry_b)
}
struct Observer{bytes:usize,phases:usize,origin:Instant,last_end_ns:u128}
impl Observer{
    fn phase(&mut self,name:&str,aliases:[&Mmap;4],d:Option<&MlxDevice>,membership:[bool;2],
        actual_classes:Option<&[String;2]>)->[Flags;2]{
        let start=Instant::now();let begin_ns=start.duration_since(self.origin).as_nanos();
        assert!(begin_ns>=self.last_end_ns,"phase clock order");
        let a0=query(aliases[0].as_ptr(),FILE_BYTES);let a1=query(aliases[1].as_ptr(),FILE_BYTES);
        let b0=query(aliases[2].as_ptr(),FILE_BYTES);let b1=query(aliases[3].as_ptr(),FILE_BYTES);
        let result=[flags(&a0,&a1),flags(&b0,&b1)];
        let current=d.map(|d|d.metal_device().current_allocated_size());
        let end_ns=self.origin.elapsed().as_nanos();
        let duration_ns=end_ns.checked_sub(begin_ns).expect("monotonic phase clock");
        assert_eq!(begin_ns.checked_add(duration_ns),Some(end_ns));self.last_end_ns=end_ns;
        let alias_identity:Vec<_>=aliases.iter().enumerate().map(|(i,a)|json!({
            "alias":(["A0","A1","B0","B1"][i]),"parent":if i<2{"A"}else{"B"},
            "base_address":a.as_ptr()as usize,"range_bytes":a.len(),"file_offset_bytes":0})).collect();
        let data=json!({"phase":name,"phase_index":self.phases,"declared_membership":membership,
            "phase_begin_ns":begin_ns,"phase_end_ns":end_ns,"query_duration_ns":duration_ns,
            "clock":"observer-origin Instant elapsed nanoseconds","alias_identity":alias_identity,"page_bytes":PAGE,"pages_per_alias":PAGES,
            "a0":a0,"a1":a1,"b0":b0,"b1":b1,"actual_resource_classes":actual_classes,
            "metal_current_allocated_size_raw":current,"actual_device_name":d.map(|d|d.metal_device().name().to_owned()),
            "entry_local_wired_exclusion":"Unavailable:no existing entry-local query helper in this probe",
            "physical_admission":false,"model_credit_bytes":0});
        let text=serde_json::to_string(&data).unwrap();assert!(text.len()<=1024*1024,"phase JSON cap");
        self.bytes=self.bytes.checked_add(text.len()).unwrap();assert!(self.bytes<=LOG_CAP);
        println!("SCOPED_RESIDENCY_PHASE {text}");self.phases+=1;result
    }
}
fn class(buffer:&MlxBuffer)->String{
    let raw=buffer.metal_buffer().as_ref()as *const metal::BufferRef as *const objc::runtime::Object;
    // SAFETY: live retained Objective-C buffer; query class only, no selector mutation.
    unsafe{(&*raw).class().name().to_owned()}
}
fn membership(d:&MlxDevice,a:&MlxBuffer,b:&MlxBuffer,selected:[bool;2]){
    // MUST use fresh UNDRAINED session: repeated commit_and_wait on a drained
    // session is a no-op and would not flush later membership mutations.
    let mut s=d.encoder_session().expect("session factory").expect("explicit session capability required");
    for (buf,on) in [(a,selected[0]),(b,selected[1])]{
        assert!(if on{s.add_to_residency_set(buf)}else{s.remove_from_residency_set(buf)},"actual residency API unavailable");
    }
    s.commit_and_wait().expect("checked membership flush");
}
fn gpu_read(d:&MlxDevice,root:&MlxBuffer,out:&MlxBuffer,seed:usize){
    let mut e=d.command_encoder().unwrap();
    for (i,page)in SENTINELS.iter().enumerate(){
        e.blit_copy_bytes(root,(page*PAGE)as u64,out,(i*16)as u64,16).unwrap();
    }
    e.commit_and_wait().expect("checked retained sentinel GPU completion");
    let actual=out.as_logical_slice::<u8>().unwrap();
    for(i,page)in SENTINELS.iter().enumerate(){for j in 0..16{
        assert_eq!(actual[i*16+j],((page+j+seed)%251)as u8,"independent sentinel oracle");
    }}
    // Completed CB/encoder owner dies BEFORE the caller demotes this parent.
    drop(e);
}
#[test]
fn scoped_membership_requires_complete_inactive_exclusion(){
    let live=Flags{zero_both:false,any_both:true,aliases_equal:true};
    let excluded=Flags{zero_both:true,any_both:false,aliases_equal:true};
    assert!(mechanism(live,excluded,live,excluded,None,None));
    assert!(!mechanism(Flags{aliases_equal:false,..live},excluded,live,excluded,None,None));
    assert!(!mechanism(live,excluded,Flags{aliases_equal:false,..live},excluded,None,None));
    assert!(!mechanism(live,live,live,excluded,None,None));
    assert!(!mechanism(live,excluded,live,live,None,None));
    assert!(!mechanism(live,excluded,live,excluded,Some(false),None));
    assert!(!mechanism(excluded,excluded,live,excluded,None,None));
    assert!(!exclusion(Flags{aliases_equal:false,..excluded},None));
    assert!(USERDATA<=LIMIT);assert!(PHASE_METADATA<=LIVE_METADATA_CAP);
}
#[test]
#[ignore="two64MiB NoCopy parents, explicit membership observation; isolated root-supervised process required"]
fn scoped_residency_two_owned_parents_later_switch_observation(){
    assert!(USERDATA<=LIMIT);assert!(PHASE_METADATA<=LIVE_METADATA_CAP);assert_eq!(host_page_size().unwrap(),PAGE);
    assert_eq!(std::env::var("HF2Q_ENCODER_SESSION").as_deref(),Ok("1"));
    assert!(!std::env::vars().any(|(k,_)|(k.starts_with("HF2Q_")&&k!="HF2Q_ENCODER_SESSION")
        ||k.starts_with("MLX_")||k.starts_with("MTL_")||k.starts_with("METAL_")),"ambient activation");
    cpu_positive();
    let mut first=Fixture::new();let mut second=SecondFile::new(&first);
    let a0=first.mapping();let a1=first.mapping();let b0=second.mapping();let b1=second.mapping();
    assert_ne!(first.file_id,second.id);assert_ne!(a0.as_ptr(),a1.as_ptr());assert_ne!(b0.as_ptr(),b1.as_ptr());
    println!("SCOPED_RESIDENCY_BEGIN {}",json!({"file_payload_union_bytes":2*FILE_BYTES,
        "requested_metal_weight_bytes":2*FILE_BYTES,"alias_VA_bytes_nonadditive":4*FILE_BYTES,
        "userdata_bound_including_retained_log_and_metadata":USERDATA,"userdata_limit":LIMIT,
        "log_cap":LOG_CAP,"live_metadata_cap":LIVE_METADATA_CAP,"default_heartbeat_seconds_source":180,
        "driver_bootstrap_bytes":"Unknown/outside userdata","files":[first.dir.join("owned.bin"),second.path],
        "file_ids":[first.file_id,second.id],"model_loaded":false,
        "file_parent_identities":[
            {"parent":"A","path":first.dir.join("owned.bin"),"dev_ino":first.file_id,
                "directory_path":first.dir,"directory_dev_ino":first.dir_id,"file_offset_bytes":0,"span_bytes":FILE_BYTES},
            {"parent":"B","path":second.path,"dev_ino":second.id,
                "directory_path":second.parent,"directory_dev_ino":second.parent_id,"file_offset_bytes":0,"span_bytes":FILE_BYTES}]}));
    let mut o=Observer{bytes:0,phases:0,origin:Instant::now(),last_end_ns:0};
    o.phase("plain_aliases_pre_device",[&a0,&a1,&b0,&b1],None,[false,false],None);
    let d=MlxDevice::new().expect("Metal required, no skip");assert!(d.residency_sets_enabled());
    let a=d.map_file_buffer(Arc::clone(&a0),0,FILE_BYTES,DType::F32,vec![FILE_BYTES/4]).unwrap();
    let b=d.map_file_buffer(Arc::clone(&b0),0,FILE_BYTES,DType::F32,vec![FILE_BYTES/4]).unwrap();
    assert!(a.is_file_backed()&&!a.is_cpu_writable()&&b.is_file_backed()&&!b.is_cpu_writable());
    let classes=[class(&a),class(&b)];
    println!("SCOPED_RESIDENCY_ROOT_IDENTITIES {}",json!({"roots":[
        {"parent":"A","metal_buffer_address":a.metal_buffer().as_ref()as *const metal::BufferRef as usize,
            "backing_alias":"A0","file_offset_bytes":0,"span_bytes":FILE_BYTES},
        {"parent":"B","metal_buffer_address":b.metal_buffer().as_ref()as *const metal::BufferRef as usize,
            "backing_alias":"B0","file_offset_bytes":0,"span_bytes":FILE_BYTES}],
        "addresses":"process-local identity only; not physical addresses or allocation credit"}));
    o.phase("both_initially_auto_enrolled_overlap_unflushed",[&a0,&a1,&b0,&b1],Some(&d),[true,true],Some(&classes));
    membership(&d,&a,&b,[true,true]);
    o.phase("initial_both_enrolled_checked_flush",[&a0,&a1,&b0,&b1],Some(&d),[true,true],Some(&classes));
    let out=d.alloc_buffer(64,DType::F32,vec![16]).unwrap();
    membership(&d,&a,&b,[true,false]);
    o.phase("A_only_membership_before_GPU",[&a0,&a1,&b0,&b1],Some(&d),[true,false],Some(&classes));
    gpu_read(&d,&a,&out,0);
    let first_read=o.phase("A_GPU_completed_owners_alive",[&a0,&a1,&b0,&b1],Some(&d),[true,false],Some(&classes));
    membership(&d,&a,&b,[false,true]);
    o.phase("B_only_membership_before_GPU",[&a0,&a1,&b0,&b1],Some(&d),[false,true],Some(&classes));
    gpu_read(&d,&b,&out,97);
    let switched_b=o.phase("B_GPU_completed_A_inactive_owners_alive",[&a0,&a1,&b0,&b1],Some(&d),[false,true],Some(&classes));
    membership(&d,&a,&b,[true,false]);gpu_read(&d,&a,&out,0);
    let switched_a=o.phase("A_rejoined_GPU_completed_B_inactive_owners_alive",[&a0,&a1,&b0,&b1],Some(&d),[true,false],Some(&classes));
    let observed=mechanism(first_read[0],switched_b[0],switched_b[1],switched_a[1],None,None);
    membership(&d,&a,&b,[false,false]);
    o.phase("both_demoted_Metal_owners_still_alive",[&a0,&a1,&b0,&b1],Some(&d),[false,false],Some(&classes));
    drop(out);drop(a);drop(b);d.residency_set().unwrap().flush_pending();
    o.phase("explicit_Metal_wrapper_drop_aliases_live",[&a0,&a1,&b0,&b1],Some(&d),[false,false],Some(&classes));
    drop(d);
    o.phase("device_queue_owner_drop_aliases_live",[&a0,&a1,&b0,&b1],None,[false,false],Some(&classes));
    drop(b1);drop(b0);drop(a1);drop(a0);
    assert!(second.cleanup(),"owned second file cleanup");assert!(first.cleanup(),"owned original fixture cleanup");
    assert_eq!(o.phases,11);
    println!("SCOPED_RESIDENCY_COMPLETE {}",json!({"protocol":"PASS","phases":o.phases,
        "classification":if observed{"MechanismObserved"}else{"NoObservedLaterSwitchExclusion"},
        "classification_scope":"later live-owner switch only; initial auto-enrolled overlap retained",
        "entry_local_wired":"Unavailable","sentinel_GPU_reads":3,"checked_drain":true,
        "cleanup":true,"model_credit_bytes":0,"model_scale_qualified":false,"physical_admission":false}));
}

#[path = "scoped_membership/startup_graph.rs"]
mod startup_graph;
