//! PRIVATE max-parent controlled-host falsification; no public mutation API or admission.
use super::{cpu_positive,host_page_size,PAGE};
use crate::{DType,MlxBuffer,MlxDevice,NativeRootAuthority,RootCreationLimits,BufferBackingSnapshot,RootOriginSnapshot,BufferProvenance,PageObservation,PageQueryRange,PageQueryLimits,observe_readonly_mapping_pages};
use memmap2::{Mmap,MmapOptions};
use sha2::{Digest,Sha256};
use serde_json::json;
use std::{fs::{self,File,OpenOptions,DirBuilder},io::Write,os::unix::fs::{MetadataExt,OpenOptionsExt,DirBuilderExt},path::PathBuf,sync::Arc,time::{Instant,Duration}};
const MIB:usize=1024*1024;
const COHORT:usize=922_746_880;
const PAGES:usize=COHORT/PAGE;
const USERDATA:usize=4*COHORT+32*MIB;
const LIMIT:usize=3_724_541_952;
const POLLS:usize=15;
const PERIOD:u64=2_000_000_000;
const QUERY_CAP:u64=50_000_000;
const SWEEP_CAP:u64=2_000_000_000;
const GATE_CAP:u64=5_000_000_000;
const IPC_CAP:u64=250_000_000;
const ACTION_CAP:u64=2_250_000_000;
const CHUNKS:usize=14;
const TOTAL_SWEEPS:usize=114;
const LOG_CAP:usize=64*MIB;
const SEQUENCE:[usize;6]=[0,1,2,0,1,2];
const MS_SYNC:i32=0x0010;
const MS_INVALIDATE:i32=0x0002;
extern "C"{fn msync(address:*mut std::ffi::c_void,length:usize,flags:i32)->i32;}
type R<T>=Result<T,String>;
fn text<T,E:std::fmt::Display>(r:Result<T,E>)->R<T>{r.map_err(|e|e.to_string())}
fn need(ok:bool,msg:&str)->R<()>{if ok{Ok(())}else{Err(msg.to_owned())}}
fn pattern(cohort:usize,offset:usize)->u8{((cohort*67+(offset/PAGE)*17+(offset%PAGE)*3)%251)as u8}
fn identity(f:&File)->R<(u64,u64,u64,i64,i64,i64,i64)>{let m=text(f.metadata())?;Ok((m.dev(),m.ino(),m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec()))}
struct OwnedDirectory{path:PathBuf,id:(u64,u64),files:Vec<(PathBuf,(u64,u64))>}
impl OwnedDirectory{
 fn new()->R<Self>{let path=std::env::temp_dir().join(format!("mlx-max-parent-reclaim-{}",std::process::id()));text(DirBuilder::new().mode(0o700).create(&path))?;let m=text(fs::symlink_metadata(&path))?;Ok(Self{path,id:(m.dev(),m.ino()),files:Vec::with_capacity(3)})}
 fn file(&mut self,index:usize)->R<(File,String)>{
  need(index<3&&self.files.len()==index,"closed file/cohort cardinality")?;
  let path=self.path.join(format!("cohort-{index}.bin"));let mut f=text(OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path))?;
  let m=text(f.metadata())?;self.files.push((path.clone(),(m.dev(),m.ino())));
  let mut page=vec![0u8;MIB];let mut hash=Sha256::new();
  for start in(0..COHORT).step_by(page.len()){let n=(COHORT-start).min(page.len());for(i,v)in page[..n].iter_mut().enumerate(){*v=pattern(index,start+i);}text(f.write_all(&page[..n]))?;hash.update(&page[..n]);}
  text(f.sync_all())?;drop(f);let f=text(OpenOptions::new().read(true).open(&path))?;
  let m=text(f.metadata())?;need((m.dev(),m.ino())==self.files[index].1&&m.len()==COHORT as u64,"same initialized readonly FD")?;
  Ok((f,format!("{:x}",hash.finalize())))
 }
 fn cleanup(&mut self)->bool{let Ok(m)=fs::symlink_metadata(&self.path)else{return false;};if !m.is_dir()||m.file_type().is_symlink()||(m.dev(),m.ino())!=self.id{return false;}
  let mut ok=true;for(p,id)in &self.files{match fs::symlink_metadata(p){Ok(m)if m.is_file()&&!m.file_type().is_symlink()&&(m.dev(),m.ino())==*id=>{ok&=fs::remove_file(p).is_ok();},_=>ok=false}}
  ok&=fs::remove_dir(&self.path).is_ok();if ok{self.files.clear();}ok
 }
}
impl Drop for OwnedDirectory{fn drop(&mut self){if !self.files.is_empty(){eprintln!("MAX_PARENT_UNWIND_FIXTURE_CLEANUP ok={}",self.cleanup());}}}
struct Cohort{file:File,id:(u64,u64,u64,i64,i64,i64,i64),a:Arc<Mmap>,b:Arc<Mmap>,root:MlxBuffer,descriptor:BufferBackingSnapshot,hash:String}
impl Cohort{
 fn new(directory:&mut OwnedDirectory,index:usize,d:&MlxDevice)->R<Self>{let(file,hash)=directory.file(index)?;let id=identity(&file)?;
  // SAFETY: private initialized immutable file, readonly FD kept through both maps.
  let a=Arc::new(text(unsafe{MmapOptions::new().len(COHORT).map(&file)})?);
  let b=Arc::new(text(unsafe{MmapOptions::new().len(COHORT).map(&file)})?);
  let root=text(d.map_file_buffer(a.clone(),0,COHORT,DType::U8,vec![COHORT]))?;let descriptor=text(root.backing_snapshot())?;
  let s=Self{file,id,a,b,root,descriptor,hash};s.verify()?;Ok(s)
 }
 fn verify(&self)->R<()>{need(identity(&self.file)?==self.id,"immutable actual FD identity")?;need(text(self.root.backing_snapshot())?==self.descriptor,"stable actual root descriptor")?;
  let s=self.descriptor;need(matches!(s.origin,RootOriginSnapshot::KnownNative{..})&&s.provenance==BufferProvenance::NoCopy&&s.logical_bytes==COHORT&&s.view_byte_offset==0&&s.metal_root_bytes==COHORT,"certified exact NoCopy root")?;
  let n=s.no_copy.ok_or("missing NoCopy range")?;need(n.mapping_base_address==self.a.as_ptr()as usize&&n.mapping_bytes==COHORT&&n.registered_base_address==self.a.as_ptr()as usize&&n.registered_bytes==COHORT&&self.a.as_ptr()!=self.b.as_ptr(),"actual mapping/registered range and independent alias")
 }
}
struct Log{bytes:usize,records:usize}
impl Log{fn emit(&mut self,kind:&str,v:serde_json::Value)->R<()>{let s=text(serde_json::to_string(&v))?;need(s.len()<=16384,"record cap")?;self.bytes=self.bytes.checked_add(s.len()+kind.len()+13).ok_or("log overflow")?;need(self.bytes<=LOG_CAP,"log cap")?;println!("MAX_PARENT_{kind} {s}");self.records+=1;Ok(())}}
fn membership(d:&MlxDevice,c:&[Cohort],on:[bool;3])->R<()>{need(c.len()==3,"exactly three owned cohorts")?;let mut session=text(d.encoder_session())?.ok_or("required EncoderSession unavailable")?;
 for(i,c)in c.iter().enumerate(){c.verify()?;need(if on[i]{session.add_to_residency_set(&c.root)}else{session.remove_from_residency_set(&c.root)},"actual membership API unavailable")?;}
 text(session.commit_and_wait())?;drop(session);Ok(())
}
fn read_all(d:&MlxDevice,c:&Cohort,index:usize,out:&MlxBuffer,step:usize,log:&mut Log)->R<()>{let whole=Instant::now();c.verify()?;let mut enc=text(d.command_encoder())?;text(enc.blit_copy_bytes(&c.root,0,out,0,COHORT as u64))?;text(enc.commit_and_wait())?;drop(enc);
 let actual=text(out.as_logical_slice::<u8>())?;need(actual.len()==COHORT,"whole readback extent")?;
 for(i,&v)in actual.iter().enumerate(){if v!=pattern(index,i){return Err(format!("whole readback mismatch step{step} byte{i}"));}}
 let hash=format!("{:x}",Sha256::digest(actual));need(hash==c.hash.as_str(),"whole readback hash")?;
 need(whole.elapsed()<=Duration::from_secs(10),"whole GPU/oracle deadline")?;log.emit("READ",json!({"step":step,"cohort":index,"bytes":COHORT,"sha256":hash,"all_bytes_checked":true,"own_cb_healthy":true,"encoder_dropped_before_return":true}))
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Arm { NoAdvice, Advice }
impl Arm { fn label(self)->&'static str{match self{Self::NoAdvice=>"no-advice",Self::Advice=>"advice"}} }
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Phase { Pre, Post }
impl Phase { fn label(self)->&'static str{match self{Self::Pre=>"PRE",Self::Post=>"POST"}} }
fn chunk(index:usize)->R<PageQueryRange>{
 need(index<CHUNKS,"chunk index")?;let offset=index.checked_mul(4096).ok_or("chunk overflow")?;
 let count=PAGES.checked_sub(offset).ok_or("chunk outside parent")?.min(4096);
 need(count>0,"empty chunk")?;Ok(PageQueryRange{page_offset:offset as u64,page_count:count as u32})
}
fn elapsed(clock:Instant)->R<u64>{u64::try_from(clock.elapsed().as_nanos()).map_err(|e|e.to_string())}
fn hash_flags(v:&[i32])->String{let mut h=Sha256::new();for b in v{h.update(b.to_le_bytes());}format!("{:x}",h.finalize())}
fn counts(o:&PageObservation)->[u32;6]{[o.counts.present,o.counts.dirty,o.counts.wired,o.counts.copied,o.counts.paged_out,o.counts.other_known]}
fn actual_pair_valid(c:&Cohort,r:PageQueryRange,a:&PageObservation,b:&PageObservation)->bool{
 let Some(offset)=(r.page_offset as usize).checked_mul(PAGE)else{return false;};
 let Some(bytes)=(r.page_count as usize).checked_mul(PAGE)else{return false;};
 let expected_root=match c.descriptor.origin{RootOriginSnapshot::KnownNative{id,..}=>id,_=>return false};
 matches!(a.source,crate::PageObservationSource::NoCopyRoot{root_id,requested_bytes,registered_base,registered_bytes}
  if root_id==expected_root&&requested_bytes==COHORT&&registered_base==c.a.as_ptr()as usize&&registered_bytes==COHORT)
 &&matches!(b.source,crate::PageObservationSource::ReadonlyMapping{mapping_base,mapping_bytes}if mapping_base==c.b.as_ptr()as usize&&mapping_bytes==COHORT)
 &&a.range==r&&b.range==r&&a.page_bytes==PAGE&&b.page_bytes==PAGE
 &&a.query_address==c.a.as_ptr()as usize+offset&&b.query_address==c.b.as_ptr()as usize+offset
 &&a.queried_bytes==bytes&&b.queried_bytes==bytes&&a.raw_mach_status==0&&b.raw_mach_status==0
 &&a.returned_count==r.page_count as u64&&b.returned_count==r.page_count as u64
 &&a.dispositions.len()==r.page_count as usize&&a.dispositions==b.dispositions
 &&a.elapsed_ns<=QUERY_CAP&&b.elapsed_ns<=QUERY_CAP
}
#[derive(Clone,Copy,Default,Debug)]
struct Sweep { totals:[u64;6], start:u64,end:u64 }
impl Sweep { fn clean(self)->bool{self.totals[1]==0&&self.totals[2]==0&&self.totals[3]==0} fn absent(self)->bool{self.clean()&&self.totals[0]==0} }
struct Census { sweeps:usize,queries:usize,reads:usize,gates:usize,advice:usize,last_parent_frame_end:Option<u64> }
fn sweep(c:&Cohort,arm:Arm,cut:usize,stage:&str,index:usize,origin:Instant,log:&mut Log,n:&mut Census)->R<Sweep>{
 c.verify()?;let mut result=Sweep{start:elapsed(origin)?,..Sweep::default()};
 for i in 0..CHUNKS{let r=chunk(i)?;let l=PageQueryLimits{max_elapsed_ns:QUERY_CAP};
  let a=text(c.root.observe_nocopy_root_pages(r,l))?;n.queries+=1;
  let b=text(observe_readonly_mapping_pages(&c.b,r,l))?;n.queries+=1;
  let pair_valid=actual_pair_valid(c,r,&a,&b);
  if !pair_valid{
   // Diagnostic only AFTER the unchanged predicate has failed. No extra page
   // query, retry, acceptance fallback or raw payload access.
   let expected_offset=usize::try_from(r.page_offset).ok().and_then(|p|p.checked_mul(PAGE));
   let expected_bytes=(r.page_count as usize).checked_mul(PAGE);
   let expected_addresses=[expected_offset.and_then(|o|(c.a.as_ptr()as usize).checked_add(o)),expected_offset.and_then(|o|(c.b.as_ptr()as usize).checked_add(o))];
   let mismatches=a.dispositions.iter().zip(&b.dispositions).filter(|(x,y)|x!=y).count()+a.dispositions.len().abs_diff(b.dispositions.len());
   let mut first=Vec::with_capacity(8);
   for j in 0..a.dispositions.len().max(b.dispositions.len()).min(4096){let x=a.dispositions.get(j).copied();let y=b.dispositions.get(j).copied();if x!=y&&first.len()<8{first.push((r.page_offset.checked_add(j as u64),x,y));}}
   let fd_now=match identity(&c.file){Ok(v)=>json!({"stat":v}),Err(e)=>json!({"error":e})};
   let root_now=match c.root.backing_snapshot(){Ok(v)=>json!({"descriptor":format!("{v:?}")}),Err(e)=>json!({"error":e.to_string()})};
   let requested=json!({"range":[r.page_offset,r.page_count as u64],"page_bytes":PAGE,"expected_addresses":expected_addresses,"expected_queried_bytes":expected_bytes,"query_cap_ns":QUERY_CAP,"expected_root_descriptor":format!("{:?}",c.descriptor),"expected_fd":c.id,"actual_fd":fd_now,"actual_root":root_now});
   let describe=|o:&PageObservation|json!({"source":format!("{:?}",o.source),"range":[o.range.page_offset,o.range.page_count as u64],"page_bytes":o.page_bytes,"address":o.query_address,"queried_bytes":o.queried_bytes,"status":o.raw_mach_status,"returned":o.returned_count,"flags_length":o.dispositions.len(),"elapsed_ns":o.elapsed_ns,"clock":o.clock,"start_ns":o.start_ns,"end_ns":o.end_ns,"raw_flags_sha256_le":hash_flags(&o.dispositions),"counts":counts(o)});
   log.emit("PAIR_FAILED",json!({"arm":arm.label(),"cut":cut,"stage":stage,"index":index,"chunk":i,"requested":requested,"root_observation":describe(&a),"alias_observation":describe(&b),"flags_mismatch_count":mismatches,"first_mismatches":first,"mismatch_offset_unit":"root_page_index","first_mismatch_cap":8,"scan_page_cap":4096,"protocol":false,"physical_credit_bytes":0,"no_retry":true}))?;
  }
  need(pair_valid,"actual complete same-file chunk pair/flags/authority disagreement")?;
  let cs=counts(&a);for(j,v)in cs.into_iter().enumerate(){result.totals[j]=result.totals[j].checked_add(v as u64).ok_or("count overflow")?;}
  log.emit("CHUNK",json!({"arm":arm.label(),"cut":cut,"stage":stage,"index":index,"chunk":i,"fd":c.id,"root":format!("{:?}",a.source),"alias":format!("{:?}",b.source),"range":[r.page_offset,r.page_count as u64],"addresses":[a.query_address,b.query_address],"page_bytes":PAGE,"queried_bytes":a.queried_bytes,"returned":[a.returned_count,b.returned_count],"status":[a.raw_mach_status,b.raw_mach_status],"counts":[counts(&a),counts(&b)],"flags_sha256_le":[hash_flags(&a.dispositions),hash_flags(&b.dispositions)],"native_clock":a.clock,"native_brackets":[[a.start_ns,a.end_ns,a.elapsed_ns],[b.start_ns,b.end_ns,b.elapsed_ns]],"equal":true,"physical_credit_bytes":0}))?;
 }
 result.end=elapsed(origin)?;need(result.end>=result.start&&result.end-result.start<=SWEEP_CAP,"whole dual-alias sweep age/skew")?;
 c.verify()?;n.sweeps+=1;need(n.sweeps<=TOTAL_SWEEPS,"fixed sweep cap")?;
 log.emit("SWEEP",json!({"arm":arm.label(),"cut":cut,"stage":stage,"index":index,"chunks":CHUNKS,"queries":2*CHUNKS,"pages_per_alias":PAGES,"totals":result.totals,"source_start_ns":result.start,"source_end_ns":result.end,"whole_ns":result.end-result.start,"clean_unwired":result.clean(),"absent":result.absent(),"native_requested_charge_unchanged":true}))?;Ok(result)
}
fn valid_schedule(index:usize,start:u64,end:u64,period:u64,count:usize,cap:u64)->bool{
 index<count&&end>=start&&end-start<=cap&&start>=index as u64*period&&start-index as u64*period<=IPC_CAP
}
fn precondition(c:&Cohort,arm:Arm,cut:usize,origin:Instant,log:&mut Log,n:&mut Census)->R<()>{
 let start=Instant::now();let mut last=Sweep::default();
 for i in 0..POLLS{let target=Duration::from_nanos(i as u64*PERIOD);if let Some(left)=target.checked_sub(start.elapsed()){std::thread::sleep(left);}
  let a=elapsed(start)?;last=sweep(c,arm,cut,"precondition",i,origin,log,n)?;let b=elapsed(start)?;
  need(valid_schedule(i,a,b,PERIOD,POLLS,SWEEP_CAP),"precondition start/whole-sweep deadline")?;
  log.emit("SCHEDULE",json!({"arm":arm.label(),"cut":cut,"stage":"precondition","index":i,"start_ns":a,"end_ns":b,"target_ns":i as u64*PERIOD}))?;
 }
 need(start.elapsed()<=Duration::from_secs(30)&&last.clean()&&last.totals[0]==PAGES as u64,"final frozen precondition not clean/unwired or >30s")
}
#[repr(C)]struct PollFd{fd:i32,events:i16,revents:i16}
extern "C" {fn poll(fds:*mut PollFd,nfds:u32,timeout:i32)->i32;#[link_name="read"]fn fd_read(fd:i32,buf:*mut u8,len:usize)->isize;}
fn within_gate(ns:u64)->bool{ns<=GATE_CAP}
fn bounded_ack(origin:Instant)->R<serde_json::Value>{
 let mut bytes=Vec::with_capacity(8192);
 loop{need(within_gate(elapsed(origin)?),"gate timeout")?;
  let mut fd=PollFd{fd:0,events:1,revents:0};
  // SAFETY: valid owned stack pollfd, fd0 is the root-owned dedicated input pipe.
  let rc=unsafe{poll(&mut fd,1,50)};need(rc>=0,"gate poll failure")?;if rc==0{continue;}
  need(fd.revents&1!=0&&fd.revents&!1==0,"gate EOF/error/unsupported readiness")?;
  let mut b=0u8;let count=unsafe{fd_read(0,&mut b,1)};need(count==1,"gate EOF/read failure")?;
  if b==b'\n'{let decoded=strict_ack_json(&bytes)?;need(within_gate(elapsed(origin)?),"gate endpoint after read/strict decode")?;return Ok(decoded);}
  need(bytes.len()<8191&&b.is_ascii()&&b!=b'\r',"gate oversized/non-ASCII ACK")?;bytes.push(b);
 }
}
fn validate_ack(v:&serde_json::Value,nonce:&str,arm:Arm,cut:usize,phase:Phase)->bool{
 let Some(o)=v.as_object()else{return false;};let keys=["schema","nonce","arm","cut","phase","frames","valid","parent_clock","last_frame_to_send_ns","ready_to_first_frame_ns","global_frames"];
 o.len()==keys.len()&&keys.iter().all(|k|o.contains_key(*k))
 &&v["schema"].as_str()==Some("mlx.max-parent-gate-ack.v1")&&v["nonce"].as_str()==Some(nonce)
 &&v["arm"].as_str()==Some(arm.label())&&v["cut"].as_u64()==Some(cut as u64)&&v["phase"].as_str()==Some(phase.label())
 &&v["frames"].as_u64()==Some(9)&&v["valid"].as_bool()==Some(true)
 &&v["parent_clock"].as_str()==Some("parent.monotonic.receipt-bounds")
 &&v["last_frame_to_send_ns"].as_u64().is_some_and(|x|x<=IPC_CAP)
 &&v["ready_to_first_frame_ns"].as_u64().is_some_and(|x|x<=IPC_CAP)
 &&plateau(&v["global_frames"]).is_ok()
}
fn gate(arm:Arm,cut:usize,phase:Phase,last_sweep:Option<u64>,origin:Instant,log:&mut Log,n:&mut Census)->R<(Instant,Plateau)>{
 use std::io::Read;
 let mut random=[0u8;16];text(File::open("/dev/urandom"))?.read_exact(&mut random).map_err(|e|e.to_string())?;
 let nonce=random.iter().map(|b|format!("{b:02x}")).collect::<String>();let ready=elapsed(origin)?;
 if let Some(last)=last_sweep{need(ready>=last&&ready-last<=50_000_000,"POST sweep-to-ready age")?;}
 need(n.gates<12,"gate count cap")?;let gate_clock=Instant::now();
 log.emit("GATE_READY",json!({"schema":"mlx.max-parent-gate-ready.v1","arm":arm.label(),"cut":cut,"phase":phase.label(),"nonce":nonce,"source_ready_ns":ready,"last_post_sweep_end_ns":last_sweep,"frames":9,"frame_period_ns":500_000_000u64,"frame_cap_ns":250_000_000u64,"plateau_cap_ns":4_250_000_000u64,"ack_deadline_ns":GATE_CAP,"parent_first_frame_receipt_age_cap_ns":IPC_CAP,"pre_last_frame_to_action_marker_receipt_cap_ns":2_500_000_000u64,"not_clock_conversion":true}))?;
 text(std::io::stdout().flush())?;let ack=bounded_ack(gate_clock)?;need(validate_ack(&ack,&nonce,arm,cut,phase),"closed exact gate ACK/source/parent-age contract")?;
 let frames=plateau(&ack["global_frames"])?;need(within_gate(elapsed(gate_clock)?),"gate endpoint before accepting validated frames")?;advance_parent_frames(&mut n.last_parent_frame_end,&frames)?;n.gates+=1;let acknowledged=Instant::now();log.emit("GATE_ACKED",json!({"arm":arm.label(),"cut":cut,"phase":phase.label(),"nonce":nonce,"ack":ack,"source_ns":elapsed(origin)?}))?;Ok((acknowledged,frames))
}
fn cut(c:&Cohort,arm:Arm,index:usize,origin:Instant,log:&mut Log,n:&mut Census)->R<(Plateau,Plateau)>{
 precondition(c,arm,index,origin,log,n)?;let (ack,pre)=gate(arm,index,Phase::Pre,None,origin,log,n)?;
 let s=sweep(c,arm,index,"immediate_before_cut",0,origin,log,n)?;need(s.clean()&&s.totals[0]==PAGES as u64,"fresh complete PRESENT/clean/unwired advice precondition")?;
 need(ack.elapsed().as_nanos()<=ACTION_CAP as u128,"PRE ACK-to-action age")?;
 log.emit("CUT_BEGIN",json!({"arm":arm.label(),"cut":index,"source_ns":elapsed(origin)?,"since_pre_ack_ns":elapsed(ack)?,"advice_calls":if arm==Arm::Advice{2}else{0},"parent_must_check_last_pre_frame_receipt_age":true}))?;
 for(alias,m)in [&c.a,&c.b].into_iter().enumerate(){let clock=Instant::now();
  need(ack.elapsed().as_nanos()<=ACTION_CAP as u128,"PRE ACK-to-each-advice age")?;
  let(rc,errno)=if arm==Arm::Advice{
   // PRIVATE root-approved undocumented PFN-coherency hypothesis ONLY. Both
   // immutable mappings/FD/Metal parents stay alive; all own CBs checked/drop;
   // same exact range, no unmap/free/protection/write or alternate advice.
   let rc=unsafe{msync(m.as_ptr()as *mut std::ffi::c_void,COHORT,MS_SYNC|MS_INVALIDATE)};
   (rc,if rc==0{None}else{std::io::Error::last_os_error().raw_os_error()})
  }else{(0,None)};
  log.emit("ADVICE",json!({"arm":arm.label(),"cut":index,"alias":alias,"invoked":arm==Arm::Advice,"rc":rc,"errno":errno,"flags":MS_SYNC|MS_INVALIDATE,"bytes":COHORT,"elapsed_ns":elapsed(clock)?,"not_release_ack":true}))?;
  need(rc==0,"advice refused; no fallback/retry")?;
 }
 let post=Instant::now();log.emit("POST_ORIGIN",json!({"arm":arm.label(),"cut":index,"source_ns":elapsed(origin)?}))?;let mut last=Sweep::default();
 for i in 0..3{let target=Duration::from_millis(i as u64*2500);if let Some(left)=target.checked_sub(post.elapsed()){std::thread::sleep(left);}
  let start=elapsed(post)?;last=sweep(c,arm,index,"post_cut",i,origin,log,n)?;let end=elapsed(post)?;
  need(valid_schedule(i,start,end,2_500_000_000,3,SWEEP_CAP),"post whole-sweep schedule")?;
  log.emit("SCHEDULE",json!({"arm":arm.label(),"cut":index,"stage":"post_cut","index":i,"start_ns":start,"end_ns":end,"target_ns":i as u64*2_500_000_000}))?;
  if arm==Arm::Advice{need(last.absent(),"necessary complete dual-alias PRESENT0 not observed")?;}
 }
 need(post.elapsed()<=Duration::from_secs(7),"post sweep horizon")?;
 let (_,post_frames)=gate(arm,index,Phase::Post,Some(last.end),origin,log,n)?;
 // No payload touch/churn/Drop before POST is complete and all six cuts finish.
 Ok((pre,post_frames))
}
fn on(current:usize,next:usize)->[bool;3]{let mut x=[false;3];x[current]=true;x[next]=true;x}
fn run()->R<()>{
 need(USERDATA==LIMIT&&PAGES==56_320&&CHUNKS==14,"predevice exact geometry")?;
 need(text(host_page_size())?==PAGE,"selected 16KiB host only")?;
 need(std::env::var("HF2Q_ENCODER_SESSION").as_deref()==Ok("1"),"explicit EncoderSession")?;
 need(!std::env::vars().any(|(k,_)|(k.starts_with("HF2Q_")&&k!="HF2Q_ENCODER_SESSION")||k.starts_with("MLX_")||k.starts_with("MTL_")||k.starts_with("METAL_")),"ambient activation")?;
 cpu_positive();let authority=text(NativeRootAuthority::install_early(RootCreationLimits{max_attempted_requested_bytes:USERDATA,max_root_attempts:5,max_raw_constructor_attempts:5,max_registered_devices:1,max_rope_cache_entries:1}))?;
 let origin=Instant::now();let mut directory=OwnedDirectory::new()?;let d=text(MlxDevice::new())?;text(authority.register_device(&d))?;
 need(d.residency_sets_enabled()&&d.metal_device().max_buffer_length()as usize>=COHORT,"unsupported residency/maxBufferLength; no segmentation")?;
 let mut cohorts=Vec::with_capacity(3);for i in 0..3{cohorts.push(Cohort::new(&mut directory,i,&d)?);}
 for i in 0..3{for j in 0..i{need(cohorts[i].descriptor.origin!=cohorts[j].descriptor.origin&&(cohorts[i].id.0,cohorts[i].id.1)!=(cohorts[j].id.0,cohorts[j].id.1),"distinct actual file/Metal roots")?;}}
 let out=text(d.alloc_buffer(COHORT,DType::U8,vec![COHORT]))?;need(!out.is_file_backed()&&out.is_cpu_writable()&&out.data_byte_len()==COHORT,"ordinary independent output")?;
 need(origin.elapsed()<=Duration::from_secs(30),"setup/file initialization/sync deadline")?;
 let mut log=Log{bytes:0,records:0};let mut n=Census{sweeps:0,queries:0,reads:0,gates:0,advice:0,last_parent_frame_end:None};
 log.emit("BEGIN",json!({"schema":"mlx.private-max-parent-conservation.v2","q":COHORT,"pages":PAGES,"chunks_per_alias":CHUNKS,"tail_pages":3072,"sequence":SEQUENCE,"arms":["no-advice","advice"],"requested_bytes":USERDATA,"roots":5,"sweeps":114,"queries":3192,"gates":12,"precondition_sweeps":90,"immediate_sweeps":6,"post_sweeps":18,"full_readbacks":15,"expected_records":1884,"fixture_directory":directory.path,"directory_identity":directory.id,"files":directory.files,"file_sha256":cohorts.iter().map(|c|c.hash.clone()).collect::<Vec<_>>(),"backing_descriptors":cohorts.iter().map(|c|format!("{:?}",c.descriptor)).collect::<Vec<_>>(),"policy_sha256":"45c83c50fe9f46152bc9ece5e43c9bd3c753c14a83ec0a0d699ffe8c48a83a74","physical_conservation_qualified":false,"physical_credit_bytes":0,"model_admission":false}))?;
 let mut control:Vec<(Plateau,Plateau)>=Vec::with_capacity(3);let mut reductions=Vec::with_capacity(3);
 let counters=(crate::encoder::dispatch_count(),crate::encoder::commit_count(),crate::encoder::sync_count());
 for arm in [Arm::NoAdvice,Arm::Advice]{
  membership(&d,&cohorts,on(0,1))?;
  for(step,&index)in SEQUENCE.iter().enumerate(){
   read_all(&d,&cohorts[index],index,&out,n.reads,&mut log)?;n.reads+=1;
   if(1..=3).contains(&step){let retired=(index+2)%3;membership(&d,&cohorts,on(index,(index+1)%3))?;let frames=cut(&cohorts[retired],arm,step-1,origin,&mut log,&mut n)?;if arm==Arm::Advice{n.advice+=2;let matched=control.get(step-1).ok_or("missing fixed control cut")?;reductions.push(compare(matched,&frames)?);}else{need(occupied_cross_l1(&frames.0,&frames.1)?<=u128::from(NUISANCE),"NoAdvice nonstationary occupied-union L1")?;control.push(frames);}}
   else if step==4{membership(&d,&cohorts,on(1,2))?;}
  }
 }
 need(n.gates==12&&n.sweeps==114&&n.queries==3192&&n.reads==12&&n.advice==6,"complete fixed cut census before churn")?;
 let mut churn=text(d.alloc_buffer(32*MIB,DType::U8,vec![32*MIB]))?;for(i,v)in text(churn.as_mut_slice::<u8>())?.iter_mut().enumerate(){*v=((i*13+191)%251)as u8;}
 log.emit("CHURN",json!({"bytes":32*MIB,"after_all_six_cuts":true,"pfns_reused_proven":false}))?;
 membership(&d,&cohorts,[true;3])?;
 for(i,c)in cohorts.iter().enumerate(){read_all(&d,c,i,&out,n.reads,&mut log)?;n.reads+=1;c.verify()?;}
 let history=text(authority.snapshot())?;need(history.covered&&!history.legacy_used&&history.unknown.is_none()&&history.denied==0&&history.inflight==0&&history.successful_roots==5,"exact complete requested root history")?;
 let delta=(crate::encoder::dispatch_count()-counters.0,crate::encoder::commit_count()-counters.1,crate::encoder::sync_count()-counters.2);
 need(delta==(0,26,26),"15 whole blits + 11 checked membership commits")?;
 need(log.records==1883,"exact source record census before COMPLETE")?;let metal=d.metal_device().current_allocated_size();drop(churn);drop(out);drop(cohorts);drop(d);need(directory.cleanup(),"identity-checked owned fixture cleanup")?;
 log.emit("COMPLETE",json!({"protocol":true,"necessary_alias_absence_and_refault":true,"full_readbacks":n.reads,"readback_bytes":n.reads*COHORT,"sweeps":n.sweeps,"queries":n.queries,"gates":n.gates,"advice_calls":n.advice,"counter_delta":[delta.0,delta.1,delta.2],"metal_allocated_size_before_drop_not_physical":metal,"controlled_conservation_comparison":if reductions.iter().all(|x|x.0>0){"PositiveUnderFrozenEmpiricalNuisance"}else{"NoPositiveConservation"},"per_cut_reduction_residual":reductions,"independent_parent_source_review_required":true,"physical_conservation_qualified":false,"driver_residual_qualified":false,"physical_credit_bytes":0,"model_admission":false,"cleanup":true,"no_retry":true}))?;Ok(())
}
#[test]fn full_chunk_census_tail_and_bounds(){let mut end=0;for i in 0..14{let r=chunk(i).unwrap();assert_eq!(r.page_offset,end);end+=r.page_count as u64;}assert_eq!(end,56_320);assert_eq!(chunk(13).unwrap().page_count,3072);assert!(chunk(14).is_err());assert_eq!(114*28,3192);assert_eq!(6*15+6+6*3,TOTAL_SWEEPS);assert_eq!(USERDATA,3_724_541_952);assert_eq!(15*COHORT,13_841_203_200);}
#[test]fn actual_schedule_rejects_late_partial_future_geometry(){assert!(valid_schedule(14,28_000_000_000,30_000_000_000,PERIOD,15,SWEEP_CAP));assert!(!valid_schedule(14,27_999_999_999,28_000_000_000,PERIOD,15,SWEEP_CAP));assert!(!valid_schedule(0,0,SWEEP_CAP+1,PERIOD,15,SWEEP_CAP));assert!(!valid_schedule(1,PERIOD+IPC_CAP+1,PERIOD+IPC_CAP+2,PERIOD,15,SWEEP_CAP));assert!(!valid_schedule(15,30_000_000_000,30_000_000_001,PERIOD,15,SWEEP_CAP));}
#[test]fn actual_closed_gate_refuses_replay_wrong_phase_typed_bool_and_age(){
 let nonce="00112233445566778899aabbccddeeff";let good=json!({"schema":"mlx.max-parent-gate-ack.v1","nonce":nonce,"arm":"advice","cut":2,"phase":"POST","frames":9,"valid":true,"parent_clock":"parent.monotonic.receipt-bounds","last_frame_to_send_ns":0,"ready_to_first_frame_ns":250_000_000,"global_frames":synthetic_frames()});
 assert!(validate_ack(&good,nonce,Arm::Advice,2,Phase::Post));assert!(!validate_ack(&good,"other",Arm::Advice,2,Phase::Post));assert!(!validate_ack(&good,nonce,Arm::NoAdvice,2,Phase::Post));assert!(!validate_ack(&good,nonce,Arm::Advice,2,Phase::Pre));
 for(k,v)in[("cut",json!(true)),("frames",json!(9.0)),("valid",json!(1)),("last_frame_to_send_ns",json!(250_000_001u64)),("ready_to_first_frame_ns",json!(false))]{let mut b=good.clone();b[k]=v;assert!(!validate_ack(&b,nonce,Arm::Advice,2,Phase::Post),"{k}");}
 let mut b=good.clone();b["extra"]=json!(0);assert!(!validate_ack(&b,nonce,Arm::Advice,2,Phase::Post));
}
#[test]fn absent_requires_all_source_counts_not_only_wired(){let good=Sweep{totals:[0;6],start:0,end:1};assert!(good.absent());for i in 0..4{let mut b=good;b.totals[i]=1;assert!(!b.absent());}assert!(Sweep{totals:[1,0,0,0,0,0],..good}.clean());}
#[test]
#[ignore="PRIVATE ONE max-parent controlled-host experiment; reviewed 600+3/5s gate/parent handshake required; not model admission"]
fn actual_private_max_parent_controlled_reclaim(){let result=run();if let Err(e)=&result{println!("MAX_PARENT_FAILED {}",json!({"error":e,"protocol":false,"physical_conservation_qualified":false,"physical_credit_bytes":0,"model_admission":false,"no_retry":true}));}assert!(result.is_ok(),"private max-parent experiment: {result:?}");}

// Parent-supplied physical vectors are evidence, NOT native kernel attribution.
// The external runner must bind/validate raw meminfo CSV/header/source separately.
const NUISANCE:u64=128*1024*1024;
#[derive(Clone,Copy,Debug)]struct Frame{start:u64,end:u64,p:u64,v:[u64;7]}
type Plateau=[Frame;9];
fn delta(a:u64,b:u64)->i128{b as i128-a as i128}
// Keep the original seven categories untouched in every ACK/record. Every S
// component is fully charged; this derived partition is NOT owner attribution.
fn charged_partition(f:&Frame)->R<[u64;3]>{
 let mut s=0u64;for i in [0,2,4,5,6]{s=s.checked_add(f.v[i]).ok_or("charged union overflow")?;}
 let total=s.checked_add(f.v[1]).and_then(|n|n.checked_add(f.v[3])).ok_or("union partition overflow")?;
 need(f.p>0&&total==f.p,"raw seven/charged union physical sum contradiction")?;Ok([s,f.v[1],f.v[3]])
}
fn occupied_cross_l1(a:&Plateau,b:&Plateau)->R<u128>{
 let mut maximum=0u128;for x in a{for y in b{let x=charged_partition(x)?;let y=charged_partition(y)?;
  let distance:u128=x.into_iter().zip(y).map(|(a,b)|delta(a,b).unsigned_abs()).sum();maximum=maximum.max(distance);
 }}Ok(maximum)
}
fn plateau(v:&serde_json::Value)->R<Plateau>{
 let a=v.as_array().ok_or("global_frames not array")?;need(a.len()==9,"exact nine global frames")?;
 let mut frames=[Frame{start:0,end:0,p:0,v:[0;7]};9];
 for(i,item)in a.iter().enumerate(){let o=item.as_object().ok_or("frame not object")?;need(o.len()==4&&["start_ns","end_ns","physical_bytes","partition"].iter().all(|k|o.contains_key(*k)),"closed physical frame")?;
  let start=item["start_ns"].as_u64().ok_or("typed frame start")?;let end=item["end_ns"].as_u64().ok_or("typed frame end")?;let p=item["physical_bytes"].as_u64().ok_or("typed independent physical bytes")?;
  let vector=item["partition"].as_array().ok_or("partition not array")?;need(vector.len()==7,"exact disjoint seven categories including U")?;let mut v=[0;7];let mut sum=0u64;
  for(j,x)in vector.iter().enumerate(){v[j]=x.as_u64().ok_or("typed category bytes")?;sum=sum.checked_add(v[j]).ok_or("partition overflow")?;}
  need(p>0&&sum==p&&end>=start&&end-start<=250_000_000,"partition/sampler bracket")?;
  frames[i]=Frame{start,end,p,v};let origin=frames[0].start;let target=origin.checked_add(i as u64*500_000_000).ok_or("frame schedule overflow")?;
  need(start>=target&&start-target<=IPC_CAP&&end-origin<=4_250_000_000,"fixed parent plateau cadence/deadline")?;
  if i>0{need(p==frames[0].p&&start>=frames[i-1].end,"stable P/nonoverlap")?;}
 }
 need(occupied_cross_l1(&frames,&frames)?<=NUISANCE as u128,"stationary plateau ALL-frame occupied-union L1")?;Ok(frames)
}
fn compare(control:&(Plateau,Plateau),advice:&(Plateau,Plateau))->R<(u64,u64)>{
 need(occupied_cross_l1(&control.0,&control.1)?<=NUISANCE as u128,"control ALL-crossframe occupied-union L1")?;
 let p=control.0[0].p;need([&control.1,&advice.0,&advice.1].iter().all(|a|a.iter().all(|f|f.p==p)),"cross-phase physical P drift")?;
 for a in &advice.0{for b in &advice.1{let a_s=charged_partition(a)?[0];let b_s=charged_partition(b)?[0];let occupied_delta=delta(a_s,b_s).unsigned_abs();
  // Fixed P makes the two terms equal: J=2*abs(delta S). This is
  // redundant conservation arithmetic, not independent corroboration.
  let signal=(delta(a.v[3],b.v[3])+delta(a.v[1],b.v[1])).unsigned_abs();need(signal==occupied_delta,"constant-P union/signal contradiction")?;
  let j=2*occupied_delta;need(j<=NUISANCE as u128,"advice ALL-crossframe occupied-union nuisance")?;
 }}
 let a_low=advice.1.iter().map(|x|x.v[3]).min().ok_or("post empty")? as i128-advice.0.iter().map(|x|x.v[3]).max().ok_or("pre empty")? as i128;
 let c_high=control.1.iter().map(|x|x.v[3]).max().ok_or("control post empty")? as i128-control.0.iter().map(|x|x.v[3]).min().ok_or("control pre empty")? as i128;
 let reduction=(a_low-c_high.max(0)-NUISANCE as i128).clamp(0,COHORT as i128)as u64;
 let residual=(COHORT as u64-reduction).checked_add(PAGE as u64-1).ok_or("residual overflow")?/PAGE as u64*PAGE as u64;Ok((reduction,residual))
}
fn synthetic_frames()->serde_json::Value{json!((0..9).map(|i|json!({"start_ns":i*500_000_000u64,"end_ns":i*500_000_000u64+1,"physical_bytes":16u64*1024*1024*1024,"partition":[0,8u64*1024*1024*1024,0,8u64*1024*1024*1024,0,0,0]})).collect::<Vec<_>>())}
#[test]fn actual_all_frame_free_signal_full_nuisance_and_page_residual(){
 let base=plateau(&synthetic_frames()).unwrap();let control=(base,base);let mut post=base;for f in &mut post{f.v[1]-=256*MIB as u64;f.v[3]+=256*MIB as u64;}
 let r=compare(&control,&(base,post)).unwrap();assert_eq!(r,(128*MIB as u64,COHORT as u64-128*MIB as u64));
 let mut control_post=base;for f in &mut control_post{f.v[1]-=32*MIB as u64;f.v[3]+=32*MIB as u64;}assert_eq!(compare(&(base,control_post),&(base,post)).unwrap().0,96*MIB as u64);
 let mut one=base;for f in &mut one{f.v[1]-=NUISANCE+1;f.v[3]+=NUISANCE+1;}assert_eq!(compare(&control,&(base,one)).unwrap(),(1,COHORT as u64));
 let mut b=post;b[8].v[0]+=129*MIB as u64;b[8].v[1]-=129*MIB as u64;assert!(compare(&control,&(base,b)).is_err());
 let mut c=base;c[8].v[1]-=129*MIB as u64;c[8].v[3]+=129*MIB as u64;assert!(compare(&(base,c),&(base,post)).is_err());
 let mut bad=synthetic_frames();bad[8]["physical_bytes"]=json!(false);assert!(plateau(&bad).is_err());
 let mut bad=synthetic_frames();bad[8]["partition"][6]=json!(1);assert!(plateau(&bad).is_err());
}


// Deserialize recursively instead of letting Value silently overwrite object keys.
// The same byte decoder is the sole actual pipe input path; serde's default
// recursion limit remains enabled and the raw line is capped at8192 bytes.
struct UniqueJson(serde_json::Value);
impl<'de> serde::Deserialize<'de> for UniqueJson {
 fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error>{
  struct V;
  impl<'de> serde::de::Visitor<'de> for V {
   type Value=UniqueJson;
   fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("JSON without duplicate object keys")}
   fn visit_bool<E:serde::de::Error>(self,v:bool)->Result<Self::Value,E>{Ok(UniqueJson(json!(v)))}
   fn visit_i64<E:serde::de::Error>(self,v:i64)->Result<Self::Value,E>{Ok(UniqueJson(json!(v)))}
   fn visit_u64<E:serde::de::Error>(self,v:u64)->Result<Self::Value,E>{Ok(UniqueJson(json!(v)))}
   fn visit_f64<E:serde::de::Error>(self,v:f64)->Result<Self::Value,E>{serde_json::Number::from_f64(v).map(|n|UniqueJson(serde_json::Value::Number(n))).ok_or_else(||E::custom("nonfinite JSON"))}
   fn visit_str<E:serde::de::Error>(self,v:&str)->Result<Self::Value,E>{Ok(UniqueJson(serde_json::Value::String(v.to_owned())))}
   fn visit_unit<E:serde::de::Error>(self)->Result<Self::Value,E>{Ok(UniqueJson(serde_json::Value::Null))}
   fn visit_seq<A:serde::de::SeqAccess<'de>>(self,mut a:A)->Result<Self::Value,A::Error>{let mut v=Vec::new();while let Some(x)=a.next_element::<UniqueJson>()?{v.push(x.0);}Ok(UniqueJson(serde_json::Value::Array(v)))}
   fn visit_map<A:serde::de::MapAccess<'de>>(self,mut a:A)->Result<Self::Value,A::Error>{let mut v=serde_json::Map::new();while let Some(k)=a.next_key::<String>()?{if v.contains_key(&k){return Err(serde::de::Error::custom("duplicate JSON object key"));}let x=a.next_value::<UniqueJson>()?;v.insert(k,x.0);}Ok(UniqueJson(serde_json::Value::Object(v)))}
  }
  d.deserialize_any(V)
 }
}
fn strict_ack_json(bytes:&[u8])->R<serde_json::Value>{
 use serde::Deserialize;
 need(bytes.len()<=8192,"ACK decoder byte cap")?;let mut d=serde_json::Deserializer::from_slice(bytes);let v=text(UniqueJson::deserialize(&mut d))?;text(d.end())?;Ok(v.0)
}
fn advance_parent_frames(last:&mut Option<u64>,frames:&Plateau)->R<()>{
 need(last.is_none_or(|end|frames[0].start>=end),"global frames replay/overlap across gates")?;*last=Some(frames[8].end);Ok(())
}
#[test]fn actual_pipe_decoder_rejects_recursive_duplicate_and_trailing_input(){
 for raw in [r#"{"nonce":"old","nonce":"new"}"#,r#"{"schema":"a","schema":"b"}"#,r#"{"global_frames":[{"start_ns":0,"start_ns":1}]}"#,r#"{"global_frames":[{"partition":{"x":0,"x":1}}]}"#,r#"{"nonce":1,"\u006eonce":2}"#,r#"{} {}"#]{assert!(strict_ack_json(raw.as_bytes()).is_err(),"{raw}");}
 let raw=serde_json::to_vec(&synthetic_frames()).unwrap();assert_eq!(strict_ack_json(&raw).unwrap(),synthetic_frames());
}
#[test]fn actual_gate_chronology_rejects_same_frames_even_with_new_nonce(){
 let first=plateau(&synthetic_frames()).unwrap();let mut last=None;advance_parent_frames(&mut last,&first).unwrap();let saved=last;
 assert!(advance_parent_frames(&mut last,&first).is_err());assert_eq!(last,saved);
 let mut next=first;for f in &mut next{f.start+=5_000_000_000;f.end+=5_000_000_000;}advance_parent_frames(&mut last,&next).unwrap();assert_eq!(last,Some(next[8].end));
 let mut overlap=next;overlap[0].start=next[8].end-1;assert!(advance_parent_frames(&mut last,&overlap).is_err());
}
#[test]fn actual_gate_endpoint_refuses_late_newline_parse_and_acceptance(){assert!(within_gate(GATE_CAP));assert!(!within_gate(GATE_CAP+1));assert!(!within_gate(GATE_CAP+50_000_000));}


#[test]fn union_internal_reclassification_keeps_every_raw_byte_charged_and_no_credit(){
 for(from,to)in[(0,4),(0,2),(6,4)]{
  let mut raw=synthetic_frames();for frame in raw.as_array_mut().unwrap(){for i in [0,2,4,5,6]{frame["partition"][i]=json!(512*MIB as u64);}frame["partition"][1]=json!(8u64*1024*1024*1024-5*512*MIB as u64);}
  let stable=plateau(&raw).unwrap();
  for (i,frame)in raw.as_array_mut().unwrap().iter_mut().enumerate(){if i%2==1{frame["partition"][from]=json!(282*MIB as u64);frame["partition"][to]=json!(742*MIB as u64);}}
  let migrated=plateau(&raw).unwrap();assert_ne!(stable[1].v,migrated[1].v);assert_eq!(charged_partition(&stable[1]).unwrap(),charged_partition(&migrated[1]).unwrap());
  assert_eq!(occupied_cross_l1(&stable,&migrated).unwrap(),0);assert_eq!(compare(&(stable,migrated),&(stable,migrated)).unwrap(),(0,COHORT as u64));
  assert_eq!(raw[1]["partition"][from].as_u64(),Some(282*MIB as u64));assert_eq!(raw[1]["partition"][to].as_u64(),Some(742*MIB as u64));
 }
}
#[test]fn union_stationary_control_and_advice_refuse_excess_movement_all_pairs(){
 let base=plateau(&synthetic_frames()).unwrap();
 let mut raw=synthetic_frames();raw[8]["partition"][0]=json!(NUISANCE+1);raw[8]["partition"][3]=json!(8u64*1024*1024*1024-NUISANCE-1);assert!(plateau(&raw).is_err());
 let mut raw=synthetic_frames();raw[8]["partition"][1]=json!(8u64*1024*1024*1024-NUISANCE-1);raw[8]["partition"][3]=json!(8u64*1024*1024*1024+NUISANCE+1);assert!(plateau(&raw).is_err());
 let mut at=base;for f in &mut at{f.v[0]=NUISANCE/2;f.v[3]-=NUISANCE/2;}assert_eq!(occupied_cross_l1(&base,&at).unwrap(),NUISANCE as u128);assert!(compare(&(base,at),&(base,at)).is_ok());
 let mut over=at;for f in &mut over{f.v[0]+=1;f.v[3]-=1;}assert!(compare(&(base,over),&(base,base)).is_err());assert!(compare(&(base,base),&(base,over)).is_err());
}
#[test]fn union_checked_raw_sum_overflow_and_full_nuisance_subtraction(){
 let base=plateau(&synthetic_frames()).unwrap();let mut corrupt=base[0];corrupt.v[0]=u64::MAX;corrupt.v[2]=1;assert!(charged_partition(&corrupt).is_err());
 let mut corrupt=base[0];corrupt.p-=1;assert!(charged_partition(&corrupt).is_err());
 let mut raw=synthetic_frames();raw[0]["partition"][0]=json!(u64::MAX);assert!(plateau(&raw).is_err());
 let mut raw=synthetic_frames();raw[0]["partition"].as_array_mut().unwrap().pop();assert!(plateau(&raw).is_err());
 let mut raw=synthetic_frames();raw[0]["partition"][6]=json!(-1);assert!(plateau(&raw).is_err());
 let mut post=base;for f in &mut post{f.v[1]-=NUISANCE+1;f.v[3]+=NUISANCE+1;}assert_eq!(compare(&(base,base),&(base,post)).unwrap(),(1,COHORT as u64));
 let mut post=base;for f in &mut post{f.v[1]-=COHORT as u64+NUISANCE+1;f.v[3]+=COHORT as u64+NUISANCE+1;}assert_eq!(compare(&(base,base),&(base,post)).unwrap(),(COHORT as u64,0));
}
