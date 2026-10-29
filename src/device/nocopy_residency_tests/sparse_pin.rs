//! Private fixed-stage observation; no reclaim or physical/model admission.
use super::{host_page_size,PAGE};
use crate::{MlxDevice,MlxBuffer,DType,NativeRootAuthority,RootCreationLimits,RootOriginSnapshot,BufferProvenance,PageObservation,PageObservationSource,PageQueryRange,PageQueryLimits,observe_readonly_mapping_pages,KNOWN_PAGE_DISPOSITION_BITS};
use memmap2::{Mmap,MmapOptions};
use sha2::{Digest,Sha256};
use serde_json::json;
use std::{fs::{self,File,OpenOptions,DirBuilder},io::Write,os::unix::fs::{MetadataExt,OpenOptionsExt,DirBuilderExt},path::PathBuf,sync::Arc,time::{Instant,Duration}};
const MIB:usize=1024*1024;const GIB:usize=1024*MIB;
const FILE_BYTES:usize=8*GIB;const OUTPUT:usize=64;const BLOCK:usize=MIB;
const PAGES:usize=FILE_BYTES/PAGE;const CHUNK:usize=4096;const CHUNKS:usize=PAGES/CHUNK;
const QUERY_CAP:u64=250_000_000;const SWEEP_CAP:u64=2_000_000_000;const PERIOD:u64=3_000_000_000;
const RECORD_CAP:usize=24*1024;const LOG_CAP:usize=64*MIB;
const ANCHORS:[usize;4]=[0,PAGES/3,2*PAGES/3,PAGES-1];
type R<T>=Result<T,String>;
fn text<T,E:std::fmt::Display>(r:Result<T,E>)->R<T>{r.map_err(|e|e.to_string())}
fn need(v:bool,m:&str)->R<()>{if v{Ok(())}else{Err(m.into())}}
fn ns(t:Instant)->R<u64>{u64::try_from(t.elapsed().as_nanos()).map_err(|e|e.to_string())}
fn word(index:u64)->u64{index.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0xd1b54a32d192ed03)|1}
fn expected_byte(offset:usize)->u8{word((offset/8)as u64).to_le_bytes()[offset%8]}
fn fill_stream_block(block:&mut[u8],start:usize)->R<()>{need(start%8==0&&block.len()%8==0,"word-aligned stream block required")?;for(j,out)in block.chunks_exact_mut(8).enumerate(){out.copy_from_slice(&word(((start/8)+j)as u64).to_le_bytes());}Ok(())}
fn stat(f:&File)->R<(u64,u64,u64,i64,i64,i64,i64)>{let m=text(f.metadata())?;need(m.is_file(),"regular owned file required")?;Ok((m.dev(),m.ino(),m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec()))}
struct Fixture{dir:PathBuf,dir_id:(u64,u64),path:PathBuf,file_id:(u64,u64),removed:bool}
impl Fixture{
 fn create()->R<(Self,File)>{let dir=std::env::temp_dir().join(format!("mlx-sparse-pin-{}",std::process::id()));text(DirBuilder::new().mode(0o700).create(&dir))?;let d=text(fs::symlink_metadata(&dir))?;let path=dir.join("owned.bin");
  let f=match OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path){Ok(f)=>f,Err(e)=>{let _=fs::remove_dir(&dir);return Err(e.to_string());}};let m=text(f.metadata())?;Ok((Self{dir,dir_id:(d.dev(),d.ino()),path,file_id:(m.dev(),m.ino()),removed:false},f))
 }
 fn cleanup(&mut self)->R<()>{let d=text(fs::symlink_metadata(&self.dir))?;need(d.is_dir()&&!d.file_type().is_symlink()&&(d.dev(),d.ino())==self.dir_id,"owned directory identity changed")?;let f=text(fs::symlink_metadata(&self.path))?;need(f.is_file()&&!f.file_type().is_symlink()&&(f.dev(),f.ino())==self.file_id,"owned file identity changed")?;text(fs::remove_file(&self.path))?;text(fs::remove_dir(&self.dir))?;self.removed=true;Ok(())}
}
impl Drop for Fixture{fn drop(&mut self){if !self.removed{let outcome=self.cleanup();eprintln!("SPARSE_PIN_UNWIND_CLEANUP {}",json!({"ok":outcome.is_ok(),"error":outcome.err()}));}}}
struct Log{bytes:usize,records:usize}
impl Log{fn emit(&mut self,kind:&str,v:serde_json::Value)->R<()>{let s=text(serde_json::to_string(&v))?;need(s.len()+kind.len()+13<=RECORD_CAP,"bounded single-alias record exceeded")?;self.bytes=self.bytes.checked_add(s.len()+kind.len()+13).ok_or("log count overflow")?;need(self.bytes<=LOG_CAP,"source log cap exceeded")?;println!("SPARSE_PIN_{kind} {s}");self.records+=1;Ok(())}}
fn raw_counters()->[u64;6]{[crate::dispatch_count(),crate::commit_count(),crate::sync_count(),crate::cmd_buf_count(),crate::residency_commit_call_count_for_test()as u64,crate::residency_allocation_count_for_test()as u64]}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]enum Stage{PreNoCopy,Registered,Flushed,Read}
impl Stage{fn label(self)->&'static str{match self{Self::PreNoCopy=>"preNoCopy",Self::Registered=>"registered-before-ourcommit",Self::Flushed=>"after-checked-empty-flush",Self::Read=>"after-checked-four-16B-blits"}}fn ordinal(self)->usize{match self{Self::PreNoCopy=>0,Self::Registered=>1,Self::Flushed=>2,Self::Read=>3}}}
#[derive(Clone,Copy)]enum Expected{Mapping{base:usize},Root{base:usize,id:usize}}
fn counts(o:&PageObservation)->[u64;6]{[o.counts.present as u64,o.counts.dirty as u64,o.counts.wired as u64,o.counts.copied as u64,o.counts.paged_out as u64,o.counts.other_known as u64]}
fn bit_counts(bits:&[i32])->[u64;6]{let mut c=[0;6];for &b in bits{for(i,mask)in[1,8,0x1000,0x20,0x10,KNOWN_PAGE_DISPOSITION_BITS&!(1|8|0x1000|0x20|0x10)].into_iter().enumerate(){c[i]+=u64::from(b&mask!=0);}}c}
fn valid(o:&PageObservation,r:PageQueryRange,expected:Expected)->bool{
 let(base,source)=match expected{
  Expected::Mapping{base}=>(base,matches!(o.source,PageObservationSource::ReadonlyMapping{mapping_base,mapping_bytes}if mapping_base==base&&mapping_bytes==FILE_BYTES)),
  Expected::Root{base,id}=>(base,matches!(o.source,PageObservationSource::NoCopyRoot{root_id,requested_bytes,registered_base,registered_bytes}if root_id==id&&requested_bytes==FILE_BYTES&&registered_base==base&&registered_bytes==FILE_BYTES)),
 };
 let address=usize::try_from(r.page_offset).ok().and_then(|n|n.checked_mul(PAGE)).and_then(|n|base.checked_add(n));
 source&&r.page_count==CHUNK as u32&&r.page_offset< PAGES as u64&&r.page_offset+CHUNK as u64<=PAGES as u64
 &&o.range==r&&o.page_bytes==PAGE&&Some(o.query_address)==address&&o.queried_bytes==CHUNK*PAGE
 &&o.raw_mach_status==0&&o.returned_count==CHUNK as u64&&o.dispositions.len()==CHUNK
 &&o.clock=="rust.std.time.Instant.process-local.page-observation-v1"&&o.end_ns>=o.start_ns&&o.end_ns-o.start_ns==o.elapsed_ns&&o.elapsed_ns<=QUERY_CAP
 &&o.dispositions.iter().all(|&b|b>=0&&b&!KNOWN_PAGE_DISPOSITION_BITS==0)&&counts(o)==bit_counts(&o.dispositions)
}
fn chunk(i:usize)->R<PageQueryRange>{need(i<CHUNKS,"chunk index outside full alias")?;Ok(PageQueryRange{page_offset:(i*CHUNK)as u64,page_count:CHUNK as u32})}
fn cadence(index:usize,start:u64,end:u64)->bool{index<2&&end>=start&&end-start<=SWEEP_CAP&&start>=index as u64*PERIOD&&start-index as u64*PERIOD<=QUERY_CAP&&end<=5_000_000_000}
fn root_id(root:&MlxBuffer,a:&Mmap)->R<usize>{let b=text(root.backing_snapshot())?;let n=b.no_copy.ok_or("missing original NoCopy backing")?;
 need(root.is_file_backed()&&!root.is_cpu_writable()&&b.storage_mode==metal::MTLStorageMode::Shared&&b.provenance==BufferProvenance::NoCopy&&b.logical_bytes==FILE_BYTES&&b.view_byte_offset==0&&b.metal_root_bytes==FILE_BYTES&&n.mapping_bytes==FILE_BYTES&&n.registered_bytes==FILE_BYTES&&n.mapping_base_address==a.as_ptr()as usize&&n.registered_base_address==a.as_ptr()as usize,"root descriptor/registered mapping disagreement")?;
 match b.origin{RootOriginSnapshot::KnownNative{id,requested_bytes}if requested_bytes==FILE_BYTES=>Ok(id),_=>Err("unknown root lineage".into())}
}
fn stage(stage:Stage,file:&File,identity:(u64,u64,u64,i64,i64,i64,i64),a:&Arc<Mmap>,b:&Arc<Mmap>,root:Option<&MlxBuffer>,log:&mut Log,queries:&mut usize)->R<()>{
 need(stat(file)?==identity&&a.len()==FILE_BYTES&&b.len()==FILE_BYTES&&a.as_ptr()!=b.as_ptr(),"same FD/alias source drift")?;
 let id=match root{Some(r)=>Some(root_id(r,a)?),None=>None};need((stage==Stage::PreNoCopy)==root.is_none(),"stage/root ownership mismatch")?;
 let before=raw_counters();let clock=Instant::now();log.emit("STAGE_BEGIN",json!({"stage":stage.label(),"ordinal":stage.ordinal(),"counters":before,"root_id":id,"constructor_staged_add_semantics_not_pending_getter":stage==Stage::Registered}))?;
 for sample in 0..2{let target=Duration::from_nanos(sample as u64*PERIOD);if let Some(left)=target.checked_sub(clock.elapsed()){std::thread::sleep(left);}
  let start=ns(clock)?;need(start>=sample as u64*PERIOD&&start-sample as u64*PERIOD<=QUERY_CAP,"snapshot start stale")?;let mut totals=[[0u64;6];2];
  for i in 0..CHUNKS{let range=chunk(i)?;for(alias,mapping)in [a,b].into_iter().enumerate(){
   let query_start=ns(clock)?;let observation=if alias==0{if let Some(r)=root{text(r.observe_nocopy_root_pages(range,PageQueryLimits{max_elapsed_ns:QUERY_CAP}))?}else{text(observe_readonly_mapping_pages(mapping,range,PageQueryLimits{max_elapsed_ns:QUERY_CAP}))?}}else{text(observe_readonly_mapping_pages(mapping,range,PageQueryLimits{max_elapsed_ns:QUERY_CAP}))?};
   let query_end=ns(clock)?;let expected=if alias==0{if let Some(id)=id{Expected::Root{base:a.as_ptr()as usize,id}}else{Expected::Mapping{base:a.as_ptr()as usize}}}else{Expected::Mapping{base:b.as_ptr()as usize}};
   need(valid(&observation,range,expected),"invalid individual page query; no alias-equality rule")?;
   for(j,v)in counts(&observation).into_iter().enumerate(){totals[alias][j]=totals[alias][j].checked_add(v).ok_or("page count overflow")?;}
   let mut hash=Sha256::new();for bit in &observation.dispositions{hash.update(bit.to_le_bytes());}
   log.emit("PAGES",json!({"stage":stage.label(),"sample":sample,"alias":alias,"chunk":i,"fd":identity,"source":format!("{:?}",observation.source),"range":[range.page_offset,range.page_count as u64],"page_bytes":observation.page_bytes,"address":observation.query_address,"queried_bytes":observation.queried_bytes,"raw_status":observation.raw_mach_status,"returned":observation.returned_count,"counts":counts(&observation),"clock":observation.clock,"outer_clock":"rust.std.time.Instant.stage-local","native_start_ns":observation.start_ns,"native_end_ns":observation.end_ns,"native_elapsed_ns":observation.elapsed_ns,"outer_stage_start_ns":query_start,"outer_stage_end_ns":query_end,"query_cap_ns":QUERY_CAP,"flags_sha256_le":format!("{:x}",hash.finalize()),"flags":observation.dispositions,"physical_credit_bytes":0}))?;
   *queries+=1;
  }}
  let end=ns(clock)?;need(cadence(sample,start,end),"whole dual-alias snapshot late/stale")?;need(stat(file)?==identity,"FD changed during snapshot")?;if let Some(r)=root{need(Some(root_id(r,a)?)==id,"root identity drift")?;}
  need(raw_counters()==before,"unexpected command/residency activity during fixed stage")?;
  log.emit("SNAPSHOT",json!({"stage":stage.label(),"sample":sample,"target_ns":sample as u64*PERIOD,"start_ns":start,"end_ns":end,"whole_cap_ns":SWEEP_CAP,"pages_per_alias":PAGES,"queries":2*CHUNKS,"counts_per_alias":totals,"alias_equality_required":false,"physical_credit_bytes":0}))?;
 }
 log.emit("STAGE_END",json!({"stage":stage.label(),"ordinal":stage.ordinal(),"counters":raw_counters(),"queries_so_far":*queries}))?;Ok(())
}
fn run()->R<()>{
 need(FILE_BYTES==8_589_934_592&&PAGES==524288&&CHUNKS==128&&8*2*CHUNKS==2048,"predevice fixed source geometry")?;need(text(host_page_size())?==PAGE,"16KiB actual host required; no alternate geometry")?;
 need(2048*RECORD_CAP+64*32768<LOG_CAP,"full raw flags/metadata log reservation")?;
 need(std::env::var("HF2Q_ENCODER_SESSION").as_deref()==Ok("1"),"explicit EncoderSession required")?;
 need(!std::env::vars().any(|(k,_)|(k.starts_with("HF2Q_")&&k!="HF2Q_ENCODER_SESSION")||k.starts_with("MLX_")||k.starts_with("MTL_")||k.starts_with("METAL_")),"ambient activation refused")?;
 let setup=Instant::now();let mut log=Log{bytes:0,records:0};let(mut fixture,mut writer)=Fixture::create()?;
 log.emit("SETUP",json!({"schema":"mlx.nocopy-sparse-pin-stages.v1","directory":fixture.dir,"directory_id":fixture.dir_id,"file":fixture.path,"file_id":fixture.file_id,"file_bytes":FILE_BYTES,"block_bytes":BLOCK,"native_requested_cap":FILE_BYTES+OUTPUT,"roots_cap":2,"device_cap":1,"record_cap":RECORD_CAP,"log_cap":LOG_CAP,"total_snapshots":8,"queries":2048,"raw_flags":8_388_608u64,"pre_free_gib":32,"runtime_free_gib":8,"process_gib":24,"wall_seconds":90,"absolute_with_cleanup_seconds":98,"disk_gib":10,"physical_admission":false}))?;
 let mut block=vec![0u8;BLOCK];let mut hash=Sha256::new();for start in(0..FILE_BYTES).step_by(BLOCK){fill_stream_block(&mut block,start)?;text(writer.write_all(&block))?;hash.update(&block);}text(writer.sync_all())?;drop(writer);drop(block);
 let input_sha=format!("{:x}",hash.finalize());let file=text(OpenOptions::new().read(true).open(&fixture.path))?;let identity=stat(&file)?;need((identity.0,identity.1)==fixture.file_id&&identity.2==FILE_BYTES as u64,"initialized original FD identity")?;
 // SAFETY: exact owned immutable regular file and readonly FD retained through
 // both complete mappings/Metal root/observations. Neither alias is CPU-read.
 let a=Arc::new(text(unsafe{MmapOptions::new().len(FILE_BYTES).map(&file)})?);let b=Arc::new(text(unsafe{MmapOptions::new().len(FILE_BYTES).map(&file)})?);
 let authority=text(NativeRootAuthority::install_early(RootCreationLimits{max_attempted_requested_bytes:FILE_BYTES+OUTPUT,max_root_attempts:2,max_raw_constructor_attempts:2,max_registered_devices:1,max_rope_cache_entries:1}))?;
 let d=text(MlxDevice::new())?;text(authority.register_device(&d))?;need(d.residency_sets_enabled()&&d.metal_device().max_buffer_length()as usize>=FILE_BYTES,"actual residency/8GiB maxBufferLength unavailable")?;
 let out=text(d.alloc_buffer(OUTPUT,DType::U8,vec![OUTPUT]))?;need(!out.is_file_backed()&&out.is_cpu_writable(),"ordinary64B output")?;need(setup.elapsed()<=Duration::from_secs(60),"fixed setup/stream-init deadline")?;
 let baseline=raw_counters();need(baseline[..4]==[0,0,0,0],"unexpected prebaseline own commands")?;let mut queries=0;
 log.emit("BEGIN",json!({"fd":identity,"input_stream_sha256":input_sha,"independent_disk_rehash":false,"map_bases":[a.as_ptr()as usize,b.as_ptr()as usize],"map_bytes":[a.len(),b.len()],"actual_page_bytes":PAGE,"anchors_pages":ANCHORS,"source_cpu_mmap_payload_read_bytes":0,"counters":baseline,"same_file_alias_equality_not_required":true,"physical_credit_bytes":0}))?;
 stage(Stage::PreNoCopy,&file,identity,&a,&b,None,&mut log,&mut queries)?;
 let root=text(d.map_file_buffer(a.clone(),0,FILE_BYTES,DType::U8,vec![FILE_BYTES]))?;let id=root_id(&root,&a)?;let registered=raw_counters();need(registered[..4]==[0,0,0,0]&&registered[4]==baseline[4],"unexpected commit during NoCopy construction")?;
 log.emit("REGISTERED",json!({"root_id":id,"descriptor":format!("{:?}",text(root.backing_snapshot())?),"counters_before":baseline,"counters_after":registered,"pending_membership_not_observed":true}))?;
 stage(Stage::Registered,&file,identity,&a,&b,Some(&root),&mut log,&mut queries)?;
 let mut flush=text(d.encoder_session())?.ok_or("required EncoderSession unavailable")?;text(flush.commit_and_wait())?;drop(flush);let flushed=raw_counters();need(flushed[..4]==[0,1,1,1]&&flushed[4]==registered[4]+1,"empty checked flush counter contract")?;
 log.emit("EMPTY_FLUSH",json!({"actual_checked_result":true,"wrapper_dropped":true,"counters_before":registered,"counters_after":flushed,"physical_ack":false}))?;
 stage(Stage::Flushed,&file,identity,&a,&b,Some(&root),&mut log,&mut queries)?;
 let mut encoder=text(d.command_encoder())?;for(i,&page)in ANCHORS.iter().enumerate(){text(encoder.blit_copy_bytes(&root,(page*PAGE)as u64,&out,(i*16)as u64,16))?;}text(encoder.commit_and_wait())?;drop(encoder);
 let actual=text(out.as_logical_slice::<u8>())?;need(actual.len()==OUTPUT,"complete64B output extent")?;for(i,&page)in ANCHORS.iter().enumerate(){for j in 0..16{need(actual[i*16+j]==expected_byte(page*PAGE+j),"sparse64B independent pattern oracle mismatch")?;}}
 let read=raw_counters();need(read[..4]==[0,2,2,2]&&read[4]==flushed[4],"four blits single checked CB counter contract")?;
 log.emit("READ",json!({"anchor_pages":ANCHORS,"bytes_per_anchor":16,"output_bytes":OUTPUT,"all64_bytes_exact":true,"actual_checked_result":true,"wrapper_dropped":true,"counters_before":flushed,"counters_after":read}))?;
 stage(Stage::Read,&file,identity,&a,&b,Some(&root),&mut log,&mut queries)?;
 let history=text(authority.snapshot())?;need(history.covered&&!history.legacy_used&&history.unknown.is_none()&&history.denied==0&&history.inflight==0&&history.successful_roots==2&&history.root_attempts==2&&history.attempted_requested_bytes==FILE_BYTES+OUTPUT,"complete exact native root history")?;
 need(queries==2048&&log.records==2069,"exact8snapshot/2048query source census")?;need(stat(&file)?==identity&&root_id(&root,&a)?==id,"final owner/source identity")?;
 drop(root);drop(out);drop(d);drop(a);drop(b);drop(file);text(fixture.cleanup())?;
 log.emit("COMPLETE",json!({"protocol":true,"records":2070,"snapshots":8,"queries":queries,"raw_flags":8_388_608u64,"explicit_gpu_read_bytes":64,"root_requested_bytes":history.attempted_requested_bytes,"successful_roots":history.successful_roots,"counters_after":read,"source_cpu_mmap_payload_read_bytes":0,"cleanup":true,"pinning_outcome_required":false,"alias_equality_required":false,"physical_admission":false,"physical_credit_bytes":0,"model_loaded":false}))?;Ok(())
}
#[test]fn sparse_geometry_coverage_stage_schedule_and_log_bound(){let mut end=0u64;for i in 0..CHUNKS{let r=chunk(i).unwrap();assert_eq!(r.page_offset,end);end+=r.page_count as u64;}assert_eq!(end,PAGES as u64);assert!(chunk(CHUNKS).is_err());assert_eq!(8*2*CHUNKS,2048);assert_eq!(8*2*PAGES,8_388_608);assert!(2048*RECORD_CAP+64*32768<LOG_CAP);assert!(cadence(0,0,1));assert!(cadence(1,PERIOD,5_000_000_000));assert!(!cadence(1,PERIOD-1,PERIOD));assert!(!cadence(1,PERIOD,5_000_000_001));assert!(!cadence(2,2*PERIOD,2*PERIOD+1));}
#[test]fn stream_word_and_sparse_byte_oracle_use_actual_offsets(){for &p in &ANCHORS{let mut actual=[0u8;16];fill_stream_block(&mut actual,p*PAGE).unwrap();for (j,&v)in actual.iter().enumerate(){assert_eq!(v,expected_byte(p*PAGE+j));}assert_ne!(word((p*PAGE/8)as u64),0);}let words=ANCHORS.map(|p|word((p*PAGE/8)as u64));for i in 0..4{for j in 0..i{assert_ne!(words[i],words[j]);}}assert!(fill_stream_block(&mut[0;7],0).is_err());assert!(fill_stream_block(&mut[0;8],1).is_err());}
#[test]fn individual_validity_refuses_corruption_but_does_not_compare_alias_flags(){
 let r=chunk(0).unwrap();let make=|base|PageObservation{source:PageObservationSource::ReadonlyMapping{mapping_base:base,mapping_bytes:FILE_BYTES},range:r,page_bytes:PAGE,query_address:base,queried_bytes:CHUNK*PAGE,raw_mach_status:0,returned_count:CHUNK as u64,dispositions:vec![0;CHUNK],counts:crate::PageDispositionCounts{present:0,dirty:0,wired:0,copied:0,paged_out:0,other_known:0},clock:"rust.std.time.Instant.process-local.page-observation-v1",start_ns:0,end_ns:1,elapsed_ns:1};
 let a=make(PAGE);let mut b=make(2*PAGE);assert!(valid(&a,r,Expected::Mapping{base:PAGE}));b.dispositions[1]=0x1001;b.counts.present=1;b.counts.wired=1;assert!(valid(&b,r,Expected::Mapping{base:2*PAGE}));assert_ne!(a.dispositions,b.dispositions);
 b.returned_count-=1;assert!(!valid(&b,r,Expected::Mapping{base:2*PAGE}));let mut b=make(PAGE);b.dispositions[0]=0x2000;assert!(!valid(&b,r,Expected::Mapping{base:PAGE}));let mut b=make(PAGE);b.elapsed_ns=QUERY_CAP+1;b.end_ns=b.elapsed_ns;assert!(!valid(&b,r,Expected::Mapping{base:PAGE}));let b=make(PAGE);assert!(!valid(&b,r,Expected::Root{base:PAGE,id:1}));let mut native=make(PAGE);native.source=PageObservationSource::NoCopyRoot{root_id:7,requested_bytes:FILE_BYTES,registered_base:PAGE,registered_bytes:FILE_BYTES};assert!(valid(&native,r,Expected::Root{base:PAGE,id:7}));assert!(!valid(&native,r,Expected::Root{base:PAGE,id:8}));let mut bad=make(PAGE);bad.clock="wrong clock";assert!(!valid(&bad,r,Expected::Mapping{base:PAGE}));
}
#[test]
#[ignore="ONE8GiB private sparse pin stage observation; freshprocess90s+3/5s guard/sourceHLDreview required; no admission"]
fn actual_nocopy_sparse_read_parent_pin_stages(){let result=run();if let Err(e)=&result{println!("SPARSE_PIN_FAILED {}",json!({"error":e,"protocol":false,"physical_credit_bytes":0,"physical_admission":false,"no_retry":true}));}assert!(result.is_ok(),"sparse pin observation: {result:?}");}
