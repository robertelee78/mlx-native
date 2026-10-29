//! One fresh-process actual mapping boundary, without command creation/GPU work.
use mlx_native::{GgufFile,MlxDevice,NativeRootAuthority,RootCreationLimits,RootOriginSnapshot,BufferProvenance,DType};
use mlx_native::gguf::GgufHeaderLimits;
use serde_json::json;
use sha2::{Digest,Sha256};
use std::{fs::{File,OpenOptions},io::{Write,Seek},os::unix::fs::{MetadataExt,OpenOptionsExt},path::PathBuf};
const MAX_PAGE:usize=65536;
const FIXTURE_CAP:usize=3*MAX_PAGE;
const ROOT_CAP:usize=4*MAX_PAGE;
const HEADER_CAP:usize=65536;
const PLAN_CELL:usize=32768;
const NAMED_UPPER:usize=ROOT_CAP+FIXTURE_CAP+2*HEADER_CAP+PLAN_CELL+8192+65536;
const USER_CAP:usize=1024*1024;
const NAMES:[&str;3]=["first","middle","last"];
const OFFSETS:[usize;3]=[32,MAX_PAGE+32,2*MAX_PAGE+32];
const VALUES:[&[u32];3]=[&[0x3f800000,0x40800000,0x41100000,0x41800000,0x41c80000],&[0x40000000,0x40400000],&[0x40e00000]];
extern "C"{fn getpagesize()->i32;}
fn stat(f:&File)->(u64,u64,u64,i64,i64,i64,i64){let m=f.metadata().expect("actual FD stat");(m.dev(),m.ino(),m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec())}
fn emit(kind:&str,value:serde_json::Value){let s=serde_json::to_string(&value).expect("bounded JSON");assert!(s.len()<=8192);println!("BOUNDED_GGUF_MAP_{kind} {s}");}
fn counters()->[u64;4]{[mlx_native::dispatch_count(),mlx_native::commit_count(),mlx_native::sync_count(),mlx_native::cmd_buf_count()]}
struct OwnedFile{path:PathBuf,id:(u64,u64),removed:bool}
impl OwnedFile{
 fn remove(&mut self){let m=std::fs::symlink_metadata(&self.path).expect("owned fixture identity");assert!(m.is_file()&&!m.file_type().is_symlink());assert_eq!((m.dev(),m.ino()),self.id);std::fs::remove_file(&self.path).expect("remove owned fixture");self.removed=true;}
}
impl Drop for OwnedFile{fn drop(&mut self){if !self.removed{if let Ok(m)=std::fs::symlink_metadata(&self.path){if m.is_file()&&!m.file_type().is_symlink()&&(m.dev(),m.ino())==self.id{let _=std::fs::remove_file(&self.path);}}}}}
fn bytes()->Vec<u8>{
 let mut b=Vec::with_capacity(FIXTURE_CAP);b.extend(b"GGUF");b.extend(3u32.to_le_bytes());b.extend(3u64.to_le_bytes());b.extend(0u64.to_le_bytes());
 for i in 0..3{b.extend((NAMES[i].len()as u64).to_le_bytes());b.extend(NAMES[i].as_bytes());b.extend(1u32.to_le_bytes());b.extend((VALUES[i].len()as u64).to_le_bytes());b.extend(0u32.to_le_bytes());b.extend((OFFSETS[i]as u64).to_le_bytes());}
 while b.len()%32!=0{b.push(0);}let data=b.len();
 for i in 0..3{let start=data+OFFSETS[i];b.resize(start,0);for v in VALUES[i]{b.extend(v.to_le_bytes());}}
 assert!(b.len()<FIXTURE_CAP&&b.capacity()==FIXTURE_CAP);b
}
#[test]
#[ignore="ONE fresh-process <=1MiB named roots/CPU fixture; actual Metal mapping only, no commands/GPU; root supervisor required"]
fn actual_bounded_header_pure_plan_matches_mapped_roots(){
 // All named source and native request reservations precede Device creation.
 assert!(NAMED_UPPER<=USER_CAP);assert!(ROOT_CAP<=USER_CAP);
 let page=usize::try_from(unsafe{getpagesize()}).expect("actual host page");assert!(page.is_power_of_two()&&(4096..=MAX_PAGE).contains(&page));
 let data=bytes();let hash=format!("{:x}",Sha256::digest(&data));
 let path=std::env::temp_dir().join(format!("mlx-bounded-map-{}.gguf",std::process::id()));
 let mut write=OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).expect("exclusive fixture");let id=stat(&write);let mut owned=OwnedFile{path:path.clone(),id:(id.0,id.1),removed:false};
 write.write_all(&data).expect("write fixture");write.sync_all().expect("sync fixture");drop(write);
 let retained=OpenOptions::new().read(true).open(&path).expect("retained readonly FD");let source=stat(&retained);assert_eq!((source.0,source.1),owned.id);assert_eq!(source.2,data.len()as u64);
 let authority=NativeRootAuthority::install_early(RootCreationLimits{max_attempted_requested_bytes:ROOT_CAP,max_root_attempts:8,max_raw_constructor_attempts:8,max_registered_devices:1,max_rope_cache_entries:1}).expect("fresh early authority");
 let limits=GgufHeaderLimits{max_read_bytes:HEADER_CAP as u64,max_declared_allocation_bytes:HEADER_CAP as u64,max_metadata_entries:4,max_tensors:8,max_string_bytes:128,max_total_array_elements:16,max_nesting_depth:4};
 let gguf=GgufFile::from_file_with_limits(retained.try_clone().expect("same FD clone"),limits).expect("bounded actual FD header");let usage=gguf.bounded_header_usage().expect("bounded usage");
 let mut cursor=&retained;let underlying_position=cursor.stream_position().expect("actual unbuffered FD cursor");assert_eq!(underlying_position,usage.read_bytes);assert!(underlying_position<=gguf.tensor_data_offset()&&gguf.tensor_data_offset()<source.2);assert_eq!(stat(&retained),source);
 let before=counters();assert_eq!(before,[0;4]);let pre=authority.snapshot().expect("predevice roots");assert_eq!(pre.root_attempts,0);assert_eq!(pre.registered_devices,0);
 emit("BEGIN",json!({"schema":"mlx.bounded-header-real-map.v1","fixture":path,"file_stat":source,"file_sha256":hash,"fixture_bytes":data.len(),"fixture_cap":FIXTURE_CAP,"named_upper":NAMED_UPPER,"named_cap":USER_CAP,"native_requested_cap":ROOT_CAP,"native_root_cap":8,"actual_page":page,"header_read_bytes":usage.read_bytes,"header_declared_reservation_bytes":usage.declared_allocation_bytes,"underlying_fd_position":underlying_position,"counters_before":before,"physical_admission":false}));
 let device=MlxDevice::new().expect("actual Device");authority.register_device(&device).expect("same Device authority");let actual_max=usize::try_from(device.metal_device().max_buffer_length()).expect("actual maxBufferLength");assert!(actual_max>=ROOT_CAP);
 let negative=gguf.mapped_tensor_storage_plan_with_limits(page,page.checked_mul(2).expect("page arithmetic"));assert!(negative.is_err());let negative_roots=authority.snapshot().expect("negative no roots");assert_eq!(negative_roots.root_attempts,0);assert_eq!(negative_roots.denied,0);assert_eq!(counters(),before);
 emit("NEGATIVE",json!({"explicit_max":2*page,"refused":true,"root_attempts":negative_roots.root_attempts,"denied":negative_roots.denied,"counters":counters(),"actual_device_max_unchanged":actual_max}));
 let pure=gguf.mapped_tensor_storage_plan_with_limits(page,actual_max).expect("pure actual-geometry plan");let legacy=gguf.mapped_tensor_storage_plan(device.metal_device()).expect("legacy actual Device summary");assert_eq!(pure.storage_plan(),&legacy);assert_eq!(pure.segments().len(),1);assert!(pure.storage_plan().physical_byte_len()<=ROOT_CAP);
 let mapped=gguf.map_tensor_data(&device).expect("actual production mapping");assert_eq!(mapped.storage_plan(),pure.storage_plan());assert_eq!(mapped.segment_count(),pure.segments().len());assert_eq!(mapped.physical_byte_len(),pure.storage_plan().physical_byte_len());
 let segment=&pure.segments()[0];assert!(segment.prefix_bytes()>0&&segment.tail_padding_bytes()>0);assert_eq!(segment.mapping_file_end(),source.2 as usize);assert!(segment.registered_file_end()>source.2 as usize);assert_eq!(segment.root_byte_len(),segment.prefix_bytes()+(segment.file_end()-segment.file_start())+segment.tail_padding_bytes());
 let mut views=Vec::with_capacity(3);let mut root_ids=std::collections::BTreeSet::new();
 for view in pure.tensor_views(){let i=NAMES.iter().position(|n|*n==view.name()).expect("exact fixture name");let buffer=mapped.load_tensor(view.name()).expect("actual typed view");let snapshot=buffer.backing_snapshot().expect("actual backing");let no_copy=snapshot.no_copy.expect("actual NoCopy owner");
  let(root_id,requested)=match snapshot.origin{RootOriginSnapshot::KnownNative{id,requested_bytes}=>(id,requested_bytes),_=>panic!("unknown root")};root_ids.insert(root_id);
  assert_eq!(snapshot.provenance,BufferProvenance::NoCopy);assert!(!buffer.is_cpu_writable()&&buffer.is_file_backed());assert_eq!(buffer.dtype(),DType::F32);assert_eq!(snapshot.logical_bytes,view.logical_bytes());assert_eq!(snapshot.view_byte_offset,view.root_byte_offset()as u64);assert_eq!(buffer.byte_offset(),snapshot.view_byte_offset);
  assert_eq!(snapshot.metal_root_bytes,segment.root_byte_len());assert_eq!(requested,segment.root_byte_len());assert_eq!(no_copy.registered_bytes,segment.root_byte_len());assert_eq!(no_copy.mapping_bytes,segment.mapping_bytes());assert_eq!(no_copy.mapping_base_address,no_copy.registered_base_address);
  let registered_file_start=view.file_start().checked_sub(snapshot.view_byte_offset as usize).expect("actual root/file join");assert_eq!(registered_file_start,segment.registered_file_start());assert_eq!(view.file_start()-segment.file_start(),view.logical_view_offset());assert_eq!(view.file_start(),gguf.tensor_data_offset()as usize+OFFSETS[i]);
  let actual=buffer.as_logical_slice::<f32>().expect("CPU file bytes only");assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>().as_slice(),VALUES[i]);assert_eq!(snapshot.logical_bytes,VALUES[i].len()*4);
  emit("TENSOR",json!({"name":view.name(),"segment":view.segment_index(),"root_id":root_id,"requested":requested,"file_start":view.file_start(),"file_end":view.file_end(),"root_view_offset":snapshot.view_byte_offset,"logical_view_offset":view.logical_view_offset(),"root_length":snapshot.metal_root_bytes,"mapping_bytes":no_copy.mapping_bytes,"mapping_base":no_copy.mapping_base_address,"registered_base":no_copy.registered_base_address,"registered_bytes":no_copy.registered_bytes,"registered_file_start":registered_file_start,"prefix":segment.prefix_bytes(),"tail":segment.tail_padding_bytes(),"exact_payload_bits":true}));views.push(buffer);
 }
 assert_eq!(root_ids.len(),1);let first=views.iter().find(|b|b.data_byte_len()==20).expect("first view");let nested=first.slice_view(4,1);let nested_backing=nested.backing_snapshot().expect("nested backing");assert_eq!(nested_backing.origin,first.backing_snapshot().unwrap().origin);assert_eq!(nested_backing.view_byte_offset,first.byte_offset()+4);assert_eq!(nested.as_logical_slice::<f32>().unwrap()[0].to_bits(),VALUES[0][1]);
 emit("SUBVIEW",json!({"same_origin":true,"offset":nested_backing.view_byte_offset,"logical_bytes":nested_backing.logical_bytes,"payload_bits":VALUES[0][1]}));
 let after=counters();assert_eq!(after,before);let final_roots=authority.snapshot().expect("actual root history");assert!(final_roots.covered&&!final_roots.legacy_used&&final_roots.unknown.is_none());assert_eq!(final_roots.successful_roots,1);assert_eq!(final_roots.root_attempts,1);assert_eq!(final_roots.registered_devices,1);assert_eq!(final_roots.inflight,0);assert_eq!(final_roots.denied,0);assert_eq!(final_roots.attempted_requested_bytes,segment.root_byte_len());assert_eq!(stat(&retained),source);
 let root_length=segment.root_byte_len();let prefix=segment.prefix_bytes();let tail=segment.tail_padding_bytes();
 drop(nested);drop(views);drop(mapped);drop(pure);drop(gguf);drop(device);drop(retained);owned.remove();
 emit("COMPLETE",json!({"protocol":true,"records":7,"tensor_views":3,"nested_views":1,"actual_root_count":final_roots.successful_roots,"root_history":{"covered":final_roots.covered,"legacy_used":final_roots.legacy_used,"unknown":final_roots.unknown,"attempts":final_roots.root_attempts,"raw_attempt_reservations":final_roots.raw_attempt_reservations,"registered_devices":final_roots.registered_devices,"inflight":final_roots.inflight,"denied":final_roots.denied},"requested_bytes":final_roots.attempted_requested_bytes,"root_length":root_length,"prefix":prefix,"tail":tail,"counters_before":before,"counters_after":after,"no_commands":true,"no_gpu_dispatch":true,"same_fd_stat_unchanged":true,"cleanup":true,"physical_admission":false,"physical_credit_bytes":0}));
}
