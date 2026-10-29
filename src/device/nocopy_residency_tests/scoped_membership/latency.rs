//! One prospective fixed latency horizon. Never a physical release acknowledgment.
use super::*;
use crate::{BufferBackingSnapshot,BufferProvenance,NativeRootAuthority,RootCreationLimits,RootOriginSnapshot};
use sha2::{Digest,Sha256};
use std::time::Duration;
const POLLS:usize=101;
const PERIOD_NS:u64=50_000_000;
const QUERY_CAP_NS:u64=250_000_000;
const GAP_CAP_NS:u64=250_000_000;
const KNOWN_BITS:i32=0x1fff;
#[derive(Clone,Copy,Debug)]
struct PageSummary {wired:usize,present:usize,dirty:usize,unknown:usize}
fn summary(bits:&[i32])->PageSummary {PageSummary{
    wired:bits.iter().filter(|x|**x&WIRED!=0).count(),
    present:bits.iter().filter(|x|**x&1!=0).count(),
    dirty:bits.iter().filter(|x|**x&8!=0).count(),
    unknown:bits.iter().filter(|x|**x<0||**x&!KNOWN_BITS!=0).count(),
}}
fn digest(bits:&[i32])->String {let mut h=Sha256::new();for b in bits{h.update(b.to_le_bytes());}format!("{:x}",h.finalize())}
fn timing_valid(index:usize,start:u64,end:u64,previous:Option<u64>)->bool {
    index<POLLS&&end>=start&&end-start<=QUERY_CAP_NS&&start>=index as u64*PERIOD_NS&&
    start-index as u64*PERIOD_NS<=GAP_CAP_NS&&previous.is_none_or(|p|start>=p&&start-p<=GAP_CAP_NS)
}
fn eligible(positives:[bool;3],cuts:&[Vec<(bool,bool)>])->bool {
    positives.into_iter().all(|v|v)&&cuts.len()==3&&cuts.iter().all(|cut|{
        if cut.len()!=POLLS||cut.iter().any(|x|!x.0){return false;}
        let Some(first)=cut.iter().position(|x|x.1)else{return false;};
        cut[first..].iter().all(|x|x.1)
    })
}
fn fd_identity(file:&File)->(u64,u64,u64,i64,i64,i64,i64){let m=file.metadata().unwrap();
    (m.dev(),m.ino(),m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec())}
fn descriptor(root:&MlxBuffer,map:&Mmap)->BufferBackingSnapshot {
    let s=root.backing_snapshot().expect("actual backing descriptor");
    assert!(matches!(s.origin,RootOriginSnapshot::KnownNative{..}),"uncertified root identity");
    assert_eq!(s.provenance,BufferProvenance::NoCopy);
    assert_eq!(s.logical_bytes,FILE_BYTES);assert_eq!(s.view_byte_offset,0);
    assert_eq!(s.metal_root_bytes,FILE_BYTES);
    let n=s.no_copy.expect("NoCopy range");
    assert_eq!(n.mapping_base_address,map.as_ptr()as usize);assert_eq!(n.mapping_bytes,FILE_BYTES);
    assert_eq!(n.registered_base_address,map.as_ptr()as usize);assert_eq!(n.registered_bytes,FILE_BYTES);
    s
}
fn descriptor_json(s:BufferBackingSnapshot)->serde_json::Value {
    let (id,requested)=match s.origin {RootOriginSnapshot::KnownNative{id,requested_bytes}=>(id,requested_bytes),_=>panic!("unknown root")};
    let n=s.no_copy.unwrap();json!({"root_id":id,"requested_bytes":requested,"provenance":"NoCopy",
        "metal_root_bytes":s.metal_root_bytes,"metal_allocated_size":s.metal_allocated_bytes,"storage_mode":format!("{:?}",s.storage_mode),
        "view_byte_offset":s.view_byte_offset,"logical_bytes":s.logical_bytes,"mapping_base":n.mapping_base_address,"mapping_bytes":n.mapping_bytes,
        "registered_base":n.registered_base_address,"registered_bytes":n.registered_bytes})
}
fn positive(name:&str,a:&Mmap,b:&Mmap)->bool {let x=query(a.as_ptr(),FILE_BYTES);let y=query(b.as_ptr(),FILE_BYTES);
    let sx=summary(&x);let sy=summary(&y);
    println!("OWNED_LATENCY_POSITIVE {}",json!({"name":name,"pages":PAGES,"wired":[sx.wired,sy.wired],"present":[sx.present,sy.present],"dirty":[sx.dirty,sy.dirty],"unknown_bits_pages":[sx.unknown,sy.unknown],"alias_equal":x==y,"disposition_sha256_le_i32":[digest(&x),digest(&y)]}));
    x==y&&sx.unknown==0&&sy.unknown==0&&sx.wired==PAGES&&sy.wired==PAGES
}
fn poll_cut(name:&str,origin:Instant,aliases:[&Mmap;2],mut verify_owners:impl FnMut(),bytes:&mut usize)->Vec<(bool,bool)> {
    let mut result=Vec::with_capacity(POLLS);let mut previous=None;
    for index in 0..POLLS {
        let target=Duration::from_nanos(index as u64*PERIOD_NS);
        if let Some(remaining)=target.checked_sub(origin.elapsed()){std::thread::sleep(remaining);}
        verify_owners();
        let start=u64::try_from(origin.elapsed().as_nanos()).unwrap();
        let a=query(aliases[0].as_ptr(),FILE_BYTES);let b=query(aliases[1].as_ptr(),FILE_BYTES);
        let end=u64::try_from(origin.elapsed().as_nanos()).unwrap();
        let x=summary(&a);let y=summary(&b);
        let valid=timing_valid(index,start,end,previous)&&a==b&&x.unknown==0&&y.unknown==0;
        let zero=x.wired==0&&y.wired==0;
        let row=json!({"schema":"mlx.nocopy.owned-latency-poll.v1","cut":name,"index":index,
            "scheduled_ns":index as u64*PERIOD_NS,"start_ns":start,"end_ns":end,"query_ns":end-start,
            "clock":"cut-origin Rust Instant elapsed","pages_per_alias":PAGES,
            "wired":[x.wired,y.wired],"present":[x.present,y.present],"dirty":[x.dirty,y.dirty],
            "unknown_bits_pages":[x.unknown,y.unknown],"alias_equal":a==b,"disposition_sha256_le_i32":[digest(&a),digest(&b)],
            "observation_valid":valid,"inactive_zero_wired":zero,"model_credit_bytes":0});
        let text=serde_json::to_string(&row).unwrap();assert!(text.len()<4096);
        *bytes=bytes.checked_add(text.len()).unwrap();assert!(*bytes<=LOG_CAP);
        println!("OWNED_LATENCY_POLL {text}");
        result.push((valid,zero));previous=Some(start);
    }result
}
#[test]
fn owned_latency_classifier_refuses_partial_stale_unknown_rebound_and_missing_positive(){
    let good=vec![vec![(true,true);POLLS];3];assert!(eligible([true;3],&good));
    let mut late=good.clone();late[1][0].1=false;assert!(eligible([true;3],&late));
    late[1][2].1=false;assert!(!eligible([true;3],&late));
    let mut bad=good.clone();bad[0][100].0=false;assert!(!eligible([true;3],&bad));
    assert!(!eligible([true,false,true],&good));assert!(!eligible([true;3],&good[..2]));
    bad=good.clone();bad[1].pop();assert!(!eligible([true;3],&bad));
    assert!(timing_valid(1,PERIOD_NS,PERIOD_NS+1,Some(0)));
    assert!(!timing_valid(1,PERIOD_NS-1,PERIOD_NS,Some(0)));
    assert!(!timing_valid(1,PERIOD_NS,PERIOD_NS+QUERY_CAP_NS+1,Some(0)));
    assert!(!timing_valid(10,10*PERIOD_NS,10*PERIOD_NS+1,Some(0)));
    let x=summary(&[WIRED|1|8,0,0x2000]);assert_eq!((x.wired,x.present,x.dirty,x.unknown),(1,1,1,1));
    assert!(USERDATA<=LIMIT);assert!(310*4096<LOG_CAP);
}
#[test]
#[ignore="ONE fresh-process 3x5s owned NoCopy latency observation <=256MiB userdata; root supervised only"]
fn actual_still_owned_nocopy_fixed_latency_and_repromotion(){
    assert!(USERDATA<=LIMIT);assert!(PHASE_METADATA<=LIVE_METADATA_CAP);assert_eq!(host_page_size().unwrap(),PAGE);
    assert_eq!(std::env::var("HF2Q_ENCODER_SESSION").as_deref(),Ok("1"));
    assert!(!std::env::vars().any(|(k,_)|(k.starts_with("HF2Q_")&&k!="HF2Q_ENCODER_SESSION")||k.starts_with("MLX_")||k.starts_with("MTL_")||k.starts_with("METAL_")),"ambient activation");
    cpu_positive();
    let authority=NativeRootAuthority::install_early(RootCreationLimits{
        max_attempted_requested_bytes:2*FILE_BYTES+65536,max_root_attempts:8,max_raw_constructor_attempts:8,max_registered_devices:1,max_rope_cache_entries:1,
    }).expect("early finite root authority");
    let mut first=Fixture::new();let mut second=SecondFile::new(&first);
    let a0=first.mapping();let a1=first.mapping();let b0=second.mapping();let b1=second.mapping();
    let fa=first.file.as_ref().unwrap();let fb=second.file.as_ref().unwrap();
    let ia=fd_identity(fa);let ib=fd_identity(fb);assert_ne!((ia.0,ia.1),(ib.0,ib.1));
    assert_eq!(Some((ia.0,ia.1)),first.file_id);assert_eq!(Some((ib.0,ib.1)),second.id);
    assert_eq!(ia.2,FILE_BYTES as u64);assert_eq!(ib.2,FILE_BYTES as u64);
    assert_ne!(a0.as_ptr(),a1.as_ptr());assert_ne!(b0.as_ptr(),b1.as_ptr());
    let d=MlxDevice::new().expect("Metal required, no skip");authority.register_device(&d).unwrap();assert!(d.residency_sets_enabled());
    let a=d.map_file_buffer(a0.clone(),0,FILE_BYTES,DType::F32,vec![FILE_BYTES/4]).unwrap();
    let b=d.map_file_buffer(b0.clone(),0,FILE_BYTES,DType::F32,vec![FILE_BYTES/4]).unwrap();
    let sa=descriptor(&a,&a0);let sb=descriptor(&b,&b0);assert_ne!(sa.origin,sb.origin);
    let out=d.alloc_buffer(64,DType::F32,vec![16]).unwrap();
    let verify=||{assert_eq!(fd_identity(fa),ia);assert_eq!(fd_identity(fb),ib);assert_eq!(descriptor(&a,&a0),sa);assert_eq!(descriptor(&b,&b0),sb);};
    println!("OWNED_LATENCY_BEGIN {}",json!({"schema":"mlx.nocopy.owned-latency.v1","prior_negative":"native-scoped-later-switch-actual-v1/65ec3f5;NOTrescored","cuts":3,"polls_each":POLLS,"period_ns":PERIOD_NS,"query_cap_ns":QUERY_CAP_NS,"gap_lateness_cap_ns":GAP_CAP_NS,"scheduled_horizon_ns":5_000_000_000u64,"full_horizon_even_after_zero":true,"payload_union_bytes":2*FILE_BYTES,"alias_va_bytes_nonadditive":4*FILE_BYTES,"userdata_bound":USERDATA,"userdata_limit":LIMIT,"log_cap":LOG_CAP,"files":[ia,ib],"native_backing":[descriptor_json(sa),descriptor_json(sb)],"same_file_alias_offset_bytes":0,"physical_admission":false,"model_credit_bytes":0}));
    let mut log_bytes=0;
    // Fresh membership session drops its completed CB before returning. The
    // checked gpu_read encoder likewise dies before any inactive-page polling.
    membership(&d,&a,&b,[true,false]);let cut=Instant::now();gpu_read(&d,&a,&out,0);let pa=positive("A_initial",&a0,&a1);
    let c0=poll_cut("startup_B_inactive_after_A_read",cut,[&b0,&b1],verify,&mut log_bytes);
    membership(&d,&a,&b,[false,true]);let cut=Instant::now();gpu_read(&d,&b,&out,97);let pb=positive("B_promoted",&b0,&b1);
    let c1=poll_cut("A_demoted_after_B_read",cut,[&a0,&a1],verify,&mut log_bytes);
    membership(&d,&a,&b,[true,false]);let cut=Instant::now();gpu_read(&d,&a,&out,0);let pa2=positive("A_repromoted",&a0,&a1);
    let c2=poll_cut("B_demoted_after_A_repromotion",cut,[&b0,&b1],verify,&mut log_bytes);
    verify();
    let root_history=authority.snapshot().unwrap();
    assert!(root_history.covered&&!root_history.legacy_used&&root_history.unknown.is_none());
    assert_eq!(root_history.denied,0);assert_eq!(root_history.inflight,0);assert_eq!(root_history.successful_roots,3);
    assert!(root_history.attempted_requested_bytes<=2*FILE_BYTES+65536);
    let cuts=vec![c0,c1,c2];let qualified=eligible([pa,pb,pa2],&cuts);
    let starts=cuts.iter().map(|c|c.iter().position(|x|x.1)).collect::<Vec<_>>();
    let suffixes=cuts.iter().map(|c|c.iter().rev().take_while(|x|x.0&&x.1).count()).collect::<Vec<_>>();
    let rebounds=cuts.iter().map(|c|c.windows(2).filter(|w|w[0].1&&!w[1].1).count()).collect::<Vec<_>>();
    drop(out);drop(a);drop(b);drop(d);drop(a1);drop(a0);drop(b1);drop(b0);
    assert!(second.cleanup());assert!(first.cleanup());
    println!("OWNED_LATENCY_COMPLETE {}",json!({"protocol":true,"polls":3*POLLS,"gpu_readbacks":3,"positive_full_wired":[pa,pb,pa2],"first_zero_indices":starts,"retained_zero_suffix_samples":suffixes,"retained_sample_rebounds":rebounds,"classification":if qualified{"ExclusionAtRetainedSamplesWithinFrozenHorizonOnly"}else{"DemotionNotEstablishedOrUnknown"},"latency_capability_only":qualified,"continuous_settling_not_proven":true,"not_residency_ack":true,"continuous_peak_proven":false,"model_credit_bytes":0,"physical_admission":false,"cleanup":true}));
}
