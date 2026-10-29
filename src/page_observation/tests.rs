use super::*;
#[test]
fn actual_range_planner_checks_covering_pages_caps_and_overflow(){
    let r=PageQueryRange{page_offset:0,page_count:1};
    assert_eq!(checked_range(16384,1,16384,r).unwrap(),(16384,16384));
    assert_eq!(checked_range(16384,32768,16384,PageQueryRange{page_offset:1,page_count:1}).unwrap(),(32768,16384));
    assert!(checked_range(1,16384,16384,r).is_err());
    assert!(checked_range(16384,16384,16384,PageQueryRange{page_offset:1,page_count:1}).is_err());
    assert!(checked_range(16384,16384,16384,PageQueryRange{page_offset:u64::MAX,page_count:1}).is_err());
    assert!(checked_range(16384,16384,16384,PageQueryRange{page_offset:0,page_count:0}).is_err());
    assert!(checked_range(16384,usize::MAX,16384,r).is_err());
    assert!(checked_range(usize::MAX-16383,16384,16384,r).is_err());
    assert!(checked_range(16384,16384,3,r).is_err());
    assert!(checked_range(16384,usize::MAX/2,16384,PageQueryRange{page_offset:0,page_count:4097}).is_err());
    assert_eq!(checked_range(16384,4096*16384,16384,PageQueryRange{page_offset:0,page_count:4096}).unwrap().1,4096*16384);
    assert_eq!(MAX_OBSERVATION_PAGES as usize*std::mem::size_of::<i32>(),16384);
}
#[test]
fn actual_decoder_refuses_error_partial_unknown_negative_and_stale(){
    let l=PageQueryLimits{max_elapsed_ns:100};
    assert!(matches!(decode(5,1,&[0],0,l),Err(PageObservationError::QueryFailed{status:5,..})));
    assert!(matches!(decode(0,0,&[0],0,l),Err(PageObservationError::Partial{..})));
    assert!(matches!(decode(0,2,&[0],0,l),Err(PageObservationError::Partial{..})));
    assert!(matches!(decode(0,1,&[-1],0,l),Err(PageObservationError::UnknownFlags{..})));
    assert!(matches!(decode(0,1,&[0x2000],0,l),Err(PageObservationError::UnknownFlags{..})));
    assert!(matches!(decode(0,1,&[0],101,l),Err(PageObservationError::Stale{..})));
    assert!(validate_limit(PageQueryLimits{max_elapsed_ns:0}).is_err());
    assert!(validate_limit(PageQueryLimits{max_elapsed_ns:MAX_OBSERVATION_ELAPSED_NS+1}).is_err());
    assert!(decode(0,0,&[],0,l).is_err());
}
#[test]
fn actual_counts_preserve_all_bits_without_physical_interpretation(){
    let c=decode(0,4,&[0,1|8|0x1000,0x20|0x10,0x1fff],10,PageQueryLimits{max_elapsed_ns:100}).unwrap();
    assert_eq!(c,PageDispositionCounts{present:2,dirty:2,wired:2,copied:2,paged_out:2,other_known:1});
}
#[test]
#[ignore="one fresh-process <=1MiB owned readonly-file/root metadata query; root supervision required"]
fn actual_borrowed_file_alias_and_nocopy_root_subview_observation(){
    use crate::{DType,MlxDevice,NativeRootAuthority,RootCreationLimits};
    use std::{fs::OpenOptions,io::Write,os::unix::fs::{MetadataExt,OpenOptionsExt},sync::Arc};
    let page=page_size().expect("Darwin page query required; no skip");
    assert!(8*page+4*16384<1024*1024);
    let authority=NativeRootAuthority::install_early(RootCreationLimits{
        max_attempted_requested_bytes:1024*1024,max_root_attempts:8,max_raw_constructor_attempts:8,max_registered_devices:1,max_rope_cache_entries:1,
    }).unwrap();
    let path=std::env::temp_dir().join(format!("mlx-page-observation-{}",std::process::id()));
    let mut write=OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).unwrap();
    let identity=write.metadata().unwrap();let id=(identity.dev(),identity.ino());
    struct Cleanup(std::path::PathBuf,(u64,u64));
    impl Drop for Cleanup{fn drop(&mut self){match std::fs::symlink_metadata(&self.0){
        Ok(m)if m.is_file()&&!m.file_type().is_symlink()&&(m.dev(),m.ino())==self.1=>{if let Err(e)=std::fs::remove_file(&self.0){eprintln!("PAGE_OBSERVATION_CLEANUP_ERROR {e}");}},
        _=>eprintln!("PAGE_OBSERVATION_CLEANUP_IDENTITY_REFUSAL"),
    }}}
    let cleanup=Cleanup(path.clone(),id);
    let chunk=vec![7u8;page];write.write_all(&chunk).unwrap();write.write_all(&chunk).unwrap();drop(chunk);drop(write);
    let file=OpenOptions::new().read(true).open(&path).unwrap();let m=file.metadata().unwrap();assert_eq!((m.dev(),m.ino()),id);
    // SAFETY: private immutable file remains owned and unchanged through aliases.
    let escrow=Arc::new(unsafe{memmap2::MmapOptions::new().len(2*page).map(&file)}.unwrap());
    let alias=unsafe{memmap2::MmapOptions::new().len(2*page).map(&file)}.unwrap();
    assert_ne!(escrow.as_ptr(),alias.as_ptr());let weak=Arc::downgrade(&escrow);
    // Metal registration removal and autorelease owners are distinct from
    // the returned metadata. Escrow remains outside their complete lifetime.
    let (a,_v,b)=objc::rc::autoreleasepool(|| {
    struct FlushPending<'a>(&'a MlxDevice);
    impl Drop for FlushPending<'_>{fn drop(&mut self){
        if let Some(set)=self.0.residency_set(){set.flush_pending();}
    }}
    let d=MlxDevice::new().unwrap();authority.register_device(&d).unwrap();
    let _flush_on_unwind=FlushPending(&d);
    let root=d.map_file_buffer(escrow.clone(),0,2*page,DType::U8,vec![2*page]).unwrap();
    let view=root.data_view(8,16,DType::U8,vec![16]).unwrap();
    let r=PageQueryRange{page_offset:0,page_count:2};let l=PageQueryLimits{max_elapsed_ns:MAX_OBSERVATION_ELAPSED_NS};
    let a=root.observe_nocopy_root_pages(r,l).unwrap();let v=view.observe_nocopy_root_pages(r,l).unwrap();
    let b=observe_readonly_mapping_pages(&alias,r,l).unwrap();
    assert_eq!(a.source,v.source);assert_eq!(a.queried_bytes,2*page);assert_eq!(b.queried_bytes,2*page);
    assert_eq!(a.raw_mach_status,0);assert_eq!(a.returned_count,2);assert_eq!(b.returned_count,2);
    for o in [&a,&v,&b]{assert_eq!(o.dispositions.len(),2);assert_eq!(o.end_ns-o.start_ns,o.elapsed_ns);assert!(o.elapsed_ns<=MAX_OBSERVATION_ELAPSED_NS);}
    let owned=d.alloc_buffer(16,DType::U8,vec![16]).unwrap();
    assert!(matches!(owned.observe_nocopy_root_pages(PageQueryRange{page_offset:0,page_count:1},l),Err(PageObservationError::NotCertifiedNoCopy)));
    drop(root);assert!(Arc::strong_count(&escrow)>1);drop(view);drop(owned);
    (a,v,b)
    });
    assert_eq!(Arc::strong_count(&escrow),1,"metadata must not retain the source mapping");
    drop(escrow);assert!(weak.upgrade().is_none());
    // Returned metadata holds no source owner. Alias observations are not atomic
    // and raw arrays need not match if the VM changes between actual calls.
    println!("BOUNDED_PAGE_OBSERVATION_ACTUAL {}",serde_json::json!({"root":format!("{:?}",a.source),"alias":format!("{:?}",b.source),"page_bytes":page,"count":2,"root_raw":a.dispositions,"alias_raw":b.dispositions,"status":0,"metadata_does_not_retain_root":true,"file_identity":[id.0,id.1],"same_fd_join_is_fixture_not_native":true,"physical_credit_bytes":0,"physical_admission":false}));
    drop(alias);drop(file);drop(cleanup);assert!(!path.exists());
}
