use super::*;
use std::io::Cursor;
fn info(name:&str,offset:u64,bytes:usize)->TensorInfo{TensorInfo{name:name.into(),shape:vec![bytes/4],ggml_type:GgmlType::F32,offset,byte_len:bytes}}
fn file(infos:Vec<TensorInfo>,data_offset:u64)->GgufFile{
 // Host-only synthetic parsed directory; no Device/file contents needed for
 // the exact same production geometry function. No physical identity claim.
 let f=std::fs::File::open("/dev/null").unwrap();
 GgufFile{metadata:HashMap::new(),tensors:infos.into_iter().map(|i|(i.name.clone(),i)).collect(),tensor_data_offset:data_offset,reader:Mutex::new(BufReader::new(f)),bounded_header_usage:None}
}
#[test]fn pure_plan_matches_production_ranges_prefix_tail_views_and_page_union(){
 let g=file(vec![info("third",12288,100),info("first",0,100),info("second",8192,100)],100);
 let p=g.mapped_tensor_storage_plan_with_limits(4096,12288).unwrap();let ranges=plan_mapped_tensor_ranges(100,g.tensors.values().collect(),4096,12288).unwrap();let old=GgufMappedTensorStoragePlan::from_ranges(&ranges).unwrap();assert_eq!(p.storage_plan(),&old);assert_eq!(p.assumed_page_size(),4096);assert_eq!(p.assumed_max_buffer_len(),12288);
 assert_eq!(p.segments().len(),2);let a=&p.segments()[0];assert_eq!((a.file_start(),a.file_end(),a.mapping_file_start(),a.mapping_file_end(),a.mapping_bytes()),(100,8392,0,8392,8392));assert_eq!((a.root_byte_len(),a.prefix_bytes(),a.tail_padding_bytes(),a.registered_file_end()),(12288,100,3896,12288));
 let last=&p.segments()[1];assert_eq!(last.mapping_file_end(),12488);assert_eq!(last.registered_file_end(),16384);assert_eq!(p.registered_file_page_union_bytes(),16384);
 for v in p.tensor_views(){let info=g.tensor_info(v.name()).unwrap();let s=&p.segments()[v.segment_index()];assert_eq!(v.file_start(),100+info.offset as usize);assert_eq!(v.file_end()-v.file_start(),v.logical_bytes());assert_eq!(v.logical_bytes(),info.byte_len);assert_eq!(v.logical_view_offset(),v.file_start()-s.file_start());assert_eq!(v.root_byte_offset(),v.file_start()-s.registered_file_start());}
}
#[test]fn shared_source_pages_overlap_and_explicit_invalid_geometry_overflow(){
 let g=file(vec![info("a",0,5000),info("overlap",4000,5000)],100);let p=g.mapped_tensor_storage_plan_with_limits(4096,12288).unwrap();assert_eq!(p.segments().len(),1);assert_eq!(p.registered_file_page_union_bytes(),12288);assert_eq!(p.storage_plan().physical_byte_len(),12288);
 assert!(g.mapped_tensor_storage_plan_with_limits(0,12288).is_err());assert!(g.mapped_tensor_storage_plan_with_limits(3000,12288).is_err());assert!(g.mapped_tensor_storage_plan_with_limits(4096,8192).is_err());assert!(g.mapped_tensor_storage_plan_with_limits(usize::MAX,usize::MAX).is_err());assert!(g.mapped_tensor_storage_plan_with_limits(1usize<<(usize::BITS-1),usize::MAX).is_err());
 let g=file(vec![info("overflow",u64::MAX,4)],1);assert!(g.mapped_tensor_storage_plan_with_limits(4096,16384).is_err());
 let g=file(vec![info("end-overflow",0,usize::MAX)],100);assert!(g.mapped_tensor_storage_plan_with_limits(4096,16384).is_err());
 let g=file(vec![info("too-large",0,20000)],0);assert!(g.mapped_tensor_storage_plan_with_limits(4096,16384).is_err());
}
#[test]fn bounded_header_and_pure_plan_work_without_device_or_payload(){
 // Shared parser: one four-byte F32 tensor, but header-only backing has no
 // payload. Pure plan does not map/read/validate physical file occupancy.
 let mut b=Vec::new();b.extend(GGUF_MAGIC.to_le_bytes());b.extend(GGUF_VERSION.to_le_bytes());b.extend(1u64.to_le_bytes());b.extend(0u64.to_le_bytes());b.extend(1u64.to_le_bytes());b.extend(b"x");b.extend(1u32.to_le_bytes());b.extend(1u64.to_le_bytes());b.extend(GGML_TYPE_F32.to_le_bytes());b.extend(0u64.to_le_bytes());
 let l=GgufHeaderLimits{max_read_bytes:1024,max_declared_allocation_bytes:4096,max_metadata_entries:1,max_tensors:1,max_string_bytes:8,max_total_array_elements:1,max_nesting_depth:1};let parsed=header::parse(Cursor::new(b),Some(l)).unwrap();let mut g=file(Vec::new(),parsed.tensor_data_offset);g.metadata=parsed.metadata;g.tensors=parsed.tensors;
 let p=g.mapped_tensor_storage_plan_with_limits(4096,16384).unwrap();assert_eq!(p.tensor_views()[0].logical_bytes(),4);assert_eq!(p.segments()[0].root_byte_len(),4096);
}

#[test]fn split_resources_can_share_a_file_page_without_double_counting_union(){
 let g=file(vec![info("a",0,8092),info("b",8000,8200)],100);
 let p=g.mapped_tensor_storage_plan_with_limits(4096,12288).unwrap();assert_eq!(p.segments().len(),2);
 assert_eq!(p.storage_plan().physical_byte_len(),20480);assert_eq!(p.registered_file_page_union_bytes(),16384);
 let b=p.tensor_views().iter().find(|v|v.name()=="b").unwrap();assert_eq!(b.root_byte_offset(),4004);assert_eq!(b.logical_view_offset(),0);
}
