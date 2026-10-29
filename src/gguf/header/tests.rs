use super::*;
use std::io::Cursor;
fn limits()->GgufHeaderLimits{GgufHeaderLimits{max_read_bytes:64*1024*1024,max_declared_allocation_bytes:64*1024*1024,max_metadata_entries:1024,max_tensors:1024,max_string_bytes:1024*1024,max_total_array_elements:4096,max_nesting_depth:8}}
fn reader(bytes:Vec<u8>,l:GgufHeaderLimits)->HeaderReader<Cursor<Vec<u8>>>{HeaderReader{inner:Cursor::new(bytes),limits:Some(l),usage:GgufHeaderUsage::default()}}
fn fixed(tensors:u64,metadata:u64)->Vec<u8>{let mut b=Vec::new();b.extend(GGUF_MAGIC.to_le_bytes());b.extend(GGUF_VERSION.to_le_bytes());b.extend(tensors.to_le_bytes());b.extend(metadata.to_le_bytes());b}
#[test]fn declared_strings_arrays_and_catalog_refuse_before_allocation(){
 let mut r=reader(1_048_577u64.to_le_bytes().to_vec(),limits());assert!(r.string().is_err());assert_eq!(r.usage.declared_allocation_bytes,0);
 let mut l=limits();l.max_read_bytes=100;let mut r=reader(1000u64.to_le_bytes().to_vec(),l);assert!(r.string().is_err());assert_eq!(r.usage.declared_allocation_bytes,0);
 let mut bytes=GGUF_TYPE_UINT8.to_le_bytes().to_vec();bytes.extend(100_000u64.to_le_bytes());let mut r=reader(bytes,limits());assert!(r.value(GGUF_TYPE_ARRAY,0).is_err());assert_eq!(r.usage.declared_allocation_bytes,0);
 let mut l=limits();l.max_declared_allocation_bytes=32;l.max_total_array_elements=1_000_000;let mut bytes=GGUF_TYPE_UINT8.to_le_bytes().to_vec();bytes.extend(100_000u64.to_le_bytes());let mut r=reader(bytes,l);assert!(r.value(GGUF_TYPE_ARRAY,0).is_err());assert_eq!(r.usage.declared_allocation_bytes,0);
 assert!(parse(Cursor::new(fixed(1025,0)),Some(limits())).is_err());assert!(parse(Cursor::new(fixed(0,1025)),Some(limits())).is_err());
 let mut l=limits();l.max_read_bytes=24;l.max_declared_allocation_bytes=0;let error=parse(Cursor::new(fixed(1,1)),Some(l)).err().unwrap().to_string();assert!(error.contains("read budget"),"{error}");
 let mut r=reader(Vec::new(),limits());assert!(r.vector::<MetadataValue>(usize::MAX).is_err());assert_eq!(r.usage.declared_allocation_bytes,0);
}
#[test]fn aggregate_reservations_depth_and_nested_elements_are_independent(){
 let mut bytes=6u64.to_le_bytes().to_vec();bytes.extend(b"abcdef");bytes.extend(6u64.to_le_bytes());bytes.extend(b"ghijkl");let mut l=limits();l.max_declared_allocation_bytes=10;let mut r=reader(bytes,l);assert_eq!(r.string().unwrap(),"abcdef");assert!(r.string().is_err());assert_eq!(r.usage.declared_allocation_bytes,6);
 let mut bytes=GGUF_TYPE_ARRAY.to_le_bytes().to_vec();bytes.extend(1u64.to_le_bytes());bytes.extend(GGUF_TYPE_UINT8.to_le_bytes());bytes.extend(1u64.to_le_bytes());bytes.push(7);
 let mut l=limits();l.max_nesting_depth=1;let mut r=reader(bytes.clone(),l);assert!(r.value(GGUF_TYPE_ARRAY,0).is_err());assert_eq!(r.usage.maximum_nesting_depth,1);
 let mut l=limits();l.max_total_array_elements=1;let mut r=reader(bytes.clone(),l);assert!(r.value(GGUF_TYPE_ARRAY,0).is_err());assert_eq!(r.usage.total_array_elements,1);
 let mut r=reader(bytes,limits());assert!(matches!(r.value(GGUF_TYPE_ARRAY,0).unwrap(),MetadataValue::Array(_)));assert_eq!(r.usage.maximum_nesting_depth,2);assert_eq!(r.usage.total_array_elements,2);
 let mut r=reader(Vec::new(),limits());r.usage.declared_allocation_bytes=u64::MAX;assert!(r.charge(1).is_err());
}
#[test]fn shared_parser_matches_legacy_and_charges_duplicate_tensor_name_buffer(){
 let mut bytes=fixed(1,0);bytes.extend(3u64.to_le_bytes());bytes.extend(b"one");bytes.extend(1u32.to_le_bytes());bytes.extend(4u64.to_le_bytes());bytes.extend(GGML_TYPE_F32.to_le_bytes());bytes.extend(0u64.to_le_bytes());
 let bounded=parse(Cursor::new(bytes.clone()),Some(limits())).unwrap();let legacy=parse(Cursor::new(bytes.clone()),None).unwrap();assert_eq!(bounded.tensor_data_offset,legacy.tensor_data_offset);let a=&bounded.tensors["one"];let b=&legacy.tensors["one"];assert_eq!((&a.name,&a.shape,a.ggml_type,a.offset,a.byte_len),(&b.name,&b.shape,b.ggml_type,b.offset,b.byte_len));
 let mut l=limits();l.max_declared_allocation_bytes=bounded.usage.declared_allocation_bytes-1;assert!(parse(Cursor::new(bytes.clone()),Some(l)).is_err());
 let mut l=limits();l.max_read_bytes=bounded.usage.read_bytes-1;assert!(parse(Cursor::new(bytes),Some(l)).is_err());
}
#[test]fn bounded_same_fd_entry_does_not_prefetch_payload_and_keeps_usage(){
 use std::{fs::OpenOptions,io::Write,sync::atomic::{AtomicU64,Ordering}};
 static NEXT:AtomicU64=AtomicU64::new(1);let path=std::env::temp_dir().join(format!("mlx-header-budget-{}-{}.gguf",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
 let mut f=OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();let mut bytes=fixed(0,0);bytes.resize(32,0);bytes.extend(b"PAYLOAD_MUST_NOT_BE_READ");f.write_all(&bytes).unwrap();f.flush().unwrap();
 let mut l=limits();l.max_read_bytes=24;let parsed=GgufFile::from_file_with_limits(f,l).unwrap();assert_eq!(parsed.bounded_header_usage().unwrap().read_bytes,24);assert_eq!(parsed.tensor_data_offset(),32);
 // Inspect the retained underlying FD cursor, not BufReader's logical position.
 assert_eq!(parsed.reader.lock().unwrap().get_mut().stream_position().unwrap(),24);
 drop(parsed);std::fs::remove_file(path).unwrap();
}
