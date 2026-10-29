//! Pointer-free descriptors derived from the production coalescing planner.
use super::*;

/// Immutable logical tensor view into the SAME first containing segment used
/// by GgufMappedTensorSet::load_tensor. Names borrow the parsed directory.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct GgufTensorViewPlan<'a> {
 name:&'a str,segment_index:usize,file_start:usize,file_end:usize,
 logical_view_offset:usize,root_byte_offset:usize,
}
impl GgufTensorViewPlan<'_>{
 pub fn name(&self)->&str{self.name}
 pub fn segment_index(&self)->usize{self.segment_index}
 pub fn file_start(&self)->usize{self.file_start}
 pub fn file_end(&self)->usize{self.file_end}
 pub fn logical_bytes(&self)->usize{self.file_end-self.file_start}
 pub fn logical_view_offset(&self)->usize{self.logical_view_offset}
 pub fn root_byte_offset(&self)->usize{self.root_byte_offset}
}
/// One source segment. Mapping ends at logical file_end; the registered Metal
/// parent includes final page padding, which is not additional file contents.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct GgufMappedSegmentPlan {
 file_start:usize,file_end:usize,mapping_file_start:usize,registered_file_end:usize,
 root_byte_len:usize,prefix_bytes:usize,tail_padding_bytes:usize,
}
impl GgufMappedSegmentPlan{
 pub fn file_start(&self)->usize{self.file_start}
 pub fn file_end(&self)->usize{self.file_end}
 pub fn mapping_file_start(&self)->usize{self.mapping_file_start}
 pub fn mapping_file_end(&self)->usize{self.file_end}
 pub fn mapping_bytes(&self)->usize{self.file_end-self.mapping_file_start}
 pub fn registered_file_start(&self)->usize{self.mapping_file_start}
 pub fn registered_file_end(&self)->usize{self.registered_file_end}
 pub fn root_byte_len(&self)->usize{self.root_byte_len}
 pub fn prefix_bytes(&self)->usize{self.prefix_bytes}
 pub fn tail_padding_bytes(&self)->usize{self.tail_padding_bytes}
}
/// Pure same-source layout under explicitly ASSUMED geometry. This does not
/// query a device, certify an FD/stat, allocate Metal, or prove physical RAM.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct GgufMappedTensorPlan<'a> {
 page_size:usize,max_buffer_len:usize,storage:GgufMappedTensorStoragePlan,
 segments:Vec<GgufMappedSegmentPlan>,views:Vec<GgufTensorViewPlan<'a>>,
 registered_file_page_union_bytes:usize,
}
impl GgufMappedTensorPlan<'_>{
 pub fn assumed_page_size(&self)->usize{self.page_size}
 pub fn assumed_max_buffer_len(&self)->usize{self.max_buffer_len}
 pub fn storage_plan(&self)->&GgufMappedTensorStoragePlan{&self.storage}
 pub fn segments(&self)->&[GgufMappedSegmentPlan]{&self.segments}
 pub fn tensor_views(&self)->&[GgufTensorViewPlan<'_>]{&self.views}
 /// Union of page intervals in this ONE source file, not unique resident RAM.
 pub fn registered_file_page_union_bytes(&self)->usize{self.registered_file_page_union_bytes}
}
pub(super) fn build<'a>(gguf:&'a GgufFile,page_size:usize,max_buffer_len:usize)->Result<GgufMappedTensorPlan<'a>>{
 if !page_size.is_power_of_two(){return Err(MlxError::InvalidArgument("explicit page size must be a positive power of two".into()));}
 let ranges=gguf.mapped_ranges(page_size,max_buffer_len)?;
 let storage=GgufMappedTensorStoragePlan::from_ranges(&ranges)?;
 let mut segments=Vec::with_capacity(ranges.len());let mut union=0usize;let mut prior:Option<(usize,usize)>=None;
 for r in &ranges{
  let start=r.file_start/page_size*page_size;let end=start.checked_add(r.physical_byte_len).ok_or_else(||MlxError::InvalidArgument("plan registered file end overflow".into()))?;
  let tail=end.checked_sub(r.file_end).ok_or_else(||MlxError::InvalidArgument("plan tail underflow".into()))?;
  segments.push(GgufMappedSegmentPlan{file_start:r.file_start,file_end:r.file_end,mapping_file_start:start,registered_file_end:end,root_byte_len:r.physical_byte_len,prefix_bytes:r.file_start-start,tail_padding_bytes:tail});
  match prior{Some((a,b))if start<=b=>prior=Some((a,b.max(end))),Some((a,b))=>{union=union.checked_add(b-a).ok_or_else(||MlxError::InvalidArgument("page union overflow".into()))?;prior=Some((start,end));},None=>prior=Some((start,end))}
 }
 if let Some((a,b))=prior{union=union.checked_add(b-a).ok_or_else(||MlxError::InvalidArgument("page union overflow".into()))?;}
 let mut infos:Vec<&TensorInfo>=gguf.tensors.values().collect();infos.sort_by(|a,b|a.name.cmp(&b.name));
 let mut views=Vec::with_capacity(infos.len());
 for info in infos{
  let start=gguf.tensor_data_offset.checked_add(info.offset).and_then(|n|usize::try_from(n).ok()).ok_or_else(||MlxError::InvalidArgument("plan tensor offset overflow".into()))?;
  let end=start.checked_add(info.byte_len).ok_or_else(||MlxError::InvalidArgument("plan tensor end overflow".into()))?;
  let(index,segment)=segments.iter().enumerate().find(|(_,s)|s.file_start<=start&&end<=s.file_end).ok_or_else(||MlxError::InvalidArgument("tensor has no production segment".into()))?;
  views.push(GgufTensorViewPlan{name:&info.name,segment_index:index,file_start:start,file_end:end,logical_view_offset:start-segment.file_start,root_byte_offset:start-segment.mapping_file_start});
 }
 Ok(GgufMappedTensorPlan{page_size,max_buffer_len,storage,segments,views,registered_file_page_union_bytes:union})
}
#[cfg(test)]mod tests;
