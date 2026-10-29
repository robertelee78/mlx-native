//! Shared legacy/bounded header parser. No tensor payload decoding/mapping.
use super::*;
use std::{hash::Hash,io,mem::size_of};

/// Explicit limits for the additional bounded same-FD header entrypoint.
/// These account cumulative bytes read and declared allocation reservations,
/// not allocator rounding/driver/physical RAM. Existing legacy caps also apply.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct GgufHeaderLimits {
 pub max_read_bytes:u64,
 pub max_declared_allocation_bytes:u64,
 pub max_metadata_entries:u64,
 pub max_tensors:u64,
 pub max_string_bytes:u64,
 /// Aggregate elements over ALL arrays, including nested arrays.
 pub max_total_array_elements:u64,
 /// Array nesting depth; zero permits scalar/string values only. Maximum64.
 pub max_nesting_depth:u32,
}
/// Cumulative accounting: overwritten entries and temporary storage get no refund.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct GgufHeaderUsage {
 pub read_bytes:u64,pub declared_allocation_bytes:u64,
 pub total_array_elements:u64,pub maximum_nesting_depth:u32,
}
pub(super) struct Parsed<R> {
 pub metadata:HashMap<String,MetadataValue>,pub tensors:HashMap<String,TensorInfo>,
 pub tensor_data_offset:u64,pub reader:R,pub usage:GgufHeaderUsage,
}
fn bad(message:&str)->MlxError{MlxError::GgufParseError(message.into())}
struct HeaderReader<R> {inner:R,limits:Option<GgufHeaderLimits>,usage:GgufHeaderUsage}
impl<R:Read> Read for HeaderReader<R>{
 fn read(&mut self,out:&mut[u8])->io::Result<usize>{
  let requested=u64::try_from(out.len()).map_err(|_|io::Error::new(io::ErrorKind::InvalidData,"header read length overflow"))?;
  let bound=self.usage.read_bytes.checked_add(requested).ok_or_else(||io::Error::new(io::ErrorKind::InvalidData,"header read counter overflow"))?;
  if self.limits.is_some_and(|l|bound>l.max_read_bytes){return Err(io::Error::new(io::ErrorKind::InvalidData,"aggregate header read budget exceeded"));}
  let count=self.inner.read(out)?;self.usage.read_bytes=self.usage.read_bytes.checked_add(count as u64).ok_or_else(||io::Error::new(io::ErrorKind::InvalidData,"header read counter overflow"))?;Ok(count)
 }
}
impl<R:Seek> Seek for HeaderReader<R>{fn seek(&mut self,where_:SeekFrom)->io::Result<u64>{self.inner.seek(where_)}fn stream_position(&mut self)->io::Result<u64>{self.inner.stream_position()}}
impl<R:Read> HeaderReader<R>{
 fn charge(&mut self,bytes:usize)->Result<()>{
  let total=self.usage.declared_allocation_bytes.checked_add(u64::try_from(bytes).map_err(|_|bad("header reservation conversion overflow"))?).ok_or_else(||bad("header reservation overflow"))?;
  if self.limits.is_some_and(|l|total>l.max_declared_allocation_bytes){return Err(bad("aggregate declared header allocation budget exceeded BEFORE allocation"));}
  self.usage.declared_allocation_bytes=total;Ok(())
 }
 fn check_read(&self,bytes:u64)->Result<()>{let total=self.usage.read_bytes.checked_add(bytes).ok_or_else(||bad("declared header wire length overflow"))?;
  if self.limits.is_some_and(|l|total>l.max_read_bytes){return Err(bad("declared header read budget exceeded BEFORE allocation"));}Ok(())
 }
 fn vector<T>(&mut self,count:usize)->Result<Vec<T>>{
  self.charge(count.checked_mul(size_of::<T>()).ok_or_else(||bad("header vector reservation overflow"))?)?;
  if self.limits.is_none(){return Ok(Vec::with_capacity(count));}
  let mut out=Vec::new();out.try_reserve_exact(count).map_err(|_|bad("bounded header vector reservation failed"))?;Ok(out)
 }
 fn table<K:Eq+Hash,V>(&mut self,count:usize)->Result<HashMap<K,V>>{
  // Explicit conservative DECLARED table reservation model: two slots/entry,
  // rounded power-of-two (minimum4), entry/control bytes and32-byte overhead.
  // It is not a promise about a future std allocator/hash-table implementation.
  let slots=if count==0{0}else{count.checked_mul(2).and_then(usize::checked_next_power_of_two).ok_or_else(||bad("header table slot overflow"))?.max(4)};
  let bytes=slots.checked_mul(size_of::<(K,V)>().checked_add(1).ok_or_else(||bad("table entry overflow"))?).and_then(|n|n.checked_add(if slots==0{0}else{32})).ok_or_else(||bad("header table reservation overflow"))?;
  self.charge(bytes)?;
  if self.limits.is_none(){return Ok(HashMap::with_capacity(count));}
  let mut out=HashMap::new();out.try_reserve(count).map_err(|_|bad("bounded header map reservation failed"))?;Ok(out)
 }
 fn string(&mut self)->Result<String>{
  let declared=read_u64(self)?;
  if declared>256*1024*1024{return Err(bad(&format!("string length {declared} exceeds 256 MiB safety limit")));}
  if self.limits.is_some_and(|l|declared>l.max_string_bytes){return Err(bad("declared header string cap exceeded BEFORE allocation"));}
  self.check_read(declared)?;let count=usize::try_from(declared).map_err(|_|bad("string length exceeds usize"))?;
  let mut out=self.vector::<u8>(count)?;out.resize(count,0);
  self.read_exact(&mut out).map_err(|e|bad(&format!("read string bytes: {e}")))?;
  String::from_utf8(out).map_err(|e|bad(&format!("invalid UTF-8 in string: {e}")))
 }
 fn value(&mut self,kind:u32,depth:u32)->Result<MetadataValue>{
  match kind{
   GGUF_TYPE_UINT8=>Ok(MetadataValue::Uint8(read_u8(self)?)),GGUF_TYPE_INT8=>Ok(MetadataValue::Int8(read_i8(self)?)),
   GGUF_TYPE_UINT16=>Ok(MetadataValue::Uint16(read_u16(self)?)),GGUF_TYPE_INT16=>Ok(MetadataValue::Int16(read_i16(self)?)),
   GGUF_TYPE_UINT32=>Ok(MetadataValue::Uint32(read_u32(self)?)),GGUF_TYPE_INT32=>Ok(MetadataValue::Int32(read_i32(self)?)),
   GGUF_TYPE_FLOAT32=>Ok(MetadataValue::Float32(read_f32(self)?)),GGUF_TYPE_BOOL=>Ok(MetadataValue::Bool(read_u8(self)?!=0)),
   GGUF_TYPE_STRING=>Ok(MetadataValue::String(self.string()?)),
   GGUF_TYPE_ARRAY=>{
    let elem=read_u32(self)?;let declared=read_u64(self)?;
    if declared>64*1024*1024{return Err(bad(&format!("array count {declared} exceeds 64M element safety limit")));}
    let next_depth=depth.checked_add(1).ok_or_else(||bad("metadata nesting depth overflow"))?;
    let aggregate=self.usage.total_array_elements.checked_add(declared).ok_or_else(||bad("aggregate array count overflow"))?;
    if self.limits.is_some_and(|l|aggregate>l.max_total_array_elements||next_depth>l.max_nesting_depth){return Err(bad("aggregate array/depth cap exceeded BEFORE allocation"));}
    if self.limits.is_some(){let min_width=match elem{GGUF_TYPE_UINT8|GGUF_TYPE_INT8|GGUF_TYPE_BOOL=>1,GGUF_TYPE_UINT16|GGUF_TYPE_INT16=>2,GGUF_TYPE_UINT32|GGUF_TYPE_INT32|GGUF_TYPE_FLOAT32=>4,GGUF_TYPE_UINT64|GGUF_TYPE_INT64|GGUF_TYPE_FLOAT64|GGUF_TYPE_STRING=>8,GGUF_TYPE_ARRAY=>12,_=>return Err(bad("unknown bounded metadata array type"))};self.check_read(declared.checked_mul(min_width).ok_or_else(||bad("array minimum wire size overflow"))?)?;}
    self.usage.total_array_elements=aggregate;self.usage.maximum_nesting_depth=self.usage.maximum_nesting_depth.max(next_depth);
    let count=usize::try_from(declared).map_err(|_|bad("array count exceeds usize"))?;let mut out=self.vector::<MetadataValue>(count)?;
    for _ in 0..count{out.push(self.value(elem,next_depth)?);}Ok(MetadataValue::Array(out))
   },
   GGUF_TYPE_UINT64=>Ok(MetadataValue::Uint64(read_u64(self)?)),GGUF_TYPE_INT64=>Ok(MetadataValue::Int64(read_i64(self)?)),GGUF_TYPE_FLOAT64=>Ok(MetadataValue::Float64(read_f64(self)?)),
   other=>Err(bad(&format!("unknown metadata value type {other}"))),
  }
 }
}

pub(super) fn parse<R:Read+Seek>(inner:R,limits:Option<GgufHeaderLimits>)->Result<Parsed<R>>{
 if limits.is_some_and(|l|l.max_read_bytes==0||l.max_nesting_depth>64){return Err(bad("invalid bounded header read/depth limits"));}
 let mut reader=HeaderReader{inner,limits,usage:GgufHeaderUsage::default()};
        // --- Header ---
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(MlxError::GgufParseError(format!(
                "bad magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }

        let version = read_u32(&mut reader)?;
        if version != GGUF_VERSION {
            return Err(MlxError::GgufParseError(format!(
                "unsupported GGUF version {version} (only v3 is supported)"
            )));
        }

        let tensor_declared = read_u64(&mut reader)?;
        let tensor_count=usize::try_from(tensor_declared).map_err(|_|bad("tensor count exceeds usize"))?;
        let metadata_declared=read_u64(&mut reader)?;
        let metadata_kv_count=usize::try_from(metadata_declared).map_err(|_|bad("metadata count exceeds usize"))?;
        if reader.limits.is_some_and(|l|tensor_declared>l.max_tensors||metadata_declared>l.max_metadata_entries){return Err(bad("bounded header catalog counts exceeded BEFORE allocation"));}

        // Sanity limits to prevent OOM on corrupted files.
        if tensor_count > 100_000 {
            return Err(MlxError::GgufParseError(format!(
                "tensor_count {tensor_count} exceeds 100k safety limit"
            )));
        }
        if metadata_kv_count > 1_000_000 {
            return Err(MlxError::GgufParseError(format!(
                "metadata_kv_count {metadata_kv_count} exceeds 1M safety limit"
            )));
        }

        if reader.limits.is_some(){
            // Minimum valid on-wire directory sizes, BEFORE either map reserve:
            // metadata key-length/type/smallest value=13, tensor name-length/
            // rank/type/offset=24 (zero dimensions is the minimum wire shape).
            let wire=metadata_declared.checked_mul(13).and_then(|n|tensor_declared.checked_mul(24).and_then(|m|n.checked_add(m))).ok_or_else(||bad("catalog minimum wire size overflow"))?;
            reader.check_read(wire)?;
        }

        // --- Metadata KV pairs ---
        let mut metadata = reader.table::<String,MetadataValue>(metadata_kv_count)?;
        for _ in 0..metadata_kv_count {
            let key = reader.string()?;
            let value_type = read_u32(&mut reader)?;
            let value = reader.value(value_type,0)?;
            metadata.insert(key, value);
        }

        // --- Determine alignment ---
        let alignment = metadata
            .get(GGUF_ALIGNMENT_KEY)
            .and_then(|v| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(MlxError::GgufParseError(format!(
                "alignment {alignment} is not a power of two"
            )));
        }

        // --- Tensor info entries ---
        let mut tensors = reader.table::<String,TensorInfo>(tensor_count)?;
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let n_dims=usize::try_from(read_u32(&mut reader)?).map_err(|_|bad("tensor rank exceeds usize"))?;

            if n_dims > 8 {
                return Err(if reader.limits.is_some(){bad("tensor rank exceeds8")}else{MlxError::GgufParseError(format!("tensor '{name}' has {n_dims} dimensions (max 8)"))});
            }

            let mut shape = reader.vector::<usize>(n_dims)?;
            for _ in 0..n_dims {
                shape.push(usize::try_from(read_u64(&mut reader)?).map_err(|_|bad("tensor dimension exceeds usize"))?);
            }
            // GGUF stores dimensions innermost-first (column-major order).
            // Reverse to match the [rows, cols] convention used by candle
            // and by the rest of hf2q's weight loading code.
            shape.reverse();

            let ggml_type_id = read_u32(&mut reader)?;
            let ggml_type = ggml_type_from_u32(ggml_type_id)
                .map_err(|e| if reader.limits.is_some(){bad(&format!("bounded tensor type/shape: {e}"))}else{MlxError::GgufParseError(format!("tensor '{name}': {e}"))})?;

            let offset = read_u64(&mut reader)?;
            let byte_len = compute_byte_len(&shape, ggml_type)
                .map_err(|e| if reader.limits.is_some(){bad(&format!("bounded tensor type/shape: {e}"))}else{MlxError::GgufParseError(format!("tensor '{name}': {e}"))})?;

            if tensors.contains_key(&name) {
                return Err(if reader.limits.is_some(){bad("duplicate tensor name in bounded GGUF directory")}else{MlxError::GgufParseError(format!("duplicate tensor name '{name}' in GGUF directory"))});
            }
            reader.charge(name.len())?; // second retained name buffer, BEFORE clone
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    ggml_type,
                    offset,
                    byte_len,
                },
            );
        }

        // --- Compute tensor_data_offset ---
        // The current file position is just past all tensor info entries.
        // Tensor data starts at the next alignment boundary.
        let pos = reader
            .stream_position()
            .map_err(|e| MlxError::GgufParseError(format!("stream_position: {e}")))?;
        let tensor_data_offset=if reader.limits.is_some(){pos.checked_add(alignment-1).map(|n|n/alignment*alignment).ok_or_else(||bad("header alignment overflow"))?}else{align_offset(pos,alignment)};

        Ok(Parsed{metadata,tensors,tensor_data_offset,reader:reader.inner,usage:reader.usage})
}
#[cfg(test)]mod tests;
