//! GGUF v3 file format parser.
//!
//! Parses GGUF headers, metadata, and tensor info on open.  Tensor data is
//! loaded lazily on demand into [`MlxBuffer`]s — either as raw GGML blocks
//! (for GPU quantized matmul) or dequantized to F32 (for norm weights etc.).
//!
//! # Example
//!
//! ```ignore
//! use mlx_native::gguf::GgufFile;
//! use std::path::Path;
//!
//! let gguf = GgufFile::open(Path::new("model.gguf"))?;
//! let names = gguf.tensor_names();
//! let buf = gguf.load_tensor("blk.0.attn_q.weight", &device)?;
//! let norm = gguf.load_tensor_f32("blk.0.attn_norm.weight", &device)?;
//! ```

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use half::f16;
use memmap2::{Mmap, MmapOptions};

use crate::ops::quantized_matmul_ggml::GgmlType;
use crate::{DType, MlxBuffer, MlxBufferPool, MlxDevice, MlxError, Result};

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

/// GGUF magic number: "GGUF" as little-endian u32 (bytes: 0x47 0x47 0x55 0x46).
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF version we support.
const GGUF_VERSION: u32 = 3;

/// Default alignment for the tensor data section.
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata key that overrides the default alignment.
const GGUF_ALIGNMENT_KEY: &str = "general.alignment";

// ---------------------------------------------------------------------------
// GGUF metadata value type IDs
// ---------------------------------------------------------------------------

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// ---------------------------------------------------------------------------
// GGML type IDs (from ggml.h)
// ---------------------------------------------------------------------------

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q5_1: u32 = 7;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q2_K: u32 = 10;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_I16: u32 = 17;
const GGML_TYPE_IQ4_NL: u32 = 20;
const GGML_TYPE_IQ4_XS: u32 = 23;
const GGML_TYPE_I32: u32 = 26;

/// IQ4_NL non-linear codebook constants. 16 signed entries selected by
/// 4-bit indices in `block_iq4_nl::qs`. Verified byte-equal with
/// `/opt/llama.cpp/ggml/src/ggml-common.h:1109-1112`. ADR-022 Phase 1.
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Try to interpret this value as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret this value as a u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Uint8(v) => Some(*v as u32),
            MetadataValue::Uint16(v) => Some(*v as u32),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to interpret this value as an f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Tensor shape in logical outermost-first order. GGUF stores dimensions
    /// innermost-first; the parser reverses them exactly once at the boundary.
    pub shape: Vec<usize>,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
    /// Total byte length of this tensor's data.
    pub byte_len: usize,
}

/// A parsed GGUF file, ready for lazy tensor loading.
///
/// The file is kept open so that tensor data can be read on demand via
/// [`load_tensor`](GgufFile::load_tensor) and
/// [`load_tensor_f32`](GgufFile::load_tensor_f32).
pub struct GgufFile {
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorInfo>,
    /// Absolute byte offset in the file where tensor data begins.
    tensor_data_offset: u64,
    reader: Mutex<BufReader<std::fs::File>>,
}

/// File-backed Metal segments and typed tensor views for one GGUF/device pair.
///
/// Large models are split only when they exceed Metal's per-buffer limit.
/// Individual tensors are views into these shared resources, matching the
/// resource topology used by llama.cpp's Metal mmap loader.
pub struct GgufMappedTensorSet<'a> {
    gguf: &'a GgufFile,
    segments: Vec<MappedTensorSegment>,
}

struct MappedTensorSegment {
    file_start: usize,
    file_end: usize,
    buffer: MlxBuffer,
}

impl GgufMappedTensorSet<'_> {
    /// Return a typed view of a raw GGUF tensor without allocating or copying
    /// its payload.
    pub fn load_tensor(&self, name: &str) -> Result<MlxBuffer> {
        let info = self.gguf.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let file_start = self
            .gguf
            .tensor_data_offset
            .checked_add(info.offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| {
                MlxError::InvalidArgument(format!(
                    "tensor '{}' file offset does not fit this host",
                    info.name
                ))
            })?;
        let file_end = file_start.checked_add(info.byte_len).ok_or_else(|| {
            MlxError::InvalidArgument(format!("tensor '{}' file range overflow", info.name))
        })?;
        let segment = self
            .segments
            .iter()
            .find(|segment| segment.file_start <= file_start && file_end <= segment.file_end)
            .ok_or_else(|| {
                MlxError::InvalidArgument(format!(
                    "tensor '{}' is not contained in a mapped GGUF segment",
                    info.name
                ))
            })?;
        segment.buffer.data_view(
            file_start - segment.file_start,
            info.byte_len,
            raw_tensor_dtype(info.ggml_type),
            info.shape.clone(),
        )
    }

    /// Number of shared Metal resources backing all tensor views.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

// ---------------------------------------------------------------------------
// Low-level read helpers
// ---------------------------------------------------------------------------

/// Read a little-endian u8.
fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read u8: {e}")))?;
    Ok(buf[0])
}

/// Read a little-endian i8.
fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

/// Read a little-endian u16.
fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read u16: {e}")))?;
    Ok(u16::from_le_bytes(buf))
}

/// Read a little-endian i16.
fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read i16: {e}")))?;
    Ok(i16::from_le_bytes(buf))
}

/// Read a little-endian u32.
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read u32: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian i32.
fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read i32: {e}")))?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read u64: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a little-endian i64.
fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read i64: {e}")))?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a little-endian f32.
fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read f32: {e}")))?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a little-endian f64.
fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read f64: {e}")))?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a GGUF-format string: u64 length followed by UTF-8 bytes (not
/// null-terminated).
fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    if len > 256 * 1024 * 1024 {
        return Err(MlxError::GgufParseError(format!(
            "string length {len} exceeds 256 MiB safety limit"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| MlxError::GgufParseError(format!("read string bytes: {e}")))?;
    String::from_utf8(buf)
        .map_err(|e| MlxError::GgufParseError(format!("invalid UTF-8 in string: {e}")))
}

// ---------------------------------------------------------------------------
// Metadata value parsing
// ---------------------------------------------------------------------------

/// Read a single metadata value of the given type.
fn read_metadata_value<R: Read>(r: &mut R, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        GGUF_TYPE_UINT8 => Ok(MetadataValue::Uint8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(MetadataValue::Int8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(MetadataValue::Uint16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(MetadataValue::Int16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(MetadataValue::Uint32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(MetadataValue::Int32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(MetadataValue::Float32(read_f32(r)?)),
        GGUF_TYPE_BOOL => {
            let byte = read_u8(r)?;
            Ok(MetadataValue::Bool(byte != 0))
        }
        GGUF_TYPE_STRING => Ok(MetadataValue::String(read_gguf_string(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            if count > 64 * 1024 * 1024 {
                return Err(MlxError::GgufParseError(format!(
                    "array count {count} exceeds 64M element safety limit"
                )));
            }
            let mut elems = Vec::with_capacity(count);
            for _ in 0..count {
                elems.push(read_metadata_value(r, elem_type)?);
            }
            Ok(MetadataValue::Array(elems))
        }
        GGUF_TYPE_UINT64 => Ok(MetadataValue::Uint64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(MetadataValue::Int64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(MetadataValue::Float64(read_f64(r)?)),
        other => Err(MlxError::GgufParseError(format!(
            "unknown metadata value type {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// GGML type mapping
// ---------------------------------------------------------------------------

/// Map a GGML type ID (u32 from the GGUF file) to our `GgmlType` enum.
fn ggml_type_from_u32(id: u32) -> Result<GgmlType> {
    match id {
        GGML_TYPE_F32 => Ok(GgmlType::F32),
        GGML_TYPE_F16 => Ok(GgmlType::F16),
        GGML_TYPE_Q4_0 => Ok(GgmlType::Q4_0),
        GGML_TYPE_Q5_1 => Ok(GgmlType::Q5_1),
        GGML_TYPE_Q8_0 => Ok(GgmlType::Q8_0),
        GGML_TYPE_Q2_K => Ok(GgmlType::Q2_K),
        GGML_TYPE_Q3_K => Ok(GgmlType::Q3_K),
        GGML_TYPE_Q4_K => Ok(GgmlType::Q4_K),
        GGML_TYPE_Q5_K => Ok(GgmlType::Q5_K),
        GGML_TYPE_Q6_K => Ok(GgmlType::Q6_K),
        GGML_TYPE_I16 => Ok(GgmlType::I16),
        GGML_TYPE_IQ4_NL => Ok(GgmlType::IQ4_NL),
        GGML_TYPE_IQ4_XS => Ok(GgmlType::IQ4_XS),
        GGML_TYPE_I32 => Ok(GgmlType::I32),
        other => Err(MlxError::GgufParseError(format!(
            "unsupported GGML type ID {other}"
        ))),
    }
}

/// Compute the byte length of a tensor from its shape and GGML type.
///
/// For quantized types, the logical innermost dimension (`shape.last()` after
/// the parser's boundary reversal) must be divisible by the block's element
/// count. GGML blocks never span rows.
fn compute_byte_len(shape: &[usize], ggml_type: GgmlType) -> Result<usize> {
    let (&innermost, outer_shape) = shape.split_last().ok_or_else(|| {
        MlxError::GgufParseError("tensor shape must contain at least one dimension".into())
    })?;
    if innermost == 0 || outer_shape.contains(&0) {
        return Err(MlxError::GgufParseError(
            "tensor shape dimensions must be non-zero".into(),
        ));
    }

    let elems_per_block = ggml_type.block_values() as usize;
    let bytes_per_block = ggml_type.block_bytes() as usize;

    if innermost % elems_per_block != 0 {
        return Err(MlxError::GgufParseError(format!(
            "innermost dimension {innermost} not divisible by block size {elems_per_block} \
             for type {:?}; GGML blocks cannot span rows",
            ggml_type
        )));
    }

    let outer_rows = outer_shape.iter().try_fold(1usize, |rows, dimension| {
        rows.checked_mul(*dimension)
            .ok_or_else(|| MlxError::GgufParseError("tensor outer-row count overflow".into()))
    })?;
    let blocks_per_row = innermost / elems_per_block;
    outer_rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(bytes_per_block))
        .ok_or_else(|| MlxError::GgufParseError("tensor byte length overflow".into()))
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Convert a raw little-endian f16 (2 bytes) to f32.
#[inline]
fn f16_from_le_bytes(bytes: [u8; 2]) -> f32 {
    f16::from_le_bytes(bytes).to_f32()
}

/// Dequantize Q4_0 blocks to f32.
///
/// Block layout (18 bytes, 32 elements):
///   f16 d          — scale
///   u8  qs[16]     — packed 4-bit values (low nibble = first 16, high nibble = last 16)
fn dequantize_q4_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_ELEMS: usize = 32;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q4_0 data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q4_0 output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_from_le_bytes([block[0], block[1]]);
        let qs = &block[2..18]; // 16 bytes

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i16 - 8;
            let x1 = (qs[j] >> 4) as i16 - 8;
            out[j] = x0 as f32 * d;
            out[j + 16] = x1 as f32 * d;
        }
    }
    Ok(())
}

/// Dequantize Q8_0 blocks to f32.
///
/// Block layout (34 bytes, 32 elements):
///   f16 d         — scale
///   i8  qs[32]    — signed 8-bit quantized values
fn dequantize_q8_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_ELEMS: usize = 32;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q8_0 data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q8_0 output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_from_le_bytes([block[0], block[1]]);
        let qs = &block[2..34]; // 32 bytes of i8

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        for j in 0..32 {
            out[j] = (qs[j] as i8) as f32 * d;
        }
    }
    Ok(())
}

/// Extract a (scale, min) pair for sub-block `j` from the 12-byte scales
/// array used by Q4_K and Q5_K.
///
/// This matches `get_scale_min_k4` from candle / llama.cpp exactly:
///
/// For j < 4:
///   scale = scales[j] & 63
///   min   = scales[j + 4] & 63
///
/// For j >= 4:
///   scale = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
///   min   = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize GGML Q2_K super-blocks to f32.
///
/// Each 84-byte block stores sixteen 16-value groups. `scales[g]` packs
/// a 4-bit multiplier in its low nibble and a 4-bit minimum multiplier
/// in its high nibble. Four consecutive groups share one two-bit plane
/// in `qs`; the two 128-value halves use separate 32-byte regions.
fn dequantize_q2_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 84;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q2_K data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q2_K output buffer too small".into(),
        ));
    }

    for block_index in 0..num_blocks {
        let block = &data[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let scales = &block[..16];
        let qs = &block[16..80];
        let d = f16_from_le_bytes([block[80], block[81]]);
        let dmin = f16_from_le_bytes([block[82], block[83]]);
        let out = &mut output[block_index * BLOCK_ELEMS..(block_index + 1) * BLOCK_ELEMS];

        for (group, &scale) in scales.iter().enumerate() {
            let scale_value = d * (scale & 0x0f) as f32;
            let min_value = dmin * (scale >> 4) as f32;
            let half = group / 8;
            let group_in_half = group % 8;
            let shift = 2 * (group_in_half / 2);
            let q_offset = half * 32 + (group_in_half % 2) * 16;

            for lane in 0..16 {
                let quant = (qs[q_offset + lane] >> shift) & 0x03;
                out[group * 16 + lane] = scale_value * quant as f32 - min_value;
            }
        }
    }

    Ok(())
}

/// Dequantize GGML Q3_K super-blocks to f32.
///
/// Each 110-byte block stores 256 values as 16 groups of 16. `qs` carries
/// the low two bits and `hmask` selects whether the decoded value is in
/// `[0, 3]` or `[-4, -1]`. The 16 group scales are signed values represented
/// by packed 6-bit integers biased by 32.
///
/// Layout and unpacking follow llama.cpp's MIT-licensed
/// `block_q3_K` / `dequantize_row_q3_K` definitions.
fn dequantize_q3_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q3_K data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q3_K output buffer too small".into(),
        ));
    }

    for block_index in 0..num_blocks {
        let block = &data[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let hmask = &block[..32];
        let qs = &block[32..96];
        let scales = &block[96..108];
        let d = f16_from_le_bytes([block[108], block[109]]);
        let out = &mut output[block_index * BLOCK_ELEMS..(block_index + 1) * BLOCK_ELEMS];

        for group in 0..16 {
            let low = if group < 8 {
                scales[group] & 0x0f
            } else {
                scales[group - 8] >> 4
            };
            let high = (scales[8 + group % 4] >> (2 * (group / 4))) & 0x03;
            let group_scale = d * ((low | (high << 4)) as f32 - 32.0);

            let half = group / 8;
            let group_in_half = group % 8;
            let q_offset = half * 32 + (group_in_half % 2) * 16;
            let shift = 2 * (group_in_half / 2);
            let high_mask = 1u8 << (group / 2);

            for lane in 0..16 {
                let low_bits = ((qs[q_offset + lane] >> shift) & 0x03) as i8;
                let quant = low_bits
                    - if hmask[(group_in_half % 2) * 16 + lane] & high_mask == 0 {
                        4
                    } else {
                        0
                    };
                out[group * 16 + lane] = group_scale * quant as f32;
            }
        }
    }

    Ok(())
}

/// Dequantize Q5_K blocks to f32.
///
/// Block layout (176 bytes, 256 elements):
///   f16 d           — super-block scale      (offset 0,  2 bytes)
///   f16 dmin        — super-block minimum     (offset 2,  2 bytes)
///   u8  scales[12]  — packed 6-bit scales/mins (offset 4,  12 bytes; shared with Q4_K)
///   u8  qh[32]      — high bits of quants      (offset 16, 32 bytes = QK_K/8)
///   u8  qs[128]     — low 4 bits of quants     (offset 48, 128 bytes = QK_K/2)
///
/// 8 sub-blocks of 32 elements each. Dequantization walks pairs of
/// sub-blocks (is, is+1), each pair consumes 32 bytes of qs (low nibble
/// for is, high nibble for is+1). The qh array is SHARED across all 4
/// pairs — the high bit per element is masked out of qh using shifting
/// selector values `u1 = 1 << (2*pair_idx)` / `u2 = 2 << (2*pair_idx)`.
///
/// Spec source: derived from `ggml/src/ggml-quants.c::dequantize_row_q5_K`.
/// No code copied — formula reproduced from the mathematical definition.
fn dequantize_q5_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 176;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q5_K data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q5_K output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];

        let d = f16_from_le_bytes([block[0], block[1]]);
        let dmin = f16_from_le_bytes([block[2], block[3]]);
        let scales = &block[4..16]; // 12 bytes
        let qh = &block[16..48]; // 32 bytes — high bit of quants
        let qs = &block[48..176]; // 128 bytes — low 4 bits

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        // Process 4 pairs of sub-blocks (256 values total).
        // u1 / u2 are the high-bit selector masks: they shift left by 2 each
        // iteration so the 4 pairs pick bits 0/1, 2/3, 4/5, 6/7 of each qh byte.
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut ys_index = 0usize;
        let mut ql_off = 0usize;

        while ql_off < 128 {
            let ql = &qs[ql_off..ql_off + 32];

            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            // Sub-block `is` (low nibble + high bit from qh masked by u1).
            for l in 0..32 {
                let low = (ql[l] & 0x0F) as u32;
                let high = if (qh[l] & u1) != 0 { 16 } else { 0 };
                let q = low + high;
                out[ys_index] = d1 * q as f32 - m1;
                ys_index += 1;
            }
            // Sub-block `is + 1` (high nibble + high bit from qh masked by u2).
            for l in 0..32 {
                let low = (ql[l] >> 4) as u32;
                let high = if (qh[l] & u2) != 0 { 16 } else { 0 };
                let q = low + high;
                out[ys_index] = d2 * q as f32 - m2;
                ys_index += 1;
            }

            is += 2;
            ql_off += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

/// Dequantize I16 tensors to f32.
///
/// Simple bitcast: `f32_val = i16_val as f32`. No scale metadata is used
/// (apex GGUF convention — raw int16 values are meaningful as-is).
///
/// ADR-013 Decision 12 originally anticipated a per-tensor scale factor,
/// but the apex GGUF does not emit one; values are stored as raw ints.
/// If future GGUFs emit a scale, extend this with a scale parameter.
fn dequantize_i16(data: &[u8], output: &mut [f32]) -> Result<()> {
    if data.len() % 2 != 0 {
        return Err(MlxError::GgufParseError(format!(
            "I16 data length {} not even",
            data.len()
        )));
    }
    let num_elements = data.len() / 2;
    if output.len() < num_elements {
        return Err(MlxError::GgufParseError(
            "I16 output buffer too small".into(),
        ));
    }
    for i in 0..num_elements {
        let v = i16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
        output[i] = v as f32;
    }
    Ok(())
}

/// Convert raw little-endian I32 values to f32 for callers that explicitly
/// request a floating-point view. The normal hash-router load path preserves
/// exact integers through [`GgufFile::load_tensor`].
fn dequantize_i32(data: &[u8], output: &mut [f32]) -> Result<()> {
    if data.len() % 4 != 0 {
        return Err(MlxError::GgufParseError(format!(
            "I32 data length {} is not divisible by four",
            data.len()
        )));
    }
    let num_elements = data.len() / 4;
    if output.len() < num_elements {
        return Err(MlxError::GgufParseError(
            "I32 output buffer too small".into(),
        ));
    }
    for (chunk, value) in data.chunks_exact(4).zip(output.iter_mut()) {
        *value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32;
    }
    Ok(())
}

/// Dequantize Q4_K blocks to f32.
///
/// Block layout (144 bytes, 256 elements):
///   f16 d          — super-block scale          (offset 0,  2 bytes)
///   f16 dmin       — super-block minimum         (offset 2,  2 bytes)
///   u8  scales[12] — packed sub-block scales/mins (offset 4, 12 bytes)
///   u8  qs[128]    — packed 4-bit quantized values (offset 16, 128 bytes)
///
/// 8 sub-blocks of 32 elements each.  Each pair of sub-blocks (64 elements)
/// shares 32 bytes of qs — the low nibble gives the first sub-block, the
/// high nibble gives the second.
fn dequantize_q4_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 144;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q4_K data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q4_K output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];

        let d = f16_from_le_bytes([block[0], block[1]]);
        let dmin = f16_from_le_bytes([block[2], block[3]]);
        let scales = &block[4..16]; // 12 bytes
        let qs = &block[16..144]; // 128 bytes

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        // Process 4 pairs of sub-blocks (8 sub-blocks total, 256 elements).
        // Each iteration handles 64 elements: sub-block `is` (low nibbles)
        // and sub-block `is+1` (high nibbles) from 32 bytes of qs.
        let mut is = 0usize;
        let mut ys_index = 0usize;

        // Step through the 256-element super-block in chunks of 64.
        // j tracks the byte offset within qs.
        let mut j = 0usize;
        while j < 128 {
            let q = &qs[j..j + 32];
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            // Low nibbles: sub-block `is` (32 elements)
            for byte in q.iter() {
                out[ys_index] = d1 * (*byte & 0xF) as f32 - min1;
                ys_index += 1;
            }
            // High nibbles: sub-block `is + 1` (32 elements)
            for byte in q.iter() {
                out[ys_index] = d2 * (*byte >> 4) as f32 - min2;
                ys_index += 1;
            }

            is += 2;
            j += 32;
        }
    }
    Ok(())
}

/// Dequantize Q6_K blocks to f32.
///
/// Block layout (210 bytes, 256 elements):
///   u8   ql[128]   — low 4 bits of quantized values  (offset 0, 128 bytes)
///   u8   qh[64]    — high 2 bits of quantized values  (offset 128, 64 bytes)
///   i8   scales[16] — sub-block scales                (offset 192, 16 bytes)
///   f16  d          — super-block scale               (offset 208, 2 bytes)
///
/// 256 elements organized as 2 groups of 128.  Each group of 128 has its own
/// ql[64], qh[32] region and produces 4 interleaved sub-groups of 32.
fn dequantize_q6_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 210;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q6_K data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q6_K output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208]; // i8 scales[16]
        let d = f16_from_le_bytes([block[208], block[209]]);

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        // Process in two groups of 128 (idx = 0 and idx = 1).
        for idx in 0..2 {
            let ql_base = &ql[64 * idx..];
            let qh_base = &qh[32 * idx..];
            let sc_base = &sc[8 * idx..];
            let out_base = &mut out[128 * idx..];

            for l in 0..32 {
                let is = l / 16; // 0 for l in 0..16, 1 for l in 16..32

                let q1 = ((ql_base[l] & 0xF) | ((qh_base[l] & 3) << 4)) as i8 - 32_i8;
                let q2 = ((ql_base[l + 32] & 0xF) | (((qh_base[l] >> 2) & 3) << 4)) as i8 - 32_i8;
                let q3 = ((ql_base[l] >> 4) | (((qh_base[l] >> 4) & 3) << 4)) as i8 - 32_i8;
                let q4 = ((ql_base[l + 32] >> 4) | (((qh_base[l] >> 6) & 3) << 4)) as i8 - 32_i8;

                out_base[l] = d * sc_base[is] as i8 as f32 * q1 as f32;
                out_base[l + 32] = d * sc_base[is + 2] as i8 as f32 * q2 as f32;
                out_base[l + 64] = d * sc_base[is + 4] as i8 as f32 * q3 as f32;
                out_base[l + 96] = d * sc_base[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

/// Dequantize F16 data to F32.
fn dequantize_f16(data: &[u8], output: &mut [f32]) -> Result<()> {
    if data.len() % 2 != 0 {
        return Err(MlxError::GgufParseError("F16 data length not even".into()));
    }
    let count = data.len() / 2;
    if output.len() < count {
        return Err(MlxError::GgufParseError(
            "F16 output buffer too small".into(),
        ));
    }
    for i in 0..count {
        output[i] = f16_from_le_bytes([data[2 * i], data[2 * i + 1]]);
    }
    Ok(())
}

/// Reinterpret F32 little-endian bytes into the output slice.
fn copy_f32(data: &[u8], output: &mut [f32]) -> Result<()> {
    if data.len() % 4 != 0 {
        return Err(MlxError::GgufParseError(
            "F32 data length not multiple of 4".into(),
        ));
    }
    let count = data.len() / 4;
    if output.len() < count {
        return Err(MlxError::GgufParseError(
            "F32 output buffer too small".into(),
        ));
    }
    for i in 0..count {
        output[i] = f32::from_le_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ]);
    }
    Ok(())
}

/// Dequantize Q5_1 blocks to f32.
///
/// Block layout (24 bytes, 32 elements):
///   f16 d   — block scale            (offset 0,  2 bytes)
///   f16 m   — block min term         (offset 2,  2 bytes)
///   u32 qh  — high-bit pack          (offset 4,  4 bytes)
///   u8  qs[16] — packed 4-bit lo nibbles (offset 8, 16 bytes)
///
/// Per-element: `out[j]      = d * x0 + m`, `out[j + 16] = d * x1 + m`,
/// where `x0 = (qs[j] & 0x0F) | ((qh >> j) << 4) & 0x10`,
///       `x1 = (qs[j] >> 4)  | ((qh >> (j + 12)) & 0x10)`.
///
/// Reference: `/opt/llama.cpp/ggml/src/ggml-quants.c:464` `dequantize_row_q5_1`.
/// ADR-022 Phase 1.
fn dequantize_q5_1(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 24;
    const BLOCK_ELEMS: usize = 32;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "Q5_1 data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "Q5_1 output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];

        let d = f16_from_le_bytes([block[0], block[1]]);
        let m = f16_from_le_bytes([block[2], block[3]]);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24]; // 16 bytes

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        for j in 0..(BLOCK_ELEMS / 2) {
            // High-bit packed: bit j of qh contributes to position j;
            // bit (j + 16) contributes to position j + 16. Mirrors
            // `dequantize_row_q5_1` byte-for-byte.
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh_0) as i32;
            let x1 = ((qs[j] >> 4) | xh_1) as i32;
            out[j] = (x0 as f32) * d + m;
            out[j + BLOCK_ELEMS / 2] = (x1 as f32) * d + m;
        }
    }
    Ok(())
}

/// Dequantize IQ4_NL blocks to f32.
///
/// Block layout (18 bytes, 32 elements):
///   f16 d      — block scale                (offset 0,  2 bytes)
///   u8  qs[16] — 16 × pair of 4-bit indices (offset 2, 16 bytes)
///
/// Per-element: `out[j]      = d * KVALUES_IQ4_NL[qs[j] & 0x0F]`,
///              `out[j + 16] = d * KVALUES_IQ4_NL[qs[j] >> 4]`.
///
/// Reference: `/opt/llama.cpp/ggml/src/ggml-quants.c:2649` `dequantize_row_iq4_nl`.
/// Codebook table verified against `ggml-common.h:1109-1112`. ADR-022 Phase 1.
fn dequantize_iq4_nl(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_ELEMS: usize = 32;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "IQ4_NL data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "IQ4_NL output buffer too small".into(),
        ));
    }

    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];

        let d = f16_from_le_bytes([block[0], block[1]]);
        let qs = &block[2..18];

        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];

        for j in 0..(BLOCK_ELEMS / 2) {
            let lo = (qs[j] & 0x0F) as usize;
            let hi = (qs[j] >> 4) as usize;
            out[j] = d * KVALUES_IQ4_NL[lo] as f32;
            out[j + BLOCK_ELEMS / 2] = d * KVALUES_IQ4_NL[hi] as f32;
        }
    }
    Ok(())
}

/// Dequantize raw IQ4_XS bytes to f32. Pure-Rust mirror of
/// `dequantize_row_iq4_xs` at `/opt/llama.cpp/ggml/src/ggml-quants.c:2667`.
///
/// Block layout (256-element super-block, 136 bytes):
///   - d:        2 bytes f16 super-block scale
///   - scales_h: 2 bytes u16, holds 8 × 2-bit sub-block scale tops
///   - scales_l: 4 bytes, holds 8 × 4-bit sub-block scale low nibbles
///     (two sub-blocks per byte: low nibble = sub-block 2k, high nibble = 2k+1)
///   - qs:     128 bytes, nibble-packed 4-bit codebook indices
///
/// Per sub-block ib32 ∈ [0,7]:
///   ls = (scales_l[ib32/2] >> 4*(ib32%2)) & 0xf
///      | ((scales_h >> 2*ib32) & 3) << 4
///   dl = d * (ls - 32)
/// Per element pair (j in [0,15]) in sub-block ib32:
///   out[j]      = dl * KVALUES_IQ4_NL[qs[16*ib32 + j] & 0xf]
///   out[j + 16] = dl * KVALUES_IQ4_NL[qs[16*ib32 + j] >> 4]
///
/// Codebook is shared with IQ4_NL. ADR-033 §Pi 2026-05-22.
fn dequantize_iq4_xs(data: &[u8], output: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 136;
    const BLOCK_ELEMS: usize = 256;

    if data.len() % BLOCK_BYTES != 0 {
        return Err(MlxError::GgufParseError(format!(
            "IQ4_XS data length {} not divisible by block size {BLOCK_BYTES}",
            data.len()
        )));
    }
    let num_blocks = data.len() / BLOCK_BYTES;
    if output.len() < num_blocks * BLOCK_ELEMS {
        return Err(MlxError::GgufParseError(
            "IQ4_XS output buffer too small".into(),
        ));
    }
    for i in 0..num_blocks {
        let block = &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d = f16_from_le_bytes([block[0], block[1]]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..];
        let out = &mut output[i * BLOCK_ELEMS..(i + 1) * BLOCK_ELEMS];
        for ib32 in 0..(BLOCK_ELEMS / 32) {
            let lo_nibble = (scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf;
            let hi_two = ((scales_h >> (2 * ib32)) & 0x3) as u8;
            let ls = (lo_nibble | (hi_two << 4)) as i32;
            let dl = d * ((ls - 32) as f32);
            let qs_sub = &qs[16 * ib32..16 * (ib32 + 1)];
            let out_sub = &mut out[32 * ib32..32 * (ib32 + 1)];
            for j in 0..16 {
                let lo = (qs_sub[j] & 0x0f) as usize;
                let hi = (qs_sub[j] >> 4) as usize;
                out_sub[j] = dl * KVALUES_IQ4_NL[lo] as f32;
                out_sub[j + 16] = dl * KVALUES_IQ4_NL[hi] as f32;
            }
        }
    }
    Ok(())
}

/// Test-only export of `dequantize_q5_1` for ADR-022 parity tests in
/// `/opt/mlx-native/tests/adr_022_phase1_dequant_parity.rs`. Hidden
/// behind a doc(hidden) marker so it's not part of the public API but
/// is accessible from integration tests via crate::gguf.
#[doc(hidden)]
pub fn test_only_dequantize_q5_1(data: &[u8], output: &mut [f32]) -> Result<()> {
    dequantize_q5_1(data, output)
}

/// Test-only export of `dequantize_iq4_xs` for ADR-033 §Pi parity tests.
#[doc(hidden)]
pub fn test_only_dequantize_iq4_xs(data: &[u8], output: &mut [f32]) -> Result<()> {
    dequantize_iq4_xs(data, output)
}

/// Test-only export of `dequantize_iq4_nl` for ADR-022 parity tests.
#[doc(hidden)]
pub fn test_only_dequantize_iq4_nl(data: &[u8], output: &mut [f32]) -> Result<()> {
    dequantize_iq4_nl(data, output)
}

/// Test-only accessor for `KVALUES_IQ4_NL` so parity tests can pin the
/// codebook bytes against the llama.cpp source of truth.
#[doc(hidden)]
pub fn test_only_kvalues_iq4_nl() -> [i8; 16] {
    KVALUES_IQ4_NL
}

/// Test-only export of `dequantize_to_f32` for ADR-022 Phase-2 Q5_K
/// dense parity tests. Routes through the same dispatch as the
/// production load path. Hidden from rustdoc.
#[doc(hidden)]
pub fn test_only_dequantize(data: &[u8], ggml_type: GgmlType, output: &mut [f32]) -> Result<()> {
    dequantize_to_f32(data, ggml_type, output)
}

/// Test-only export for pinning GGUF type-ID and byte-sizing contracts.
#[doc(hidden)]
pub fn test_only_ggml_type_from_u32(id: u32) -> Result<GgmlType> {
    ggml_type_from_u32(id)
}

/// Test-only export for pinning quantized tensor byte sizing.
#[doc(hidden)]
pub fn test_only_compute_byte_len(shape: &[usize], ggml_type: GgmlType) -> Result<usize> {
    compute_byte_len(shape, ggml_type)
}

/// Dequantize raw GGML block data to f32.
fn dequantize_to_f32(data: &[u8], ggml_type: GgmlType, output: &mut [f32]) -> Result<()> {
    match ggml_type {
        GgmlType::F32 => copy_f32(data, output),
        GgmlType::F16 => dequantize_f16(data, output),
        GgmlType::Q4_0 => dequantize_q4_0(data, output),
        GgmlType::Q8_0 => dequantize_q8_0(data, output),
        GgmlType::Q2_K => dequantize_q2_k(data, output),
        GgmlType::Q3_K => dequantize_q3_k(data, output),
        GgmlType::Q4_K => dequantize_q4_k(data, output),
        GgmlType::Q6_K => dequantize_q6_k(data, output),
        GgmlType::Q5_K => dequantize_q5_k(data, output),
        GgmlType::I16 => dequantize_i16(data, output),
        GgmlType::I32 => dequantize_i32(data, output),
        GgmlType::Q5_1 => dequantize_q5_1(data, output),
        GgmlType::IQ4_NL => dequantize_iq4_nl(data, output),
        GgmlType::IQ4_XS => dequantize_iq4_xs(data, output),
    }
}

// ---------------------------------------------------------------------------
// GgufFile implementation
// ---------------------------------------------------------------------------

impl GgufFile {
    /// Open and parse a GGUF v3 file.
    ///
    /// This reads the full header (magic, version, tensor count, metadata KV
    /// pairs, tensor info entries) but does **not** read any tensor data.
    /// Tensor data is loaded lazily via [`load_tensor`](Self::load_tensor) or
    /// [`load_tensor_f32`](Self::load_tensor_f32).
    ///
    /// # Errors
    ///
    /// Returns `MlxError::IoError` if the file cannot be opened.
    /// Returns `MlxError::GgufParseError` if the file is not valid GGUF v3.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            MlxError::IoError(format!("cannot open GGUF file '{}': {e}", path.display()))
        })?;
        let mut reader = BufReader::new(file);

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

        let tensor_count = read_u64(&mut reader)? as usize;
        let metadata_kv_count = read_u64(&mut reader)? as usize;

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

        // --- Metadata KV pairs ---
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader)?;
            let value_type = read_u32(&mut reader)?;
            let value = read_metadata_value(&mut reader, value_type)?;
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
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)? as usize;

            if n_dims > 8 {
                return Err(MlxError::GgufParseError(format!(
                    "tensor '{name}' has {n_dims} dimensions (max 8)"
                )));
            }

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut reader)? as usize);
            }
            // GGUF stores dimensions innermost-first (column-major order).
            // Reverse to match the [rows, cols] convention used by candle
            // and by the rest of hf2q's weight loading code.
            shape.reverse();

            let ggml_type_id = read_u32(&mut reader)?;
            let ggml_type = ggml_type_from_u32(ggml_type_id)
                .map_err(|e| MlxError::GgufParseError(format!("tensor '{name}': {e}")))?;

            let offset = read_u64(&mut reader)?;
            let byte_len = compute_byte_len(&shape, ggml_type)
                .map_err(|e| MlxError::GgufParseError(format!("tensor '{name}': {e}")))?;

            if tensors.contains_key(&name) {
                return Err(MlxError::GgufParseError(format!(
                    "duplicate tensor name '{name}' in GGUF directory"
                )));
            }
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
        let tensor_data_offset = align_offset(pos, alignment);

        Ok(GgufFile {
            metadata,
            tensors,
            tensor_data_offset,
            reader: Mutex::new(reader),
        })
    }

    // -----------------------------------------------------------------------
    // Metadata accessors
    // -----------------------------------------------------------------------

    /// Look up a metadata value by key.
    pub fn metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Look up a metadata string value by key.
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Look up a metadata u32 value by key.
    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Look up a metadata f32 value by key.
    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    // -----------------------------------------------------------------------
    // Tensor info accessors
    // -----------------------------------------------------------------------

    /// Return the names of all tensors in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Look up info for a specific tensor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Number of metadata key-value pairs.
    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }

    /// Absolute byte offset (from the start of the file) where the
    /// tensor data region begins. Add a [`TensorInfo::offset`] to this
    /// to get the absolute on-disk offset of a tensor's first byte —
    /// useful for tests that want to verify raw payload bytes without
    /// going through `load_tensor` (which requires an `MlxDevice`).
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    // -----------------------------------------------------------------------
    // Tensor loading
    // -----------------------------------------------------------------------

    /// Read raw tensor bytes from the file.
    ///
    /// This is a private helper that seeks to the tensor's location and reads
    /// `byte_len` bytes.
    fn read_tensor_bytes(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let abs_offset = self
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| {
                MlxError::GgufParseError(format!(
                    "tensor '{}' absolute file offset overflow",
                    info.name
                ))
            })?;
        let byte_len = u64::try_from(info.byte_len).map_err(|_| {
            MlxError::GgufParseError(format!(
                "tensor '{}' byte length is not representable",
                info.name
            ))
        })?;
        let abs_end = abs_offset.checked_add(byte_len).ok_or_else(|| {
            MlxError::GgufParseError(format!(
                "tensor '{}' absolute file range overflow",
                info.name
            ))
        })?;
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| MlxError::GgufParseError("reader mutex poisoned".into()))?;
        let file_len = reader
            .get_ref()
            .metadata()
            .map_err(|error| {
                MlxError::IoError(format!(
                    "stat GGUF before reading tensor '{}': {error}",
                    info.name
                ))
            })?
            .len();
        if abs_end > file_len {
            return Err(MlxError::GgufParseError(format!(
                "tensor '{}' range [{abs_offset}, {abs_end}) exceeds file length {file_len}",
                info.name
            )));
        }

        reader
            .seek(SeekFrom::Start(abs_offset))
            .map_err(|e| MlxError::IoError(format!("seek to tensor '{}': {e}", info.name)))?;

        let mut buf = vec![0u8; info.byte_len];
        reader.read_exact(&mut buf).map_err(|e| {
            MlxError::IoError(format!(
                "read tensor '{}' ({} bytes at offset {}): {e}",
                info.name, info.byte_len, abs_offset
            ))
        })?;

        Ok(buf)
    }

    /// Read one tensor's exact packed payload bytes without allocating a Metal
    /// buffer.
    ///
    /// The returned bytes exclude GGUF alignment padding and are read from the
    /// checked tensor region described by [`TensorInfo`]. This is intended for
    /// artifact verification, conversion receipts, and other host-side
    /// provenance work that must bind the bytes actually stored on disk.
    pub fn read_tensor_bytes_host(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;
        self.read_tensor_bytes(info)
    }

    /// Decode one tensor to host-resident F32 values without requiring Metal.
    ///
    /// This uses the same GGML decoder as [`Self::load_tensor_f32`], but
    /// returns an owned vector. It is the canonical device-free path for
    /// computing logical tensor hashes from finalized GGUF payload bytes.
    pub fn read_tensor_f32_host(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let data = self.read_tensor_bytes(info)?;
        let total_elements = info
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| {
                MlxError::GgufParseError(format!("tensor '{}' element count overflow", info.name))
            })?;
        if total_elements == 0 {
            return Err(MlxError::GgufParseError(format!(
                "tensor '{}' has zero elements",
                info.name
            )));
        }
        let mut values = vec![0.0_f32; total_elements];
        dequantize_to_f32(&data, info.ggml_type, &mut values)?;
        Ok(values)
    }

    /// Map one tensor through its own virtual-address range.
    ///
    /// Metal resources must not overlap in virtual memory. Multiple tensors
    /// commonly share one file page, so reusing a whole-file mapping for every
    /// tensor would create aliased `MTLBuffer` resources. Independent mmap
    /// views retain file-backed paging while giving each resource a distinct
    /// address range.
    fn tensor_mapping(&self, abs_offset: usize, byte_len: usize) -> Result<(Arc<Mmap>, usize)> {
        let page_size = host_page_size()?;
        let abs_end = abs_offset
            .checked_add(byte_len)
            .ok_or_else(|| MlxError::InvalidArgument("GGUF tensor file range overflow".into()))?;
        let aligned_start = abs_offset / page_size * page_size;
        let offset_in_mapping = abs_offset - aligned_start;
        let mapping_len = offset_in_mapping.checked_add(byte_len).ok_or_else(|| {
            MlxError::InvalidArgument("GGUF tensor mapping length overflow".into())
        })?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| MlxError::GgufParseError("reader mutex poisoned".into()))?;
        let file_len = reader
            .get_ref()
            .metadata()
            .map_err(|e| MlxError::IoError(format!("stat GGUF before mapping: {e}")))?
            .len();
        if u64::try_from(abs_end).map_or(true, |end| end > file_len) {
            return Err(MlxError::GgufParseError(format!(
                "GGUF tensor range [{abs_offset}, {abs_end}) exceeds file length {file_len}"
            )));
        }
        // SAFETY: `aligned_start` is page-aligned, the complete logical range
        // was checked against the current file length above, and the read-only
        // mapping owns its virtual range independently of the file descriptor.
        let mapping = unsafe {
            MmapOptions::new()
                .offset(aligned_start as u64)
                .len(mapping_len)
                .map(reader.get_ref())
        }
        .map_err(|e| MlxError::IoError(format!("map GGUF tensor range: {e}")))?;
        Ok((Arc::new(mapping), offset_in_mapping))
    }

    /// Map all GGUF tensor payloads into the minimum practical number of
    /// no-copy Metal resources for this device.
    ///
    /// The returned buffers are read-only weight views. Callers must not bind
    /// them as kernel outputs or otherwise request GPU writes to the mapped
    /// file pages.
    pub fn map_tensor_data<'a>(&'a self, device: &MlxDevice) -> Result<GgufMappedTensorSet<'a>> {
        let page_size = host_page_size()?;
        let max_buffer_len = usize::try_from(device.metal_device().max_buffer_length())
            .map_err(|_| MlxError::InvalidArgument("Metal buffer limit exceeds usize".into()))?;
        if max_buffer_len <= page_size * 2 {
            return Err(MlxError::BufferAllocationError {
                bytes: max_buffer_len,
            });
        }

        let mut infos: Vec<&TensorInfo> = self.tensors.values().collect();
        infos.sort_by_key(|info| info.offset);
        if infos.is_empty() {
            return Err(MlxError::GgufParseError(
                "cannot map a GGUF with no tensors".into(),
            ));
        }

        let absolute_range = |info: &TensorInfo| -> Result<(usize, usize)> {
            let start = self
                .tensor_data_offset
                .checked_add(info.offset)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| {
                    MlxError::InvalidArgument(format!(
                        "tensor '{}' file offset does not fit this host",
                        info.name
                    ))
                })?;
            let end = start.checked_add(info.byte_len).ok_or_else(|| {
                MlxError::InvalidArgument(format!("tensor '{}' file range overflow", info.name))
            })?;
            Ok((start, end))
        };
        let metal_span = |start: usize, end: usize| -> Result<usize> {
            let aligned_start = start / page_size * page_size;
            let aligned_end = end
                .checked_add(page_size - 1)
                .ok_or_else(|| MlxError::InvalidArgument("Mapped range overflow".into()))?
                / page_size
                * page_size;
            Ok(aligned_end - aligned_start)
        };

        let (mut range_start, mut range_end) = absolute_range(infos[0])?;
        if metal_span(range_start, range_end)? > max_buffer_len {
            return Err(MlxError::BufferAllocationError {
                bytes: range_end - range_start,
            });
        }
        let mut ranges = Vec::new();
        for info in infos.into_iter().skip(1) {
            let (tensor_start, tensor_end) = absolute_range(info)?;
            if metal_span(range_start, tensor_end)? <= max_buffer_len {
                range_end = range_end.max(tensor_end);
            } else {
                ranges.push((range_start, range_end));
                range_start = tensor_start;
                range_end = tensor_end;
                if metal_span(range_start, range_end)? > max_buffer_len {
                    return Err(MlxError::BufferAllocationError {
                        bytes: range_end - range_start,
                    });
                }
            }
        }
        ranges.push((range_start, range_end));

        let mut segments = Vec::with_capacity(ranges.len());
        for (file_start, file_end) in ranges {
            let data_len = file_end - file_start;
            let (mapping, offset_in_mapping) = self.tensor_mapping(file_start, data_len)?;
            let buffer = device.map_file_buffer(
                mapping,
                offset_in_mapping,
                data_len,
                DType::U8,
                vec![data_len],
            )?;
            segments.push(MappedTensorSegment {
                file_start,
                file_end,
                buffer,
            });
        }
        Ok(GgufMappedTensorSet {
            gguf: self,
            segments,
        })
    }

    /// Load a tensor as a raw buffer on the Metal device.
    ///
    /// For quantized types (Q4_0, Q8_0, Q4_K, Q6_K) the buffer contains raw
    /// GGML blocks with dtype `U8` — these are consumed directly by
    /// `quantized_matmul_ggml` kernels.
    ///
    /// F32, F16, and I32 tensors retain their corresponding typed dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor name is not found, or if reading fails.
    pub fn load_tensor(&self, name: &str, device: &MlxDevice) -> Result<MlxBuffer> {
        let info = self.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;

        let data = self.read_tensor_bytes(info)?;

        match info.ggml_type {
            GgmlType::F32 => {
                let mut buf =
                    device.alloc_buffer(info.byte_len, DType::F32, info.shape.clone())?;
                {
                    let slice: &mut [u8] = buf.as_mut_slice()?;
                    slice.copy_from_slice(&data);
                }
                Ok(buf)
            }
            GgmlType::F16 => {
                let mut buf =
                    device.alloc_buffer(info.byte_len, DType::F16, info.shape.clone())?;
                {
                    let slice: &mut [u8] = buf.as_mut_slice()?;
                    slice.copy_from_slice(&data);
                }
                Ok(buf)
            }
            GgmlType::I32 => {
                let mut buf =
                    device.alloc_buffer(info.byte_len, DType::I32, info.shape.clone())?;
                {
                    let slice: &mut [u8] = buf.as_mut_slice()?;
                    slice.copy_from_slice(&data);
                }
                Ok(buf)
            }
            GgmlType::Q4_0
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::I16
            | GgmlType::Q5_1
            | GgmlType::IQ4_NL
            // ADR-033 §Pi 2026-05-22 — IQ4_XS uses the same opaque-U8
            // on-device storage as the other quantized blocks; the
            // Metal matmul kernel reads them directly. (Kernel port:
            // Task #16. Until then, hf2q's runtime will surface a
            // typed "unsupported" at the dispatch site.)
            | GgmlType::IQ4_XS => {
                // Store raw GGML blocks as a U8 buffer. Where a Metal
                // quantized-matmul kernel exists for the type, it consumes
                // these blocks directly without an explicit dequant pass on
                // the GPU; otherwise the U8 view is opaque on-device storage
                // pending either a kernel port or a host-side dequant.
                //
                // Coverage status (ADR-022 in-flight; see ADR for the live
                // matrix). Per-type Metal kernel coverage is owned by
                // `quantized_matmul_ggml.rs` `kernel_name` / `mm_kernel_name`
                // / `mm_tensor_kernel_name` and the matmul-id counterparts;
                // host-side dequant for parity / no-kernel-yet paths is
                // wired into `dequantize_to_f32` directly above.
                let mut buf =
                    device.alloc_buffer(info.byte_len, DType::U8, info.shape.clone())?;
                {
                    let slice: &mut [u8] = buf.as_mut_slice()?;
                    slice.copy_from_slice(&data);
                }
                Ok(buf)
            }
        }
    }

    /// Load a raw tensor as a read-only, file-backed Metal buffer.
    ///
    /// This avoids copying GGUF weights into anonymous memory. Quantized
    /// tensors retain their packed GGML bytes as `U8`; F32, F16, and I32 keep
    /// their native dtype. The returned buffer owns a shared mapping reference
    /// and remains valid after `GgufFile` and the on-disk directory entry are
    /// dropped. It must only be bound as a kernel input; GPU writes to the
    /// read-only mapped pages are unsupported.
    pub fn load_tensor_mapped(&self, name: &str, device: &MlxDevice) -> Result<MlxBuffer> {
        let info = self.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;
        let dtype = raw_tensor_dtype(info.ggml_type);
        let abs_offset = self
            .tensor_data_offset
            .checked_add(info.offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| {
                MlxError::InvalidArgument(format!(
                    "tensor '{}' file offset does not fit this host",
                    info.name
                ))
            })?;

        let (mapping, offset_in_mapping) = self.tensor_mapping(abs_offset, info.byte_len)?;
        device.map_file_buffer(
            mapping,
            offset_in_mapping,
            info.byte_len,
            dtype,
            info.shape.clone(),
        )
    }

    /// Load a tensor, dequantizing to F32 on the CPU, then upload to the
    /// Metal device.
    ///
    /// This is used for norm weights, embedding tables, and other tensors
    /// where the inference kernels operate on F32 directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor name is not found, reading fails, or
    /// dequantization encounters malformed data.
    pub fn load_tensor_f32(&self, name: &str, device: &MlxDevice) -> Result<MlxBuffer> {
        let info = self.tensors.get(name).ok_or_else(|| {
            MlxError::GgufParseError(format!("tensor '{name}' not found in GGUF file"))
        })?;

        let data = self.read_tensor_bytes(info)?;
        let total_elements: usize = info.shape.iter().product();

        if total_elements == 0 {
            return Err(MlxError::GgufParseError(format!(
                "tensor '{name}' has zero elements"
            )));
        }

        let f32_byte_len = total_elements * 4;
        let mut buf = device.alloc_buffer(f32_byte_len, DType::F32, info.shape.clone())?;

        {
            let out_slice: &mut [f32] = buf.as_mut_slice()?;
            dequantize_to_f32(&data, info.ggml_type, out_slice)?;
        }

        Ok(buf)
    }

    /// Load a tensor and register its underlying Metal buffer with `pool`'s
    /// residency set, returning the [`MlxBuffer`] to the caller.
    ///
    /// This is functionally equivalent to:
    ///
    /// ```ignore
    /// let buf = gguf.load_tensor(name, device)?;
    /// pool.register_existing(device, &buf)?;
    /// ```
    ///
    /// but exists as a single call so callers don't need to reach for the
    /// underlying [`MlxBufferPool::register_existing`] API directly.  See
    /// that method's docs for the residency-set ownership contract.
    ///
    /// # Why a separate method instead of a `pool` parameter on `load_tensor`
    ///
    /// `load_tensor` has stable callers across the codebase that pass only
    /// `&MlxDevice`; making the pool registration optional via a new method
    /// keeps the existing signature wire-compatible.
    ///
    /// # Note on bucket-rounding
    ///
    /// The buffer is allocated at exactly `info.byte_len` via
    /// [`MlxDevice::alloc_buffer`](crate::MlxDevice::alloc_buffer) (no
    /// bucket-rounding) and added to the pool's residency set only —
    /// it is not placed in the recycling free list.  This is the path
    /// hf2q's static weight loader uses to gain MTLResidencySet hints
    /// without paying the 48% bucket-rounding tax that would have
    /// inflated 17 GB of weights to 25 GB.
    ///
    /// # Errors
    ///
    /// Same as [`load_tensor`](Self::load_tensor), plus any
    /// [`MlxError::InvalidArgument`] from
    /// [`MlxBufferPool::register_existing`].
    pub fn load_tensor_into_pool(
        &self,
        name: &str,
        device: &MlxDevice,
        pool: &mut MlxBufferPool,
    ) -> Result<MlxBuffer> {
        let buf = self.load_tensor(name, device)?;
        pool.register_existing(device, &buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn raw_tensor_dtype(ggml_type: GgmlType) -> DType {
    match ggml_type {
        GgmlType::F32 => DType::F32,
        GgmlType::F16 => DType::F16,
        GgmlType::I32 => DType::I32,
        GgmlType::Q4_0
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::I16
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS => DType::U8,
    }
}

fn host_page_size() -> Result<usize> {
    extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }

    // SAFETY: `getpagesize` has no arguments or side effects and is available
    // on every macOS version supported by mlx-native.
    let size = unsafe { getpagesize() };
    if size <= 0 {
        return Err(MlxError::InvalidArgument(
            "Operating system reported an invalid page size".into(),
        ));
    }
    Ok(size as usize)
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (offset + mask) & !mask
}
