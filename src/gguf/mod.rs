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
use std::sync::Mutex;

use half::f16;

use crate::ops::quantized_matmul_ggml::GgmlType;
use crate::{DType, MlxBuffer, MlxDevice, MlxError, Result};

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
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_I16: u32 = 17;

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
    /// Tensor shape, innermost dimension first (as stored in GGUF).
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
        GGML_TYPE_Q8_0 => Ok(GgmlType::Q8_0),
        GGML_TYPE_Q4_K => Ok(GgmlType::Q4_K),
        GGML_TYPE_Q5_K => Ok(GgmlType::Q5_K),
        GGML_TYPE_Q6_K => Ok(GgmlType::Q6_K),
        GGML_TYPE_I16 => Ok(GgmlType::I16),
        other => Err(MlxError::GgufParseError(format!(
            "unsupported GGML type ID {other}"
        ))),
    }
}

/// Compute the byte length of a tensor from its shape and GGML type.
///
/// For quantized types, the innermost dimension (shape[0] in GGUF's row-major
/// convention) must be divisible by the block's element count.
fn compute_byte_len(shape: &[usize], ggml_type: GgmlType) -> Result<usize> {
    let total_elements: usize = shape.iter().product();
    if total_elements == 0 {
        return Ok(0);
    }

    let elems_per_block = ggml_type.block_values() as usize;
    let bytes_per_block = ggml_type.block_bytes() as usize;

    if total_elements % elems_per_block != 0 {
        return Err(MlxError::GgufParseError(format!(
            "total elements {total_elements} not divisible by block size {elems_per_block} \
             for type {:?}",
            ggml_type
        )));
    }

    Ok((total_elements / elems_per_block) * bytes_per_block)
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
        let scales = &block[4..16];   // 12 bytes
        let qs = &block[16..144];     // 128 bytes

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
                let q2 = ((ql_base[l + 32] & 0xF) | (((qh_base[l] >> 2) & 3) << 4)) as i8
                    - 32_i8;
                let q3 = ((ql_base[l] >> 4) | (((qh_base[l] >> 4) & 3) << 4)) as i8 - 32_i8;
                let q4 = ((ql_base[l + 32] >> 4) | (((qh_base[l] >> 6) & 3) << 4)) as i8
                    - 32_i8;

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
        return Err(MlxError::GgufParseError(
            "F16 data length not even".into(),
        ));
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

/// Dequantize raw GGML block data to f32.
fn dequantize_to_f32(data: &[u8], ggml_type: GgmlType, output: &mut [f32]) -> Result<()> {
    match ggml_type {
        GgmlType::F32 => copy_f32(data, output),
        GgmlType::F16 => dequantize_f16(data, output),
        GgmlType::Q4_0 => dequantize_q4_0(data, output),
        GgmlType::Q8_0 => dequantize_q8_0(data, output),
        GgmlType::Q4_K => dequantize_q4_k(data, output),
        GgmlType::Q6_K => dequantize_q6_k(data, output),
        GgmlType::Q5_K => dequantize_q5_k(data, output),
        GgmlType::I16 => dequantize_i16(data, output),
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
            let ggml_type = ggml_type_from_u32(ggml_type_id).map_err(|e| {
                MlxError::GgufParseError(format!("tensor '{name}': {e}"))
            })?;

            let offset = read_u64(&mut reader)?;
            let byte_len = compute_byte_len(&shape, ggml_type).map_err(|e| {
                MlxError::GgufParseError(format!("tensor '{name}': {e}"))
            })?;

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

    // -----------------------------------------------------------------------
    // Tensor loading
    // -----------------------------------------------------------------------

    /// Read raw tensor bytes from the file.
    ///
    /// This is a private helper that seeks to the tensor's location and reads
    /// `byte_len` bytes.
    fn read_tensor_bytes(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let abs_offset = self.tensor_data_offset + info.offset;
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| MlxError::GgufParseError("reader mutex poisoned".into()))?;

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

    /// Load a tensor as a raw buffer on the Metal device.
    ///
    /// For quantized types (Q4_0, Q8_0, Q4_K, Q6_K) the buffer contains raw
    /// GGML blocks with dtype `U8` — these are consumed directly by
    /// `quantized_matmul_ggml` kernels.
    ///
    /// For F32 and F16 tensors the buffer has the corresponding typed dtype.
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
            GgmlType::Q4_0
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::I16 => {
                // Store raw GGML blocks as a U8 buffer. Quantized matmul
                // kernels consume these blocks directly without an explicit
                // dequant pass on the GPU; the U8 view is the on-device
                // storage contract, not an "unsupported" placeholder.
                //
                // Coverage status (2026-04-25):
                //   * Q4_0 / Q8_0 / Q4_K / Q6_K — full mat-vec + mat-mat.
                //   * Q5_K — host-side dequant-to-F32 implemented in
                //     `dequantize_q5_k` (`src/gguf/mod.rs:469`), wired into
                //     `dequantize_to_f32` at `src/gguf/mod.rs:763`. An
                //     expert-indexed mat-vec (`mv_id`) kernel exists at
                //     `src/ops/quantized_matmul_id_ggml.rs:69` and
                //     `src/shaders/quantized_matmul_id_ggml.metal:250`. The
                //     dense `mm_id` (large-batch matmul) variant for Q5_K is
                //     not yet ported — see ADR-013 Decision 12 for the
                //     remaining gap.
                //   * I16 — held opaque; no dequant kernel has landed yet.
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
        let mut buf =
            device.alloc_buffer(f32_byte_len, DType::F32, info.shape.clone())?;

        {
            let out_slice: &mut [f32] = buf.as_mut_slice()?;
            dequantize_to_f32(&data, info.ggml_type, out_slice)?;
        }

        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Round `offset` up to the next multiple of `alignment`.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (offset + mask) & !mask
}
