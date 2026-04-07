//! Weight loading from safetensors files into Metal GPU buffers.
//!
//! This module provides utilities for loading quantized model weights from
//! the [safetensors](https://huggingface.co/docs/safetensors) file format
//! into Metal `StorageModeShared` buffers for GPU inference.
//!
//! # Architecture
//!
//! The loading pipeline is:
//!
//! 1. Memory-map the safetensors file(s) via `memmap2` (no full read into RAM).
//! 2. Parse the header to discover tensor names, shapes, dtypes, and byte offsets.
//! 3. For each tensor, create a Metal `StorageModeShared` buffer and copy the
//!    raw bytes from the mmap region into it.
//! 4. Attach quantization metadata (bits, group_size) from the
//!    `quantization_config.json` file.
//!
//! # Zero-Copy Consideration
//!
//! On Apple Silicon, Metal shared-mode buffers reside in unified memory.  We
//! *could* create a Metal buffer that wraps the mmap pointer directly, but this
//! is unsafe because the mmap lifetime is tied to the file mapping.  Instead we
//! copy the tensor bytes into a fresh Metal buffer, which is a single memcpy on
//! unified memory and guarantees the buffer outlives the file mapping.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use metal::MTLResourceOptions;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::error::{MlxError, Result};

// ---------------------------------------------------------------------------
// Quantization config parsing
// ---------------------------------------------------------------------------

/// Per-tensor quantization configuration from `quantization_config.json`.
///
/// This mirrors the JSON structure produced by hf2q's `--quant auto` mode,
/// where each tensor may have a different bit-width and group size.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    /// Default bit-width applied when a tensor has no per-tensor override.
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Default group size applied when a tensor has no per-tensor override.
    #[serde(default = "default_group_size")]
    pub group_size: usize,

    /// Per-tensor overrides keyed by tensor name pattern.
    /// Each entry maps a tensor name (or glob pattern) to its quant config.
    #[serde(default)]
    pub per_tensor: HashMap<String, TensorQuantConfig>,
}

/// Quantization parameters for an individual tensor.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorQuantConfig {
    /// Bit-width for this tensor (3, 4, 6, or 8).
    pub bits: u8,
    /// Number of consecutive values sharing one scale/bias pair.
    pub group_size: usize,
}

fn default_bits() -> u8 {
    4
}

fn default_group_size() -> usize {
    64
}

impl QuantizationConfig {
    /// Load and parse a `quantization_config.json` file from disk.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::IoError` if the file cannot be read, or
    /// `MlxError::QuantConfigError` if the JSON is malformed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path).map_err(|e| {
            MlxError::IoError(format!("Failed to read quantization config at {}: {}", path.display(), e))
        })?;
        Self::from_json(&contents)
    }

    /// Parse a `QuantizationConfig` from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::QuantConfigError` if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            MlxError::QuantConfigError(format!("Failed to parse quantization config JSON: {e}"))
        })
    }

    /// Look up the quantization parameters for a specific tensor name.
    ///
    /// First checks for an exact match in `per_tensor`.  If not found, returns
    /// the default bits and group_size.
    pub fn config_for_tensor(&self, tensor_name: &str) -> (u8, usize) {
        if let Some(tc) = self.per_tensor.get(tensor_name) {
            (tc.bits, tc.group_size)
        } else {
            (self.bits, self.group_size)
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedWeight
// ---------------------------------------------------------------------------

/// A quantized weight tensor loaded into Metal GPU buffers.
///
/// Tracks the tensor name, logical shape, original dtype, quantization
/// parameters, and the Metal buffers holding the packed data, scales, and
/// optional biases.
///
/// # Layout
///
/// * `packed_data` — Packed quantized integers (e.g. 4-bit values packed
///   8-per-uint32, or 6-bit values packed 4-per-uint32).
/// * `scales` — Per-group scale factors as f16 values.
/// * `biases` — Per-group biases as f16 values (present for affine quant).
pub struct QuantizedWeight {
    /// Full tensor path, e.g. `model.layers.0.self_attn.q_proj.weight`.
    tensor_name: String,
    /// Logical tensor dimensions before quantization.
    shape: Vec<usize>,
    /// Original dtype before quantization (e.g. `F16` or `BF16`).
    dtype: DType,
    /// Quantization bit-width (3, 4, 6, or 8).
    bits: u8,
    /// Number of consecutive values sharing one scale/bias pair.
    group_size: usize,
    /// Per-group scale factors (f16 Metal buffer).
    scales: MlxBuffer,
    /// Per-group biases (f16 Metal buffer), if asymmetric quantization.
    biases: Option<MlxBuffer>,
    /// Packed quantized weight data (Metal buffer).
    packed_data: MlxBuffer,
}

impl QuantizedWeight {
    /// Construct a new `QuantizedWeight` with all fields specified.
    ///
    /// This is the primary constructor used by [`load_quantized_weights`].
    /// It does not validate buffer sizes — the caller is responsible for
    /// ensuring the buffers match the declared shape, bits, and group_size.
    pub fn new(
        tensor_name: String,
        shape: Vec<usize>,
        dtype: DType,
        bits: u8,
        group_size: usize,
        scales: MlxBuffer,
        biases: Option<MlxBuffer>,
        packed_data: MlxBuffer,
    ) -> Self {
        Self {
            tensor_name,
            shape,
            dtype,
            bits,
            group_size,
            scales,
            biases,
            packed_data,
        }
    }

    /// Full tensor name path.
    #[inline]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    /// Logical tensor shape (dimensions before quantization).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Original element dtype before quantization.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Quantization bit-width.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Quantization group size.
    #[inline]
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Borrow the per-group scales buffer.
    #[inline]
    pub fn scales(&self) -> &MlxBuffer {
        &self.scales
    }

    /// Borrow the per-group biases buffer, if present.
    #[inline]
    pub fn biases(&self) -> Option<&MlxBuffer> {
        self.biases.as_ref()
    }

    /// Borrow the packed quantized data buffer.
    #[inline]
    pub fn packed_data(&self) -> &MlxBuffer {
        &self.packed_data
    }

    /// Number of logical elements in the weight tensor (product of shape dims).
    pub fn element_count(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Number of quantization groups along the last dimension.
    ///
    /// This is `ceil(last_dim / group_size)`.
    pub fn num_groups(&self) -> usize {
        let last_dim = self.shape.last().copied().unwrap_or(0);
        if self.group_size == 0 {
            return 0;
        }
        (last_dim + self.group_size - 1) / self.group_size
    }
}

impl std::fmt::Debug for QuantizedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedWeight")
            .field("tensor_name", &self.tensor_name)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("bits", &self.bits)
            .field("group_size", &self.group_size)
            .field("packed_data_bytes", &self.packed_data.byte_len())
            .field("scales_bytes", &self.scales.byte_len())
            .field("has_biases", &self.biases.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DType conversion
// ---------------------------------------------------------------------------

/// Convert a safetensors `Dtype` to our `DType`.
///
/// Returns `Err(MlxError::UnsupportedDtype)` for types we don't handle.
fn safetensors_dtype_to_dtype(st_dtype: safetensors::Dtype) -> Result<DType> {
    match st_dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::U16 => Ok(DType::U16),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I32 => Ok(DType::I32),
        other => Err(MlxError::UnsupportedDtype(format!("{other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Buffer creation
// ---------------------------------------------------------------------------

/// Copy raw bytes from a safetensors tensor view into a new Metal
/// `StorageModeShared` buffer.
///
/// This is the core data-transfer function.  It:
/// 1. Allocates a Metal buffer of the exact byte length.
/// 2. Copies the tensor data from the (mmap'd) safetensors region into the
///    Metal buffer via a single `std::ptr::copy_nonoverlapping`.
///
/// # Arguments
///
/// * `device`   — The Metal device to allocate from.
/// * `data`     — Raw tensor bytes (borrowed from the safetensors mmap).
/// * `dtype`    — Element data type for metadata tracking.
/// * `shape`    — Tensor dimensions for metadata tracking.
///
/// # Errors
///
/// * `MlxError::InvalidArgument` if `data` is empty.
/// * `MlxError::BufferAllocationError` if Metal allocation fails.
pub fn safetensors_to_metal_buffer(
    device: &MlxDevice,
    data: &[u8],
    dtype: DType,
    shape: Vec<usize>,
) -> Result<MlxBuffer> {
    if data.is_empty() {
        return Err(MlxError::InvalidArgument(
            "Cannot create Metal buffer from empty data".into(),
        ));
    }

    let byte_len = data.len();
    let metal_buf = device
        .metal_device()
        .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

    if metal_buf.contents().is_null() {
        return Err(MlxError::BufferAllocationError { bytes: byte_len });
    }

    // Copy tensor bytes into the Metal buffer.
    // SAFETY: Metal guarantees the buffer contents pointer is valid for
    // `byte_len` bytes.  The source slice is valid for `byte_len` bytes.
    // The regions do not overlap (one is mmap, the other is Metal allocation).
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), metal_buf.contents() as *mut u8, byte_len);
    }

    Ok(MlxBuffer::from_raw(metal_buf, dtype, shape))
}

// ---------------------------------------------------------------------------
// Memory-mapped safetensors file handle
// ---------------------------------------------------------------------------

/// A memory-mapped safetensors file ready for tensor extraction.
///
/// This struct owns the mmap and parsed header.  Individual tensors can be
/// loaded into Metal buffers on demand via [`load_tensor`](Self::load_tensor)
/// or all at once via [`load_all_tensors`](Self::load_all_tensors).
pub struct SafetensorsFile {
    /// The memory-mapped file data.
    #[allow(dead_code)]
    mmap: Mmap,
}

impl std::fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetensorsFile")
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl SafetensorsFile {
    /// Open and memory-map a safetensors file.
    ///
    /// The file is mapped read-only.  No tensor data is copied until
    /// `load_tensor` or `load_all_tensors` is called.
    ///
    /// # Errors
    ///
    /// * `MlxError::IoError` if the file cannot be opened or mapped.
    pub fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).map_err(|e| {
            MlxError::IoError(format!("Failed to open safetensors file {}: {}", path.display(), e))
        })?;

        // SAFETY: We are mapping a regular file read-only.  The only hazard is
        // if another process truncates the file while mapped, which is
        // undefined behavior.  This is the standard use case for memmap2.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                MlxError::IoError(format!("Failed to mmap safetensors file {}: {}", path.display(), e))
            })?
        };

        Ok(Self { mmap })
    }

    /// Parse the safetensors header and return the deserialized view.
    ///
    /// This borrows from the mmap and is cheap (no tensor data is copied).
    fn parse(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.mmap).map_err(|e| {
            MlxError::SafetensorsError(format!("Failed to parse safetensors header: {e}"))
        })
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let st = self.parse()?;
        Ok(st.names().into_iter().map(|s| s.to_string()).collect())
    }

    /// Load a single named tensor into a Metal buffer.
    ///
    /// Returns the dtype, shape, and a Metal buffer containing the raw bytes.
    ///
    /// # Errors
    ///
    /// * `MlxError::SafetensorsError` if the tensor name is not found.
    /// * `MlxError::UnsupportedDtype` if the tensor's dtype is not supported.
    /// * `MlxError::BufferAllocationError` if Metal allocation fails.
    pub fn load_tensor(
        &self,
        name: &str,
        device: &MlxDevice,
    ) -> Result<(DType, Vec<usize>, MlxBuffer)> {
        let st = self.parse()?;
        let view = st.tensor(name).map_err(|e| {
            MlxError::SafetensorsError(format!("Tensor '{}' not found: {}", name, e))
        })?;

        let dtype = safetensors_dtype_to_dtype(view.dtype())?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();

        let buffer = safetensors_to_metal_buffer(device, data, dtype, shape.clone())?;
        Ok((dtype, shape, buffer))
    }

    /// Load all tensors from the file into Metal buffers.
    ///
    /// Returns a map from tensor name to `(DType, shape, MlxBuffer)`.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during loading.
    pub fn load_all_tensors(
        &self,
        device: &MlxDevice,
    ) -> Result<HashMap<String, (DType, Vec<usize>, MlxBuffer)>> {
        let st = self.parse()?;
        let mut result = HashMap::new();

        for (name, view) in st.tensors() {
            let dtype = safetensors_dtype_to_dtype(view.dtype())?;
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let buffer = safetensors_to_metal_buffer(device, data, dtype, shape.clone())?;
            result.insert(name, (dtype, shape, buffer));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// High-level quantized weight loading
// ---------------------------------------------------------------------------

/// Load quantized weights from a directory containing safetensors file(s) and
/// a `quantization_config.json`.
///
/// This is the primary entry point for weight loading.  It:
///
/// 1. Reads `quantization_config.json` from the directory to determine
///    per-tensor bit-widths and group sizes.
/// 2. Discovers all `*.safetensors` files in the directory.
/// 3. Memory-maps each file and loads tensors that look like quantized weight
///    components (packed data, scales, biases) into Metal buffers.
/// 4. Groups the components by base tensor name and constructs
///    [`QuantizedWeight`] instances.
///
/// # Tensor Naming Convention
///
/// Quantized weights in safetensors use a naming convention:
/// - `<base_name>.weight` — packed quantized data
/// - `<base_name>.scales` — per-group scale factors
/// - `<base_name>.biases` — per-group biases (optional, for affine quant)
///
/// # Arguments
///
/// * `model_dir` — Path to the directory containing safetensors files and config.
/// * `device`    — The Metal device for buffer allocation.
///
/// # Errors
///
/// * `MlxError::IoError` if the directory or files cannot be read.
/// * `MlxError::QuantConfigError` if the quantization config is invalid.
/// * `MlxError::SafetensorsError` if a safetensors file is malformed.
pub fn load_quantized_weights(
    model_dir: &Path,
    device: &MlxDevice,
) -> Result<Vec<QuantizedWeight>> {
    // 1. Load quantization config.
    let config_path = model_dir.join("quantization_config.json");
    let quant_config = QuantizationConfig::from_file(&config_path)?;

    // 2. Discover safetensors files.
    let safetensors_files = discover_safetensors_files(model_dir)?;
    if safetensors_files.is_empty() {
        return Err(MlxError::IoError(format!(
            "No .safetensors files found in {}",
            model_dir.display()
        )));
    }

    // 3. Load all tensors from all files into a flat map.
    let mut all_tensors: HashMap<String, (DType, Vec<usize>, MlxBuffer)> = HashMap::new();
    for sf_path in &safetensors_files {
        let sf = SafetensorsFile::open(sf_path)?;
        let tensors = sf.load_all_tensors(device)?;
        all_tensors.extend(tensors);
    }

    // 4. Group by base tensor name and construct QuantizedWeight instances.
    //
    // We look for groups of related tensors.  The convention is:
    //   - `<base>.weight` or just `<base>` — packed quantized data
    //   - `<base>.scales` — per-group scales (f16)
    //   - `<base>.biases` — per-group biases (f16, optional)
    //
    // A tensor is considered quantized if it has a corresponding `.scales` entry.

    let mut weights = Vec::new();
    let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Collect all base names that have .scales entries.
    let scale_suffix = ".scales";
    let scale_bases: Vec<String> = all_tensors
        .keys()
        .filter(|k| k.ends_with(scale_suffix))
        .map(|k| k[..k.len() - scale_suffix.len()].to_string())
        .collect();

    for base_name in &scale_bases {
        let scales_key = format!("{base_name}.scales");
        let biases_key = format!("{base_name}.biases");

        // The packed data might be at `<base>.weight` or just `<base>`.
        let weight_key = if all_tensors.contains_key(&format!("{base_name}.weight")) {
            format!("{base_name}.weight")
        } else if all_tensors.contains_key(base_name) {
            base_name.clone()
        } else {
            // Scales without a weight tensor — skip.
            continue;
        };

        // Extract the packed data buffer.
        let (packed_dtype, packed_shape, packed_data) = match all_tensors.remove(&weight_key) {
            Some(t) => t,
            None => continue,
        };

        // Extract scales buffer.
        let (_scales_dtype, _scales_shape, scales_buf) = match all_tensors.remove(&scales_key) {
            Some(t) => t,
            None => continue,
        };

        // Extract biases buffer (optional).
        let biases_buf = all_tensors.remove(&biases_key).map(|(_, _, buf)| buf);

        // Look up quant config for this tensor.
        let (bits, group_size) = quant_config.config_for_tensor(&weight_key);

        weights.push(QuantizedWeight::new(
            weight_key.clone(),
            packed_shape,
            packed_dtype,
            bits,
            group_size,
            scales_buf,
            biases_buf,
            packed_data,
        ));

        processed.insert(weight_key);
        processed.insert(scales_key);
        processed.insert(biases_key);
    }

    Ok(weights)
}

/// Discover all `*.safetensors` files in a directory, sorted by name.
fn discover_safetensors_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let entries = fs::read_dir(dir).map_err(|e| {
        MlxError::IoError(format!("Failed to read directory {}: {}", dir.display(), e))
    })?;

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            MlxError::IoError(format!("Failed to read directory entry: {e}"))
        })?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype as StDtype, TensorView};

    // ---- QuantizedWeight construction and accessors ----

    #[test]
    fn test_quantized_weight_construction() {
        let device = MlxDevice::new().expect("device");

        // Create minimal buffers for testing.
        let packed = device.alloc_buffer(64, DType::U32, vec![4, 4]).expect("packed");
        let scales = device.alloc_buffer(16, DType::F16, vec![4, 2]).expect("scales");
        let biases = device.alloc_buffer(16, DType::F16, vec![4, 2]).expect("biases");

        let qw = QuantizedWeight::new(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![2816, 2816],
            DType::F16,
            4,
            64,
            scales,
            Some(biases),
            packed,
        );

        assert_eq!(qw.tensor_name(), "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(qw.shape(), &[2816, 2816]);
        assert_eq!(qw.dtype(), DType::F16);
        assert_eq!(qw.bits(), 4);
        assert_eq!(qw.group_size(), 64);
        assert!(qw.biases().is_some());
        assert_eq!(qw.element_count(), 2816 * 2816);
        assert_eq!(qw.num_groups(), (2816 + 64 - 1) / 64);
    }

    #[test]
    fn test_quantized_weight_no_biases() {
        let device = MlxDevice::new().expect("device");

        let packed = device.alloc_buffer(32, DType::U32, vec![4, 2]).expect("packed");
        let scales = device.alloc_buffer(8, DType::F16, vec![4, 1]).expect("scales");

        let qw = QuantizedWeight::new(
            "test.weight".to_string(),
            vec![128, 128],
            DType::BF16,
            6,
            32,
            scales,
            None,
            packed,
        );

        assert!(qw.biases().is_none());
        assert_eq!(qw.bits(), 6);
        assert_eq!(qw.group_size(), 32);
        assert_eq!(qw.num_groups(), (128 + 32 - 1) / 32);
    }

    #[test]
    fn test_quantized_weight_debug() {
        let device = MlxDevice::new().expect("device");
        let packed = device.alloc_buffer(16, DType::U32, vec![4]).expect("packed");
        let scales = device.alloc_buffer(4, DType::F16, vec![2]).expect("scales");

        let qw = QuantizedWeight::new(
            "test.w".to_string(),
            vec![64],
            DType::F32,
            4,
            64,
            scales,
            None,
            packed,
        );

        let debug_str = format!("{:?}", qw);
        assert!(debug_str.contains("QuantizedWeight"));
        assert!(debug_str.contains("test.w"));
        assert!(debug_str.contains("bits: 4"));
    }

    // ---- QuantizationConfig parsing ----

    #[test]
    fn test_quant_config_defaults() {
        let json = r#"{}"#;
        let config = QuantizationConfig::from_json(json).expect("parse");
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert!(config.per_tensor.is_empty());
    }

    #[test]
    fn test_quant_config_with_per_tensor() {
        let json = r#"{
            "bits": 4,
            "group_size": 64,
            "per_tensor": {
                "model.layers.0.self_attn.v_proj.weight": {"bits": 6, "group_size": 128},
                "model.embed_tokens.weight": {"bits": 8, "group_size": 32}
            }
        }"#;

        let config = QuantizationConfig::from_json(json).expect("parse");
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);

        // Per-tensor override.
        let (bits, gs) = config.config_for_tensor("model.layers.0.self_attn.v_proj.weight");
        assert_eq!(bits, 6);
        assert_eq!(gs, 128);

        // Default for unknown tensor.
        let (bits, gs) = config.config_for_tensor("model.layers.5.mlp.gate_proj.weight");
        assert_eq!(bits, 4);
        assert_eq!(gs, 64);
    }

    #[test]
    fn test_quant_config_invalid_json() {
        let result = QuantizationConfig::from_json("not json at all {{{");
        assert!(result.is_err());
        match result {
            Err(MlxError::QuantConfigError(msg)) => {
                assert!(msg.contains("parse"), "msg: {msg}");
            }
            other => panic!("Expected QuantConfigError, got {:?}", other),
        }
    }

    // ---- DType conversion ----

    #[test]
    fn test_safetensors_dtype_conversion() {
        assert_eq!(safetensors_dtype_to_dtype(StDtype::F32).unwrap(), DType::F32);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::F16).unwrap(), DType::F16);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::BF16).unwrap(), DType::BF16);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::U8).unwrap(), DType::U8);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::U16).unwrap(), DType::U16);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::U32).unwrap(), DType::U32);
        assert_eq!(safetensors_dtype_to_dtype(StDtype::I32).unwrap(), DType::I32);
    }

    #[test]
    fn test_safetensors_dtype_unsupported() {
        let result = safetensors_dtype_to_dtype(StDtype::BOOL);
        assert!(result.is_err());
        match result {
            Err(MlxError::UnsupportedDtype(_)) => {}
            other => panic!("Expected UnsupportedDtype, got {:?}", other),
        }
    }

    // ---- safetensors_to_metal_buffer ----

    #[test]
    fn test_safetensors_to_metal_buffer_roundtrip() {
        let device = MlxDevice::new().expect("device");

        // Create test data: 4 f32 values.
        let values: [f32; 4] = [1.0, 2.5, -3.0, 4.125];
        let bytes: &[u8] = bytemuck::cast_slice(&values);

        let buf = safetensors_to_metal_buffer(&device, bytes, DType::F32, vec![4])
            .expect("to_metal_buffer");

        assert_eq!(buf.byte_len(), 16);
        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.shape(), &[4]);

        // Verify data integrity.
        let read_back: &[f32] = buf.as_slice().expect("as_slice");
        assert_eq!(read_back.len(), 4);
        assert_eq!(read_back[0], 1.0);
        assert_eq!(read_back[1], 2.5);
        assert_eq!(read_back[2], -3.0);
        assert_eq!(read_back[3], 4.125);
    }

    #[test]
    fn test_safetensors_to_metal_buffer_empty_error() {
        let device = MlxDevice::new().expect("device");
        let result = safetensors_to_metal_buffer(&device, &[], DType::F32, vec![0]);
        assert!(result.is_err());
        match result {
            Err(MlxError::InvalidArgument(msg)) => {
                assert!(msg.contains("empty"), "msg: {msg}");
            }
            other => panic!("Expected InvalidArgument, got {:?}", other),
        }
    }

    #[test]
    fn test_safetensors_to_metal_buffer_u8_data() {
        let device = MlxDevice::new().expect("device");
        let data: Vec<u8> = (0..128).collect();

        let buf = safetensors_to_metal_buffer(&device, &data, DType::U8, vec![128])
            .expect("to_metal_buffer");

        assert_eq!(buf.byte_len(), 128);
        let read_back: &[u8] = buf.as_slice().expect("as_slice");
        for (i, &val) in read_back.iter().enumerate() {
            assert_eq!(val, i as u8, "mismatch at index {i}");
        }
    }

    // ---- SafetensorsFile with synthetic test file ----

    /// Create a minimal safetensors file in a temp directory for testing.
    fn create_test_safetensors(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("test_model.safetensors");

        // Build tensors: two small f32 tensors.
        let tensor_a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor_a_bytes: &[u8] = bytemuck::cast_slice(&tensor_a_data);
        let tensor_b_data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let tensor_b_bytes: &[u8] = bytemuck::cast_slice(&tensor_b_data);

        let tensors = vec![
            (
                "layer.weight",
                TensorView::new(StDtype::F32, vec![2, 3], tensor_a_bytes).unwrap(),
            ),
            (
                "layer.bias",
                TensorView::new(StDtype::F32, vec![3], tensor_b_bytes).unwrap(),
            ),
        ];

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        fs::write(&path, &serialized).unwrap();

        path
    }

    #[test]
    fn test_safetensors_file_open_and_list() {
        let tmp = tempdir();
        let st_path = create_test_safetensors(&tmp);

        let sf = SafetensorsFile::open(&st_path).expect("open");
        let names = sf.tensor_names().expect("names");

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"layer.weight".to_string()));
        assert!(names.contains(&"layer.bias".to_string()));
    }

    #[test]
    fn test_safetensors_file_load_tensor() {
        let device = MlxDevice::new().expect("device");
        let tmp = tempdir();
        let st_path = create_test_safetensors(&tmp);

        let sf = SafetensorsFile::open(&st_path).expect("open");
        let (dtype, shape, buf) = sf.load_tensor("layer.weight", &device).expect("load");

        assert_eq!(dtype, DType::F32);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(buf.byte_len(), 24); // 6 * 4 bytes

        let data: &[f32] = buf.as_slice().expect("as_slice");
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_safetensors_file_load_all() {
        let device = MlxDevice::new().expect("device");
        let tmp = tempdir();
        let st_path = create_test_safetensors(&tmp);

        let sf = SafetensorsFile::open(&st_path).expect("open");
        let all = sf.load_all_tensors(&device).expect("load_all");

        assert_eq!(all.len(), 2);

        let (dtype, shape, buf) = all.get("layer.bias").expect("bias");
        assert_eq!(*dtype, DType::F32);
        assert_eq!(*shape, vec![3]);
        let data: &[f32] = buf.as_slice().expect("as_slice");
        assert_eq!(data, &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_safetensors_file_tensor_not_found() {
        let tmp = tempdir();
        let st_path = create_test_safetensors(&tmp);
        let device = MlxDevice::new().expect("device");

        let sf = SafetensorsFile::open(&st_path).expect("open");
        let result = sf.load_tensor("nonexistent", &device);
        assert!(result.is_err());
        match result {
            Err(MlxError::SafetensorsError(msg)) => {
                assert!(msg.contains("nonexistent"), "msg: {msg}");
            }
            other => panic!("Expected SafetensorsError, got {:?}", other),
        }
    }

    #[test]
    fn test_safetensors_file_open_missing() {
        let result = SafetensorsFile::open(Path::new("/tmp/does_not_exist_8f3a2b1c.safetensors"));
        assert!(result.is_err());
        match result {
            Err(MlxError::IoError(_)) => {}
            other => panic!("Expected IoError, got {:?}", other),
        }
    }

    // ---- load_quantized_weights with synthetic directory ----

    /// Create a synthetic quantized model directory for integration testing.
    fn create_test_quant_dir(dir: &Path) {
        // Create quantization_config.json.
        let config_json = r#"{
            "bits": 4,
            "group_size": 64,
            "per_tensor": {
                "proj.weight": {"bits": 4, "group_size": 64}
            }
        }"#;
        fs::write(dir.join("quantization_config.json"), config_json).unwrap();

        // Create a safetensors file with weight, scales, and biases tensors.
        //
        // proj.weight — packed quantized data (stored as U32)
        // proj.scales — per-group scale factors (stored as F16)
        // proj.biases — per-group biases (stored as F16)
        let weight_data: Vec<u32> = vec![0xAAAA_BBBB; 8]; // 8 uint32s = 32 bytes
        let weight_bytes: &[u8] = bytemuck::cast_slice(&weight_data);

        // Scales: 2 f16 values (4 bytes).
        let scales_data: Vec<u16> = vec![0x3C00, 0x3C00]; // f16 = 1.0
        let scales_bytes: &[u8] = bytemuck::cast_slice(&scales_data);

        // Biases: 2 f16 values (4 bytes).
        let biases_data: Vec<u16> = vec![0x0000, 0x0000]; // f16 = 0.0
        let biases_bytes: &[u8] = bytemuck::cast_slice(&biases_data);

        let tensors = vec![
            (
                "proj.weight",
                TensorView::new(StDtype::U32, vec![2, 4], weight_bytes).unwrap(),
            ),
            (
                "proj.scales",
                TensorView::new(StDtype::F16, vec![2, 1], scales_bytes).unwrap(),
            ),
            (
                "proj.biases",
                TensorView::new(StDtype::F16, vec![2, 1], biases_bytes).unwrap(),
            ),
        ];

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        fs::write(dir.join("model.safetensors"), &serialized).unwrap();
    }

    #[test]
    fn test_load_quantized_weights_integration() {
        let device = MlxDevice::new().expect("device");
        let tmp = tempdir();
        create_test_quant_dir(&tmp);

        let weights = load_quantized_weights(&tmp, &device).expect("load");

        assert_eq!(weights.len(), 1);
        let qw = &weights[0];
        assert_eq!(qw.tensor_name(), "proj.weight");
        assert_eq!(qw.bits(), 4);
        assert_eq!(qw.group_size(), 64);
        assert_eq!(qw.packed_data().byte_len(), 32); // 8 * 4 bytes
        assert_eq!(qw.scales().byte_len(), 4); // 2 * 2 bytes
        assert!(qw.biases().is_some());
    }

    #[test]
    fn test_load_quantized_weights_no_safetensors() {
        let tmp = tempdir();

        // Create config but no safetensors files.
        fs::write(tmp.join("quantization_config.json"), "{}").unwrap();

        let device = MlxDevice::new().expect("device");
        let result = load_quantized_weights(&tmp, &device);
        assert!(result.is_err());
        match result {
            Err(MlxError::IoError(msg)) => {
                assert!(msg.contains("No .safetensors files"), "msg: {msg}");
            }
            other => panic!("Expected IoError, got {:?}", other),
        }
    }

    #[test]
    fn test_load_quantized_weights_missing_config() {
        let tmp = tempdir();
        // Create a dummy safetensors file but no config.
        let data: Vec<u8> = vec![0; 16];
        let tensors = vec![(
            "dummy",
            TensorView::new(StDtype::U8, vec![16], &data).unwrap(),
        )];
        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        fs::write(tmp.join("model.safetensors"), &serialized).unwrap();

        let device = MlxDevice::new().expect("device");
        let result = load_quantized_weights(&tmp, &device);
        assert!(result.is_err());
        match result {
            Err(MlxError::IoError(msg)) => {
                assert!(msg.contains("quantization_config"), "msg: {msg}");
            }
            other => panic!("Expected IoError for missing config, got {:?}", other),
        }
    }

    // ---- Helper: create a temp directory and return its path ----

    fn tempdir() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("mlx_native_test_{}", std::process::id()));
        path.push(format!("{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }
}
