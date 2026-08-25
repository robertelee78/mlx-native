//! GGML block-format quantized matrix-vector multiply dispatch.
//!
//! Encodes GPU compute commands for GGML quantized mat-vec:
//!   output[row] = dot(dequant(weight[row]), input)
//!
//! Weight buffers contain raw GGML blocks — the same bytes that come from
//! GGUF mmap. No intermediate conversion.
//!
//! Supported formats include Q2_K, Q4_0, Q5_0, Q8_0, and K-quants through Q6_K.
//!
//! Portions derived from candle-metal-kernels v0.10.2 (Apache-2.0).
//! See src/shaders/quantized_matmul_ggml.metal for full attribution.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, DispatchRecord, KernelArg};
use crate::env_flags::{cached_env_default_true, cached_env_eq_one};
use crate::ggml_capability::{
    plan_dense_auto_route, tensor_mm_auto_selected, DenseAutoPlan, GgmlCapabilityRequest,
    GgmlInvocation, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlWorkloadClass,
    GGML_CAPABILITY_SCHEMA_VERSION,
};
use crate::ggml_dispatch_trace::{trace_ggml_operation, GgmlResolvedDispatchTrace};
use crate::ggml_routing_policy::ggml_routing_policy_for_registry;
use crate::ops::dense_mm_capability::is_unavailable_tensor_header;
use crate::ops::dense_q4_auto::{self, DenseQ4Route};
use std::sync::atomic::AtomicI8;

// ADR-029: cached hot-path env-flag gates for dispatch_mv.
// Same pattern as Step 1an (dispatch_id_mv). ~120 attn dispatches per token
// each hitting these 2 env reads.
static CACHED_Q6K_MV_NR2: AtomicI8 = AtomicI8::new(-1);
static CACHED_Q8_0_MV_NR2: AtomicI8 = AtomicI8::new(-1);
// ADR-040 §0.21 decode mul_mv_ext lever (opt-in, default off — see routing site).
static CACHED_DECODE_MV_EXT: AtomicI8 = AtomicI8::new(-1);
// Decode mvN lever (bit-identical column-amortizing Q4_K/Q6_K mv).
static CACHED_DECODE_MVN: AtomicI8 = AtomicI8::new(-1);
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

fn checked_byte_extent(label: &str, factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |total, factor| {
        total.checked_mul(*factor).ok_or_else(|| {
            MlxError::InvalidArgument(format!("quantized_matmul_ggml: {label} size overflow"))
        })
    })
}

fn validate_native_quantized_dtypes(
    operation: &str,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    if input.dtype() != DType::F32 || weight.dtype() != DType::U8 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation} requires F32 input, native U8 GGUF blocks, and F32 output; got {:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            output.dtype(),
        )));
    }
    Ok(())
}

fn validate_signed_metal_dimensions(operation: &str, dimensions: &[(&str, u32)]) -> Result<()> {
    for (label, value) in dimensions {
        if *value > i32::MAX as u32 {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: {label} exceeds the signed Metal ABI"
            )));
        }
    }
    Ok(())
}

// ---- Block format constants ----

/// Q4_0: 32 values per block, 18 bytes per block (2 byte f16 scale + 16 bytes quants).
const QK4_0: u32 = 32;
const BLOCK_Q4_0_BYTES: u32 = 18;

/// Q5_0: 32 values per block, 22 bytes per block
/// (2-byte f16 scale + 4-byte high-bit mask + 16 bytes of low nibbles).
const QK5_0: u32 = 32;
const BLOCK_Q5_0_BYTES: u32 = 22;

/// Q8_0: 32 values per block, 34 bytes per block (2 byte f16 scale + 32 bytes quants).
const QK8_0: u32 = 32;
const BLOCK_Q8_0_BYTES: u32 = 34;

/// Q4_K: 256 values per block, 144 bytes per block.
const QK4_K: u32 = 256;
const BLOCK_Q4_K_BYTES: u32 = 144;

/// Q2_K: 256 values per block, 84 bytes per block.
/// Layout: scales[16] + qs[64] + d(f16) + dmin(f16).
const QK2_K: u32 = 256;
const BLOCK_Q2_K_BYTES: u32 = 84;

/// Q3_K: 256 values per block, 110 bytes per block.
/// Layout: hmask[32] + qs[64] + scales[12] + d(f16).
const QK3_K: u32 = 256;
const BLOCK_Q3_K_BYTES: u32 = 110;

/// Q5_K: 256 values per block, 176 bytes per block.
/// Block layout: d(fp16) + dmin(fp16) + scales[12] + qh[32] + qs[128] = 176.
const QK5_K: u32 = 256;
const BLOCK_Q5_K_BYTES: u32 = 176;

/// Q6_K: 256 values per block, 210 bytes per block.
const QK6_K: u32 = 256;
const BLOCK_Q6_K_BYTES: u32 = 210;

/// Q5_1 (legacy ggml 5-bit asymmetric, 32-element block).
/// Block layout: d(fp16) + m(fp16) + qh(u32) + qs[16] = 24 bytes.
/// 6 effective bpw (5 payload bits + scale + min term).
/// ADR-022 Phase 1 — added 2026-05-08 to support APEX-Q5_K_M
/// MoE expert tensors that fall through the layer-mix policy into
/// Q5_1 (e.g. `gemma4-ara-2pass-APEX-Q5_K_M.gguf` blk.{5..9, 20..24}.ffn_down_exps.weight).
/// Reference: the ggml `block_q5_1` block format.
const QK5_1: u32 = 32;
const BLOCK_Q5_1_BYTES: u32 = 24;

/// IQ4_NL (4-bit non-linear codebook, 32-element block).
/// Block layout: d(fp16) + qs[16] = 18 bytes.
/// 4.5 effective bpw — 16 4-bit indices into a fixed 16-entry signed
/// codebook (the ggml `kvalues_iq4nl` codebook).
/// ADR-022 Phase 1 — added 2026-05-08 alongside Q5_1.
/// Reference: the ggml `block_iq4_nl` block format.
const QK4_NL: u32 = 32;
const BLOCK_IQ4_NL_BYTES: u32 = 18;

/// IQ4_XS (4-bit non-linear codebook, 256-element super-block).
/// Block layout: d(fp16) + scales_h(u16) + scales_l[4] + qs[128] = 136 bytes.
/// 4.25 effective bpw — 8 sub-blocks of 32 elements, each with a 6-bit
/// scale (4 bits in scales_l + 2 bits in scales_h), 4-bit indices into
/// the SAME `kvalues_iq4nl` codebook used by IQ4_NL.
/// ADR-033 §Pi 2026-05-22 — added to unblock apex-i-quality on Qwen MoE
/// (every quality-tier mudler config uses IQ4_XS for mid-layer experts).
/// Reference: the ggml `block_iq4_xs` block format.
const BLOCK_IQ4_XS_BYTES: u32 = 136;

// ---- Public types ----

/// GGML quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum GgmlType {
    /// 32-bit float (unquantized). 1 element per block, 4 bytes per block.
    F32,
    /// 16-bit float (unquantized). 1 element per block, 2 bytes per block.
    F16,
    /// 16-bit brain float (unquantized, GGML type ID 30).
    /// 1 element per block, 2 bytes per block.
    BF16,
    /// 4-bit quantization. 32 values per block, 18 bytes per block.
    Q4_0,
    /// Legacy symmetric 5-bit quantization (GGML type ID 6).
    /// 32 values per block, 22 bytes per block.
    Q5_0,
    /// 8-bit quantization. 32 values per block, 34 bytes per block.
    Q8_0,
    /// 2-bit super-block quantization. 256 values per block, 84 bytes per block.
    Q2_K,
    /// 3-bit super-block quantization. 256 values per block, 110 bytes per block.
    Q3_K,
    /// 4-bit super-block quantization. 256 values per block, 144 bytes per block.
    Q4_K,
    /// 5-bit super-block quantization. 256 values per block, 176 bytes per block.
    /// Supported by dense, expert-routed, and embedding kernels.
    Q5_K,
    /// 6-bit super-block quantization. 256 values per block, 210 bytes per block.
    Q6_K,
    /// 16-bit signed integer (1 element per block, 2 bytes per block).
    /// Recognized for GGUF header parsing; dequant depends on per-tensor
    /// scale metadata (ADR-013 Decision 12). No matmul kernel.
    I16,
    /// 32-bit signed integer (GGML type ID 26). Used by DeepSeek-V4 hash
    /// router tables; loaded as raw `DType::I32`, not a matmul weight.
    I32,
    /// Legacy 5-bit asymmetric quant (id 7 in GGML). 32 values per block,
    /// 24 bytes per block. Carries a per-block `m` (min) term in addition
    /// to the scale `d`. ADR-022 Phase 1.
    Q5_1,
    /// Non-linear 4-bit codebook quant (id 20 in GGML). 32 values per
    /// block, 18 bytes per block. Each 4-bit index selects from a fixed
    /// 16-entry signed codebook `kvalues_iq4nl`. ADR-022 Phase 1.
    IQ4_NL,
    /// Super-block 4-bit non-linear codebook quant (id 23 in GGML).
    /// 256 values per super-block, 136 bytes per block. 8 sub-blocks of
    /// 32 elements each, with 6-bit per-sub-block scales (4-bit
    /// scales_l + 2-bit scales_h). Shares the `kvalues_iq4nl` codebook
    /// with IQ4_NL. ADR-033 §Pi 2026-05-22 — added to unblock
    /// apex-i-quality on Qwen MoE (canonical mid-layer expert quant).
    /// Reader-side recognized for GGUF header parsing; Metal mul_mv /
    /// mul_mm kernels follow in a subsequent commit.
    IQ4_XS,
}

impl GgmlType {
    /// Number of dequantized values per GGML block.
    pub fn block_values(self) -> u32 {
        match self {
            GgmlType::F32 => 1,
            GgmlType::F16 => 1,
            GgmlType::BF16 => 1,
            GgmlType::Q4_0 => QK4_0,
            GgmlType::Q5_0 => QK5_0,
            GgmlType::Q8_0 => QK8_0,
            GgmlType::Q2_K => QK2_K,
            GgmlType::Q3_K => QK3_K,
            GgmlType::Q4_K => QK4_K,
            GgmlType::Q5_K => QK5_K,
            GgmlType::Q6_K => QK6_K,
            GgmlType::I16 => 1,
            GgmlType::I32 => 1,
            GgmlType::Q5_1 => QK5_1,
            GgmlType::IQ4_NL => QK4_NL,
            GgmlType::IQ4_XS => QK6_K, // super-block size = QK_K = 256
        }
    }

    /// Number of bytes per GGML block.
    pub fn block_bytes(self) -> u32 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::BF16 => 2,
            GgmlType::Q4_0 => BLOCK_Q4_0_BYTES,
            GgmlType::Q5_0 => BLOCK_Q5_0_BYTES,
            GgmlType::Q8_0 => BLOCK_Q8_0_BYTES,
            GgmlType::Q2_K => BLOCK_Q2_K_BYTES,
            GgmlType::Q3_K => BLOCK_Q3_K_BYTES,
            GgmlType::Q4_K => BLOCK_Q4_K_BYTES,
            GgmlType::Q5_K => BLOCK_Q5_K_BYTES,
            GgmlType::Q6_K => BLOCK_Q6_K_BYTES,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::Q5_1 => BLOCK_Q5_1_BYTES,
            GgmlType::IQ4_NL => BLOCK_IQ4_NL_BYTES,
            GgmlType::IQ4_XS => BLOCK_IQ4_XS_BYTES,
        }
    }

    /// Metal kernel function name for the matrix-vector (mv) kernel
    /// — used for `m <= MM_ROUTING_THRESHOLD`.
    pub(crate) fn kernel_name(self) -> &'static str {
        match self {
            // Scalar/non-quantized types are not applicable to this dispatch.
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 | GgmlType::I32 => {
                "unsupported"
            }
            GgmlType::Q4_0 => "kernel_mul_mv_q4_0_f32",
            GgmlType::Q5_0 => "kernel_mul_mv_q5_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mv_q8_0_f32",
            GgmlType::Q2_K => "kernel_mul_mv_q2_K_f32",
            GgmlType::Q3_K => "kernel_mul_mv_q3_K_f32",
            // ADR-013 P7 — Q4_K mv peer-pattern kernel.
            GgmlType::Q4_K => "kernel_mul_mv_q4_K_f32",
            // ADR-022 Phase 2 — Q5_K dense mv ported.
            GgmlType::Q5_K => "kernel_mul_mv_q5_K_f32",
            GgmlType::Q6_K => "kernel_mul_mv_q6_K_f32",
            // ADR-022 Phase 1 P1.5 — Q5_1 / IQ4_NL dense mv ports.
            GgmlType::Q5_1 => "kernel_mul_mv_q5_1_f32",
            GgmlType::IQ4_NL => "kernel_mul_mv_iq4_nl_f32",
            // ADR-033 §Pi Task #16 SHIPPED 2026-05-22 — mirrors IQ4_NL
            // geometry (N_SIMDGROUP=2, N_DST=4, threadgroup=(8,8,1)).
            // Byte-cmp parity tested against the reference Metal IQ4_XS
            // kernel via tests/iq4_xs_metal_parity.rs.
            GgmlType::IQ4_XS => "kernel_mul_mv_iq4_xs_f32",
        }
    }

    /// Metal kernel function name for the matrix-matrix (mm) kernel
    /// — used for `m > MM_ROUTING_THRESHOLD`.  Peer port of the
    /// `kernel_mul_mm_<qtype>_f32` template (ADR-011 Phase 3).
    pub(crate) fn mm_kernel_name(self) -> &'static str {
        match self {
            // ADR-022 Phase 2 — Q5_K dense mm ported.
            // ADR-022 Phase 3 — Q4_K dense mm ported.
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 | GgmlType::I32 => {
                "unsupported"
            }
            GgmlType::Q2_K => "kernel_mul_mm_q2_K_f32",
            GgmlType::Q3_K => "kernel_mul_mm_q3_K_f32",
            GgmlType::Q4_0 => "kernel_mul_mm_q4_0_f32",
            GgmlType::Q5_0 => "kernel_mul_mm_q5_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_q8_0_f32",
            GgmlType::Q4_K => "kernel_mul_mm_q4_K_f32",
            GgmlType::Q5_K => "kernel_mul_mm_q5_K_f32",
            GgmlType::Q6_K => "kernel_mul_mm_q6_K_f32",
            GgmlType::Q5_1 => "kernel_mul_mm_q5_1_f32",
            GgmlType::IQ4_NL => "kernel_mul_mm_iq4_nl_f32",
            // ADR-033 §Pi 2026-05-22 — Metal mm kernel port pending (Task #16).
            GgmlType::IQ4_XS => "unsupported",
        }
    }

    /// Metal kernel function name for the tensor-API matrix-matrix
    /// variant (ADR-011 Phase 3 Wave P3b-tensor).  On M3+ this path uses
    /// `mpp::tensor_ops::matmul2d<>` which hits the hardware tensor cores
    /// for 2-3× the FLOP throughput of the simdgroup MMA variant.
    pub(crate) fn mm_tensor_kernel_name(self) -> &'static str {
        match self {
            // ADR-022 Phase 2: Q5_K tensor mm landed.
            // ADR-022 Phase 3: Q4_K tensor mm landed.
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 | GgmlType::I32 => {
                "unsupported"
            }
            GgmlType::Q2_K => "kernel_mul_mm_q2_K_tensor_f32",
            GgmlType::Q3_K => "kernel_mul_mm_q3_K_tensor_f32",
            GgmlType::Q4_0 => "kernel_mul_mm_q4_0_tensor_f32",
            GgmlType::Q5_0 => "kernel_mul_mm_q5_0_tensor_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_q8_0_tensor_f32",
            GgmlType::Q4_K => "kernel_mul_mm_q4_K_tensor_f32",
            GgmlType::Q5_K => "kernel_mul_mm_q5_K_tensor_f32",
            GgmlType::Q6_K => "kernel_mul_mm_q6_K_tensor_f32",
            GgmlType::Q5_1 => "kernel_mul_mm_q5_1_tensor_f32",
            GgmlType::IQ4_NL => "kernel_mul_mm_iq4_nl_tensor_f32",
            // ADR-033 §Pi 2026-05-22 — Metal tensor mm port pending (Task #16).
            GgmlType::IQ4_XS => "unsupported",
        }
    }

    /// ADR-029 H28-A — V2 large-tile tensor mm-kernel names.
    /// 64 (M tile) × 128 (N tile) output tile, direct-device B-read (no
    /// shmem staging), 4 simdgroups.  Peer port of the modern tensor
    /// kernel layout.
    pub(crate) fn mm_tensor_v2_kernel_name(self) -> &'static str {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 | GgmlType::I32 => {
                "unsupported"
            }
            GgmlType::Q2_K => "kernel_mul_mm_q2_K_tensor_v2_f32",
            GgmlType::Q3_K => "kernel_mul_mm_q3_K_tensor_v2_f32",
            GgmlType::Q4_0 => "kernel_mul_mm_q4_0_tensor_v2_f32",
            GgmlType::Q5_0 => "kernel_mul_mm_q5_0_tensor_v2_f32",
            GgmlType::Q8_0 => "kernel_mul_mm_q8_0_tensor_v2_f32",
            GgmlType::Q4_K => "kernel_mul_mm_q4_K_tensor_v2_f32",
            GgmlType::Q5_K => "kernel_mul_mm_q5_K_tensor_v2_f32",
            GgmlType::Q6_K => "kernel_mul_mm_q6_K_tensor_v2_f32",
            GgmlType::Q5_1 => "kernel_mul_mm_q5_1_tensor_v2_f32",
            GgmlType::IQ4_NL => "kernel_mul_mm_iq4_nl_tensor_v2_f32",
            // ADR-033 §Pi 2026-05-22 — Metal tensor-v2 mm port pending (Task #16).
            GgmlType::IQ4_XS => "unsupported",
        }
    }
}

/// Device-bound tensor-API availability. The registry caches the exact
/// specialized-pipeline result; later calls still perform the registry-id and
/// cache-key lookup but do not recompile the probe pipeline.
fn probe_tensor_mm(registry: &mut KernelRegistry, device: &MlxDevice) -> Result<bool> {
    let probe = registry.probe_optional_pipeline_with_constants(
        "kernel_mul_mm_q4_0_tensor_f32",
        device.metal_device(),
        device.registry_id(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
        is_unavailable_tensor_header,
    )?;
    if probe.newly_probed && std::env::var("MLX_LOG_TENSOR_PROBE").is_ok() {
        eprintln!(
            "[mlx-native] tensor_mm probe: {}",
            if probe.available {
                "OK (using tensor variant)"
            } else {
                "FAILED (falling back to simdgroup MMA)"
            }
        );
    }
    Ok(probe.available)
}

pub(crate) fn dense_routing_policy_from_environment() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        dense_decode_mvn: cached_env_default_true(&CACHED_DECODE_MVN, "HF2Q_DECODE_MVN"),
        dense_decode_mv_ext: cached_env_eq_one(&CACHED_DECODE_MV_EXT, "HF2Q_DECODE_MV_EXT"),
        dense_q6k_mv_nr2: cached_env_default_true(&CACHED_Q6K_MV_NR2, "HF2Q_Q6K_MV_NR2"),
        dense_q8_0_mv_nr2: cached_env_default_true(&CACHED_Q8_0_MV_NR2, "HF2Q_Q8_0_MV_NR2"),
        dense_tensor_mm: if std::env::var("HF2Q_DISABLE_TENSOR_MM").as_deref() == Ok("1") {
            GgmlTensorMmPreference::ForceSimd
        } else {
            GgmlTensorMmPreference::AutoProbe
        },
        allow_dense_large_tile_mm: !matches!(
            std::env::var("HF2Q_LARGE_TILE_MM").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        ),
        ..GgmlRoutingPolicy::default()
    }
}

/// The `ne11_mm_min` threshold (reference-implementation convention) for
/// routing between mat-vec and mat-mat.
/// At prefill m > 8, the mm kernel's threadgroup-staged weight tile reuse
/// beats the mv kernel's per-row DRAM re-read by 5-30x on Apple Silicon.
/// At m <= 8 the mv kernel wins on launch overhead for narrow inputs.
pub const MM_ROUTING_THRESHOLD: u32 = 8;

/// Parameters for GGML block-format quantized mat-vec.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulParams {
    /// Number of input rows (1 for decode).
    pub m: u32,
    /// Number of output columns (weight rows).
    pub n: u32,
    /// Input dimension (weight cols before quantization).
    /// Must be divisible by the block's QK value.
    pub k: u32,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
}

/// Parameters for independent batched GGML block-format matrix products.
///
/// Buffers use contiguous `[batch, m, k]` input, `[batch, n, k]` weight,
/// and `[batch, m, n]` output layouts. Each batch entry is an independent
/// matrix product; no weight or activation broadcasting is performed.
#[derive(Debug, Clone, Copy)]
pub struct GgmlBatchedQuantizedMatmulParams {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub ggml_type: GgmlType,
}

/// Byte strides for an F32 batched-MM input view.
///
/// `row_bytes` advances between logical input rows and `batch_bytes`
/// advances between independent products. This supports both packed
/// `[batch, m, k]` storage and interleaved token-major `[m, batch, k]`
/// storage without materializing a permutation.
#[derive(Debug, Clone, Copy)]
pub struct GgmlBatchedQuantizedMatmulInputStrides {
    pub row_bytes: u64,
    pub batch_bytes: u64,
}

/// GPU-side params struct — must match the Metal shader's `GgmlMatvecParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatvecGpuParams {
    ne00: i64, // K
    ne01: i64, // N
    ne02: i64, // batch (weights)
    ne10: i64, // K
    ne12: i64, // batch (input)
    ne0: i64,  // N (output stride)
    ne1: i64,  // M
    r2: u32,   // ne12/ne02
    r3: u32,   // always 1
}

/// GPU-side params struct for the mat-mat (mm) kernel.
/// Must match `GgmlMatmulMmParams` in
/// `/opt/mlx-native/src/shaders/quantized_matmul_mm.metal`.
///
/// Explicit 4-byte padding is inserted between `ne12` and `nb10` so the
/// Rust struct has deterministic layout and matches the natural Metal
/// struct alignment (u64 members align to 8 bytes).  bytemuck::Pod
/// requires no implicit padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatmulMmGpuParams {
    ne00: i32,  // K
    ne02: i32,  // batch(src0), always 1 for our projections
    nb01: u64,  // bytes per weight row
    nb02: u64,  // bytes per weight batch
    nb03: u64,  // unused (always 0)
    ne12: i32,  // batch(src1), always 1
    _pad0: u32, // align nb10 to 8
    nb10: u64,  // = sizeof(float) = 4
    nb11: u64,  // bytes per input row = K * sizeof(float)
    nb12: u64,  // bytes per input batch = M * nb11
    nb13: u64,  // unused
    ne0: i32,   // N (output stride)
    ne1: i32,   // M
    r2: i16,    // 1
    r3: i16,    // 1
    _pad1: u32, // trailing pad so sizeof == multiple of 8 (u64 align)
}

/// Quantized matmul for GGML block format weights.
///
/// Weight buffer contains raw GGML blocks (same bytes as GGUF on disk).
/// Input is f32, output is f32.
///
/// Routes between two Metal kernels based on `m`:
///
/// - `m <= MM_ROUTING_THRESHOLD` (8) -> uses the matrix-vector kernel
///   (`kernel_mul_mv_q*_f32`).  Lower launch overhead; one output row
///   per threadgroup-block in the M axis.
/// - `m > MM_ROUTING_THRESHOLD` -> uses the matrix-matrix kernel
///   (`kernel_mul_mm_q*_f32`, ADR-011 Phase 3 peer port).
///   Tiles the input at 64x32 and stages a dequantized weight tile into
///   threadgroup shared memory, reusing each weight block across a 32-row
///   block of inputs.  At prefill m=2455 this is ~32x less DRAM traffic.
///
/// The threshold matches the reference `ne11_mm_min = 8`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - K is not divisible by the GGML block QK value
/// - Buffer sizes don't match expected dimensions
/// - M, K, or N are zero
pub fn quantized_matmul_ggml(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_ggml_with_policy(
        encoder, registry, device, input, weight, output, params, &routing,
    )
}

/// Execute the canonical dense GGUF operation under an explicit routing
/// policy. This is the receipt-bindable entry point used by allocators and
/// benchmark harnesses; unlike [`quantized_matmul_ggml`], it does not consult
/// process-global routing overrides.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_ggml_with_policy(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    // ADR-028: output: &MlxBuffer (was &mut).  Encoders never mutate
    // through Rust refs — only via metal_buffer() / contents_ptr() (&self).
    // Relaxing to &MlxBuffer enables Arc<MlxBuffer> sharing across threads
    // for the multi-thread encoding port (peer's n_cb=2 pattern).
    validate_native_quantized_dtypes("quantized_matmul_ggml", input, weight, output)?;
    validate_signed_metal_dimensions(
        "quantized_matmul_ggml",
        &[("M", params.m), ("N", params.n), ("K", params.k)],
    )?;

    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();

    // --- Validate (common to mv and mm paths) ---
    // ADR-013 P7 — Q4_K added (mv only; mm path falls back to mv at m <= 8
    // and Q4_K's mm/mm_tensor kernels are not yet ported, so we only allow
    // Q4_K when the dispatcher would route to mv).
    match params.ggml_type {
        // ADR-022 Phase 1 — Q5_1 / IQ4_NL added (mv-only; mm/mm_tensor
        // come in P1.6, dispatcher already routes to mv at m ≤ 8).
        // ADR-022 Phase 2 — Q5_K added (mv + mm + mm_tensor).
        GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        // ADR-033 §Pi Task #16 2026-05-22 — mv kernel ported. mm
        // kernel pending; dispatcher routes to mv path at m ≤
        // MM_ROUTING_THRESHOLD = 8 (the critical decode-time hot
        // path on Qwen MoE expert tensors).
        | GgmlType::IQ4_XS => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "quantized_matmul_ggml does not support {:?} — use a different dispatch path",
                other
            )));
        }
    }
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }

    let blocks_per_row = params.k / qk;
    let expected_weight_bytes = checked_byte_extent(
        "weight",
        &[
            params.n as usize,
            blocks_per_row as usize,
            block_bytes as usize,
        ],
    )?;
    if weight.data_byte_len() < expected_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Weight buffer too small: expected {} bytes for {:?} [{}x{}], got {}",
            expected_weight_bytes,
            params.ggml_type,
            params.n,
            params.k,
            weight.data_byte_len()
        )));
    }

    let expected_input_bytes = checked_byte_extent(
        "input",
        &[params.m as usize, params.k as usize, DType::F32.size_of()],
    )?;
    if input.data_byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Input buffer too small: expected {} bytes for [{}x{}] f32, got {}",
            expected_input_bytes,
            params.m,
            params.k,
            input.data_byte_len()
        )));
    }

    let expected_output_bytes = checked_byte_extent(
        "output",
        &[params.m as usize, params.n as usize, DType::F32.size_of()],
    )?;
    if output.data_byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "Output buffer too small: expected {} bytes for [{}x{}] f32, got {}",
            expected_output_bytes,
            params.m,
            params.n,
            output.data_byte_len()
        )));
    }

    // ADR-011 Phase 3 Wave P3a — route on m threshold.
    //
    // The mm kernel stages a 64x32 weight tile into threadgroup shared
    // memory and reuses it across a 32-row block of M.  This cuts DRAM
    // weight-read bandwidth by ~32x at prefill m=2455 and delivers a
    // 5-30x per-kernel speedup over the mv path (which re-reads every
    // weight block once per M row).  The mv path is still preferable at
    // low M (decode m=1, short-prompt prefill m<=8) where launch overhead
    // dominates tile reuse savings.
    //
    // Threshold matches the reference `ne11_mm_min = 8`.
    // The mm kernel also requires K >= NK=32,
    // which every projection in our Gemma 4 DWQ model satisfies — guard
    // kept so any future shape smaller than 32 falls back to mv.
    // ADR-013 P7 — Q4_K mm/mm_tensor not yet ported; Q4_K always
    // routes to mv (correct but slower for large m than a fully ported
    // mm).  dwq46/dwq48 dense Q4_K shexp/gate_inp tensors are small
    // (router weights with N <= 256), so the perf delta is negligible
    // in practice.  Other ggml types route on m as before.
    // ADR-022 Phase 3 — Q4_K dense mm + mm_tensor ported. All quantized
    // types now have a real mm path; the mm_supported guard is a
    // compatibility no-op kept for future "type not yet ported" cases.
    // Types without a parity-proven MM kernel remain on the matvec fallback.
    // Q2_K joins the proven K-quant MM family; IQ4_XS is still mv-only here.
    let mm_supported = !matches!(params.ggml_type, GgmlType::IQ4_XS);
    // ADR-040 §0.21 decode F3 lever: at small continuous-batching decode width
    // m∈[2,8], route quantized matmuls to the weight-amortizing `mul_mv_ext`
    // kernel (the reference small-batch path:
    // `r1ptg` src1 columns processed per threadgroup, so each quantized weight
    // block is read ONCE across the m columns — vs the regular `mv` which
    // re-reads the weight per column, measured ~5× at m=8). mul_mv_ext is
    // Numerically close but not generally byte-identical to mv: the wider
    // kernel uses a different reduction tree. A family must therefore qualify
    // model decisions and cache handoff before enabling it in production.
    // K-quants use m=4..8; legacy Q4_0/Q5_0/Q8_0 retain m=2..8. Unsupported shapes
    // fall through unchanged.
    // ADR-040 §0.21 decode F3 lever (mul_mv_ext weight-amortization) — opt-in,
    // DEFAULT OFF (HF2Q_DECODE_MV_EXT=1). MEASURED: routing decode m∈[2,8] to
    // mul_mv_ext gives N=8 decode 197→245.8 t/s (+25%, toward llama 291) — the
    // weight-reload gap is real and this closes a big chunk. BUT it currently
    // BREAKS slot_aware_n8_per_slot_parity (bit-exact) in the gemma4 model —
    // mul_mv_ext is NOT bit-identical to the regular mv for the model's shapes
    // (the adr_022_phase4 mv_ext parity tests use fp-tolerance, and Q6_K mv_ext
    // has no parity test at all). So it stays OFF by default (byte-identical
    // bar) until the mv_ext kernels are proven/made bit-exact (Q6_K especially);
    // the env preserves the validated speedup for that follow-up. NON-blocking.
    // Exact decode mvN lever — BIT-IDENTICAL column-amortizing Q4_K, Q5_K,
    // and Q6_K mat-vec. Each physical kernel preserves its production scalar
    // accumulation tree independently per src1 column while sharing packed
    // weight reads. Q4_K and Q5_K clone their ordinary scalar kernels; Q6_K
    // clones its default NR2 kernel. The to_bits gates cover every logical
    // width m∈[2,8], including multi-tile buffer offsets. mul_mv_ext remains a
    // separate non-bit-identical diagnostic route.
    // ADR-040 §0.21c-track2: DEFAULT-ON (lead-approved 2026-06-26). Opt out with
    // HF2Q_DECODE_MVN=0. Bar met: byte-equal spike GREEN under the precompiled
    // -O3 metallib + release-deterministic parity (after the encoder-retain root
    // fix that the mvN flake surfaced — encoder.rs §0.21c-track2) + measured
    // +12.5% net N=8 decode throughput (clean single-tenant).
    match plan_dense_auto_route(params.ggml_type, params.m, params.k, routing) {
        DenseAutoPlan::Q4kWidthMn => {
            debug_assert!(params.k % QK4_K == 0);
            if std::env::var("HF2Q_DECODE_MVN_TRACE").is_ok() {
                eprintln!(
                    "[mvN-route] Q4_K m={} n={} k={} → mN tiles={:?}",
                    params.m,
                    params.n,
                    params.k,
                    mn_column_tiling(params.m as usize)
                );
            }
            dispatch_mv_q4k_mn_adaptive(encoder, registry, device, input, weight, output, params)
        }
        DenseAutoPlan::Q5kWidthMn => {
            debug_assert!(params.k % QK5_K == 0);
            if std::env::var("HF2Q_DECODE_MVN_TRACE").is_ok() {
                eprintln!(
                    "[mvN-route] Q5_K m={} n={} k={} → mN tiles={:?}",
                    params.m,
                    params.n,
                    params.k,
                    mn_column_tiling(params.m as usize)
                );
            }
            dispatch_mv_q5k_mn_adaptive(encoder, registry, device, input, weight, output, params)
        }
        DenseAutoPlan::Q6kWidthMn => {
            debug_assert!(params.k % QK6_K == 0);
            if std::env::var("HF2Q_DECODE_MVN_TRACE").is_ok() {
                eprintln!(
                    "[mvN-route] Q6_K m={} n={} k={} → mN tiles={:?}",
                    params.m,
                    params.n,
                    params.k,
                    mn_column_tiling(params.m as usize)
                );
            }
            dispatch_mv_q6k_mn_adaptive(encoder, registry, device, input, weight, output, params)
        }
        DenseAutoPlan::WidthMvExt => {
            let ext_params = crate::ops::mul_mv_ext::MulMvExtParams {
                m: params.m,
                n: params.n,
                k: params.k,
                batch: 1,
                ggml_type: params.ggml_type,
            };
            crate::ops::mul_mv_ext::mul_mv_ext_dispatch(
                encoder,
                registry,
                device,
                weight,
                input,
                output,
                &ext_params,
            )
        }
        DenseAutoPlan::Mm => {
            debug_assert!(mm_supported);
            dispatch_mm(
                encoder, registry, device, input, weight, output, params, 1, None, routing,
            )
        }
        DenseAutoPlan::Mv => dispatch_mv(
            encoder, registry, device, input, weight, output, params, routing,
        ),
    }
}

/// Execute the canonical dense GGUF operation under an explicit routing
/// policy and return typed evidence for the exact Metal dispatches encoded by
/// this call. The trace scope is fail-closed and cannot coexist with graph
/// capture, which records instead of executing. The receipt proves encoding,
/// not command-buffer completion or latency.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_ggml_with_policy_and_trace(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
    workload: GgmlWorkloadClass,
) -> Result<GgmlResolvedDispatchTrace> {
    let request = GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::DenseAuto {
            m: params.m,
            n: params.n,
            k: params.k,
        },
        ggml_type: params.ggml_type,
        workload,
        routing: *routing,
    };
    trace_ggml_operation(encoder, registry, device, request, |encoder, registry| {
        quantized_matmul_ggml_with_policy(
            encoder, registry, device, input, weight, output, params, routing,
        )
    })
}

/// Execute independent quantized matrix products through the MM kernel's
/// native batch dimension.
///
/// This is the GGML/Metal 3-D `mul_mat` contract used for grouped
/// projections: one command dispatch spans `batch` independent products while
/// preserving the same per-product MM arithmetic as `quantized_matmul_ggml`.
/// The current entry point is deliberately MM-only (`m > 8`); small batches
/// retain their format-specific mat-vec routing through the scalar API.
pub fn quantized_matmul_ggml_batched_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_ggml_batched_mm_with_policy(
        encoder, registry, device, input, weight, output, params, &routing,
    )
}

/// Batched MM under an explicit receipt-bindable routing policy.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_ggml_batched_mm_with_policy(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    let element_bytes = DType::F32.size_of() as u64;
    let row_bytes = u64::from(params.k)
        .checked_mul(element_bytes)
        .ok_or_else(|| {
            MlxError::InvalidArgument("batched quantized MM input row stride overflows".into())
        })?;
    let batch_bytes = row_bytes.checked_mul(u64::from(params.m)).ok_or_else(|| {
        MlxError::InvalidArgument("batched quantized MM input batch stride overflows".into())
    })?;
    quantized_matmul_ggml_batched_mm_strided_input_with_policy(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        &GgmlBatchedQuantizedMatmulInputStrides {
            row_bytes,
            batch_bytes,
        },
        routing,
    )
}

/// Execute independent quantized matrix products from an explicitly-strided
/// F32 input view through the MM kernel's native batch dimension.
pub fn quantized_matmul_ggml_batched_mm_strided_input(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
    input_strides: &GgmlBatchedQuantizedMatmulInputStrides,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_ggml_batched_mm_strided_input_with_policy(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        input_strides,
        &routing,
    )
}

/// Strided-input batched MM under an explicit routing policy.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_ggml_batched_mm_strided_input_with_policy(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
    input_strides: &GgmlBatchedQuantizedMatmulInputStrides,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    validate_signed_metal_dimensions(
        "batched quantized MM",
        &[
            ("batch", params.batch),
            ("M", params.m),
            ("N", params.n),
            ("K", params.k),
        ],
    )?;
    if params.batch == 0 || params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "batched quantized MM dimensions must all be nonzero".into(),
        ));
    }
    if params.m <= MM_ROUTING_THRESHOLD {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MM requires m > {MM_ROUTING_THRESHOLD}, got {}",
            params.m
        )));
    }
    if matches!(params.ggml_type, GgmlType::IQ4_XS) {
        return Err(MlxError::InvalidArgument(
            "batched quantized MM does not support IQ4_XS".into(),
        ));
    }
    if input.dtype() != DType::F32 || weight.dtype() != DType::U8 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MM requires F32/U8/F32 buffers, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            output.dtype()
        )));
    }

    let qk = params.ggml_type.block_values();
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({qk})",
            params.k
        )));
    }
    let checked_bytes = |dimensions: &[u32], element_bytes: usize, label: &str| -> Result<usize> {
        dimensions
            .iter()
            .try_fold(element_bytes, |bytes, &dimension| {
                bytes.checked_mul(dimension as usize).ok_or_else(|| {
                    MlxError::InvalidArgument(format!(
                        "batched quantized MM {label} byte length overflows"
                    ))
                })
            })
    };
    let element_bytes = DType::F32.size_of() as u64;
    let logical_row_bytes = u64::from(params.k)
        .checked_mul(element_bytes)
        .ok_or_else(|| {
            MlxError::InvalidArgument("batched quantized MM input row length overflows".into())
        })?;
    const INPUT_STRIDE_ALIGNMENT_BYTES: u64 = 32;
    if input_strides.row_bytes < logical_row_bytes
        || input_strides.batch_bytes < logical_row_bytes
        || input_strides.row_bytes % INPUT_STRIDE_ALIGNMENT_BYTES != 0
        || input_strides.batch_bytes % INPUT_STRIDE_ALIGNMENT_BYTES != 0
    {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MM input strides must be {INPUT_STRIDE_ALIGNMENT_BYTES}-byte aligned and at least one logical row ({logical_row_bytes} bytes), got row={} batch={}",
            input_strides.row_bytes, input_strides.batch_bytes
        )));
    }
    let input_bytes_u64 = u64::from(params.batch - 1)
        .checked_mul(input_strides.batch_bytes)
        .and_then(|bytes| {
            u64::from(params.m - 1)
                .checked_mul(input_strides.row_bytes)
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| bytes.checked_add(logical_row_bytes))
        .ok_or_else(|| {
            MlxError::InvalidArgument("batched quantized MM strided input range overflows".into())
        })?;
    let input_bytes = usize::try_from(input_bytes_u64).map_err(|_| {
        MlxError::InvalidArgument(
            "batched quantized MM strided input range exceeds address space".into(),
        )
    })?;
    let weight_bytes = checked_bytes(
        &[params.batch, params.n, params.k / qk],
        params.ggml_type.block_bytes() as usize,
        "weight",
    )?;
    let output_bytes = checked_bytes(
        &[params.batch, params.m, params.n],
        DType::F32.size_of(),
        "output",
    )?;
    for (buffer, required, label) in [
        (input, input_bytes, "input"),
        (weight, weight_bytes, "weight"),
        (output, output_bytes, "output"),
    ] {
        if buffer.data_byte_len() < required {
            return Err(MlxError::InvalidArgument(format!(
                "batched quantized MM {label} buffer needs {required} bytes, got {}",
                buffer.data_byte_len()
            )));
        }
    }

    let scalar = GgmlQuantizedMatmulParams {
        m: params.m,
        n: params.n,
        k: params.k,
        ggml_type: params.ggml_type,
    };
    dispatch_mm(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        &scalar,
        params.batch,
        Some(input_strides),
        routing,
    )
}

/// Execute independent matrix-vector products through the GGML kernel's
/// native batch dimension.
///
/// Layouts are input `[batch, m, k]`, weights `[batch, n, k]` in GGML block
/// storage, and output `[batch, m, n]`. Each z-grid slice executes the same
/// format-specific mat-vec kernel and accumulation order as an independent
/// `quantized_matmul_ggml` call.
pub fn quantized_matmul_ggml_batched_mv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_ggml_batched_mv_with_policy(
        encoder, registry, device, input, weight, output, params, &routing,
    )
}

/// Batched MV under an explicit receipt-bindable routing policy.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_ggml_batched_mv_with_policy(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlBatchedQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    validate_signed_metal_dimensions(
        "batched quantized MV",
        &[
            ("batch", params.batch),
            ("M", params.m),
            ("N", params.n),
            ("K", params.k),
        ],
    )?;
    if params.batch == 0 || params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "batched quantized MV dimensions must all be nonzero".into(),
        ));
    }
    if params.m > MM_ROUTING_THRESHOLD {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MV supports m <= {MM_ROUTING_THRESHOLD}, got {}",
            params.m
        )));
    }
    if matches!(
        params.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::I16 | GgmlType::I32
    ) {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MV does not support {:?}",
            params.ggml_type
        )));
    }
    let qk = params.ggml_type.block_values();
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MV input width {} is not divisible by {qk}",
            params.k
        )));
    }
    if input.dtype() != DType::F32 || weight.dtype() != DType::U8 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "batched quantized MV requires F32/U8/F32 buffers, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight.dtype(),
            output.dtype()
        )));
    }
    let checked_bytes = |dimensions: &[u32], element_bytes: usize, label: &str| -> Result<usize> {
        dimensions
            .iter()
            .try_fold(element_bytes, |bytes, &dimension| {
                bytes.checked_mul(dimension as usize).ok_or_else(|| {
                    MlxError::InvalidArgument(format!(
                        "batched quantized MV {label} byte length overflows"
                    ))
                })
            })
    };
    let input_bytes = checked_bytes(
        &[params.batch, params.m, params.k],
        DType::F32.size_of(),
        "input",
    )?;
    let weight_bytes = checked_bytes(
        &[params.batch, params.n, params.k / qk],
        params.ggml_type.block_bytes() as usize,
        "weight",
    )?;
    let output_bytes = checked_bytes(
        &[params.batch, params.m, params.n],
        DType::F32.size_of(),
        "output",
    )?;
    for (buffer, required, label) in [
        (input, input_bytes, "input"),
        (weight, weight_bytes, "weight"),
        (output, output_bytes, "output"),
    ] {
        if buffer.data_byte_len() < required {
            return Err(MlxError::InvalidArgument(format!(
                "batched quantized MV {label} buffer needs {required} bytes, got {}",
                buffer.data_byte_len()
            )));
        }
    }

    let scalar = GgmlQuantizedMatmulParams {
        m: params.m,
        n: params.n,
        k: params.k,
        ggml_type: params.ggml_type,
    };
    dispatch_mv_batched(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        &scalar,
        params.batch,
        routing,
    )
}

/// Compatibility entry point for the original DeepSeek Q2_K grouped matvec.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_q2_k_batched_mv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    quantized_matmul_ggml_batched_mv(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        &GgmlBatchedQuantizedMatmulParams {
            batch,
            m,
            n,
            k,
            ggml_type: GgmlType::Q2_K,
        },
    )
}

/// ADR-029 H29-speed: dispatch the V2 64×128 large-tile mm-tensor
/// kernel with F16 weight input (no dequant — reads from a pre-materialized
/// F16 shadow buffer).  Mirrors `dispatch_mm` geometry / shmem / dispatch
/// for the V2 path but skips the per-call dequantize_func work.
///
/// `f16_weight` is the F16-typed MlxBuffer (per-row stride = K halfs =
/// 2K bytes).  `input` is F32 [m, k].  `output` is F32 [m, n].
pub fn dispatch_mm_v2_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    f16_weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if f16_weight.dtype() != DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mm_v2_f16: f16_weight must be F16, got {:?}",
            f16_weight.dtype()
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(MlxError::InvalidArgument(
            "dispatch_mm_v2_f16: M, K, N must all be > 0".into(),
        ));
    }
    // F16 weight row stride (per-row bytes) = 2K.
    let nb01 = (k as u64) * (DType::F16.size_of() as u64);
    let nb11 = (k as u64) * (DType::F32.size_of() as u64);

    let gpu_params = GgmlMatmulMmGpuParams {
        ne00: k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * (n as u64),
        nb03: 0,
        ne12: 1,
        _pad0: 0,
        nb10: DType::F32.size_of() as u64,
        nb11,
        nb12: nb11 * (m as u64),
        nb13: 0,
        ne0: n as i32,
        ne1: m as i32,
        r2: 1,
        r3: 1,
        _pad1: 0,
    };

    let pipeline = registry.get_pipeline_with_constants(
        "hf2q_mul_mm_tensor_v2_f16",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    const THREADS_PER_TG: u64 = 128;
    let nra: u64 = 64; // M_peer tile
    let nrb: u64 = 128; // N_peer tile
    let tg_x = (m as u64 + nrb - 1) / nrb;
    let tg_y = (n as u64 + nra - 1) / nra;
    let threadgroups = metal::MTLSize::new(tg_x, tg_y, 1);
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);
    const SHMEM_BYTES: u64 = 4096; // only A tile in shmem

    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(f16_weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Test-only helper: force the mm dispatch path.  Used by the mm parity
/// tests (`tests/test_quantized_matmul_mm.rs`).  This entry point
/// intentionally bypasses the public dispatcher's routing decision so
/// that tests can verify mm vs mv parity at every M (including the
/// m <= 8 range where the production dispatcher normally picks mv).
///
/// Not intended for production callers — use `quantized_matmul_ggml`
/// above, which routes by m.
#[doc(hidden)]
pub fn dispatch_mm_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    validate_native_quantized_dtypes("dispatch_mm_for_test", input, weight, output)?;
    validate_mm_for_test(params)?;
    let routing = GgmlRoutingPolicy::default();
    dispatch_mm(
        encoder, registry, device, input, weight, output, params, 1, None, &routing,
    )
}

/// Test-only Q4_0 tensor-MM candidate with a 64-output by 32-token tile.
///
/// This retains the V2 kernel's native Q4_0 reads, direct F32 activation
/// tensor, 32-wide K loop, reduction order, and output layout. Only the token
/// tile and corresponding grid width differ.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mm_q4_0_tensor_64x32_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    validate_native_quantized_dtypes(
        "dispatch_mm_q4_0_tensor_64x32_for_test",
        input,
        weight,
        output,
    )?;
    validate_mm_for_test(params)?;
    dispatch_mm_q4_route_internal(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        DenseQ4Route::Tensor64x32,
    )
}

/// Test-only helper that forces the non-tensor simdgroup-MMA fallback.
/// This proves the path used on devices without Metal tensor support while
/// leaving the production capability probe and routing unchanged.
#[doc(hidden)]
pub fn dispatch_mm_simd_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    validate_native_quantized_dtypes("dispatch_mm_simd_for_test", input, weight, output)?;
    validate_mm_for_test(params)?;
    let routing = GgmlRoutingPolicy {
        dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    };
    dispatch_mm(
        encoder, registry, device, input, weight, output, params, 1, None, &routing,
    )
}

fn validate_mm_for_test(params: &GgmlQuantizedMatmulParams) -> Result<()> {
    // Re-run common validation so this entry point is safe on its own.
    let qk = params.ggml_type.block_values();
    match params.ggml_type {
        GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL => {}
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_mm_for_test does not support {:?}",
                other
            )));
        }
    }
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }
    Ok(())
}

/// Matrix-vector dispatch (original path, unchanged from pre-Phase-3).
/// Used for decode (m=1) and small-prompt prefills (m <= 8).
fn dispatch_mv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer, // ADR-028: was &mut, see public fn comment
    params: &GgmlQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_mv_batched(
        encoder, registry, device, input, weight, output, params, 1, routing,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mv_batched(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    batch: u32,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    // ADR-028 —nr0=2 variant for q6_K mat-vec.  Peer pattern:
    // 4 rows/TG (vs baseline's 2) + cached `yl[16]` (vs no cache + device
    // re-reads).  Bit-exact-equivalent to the baseline at HEAD (parity
    // test in tests/adr_028_iter309_q6k_mv_nr2_parity.rs).
    //
    // ADR-028 default-flipped to ON (operator REFRAME #2:
    // "default should have the best things on that provide the best
    // mantra-aligned outcome for users").  Opt out with
    // `HF2Q_Q6K_MV_NR2=0` / `=false` / `=off`.
    let use_q6k_nr2 = matches!(params.ggml_type, GgmlType::Q6_K) && routing.dense_q6k_mv_nr2;
    // Q8_0 NSG=4 NR=2 is the peer-pattern geometry. Exact-output
    // parity and DeepSeek-V4 end-to-end decode validation make it the
    // production default; operators can retain the legacy kernel with
    // `HF2Q_Q8_0_MV_NR2=0` / `=false` / `=off` for diagnostics.
    let use_q8_0_nr2 = matches!(params.ggml_type, GgmlType::Q8_0) && routing.dense_q8_0_mv_nr2;
    let kernel_name = if use_q6k_nr2 {
        "kernel_mul_mv_q6_K_f32_nr2"
    } else if use_q8_0_nr2 {
        "kernel_mul_mv_q8_0_f32_nr2"
    } else {
        params.ggml_type.kernel_name()
    };
    // ADR-029 H93: PSO-specialize batch divisors (ne12/r2/r3) at
    // function-constant slots 700/701/702. Peer-grounded port.
    // `ne12=batch` specializes the z-grid divisor for both
    // the ordinary batch=1 path and independent grouped products.
    // The redundant .clone() is omitted — registry is not
    // accessed again after pipeline lookup, so we can hold the &ComputePipelineState
    // reference across the rest of the function. Saves one objc retain/release
    // pair per dispatch.
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(700, batch as i32), (701, 1), (702, 1)],
    )?;

    let batch_i64 = i64::from(batch);
    let gpu_params = GgmlMatvecGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: batch_i64,
        ne10: params.k as i64,
        ne12: batch_i64,
        ne0: params.n as i64,
        ne1: params.m as i64,
        r2: 1,
        r3: 1,
    };

    let n = params.n as usize;
    let m = params.m as usize;

    let (nth0, nth1, align) = match params.ggml_type {
        // Q4_0 / Q5_0 / Q8_0 / Q5_1 / IQ4_NL all use legacy 32-element blocks
        // and the Q4_0-style (8, 8) threadgroup geometry: 2 simdgroups ×
        // 4 rows per simdgroup = 8 rows per threadgroup.
        GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::Q8_0
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        // ADR-033 §Pi Task #16 — kernel_mul_mv_iq4_xs_f32 ports
        // IQ4_NL's (N_SIMDGROUP=2, N_DST=4) geometry so the
        // launch tuple (8, 8, 8) is shared.
        | GgmlType::IQ4_XS => (8u64, 8u64, 8usize),
        // Q2_K uses two simdgroups with four output rows per simdgroup.
        GgmlType::Q2_K => (2u64, 32u64, 8usize),
        // Q4_K / Q5_K (ADR-022 Phase 2) mirror Q6_K's 2-row-per-tg geometry.
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => (2u64, 32u64, 2usize),
        GgmlType::Q3_K => (2u64, 32u64, 4usize),
        _ => unreachable!(),
    };
    // ADR-028 —nr0=2 variant doubles rows-per-TG to 4.  Same
    // 2 SGs × 32 threads layout, but each SG handles 2 rows so align=4.
    let align = if use_q6k_nr2 { 4usize } else { align };
    // ADR-028 —Q8_0 NR2 uses 32×4=128 threads/TG, 2 rows/TG.
    let (nth0, nth1, align) = if use_q8_0_nr2 {
        (32u64, 4u64, 2usize)
    } else {
        (nth0, nth1, align)
    };

    let threadgroups = metal::MTLSize::new(div_ceil(n, align) as u64, m as u64, batch as u64);
    let threads_per_tg = metal::MTLSize::new(nth0, nth1, 1);

    if use_q8_0_nr2 {
        // Cross-SG reduction needs threadgroup memory: NR0 * NW * sizeof(float).
        let smem_bytes: u64 = 2 * 32 * std::mem::size_of::<f32>() as u64;
        encoder.dispatch_tracked_threadgroups_with_args_and_shared(
            &pipeline,
            &[
                (0, KernelArg::Buffer(weight)),
                (1, KernelArg::Buffer(input)),
                (2, KernelArg::Buffer(output)),
                (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            ],
            &[(0, smem_bytes)],
            &[weight, input],
            &[output],
            threadgroups,
            threads_per_tg,
        );
    } else {
        // ADR-029: dataflow-tracked dispatch.  When
        // HF2Q_AUTO_BARRIER=1, the MemRanges tracker checks weight/input
        // against the cumulative state and auto-emits a barrier on RAW.
        // When HF2Q_AUTO_BARRIER=0 (default), this is identical to the
        // prior `encode_threadgroups_with_args` call (zero behavioral
        // diff in production until the env-flag default flips).
        encoder.dispatch_tracked_threadgroups_with_args(
            &pipeline,
            &[
                (0, KernelArg::Buffer(weight)),
                (1, KernelArg::Buffer(input)),
                (2, KernelArg::Buffer(output)),
                (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            ],
            &[weight, input],
            &[output],
            threadgroups,
            threads_per_tg,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mv_q4k_mn_chunk(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    r1ptg: usize,
    col0: usize,
    width: usize,
) -> Result<()> {
    if params.ggml_type != GgmlType::Q4_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q4k_mn: expected Q4_K weights, got {:?}",
            params.ggml_type
        )));
    }
    if !(2..=5).contains(&r1ptg) || r1ptg != width {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q4k_mn: tile width must equal r1ptg in 2..=5, got width={width}, r1ptg={r1ptg}"
        )));
    }

    let kernel_name = match r1ptg {
        2 => "kernel_mul_mv_q4_K_f32_mN_r1_2",
        3 => "kernel_mul_mv_q4_K_f32_mN_r1_3",
        4 => "kernel_mul_mv_q4_K_f32_mN_r1_4",
        5 => "kernel_mul_mv_q4_K_f32_mN_r1_5",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_mv_q4k_mn: r1ptg must be 2..=5, got {r1ptg}"
            )))
        }
    };

    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = GgmlMatvecGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: width as i64,
        r2: 1,
        r3: 1,
    };

    let f32_sz = DType::F32.size_of() as u64;
    let input_off = (col0 as u64) * (params.k as u64) * f32_sz;
    let output_off = (col0 as u64) * (params.n as u64) * f32_sz;
    let threadgroups = metal::MTLSize::new(
        div_ceil(params.n as usize, 2) as u64,
        div_ceil(width, r1ptg) as u64,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(2, 32, 1);

    encoder.dispatch_tracked_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::BufferWithOffset(input, input_off)),
            (2, KernelArg::BufferWithOffset(output, output_off)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[weight, input],
        &[output],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Tile a Q4_K m∈[2,8] batch into the same register-safe widths used by
/// Q6_K. Columns are independent, so tiling preserves serial byte identity.
pub fn dispatch_mv_q4k_mn_adaptive(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    let m = params.m as usize;
    if params.ggml_type != GgmlType::Q4_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q4k_mn_adaptive: expected Q4_K weights, got {:?}",
            params.ggml_type
        )));
    }
    if !(2..=MM_ROUTING_THRESHOLD as usize).contains(&m) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q4k_mn_adaptive: m must be 2..={MM_ROUTING_THRESHOLD}, got {m}"
        )));
    }
    for (col0, width) in mn_column_tiling(m) {
        dispatch_mv_q4k_mn_chunk(
            encoder, registry, device, input, weight, output, params, width, col0, width,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mv_q5k_mn_chunk(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    r1ptg: usize,
    col0: usize,
    width: usize,
) -> Result<()> {
    if params.ggml_type != GgmlType::Q5_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q5k_mn: expected Q5_K weights, got {:?}",
            params.ggml_type
        )));
    }
    if !(2..=5).contains(&r1ptg) || r1ptg != width {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q5k_mn: tile width must equal r1ptg in 2..=5, got width={width}, r1ptg={r1ptg}"
        )));
    }

    let kernel_name = match r1ptg {
        2 => "kernel_mul_mv_q5_K_f32_mN_r1_2",
        3 => "kernel_mul_mv_q5_K_f32_mN_r1_3",
        4 => "kernel_mul_mv_q5_K_f32_mN_r1_4",
        5 => "kernel_mul_mv_q5_K_f32_mN_r1_5",
        _ => unreachable!(),
    };
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;
    let gpu_params = GgmlMatvecGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: width as i64,
        r2: 1,
        r3: 1,
    };
    let f32_sz = DType::F32.size_of() as u64;
    let input_off = (col0 as u64) * (params.k as u64) * f32_sz;
    let output_off = (col0 as u64) * (params.n as u64) * f32_sz;
    encoder.dispatch_tracked_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::BufferWithOffset(input, input_off)),
            (2, KernelArg::BufferWithOffset(output, output_off)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[weight, input],
        &[output],
        metal::MTLSize::new(
            div_ceil(params.n as usize, 2) as u64,
            div_ceil(width, r1ptg) as u64,
            1,
        ),
        metal::MTLSize::new(2, 32, 1),
    );
    Ok(())
}

/// Tile a Q5_K m∈[2,8] batch into register-safe widths while preserving the
/// scalar Q5_K floating-point tree independently for every column.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mv_q5k_mn_adaptive(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    let m = params.m as usize;
    if params.ggml_type != GgmlType::Q5_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q5k_mn_adaptive: expected Q5_K weights, got {:?}",
            params.ggml_type
        )));
    }
    if !(2..=MM_ROUTING_THRESHOLD as usize).contains(&m) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_mv_q5k_mn_adaptive: m must be 2..={MM_ROUTING_THRESHOLD}, got {m}"
        )));
    }
    for (col0, width) in mn_column_tiling(m) {
        dispatch_mv_q5k_mn_chunk(
            encoder, registry, device, input, weight, output, params, width, col0, width,
        )?;
    }
    Ok(())
}

/// ADR-040 §0.21c — dispatch the bit-identical column-amortizing Q6_K mat-vec
/// (`kernel_mul_mv_q6_K_f32_mN_r1_{R1}`) as a SINGLE tile of width `r1ptg` over
/// a width-`params.m` batch (so `r1ptg` must equal `params.m`). For arbitrary
/// m∈[2,8] with register-safe tiling, prefer [`dispatch_mv_q6k_mn_adaptive`].
///
/// The kernel is BYTE-IDENTICAL to `kernel_mul_mv_q6_K_f32_nr2` (the gemma4
/// model's default serial decode kernel) — it is a literal clone of NR2's
/// per-row block body (same `yl` cache, same operand form, same array `sumf`,
/// same `short` indexing) specialized across R1 columns instead of NR2's nr0=2
/// rows. NR2 is NOT byte-equal to plain mv in general, so cloning NR2 (not plain
/// mv) is what keeps the model's m=1-serial-vs-m=8-batched parity bit-exact.
/// The weight bytes for this SG's nr0 rows are read + dequantized once per block
/// and reused across all R1 columns (the amortization).
///
/// Geometry matches NR2: (2,32) threadgroup, 2 SGs × nr0=2 rows/SG → 4 rows/TG
/// (align=4 on N); the M axis is tiled by R1 (grid.y = ceil(m / R1)).
pub fn dispatch_mv_q6k_mn(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    r1ptg: usize,
) -> Result<()> {
    dispatch_mv_q6k_mn_chunk(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        r1ptg,
        0,
        params.m as usize,
    )
}

/// Dispatch a single mN tile for a contiguous column range `[col0, col0+width)`
/// of the m=`params.m` batch. `r1ptg` is the template width (must equal `width`,
/// in 2..=8). The src1/dst buffers are bound with byte offsets so the kernel's
/// chunk-local column index `c ∈ [0, width)` maps to the global column `col0+c`;
/// since columns are independent, any such tiling stays bit-identical.
#[allow(clippy::too_many_arguments)]
fn dispatch_mv_q6k_mn_chunk(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    r1ptg: usize,
    col0: usize,
    width: usize,
) -> Result<()> {
    debug_assert!(matches!(params.ggml_type, GgmlType::Q6_K));
    debug_assert!((2..=8).contains(&r1ptg));
    debug_assert_eq!(r1ptg, width);

    let kernel_name = match r1ptg {
        2 => "kernel_mul_mv_q6_K_f32_mN_r1_2",
        3 => "kernel_mul_mv_q6_K_f32_mN_r1_3",
        4 => "kernel_mul_mv_q6_K_f32_mN_r1_4",
        5 => "kernel_mul_mv_q6_K_f32_mN_r1_5",
        6 => "kernel_mul_mv_q6_K_f32_mN_r1_6",
        7 => "kernel_mul_mv_q6_K_f32_mN_r1_7",
        8 => "kernel_mul_mv_q6_K_f32_mN_r1_8",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "dispatch_mv_q6k_mn: r1ptg must be 2..=8, got {r1ptg}"
            )))
        }
    };

    // Same FC specialization (ne12/r2/r3 = 1) as dispatch_mv's hot path.
    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    // Chunk-local params: ne1 = chunk width (so the kernel's column boundary
    // guard `(r1_base + c) < ne1` is correct for this chunk's local indexing).
    let gpu_params = GgmlMatvecGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: width as i64,
        r2: 1,
        r3: 1,
    };

    let n = params.n as usize;

    // Byte offsets that shift src1/dst to the chunk's first column. The kernel's
    // chunk-local column index c∈[0,width) then maps to global column col0+c.
    //   src1 column stride = ne10 = k f32 elements
    //   dst  column stride = ne0  = n f32 elements
    let f32_sz = DType::F32.size_of() as u64;
    let input_off = (col0 as u64) * (params.k as u64) * f32_sz;
    let output_off = (col0 as u64) * (params.n as u64) * f32_sz;

    // Geometry matches NR2 (the bit-identity target): 2 SGs × 32 threads,
    // nr0=2 rows/SG → 4 rows/TG (align=4 on N). grid.y tiles this chunk's
    // `width` columns by R1 (= 1 TG-row since width == r1ptg).
    let align = 4usize;
    let threadgroups =
        metal::MTLSize::new(div_ceil(n, align) as u64, div_ceil(width, r1ptg) as u64, 1);
    let threads_per_tg = metal::MTLSize::new(2, 32, 1);

    encoder.dispatch_tracked_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::BufferWithOffset(input, input_off)),
            (2, KernelArg::BufferWithOffset(output, output_off)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[weight, input],
        &[output],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// Pick the column-tiling for a width-`m` batch that keeps each tile within the
/// register-safe single-tile width and balances tile widths. Empirically the
/// `yl_c[R1][16]` register cache spills past R1≈5 (measured: a throughput
/// cliff at R1≥6), so tiles are capped at 5 and split as evenly as possible.
/// Every tile width lands in 2..=5, EXCEPT a width-1 remainder which is handled
/// by the caller (m=1 never reaches mN; a tail width of 1 is merged up).
fn mn_column_tiling(m: usize) -> Vec<(usize, usize)> {
    // (col0, width) tiles. m ∈ [2,8] only.
    match m {
        2 => vec![(0, 2)],
        3 => vec![(0, 3)],
        4 => vec![(0, 4)],
        5 => vec![(0, 5)],
        6 => vec![(0, 3), (3, 3)],
        7 => vec![(0, 4), (4, 3)],
        8 => vec![(0, 4), (4, 4)],
        _ => vec![(0, m)], // unreachable for the gated m-range
    }
}

pub(crate) fn dense_mn_dispatch_count(m: u32) -> u32 {
    mn_column_tiling(m as usize).len() as u32
}

/// Adaptive entry point: tile the m∈[2,8] batch into register-safe column
/// chunks (each width 2..=5) and dispatch one mN tile per chunk. Bit-identity
/// is preserved because columns are independent — any column-tiling produces
/// the same per-column output. This is the routing target for HF2Q_DECODE_MVN.
pub fn dispatch_mv_q6k_mn_adaptive(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    for (col0, width) in mn_column_tiling(params.m as usize) {
        dispatch_mv_q6k_mn_chunk(
            encoder, registry, device, input, weight, output, params, width, col0, width,
        )?;
    }
    Ok(())
}

/// Build a pre-baked `DispatchRecord` for the Q6_K NR2 mat-vec
/// decode-m=1 path.
///
/// ADR-029: first concrete consumer of
/// [`DispatchRecord`].  The Q6_K NR2 path is the hottest single
/// per-token dispatch shape on gemma4-APEX-Q5_K_M decode
/// (Q/K/V proj × 30 layers + lm_head Q6_K = up to 91 dispatches/tok
/// at this kernel, plus an additional ~240 for MoE expert variants —
/// see `quantized_matmul_id_ggml::build_q6k_id_nr2_m1_record` once
/// that variant lands in Step 1e).
///
/// Pre-bakes:
///   - Pipeline reference (skips registry HashMap lookup per call)
///   - MTLSize threadgroups + threads_per_tg (skips MTLSize::new + match)
///   - GgmlMatvecGpuParams bytes (skips struct construction + bytemuck)
///   - Binding slot order: weight=0, input=1, output=2, params=3
///
/// Returns `None` if `HF2Q_Q6K_MV_NR2` is set to off (in which case
/// the legacy NR1 kernel is selected at dispatch_mv time and this
/// record would be wrong); the caller must fall back to the unbaked
/// path.
///
/// Bake-time validation: pipeline lookup must succeed; threadgroup
/// size is hard-coded to the Q6_K NR2 contract (2 × 32 = 64 threads,
/// align=4 rows/TG).
pub fn build_q6k_nr2_m1_record(
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    n: u32,
    k: u32,
) -> Result<Option<DispatchRecord>> {
    let routing = ggml_routing_policy_for_registry(registry);
    build_q6k_nr2_m1_record_with_policy(registry, device, n, k, &routing)
}

/// Explicit-policy form of [`build_q6k_nr2_m1_record`].
///
/// Use this when the dispatch record is part of a capability or performance
/// receipt. The record and its eventual execution then remain bound to the
/// same serialized policy even if the process environment differs.
pub fn build_q6k_nr2_m1_record_with_policy(
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    n: u32,
    k: u32,
    routing: &GgmlRoutingPolicy,
) -> Result<Option<DispatchRecord>> {
    // Only bakeable when the NR2 variant is the selected one.
    // (Matches the `use_q6k_nr2` decision in dispatch_mv.)
    if !routing.dense_q6k_mv_nr2 {
        return Ok(None);
    }

    // Pipeline lookup — same constants as the dispatch_mv hot path.
    let pipeline = registry
        .get_pipeline_with_constants(
            "kernel_mul_mv_q6_K_f32_nr2",
            device,
            &[],
            &[(700, 1), (701, 1), (702, 1)],
        )?
        .clone();

    // GgmlMatvecGpuParams for m=1.
    let gpu_params = GgmlMatvecGpuParams {
        ne00: k as i64,
        ne01: n as i64,
        ne02: 1,
        ne10: k as i64,
        ne12: 1,
        ne0: n as i64,
        ne1: 1,
        r2: 1,
        r3: 1,
    };
    let params_bytes = as_bytes(&gpu_params).to_vec();

    // Q6_K NR2: align=4 rows per TG, threads = (nth0=2, nth1=32, 1)
    // (matches dispatch_mv's Q6_K NR2 branch).
    const ALIGN: u32 = 4;
    let threadgroups = metal::MTLSize::new(div_ceil(n as usize, ALIGN as usize) as u64, 1, 1);
    let threads_per_tg = metal::MTLSize::new(2, 32, 1);

    Ok(Some(DispatchRecord {
        pipeline,
        threadgroups,
        threads_per_tg,
        threadgroup_mem: Vec::new(), // NR2 path doesn't use shmem
        params_bytes,
        params_slot: 3,
        buffer_slots: vec![0, 1, 2], // weight, input, output
        op_kind: CapturedOpKind::Other,
        kernel_name: "kernel_mul_mv_q6_K_f32_nr2".to_string(),
    }))
}

/// Matrix-matrix (mm) dispatch.  ADR-011 Phase 3 Wave P3a: peer port
/// of `kernel_mul_mm_<qtype>_f32`.  64x32 output tile, 4
/// simdgroups (128 threads), threadgroup-staged A+B with simdgroup MMA.
/// See `/opt/mlx-native/src/shaders/quantized_matmul_mm.metal`.
fn dispatch_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer, // ADR-028: was &mut, see public fn comment
    params: &GgmlQuantizedMatmulParams,
    batch: u32,
    input_strides: Option<&GgmlBatchedQuantizedMatmulInputStrides>,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_mm_impl(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        batch,
        input_strides,
        routing,
        None,
    )
}

/// Exact Q4 tensor route used by calibration and evidence tooling.
/// Ordinary inference calls [`dispatch_mm`] and may select the candidate only
/// through a frozen exact-shape plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mm_q4_route_internal(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    route: DenseQ4Route,
) -> Result<()> {
    if params.ggml_type != GgmlType::Q4_0 || params.m <= MM_ROUTING_THRESHOLD {
        return Err(MlxError::InvalidArgument(
            "forced dense Q4 MM route requires Q4_0 and M > 8".into(),
        ));
    }
    if !matches!(
        route,
        DenseQ4Route::CompatibilityV2 | DenseQ4Route::Tensor64x32
    ) {
        return Err(MlxError::InvalidArgument(
            "forced dense Q4 calibration route must be V2 or Tensor64x32".into(),
        ));
    }
    dispatch_mm_impl(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        params,
        1,
        None,
        &GgmlRoutingPolicy::default(),
        Some(route),
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mm_impl(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    batch: u32,
    input_strides: Option<&GgmlBatchedQuantizedMatmulInputStrides>,
    routing: &GgmlRoutingPolicy,
    forced_q4_route: Option<DenseQ4Route>,
) -> Result<()> {
    // ADR-011 Phase 3 Wave P3b-tensor — prefer the tensor_ops::matmul2d
    // variant on M3+ (hardware tensor cores); fall back to the simdgroup
    // MMA kernel if the probe fails or the tensor kernel can't compile
    // on this device.
    let use_tensor = tensor_mm_auto_selected(params.ggml_type, routing.dense_tensor_mm)
        && probe_tensor_mm(registry, device)?;
    // ADR-029 iter-23 H28-A — large-tile v2 mm-tensor kernel (64×128
    // output tile vs the v1 32×64).  Reduces threadgroup count by 4× at
    // prefill shapes (m=4213, n=5760: 11,880 → 2,970 tg).
    //
    // ADR-029 default-flip: validated across regimes —
    //   gemma4-APEX-Q5_K_M (2K/4K/8K prefill): +6–7% byte-identical
    //   qwen3.6-APEX-Q5_K_M (4K prefill): +0–2% byte-identical
    //   decode m=1 unaffected (V2 only fires at m > MM_ROUTING_THRESHOLD=8)
    // 3457/0/11 unit tests pass.  Default ON; opt-out via
    // `HF2Q_LARGE_TILE_MM=0` / `false` / `off`.
    let use_v2_large_tile = use_tensor && routing.allow_dense_large_tile_mm;
    let q4_decision = dense_q4_auto::select_route(
        registry,
        device,
        params,
        batch,
        input_strides.is_none(),
        routing,
    );
    let q4_route = forced_q4_route.unwrap_or(q4_decision.route);
    let mut use_q4_short_tile = use_v2_large_tile
        && params.ggml_type == GgmlType::Q4_0
        && batch == 1
        && input_strides.is_none()
        && q4_route == DenseQ4Route::Tensor64x32;
    if forced_q4_route == Some(DenseQ4Route::Tensor64x32) && !use_q4_short_tile {
        return Err(MlxError::InvalidArgument(
            "forced Q4 tensor 64x32 route is unavailable for this device or layout".into(),
        ));
    }
    let mut kernel_name = if use_q4_short_tile {
        DenseQ4Route::Tensor64x32.kernel_name()
    } else if use_v2_large_tile {
        params.ggml_type.mm_tensor_v2_kernel_name()
    } else if use_tensor {
        params.ggml_type.mm_tensor_kernel_name()
    } else {
        params.ggml_type.mm_kernel_name()
    };
    let pipeline = match registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        dense_q4_auto::Q4_MM_PIPELINE_INT_CONSTANTS,
    ) {
        Ok(pipeline) => pipeline,
        Err(_error) if use_q4_short_tile && forced_q4_route.is_none() => {
            // The candidate is optional performance metadata. Even after a
            // successful activation, an unexpected lookup failure must not
            // make a request less executable than the compatibility path.
            use_q4_short_tile = false;
            kernel_name = params.ggml_type.mm_tensor_v2_kernel_name();
            registry.get_pipeline_with_constants(
                kernel_name,
                device.metal_device(),
                &[],
                dense_q4_auto::Q4_MM_PIPELINE_INT_CONSTANTS,
            )?
        }
        Err(error) => return Err(error),
    };

    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();
    let blocks_per_row = params.k / qk;
    let nb01 = (blocks_per_row as u64) * (block_bytes as u64);
    let packed_row_bytes = (params.k as u64) * DType::F32.size_of() as u64;
    let (nb11, nb12) = input_strides
        .map(|strides| (strides.row_bytes, strides.batch_bytes))
        .unwrap_or((packed_row_bytes, packed_row_bytes * params.m as u64));

    let gpu_params = GgmlMatmulMmGpuParams {
        ne00: params.k as i32,
        ne02: batch as i32,
        nb01,
        nb02: nb01 * (params.n as u64),
        nb03: 0,
        ne12: batch as i32,
        _pad0: 0,
        nb10: DType::F32.size_of() as u64,
        nb11,
        nb12,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        _pad1: 0,
    };

    // V1 tile geometry: NR0=64 (output-N per tg), NR1=32 (M per tg).
    // V2 tile geometry: NRA=64 (M per tg), NRB=128 (N per tg).
    // Both use 4 simdgroups / 128 threads per threadgroup.
    const THREADS_PER_TG: u64 = 128;

    let (tg_x, tg_y, shmem_bytes) = if use_q4_short_tile {
        (
            u64::from(params.m).div_ceil(32),
            u64::from(params.n).div_ceil(64),
            4096u64,
        )
    } else if use_v2_large_tile {
        // V2 in peer-convention coordinates:
        //   gx covers N_peer with stride NRB=128 → N_peer is the SLOWER axis
        //     (hf2q-M = tokens = params.m).
        //   gy covers M_peer with stride NRA=64  → M_peer is the FASTER axis
        //     (hf2q-N = output features = params.n).
        // Only A goes through shmem: 64 × 32 halfs = 4096 B.  B is read
        // directly from device via the tensor view (no shmem staging).
        let nra: u64 = 64; // M_peer = hf2q-N
        let nrb: u64 = 128; // N_peer = hf2q-M
        (
            (params.m as u64 + nrb - 1) / nrb, // gx → N_peer = hf2q-M tiles
            (params.n as u64 + nra - 1) / nra, // gy → M_peer = hf2q-N tiles
            4096u64,
        )
    } else {
        // V1: gx = M tiles (NR1=32), gy = N tiles (NR0=64).  sa (A tile
        // 4096 B) + sb (B tile 4096 B as f32 → half cast) = 8192 B.
        let nr0: u64 = 64;
        let nr1: u64 = 32;
        (
            (params.m as u64 + nr1 - 1) / nr1,
            (params.n as u64 + nr0 - 1) / nr0,
            8192u64,
        )
    };

    let threadgroups = metal::MTLSize::new(tg_x, tg_y, batch as u64);
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, shmem_bytes)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ===========================================================================
// Wave P4.19 — bf16-input perm021 entry point for tensor-mm
//
// Used by the hf2q batched prefill's O-projection to consume the flash-
// attention output buffer (bf16 at [n_heads, seq_len, head_dim] physical
// layout) directly, eliminating the dedicated `permute_021_bf16_to_f32`
// dispatch that otherwise runs every layer.
//
// Semantics:
//   output[t, c] = sum_{i=0..K-1} weight[c, i] * src1_logical[t, i]
// where src1_logical[t, i] is obtained from the physical bf16 buffer at
//   src1_bf16[h * seq_len * head_dim + t * head_dim + f],  h = i / head_dim,
//                                                          f = i mod head_dim.
// K must equal n_heads * head_dim, and head_dim must be a multiple of NK=32
// (Gemma 4: head_dim ∈ {256 sliding, 512 global} — both satisfy).
//
// See /opt/mlx-native/src/shaders/quantized_matmul_mm_tensor.metal kernel
// `hf2q_mul_mm_tensor_perm021_impl` for the byte-exact equivalence proof.
// ===========================================================================

/// GPU-side params for the perm021 tensor-mm kernel — must match the
/// shader's `GgmlMatmulMmTensorPerm021Params`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatmulMmTensorPerm021GpuParams {
    ne00: i32, // K = n_heads * head_dim
    ne02: i32,
    nb01: u64, // bytes per weight row
    nb02: u64,
    nb03: u64,
    ne12: i32,
    _pad0: u32,
    nb10: u64, // = sizeof(bfloat) = 2
    nb11: u64, // unused (kept for struct symmetry)
    nb12: u64,
    nb13: u64,
    ne0: i32, // N = hidden_size
    ne1: i32, // M = seq_len
    r2: i16,
    r3: i16,
    // NO _pad between r3 and head_dim: Metal auto-aligns int32_t after
    // two int16_t at 2-byte boundary; the next int32_t naturally lands
    // at offset 84 (= 80 + 2 + 2).  Adding a u32 pad here would slide
    // head_dim to byte 88, mismatching the Metal struct layout and
    // causing the GPU to read head_dim = 0 (verified empirically
    // 2026-04-20: an earlier version with _pad1 produced first_token
    // 236772 instead of the expected 29294; removing the pad restored
    // byte-identity).
    head_dim: i32,
    seq_len: i32,
    // Trailing pad to bring struct size to a multiple of 8 (largest
    // member alignment = u64).  Rust's repr(C) auto-inserts this to 96
    // bytes anyway, but bytemuck::Pod rejects implicit trailing padding;
    // an explicit pad makes the derive compile and matches Metal's
    // struct size exactly.
    _pad_trailing: u32,
}

/// Params for the perm021 tensor-mm dispatch.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulPerm021Params {
    /// M — number of rows / tokens.
    pub m: u32,
    /// N — number of output cols (= hidden_size).
    pub n: u32,
    /// K — hidden_size (= n_heads * head_dim).  Must be divisible by
    /// the block's QK and by `head_dim`.
    pub k: u32,
    /// Head dimension.  Must be a multiple of NK=32.
    pub head_dim: u32,
    /// GGML quantization type of the weight.
    pub ggml_type: GgmlType,
}

/// Dispatch the bf16-input permuted-021 variant of the tensor-mm kernel.
///
/// `weight` is the quantized O-projection weight `[n, k]`.
/// `input_bf16` is the flash-attention output at physical layout
///   `[n_heads, seq_len, head_dim]` bf16.
/// `output` is the standard `[m, n]` f32 O-proj result.
///
/// # Errors
/// Returns `InvalidArgument` if:
/// - `ggml_type` is not Q4_0, Q5_0, Q8_0, or Q6_K
/// - `head_dim` is not a positive multiple of 32
/// - `k != n_heads * head_dim`  (we infer n_heads = k / head_dim)
/// - buffer sizes don't match the declared shapes
pub fn quantized_matmul_mm_tensor_perm021(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulPerm021Params,
) -> Result<()> {
    validate_signed_metal_dimensions(
        "quantized_matmul_mm_tensor_perm021",
        &[("M", params.m), ("N", params.n), ("K", params.k)],
    )?;
    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_mm_tensor_perm021: M, N, and K must be non-zero".into(),
        ));
    }
    let kernel_name = match params.ggml_type {
        GgmlType::Q4_0 => "kernel_mul_mm_q4_0_tensor_bf16_perm021",
        GgmlType::Q5_0 => "kernel_mul_mm_q5_0_tensor_bf16_perm021",
        // ADR-022 Phase 3 — Q8_0 perm021 instantiation added so the
        // Q8_0-quantized attention path (e.g. iter-21 Track B HB-encoded
        // K cache for Qwen 3.5 / 3.6) can use the same tensor-tile
        // permuted Q@K^T kernel as Q4_0 / Q6_K.
        GgmlType::Q8_0 => "kernel_mul_mm_q8_0_tensor_bf16_perm021",
        GgmlType::Q6_K => "kernel_mul_mm_q6_K_tensor_bf16_perm021",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "quantized_matmul_mm_tensor_perm021: unsupported ggml_type {:?} \
                 (only Q4_0 / Q5_0 / Q8_0 / Q6_K are instantiated)",
                other
            )));
        }
    };

    if params.head_dim == 0 || params.head_dim % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: head_dim {} must be a positive \
             multiple of 32 (NK tile width)",
            params.head_dim
        )));
    }
    if params.k % params.head_dim != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: k ({}) must be divisible by \
             head_dim ({})",
            params.k, params.head_dim
        )));
    }
    let qk = params.ggml_type.block_values();
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: K ({}) must be divisible by block QK ({})",
            params.k, qk,
        )));
    }
    if input_bf16.dtype() != DType::BF16
        || weight.dtype() != DType::U8
        || output.dtype() != DType::F32
    {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021 requires BF16 input, native U8 GGUF blocks, and F32 output; got {:?}/{:?}/{:?}",
            input_bf16.dtype(),
            weight.dtype(),
            output.dtype(),
        )));
    }

    // Input-buffer size check: n_heads * seq_len * head_dim * sizeof(bfloat).
    let n_heads = params.k / params.head_dim;
    let expected_input_bytes = checked_byte_extent(
        "perm021 input",
        &[
            n_heads as usize,
            params.m as usize,
            params.head_dim as usize,
            DType::BF16.size_of(),
        ],
    )?;
    if input_bf16.data_byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: input_bf16 buffer too small \
             (have {}, need {})",
            input_bf16.data_byte_len(),
            expected_input_bytes
        )));
    }

    let blocks_per_row = params.k / qk;
    let block_bytes = params.ggml_type.block_bytes();
    let expected_weight_bytes = (params.n as usize)
        .checked_mul(blocks_per_row as usize)
        .and_then(|blocks| blocks.checked_mul(block_bytes as usize))
        .ok_or_else(|| MlxError::InvalidArgument("perm021 weight bytes overflow".into()))?;
    if weight.data_byte_len() < expected_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: weight buffer too small (have {}, need {})",
            weight.data_byte_len(),
            expected_weight_bytes,
        )));
    }
    let expected_output_bytes = (params.m as usize)
        .checked_mul(params.n as usize)
        .and_then(|elements| elements.checked_mul(DType::F32.size_of()))
        .ok_or_else(|| MlxError::InvalidArgument("perm021 output bytes overflow".into()))?;
    if output.data_byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021: output buffer too small (have {}, need {})",
            output.data_byte_len(),
            expected_output_bytes,
        )));
    }

    let pipeline = registry.get_pipeline_with_constants(
        kernel_name,
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let nb01 = (blocks_per_row as u64) * (block_bytes as u64);

    let gpu_params = GgmlMatmulMmTensorPerm021GpuParams {
        ne00: params.k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * (params.n as u64),
        nb03: 0,
        ne12: 1,
        _pad0: 0,
        nb10: 2, // sizeof(bfloat)
        nb11: 0, // unused; B-stage computes addresses directly
        nb12: 0,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        head_dim: params.head_dim as i32,
        seq_len: params.m as i32,
        _pad_trailing: 0,
    };

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_BYTES: u64 = 8192;

    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input_bf16)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

/// ADR-029 iter-36 H28-D — F16-shadow variant of the perm021 tensor-mm.
///
/// Same contract as `quantized_matmul_mm_tensor_perm021`, but reads weights
/// from a caller-supplied F16 shadow buffer instead of dequantizing the
/// quantized weight in the kernel.  Mirrors the H29-speed pattern
/// applied to the perm021 layout — used for the O-projection prefill matmul
/// when `MlxQWeight.f16_shadow` is populated.
///
/// # Arguments
///
/// * `input_bf16` — bf16 input at physical layout `[n_heads, seq_len, head_dim]`
///   (same as the quantized perm021 variant; produced by flash-attention).
/// * `weight_f16` — F16 weight buffer at row-major `[n, k]`, `nb01 = 2*k` bytes
///   per row.  Caller is responsible for ensuring the shadow was populated.
/// * `output` — f32 `[m, n]` O-proj result.
/// * `params` — Same dimensions as `quantized_matmul_mm_tensor_perm021`; the
///   `ggml_type` field is ignored on this path (F16 has no GGML type).
///
/// # Errors
/// Same as `quantized_matmul_mm_tensor_perm021` minus the per-type kernel
/// resolution (this fn uses a single `kernel_mul_mm_f16_tensor_bf16_perm021`).
pub fn quantized_matmul_mm_tensor_perm021_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input_bf16: &MlxBuffer,
    weight_f16: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulPerm021Params,
) -> Result<()> {
    if params.head_dim == 0 || params.head_dim % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021_f16: head_dim {} must be a positive \
             multiple of 32 (NK tile width)",
            params.head_dim
        )));
    }
    if params.k % params.head_dim != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021_f16: k ({}) must be divisible by \
             head_dim ({})",
            params.k, params.head_dim
        )));
    }

    let n_heads = params.k / params.head_dim;
    let expected_input_bytes =
        (n_heads as usize) * (params.m as usize) * (params.head_dim as usize) * 2;
    if input_bf16.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021_f16: input_bf16 buffer too small \
             (have {}, need {})",
            input_bf16.byte_len(),
            expected_input_bytes
        )));
    }
    let expected_weight_bytes = (params.n as usize) * (params.k as usize) * 2;
    if weight_f16.byte_len() < expected_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_mm_tensor_perm021_f16: weight_f16 buffer too small \
             (have {}, need {} bytes for [n={}, k={}] half)",
            weight_f16.byte_len(),
            expected_weight_bytes,
            params.n,
            params.k
        )));
    }

    let pipeline = registry.get_pipeline_with_constants(
        "kernel_mul_mm_f16_tensor_bf16_perm021",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    // nb01 = bytes per F16 weight row = k * sizeof(half)
    let nb01: u64 = (params.k as u64) * 2;

    let gpu_params = GgmlMatmulMmTensorPerm021GpuParams {
        ne00: params.k as i32,
        ne02: 1,
        nb01,
        nb02: nb01 * (params.n as u64),
        nb03: 0,
        ne12: 1,
        _pad0: 0,
        nb10: 2, // sizeof(bfloat)
        nb11: 0,
        nb12: 0,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        head_dim: params.head_dim as i32,
        seq_len: params.m as i32,
        _pad_trailing: 0,
    };

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_BYTES: u64 = 8192;

    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(weight_f16)),
            (2, KernelArg::Buffer(input_bf16)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}
