//! Machine-readable execution contract for MLX-compatible packed affine weights.
//!
//! This module describes what the public packed-affine dispatch paths can
//! execute. It intentionally does not make a throughput claim: callers still
//! have to benchmark the exact artifact, shape, and workload on the target
//! Apple Silicon device.

use serde::{Deserialize, Serialize};

/// Schema version for serialized [`PackedAffineCapability`] receipts.
pub const PACKED_AFFINE_CAPABILITY_SCHEMA_VERSION: u32 = 1;

/// Activation/output dtype at the packed-affine dispatch boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AffineIoDType {
    F32,
    Bf16,
}

/// The graph operation requesting packed-affine execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AffineOperation {
    Dense,
    /// One expert selected by byte offsets into a packed 3-D tensor.
    ExpertOffset,
    /// Per-token expert selection through an explicit expert-ID buffer.
    ExpertRoutedId,
    Embedding,
}

/// Workload regime for which the caller needs a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AffineExecutionRegime {
    DecodeQmv,
    PromptQmm,
    WidthN,
    EmbeddingGather,
}

/// Concrete kernel route selected by the current runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PackedAffineKernelRoute {
    /// Correctness-first dense kernel, with F32 input/output.
    DenseScalarF32,
    /// BF16 dense input/output routed through F32 casts and the scalar kernel.
    DenseScalarViaF32,
    /// Row-wise SIMD route with F32 input/output.
    DenseRowWiseSimdF32,
    /// Row-wise SIMD route with BF16 input/output.
    DenseRowWiseSimdBf16,
    /// BF16 row-wise SIMD route with byte offsets into an expert tensor.
    ExpertOffsetRowWiseSimdBf16,
    /// F32 scalar route with per-token expert IDs.
    ExpertRoutedIdScalarF32,
    /// F32 embedding gather over packed 4-bit or 6-bit rows.
    EmbeddingGatherF32,
}

/// Exact tensor and workload request presented to the runtime capability API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedAffineRequest {
    pub operation: AffineOperation,
    pub regime: AffineExecutionRegime,
    pub io_dtype: AffineIoDType,
    pub bits: u32,
    pub group_size: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Current packed-affine kernels require an explicit per-group bias buffer.
    pub has_biases: bool,
}

/// Serializable capability decision for one exact request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PackedAffineCapability {
    pub schema_version: u32,
    pub executable: bool,
    pub route: Option<PackedAffineKernelRoute>,
    /// True only when the route is specialized for the requested regime.
    /// A row-wise QMV kernel can execute multiple rows, but that does not make
    /// it a prompt-QMM or width-N optimized kernel.
    pub specialized_for_regime: bool,
    /// Stable rejection category. `None` means the request is executable.
    pub rejection_code: Option<PackedAffineRejectionCode>,
    pub diagnostic: String,
}

/// Machine-readable reason why a packed-affine request is not executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PackedAffineRejectionCode {
    InvalidDimensions,
    InvalidGroupSize,
    UnsupportedBits,
    MissingBiases,
    UnsupportedIoDtype,
    UnsupportedRegime,
    UnsupportedLayout,
}

impl PackedAffineCapability {
    fn supported(
        route: PackedAffineKernelRoute,
        specialized_for_regime: bool,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: PACKED_AFFINE_CAPABILITY_SCHEMA_VERSION,
            executable: true,
            route: Some(route),
            specialized_for_regime,
            rejection_code: None,
            diagnostic: diagnostic.into(),
        }
    }

    fn unsupported(
        rejection_code: PackedAffineRejectionCode,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: PACKED_AFFINE_CAPABILITY_SCHEMA_VERSION,
            executable: false,
            route: None,
            specialized_for_regime: false,
            rejection_code: Some(rejection_code),
            diagnostic: diagnostic.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimdContract {
    block_size: u32,
    values_per_thread: u32,
}

fn simd_contract(bits: u32, io_dtype: AffineIoDType) -> Option<SimdContract> {
    match (bits, io_dtype) {
        (4, AffineIoDType::F32) => Some(SimdContract {
            block_size: 256,
            values_per_thread: 8,
        }),
        (8, AffineIoDType::F32) => Some(SimdContract {
            block_size: 256,
            values_per_thread: 8,
        }),
        (4, AffineIoDType::Bf16) => Some(SimdContract {
            block_size: 512,
            values_per_thread: 16,
        }),
        (8, AffineIoDType::Bf16) => Some(SimdContract {
            block_size: 256,
            values_per_thread: 8,
        }),
        _ => None,
    }
}

fn simd_layout_supported(request: &PackedAffineRequest, contract: SimdContract) -> bool {
    request.n % 8 == 0
        && request.k % contract.block_size == 0
        && request.group_size.is_power_of_two()
        && request.group_size >= contract.values_per_thread
        && request.group_size <= contract.block_size
        && request.group_size % contract.values_per_thread == 0
        && contract.block_size % request.group_size == 0
}

fn row_wise_specialized(request: &PackedAffineRequest) -> bool {
    request.regime == AffineExecutionRegime::DecodeQmv && request.m == 1
}

pub(crate) const fn packed_row_quantum(bits: u32) -> Option<u32> {
    match bits {
        4 => Some(8),
        6 | 8 => Some(4),
        _ => None,
    }
}

fn matmul_regime_supported(request: &PackedAffineRequest) -> bool {
    request.regime != AffineExecutionRegime::EmbeddingGather
}

/// Return the exact packed-affine execution capability for `request`.
///
/// Packed storage uses the MLX-compatible bit layout with BF16 per-group scales
/// and biases: U32 words for 4/8-bit data and 3-byte triplets for 6-bit data.
/// For embeddings, `m` is the token count, `n` is the vocabulary size, and `k`
/// is the embedding width. For matmuls they retain their conventional M/N/K
/// meanings. Buffer sizes remain dispatch-time checks; expert-ID ranges are
/// defended in the routed-ID kernel.
pub fn packed_affine_capability(request: PackedAffineRequest) -> PackedAffineCapability {
    if request.m == 0 || request.n == 0 || request.k == 0 {
        return PackedAffineCapability::unsupported(
            PackedAffineRejectionCode::InvalidDimensions,
            "M, N, and K must all be non-zero",
        );
    }
    if request.group_size == 0 {
        return PackedAffineCapability::unsupported(
            PackedAffineRejectionCode::InvalidGroupSize,
            "group_size must be non-zero",
        );
    }
    if !matches!(request.bits, 4 | 6 | 8) {
        return PackedAffineCapability::unsupported(
            PackedAffineRejectionCode::UnsupportedBits,
            format!(
                "packed affine storage supports bits 4, 6, and 8; got {}",
                request.bits
            ),
        );
    }
    if !request.has_biases {
        return PackedAffineCapability::unsupported(
            PackedAffineRejectionCode::MissingBiases,
            "current packed affine dispatch requires an explicit bias buffer",
        );
    }

    if request.operation == AffineOperation::Embedding {
        if request.regime != AffineExecutionRegime::EmbeddingGather {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedRegime,
                "embedding execution requires the embedding-gather regime",
            );
        }
        if request.io_dtype != AffineIoDType::F32 {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedIoDtype,
                "packed affine embedding gather writes F32 output",
            );
        }
        if !matches!(request.bits, 4 | 6) {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedBits,
                "packed affine embedding gather supports bits 4 and 6",
            );
        }
        if request.k % request.group_size != 0 {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedLayout,
                "embedding width must be divisible by group_size",
            );
        }
        let packing_quantum = packed_row_quantum(request.bits)
            .expect("embedding bit widths were validated above");
        if request.k % packing_quantum != 0 {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedLayout,
                format!(
                    "{}-bit embedding width must be divisible by its {}-value packing quantum",
                    request.bits, packing_quantum
                ),
            );
        }
        return PackedAffineCapability::supported(
            PackedAffineKernelRoute::EmbeddingGatherF32,
            true,
            "dedicated packed affine embedding-gather route",
        );
    }

    if !matmul_regime_supported(&request) {
        return PackedAffineCapability::unsupported(
            PackedAffineRejectionCode::UnsupportedRegime,
            "matmul execution does not use the embedding-gather regime",
        );
    }

    if request.operation == AffineOperation::ExpertRoutedId {
        if request.io_dtype != AffineIoDType::F32 {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedIoDtype,
                "expert-ID packed affine execution requires F32 I/O",
            );
        }
        return PackedAffineCapability::supported(
            PackedAffineKernelRoute::ExpertRoutedIdScalarF32,
            false,
            "correctness-first F32 route with per-token expert IDs",
        );
    }

    let simd = simd_contract(request.bits, request.io_dtype)
        .filter(|contract| simd_layout_supported(&request, *contract));

    if request.operation == AffineOperation::ExpertOffset {
        if request.io_dtype != AffineIoDType::Bf16 {
            return PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedIoDtype,
                "expert-offset packed affine execution requires BF16 I/O",
            );
        }
        return match simd {
            Some(_) => PackedAffineCapability::supported(
                PackedAffineKernelRoute::ExpertOffsetRowWiseSimdBf16,
                row_wise_specialized(&request),
                if row_wise_specialized(&request) {
                    "dedicated expert-offset BF16 QMV route"
                } else {
                    "expert-offset route is row-wise SIMD, not a specialized QMM/width-N kernel"
                },
            ),
            None => PackedAffineCapability::unsupported(
                PackedAffineRejectionCode::UnsupportedLayout,
                "expert-offset route requires bits 4/8, N divisible by 8, and a supported SIMD K/group layout",
            ),
        };
    }

    match (request.io_dtype, simd) {
        (AffineIoDType::F32, Some(_)) => PackedAffineCapability::supported(
            PackedAffineKernelRoute::DenseRowWiseSimdF32,
            row_wise_specialized(&request),
            if row_wise_specialized(&request) {
                "dedicated F32 QMV route"
            } else {
                "dense route is row-wise SIMD, not a specialized QMM/width-N kernel"
            },
        ),
        (AffineIoDType::Bf16, Some(_)) => PackedAffineCapability::supported(
            PackedAffineKernelRoute::DenseRowWiseSimdBf16,
            row_wise_specialized(&request),
            if row_wise_specialized(&request) {
                "dedicated BF16 QMV route"
            } else {
                "dense route is row-wise SIMD, not a specialized QMM/width-N kernel"
            },
        ),
        (AffineIoDType::F32, None) => PackedAffineCapability::supported(
            PackedAffineKernelRoute::DenseScalarF32,
            false,
            "correctness fallback; no matching packed-affine SIMD route",
        ),
        (AffineIoDType::Bf16, None) => PackedAffineCapability::supported(
            PackedAffineKernelRoute::DenseScalarViaF32,
            false,
            "correctness fallback with BF16/F32 casts; no matching packed-affine SIMD route",
        ),
    }
}

#[cfg(test)]
mod tests;
