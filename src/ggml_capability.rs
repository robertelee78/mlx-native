//! Machine-readable execution contract for GGUF block-quantized weights.
//!
//! Capability is defined for an exact public entry point, shape, workload
//! regime, and explicit routing policy. This avoids a false promise that a
//! `(qtype, M, N, K)` tuple alone determines the program that Metal executes:
//! batched layouts, fused gate/up, expert schedule reuse, and diagnostic
//! routing overrides all change the real operation.
//!
//! The result is a structural capability decision, not a performance model.
//! Device-selected tensor-API routes still require an exact-device runtime
//! trace and benchmark receipt before a caller may rank encodings.

use serde::{Deserialize, Serialize};

use crate::error::{MlxError, Result};
use crate::ops::quantized_matmul_ggml::{dense_mn_dispatch_count, GgmlType, MM_ROUTING_THRESHOLD};
use crate::ops::quantized_matmul_id_ggml::MM_ID_ROUTING_THRESHOLD;

/// Schema version for serialized [`GgmlCapability`] receipts.
pub const GGML_CAPABILITY_SCHEMA_VERSION: u32 = 1;

/// Workload class for which the caller requires execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgmlWorkloadClass {
    DecodeSingle,
    Prompt,
    ContinuousWidth,
    Embedding,
}

/// Physical input layout for the public batched-MM entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GgmlBatchedInputLayout {
    Contiguous,
    Strided { row_bytes: u64, batch_bytes: u64 },
}

/// Physical input layout for the pooled expert entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgmlExpertInputLayout {
    SharedPerToken,
    Slotted,
}

/// Shape and layout contract shared by expert-routed entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlExpertShape {
    pub n_tokens: u32,
    pub n: u32,
    pub k: u32,
    pub top_k: u32,
    pub n_experts: u32,
    pub expert_stride_bytes: u64,
    /// Required by the current MM_ID schedule: each token may contribute at
    /// most one row to any expert's scratch list.
    pub ids_are_distinct_per_token: bool,
    /// Caller assertion required by every expert kernel before dispatch.
    pub ids_within_expert_range: bool,
}

/// Exact public invocation that consumes the weight.
///
/// Dimensions live inside the tagged variant so callers cannot combine an
/// embedding entry point with dense-only fields, omit an expert stride, or
/// accidentally price a fused pair as two independent matmuls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "entrypoint", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum GgmlInvocation {
    /// `quantized_matmul_ggml`; runtime routing uses [`GgmlRoutingPolicy`].
    DenseAuto { m: u32, n: u32, k: u32 },
    /// `quantized_matmul_ggml_batched_mv` with independent weight matrices.
    DenseBatchedMv { batch: u32, m: u32, n: u32, k: u32 },
    /// Batched MM, either tightly packed or explicitly strided on input.
    DenseBatchedMm {
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        input_layout: GgmlBatchedInputLayout,
    },
    /// BF16-input permuted-021 output projection.
    DensePerm021Bf16 {
        m: u32,
        n: u32,
        k: u32,
        head_dim: u32,
    },
    /// Two same-codec weights consumed by one fused gate+up+SiLU dispatch.
    DenseGateUpSiluPair { m: u32, n: u32, k: u32 },
    /// Auto-allocating expert route.
    ExpertAutoAllocated { shape: GgmlExpertShape },
    /// Byte-identity entry point that always chooses expert matvec.
    ExpertForceMv { shape: GgmlExpertShape },
    /// Caller-owned expert MM scratch, with shared or slotted activations.
    ExpertPooled {
        shape: GgmlExpertShape,
        input_layout: GgmlExpertInputLayout,
    },
    /// Two projections sharing one prepared expert routing schedule.
    ExpertPooledPair { shape: GgmlExpertShape },
    /// Q4_0-only fused SwiGLU plus expert-down matvec.
    ExpertSwiGluDownQ4 { shape: GgmlExpertShape },
    /// Direct block-quantized embedding gather.
    EmbeddingGather {
        n_tokens: u32,
        vocab_size: u32,
        embed_dim: u32,
    },
}

impl GgmlInvocation {
    fn dimensions(self) -> (u32, u32, u32) {
        match self {
            Self::DenseAuto { m, n, k }
            | Self::DenseBatchedMv { m, n, k, .. }
            | Self::DenseBatchedMm { m, n, k, .. }
            | Self::DensePerm021Bf16 { m, n, k, .. }
            | Self::DenseGateUpSiluPair { m, n, k } => (m, n, k),
            Self::ExpertAutoAllocated { shape }
            | Self::ExpertForceMv { shape }
            | Self::ExpertPooled { shape, .. }
            | Self::ExpertPooledPair { shape }
            | Self::ExpertSwiGluDownQ4 { shape } => (shape.n_tokens, shape.n, shape.k),
            Self::EmbeddingGather {
                n_tokens,
                vocab_size,
                embed_dim,
            } => (n_tokens, vocab_size, embed_dim),
        }
    }
}

/// Explicit routing knobs that otherwise come from process-global env state.
///
/// A receipt must serialize the effective values used by the measured
/// process. [`Default`] matches the production defaults at this crate
/// revision; diagnostic overrides must be represented by a different value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlRoutingPolicy {
    pub dense_decode_mvn: bool,
    pub dense_decode_mv_ext: bool,
    pub dense_q6k_mv_nr2: bool,
    pub dense_q8_0_mv_nr2: bool,
    pub dense_tensor_mm: GgmlTensorMmPreference,
    pub allow_dense_large_tile_mm: bool,
    pub expert_mm_threshold: u32,
    pub expert_q6k_mv_nr2: bool,
    pub expert_q8_0_mv_nr2: bool,
    pub expert_tensor_mm: GgmlTensorMmPreference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlTensorMmPreference {
    AutoProbe,
    ForceSimd,
}

impl Default for GgmlRoutingPolicy {
    fn default() -> Self {
        Self {
            dense_decode_mvn: true,
            dense_decode_mv_ext: false,
            dense_q6k_mv_nr2: true,
            dense_q8_0_mv_nr2: true,
            dense_tensor_mm: GgmlTensorMmPreference::AutoProbe,
            allow_dense_large_tile_mm: true,
            expert_mm_threshold: MM_ID_ROUTING_THRESHOLD,
            expert_q6k_mv_nr2: true,
            expert_q8_0_mv_nr2: false,
            expert_tensor_mm: GgmlTensorMmPreference::AutoProbe,
        }
    }
}

/// Exact tensor, graph operation, and workload request.
///
/// `M`, `N`, and `K` follow the public matmul contracts. For expert
/// operations `M` is `n_tokens`; top-k expansion is carried separately in the
/// operation. For embeddings `M` is token count, `N` vocabulary rows, and `K`
/// embedding width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlCapabilityRequest {
    pub schema_version: u32,
    pub invocation: GgmlInvocation,
    pub ggml_type: GgmlType,
    pub workload: GgmlWorkloadClass,
    pub routing: GgmlRoutingPolicy,
}

/// Structural kernel family selected by the exact entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgmlKernelRoute {
    DenseF32Mv,
    DenseF32Mm,
    DenseF16Mv,
    DenseF16Mm,
    DenseBF16Mv,
    DenseBF16Mm,
    DenseMv,
    DenseMvNr2,
    DenseQ4kWidthMn,
    DenseQ5kWidthMn,
    DenseQ6kWidthMn,
    DenseWidthMvExt,
    DenseMmSimdgroup,
    DenseMmDeviceSelected,
    DenseBatchedMv,
    DenseBatchedMvNr2,
    DenseBatchedMmSimdgroup,
    DenseBatchedMmDeviceSelected,
    DensePerm021TensorMm,
    FusedGateUpSilu,
    ExpertMv,
    ExpertMvNr2,
    ExpertMmSimdgroup,
    ExpertMmDeviceSelected,
    ExpertPooledMmSimdgroup,
    ExpertPooledMmDeviceSelected,
    ExpertPooledPairMmSimdgroup,
    ExpertPooledPairMmDeviceSelected,
    ExpertPooledSlottedMmSimdgroup,
    ExpertPooledSlottedMmDeviceSelected,
    ExpertSwiGluDownQ4,
    EmbeddingF32,
    EmbeddingF16,
    EmbeddingBF16,
    EmbeddingQ2K,
    EmbeddingQ4_0,
    EmbeddingQ5_0,
    EmbeddingQ4K,
    EmbeddingQ5K,
    EmbeddingQ6K,
    EmbeddingQ8_0,
}

/// Stable reason why an exact GGUF request is not executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgmlRejectionCode {
    InvalidDimensions,
    InvalidOperationContract,
    UnsupportedType,
    UnsupportedLayout,
    UnsupportedRegime,
    ArithmeticOverflow,
}

/// Transient storage required by the logical route. `caller_owned` says
/// whether the public entry point receives it from the caller or allocates it
/// internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GgmlScratchRequirement {
    None,
    ExpertMm {
        htpe_bytes: u64,
        hids_bytes: u64,
        caller_owned: bool,
        schedule_reused: bool,
    },
}

/// Serializable structural decision for one exact request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlCapability {
    pub schema_version: u32,
    pub request: GgmlCapabilityRequest,
    pub executable: bool,
    pub route: Option<GgmlKernelRoute>,
    /// True when the selected route is designed for the requested regime.
    pub specialized_for_workload: bool,
    /// True when execution is correct but falls back from that regime's
    /// specialized path.
    pub correctness_fallback: bool,
    /// Tensor-API routes probe the exact device and metallib at runtime. A
    /// cost receipt must record the resolved kernel, not only this family.
    pub requires_device_probe: bool,
    pub block_values: u32,
    pub block_bytes: u32,
    pub weight_buffer_count: u32,
    /// Minimum data bytes required in each weight buffer. Expert strides may
    /// make this larger than tightly packed logical payload bytes.
    pub minimum_weight_buffer_bytes: u64,
    pub minimum_total_weight_bytes: u64,
    pub scratch: GgmlScratchRequirement,
    /// Dispatches internal to this invocation. Graph-level copies, casts,
    /// automatic dependency barriers, and caller fusion are outside this
    /// count and must be measured in the whole execution trace.
    pub dispatches: u32,
    /// Explicit barriers internal to this invocation. Encoder-inferred or
    /// surrounding graph barriers are not included.
    pub barriers: u32,
    pub rejection_code: Option<GgmlRejectionCode>,
    pub diagnostic: String,
}

impl GgmlCapability {
    fn supported(
        request: &GgmlCapabilityRequest,
        route: GgmlKernelRoute,
        specialized_for_workload: bool,
        correctness_fallback: bool,
        requires_device_probe: bool,
        weight_buffer_count: u32,
        minimum_weight_buffer_bytes: u64,
        scratch: GgmlScratchRequirement,
        dispatches: u32,
        barriers: u32,
        diagnostic: impl Into<String>,
    ) -> Self {
        let Some(minimum_total_weight_bytes) =
            minimum_weight_buffer_bytes.checked_mul(u64::from(weight_buffer_count))
        else {
            return Self::unsupported(
                request,
                GgmlRejectionCode::ArithmeticOverflow,
                "total GGUF weight byte count overflows u64",
            );
        };
        Self {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            request: *request,
            executable: true,
            route: Some(route),
            specialized_for_workload,
            correctness_fallback,
            requires_device_probe,
            block_values: request.ggml_type.block_values(),
            block_bytes: request.ggml_type.block_bytes(),
            weight_buffer_count,
            minimum_weight_buffer_bytes,
            minimum_total_weight_bytes,
            scratch,
            dispatches,
            barriers,
            rejection_code: None,
            diagnostic: diagnostic.into(),
        }
    }

    fn unsupported(
        request: &GgmlCapabilityRequest,
        code: GgmlRejectionCode,
        diagnostic: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            request: *request,
            executable: false,
            route: None,
            specialized_for_workload: false,
            correctness_fallback: false,
            requires_device_probe: false,
            block_values: request.ggml_type.block_values(),
            block_bytes: request.ggml_type.block_bytes(),
            weight_buffer_count: 0,
            minimum_weight_buffer_bytes: 0,
            minimum_total_weight_bytes: 0,
            scratch: GgmlScratchRequirement::None,
            dispatches: 0,
            barriers: 0,
            rejection_code: Some(code),
            diagnostic: diagnostic.into(),
        }
    }
}

fn quantized_matmul_type(ggml_type: GgmlType) -> bool {
    matches!(
        ggml_type,
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
            | GgmlType::IQ4_XS
    )
}

/// Checked packed bytes for one GGUF weight row containing `K` values.
pub fn ggml_packed_row_bytes(ggml_type: GgmlType, k: u32) -> Result<u64> {
    if !quantized_matmul_type(ggml_type) {
        return Err(MlxError::InvalidArgument(format!(
            "{ggml_type:?} is not a block-quantized matmul type"
        )));
    }
    if k == 0 || k % ggml_type.block_values() != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "K ({k}) must be non-zero and divisible by {:?} block quantum {}",
            ggml_type,
            ggml_type.block_values()
        )));
    }
    u64::from(k / ggml_type.block_values())
        .checked_mul(u64::from(ggml_type.block_bytes()))
        .ok_or_else(|| MlxError::InvalidArgument("packed GGUF row bytes overflow u64".into()))
}

/// Checked packed bytes for one `[N, K]` GGUF matrix.
pub fn ggml_matrix_bytes(ggml_type: GgmlType, n: u32, k: u32) -> Result<u64> {
    if n == 0 {
        return Err(MlxError::InvalidArgument("N must be non-zero".into()));
    }
    ggml_packed_row_bytes(ggml_type, k)?
        .checked_mul(u64::from(n))
        .ok_or_else(|| MlxError::InvalidArgument("packed GGUF matrix bytes overflow u64".into()))
}

/// Checked packed bytes for `batch` independent `[N, K]` matrices.
pub fn ggml_batched_matrix_bytes(ggml_type: GgmlType, batch: u32, n: u32, k: u32) -> Result<u64> {
    if batch == 0 {
        return Err(MlxError::InvalidArgument("batch must be non-zero".into()));
    }
    ggml_matrix_bytes(ggml_type, n, k)?
        .checked_mul(u64::from(batch))
        .ok_or_else(|| MlxError::InvalidArgument("batched GGUF matrix bytes overflow u64".into()))
}

/// Checked minimum buffer bytes for a padded expert stack.
pub fn ggml_expert_bytes(
    ggml_type: GgmlType,
    n_experts: u32,
    n: u32,
    k: u32,
    expert_stride_bytes: u64,
) -> Result<u64> {
    if n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "n_experts must be non-zero".into(),
        ));
    }
    let matrix_bytes = ggml_matrix_bytes(ggml_type, n, k)?;
    if expert_stride_bytes < matrix_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "expert stride ({expert_stride_bytes}) is smaller than one matrix ({matrix_bytes})"
        )));
    }
    u64::from(n_experts - 1)
        .checked_mul(expert_stride_bytes)
        .and_then(|offset| offset.checked_add(matrix_bytes))
        .ok_or_else(|| MlxError::InvalidArgument("expert GGUF bytes overflow u64".into()))
}

fn workload_shape_valid(request: &GgmlCapabilityRequest) -> bool {
    let (m, _, _) = request.invocation.dimensions();
    match request.workload {
        GgmlWorkloadClass::DecodeSingle => m == 1,
        GgmlWorkloadClass::ContinuousWidth => (2..=MM_ROUTING_THRESHOLD).contains(&m),
        GgmlWorkloadClass::Prompt => {
            !matches!(request.invocation, GgmlInvocation::EmbeddingGather { .. })
        }
        GgmlWorkloadClass::Embedding => {
            matches!(request.invocation, GgmlInvocation::EmbeddingGather { .. })
        }
    }
}

fn packed_matrix_bytes(request: &GgmlCapabilityRequest) -> Option<u64> {
    let (_, n, k) = request.invocation.dimensions();
    if matches!(request.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return u64::from(n)
            .checked_mul(u64::from(k))
            .and_then(|elements| {
                elements.checked_mul(u64::from(request.ggml_type.block_bytes()))
            });
    }
    ggml_matrix_bytes(request.ggml_type, n, k).ok()
}

fn validate_common(request: &GgmlCapabilityRequest) -> Option<GgmlCapability> {
    let (m, n, k) = request.invocation.dimensions();
    if request.schema_version != GGML_CAPABILITY_SCHEMA_VERSION {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "request schema does not match the capability schema",
        ));
    }
    if m == 0 || n == 0 || k == 0 {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidDimensions,
            "M, N, and K must all be non-zero",
        ));
    }
    let native_scalar = matches!(
        request.ggml_type,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    );
    let native_scalar_operation = matches!(
        request.invocation,
        GgmlInvocation::DenseAuto { .. } | GgmlInvocation::EmbeddingGather { .. }
    );
    if !quantized_matmul_type(request.ggml_type) && !(native_scalar && native_scalar_operation) {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedType,
            format!(
                "{:?} has no native route for this GGUF operation",
                request.ggml_type
            ),
        ));
    }
    if k % request.ggml_type.block_values() != 0 {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedLayout,
            format!(
                "K ({}) must be divisible by the {:?} block quantum ({})",
                k,
                request.ggml_type,
                request.ggml_type.block_values()
            ),
        ));
    }
    if !workload_shape_valid(request) {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "workload class does not match the exact M shape or entry point",
        ));
    }
    if request.routing.expert_mm_threshold == 0 {
        return Some(GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "expert MM threshold must be non-zero",
        ));
    }
    None
}

fn dense_mv_route(request: &GgmlCapabilityRequest, batched: bool) -> GgmlKernelRoute {
    if (request.ggml_type == GgmlType::Q6_K && request.routing.dense_q6k_mv_nr2)
        || (request.ggml_type == GgmlType::Q8_0 && request.routing.dense_q8_0_mv_nr2)
    {
        if batched {
            GgmlKernelRoute::DenseBatchedMvNr2
        } else {
            GgmlKernelRoute::DenseMvNr2
        }
    } else if batched {
        GgmlKernelRoute::DenseBatchedMv
    } else {
        GgmlKernelRoute::DenseMv
    }
}

/// Whether the default device-selected MM route may use the tensor pipeline.
///
/// Q5_0 stays on the native simdgroup kernels: the M5 tensor path exceeded
/// the independent F32 parity bound at dense and expert prompt widths, while
/// the simdgroup path passed the same bytes and shapes. An explicit future
/// qualification can reopen this predicate without changing the artifact.
pub(crate) fn tensor_mm_auto_selected(
    ggml_type: GgmlType,
    preference: GgmlTensorMmPreference,
) -> bool {
    preference == GgmlTensorMmPreference::AutoProbe && ggml_type != GgmlType::Q5_0
}

fn dense_mm_route(request: &GgmlCapabilityRequest, batched: bool) -> (GgmlKernelRoute, bool) {
    if tensor_mm_auto_selected(request.ggml_type, request.routing.dense_tensor_mm) {
        (
            if batched {
                GgmlKernelRoute::DenseBatchedMmDeviceSelected
            } else {
                GgmlKernelRoute::DenseMmDeviceSelected
            },
            true,
        )
    } else {
        (
            if batched {
                GgmlKernelRoute::DenseBatchedMmSimdgroup
            } else {
                GgmlKernelRoute::DenseMmSimdgroup
            },
            false,
        )
    }
}

/// Device-independent branch selected by the canonical dense GGUF entry
/// point. The dispatcher and the capability API share this planner so a
/// receipt cannot describe a different routing predicate than production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseAutoPlan {
    Mv,
    Q4kWidthMn,
    Q5kWidthMn,
    Q6kWidthMn,
    WidthMvExt,
    Mm,
}

pub(crate) fn plan_dense_auto_route(
    ggml_type: GgmlType,
    m: u32,
    k: u32,
    routing: &GgmlRoutingPolicy,
) -> DenseAutoPlan {
    if routing.dense_decode_mvn
        && ggml_type == GgmlType::Q4_K
        && (2..=MM_ROUTING_THRESHOLD).contains(&m)
    {
        return DenseAutoPlan::Q4kWidthMn;
    }
    if routing.dense_decode_mvn
        && ggml_type == GgmlType::Q5_K
        && (2..=MM_ROUTING_THRESHOLD).contains(&m)
    {
        return DenseAutoPlan::Q5kWidthMn;
    }
    if routing.dense_decode_mvn
        && ggml_type == GgmlType::Q6_K
        && (2..=MM_ROUTING_THRESHOLD).contains(&m)
    {
        return DenseAutoPlan::Q6kWidthMn;
    }
    let mv_ext_width_supported = match ggml_type {
        // K-quants use mul_mv_ext only once four columns can amortize the
        // wider dequantization path.
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => (4..=MM_ROUTING_THRESHOLD).contains(&m),
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => (2..=MM_ROUTING_THRESHOLD).contains(&m),
        _ => false,
    };
    if routing.dense_decode_mv_ext && mv_ext_width_supported && k >= 32 {
        return DenseAutoPlan::WidthMvExt;
    }
    if m > MM_ROUTING_THRESHOLD && k >= 32 && ggml_type != GgmlType::IQ4_XS {
        DenseAutoPlan::Mm
    } else {
        DenseAutoPlan::Mv
    }
}

fn dense_auto(request: &GgmlCapabilityRequest, bytes: u64) -> GgmlCapability {
    let (m, _, k) = request.invocation.dimensions();
    if request.workload == GgmlWorkloadClass::Embedding {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedRegime,
            "dense auto matmul does not use embedding-gather",
        );
    }
    let scalar_route = match (request.ggml_type, m == 1) {
        (GgmlType::F32, true) => Some(GgmlKernelRoute::DenseF32Mv),
        (GgmlType::F32, false) => Some(GgmlKernelRoute::DenseF32Mm),
        (GgmlType::F16, true) => Some(GgmlKernelRoute::DenseF16Mv),
        (GgmlType::F16, false) => Some(GgmlKernelRoute::DenseF16Mm),
        (GgmlType::BF16, true) => Some(GgmlKernelRoute::DenseBF16Mv),
        (GgmlType::BF16, false) => Some(GgmlKernelRoute::DenseBF16Mm),
        _ => None,
    };
    if let Some(route) = scalar_route {
        return GgmlCapability::supported(
            request,
            route,
            true,
            false,
            false,
            1,
            bytes,
            GgmlScratchRequirement::None,
            1,
            0,
            "native scalar GGUF dense projection route",
        );
    }
    match plan_dense_auto_route(request.ggml_type, m, k, &request.routing) {
        DenseAutoPlan::Q4kWidthMn => GgmlCapability::supported(
            request,
            GgmlKernelRoute::DenseQ4kWidthMn,
            request.workload == GgmlWorkloadClass::ContinuousWidth,
            request.workload != GgmlWorkloadClass::ContinuousWidth,
            false,
            1,
            bytes,
            GgmlScratchRequirement::None,
            dense_mn_dispatch_count(m),
            0,
            "Q4_K byte-identical multi-column matvec route",
        ),
        DenseAutoPlan::Q5kWidthMn => GgmlCapability::supported(
            request,
            GgmlKernelRoute::DenseQ5kWidthMn,
            request.workload == GgmlWorkloadClass::ContinuousWidth,
            request.workload != GgmlWorkloadClass::ContinuousWidth,
            false,
            1,
            bytes,
            GgmlScratchRequirement::None,
            dense_mn_dispatch_count(m),
            0,
            "Q5_K byte-identical multi-column matvec route",
        ),
        DenseAutoPlan::Q6kWidthMn => GgmlCapability::supported(
            request,
            GgmlKernelRoute::DenseQ6kWidthMn,
            request.workload == GgmlWorkloadClass::ContinuousWidth,
            request.workload != GgmlWorkloadClass::ContinuousWidth,
            false,
            1,
            bytes,
            GgmlScratchRequirement::None,
            dense_mn_dispatch_count(m),
            0,
            "Q6_K byte-identical multi-column matvec route",
        ),
        DenseAutoPlan::WidthMvExt => GgmlCapability::supported(
            request,
            GgmlKernelRoute::DenseWidthMvExt,
            request.workload == GgmlWorkloadClass::ContinuousWidth,
            request.workload != GgmlWorkloadClass::ContinuousWidth,
            false,
            1,
            bytes,
            GgmlScratchRequirement::None,
            1,
            0,
            "opt-in multi-column mul_mv_ext route",
        ),
        DenseAutoPlan::Mm => {
            let (route, probe) = dense_mm_route(request, false);
            GgmlCapability::supported(
                request,
                route,
                request.workload == GgmlWorkloadClass::Prompt,
                request.workload != GgmlWorkloadClass::Prompt,
                probe,
                1,
                bytes,
                GgmlScratchRequirement::None,
                1,
                0,
                if request.routing.allow_dense_large_tile_mm {
                    "dense MM route; tensor-capable devices use a frozen exact-shape Q4 plan or the compatibility large-tile tensor kernel"
                } else {
                    "dense MM route; large-tile tensor kernel disabled by routing policy"
                },
            )
        }
        DenseAutoPlan::Mv => {
            let specialized = request.workload == GgmlWorkloadClass::DecodeSingle;
            GgmlCapability::supported(
                request,
                dense_mv_route(request, false),
                specialized,
                !specialized,
                false,
                1,
                bytes,
                GgmlScratchRequirement::None,
                1,
                0,
                if request.ggml_type == GgmlType::IQ4_XS && m > MM_ROUTING_THRESHOLD {
                    "IQ4_XS dense prompt falls back to matvec because no dense MM kernel exists"
                } else {
                    "GGUF dense matvec route"
                },
            )
        }
    }
}

fn batched_mv(request: &GgmlCapabilityRequest, batch: u32, bytes: u64) -> GgmlCapability {
    let (m, _, _) = request.invocation.dimensions();
    if batch == 0 || m > MM_ROUTING_THRESHOLD {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "batched MV requires batch > 0 and M <= 8",
        );
    }
    let Some(total_bytes) = bytes.checked_mul(u64::from(batch)) else {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::ArithmeticOverflow,
            "batched GGUF weight bytes overflow u64",
        );
    };
    let specialized = request.workload == GgmlWorkloadClass::DecodeSingle;
    GgmlCapability::supported(
        request,
        dense_mv_route(request, true),
        specialized,
        !specialized,
        false,
        1,
        total_bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "native independent-weight batched matvec entry point",
    )
}

fn batched_mm(
    request: &GgmlCapabilityRequest,
    batch: u32,
    input_layout: GgmlBatchedInputLayout,
    bytes: u64,
) -> GgmlCapability {
    let (m, _, k) = request.invocation.dimensions();
    if batch == 0 || m <= MM_ROUTING_THRESHOLD || request.ggml_type == GgmlType::IQ4_XS {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "batched MM needs batch > 0, M > 8, and a type with a dense MM kernel",
        );
    }
    let Some(total_bytes) = bytes.checked_mul(u64::from(batch)) else {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::ArithmeticOverflow,
            "batched GGUF weight bytes overflow u64",
        );
    };
    if let GgmlBatchedInputLayout::Strided {
        row_bytes,
        batch_bytes,
    } = input_layout
    {
        let logical_row = u64::from(k) * 4;
        if row_bytes < logical_row
            || batch_bytes < logical_row
            || row_bytes % 32 != 0
            || batch_bytes % 32 != 0
        {
            return GgmlCapability::unsupported(
                request,
                GgmlRejectionCode::UnsupportedLayout,
                "strided batched MM input strides must be 32-byte aligned and cover one F32 row",
            );
        }
    }
    let (route, probe) = dense_mm_route(request, true);
    GgmlCapability::supported(
        request,
        route,
        request.workload == GgmlWorkloadClass::Prompt,
        request.workload != GgmlWorkloadClass::Prompt,
        probe,
        1,
        total_bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "native independent-weight batched MM entry point",
    )
}

fn perm021(request: &GgmlCapabilityRequest, head_dim: u32, bytes: u64) -> GgmlCapability {
    let (_, _, k) = request.invocation.dimensions();
    if !matches!(
        request.ggml_type,
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q6_K
    ) || head_dim == 0
        || head_dim % 32 != 0
        || k % head_dim != 0
    {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "perm021 requires Q4_0/Q5_0/Q8_0/Q6_K and a 32-aligned head dimension dividing K",
        );
    }
    let specialized = request.workload == GgmlWorkloadClass::Prompt;
    GgmlCapability::supported(
        request,
        GgmlKernelRoute::DensePerm021TensorMm,
        specialized,
        !specialized,
        true,
        1,
        bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "dedicated BF16-input permuted-021 tensor-MM entry point",
    )
}

fn fused_gate_up(request: &GgmlCapabilityRequest, bytes: u64) -> GgmlCapability {
    if !matches!(
        request.ggml_type,
        GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::IQ4_NL
    ) || request.workload == GgmlWorkloadClass::Embedding
    {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedType,
            "fused gate+up+SiLU supports Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL",
        );
    }
    let specialized = request.workload == GgmlWorkloadClass::DecodeSingle;
    GgmlCapability::supported(
        request,
        GgmlKernelRoute::FusedGateUpSilu,
        specialized,
        !specialized,
        false,
        2,
        bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "two same-codec weights execute in one fused gate+up+SiLU dispatch",
    )
}

fn expert_mv_route(request: &GgmlCapabilityRequest) -> GgmlKernelRoute {
    if (request.ggml_type == GgmlType::Q6_K && request.routing.expert_q6k_mv_nr2)
        || (request.ggml_type == GgmlType::Q8_0 && request.routing.expert_q8_0_mv_nr2)
    {
        GgmlKernelRoute::ExpertMvNr2
    } else {
        GgmlKernelRoute::ExpertMv
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertEntrypoint {
    AutoAllocated,
    ForcedMv,
    PooledShared,
    PooledPair,
    PooledSlotted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertAutoPlan {
    Mv,
    Mm,
}

pub(crate) fn plan_expert_auto_route(
    n_tokens: u32,
    top_k: u32,
    k: u32,
    force_mv: bool,
    routing: &GgmlRoutingPolicy,
) -> ExpertAutoPlan {
    if !force_mv && n_tokens > routing.expert_mm_threshold && matches!(top_k, 1 | 6 | 8) && k >= 32
    {
        ExpertAutoPlan::Mm
    } else {
        ExpertAutoPlan::Mv
    }
}

fn expert_mm_route(
    request: &GgmlCapabilityRequest,
    entrypoint: ExpertEntrypoint,
) -> (GgmlKernelRoute, bool) {
    let tensor = tensor_mm_auto_selected(request.ggml_type, request.routing.expert_tensor_mm);
    let route = match (entrypoint, tensor) {
        (ExpertEntrypoint::AutoAllocated, true) => GgmlKernelRoute::ExpertMmDeviceSelected,
        (ExpertEntrypoint::AutoAllocated, false) => GgmlKernelRoute::ExpertMmSimdgroup,
        (ExpertEntrypoint::PooledShared, true) => GgmlKernelRoute::ExpertPooledMmDeviceSelected,
        (ExpertEntrypoint::PooledShared, false) => GgmlKernelRoute::ExpertPooledMmSimdgroup,
        (ExpertEntrypoint::PooledPair, true) => GgmlKernelRoute::ExpertPooledPairMmDeviceSelected,
        (ExpertEntrypoint::PooledPair, false) => GgmlKernelRoute::ExpertPooledPairMmSimdgroup,
        (ExpertEntrypoint::PooledSlotted, true) => {
            GgmlKernelRoute::ExpertPooledSlottedMmDeviceSelected
        }
        (ExpertEntrypoint::PooledSlotted, false) => GgmlKernelRoute::ExpertPooledSlottedMmSimdgroup,
        (ExpertEntrypoint::ForcedMv, _) => unreachable!("forced MV has no MM route"),
    };
    (route, tensor)
}

fn expert(
    request: &GgmlCapabilityRequest,
    entrypoint: ExpertEntrypoint,
    shape: GgmlExpertShape,
    packed_expert_bytes: u64,
) -> GgmlCapability {
    if request.workload == GgmlWorkloadClass::Embedding
        || shape.top_k == 0
        || shape.n_experts == 0
        || shape.top_k > shape.n_experts
        || !shape.ids_within_expert_range
        || shape.expert_stride_bytes > i64::MAX as u64
    {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "expert execution requires in-range ids, top_k <= n_experts, an i64-safe stride, and a matmul regime",
        );
    }
    let buffer_bytes = match ggml_expert_bytes(
        request.ggml_type,
        shape.n_experts,
        shape.n,
        shape.k,
        shape.expert_stride_bytes,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return GgmlCapability::unsupported(
                request,
                if shape.expert_stride_bytes < packed_expert_bytes {
                    GgmlRejectionCode::UnsupportedLayout
                } else {
                    GgmlRejectionCode::ArithmeticOverflow
                },
                error.to_string(),
            );
        }
    };
    let has_map = matches!(shape.top_k, 1 | 6 | 8);
    let mm_eligible = plan_expert_auto_route(
        shape.n_tokens,
        shape.top_k,
        shape.k,
        entrypoint == ExpertEntrypoint::ForcedMv,
        &request.routing,
    ) == ExpertAutoPlan::Mm;
    let requires_mm = matches!(
        entrypoint,
        ExpertEntrypoint::PooledPair | ExpertEntrypoint::PooledSlotted
    );
    if requires_mm && !mm_eligible {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "paired/slotted pooled expert entry point requires the mm_id route",
        );
    }
    if mm_eligible && !shape.ids_are_distinct_per_token {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "expert MM requires distinct expert ids within each token's top-k",
        );
    }
    if entrypoint != ExpertEntrypoint::ForcedMv && mm_eligible {
        let (route, tensor_probe) = expert_mm_route(request, entrypoint);
        let Some(htpe_bytes) = u64::from(shape.n_experts).checked_mul(4) else {
            return GgmlCapability::unsupported(
                request,
                GgmlRejectionCode::ArithmeticOverflow,
                "expert htpe scratch bytes overflow u64",
            );
        };
        let Some(hids_bytes) = u64::from(shape.n_experts)
            .checked_mul(u64::from(shape.n_tokens))
            .and_then(|elements| elements.checked_mul(4))
        else {
            return GgmlCapability::unsupported(
                request,
                GgmlRejectionCode::ArithmeticOverflow,
                "expert hids scratch bytes overflow u64",
            );
        };
        let scratch = GgmlScratchRequirement::ExpertMm {
            htpe_bytes,
            hids_bytes,
            caller_owned: entrypoint != ExpertEntrypoint::AutoAllocated,
            schedule_reused: entrypoint == ExpertEntrypoint::PooledPair,
        };
        return GgmlCapability::supported(
            request,
            route,
            request.workload == GgmlWorkloadClass::Prompt,
            request.workload != GgmlWorkloadClass::Prompt,
            tensor_probe,
            if entrypoint == ExpertEntrypoint::PooledPair {
                2
            } else {
                1
            },
            buffer_bytes,
            scratch,
            if entrypoint == ExpertEntrypoint::PooledPair {
                3
            } else {
                2
            },
            1,
            "expert mm_id route with explicit schedule/layout entry point",
        );
    }
    if entrypoint == ExpertEntrypoint::PooledSlotted {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedLayout,
            "slotted expert input has no matvec fallback",
        );
    }
    let specialized = request.workload == GgmlWorkloadClass::DecodeSingle;
    GgmlCapability::supported(
        request,
        expert_mv_route(request),
        specialized,
        !specialized,
        false,
        1,
        buffer_bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        if shape.n_tokens > request.routing.expert_mm_threshold && !has_map {
            "top_k has no mm_id map kernel; expert execution falls back to matvec"
        } else {
            "expert-routed matvec entry point"
        },
    )
}

fn expert_swiglu_down_q4(
    request: &GgmlCapabilityRequest,
    shape: GgmlExpertShape,
    packed_expert_bytes: u64,
) -> GgmlCapability {
    if request.ggml_type != GgmlType::Q4_0 || request.workload == GgmlWorkloadClass::Embedding {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedType,
            "fused expert SwiGLU-down entry point supports Q4_0 only",
        );
    }
    if shape.top_k == 0
        || shape.n_experts == 0
        || shape.top_k > shape.n_experts
        || !shape.ids_within_expert_range
        || shape.expert_stride_bytes > i64::MAX as u64
        || shape.expert_stride_bytes < packed_expert_bytes
    {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::InvalidOperationContract,
            "fused expert SwiGLU-down requires valid expert dimensions and stride",
        );
    }
    let buffer_bytes = match ggml_expert_bytes(
        request.ggml_type,
        shape.n_experts,
        shape.n,
        shape.k,
        shape.expert_stride_bytes,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return GgmlCapability::unsupported(
                request,
                GgmlRejectionCode::ArithmeticOverflow,
                error.to_string(),
            );
        }
    };
    GgmlCapability::supported(
        request,
        GgmlKernelRoute::ExpertSwiGluDownQ4,
        request.workload == GgmlWorkloadClass::DecodeSingle,
        request.workload != GgmlWorkloadClass::DecodeSingle,
        false,
        1,
        buffer_bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "Q4_0 fused SwiGLU plus expert-routed down projection",
    )
}

fn embedding(request: &GgmlCapabilityRequest, bytes: u64) -> GgmlCapability {
    if request.workload != GgmlWorkloadClass::Embedding {
        return GgmlCapability::unsupported(
            request,
            GgmlRejectionCode::UnsupportedRegime,
            "embedding gather requires the embedding-gather regime",
        );
    }
    let route = match request.ggml_type {
        GgmlType::F32 => GgmlKernelRoute::EmbeddingF32,
        GgmlType::F16 => GgmlKernelRoute::EmbeddingF16,
        GgmlType::BF16 => GgmlKernelRoute::EmbeddingBF16,
        GgmlType::Q2_K => GgmlKernelRoute::EmbeddingQ2K,
        GgmlType::Q4_0 => GgmlKernelRoute::EmbeddingQ4_0,
        GgmlType::Q5_0 => GgmlKernelRoute::EmbeddingQ5_0,
        GgmlType::Q4_K => GgmlKernelRoute::EmbeddingQ4K,
        GgmlType::Q5_K => GgmlKernelRoute::EmbeddingQ5K,
        GgmlType::Q6_K => GgmlKernelRoute::EmbeddingQ6K,
        GgmlType::Q8_0 => GgmlKernelRoute::EmbeddingQ8_0,
        other => {
            return GgmlCapability::unsupported(
                request,
                GgmlRejectionCode::UnsupportedType,
                format!("GGUF embedding gather does not support {other:?}"),
            );
        }
    };
    GgmlCapability::supported(
        request,
        route,
        true,
        false,
        false,
        1,
        bytes,
        GgmlScratchRequirement::None,
        1,
        0,
        "dedicated native-storage embedding-gather route",
    )
}

/// Return structural GGUF execution capability for one exact graph operation.
///
/// The function is pure: it never reads process environment or probes a
/// device. Callers must pass the effective routing policy, and must bind any
/// `requires_device_probe` result to a runtime trace before using measured
/// costs for allocation.
pub fn ggml_capability(request: GgmlCapabilityRequest) -> GgmlCapability {
    if let Some(rejection) = validate_common(&request) {
        return rejection;
    }
    let Some(matrix_bytes) = packed_matrix_bytes(&request) else {
        return GgmlCapability::unsupported(
            &request,
            GgmlRejectionCode::ArithmeticOverflow,
            "packed matrix byte count overflows u64",
        );
    };
    match request.invocation {
        GgmlInvocation::DenseAuto { .. } => dense_auto(&request, matrix_bytes),
        GgmlInvocation::DenseBatchedMv { batch, .. } => batched_mv(&request, batch, matrix_bytes),
        GgmlInvocation::DenseBatchedMm {
            batch,
            input_layout,
            ..
        } => batched_mm(&request, batch, input_layout, matrix_bytes),
        GgmlInvocation::DensePerm021Bf16 { head_dim, .. } => {
            perm021(&request, head_dim, matrix_bytes)
        }
        GgmlInvocation::DenseGateUpSiluPair { .. } => fused_gate_up(&request, matrix_bytes),
        GgmlInvocation::ExpertAutoAllocated { shape } => expert(
            &request,
            ExpertEntrypoint::AutoAllocated,
            shape,
            matrix_bytes,
        ),
        GgmlInvocation::ExpertForceMv { shape } => {
            expert(&request, ExpertEntrypoint::ForcedMv, shape, matrix_bytes)
        }
        GgmlInvocation::ExpertPooled {
            shape,
            input_layout,
        } => expert(
            &request,
            match input_layout {
                GgmlExpertInputLayout::SharedPerToken => ExpertEntrypoint::PooledShared,
                GgmlExpertInputLayout::Slotted => ExpertEntrypoint::PooledSlotted,
            },
            shape,
            matrix_bytes,
        ),
        GgmlInvocation::ExpertPooledPair { shape } => {
            expert(&request, ExpertEntrypoint::PooledPair, shape, matrix_bytes)
        }
        GgmlInvocation::ExpertSwiGluDownQ4 { shape } => {
            expert_swiglu_down_q4(&request, shape, matrix_bytes)
        }
        GgmlInvocation::EmbeddingGather { .. } => embedding(&request, matrix_bytes),
    }
}

#[cfg(test)]
mod tests;
