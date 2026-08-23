//! Activation-scoped calibration for native scalar expert-ID matmul.
//!
//! Callers declare the exact expert shapes reachable by one loaded model and
//! lend native weights for extent/identity validation during activation only.
//! Calibration proves Direct/Grouped bit identity on one current representative
//! per exact shape using adversarial F32 activations and routing before timing
//! both balanced and maximally skewed distinct routing. The shared shader theorem
//! then authorizes every declared identity of that exact shape. The frozen plan
//! is pointer-free and bound to the prepared pipeline identities and theorem
//! digest; request-path dispatch never benchmarks. Process-wide reuse contains
//! timing metadata only; activation epochs remain registry-local.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, DispatchKind, EncodedKernelDispatch};
use crate::error::{MlxError, Result};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::dense_matmul_id::{
    dense_matmul_id, dense_matmul_id_capability, pipeline_names, DenseMatmulIdDispatchReceipt,
    DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity, DenseMatmulIdParams, DenseMatmulIdRoute,
    DenseMatmulIdScratch, DENSE_MATMUL_ID_SCHEMA_VERSION,
};

pub const DENSE_MATMUL_ID_ROUTE_SCHEMA_VERSION: u32 = DENSE_MATMUL_ID_SCHEMA_VERSION;
/// Canonical authority statement for value-independent Grouped routing.
///
/// The theorem is deliberately narrow: it applies only when the full
/// `DenseMatmulIdShape`, native weight dtype, compiled direct/map/grouped
/// pipeline identities, and this source statement are identical. Direct and
/// Grouped call the same shader helpers for input addressing, weight scalar
/// addressing, native-scalar-to-F32 widening, F32 fused multiply-add, and
/// `simd_sum`; Grouped only reorders independent output rows and stages the
/// already-widened weight scalar. It does not authorize a different dtype,
/// pipeline label, shape, stride, layout, multiplicity, K-tail, or N-tail.
pub const DENSE_MATMUL_ID_VALUE_INDEPENDENCE_THEOREM: &str =
    "dense-matmul-id-value-independent-v1:exact-shape+exact-native-dtype+exact-pipeline-identities;shared-input-offset+expert-base+weight-index+widen+f32-fma+simd-sum;grouped-reorders-independent-rows-only;no-cross-dtype-pipeline-shape-stride-layout-multiplicity-or-tail-authority";
const CALIBRATION_SAMPLES: usize = 5;
const MATERIAL_WIN_FRACTION: f64 = 0.05;
const GPU_CONTRARY_TOLERANCE: f64 = 0.02;
const OUTPUT_GUARD_ELEMENTS: usize = 16;
const OUTPUT_POISON_BITS: u32 = 0x7fc0_00c7;

/// Full pointer-free key for one native scalar expert-ID execution case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdShape {
    pub weight_dtype: DType,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub top_k: u32,
    pub n_experts: u32,
    pub expert_stride_bytes: u64,
    pub input_layout: DenseMatmulIdInputLayout,
    pub id_multiplicity: DenseMatmulIdMultiplicity,
}

/// Width-independent execution contract admitted at activation. An unseen M
/// may use Direct only when every other field matches one of these bases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DenseMatmulIdBaseShape {
    weight_dtype: DType,
    n: u32,
    k: u32,
    top_k: u32,
    n_experts: u32,
    expert_stride_bytes: u64,
    input_layout: DenseMatmulIdInputLayout,
    id_multiplicity: DenseMatmulIdMultiplicity,
}

impl From<DenseMatmulIdShape> for DenseMatmulIdBaseShape {
    fn from(shape: DenseMatmulIdShape) -> Self {
        Self {
            weight_dtype: shape.weight_dtype,
            n: shape.n,
            k: shape.k,
            top_k: shape.top_k,
            n_experts: shape.n_experts,
            expert_stride_bytes: shape.expert_stride_bytes,
            input_layout: shape.input_layout,
            id_multiplicity: shape.id_multiplicity,
        }
    }
}

impl DenseMatmulIdShape {
    fn params(self, route: DenseMatmulIdRoute) -> DenseMatmulIdParams {
        DenseMatmulIdParams {
            m: self.m,
            n: self.n,
            k: self.k,
            top_k: self.top_k,
            n_experts: self.n_experts,
            expert_stride_bytes: self.expert_stride_bytes,
            input_layout: self.input_layout,
            id_multiplicity: self.id_multiplicity,
            route,
        }
    }
}

/// One exact activation-time case. `params.route` is ignored: calibration owns
/// route selection and keys every other field plus `weight.dtype()`.
pub struct DenseMatmulIdCalibrationCase<'a> {
    pub weight: &'a MlxBuffer,
    pub params: DenseMatmulIdParams,
}

/// Fail-closed activation-cost limits. `max_submissions` includes the final
/// cleanup boundary when calibration submitted GPU work. Metal compilation
/// and one GPU submission are indivisible; any elapsed-time overrun they cause
/// is receipted and prevents a Grouped selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdCalibrationLimits {
    pub max_elapsed_ms: u64,
    pub max_cases: u32,
    pub max_submissions: u32,
}

impl Default for DenseMatmulIdCalibrationLimits {
    fn default() -> Self {
        Self {
            max_elapsed_ms: 15_000,
            max_cases: 256,
            max_submissions: 6_145,
        }
    }
}

/// Runtime ID distributions exercised by the activation-time performance
/// gate. A route is admitted only when it wins both profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdRoutingProfile {
    Balanced,
    MaximallySkewedDistinct,
}

const ROUTING_PROFILES: [DenseMatmulIdRoutingProfile; 2] = [
    DenseMatmulIdRoutingProfile::Balanced,
    DenseMatmulIdRoutingProfile::MaximallySkewedDistinct,
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdTimingDistribution {
    pub p25_us: f64,
    pub median_us: f64,
    pub p75_us: f64,
    pub samples: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdRouteTiming {
    pub profile: DenseMatmulIdRoutingProfile,
    pub route: DenseMatmulIdRoute,
    pub wall: DenseMatmulIdTimingDistribution,
    pub gpu: DenseMatmulIdTimingDistribution,
    pub encoded: Vec<EncodedKernelDispatch>,
    pub pipelines: Vec<KernelPipelineIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdSelectionStatus {
    CalibratedWinner,
    DirectFastest,
    NoStableWinner,
    DirectOnly,
    IncoherentGrouped,
    BudgetFallback,
    ErrorFallback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdCalibrationDecision {
    pub shape: DenseMatmulIdShape,
    /// Distinct logical weight-buffer identities admitted for this shape in
    /// the current activation. No identity is retained by the plan/cache.
    pub declared_weight_identities: u32,
    /// Declared identities authorized by the exact-shape value-independence
    /// theorem when Grouped is selected. No identity or buffer is retained.
    pub theorem_authorized_weight_identities: u32,
    pub selected_route: DenseMatmulIdRoute,
    pub status: DenseMatmulIdSelectionStatus,
    pub timings: Vec<DenseMatmulIdRouteTiming>,
    pub process_cache_hit: bool,
    /// GPU work executed against one current representative for this exact
    /// shape, with adversarial activations/routing, to prove Direct/Grouped bit
    /// identity and tail coverage.
    pub empirical_shape_proof_submissions: u32,
    pub empirical_shape_proof_dispatches: u32,
    /// Comparative timing work executed during this activation.
    pub current_timing_submissions: u32,
    pub current_timing_dispatches: u32,
    /// Timing sample counts reused as process metadata. These are historical
    /// observations, not GPU submissions executed by this activation.
    pub cached_timing_submissions: u32,
    pub cached_timing_dispatches: u32,
    pub calibration_submissions: u32,
    pub calibration_dispatches: u32,
    pub fallback_reason: Option<String>,
}

/// Immutable pointer-free plan scoped to one activation epoch.
#[derive(Clone, Debug)]
pub struct DenseMatmulIdRoutePlan {
    plan_id: String,
    build_fingerprint: String,
    device_name: String,
    device_registry_id: u64,
    activation_epoch: u64,
    activation_authority_digest: String,
    pipeline_set_fingerprint: String,
    value_independence_theorem_sha256: String,
    pipeline_identities: Vec<KernelPipelineIdentity>,
    decisions: HashMap<DenseMatmulIdShape, DenseMatmulIdRoute>,
    admitted_bases: HashSet<DenseMatmulIdBaseShape>,
}

impl DenseMatmulIdRoutePlan {
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    pub fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }

    pub fn activation_authority_digest(&self) -> &str {
        &self.activation_authority_digest
    }

    pub fn route_for(&self, shape: DenseMatmulIdShape) -> Option<DenseMatmulIdRoute> {
        self.decisions.get(&shape).copied()
    }

    pub fn value_independence_theorem_sha256(&self) -> &str {
        &self.value_independence_theorem_sha256
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdCalibrationBatchReceipt {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub build_fingerprint: String,
    pub pipeline_set_fingerprint: String,
    pub pipeline_identities: Vec<KernelPipelineIdentity>,
    pub plan_id: String,
    pub activation_epoch: u64,
    pub activation_authority_digest: String,
    pub device_name: String,
    pub device_registry_id: u64,
    pub declared_cases: u32,
    pub declared_weight_identities: u32,
    pub theorem_authorized_weight_identities: u32,
    pub value_independence_theorem_sha256: String,
    pub calibrated_decisions: u32,
    pub process_cache_hits: u32,
    pub fallback_decisions: u32,
    pub empirical_shape_proof_submissions: u32,
    pub empirical_shape_proof_dispatches: u32,
    pub current_timing_submissions: u32,
    pub current_timing_dispatches: u32,
    pub cached_timing_submissions: u32,
    pub cached_timing_dispatches: u32,
    /// Empty command-buffer boundaries committed after calibration attempts
    /// so dropped proof/scratch allocations cannot leak into serving.
    pub cleanup_submissions: u32,
    pub calibration_submissions: u32,
    pub calibration_dispatches: u32,
    pub elapsed_ms: f64,
    pub deadline_overrun_ms: f64,
    pub decisions: Vec<DenseMatmulIdCalibrationDecision>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdAutoDispatchReceipt {
    pub route: DenseMatmulIdRoute,
    pub decision_source: DenseMatmulIdDecisionSource,
    pub activation_epoch: u64,
    pub primitive: DenseMatmulIdDispatchReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdDecisionSource {
    FrozenPlan,
    UndeclaredDirect,
}

/// Exact host-side encoding and pipeline identity for one native scalar
/// expert-ID call. Primitive traces have no plan fields; auto traces bind the
/// exact frozen activation plan as well.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdDispatchTrace {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub device_name: String,
    pub device_registry_id: u64,
    pub shape: DenseMatmulIdShape,
    pub route: DenseMatmulIdRoute,
    pub decision_source: Option<DenseMatmulIdDecisionSource>,
    pub pipeline_set_fingerprint: String,
    pub encoded: Vec<EncodedKernelDispatch>,
    pub pipelines: Vec<KernelPipelineIdentity>,
    pub plan_id: Option<String>,
    pub plan_build_fingerprint: Option<String>,
    pub plan_pipeline_set_fingerprint: Option<String>,
    pub plan_value_independence_theorem_sha256: Option<String>,
    pub plan_activation_authority_digest: Option<String>,
    pub activation_epoch: Option<u64>,
}

#[derive(Default)]
pub(crate) struct DenseMatmulIdAutoState {
    frozen_plan: Option<Arc<DenseMatmulIdRoutePlan>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProcessKey {
    build_fingerprint: String,
    device_name: String,
    device_registry_id: u64,
    pipeline_set_fingerprint: String,
    shape: DenseMatmulIdShape,
}

/// Process-wide reuse is deliberately limited to route/timing metadata. It
/// never retains a weight buffer or broadens the exact-shape theorem.
#[derive(Clone)]
struct CachedRouteMetadata {
    shape: DenseMatmulIdShape,
    selected_route: DenseMatmulIdRoute,
    status: DenseMatmulIdSelectionStatus,
    timings: Vec<DenseMatmulIdRouteTiming>,
    timing_submissions: u32,
    timing_dispatches: u32,
    fallback_reason: Option<String>,
}

impl CachedRouteMetadata {
    fn from_decision(decision: &DenseMatmulIdCalibrationDecision) -> Self {
        Self {
            shape: decision.shape,
            selected_route: decision.selected_route,
            status: decision.status,
            timings: decision.timings.clone(),
            timing_submissions: decision.current_timing_submissions,
            timing_dispatches: decision.current_timing_dispatches,
            fallback_reason: decision.fallback_reason.clone(),
        }
    }

    fn activation_decision(&self) -> DenseMatmulIdCalibrationDecision {
        DenseMatmulIdCalibrationDecision {
            shape: self.shape,
            declared_weight_identities: 0,
            theorem_authorized_weight_identities: 0,
            selected_route: self.selected_route,
            status: self.status,
            timings: self.timings.clone(),
            process_cache_hit: true,
            empirical_shape_proof_submissions: 0,
            empirical_shape_proof_dispatches: 0,
            current_timing_submissions: 0,
            current_timing_dispatches: 0,
            cached_timing_submissions: self.timing_submissions,
            cached_timing_dispatches: self.timing_dispatches,
            calibration_submissions: 0,
            calibration_dispatches: 0,
            fallback_reason: self.fallback_reason.clone(),
        }
    }
}

enum CachedRouteEntry {
    Ready(CachedRouteMetadata),
    Failed(String),
}

type CalibrationCell = OnceLock<CachedRouteEntry>;

fn process_cache() -> &'static Mutex<HashMap<ProcessKey, Arc<CalibrationCell>>> {
    static CACHE: OnceLock<Mutex<HashMap<ProcessKey, Arc<CalibrationCell>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn process_cell(key: ProcessKey) -> Result<Arc<CalibrationCell>> {
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense_matmul_id calibration cache mutex is poisoned".into())
    })?;
    Ok(cache
        .entry(key)
        .or_insert_with(|| Arc::new(OnceLock::new()))
        .clone())
}

fn evict_cell_if_same(key: &ProcessKey, cell: &Arc<CalibrationCell>) -> Result<()> {
    #[cfg(test)]
    if take_test_failure(TestFailurePoint::CacheEviction) {
        return Err(MlxError::InvalidArgument(
            "injected dense_matmul_id cache eviction failure".into(),
        ));
    }
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense_matmul_id calibration cache mutex is poisoned".into())
    })?;
    if cache
        .get(key)
        .is_some_and(|current| Arc::ptr_eq(current, cell))
    {
        cache.remove(key);
    }
    Ok(())
}

fn source_build_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            let mut digest = Sha256::new();
            digest.update(env!("CARGO_PKG_VERSION").as_bytes());
            digest.update(DENSE_MATMUL_ID_ROUTE_SCHEMA_VERSION.to_le_bytes());
            digest.update(DENSE_MATMUL_ID_VALUE_INDEPENDENCE_THEOREM.as_bytes());
            digest.update(include_bytes!("dense_matmul_id.rs"));
            digest.update(include_bytes!("dense_matmul_id_auto.rs"));
            digest.update(include_bytes!("../shaders/dense_matmul_id.metal"));
            hex::encode(digest.finalize())
        })
        .as_str()
}

/// Stable digest embedded in activation plans and dispatch traces. This is
/// metadata only and never retains activation buffers.
pub fn dense_matmul_id_value_independence_theorem_sha256() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST
        .get_or_init(|| {
            hex::encode(Sha256::digest(
                DENSE_MATMUL_ID_VALUE_INDEPENDENCE_THEOREM.as_bytes(),
            ))
        })
        .as_str()
}

fn fingerprint_pipeline_identities(
    identities: &[KernelPipelineIdentity],
) -> Result<(String, String)> {
    let mut ordered = identities.to_vec();
    ordered.sort_by(|left, right| left.pipeline_label.cmp(&right.pipeline_label));
    if let Some(conflict) = ordered
        .windows(2)
        .find(|pair| pair[0].pipeline_label == pair[1].pipeline_label && pair[0] != pair[1])
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id has conflicting identities for pipeline {}",
            conflict[0].pipeline_label
        )));
    }
    ordered.dedup_by(|left, right| left.pipeline_label == right.pipeline_label);
    let encoded = serde_json::to_vec(&ordered).map_err(|error| {
        MlxError::InvalidArgument(format!(
            "serialize dense_matmul_id pipeline identities: {error}"
        ))
    })?;
    let pipeline_set_fingerprint = hex::encode(Sha256::digest(encoded));
    let mut digest = Sha256::new();
    digest.update(source_build_fingerprint().as_bytes());
    digest.update(pipeline_set_fingerprint.as_bytes());
    Ok((pipeline_set_fingerprint, hex::encode(digest.finalize())))
}

fn checked_product(label: &str, factors: &[u64]) -> Result<usize> {
    let bytes = factors.iter().try_fold(1u64, |value, factor| {
        value.checked_mul(*factor).ok_or_else(|| {
            MlxError::InvalidArgument(format!("dense_matmul_id calibration {label} size overflow"))
        })
    })?;
    usize::try_from(bytes).map_err(|_| {
        MlxError::InvalidArgument(format!("dense_matmul_id calibration {label} exceeds usize"))
    })
}

fn shape_from_case(case: &DenseMatmulIdCalibrationCase<'_>) -> DenseMatmulIdShape {
    DenseMatmulIdShape {
        weight_dtype: case.weight.dtype(),
        m: case.params.m,
        n: case.params.n,
        k: case.params.k,
        top_k: case.params.top_k,
        n_experts: case.params.n_experts,
        expert_stride_bytes: case.params.expert_stride_bytes,
        input_layout: case.params.input_layout,
        id_multiplicity: case.params.id_multiplicity,
    }
}

fn shape_from_call(weights: &MlxBuffer, params: &DenseMatmulIdParams) -> DenseMatmulIdShape {
    DenseMatmulIdShape {
        weight_dtype: weights.dtype(),
        m: params.m,
        n: params.n,
        k: params.k,
        top_k: params.top_k,
        n_experts: params.n_experts,
        expert_stride_bytes: params.expert_stride_bytes,
        input_layout: params.input_layout,
        id_multiplicity: params.id_multiplicity,
    }
}

struct ValidatedCase<'a> {
    weights: Vec<&'a MlxBuffer>,
    shape: DenseMatmulIdShape,
    grouped_legal: bool,
}

fn logical_weight_identity(weight: &MlxBuffer) -> (usize, u64, usize) {
    (
        weight.contents_ptr() as usize,
        weight.byte_offset(),
        weight.data_byte_len(),
    )
}

fn validate_cases<'a>(
    cases: &'a [DenseMatmulIdCalibrationCase<'a>],
    limits: DenseMatmulIdCalibrationLimits,
) -> Result<Vec<ValidatedCase<'a>>> {
    if limits.max_elapsed_ms == 0 || limits.max_cases == 0 || limits.max_submissions == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id calibration limits must be nonzero".into(),
        ));
    }
    if cases.is_empty() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id calibration requires at least one case".into(),
        ));
    }
    let mut validated = HashMap::<DenseMatmulIdShape, ValidatedCase<'a>>::new();
    for case in cases {
        let shape = shape_from_case(case);
        let direct = shape.params(DenseMatmulIdRoute::Direct);
        let capability = dense_matmul_id_capability(shape.weight_dtype, &direct)?;
        if case.weight.data_byte_len() < capability.required_weight_bytes
            || case.weight.byte_offset() % shape.weight_dtype.size_of() as u64 != 0
        {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id calibration weight does not satisfy {shape:?}"
            )));
        }
        let grouped_legal = dense_matmul_id_capability(
            shape.weight_dtype,
            &shape.params(DenseMatmulIdRoute::GroupedPrefill),
        )
        .is_ok();
        if let Some(existing) = validated.get_mut(&shape) {
            let identity = logical_weight_identity(case.weight);
            if existing
                .weights
                .iter()
                .all(|weight| logical_weight_identity(weight) != identity)
            {
                existing.weights.push(case.weight);
            }
        } else {
            validated.insert(
                shape,
                ValidatedCase {
                    weights: vec![case.weight],
                    shape,
                    grouped_legal,
                },
            );
        }
    }
    if validated.len() > limits.max_cases as usize {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id calibration declares {} cases, limit is {}",
            validated.len(),
            limits.max_cases
        )));
    }
    let mut validated: Vec<_> = validated.into_values().collect();
    validated.sort_by_key(|case| case.shape);
    Ok(validated)
}

fn activation_authority_digest(
    activation_epoch: u64,
    device: &MlxDevice,
    cases: &[ValidatedCase<'_>],
) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"dense-matmul-id-activation-authority-v1");
    digest.update(activation_epoch.to_le_bytes());
    digest.update(device.registry_id().to_le_bytes());
    digest.update(device.name().as_bytes());
    let mut ordered: Vec<_> = cases.iter().collect();
    ordered.sort_by_key(|case| case.shape);
    for case in ordered {
        let encoded_shape = serde_json::to_vec(&case.shape).map_err(|error| {
            MlxError::InvalidArgument(format!(
                "serialize dense_matmul_id activation shape: {error}"
            ))
        })?;
        digest.update((encoded_shape.len() as u64).to_le_bytes());
        digest.update(encoded_shape);
        let mut identities: Vec<_> = case
            .weights
            .iter()
            .map(|weight| logical_weight_identity(weight))
            .collect();
        identities.sort_unstable();
        digest.update((identities.len() as u64).to_le_bytes());
        for (pointer, byte_offset, logical_bytes) in identities {
            let pointer = u64::try_from(pointer).map_err(|_| {
                MlxError::InvalidArgument(
                    "dense_matmul_id logical weight pointer exceeds u64".into(),
                )
            })?;
            let logical_bytes = u64::try_from(logical_bytes).map_err(|_| {
                MlxError::InvalidArgument(
                    "dense_matmul_id logical weight extent exceeds u64".into(),
                )
            })?;
            digest.update(pointer.to_le_bytes());
            digest.update(byte_offset.to_le_bytes());
            digest.update(logical_bytes.to_le_bytes());
        }
    }
    Ok(hex::encode(digest.finalize()))
}

struct PreparedDenseMatmulIdRoutes {
    pipeline_set_fingerprint: String,
    build_fingerprint: String,
    identities: Vec<KernelPipelineIdentity>,
    grouped_preparation_errors: HashMap<DType, String>,
}

fn prepare_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cases: &[ValidatedCase<'_>],
) -> Result<PreparedDenseMatmulIdRoutes> {
    let mut direct_names = Vec::<&'static str>::new();
    let mut grouped_dtypes = Vec::<DType>::new();
    for case in cases {
        let (direct, _) = pipeline_names(case.shape.weight_dtype)?;
        if !direct_names.contains(&direct) {
            direct_names.push(direct);
        }
        if case.grouped_legal && !grouped_dtypes.contains(&case.shape.weight_dtype) {
            grouped_dtypes.push(case.shape.weight_dtype);
        }
    }
    direct_names.sort_unstable();
    grouped_dtypes.sort_unstable();
    let mut identities = Vec::with_capacity(direct_names.len() + grouped_dtypes.len() + 1);
    for name in direct_names {
        registry.get_pipeline(name, device.metal_device())?;
        identities.push(registry.pipeline_identity(name)?);
    }
    let mut grouped_preparation_errors = HashMap::new();
    if !grouped_dtypes.is_empty() {
        let map_name = "dense_matmul_id_map_distinct";
        match registry.get_pipeline(map_name, device.metal_device()) {
            Ok(_) => identities.push(registry.pipeline_identity(map_name)?),
            Err(error) => {
                let reason = format!("prepare grouped route map pipeline: {error}");
                for dtype in grouped_dtypes {
                    grouped_preparation_errors.insert(dtype, reason.clone());
                }
                let (pipeline_set_fingerprint, build_fingerprint) =
                    fingerprint_pipeline_identities(&identities)?;
                return Ok(PreparedDenseMatmulIdRoutes {
                    pipeline_set_fingerprint,
                    build_fingerprint,
                    identities,
                    grouped_preparation_errors,
                });
            }
        }
        for dtype in grouped_dtypes {
            let (_, grouped) = pipeline_names(dtype)?;
            match registry.get_pipeline(grouped, device.metal_device()) {
                Ok(_) => identities.push(registry.pipeline_identity(grouped)?),
                Err(error) => {
                    grouped_preparation_errors
                        .insert(dtype, format!("prepare grouped route pipeline: {error}"));
                }
            }
        }
    }
    let (pipeline_set_fingerprint, build_fingerprint) =
        fingerprint_pipeline_identities(&identities)?;
    Ok(PreparedDenseMatmulIdRoutes {
        pipeline_set_fingerprint,
        build_fingerprint,
        identities,
        grouped_preparation_errors,
    })
}

fn verify_prepared_pipeline_identities(
    registry: &KernelRegistry,
    expected: &[KernelPipelineIdentity],
) -> Result<(String, String)> {
    let mut current = Vec::with_capacity(expected.len());
    for expected_identity in expected {
        let current_identity = registry.pipeline_identity(&expected_identity.pipeline_label)?;
        if current_identity != *expected_identity {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id pipeline identity changed for {}",
                expected_identity.pipeline_label
            )));
        }
        current.push(current_identity);
    }
    fingerprint_pipeline_identities(&current)
}

fn poison_output(output: &mut MlxBuffer) -> Result<()> {
    output.as_mut_slice::<u32>()?.fill(OUTPUT_POISON_BITS);
    Ok(())
}

fn verified_bits(output: &MlxBuffer, logical_elements: usize) -> Result<Vec<u32>> {
    let values = output.as_slice::<f32>()?;
    let expected = logical_elements
        .checked_add(OUTPUT_GUARD_ELEMENTS)
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id guard overflow".into()))?;
    if values.len() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id proof output has {} elements, expected {expected}",
            values.len()
        )));
    }
    let mut bits = Vec::with_capacity(logical_elements);
    for (index, value) in values[..logical_elements].iter().enumerate() {
        if value.to_bits() == OUTPUT_POISON_BITS || !value.is_finite() {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id proof output element {index} was not finitely overwritten"
            )));
        }
        bits.push(value.to_bits());
    }
    if values[logical_elements..]
        .iter()
        .any(|value| value.to_bits() != OUTPUT_POISON_BITS)
    {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id proof overwrote its output guard".into(),
        ));
    }
    Ok(bits)
}

struct CalibrationBuffers {
    input: MlxBuffer,
    expert_ids: MlxBuffer,
    output: MlxBuffer,
    scratch: DenseMatmulIdScratch,
    output_elements: usize,
}

fn calibration_buffers(
    device: &MlxDevice,
    shape: DenseMatmulIdShape,
    profile: DenseMatmulIdRoutingProfile,
) -> Result<CalibrationBuffers> {
    let routed_rows =
        checked_product("routed rows", &[u64::from(shape.m), u64::from(shape.top_k)])?;
    let input_rows = match shape.input_layout {
        DenseMatmulIdInputLayout::SharedPerToken => shape.m as usize,
        DenseMatmulIdInputLayout::Slotted => routed_rows,
    };
    let input_elements = input_rows
        .checked_mul(shape.k as usize)
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id input overflow".into()))?;
    let output_elements = routed_rows
        .checked_mul(shape.n as usize)
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id output overflow".into()))?;
    let mut input = device.alloc_buffer(
        input_elements
            .checked_mul(DType::F32.size_of())
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense_matmul_id input bytes overflow".into())
            })?,
        DType::F32,
        vec![input_rows, shape.k as usize],
    )?;
    let adversarial = [
        1.000_976_6f32,
        -1.000_976_6,
        f32::from_bits(0x3f80_0001),
        f32::from_bits(0xbf80_0001),
        0.333_333_34,
        -0.142_857_15,
        0.000_012_345_679,
        -0.000_009_765_625,
    ];
    for (index, value) in input.as_mut_slice::<f32>()?.iter_mut().enumerate() {
        let base = adversarial[(index * 5 + index / shape.k as usize) % adversarial.len()];
        *value = base * (1.0 + ((index * 17) % 11) as f32 / 4096.0);
    }
    let mut expert_ids = device.alloc_buffer(
        routed_rows
            .checked_mul(DType::U32.size_of())
            .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id ID bytes overflow".into()))?,
        DType::U32,
        vec![shape.m as usize, shape.top_k as usize],
    )?;
    for token in 0..shape.m as usize {
        for slot in 0..shape.top_k as usize {
            expert_ids.as_mut_slice::<u32>()?[token * shape.top_k as usize + slot] =
                match (shape.id_multiplicity, profile) {
                    (
                        DenseMatmulIdMultiplicity::DistinctPerToken,
                        DenseMatmulIdRoutingProfile::Balanced,
                    ) => ((token + slot) % shape.n_experts as usize) as u32,
                    (
                        DenseMatmulIdMultiplicity::DistinctPerToken,
                        DenseMatmulIdRoutingProfile::MaximallySkewedDistinct,
                    ) => slot as u32,
                    (DenseMatmulIdMultiplicity::MayRepeat, _) => {
                        (token % shape.n_experts as usize) as u32
                    }
                };
        }
    }
    let guarded = output_elements
        .checked_add(OUTPUT_GUARD_ELEMENTS)
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id output guard overflow".into()))?;
    let output = device.alloc_buffer(
        guarded.checked_mul(DType::F32.size_of()).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id output bytes overflow".into())
        })?,
        DType::F32,
        vec![guarded],
    )?;
    Ok(CalibrationBuffers {
        input,
        expert_ids,
        output,
        scratch: DenseMatmulIdScratch::new(device, shape.n_experts, shape.m)?,
        output_elements,
    })
}

struct Sample {
    wall_us: f64,
    gpu_us: f64,
    encoded: Vec<EncodedKernelDispatch>,
    pipelines: Vec<KernelPipelineIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationAttemptKind {
    Proof,
    Timing,
}

#[derive(Default)]
struct AttemptLedger {
    submissions: u32,
    dispatches: u32,
    cleanup_submissions: u32,
}

fn commit_cleanup_boundary(device: &MlxDevice, attempts: &mut AttemptLedger) -> Result<()> {
    if attempts.cleanup_submissions != 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id cleanup boundary was requested more than once".into(),
        ));
    }
    let mut cleanup = device.command_encoder()?;
    attempts.submissions = attempts.submissions.checked_add(1).ok_or_else(|| {
        MlxError::InvalidArgument("dense_matmul_id cleanup attempt count overflow".into())
    })?;
    attempts.cleanup_submissions = 1;
    cleanup.commit_and_wait()?;
    #[cfg(test)]
    {
        TEST_CLEANUP_BOUNDARIES.with(|count| count.set(count.get() + 1));
        TEST_LAST_CLEANUP_LEDGER
            .with(|ledger| ledger.set((attempts.submissions, attempts.dispatches)));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestFailurePoint {
    DirectProofAfterCommit = 1,
    GroupedProofAfterCommit = 2,
    CacheEviction = 4,
    DirectTimingInvalidIntervalAfterCommit = 8,
    GroupedTimingInvalidIntervalAfterCommit = 16,
}

#[cfg(test)]
thread_local! {
    static TEST_FAILURE_MASK: Cell<u8> = const { Cell::new(0) };
    static TEST_CLEANUP_BOUNDARIES: Cell<u32> = const { Cell::new(0) };
    static TEST_LAST_CLEANUP_LEDGER: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

#[cfg(test)]
fn take_test_failure(point: TestFailurePoint) -> bool {
    TEST_FAILURE_MASK.with(|mask| {
        let bit = point as u8;
        let current = mask.get();
        let active = current & bit != 0;
        mask.set(current & !bit);
        active
    })
}

#[cfg(test)]
struct TestFailureGuard;

#[cfg(test)]
impl Drop for TestFailureGuard {
    fn drop(&mut self) {
        TEST_FAILURE_MASK.with(|current| current.set(0));
    }
}

#[cfg(test)]
fn set_test_failures(points: &[TestFailurePoint]) -> TestFailureGuard {
    let mask = points.iter().fold(0u8, |mask, point| mask | *point as u8);
    TEST_FAILURE_MASK.with(|current| current.set(mask));
    TEST_CLEANUP_BOUNDARIES.with(|count| count.set(0));
    TEST_LAST_CLEANUP_LEDGER.with(|ledger| ledger.set((0, 0)));
    TestFailureGuard
}

fn return_after_attempt_cleanup<T>(
    device: &MlxDevice,
    attempts: &mut AttemptLedger,
    pending_evictions: &[(ProcessKey, Arc<CalibrationCell>)],
    error: MlxError,
) -> Result<T> {
    let mut diagnostics = Vec::new();
    if attempts.submissions != 0 && attempts.cleanup_submissions == 0 {
        if let Err(cleanup_error) = commit_cleanup_boundary(device, attempts) {
            diagnostics.push(format!(
                "dense_matmul_id cleanup boundary also failed: {cleanup_error}"
            ));
        }
    }
    for (key, cell) in pending_evictions {
        if let Err(eviction_error) = evict_cell_if_same(key, cell) {
            diagnostics.push(format!(
                "dense_matmul_id deferred cache eviction also failed: {eviction_error}"
            ));
        }
    }
    if diagnostics.is_empty() {
        Err(error)
    } else {
        Err(MlxError::InvalidArgument(format!(
            "{error}; {}",
            diagnostics.join("; ")
        )))
    }
}

fn expected_dispatches(
    shape: DenseMatmulIdShape,
    route: DenseMatmulIdRoute,
) -> Result<Vec<EncodedKernelDispatch>> {
    let (direct, grouped) = pipeline_names(shape.weight_dtype)?;
    Ok(match route {
        DenseMatmulIdRoute::Direct => vec![EncodedKernelDispatch {
            pipeline_label: direct.to_string(),
            dispatch_kind: DispatchKind::ThreadGroups,
            grid: [
                (u64::from(shape.n) + 7) / 8,
                u64::from(shape.m) * u64::from(shape.top_k),
                1,
            ],
            threads_per_threadgroup: [64, 1, 1],
            threadgroup_memory: Vec::new(),
        }],
        DenseMatmulIdRoute::GroupedPrefill => vec![
            EncodedKernelDispatch {
                pipeline_label: "dense_matmul_id_map_distinct".to_string(),
                dispatch_kind: DispatchKind::ThreadGroups,
                grid: [1, 1, 1],
                threads_per_threadgroup: [u64::from(shape.n_experts), 1, 1],
                threadgroup_memory: Vec::new(),
            },
            EncodedKernelDispatch {
                pipeline_label: grouped.to_string(),
                dispatch_kind: DispatchKind::ThreadGroups,
                grid: [
                    (u64::from(shape.m) + 7) / 8,
                    (u64::from(shape.n) + 7) / 8,
                    u64::from(shape.n_experts),
                ],
                threads_per_threadgroup: [256, 1, 1],
                threadgroup_memory: vec![(0, 8 * 128 * DType::F32.size_of() as u64)],
            },
        ],
    })
}

fn identities_for_dispatches(
    registry: &KernelRegistry,
    encoded: &[EncodedKernelDispatch],
) -> Result<Vec<KernelPipelineIdentity>> {
    encoded
        .iter()
        .map(|dispatch| {
            let identity = registry.pipeline_identity(&dispatch.pipeline_label)?;
            if identity.pipeline_label != dispatch.pipeline_label {
                return Err(MlxError::InvalidArgument(format!(
                    "dense_matmul_id pipeline identity mismatch for {}",
                    dispatch.pipeline_label
                )));
            }
            Ok(identity)
        })
        .collect()
}

fn execute_route(
    route: DenseMatmulIdRoute,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    buffers: &CalibrationBuffers,
    shape: DenseMatmulIdShape,
    budget: &mut Budget,
    attempts: &mut AttemptLedger,
    kind: CalibrationAttemptKind,
) -> Result<Sample> {
    let started = Instant::now();
    let mut encoder = device.command_encoder()?;
    encoder.start_encoded_dispatch_receipt(2)?;
    let operation = dense_matmul_id(
        &mut encoder,
        registry,
        device,
        weight,
        &buffers.input,
        &buffers.expert_ids,
        &buffers.output,
        Some(&buffers.scratch),
        &shape.params(route),
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    let receipt = operation?;
    let encoded = encoded?;
    let expected = expected_dispatches(shape, route)?;
    if encoded != expected {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id {route:?} encoded unexpected geometry: {encoded:?}"
        )));
    }
    let pipelines = identities_for_dispatches(registry, &encoded)?;
    // The command buffer becomes an attempt at this exact point. Count it
    // before commit so commit/wait and invalid-timing failures cannot vanish
    // from the cleanup/error authority ledger.
    budget.record_attempt(attempts, receipt.dispatch_count, kind)?;
    let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time()?;
    #[cfg(test)]
    {
        let point = match (route, kind) {
            (DenseMatmulIdRoute::Direct, CalibrationAttemptKind::Proof) => {
                Some(TestFailurePoint::DirectProofAfterCommit)
            }
            (DenseMatmulIdRoute::GroupedPrefill, CalibrationAttemptKind::Proof) => {
                Some(TestFailurePoint::GroupedProofAfterCommit)
            }
            (DenseMatmulIdRoute::Direct, CalibrationAttemptKind::Timing) => {
                Some(TestFailurePoint::DirectTimingInvalidIntervalAfterCommit)
            }
            (DenseMatmulIdRoute::GroupedPrefill, CalibrationAttemptKind::Timing) => {
                Some(TestFailurePoint::GroupedTimingInvalidIntervalAfterCommit)
            }
        };
        if point.is_some_and(take_test_failure) {
            let reason = match kind {
                CalibrationAttemptKind::Proof => {
                    format!("injected dense_matmul_id {route:?} proof failure after commit")
                }
                CalibrationAttemptKind::Timing => format!(
                    "injected dense_matmul_id {route:?} timing invalid interval after commit"
                ),
            };
            return Err(MlxError::InvalidArgument(reason));
        }
    }
    let wall_us = started.elapsed().as_secs_f64() * 1e6;
    let gpu_us = (gpu_end - gpu_start) * 1e6;
    if !wall_us.is_finite() || wall_us <= 0.0 || !gpu_us.is_finite() || gpu_us <= 0.0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id calibration returned an invalid timing interval".into(),
        ));
    }
    Ok(Sample {
        wall_us,
        gpu_us,
        encoded,
        pipelines,
    })
}

struct Budget {
    started: Instant,
    max_elapsed_ms: u64,
    max_submissions: u32,
    submissions: u32,
    dispatches: u32,
    proof_submissions: u32,
    proof_dispatches: u32,
    timing_submissions: u32,
    timing_dispatches: u32,
}

impl Budget {
    fn exhausted(&self) -> bool {
        self.submissions >= self.max_submissions
            || self.started.elapsed().as_secs_f64() * 1000.0 >= self.max_elapsed_ms as f64
    }

    fn record_attempt(
        &mut self,
        attempts: &mut AttemptLedger,
        dispatches: u32,
        kind: CalibrationAttemptKind,
    ) -> Result<()> {
        let submissions = self.submissions.checked_add(1).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id submission count overflow".into())
        })?;
        let all_dispatches = self.dispatches.checked_add(dispatches).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id dispatch count overflow".into())
        })?;
        let ledger_submissions = attempts.submissions.checked_add(1).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id attempt count overflow".into())
        })?;
        let ledger_dispatches = attempts.dispatches.checked_add(dispatches).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id attempt dispatch count overflow".into())
        })?;
        let (proof_submissions, proof_dispatches, timing_submissions, timing_dispatches) =
            match kind {
                CalibrationAttemptKind::Proof => (
                    self.proof_submissions.checked_add(1).ok_or_else(|| {
                        MlxError::InvalidArgument(
                            "dense_matmul_id proof submission count overflow".into(),
                        )
                    })?,
                    self.proof_dispatches
                        .checked_add(dispatches)
                        .ok_or_else(|| {
                            MlxError::InvalidArgument(
                                "dense_matmul_id proof dispatch count overflow".into(),
                            )
                        })?,
                    self.timing_submissions,
                    self.timing_dispatches,
                ),
                CalibrationAttemptKind::Timing => (
                    self.proof_submissions,
                    self.proof_dispatches,
                    self.timing_submissions.checked_add(1).ok_or_else(|| {
                        MlxError::InvalidArgument(
                            "dense_matmul_id timing submission count overflow".into(),
                        )
                    })?,
                    self.timing_dispatches
                        .checked_add(dispatches)
                        .ok_or_else(|| {
                            MlxError::InvalidArgument(
                                "dense_matmul_id timing dispatch count overflow".into(),
                            )
                        })?,
                ),
            };
        self.submissions = submissions;
        self.dispatches = all_dispatches;
        self.proof_submissions = proof_submissions;
        self.proof_dispatches = proof_dispatches;
        self.timing_submissions = timing_submissions;
        self.timing_dispatches = timing_dispatches;
        attempts.submissions = ledger_submissions;
        attempts.dispatches = ledger_dispatches;
        Ok(())
    }
}

fn distribution(mut values: Vec<f64>) -> Result<DenseMatmulIdTimingDistribution> {
    if values.len() != CALIBRATION_SAMPLES {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id timing has {} samples, expected {CALIBRATION_SAMPLES}",
            values.len()
        )));
    }
    values.sort_by(f64::total_cmp);
    Ok(DenseMatmulIdTimingDistribution {
        p25_us: values[(values.len() - 1) / 4],
        median_us: values[values.len() / 2],
        p75_us: values[(values.len() - 1) * 3 / 4],
        samples: values.len() as u32,
    })
}

fn fallback_decision(
    shape: DenseMatmulIdShape,
    status: DenseMatmulIdSelectionStatus,
    budget: &Budget,
    reason: Option<String>,
) -> DenseMatmulIdCalibrationDecision {
    DenseMatmulIdCalibrationDecision {
        shape,
        declared_weight_identities: 1,
        theorem_authorized_weight_identities: 0,
        selected_route: DenseMatmulIdRoute::Direct,
        status,
        timings: Vec::new(),
        process_cache_hit: false,
        empirical_shape_proof_submissions: budget.proof_submissions,
        empirical_shape_proof_dispatches: budget.proof_dispatches,
        current_timing_submissions: budget.timing_submissions,
        current_timing_dispatches: budget.timing_dispatches,
        cached_timing_submissions: 0,
        cached_timing_dispatches: 0,
        calibration_submissions: budget.submissions,
        calibration_dispatches: budget.dispatches,
        fallback_reason: reason,
    }
}

fn calibrate_one_inner(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    shape: DenseMatmulIdShape,
    grouped_legal: bool,
    grouped_preparation_error: Option<&str>,
    budget: &mut Budget,
    attempts: &mut AttemptLedger,
) -> Result<DenseMatmulIdCalibrationDecision> {
    if let Some(reason) = grouped_preparation_error {
        return Ok(fallback_decision(
            shape,
            DenseMatmulIdSelectionStatus::ErrorFallback,
            budget,
            Some(reason.to_string()),
        ));
    }
    if !grouped_legal {
        return Ok(fallback_decision(
            shape,
            DenseMatmulIdSelectionStatus::DirectOnly,
            budget,
            None,
        ));
    }
    let submissions_per_profile = 2 + (CALIBRATION_SAMPLES as u32 * 2);
    let required_submissions = submissions_per_profile * ROUTING_PROFILES.len() as u32;
    // Both routing distributions are one indivisible authority gate. Refuse to
    // begin when the declared budget cannot prove and time them completely.
    if budget.max_submissions < required_submissions || budget.exhausted() {
        return Ok(fallback_decision(
            shape,
            DenseMatmulIdSelectionStatus::BudgetFallback,
            budget,
            Some("insufficient activation budget for a complete AB gate".into()),
        ));
    }
    let mut timings = Vec::with_capacity(ROUTING_PROFILES.len() * 2);
    let mut grouped_wins_all = true;
    let mut direct_fastest_any = false;
    for profile in ROUTING_PROFILES {
        let mut buffers = calibration_buffers(device, shape, profile)?;
        poison_output(&mut buffers.output)?;
        let direct_proof = execute_route(
            DenseMatmulIdRoute::Direct,
            registry,
            device,
            weight,
            &buffers,
            shape,
            budget,
            attempts,
            CalibrationAttemptKind::Proof,
        )?;
        let direct_bits = verified_bits(&buffers.output, buffers.output_elements)?;
        if budget.exhausted() {
            return Ok(fallback_decision(
                shape,
                DenseMatmulIdSelectionStatus::BudgetFallback,
                &budget,
                Some(format!(
                    "activation budget ended after {profile:?} Direct proof"
                )),
            ));
        }
        poison_output(&mut buffers.output)?;
        let grouped_proof = match execute_route(
            DenseMatmulIdRoute::GroupedPrefill,
            registry,
            device,
            weight,
            &buffers,
            shape,
            budget,
            attempts,
            CalibrationAttemptKind::Proof,
        ) {
            Ok(proof) => proof,
            Err(error) => {
                return Ok(fallback_decision(
                    shape,
                    DenseMatmulIdSelectionStatus::ErrorFallback,
                    budget,
                    Some(format!(
                        "optional Grouped proof failed after Direct was proven for {profile:?}: {error}"
                    )),
                ));
            }
        };
        let grouped_bits = match verified_bits(&buffers.output, buffers.output_elements) {
            Ok(bits) => bits,
            Err(error) => {
                return Ok(fallback_decision(
                    shape,
                    DenseMatmulIdSelectionStatus::ErrorFallback,
                    budget,
                    Some(format!(
                        "Grouped proof failed after Direct was proven for {profile:?}: {error}"
                    )),
                ));
            }
        };
        if direct_bits != grouped_bits {
            return Ok(fallback_decision(
                shape,
                DenseMatmulIdSelectionStatus::IncoherentGrouped,
                &budget,
                Some(format!(
                    "Grouped output was not bit-identical to Direct for {profile:?} routing"
                )),
            ));
        }
        let mut direct_wall = Vec::with_capacity(CALIBRATION_SAMPLES);
        let mut direct_gpu = Vec::with_capacity(CALIBRATION_SAMPLES);
        let mut grouped_wall = Vec::with_capacity(CALIBRATION_SAMPLES);
        let mut grouped_gpu = Vec::with_capacity(CALIBRATION_SAMPLES);
        for round in 0..CALIBRATION_SAMPLES {
            for route in if round % 2 == 0 {
                [
                    DenseMatmulIdRoute::Direct,
                    DenseMatmulIdRoute::GroupedPrefill,
                ]
            } else {
                [
                    DenseMatmulIdRoute::GroupedPrefill,
                    DenseMatmulIdRoute::Direct,
                ]
            } {
                if budget.exhausted() {
                    return Ok(fallback_decision(
                        shape,
                        DenseMatmulIdSelectionStatus::BudgetFallback,
                        &budget,
                        Some(format!(
                            "activation budget ended during {profile:?} AB timing"
                        )),
                    ));
                }
                let sample = match execute_route(
                    route,
                    registry,
                    device,
                    weight,
                    &buffers,
                    shape,
                    budget,
                    attempts,
                    CalibrationAttemptKind::Timing,
                ) {
                    Ok(sample) => sample,
                    Err(error) if route == DenseMatmulIdRoute::GroupedPrefill => {
                        return Ok(fallback_decision(
                            shape,
                            DenseMatmulIdSelectionStatus::ErrorFallback,
                            budget,
                            Some(format!(
                                "optional Grouped timing failed after Direct was proven for {profile:?}: {error}"
                            )),
                        ));
                    }
                    Err(error) => return Err(error),
                };
                match route {
                    DenseMatmulIdRoute::Direct => {
                        direct_wall.push(sample.wall_us);
                        direct_gpu.push(sample.gpu_us);
                    }
                    DenseMatmulIdRoute::GroupedPrefill => {
                        grouped_wall.push(sample.wall_us);
                        grouped_gpu.push(sample.gpu_us);
                    }
                }
            }
        }
        let direct = DenseMatmulIdRouteTiming {
            profile,
            route: DenseMatmulIdRoute::Direct,
            wall: distribution(direct_wall)?,
            gpu: distribution(direct_gpu)?,
            encoded: direct_proof.encoded,
            pipelines: direct_proof.pipelines,
        };
        let grouped = DenseMatmulIdRouteTiming {
            profile,
            route: DenseMatmulIdRoute::GroupedPrefill,
            wall: distribution(grouped_wall)?,
            gpu: distribution(grouped_gpu)?,
            encoded: grouped_proof.encoded,
            pipelines: grouped_proof.pipelines,
        };
        let material =
            grouped.wall.median_us <= direct.wall.median_us * (1.0 - MATERIAL_WIN_FRACTION);
        let stable = grouped.wall.p75_us < direct.wall.p25_us;
        let no_contrary_gpu =
            grouped.gpu.median_us <= direct.gpu.median_us * (1.0 + GPU_CONTRARY_TOLERANCE);
        grouped_wins_all &= material && stable && no_contrary_gpu;
        direct_fastest_any |= grouped.wall.median_us >= direct.wall.median_us;
        timings.push(direct);
        timings.push(grouped);
    }
    // The final timed submission is indivisible and may cross the wall-clock
    // deadline. Never freeze Grouped from samples completed after authority
    // expired, even when every comparative result otherwise favored it.
    if budget.started.elapsed().as_secs_f64() * 1000.0 >= budget.max_elapsed_ms as f64 {
        return Ok(fallback_decision(
            shape,
            DenseMatmulIdSelectionStatus::BudgetFallback,
            budget,
            Some("activation deadline ended before route selection".into()),
        ));
    }
    let (selected_route, status) = if grouped_wins_all {
        (
            DenseMatmulIdRoute::GroupedPrefill,
            DenseMatmulIdSelectionStatus::CalibratedWinner,
        )
    } else if direct_fastest_any {
        (
            DenseMatmulIdRoute::Direct,
            DenseMatmulIdSelectionStatus::DirectFastest,
        )
    } else {
        (
            DenseMatmulIdRoute::Direct,
            DenseMatmulIdSelectionStatus::NoStableWinner,
        )
    };
    Ok(DenseMatmulIdCalibrationDecision {
        shape,
        declared_weight_identities: 1,
        theorem_authorized_weight_identities: u32::from(
            selected_route == DenseMatmulIdRoute::GroupedPrefill,
        ),
        selected_route,
        status,
        timings,
        process_cache_hit: false,
        empirical_shape_proof_submissions: budget.proof_submissions,
        empirical_shape_proof_dispatches: budget.proof_dispatches,
        current_timing_submissions: budget.timing_submissions,
        current_timing_dispatches: budget.timing_dispatches,
        cached_timing_submissions: 0,
        cached_timing_dispatches: 0,
        calibration_submissions: budget.submissions,
        calibration_dispatches: budget.dispatches,
        fallback_reason: None,
    })
}

fn calibrate_one(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    shape: DenseMatmulIdShape,
    grouped_legal: bool,
    grouped_preparation_error: Option<&str>,
    started: Instant,
    max_elapsed_ms: u64,
    max_submissions: u32,
    attempts: &mut AttemptLedger,
) -> Result<DenseMatmulIdCalibrationDecision> {
    let mut budget = Budget {
        started,
        max_elapsed_ms,
        max_submissions,
        submissions: 0,
        dispatches: 0,
        proof_submissions: 0,
        proof_dispatches: 0,
        timing_submissions: 0,
        timing_dispatches: 0,
    };
    calibrate_one_inner(
        registry,
        device,
        weight,
        shape,
        grouped_legal,
        grouped_preparation_error,
        &mut budget,
        attempts,
    )
}

#[derive(Debug)]
enum RepresentativeShapeProofFailure {
    Budget(String),
    Incoherent(String),
    Grouped(String),
    Required(String),
}

fn require_representative_bit_identity(
    profile: DenseMatmulIdRoutingProfile,
    direct: &[u32],
    grouped: &[u32],
) -> std::result::Result<(), RepresentativeShapeProofFailure> {
    if direct == grouped {
        Ok(())
    } else {
        Err(RepresentativeShapeProofFailure::Incoherent(format!(
            "cached Grouped candidate was not bit-identical to Direct for the exact-shape representative under {profile:?} routing"
        )))
    }
}

/// Re-establish empirical exact-shape authority for a cached Grouped timing
/// winner using one current activation representative. Exactly two routes are run
/// for each of the two declared routing profiles: four submissions and six
/// dispatches when the proof completes.
fn prove_cached_grouped_shape_representative(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    shape: DenseMatmulIdShape,
    budget: &mut Budget,
    attempts: &mut AttemptLedger,
) -> std::result::Result<(), RepresentativeShapeProofFailure> {
    const REQUIRED_PROOF_SUBMISSIONS: u32 = 4;
    let submissions_before = budget.proof_submissions;
    let dispatches_before = budget.proof_dispatches;
    if budget.max_submissions < REQUIRED_PROOF_SUBMISSIONS || budget.exhausted() {
        return Err(RepresentativeShapeProofFailure::Budget(
            "insufficient activation budget for cached-Grouped exact-shape proof".into(),
        ));
    }
    for profile in ROUTING_PROFILES {
        let mut buffers = calibration_buffers(device, shape, profile)
            .map_err(|error| RepresentativeShapeProofFailure::Required(error.to_string()))?;
        if budget.exhausted() {
            return Err(RepresentativeShapeProofFailure::Budget(format!(
                "activation budget ended before representative {profile:?} Direct proof"
            )));
        }
        poison_output(&mut buffers.output)
            .map_err(|error| RepresentativeShapeProofFailure::Required(error.to_string()))?;
        let _direct = execute_route(
            DenseMatmulIdRoute::Direct,
            registry,
            device,
            weight,
            &buffers,
            shape,
            budget,
            attempts,
            CalibrationAttemptKind::Proof,
        )
        .map_err(|error| RepresentativeShapeProofFailure::Required(error.to_string()))?;
        let direct_bits = verified_bits(&buffers.output, buffers.output_elements)
            .map_err(|error| RepresentativeShapeProofFailure::Required(error.to_string()))?;

        if budget.exhausted() {
            return Err(RepresentativeShapeProofFailure::Budget(format!(
                "activation budget ended before representative {profile:?} Grouped proof"
            )));
        }
        poison_output(&mut buffers.output)
            .map_err(|error| RepresentativeShapeProofFailure::Required(error.to_string()))?;
        let _grouped = execute_route(
            DenseMatmulIdRoute::GroupedPrefill,
            registry,
            device,
            weight,
            &buffers,
            shape,
            budget,
            attempts,
            CalibrationAttemptKind::Proof,
        )
        .map_err(|error| RepresentativeShapeProofFailure::Grouped(error.to_string()))?;
        let grouped_bits = verified_bits(&buffers.output, buffers.output_elements)
            .map_err(|error| RepresentativeShapeProofFailure::Grouped(error.to_string()))?;
        require_representative_bit_identity(profile, &direct_bits, &grouped_bits)?;
    }
    debug_assert_eq!(budget.proof_submissions - submissions_before, 4);
    debug_assert_eq!(budget.proof_dispatches - dispatches_before, 6);
    Ok(())
}

fn apply_representative_proof_result(
    decision: &mut DenseMatmulIdCalibrationDecision,
    budget: &Budget,
    proof: std::result::Result<(), RepresentativeShapeProofFailure>,
) -> Result<()> {
    decision.empirical_shape_proof_submissions = decision
        .empirical_shape_proof_submissions
        .checked_add(budget.proof_submissions)
        .ok_or_else(|| MlxError::InvalidArgument("proof submission count overflow".into()))?;
    decision.empirical_shape_proof_dispatches = decision
        .empirical_shape_proof_dispatches
        .checked_add(budget.proof_dispatches)
        .ok_or_else(|| MlxError::InvalidArgument("proof dispatch count overflow".into()))?;
    decision.calibration_submissions = decision
        .calibration_submissions
        .checked_add(budget.submissions)
        .ok_or_else(|| MlxError::InvalidArgument("calibration submission count overflow".into()))?;
    decision.calibration_dispatches = decision
        .calibration_dispatches
        .checked_add(budget.dispatches)
        .ok_or_else(|| MlxError::InvalidArgument("calibration dispatch count overflow".into()))?;
    if let Err(failure) = proof {
        let (status, reason) = match failure {
            RepresentativeShapeProofFailure::Budget(reason) => {
                (DenseMatmulIdSelectionStatus::BudgetFallback, reason)
            }
            RepresentativeShapeProofFailure::Incoherent(reason) => {
                (DenseMatmulIdSelectionStatus::IncoherentGrouped, reason)
            }
            RepresentativeShapeProofFailure::Grouped(reason) => {
                (DenseMatmulIdSelectionStatus::ErrorFallback, reason)
            }
            RepresentativeShapeProofFailure::Required(reason) => {
                return Err(MlxError::InvalidArgument(format!(
                    "dense_matmul_id required Direct/representative proof failed: {reason}"
                )));
            }
        };
        decision.selected_route = DenseMatmulIdRoute::Direct;
        decision.status = status;
        decision.fallback_reason = Some(reason);
    }
    Ok(())
}

fn downgrade_grouped_after_deadline(
    decisions: &mut [DenseMatmulIdCalibrationDecision],
    plan_decisions: &mut HashMap<DenseMatmulIdShape, DenseMatmulIdRoute>,
    elapsed_ms: f64,
    max_elapsed_ms: u64,
) {
    if elapsed_ms <= max_elapsed_ms as f64 {
        return;
    }
    for decision in decisions {
        if decision.selected_route == DenseMatmulIdRoute::GroupedPrefill {
            decision.selected_route = DenseMatmulIdRoute::Direct;
            decision.theorem_authorized_weight_identities = 0;
            decision.status = DenseMatmulIdSelectionStatus::BudgetFallback;
            decision.fallback_reason =
                Some("activation deadline was exceeded before plan publication".into());
            plan_decisions.insert(decision.shape, DenseMatmulIdRoute::Direct);
        }
    }
}

fn receipt_category_counts(
    decisions: &[DenseMatmulIdCalibrationDecision],
) -> Result<(u32, u32, u32)> {
    let mut calibrated = 0u32;
    let mut process_cache_hits = 0u32;
    let mut fallback = 0u32;
    for decision in decisions {
        let category = if decision.process_cache_hit {
            &mut process_cache_hits
        } else {
            match decision.status {
                DenseMatmulIdSelectionStatus::CalibratedWinner
                | DenseMatmulIdSelectionStatus::DirectFastest
                | DenseMatmulIdSelectionStatus::NoStableWinner => &mut calibrated,
                DenseMatmulIdSelectionStatus::DirectOnly
                | DenseMatmulIdSelectionStatus::BudgetFallback
                | DenseMatmulIdSelectionStatus::ErrorFallback
                | DenseMatmulIdSelectionStatus::IncoherentGrouped => &mut fallback,
            }
        };
        *category = category.checked_add(1).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id receipt category count overflow".into())
        })?;
    }
    let categorized = calibrated
        .checked_add(process_cache_hits)
        .and_then(|count| count.checked_add(fallback))
        .ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id receipt category count overflow".into())
        })?;
    let expected = u32::try_from(decisions.len()).map_err(|_| {
        MlxError::InvalidArgument("dense_matmul_id decision count exceeds u32".into())
    })?;
    if categorized != expected {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id receipt categories cover {categorized} of {expected} decisions"
        )));
    }
    Ok((calibrated, process_cache_hits, fallback))
}

fn plan_id(
    build: &str,
    device_registry_id: u64,
    activation_epoch: u64,
    activation_authority_digest: &str,
    decisions: &HashMap<DenseMatmulIdShape, DenseMatmulIdRoute>,
) -> String {
    let mut ordered: Vec<_> = decisions
        .iter()
        .map(|(shape, route)| (*shape, *route))
        .collect();
    ordered.sort_by_key(|(shape, _)| *shape);
    let mut digest = Sha256::new();
    digest.update(build.as_bytes());
    digest.update(device_registry_id.to_le_bytes());
    digest.update(activation_epoch.to_le_bytes());
    digest.update(activation_authority_digest.as_bytes());
    for (shape, route) in ordered {
        digest.update(format!("{shape:?}:{route:?}").as_bytes());
    }
    hex::encode(digest.finalize())
}

fn build_route_plan(
    prepared: &PreparedDenseMatmulIdRoutes,
    device: &MlxDevice,
    activation_epoch: u64,
    activation_authority_digest: &str,
    decisions: HashMap<DenseMatmulIdShape, DenseMatmulIdRoute>,
) -> Arc<DenseMatmulIdRoutePlan> {
    let id = plan_id(
        &prepared.build_fingerprint,
        device.registry_id(),
        activation_epoch,
        activation_authority_digest,
        &decisions,
    );
    Arc::new(DenseMatmulIdRoutePlan {
        plan_id: id,
        build_fingerprint: prepared.build_fingerprint.clone(),
        device_name: device.name(),
        device_registry_id: device.registry_id(),
        activation_epoch,
        activation_authority_digest: activation_authority_digest.to_string(),
        pipeline_set_fingerprint: prepared.pipeline_set_fingerprint.clone(),
        value_independence_theorem_sha256: dense_matmul_id_value_independence_theorem_sha256()
            .to_string(),
        pipeline_identities: prepared.identities.clone(),
        admitted_bases: decisions.keys().copied().map(Into::into).collect(),
        decisions,
    })
}

impl KernelRegistry {
    fn validate_dense_matmul_id_plan_candidates(
        &self,
        device: &MlxDevice,
        plans: &[&DenseMatmulIdRoutePlan],
    ) -> Result<()> {
        if self.dense_matmul_id_auto.frozen_plan.is_some() {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id route plan is already frozen as {}",
                self.dense_matmul_id_auto
                    .frozen_plan
                    .as_ref()
                    .expect("checked")
                    .plan_id
            )));
        }
        let first = plans.first().ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id requires a plan candidate".into())
        })?;
        let (pipeline_set_fingerprint, build_fingerprint) =
            verify_prepared_pipeline_identities(self, &first.pipeline_identities)?;
        for plan in plans {
            if plan.device_registry_id != device.registry_id() || plan.device_name != device.name()
            {
                return Err(MlxError::InvalidArgument(
                    "dense_matmul_id route plan belongs to a different Metal device".into(),
                ));
            }
            if pipeline_set_fingerprint != plan.pipeline_set_fingerprint
                || build_fingerprint != plan.build_fingerprint
                || plan.pipeline_identities != first.pipeline_identities
                || plan.value_independence_theorem_sha256
                    != dense_matmul_id_value_independence_theorem_sha256()
            {
                return Err(MlxError::InvalidArgument(
                    "dense_matmul_id route plan build/pipeline identity mismatch".into(),
                ));
            }
            let expected_id = plan_id(
                &plan.build_fingerprint,
                plan.device_registry_id,
                plan.activation_epoch,
                &plan.activation_authority_digest,
                &plan.decisions,
            );
            let expected_bases: HashSet<_> =
                plan.decisions.keys().copied().map(Into::into).collect();
            if plan.plan_id != expected_id || plan.admitted_bases != expected_bases {
                return Err(MlxError::InvalidArgument(
                    "dense_matmul_id route plan identity/base contract mismatch".into(),
                ));
            }
        }
        Ok(())
    }

    fn install_validated_dense_matmul_id_plan(&mut self, plan: Arc<DenseMatmulIdRoutePlan>) {
        debug_assert!(self.dense_matmul_id_auto.frozen_plan.is_none());
        self.dense_matmul_id_auto.frozen_plan = Some(plan);
    }

    /// Freeze a calibrated plan into another registry for the same live model
    /// activation without repeating timing. The caller must lend the exact
    /// declared logical weight identities again; they are hashed for authority
    /// validation and are not retained by the registry or plan.
    pub fn freeze_dense_matmul_id_plan_for_cases(
        &mut self,
        device: &MlxDevice,
        activation_epoch: u64,
        plan: Arc<DenseMatmulIdRoutePlan>,
        cases: &[DenseMatmulIdCalibrationCase<'_>],
    ) -> Result<()> {
        if activation_epoch == 0 || activation_epoch != plan.activation_epoch {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id worker freeze activation epoch mismatch".into(),
            ));
        }
        let validated = validate_cases(
            cases,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 1,
                max_cases: u32::MAX,
                max_submissions: 1,
            },
        )?;
        let digest = activation_authority_digest(activation_epoch, device, &validated)?;
        if digest != plan.activation_authority_digest {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id worker freeze activation authority mismatch".into(),
            ));
        }
        let declared_shapes: HashSet<_> = validated.iter().map(|case| case.shape).collect();
        let planned_shapes: HashSet<_> = plan.decisions.keys().copied().collect();
        if declared_shapes != planned_shapes {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id worker freeze exact case union mismatch".into(),
            ));
        }
        if let Some(existing) = self.dense_matmul_id_auto.frozen_plan.as_ref() {
            if existing.plan_id == plan.plan_id {
                return Ok(());
            }
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id route plan is already frozen as {}",
                existing.plan_id
            )));
        }
        let prepared = prepare_routes(self, device, &validated)?;
        if prepared.build_fingerprint != plan.build_fingerprint
            || prepared.pipeline_set_fingerprint != plan.pipeline_set_fingerprint
            || prepared.identities != plan.pipeline_identities
        {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id worker freeze build/pipeline identity mismatch".into(),
            ));
        }
        self.validate_dense_matmul_id_plan_candidates(device, &[plan.as_ref()])?;
        self.install_validated_dense_matmul_id_plan(plan);
        Ok(())
    }

    pub fn dense_matmul_id_plan(&self) -> Option<Arc<DenseMatmulIdRoutePlan>> {
        self.dense_matmul_id_auto.frozen_plan.clone()
    }
}

/// Calibrate the exact union of declared native scalar expert cases and freeze
/// it into `registry`. Borrowed weights are not retained in the plan or cache.
pub fn calibrate_dense_matmul_id_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    limits: DenseMatmulIdCalibrationLimits,
    cases: &[DenseMatmulIdCalibrationCase<'_>],
) -> Result<(
    Arc<DenseMatmulIdRoutePlan>,
    DenseMatmulIdCalibrationBatchReceipt,
)> {
    if activation_epoch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id activation epoch must be nonzero".into(),
        ));
    }
    if registry.dense_matmul_id_plan().is_some() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id registry is already frozen; calibration is one-shot".into(),
        ));
    }
    let validated = validate_cases(cases, limits)?;
    let activation_authority_digest =
        activation_authority_digest(activation_epoch, device, &validated)?;
    let started = Instant::now();
    // Prepare and bind every pipeline the declared cases may execute before
    // consulting process-wide timing metadata. A cache hit is authoritative
    // only for this exact compiled pipeline set.
    let prepared = prepare_routes(registry, device, &validated)?;
    let declared_cases = validated.len() as u32;
    let declared_weight_identities = validated.iter().try_fold(0u32, |total, case| {
        let count = u32::try_from(case.weights.len()).map_err(|_| {
            MlxError::InvalidArgument("dense_matmul_id weight identity count exceeds u32".into())
        })?;
        total.checked_add(count).ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id weight identity count overflow".into())
        })
    })?;
    let mut decisions = Vec::with_capacity(validated.len());
    let mut plan_decisions = HashMap::with_capacity(validated.len());
    let mut calibration_submissions = 0u32;
    let mut calibration_dispatches = 0u32;
    let mut empirical_shape_proof_submissions = 0u32;
    let mut empirical_shape_proof_dispatches = 0u32;
    let mut current_timing_submissions = 0u32;
    let mut current_timing_dispatches = 0u32;
    let mut cached_timing_submissions = 0u32;
    let mut cached_timing_dispatches = 0u32;
    let mut attempts = AttemptLedger::default();
    let mut pending_evictions = Vec::<(ProcessKey, Arc<CalibrationCell>)>::new();

    let calibration = (|| -> Result<()> {
        for case in validated {
            let key = ProcessKey {
                build_fingerprint: prepared.build_fingerprint.clone(),
                device_name: device.name(),
                device_registry_id: device.registry_id(),
                pipeline_set_fingerprint: prepared.pipeline_set_fingerprint.clone(),
                shape: case.shape,
            };
            let cell = process_cell(key.clone())?;
            let initialized_here = Cell::new(false);
            let initialized_decision = RefCell::new(None);
            let reserve_cleanup = u32::from(calibration_submissions > 0);
            let remaining = limits
                .max_submissions
                .saturating_sub(calibration_submissions)
                .saturating_sub(reserve_cleanup.max(1));
            let cached_entry = cell.get_or_init(|| {
                initialized_here.set(true);
                match calibrate_one(
                    registry,
                    device,
                    case.weights[0],
                    case.shape,
                    case.grouped_legal,
                    prepared
                        .grouped_preparation_errors
                        .get(&case.shape.weight_dtype)
                        .map(String::as_str),
                    started,
                    limits.max_elapsed_ms,
                    remaining,
                    &mut attempts,
                ) {
                    Ok(decision) => {
                        *initialized_decision.borrow_mut() = Some(decision.clone());
                        CachedRouteEntry::Ready(CachedRouteMetadata::from_decision(&decision))
                    }
                    Err(error) => CachedRouteEntry::Failed(error.to_string()),
                }
            });
            let cached = match cached_entry {
                CachedRouteEntry::Ready(cached) => cached,
                CachedRouteEntry::Failed(reason) => {
                    pending_evictions.push((key.clone(), cell.clone()));
                    return Err(MlxError::InvalidArgument(format!(
                        "dense_matmul_id required calibration failed: {reason}"
                    )));
                }
            };
            let mut decision = if initialized_here.get() {
                initialized_decision.into_inner().ok_or_else(|| {
                    MlxError::InvalidArgument(
                        "dense_matmul_id process cache initializer lost its decision".into(),
                    )
                })?
            } else {
                let mut decision = cached.activation_decision();
                if decision.selected_route == DenseMatmulIdRoute::GroupedPrefill {
                    let mut proof_budget = Budget {
                        started,
                        max_elapsed_ms: limits.max_elapsed_ms,
                        max_submissions: remaining,
                        submissions: 0,
                        dispatches: 0,
                        proof_submissions: 0,
                        proof_dispatches: 0,
                        timing_submissions: 0,
                        timing_dispatches: 0,
                    };
                    let proof = prove_cached_grouped_shape_representative(
                        registry,
                        device,
                        case.weights[0],
                        case.shape,
                        &mut proof_budget,
                        &mut attempts,
                    );
                    apply_representative_proof_result(&mut decision, &proof_budget, proof)?;
                } else if started.elapsed().as_secs_f64() * 1000.0 >= limits.max_elapsed_ms as f64 {
                    decision.status = DenseMatmulIdSelectionStatus::BudgetFallback;
                    decision.fallback_reason = Some(
                        "activation budget ended before applying cached route metadata".into(),
                    );
                }
                decision
            };
            decision.declared_weight_identities = case.weights.len() as u32;
            decision.theorem_authorized_weight_identities =
                if decision.selected_route == DenseMatmulIdRoute::GroupedPrefill {
                    decision.declared_weight_identities
                } else {
                    0
                };

            if initialized_here.get() {
                // Inspect the representative calibration outcome, not the
                // final activation decision. A representative shape proof may
                // downgrade this activation to Direct, but that must not poison
                // otherwise reusable pointer-free timing metadata.
                if matches!(
                    cached.status,
                    DenseMatmulIdSelectionStatus::BudgetFallback
                        | DenseMatmulIdSelectionStatus::ErrorFallback
                        | DenseMatmulIdSelectionStatus::IncoherentGrouped
                ) {
                    // These outcomes are activation-local and carry no reusable
                    // route authority for a future activation.
                    pending_evictions.push((key.clone(), cell.clone()));
                }
            }

            calibration_submissions = calibration_submissions
                .checked_add(decision.calibration_submissions)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("calibration submissions overflow".into())
                })?;
            calibration_dispatches = calibration_dispatches
                .checked_add(decision.calibration_dispatches)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("calibration dispatches overflow".into())
                })?;
            empirical_shape_proof_submissions = empirical_shape_proof_submissions
                .checked_add(decision.empirical_shape_proof_submissions)
                .ok_or_else(|| MlxError::InvalidArgument("proof submissions overflow".into()))?;
            empirical_shape_proof_dispatches = empirical_shape_proof_dispatches
                .checked_add(decision.empirical_shape_proof_dispatches)
                .ok_or_else(|| MlxError::InvalidArgument("proof dispatches overflow".into()))?;
            current_timing_submissions = current_timing_submissions
                .checked_add(decision.current_timing_submissions)
                .ok_or_else(|| MlxError::InvalidArgument("timing submissions overflow".into()))?;
            current_timing_dispatches = current_timing_dispatches
                .checked_add(decision.current_timing_dispatches)
                .ok_or_else(|| MlxError::InvalidArgument("timing dispatches overflow".into()))?;
            cached_timing_submissions = cached_timing_submissions
                .checked_add(decision.cached_timing_submissions)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("cached timing submissions overflow".into())
                })?;
            cached_timing_dispatches = cached_timing_dispatches
                .checked_add(decision.cached_timing_dispatches)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("cached timing dispatches overflow".into())
                })?;
            plan_decisions.insert(case.shape, decision.selected_route);
            decisions.push(decision);
        }
        if calibration_submissions != attempts.submissions
            || calibration_dispatches != attempts.dispatches
        {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id aggregate attempt receipt is inconsistent".into(),
            ));
        }
        if calibration_submissions > 0 && calibration_submissions >= limits.max_submissions {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id calibration left no cleanup submission budget".into(),
            ));
        }
        Ok(())
    })();
    if let Err(error) = calibration {
        return return_after_attempt_cleanup(device, &mut attempts, &pending_evictions, error);
    }
    if attempts.submissions > 0 {
        if let Err(error) = commit_cleanup_boundary(device, &mut attempts) {
            return return_after_attempt_cleanup(device, &mut attempts, &pending_evictions, error);
        }
    }
    for (key, cell) in &pending_evictions {
        evict_cell_if_same(key, cell)?;
    }
    decisions.sort_by_key(|decision| decision.shape);
    let mut deadline_decisions = decisions.clone();
    let mut deadline_plan_decisions = plan_decisions.clone();
    downgrade_grouped_after_deadline(
        &mut deadline_decisions,
        &mut deadline_plan_decisions,
        f64::INFINITY,
        limits.max_elapsed_ms,
    );
    let primary_plan = build_route_plan(
        &prepared,
        device,
        activation_epoch,
        &activation_authority_digest,
        plan_decisions,
    );
    let deadline_plan = build_route_plan(
        &prepared,
        device,
        activation_epoch,
        &activation_authority_digest,
        deadline_plan_decisions,
    );
    let primary_category_counts = receipt_category_counts(&decisions)?;
    let deadline_category_counts = receipt_category_counts(&deadline_decisions)?;
    // Validate both possible publication outcomes before the authoritative
    // deadline sample. No pipeline lookup, compilation, hashing, or other
    // fallible work may occur between that sample and the plan install.
    registry.validate_dense_matmul_id_plan_candidates(
        device,
        &[primary_plan.as_ref(), deadline_plan.as_ref()],
    )?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let deadline_exceeded = elapsed_ms > limits.max_elapsed_ms as f64;
    let (plan, decisions, category_counts) = if deadline_exceeded {
        (deadline_plan, deadline_decisions, deadline_category_counts)
    } else {
        (primary_plan, decisions, primary_category_counts)
    };
    registry.install_validated_dense_matmul_id_plan(plan.clone());
    debug_assert!(
        !deadline_exceeded
            || plan
                .decisions
                .values()
                .all(|route| *route != DenseMatmulIdRoute::GroupedPrefill)
    );
    let (calibrated_decisions, process_cache_hits, fallback_decisions) = category_counts;
    let theorem_authorized_weight_identities =
        decisions.iter().try_fold(0u32, |total, decision| {
            total
                .checked_add(decision.theorem_authorized_weight_identities)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("theorem-authorized identity count overflow".into())
                })
        })?;
    let id = plan.plan_id.clone();
    Ok((
        plan,
        DenseMatmulIdCalibrationBatchReceipt {
            schema_version: DENSE_MATMUL_ID_ROUTE_SCHEMA_VERSION,
            mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: prepared.build_fingerprint,
            pipeline_set_fingerprint: prepared.pipeline_set_fingerprint,
            pipeline_identities: prepared.identities,
            plan_id: id,
            activation_epoch,
            activation_authority_digest,
            device_name: device.name(),
            device_registry_id: device.registry_id(),
            declared_cases,
            declared_weight_identities,
            theorem_authorized_weight_identities,
            value_independence_theorem_sha256: dense_matmul_id_value_independence_theorem_sha256()
                .to_string(),
            calibrated_decisions,
            process_cache_hits,
            fallback_decisions,
            empirical_shape_proof_submissions,
            empirical_shape_proof_dispatches,
            current_timing_submissions,
            current_timing_dispatches,
            cached_timing_submissions,
            cached_timing_dispatches,
            cleanup_submissions: attempts.cleanup_submissions,
            calibration_submissions: attempts.submissions,
            calibration_dispatches: attempts.dispatches,
            elapsed_ms,
            deadline_overrun_ms: (elapsed_ms - limits.max_elapsed_ms as f64).max(0.0),
            decisions,
        },
    ))
}

fn validate_resolved_weight(
    weights: &MlxBuffer,
    shape: DenseMatmulIdShape,
    route: DenseMatmulIdRoute,
) -> Result<()> {
    let capability = dense_matmul_id_capability(shape.weight_dtype, &shape.params(route))?;
    if weights.data_byte_len() < capability.required_weight_bytes
        || weights.byte_offset() % shape.weight_dtype.size_of() as u64 != 0
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id auto weight does not satisfy resolved {shape:?} {route:?} route"
        )));
    }
    Ok(())
}

/// Resolve and validate an exact declared shape, or Direct for an unseen width
/// whose full width-independent base was admitted by the frozen activation
/// plan. This performs no encoder mutation or GPU submission, so callers may
/// fail before adding a barrier/commit to an existing graph.
/// Undeclared bases, insufficient weight views, and activation/device/theorem
/// mismatches fail closed.
pub fn resolve_dense_matmul_id_auto_route(
    registry: &KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    weights: &MlxBuffer,
    params: &DenseMatmulIdParams,
) -> Result<(DenseMatmulIdRoute, DenseMatmulIdDecisionSource)> {
    let plan = registry
        .dense_matmul_id_auto
        .frozen_plan
        .as_ref()
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "dense_matmul_id auto routing requires a frozen activation plan".into(),
            )
        })?;
    if plan.activation_epoch != activation_epoch
        || plan.device_registry_id != device.registry_id()
        || plan.device_name != device.name()
        || plan.value_independence_theorem_sha256
            != dense_matmul_id_value_independence_theorem_sha256()
    {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id activation plan epoch/device/theorem mismatch".into(),
        ));
    }
    let shape = shape_from_call(weights, params);
    if let Some(route) = plan.decisions.get(&shape).copied() {
        validate_resolved_weight(weights, shape, route)?;
        return Ok((route, DenseMatmulIdDecisionSource::FrozenPlan));
    }
    if plan.admitted_bases.contains(&shape.into()) {
        // M is deliberately absent from the admitted base. Revalidate the
        // exact unseen width against the Direct primitive before granting the
        // compatibility route; Grouped is never inferred for an unseen M.
        validate_resolved_weight(weights, shape, DenseMatmulIdRoute::Direct)?;
        return Ok((
            DenseMatmulIdRoute::Direct,
            DenseMatmulIdDecisionSource::UndeclaredDirect,
        ));
    }
    Err(MlxError::InvalidArgument(format!(
        "dense_matmul_id base was not admitted at activation: {shape:?}"
    )))
}

/// Encode one exact declared expert case through its frozen route, or an
/// admitted-base unseen width through Direct. Undeclared bases and activation
/// epoch/device mismatches fail closed.
#[allow(clippy::too_many_arguments)]
pub fn dense_matmul_id_auto(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    weights: &MlxBuffer,
    input: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: Option<&DenseMatmulIdScratch>,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdAutoDispatchReceipt> {
    let (route, decision_source) =
        resolve_dense_matmul_id_auto_route(registry, device, activation_epoch, weights, params)?;
    let effective = DenseMatmulIdParams { route, ..*params };
    let primitive = dense_matmul_id(
        encoder, registry, device, weights, input, expert_ids, output, scratch, &effective,
    )?;
    Ok(DenseMatmulIdAutoDispatchReceipt {
        route,
        decision_source,
        activation_epoch,
        primitive,
    })
}

fn finish_dispatch_trace(
    registry: &KernelRegistry,
    device: &MlxDevice,
    shape: DenseMatmulIdShape,
    route: DenseMatmulIdRoute,
    encoded: Vec<EncodedKernelDispatch>,
    decision_source: Option<DenseMatmulIdDecisionSource>,
    plan: Option<&DenseMatmulIdRoutePlan>,
) -> Result<DenseMatmulIdDispatchTrace> {
    let expected = expected_dispatches(shape, route)?;
    if encoded != expected {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id trace encoded unexpected geometry: {encoded:?}"
        )));
    }
    let pipelines = identities_for_dispatches(registry, &encoded)?;
    let (pipeline_set_fingerprint, _) = fingerprint_pipeline_identities(&pipelines)?;
    Ok(DenseMatmulIdDispatchTrace {
        schema_version: DENSE_MATMUL_ID_ROUTE_SCHEMA_VERSION,
        mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
        device_name: device.name(),
        device_registry_id: device.registry_id(),
        shape,
        route,
        decision_source,
        pipeline_set_fingerprint,
        encoded,
        pipelines,
        plan_id: plan.map(|plan| plan.plan_id.clone()),
        plan_build_fingerprint: plan.map(|plan| plan.build_fingerprint.clone()),
        plan_pipeline_set_fingerprint: plan.map(|plan| plan.pipeline_set_fingerprint.clone()),
        plan_value_independence_theorem_sha256: plan
            .map(|plan| plan.value_independence_theorem_sha256.clone()),
        plan_activation_authority_digest: plan.map(|plan| plan.activation_authority_digest.clone()),
        activation_epoch: plan.map(|plan| plan.activation_epoch),
    })
}

/// Trace one explicit native scalar expert-ID route. This evidence path
/// allocates receipt metadata; ordinary dispatch does not.
#[allow(clippy::too_many_arguments)]
pub fn trace_dense_matmul_id(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &MlxBuffer,
    input: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: Option<&DenseMatmulIdScratch>,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdDispatchTrace> {
    if encoder.device_registry_id() != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id trace encoder/device mismatch".into(),
        ));
    }
    encoder.start_encoded_dispatch_receipt(2)?;
    let operation = dense_matmul_id(
        encoder, registry, device, weights, input, expert_ids, output, scratch, params,
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    let receipt = operation?;
    let encoded = encoded?;
    let shape = shape_from_call(weights, params);
    if receipt.route != params.route || receipt.dispatch_count as usize != encoded.len() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id trace receipt disagrees with encoded dispatches".into(),
        ));
    }
    finish_dispatch_trace(registry, device, shape, receipt.route, encoded, None, None)
}

/// Trace one activation-declared native scalar expert-ID call through its
/// frozen plan and bind the result to the exact plan and compiled pipelines.
#[allow(clippy::too_many_arguments)]
pub fn trace_dense_matmul_id_auto(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    weights: &MlxBuffer,
    input: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: Option<&DenseMatmulIdScratch>,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdDispatchTrace> {
    if encoder.device_registry_id() != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id auto trace encoder/device mismatch".into(),
        ));
    }
    encoder.start_encoded_dispatch_receipt(2)?;
    let operation = dense_matmul_id_auto(
        encoder,
        registry,
        device,
        activation_epoch,
        weights,
        input,
        expert_ids,
        output,
        scratch,
        params,
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    let receipt = operation?;
    let encoded = encoded?;
    let shape = shape_from_call(weights, params);
    if receipt.primitive.dispatch_count as usize != encoded.len() {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id auto trace receipt disagrees with encoded dispatches".into(),
        ));
    }
    let plan = registry
        .dense_matmul_id_auto
        .frozen_plan
        .clone()
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id plan disappeared".into()))?;
    finish_dispatch_trace(
        registry,
        device,
        shape,
        receipt.route,
        encoded,
        Some(receipt.decision_source),
        Some(&plan),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::KernelPipelineOrigin;

    fn decision(
        shape: DenseMatmulIdShape,
        route: DenseMatmulIdRoute,
        status: DenseMatmulIdSelectionStatus,
        process_cache_hit: bool,
    ) -> DenseMatmulIdCalibrationDecision {
        DenseMatmulIdCalibrationDecision {
            shape,
            declared_weight_identities: 1,
            theorem_authorized_weight_identities: 0,
            selected_route: route,
            status,
            timings: Vec::new(),
            process_cache_hit,
            empirical_shape_proof_submissions: 0,
            empirical_shape_proof_dispatches: 0,
            current_timing_submissions: 0,
            current_timing_dispatches: 0,
            cached_timing_submissions: 20,
            cached_timing_dispatches: 30,
            calibration_submissions: 0,
            calibration_dispatches: 0,
            fallback_reason: None,
        }
    }

    fn test_shape() -> DenseMatmulIdShape {
        DenseMatmulIdShape {
            weight_dtype: DType::BF16,
            m: 9,
            n: 11,
            k: 37,
            top_k: 6,
            n_experts: 8,
            expert_stride_bytes: 832,
            input_layout: DenseMatmulIdInputLayout::SharedPerToken,
            id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        }
    }

    fn identity(source_sha: &str) -> KernelPipelineIdentity {
        KernelPipelineIdentity {
            schema_version: crate::KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION,
            pipeline_label: "dense_matmul_id_direct_bf16_f32".into(),
            kernel_name: "dense_matmul_id_direct_bf16_f32".into(),
            origin: KernelPipelineOrigin::RuntimeSource,
            runtime_source_sha256: Some(source_sha.into()),
            embedded_metallib_sha256: None,
            precise_fp32_math: true,
            threadgroup_size_multiple_hint: false,
        }
    }

    #[test]
    fn shader_routes_share_the_canonical_address_widen_madd_helpers() {
        let shader = include_str!("../shaders/dense_matmul_id.metal");
        // One definition plus one Direct and one Grouped call. A route-local
        // replacement changes these counts and, because the complete source is
        // in the build/pipeline fingerprints, invalidates every cached plan.
        assert_eq!(
            shader.matches("dense_matmul_id_input_byte_offset(").count(),
            3
        );
        assert_eq!(
            shader
                .matches("dense_matmul_id_weight_scalar_index(")
                .count(),
            3
        );
        assert_eq!(
            shader
                .matches("dense_matmul_id_expert_byte_offset(")
                .count(),
            3
        );
        assert_eq!(shader.matches("dense_matmul_id_f32_madd(").count(), 3);
        assert_eq!(shader.matches("dense_matmul_id_widen_weight(").count(), 3);
        assert!(shader.contains("return fma(weight, activation, sum);"));
        assert_eq!(
            dense_matmul_id_value_independence_theorem_sha256().len(),
            64
        );
    }

    #[test]
    fn compiled_pipeline_identity_changes_plan_and_process_cache_authority() -> Result<()> {
        let first = identity("first");
        let second = identity("second");
        assert!(fingerprint_pipeline_identities(&[first.clone(), second.clone()]).is_err());
        let (first_pipeline_set, first_build) =
            fingerprint_pipeline_identities(std::slice::from_ref(&first))?;
        let (second_pipeline_set, second_build) =
            fingerprint_pipeline_identities(std::slice::from_ref(&second))?;
        assert_ne!(first_pipeline_set, second_pipeline_set);
        assert_ne!(first_build, second_build);

        let shape = test_shape();
        let first_key = ProcessKey {
            build_fingerprint: first_build,
            device_name: "test".into(),
            device_registry_id: 7,
            pipeline_set_fingerprint: first_pipeline_set,
            shape,
        };
        let second_key = ProcessKey {
            build_fingerprint: second_build,
            device_name: "test".into(),
            device_registry_id: 7,
            pipeline_set_fingerprint: second_pipeline_set,
            shape,
        };
        assert_ne!(first_key, second_key);
        Ok(())
    }

    #[test]
    fn admitted_base_excludes_only_m_and_keys_every_other_contract_field() {
        let shape = test_shape();
        let base = DenseMatmulIdBaseShape::from(shape);
        assert_eq!(
            base,
            DenseMatmulIdBaseShape::from(DenseMatmulIdShape { m: 33, ..shape })
        );
        for mutation in [
            DenseMatmulIdShape {
                weight_dtype: DType::F16,
                ..shape
            },
            DenseMatmulIdShape { n: 12, ..shape },
            DenseMatmulIdShape { k: 38, ..shape },
            DenseMatmulIdShape { top_k: 5, ..shape },
            DenseMatmulIdShape {
                n_experts: 9,
                ..shape
            },
            DenseMatmulIdShape {
                expert_stride_bytes: 834,
                ..shape
            },
            DenseMatmulIdShape {
                input_layout: DenseMatmulIdInputLayout::Slotted,
                ..shape
            },
            DenseMatmulIdShape {
                id_multiplicity: DenseMatmulIdMultiplicity::MayRepeat,
                ..shape
            },
        ] {
            assert_ne!(base, DenseMatmulIdBaseShape::from(mutation));
        }
    }

    #[test]
    fn cached_grouped_representative_proof_counts_and_mismatch_are_local() {
        let shape = test_shape();
        let cached = CachedRouteMetadata {
            shape,
            selected_route: DenseMatmulIdRoute::GroupedPrefill,
            status: DenseMatmulIdSelectionStatus::CalibratedWinner,
            timings: Vec::new(),
            timing_submissions: 20,
            timing_dispatches: 30,
            fallback_reason: None,
        };
        let mut accepted = cached.activation_decision();
        let proof_budget = Budget {
            started: Instant::now(),
            max_elapsed_ms: 1_000,
            max_submissions: 4,
            submissions: 4,
            dispatches: 6,
            proof_submissions: 4,
            proof_dispatches: 6,
            timing_submissions: 0,
            timing_dispatches: 0,
        };
        apply_representative_proof_result(&mut accepted, &proof_budget, Ok(())).unwrap();
        assert_eq!(accepted.selected_route, DenseMatmulIdRoute::GroupedPrefill);
        assert_eq!(accepted.empirical_shape_proof_submissions, 4);
        assert_eq!(accepted.empirical_shape_proof_dispatches, 6);
        assert_eq!(accepted.current_timing_submissions, 0);
        assert_eq!(accepted.cached_timing_submissions, 20);
        assert_eq!(accepted.cached_timing_dispatches, 30);

        let mut rejected = cached.activation_decision();
        let mismatch = require_representative_bit_identity(
            DenseMatmulIdRoutingProfile::MaximallySkewedDistinct,
            &[1, 2, 3],
            &[1, 2, 4],
        );
        let mismatch_budget = Budget {
            started: Instant::now(),
            max_elapsed_ms: 1_000,
            max_submissions: 4,
            submissions: 4,
            dispatches: 6,
            proof_submissions: 4,
            proof_dispatches: 6,
            timing_submissions: 0,
            timing_dispatches: 0,
        };
        apply_representative_proof_result(&mut rejected, &mismatch_budget, mismatch).unwrap();
        assert_eq!(rejected.selected_route, DenseMatmulIdRoute::Direct);
        assert_eq!(
            rejected.status,
            DenseMatmulIdSelectionStatus::IncoherentGrouped
        );
        assert_eq!(rejected.empirical_shape_proof_submissions, 4);
        assert_eq!(rejected.theorem_authorized_weight_identities, 0);
        assert_eq!(rejected.cached_timing_submissions, 20);
        assert_eq!(cached.selected_route, DenseMatmulIdRoute::GroupedPrefill);
        assert_eq!(
            cached.status,
            DenseMatmulIdSelectionStatus::CalibratedWinner
        );
    }

    #[test]
    fn post_cleanup_deadline_downgrades_grouped_and_categories_are_disjoint() {
        let shape = test_shape();
        let direct_shape = DenseMatmulIdShape { m: 2, ..shape };
        let incoherent_shape = DenseMatmulIdShape { m: 3, ..shape };
        let direct_only_shape = DenseMatmulIdShape { m: 4, ..shape };
        let cached_shape = DenseMatmulIdShape { m: 5, ..shape };
        let mut decisions = vec![
            decision(
                shape,
                DenseMatmulIdRoute::GroupedPrefill,
                DenseMatmulIdSelectionStatus::CalibratedWinner,
                false,
            ),
            decision(
                direct_shape,
                DenseMatmulIdRoute::Direct,
                DenseMatmulIdSelectionStatus::DirectFastest,
                false,
            ),
            decision(
                incoherent_shape,
                DenseMatmulIdRoute::Direct,
                DenseMatmulIdSelectionStatus::IncoherentGrouped,
                false,
            ),
            decision(
                direct_only_shape,
                DenseMatmulIdRoute::Direct,
                DenseMatmulIdSelectionStatus::DirectOnly,
                false,
            ),
            decision(
                cached_shape,
                DenseMatmulIdRoute::Direct,
                DenseMatmulIdSelectionStatus::ErrorFallback,
                true,
            ),
        ];
        let mut routes = HashMap::from([
            (shape, DenseMatmulIdRoute::GroupedPrefill),
            (direct_shape, DenseMatmulIdRoute::Direct),
            (incoherent_shape, DenseMatmulIdRoute::Direct),
            (direct_only_shape, DenseMatmulIdRoute::Direct),
            (cached_shape, DenseMatmulIdRoute::Direct),
        ]);
        downgrade_grouped_after_deadline(&mut decisions, &mut routes, 100.001, 100);
        assert_eq!(routes[&shape], DenseMatmulIdRoute::Direct);
        assert!(decisions
            .iter()
            .all(|decision| decision.selected_route != DenseMatmulIdRoute::GroupedPrefill));
        assert_eq!(
            decisions[0].status,
            DenseMatmulIdSelectionStatus::BudgetFallback
        );
        let (calibrated, cache_hits, fallback) = receipt_category_counts(&decisions).unwrap();
        assert_eq!(calibrated, 1);
        assert_eq!(cache_hits, 1);
        assert_eq!(fallback, 3);
        assert_eq!(calibrated + cache_hits + fallback, decisions.len() as u32);
        assert!(!matches!(
            decisions[2].status,
            DenseMatmulIdSelectionStatus::CalibratedWinner
                | DenseMatmulIdSelectionStatus::DirectFastest
                | DenseMatmulIdSelectionStatus::NoStableWinner
        ));
    }

    #[cfg(target_vendor = "apple")]
    fn injected_failure_case(
        device: &MlxDevice,
        m: u32,
        n: u32,
        k: u32,
        seed: u32,
    ) -> Result<(MlxBuffer, DenseMatmulIdParams)> {
        let n_experts = 5;
        let expert_stride_bytes = u64::from(n) * u64::from(k) * DType::F32.size_of() as u64;
        let bytes = usize::try_from(u64::from(n_experts) * expert_stride_bytes)
            .map_err(|_| MlxError::InvalidArgument("test weight extent overflow".into()))?;
        let mut weight = device.alloc_buffer(bytes, DType::F32, vec![bytes / 4])?;
        for (index, value) in weight.as_mut_slice::<f32>()?.iter_mut().enumerate() {
            *value = (((index as u32 * 19 + seed * 37) % 257) as f32 - 128.0) / 263.0;
        }
        Ok((
            weight,
            DenseMatmulIdParams {
                m,
                n,
                k,
                top_k: 3,
                n_experts,
                expert_stride_bytes,
                input_layout: DenseMatmulIdInputLayout::SharedPerToken,
                id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
                route: DenseMatmulIdRoute::Direct,
            },
        ))
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn required_direct_failure_after_attempt_cleans_once_and_does_not_freeze() -> Result<()> {
        let _failure = set_test_failures(&[TestFailurePoint::DirectProofAfterCommit]);
        let device = MlxDevice::new()?;
        let (weight, params) = injected_failure_case(&device, 13, 31, 47, 101)?;
        let mut registry = KernelRegistry::new();
        let error = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_001,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &weight,
                params,
            }],
        )
        .expect_err("required Direct attempt failure must abort activation");
        assert!(error
            .to_string()
            .contains("Direct proof failure after commit"));
        assert!(registry.dense_matmul_id_plan().is_none());
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn optional_grouped_failure_after_direct_proof_falls_back_with_exact_attempts() -> Result<()> {
        let _failure = set_test_failures(&[TestFailurePoint::GroupedProofAfterCommit]);
        let device = MlxDevice::new()?;
        let (weight, params) = injected_failure_case(&device, 14, 33, 49, 102)?;
        let shape = shape_from_call(&weight, &params);
        let mut registry = KernelRegistry::new();
        let (plan, receipt) = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_002,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &weight,
                params,
            }],
        )?;
        assert_eq!(plan.route_for(shape), Some(DenseMatmulIdRoute::Direct));
        assert_eq!(
            receipt.decisions[0].status,
            DenseMatmulIdSelectionStatus::ErrorFallback
        );
        assert_eq!(receipt.empirical_shape_proof_submissions, 2);
        assert_eq!(receipt.empirical_shape_proof_dispatches, 3);
        assert_eq!(receipt.cleanup_submissions, 1);
        assert_eq!(receipt.calibration_submissions, 3);
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn required_direct_timing_failure_cleans_once_and_does_not_freeze() -> Result<()> {
        let _failure =
            set_test_failures(&[TestFailurePoint::DirectTimingInvalidIntervalAfterCommit]);
        let device = MlxDevice::new()?;
        let (weight, params) = injected_failure_case(&device, 17, 39, 55, 106)?;
        let mut registry = KernelRegistry::new();
        let error = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_006,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &weight,
                params,
            }],
        )
        .expect_err("required Direct timing failure must abort activation");
        assert!(error
            .to_string()
            .contains("Direct timing invalid interval after commit"));
        assert!(registry.dense_matmul_id_plan().is_none());
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        // Direct+Grouped proof, the failed Direct timing attempt, then one
        // cleanup submission. Dispatches exclude the empty cleanup boundary.
        TEST_LAST_CLEANUP_LEDGER.with(|ledger| assert_eq!(ledger.get(), (4, 4)));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn optional_grouped_timing_failure_falls_back_with_exact_attempts() -> Result<()> {
        let _failure =
            set_test_failures(&[TestFailurePoint::GroupedTimingInvalidIntervalAfterCommit]);
        let device = MlxDevice::new()?;
        let (weight, params) = injected_failure_case(&device, 18, 41, 57, 107)?;
        let shape = shape_from_call(&weight, &params);
        let mut registry = KernelRegistry::new();
        let (plan, receipt) = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_007,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &weight,
                params,
            }],
        )?;
        assert_eq!(plan.route_for(shape), Some(DenseMatmulIdRoute::Direct));
        assert_eq!(
            receipt.decisions[0].status,
            DenseMatmulIdSelectionStatus::ErrorFallback
        );
        assert!(receipt.decisions[0]
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("optional Grouped timing failed")));
        assert_eq!(receipt.empirical_shape_proof_submissions, 2);
        assert_eq!(receipt.empirical_shape_proof_dispatches, 3);
        assert_eq!(receipt.current_timing_submissions, 2);
        assert_eq!(receipt.current_timing_dispatches, 3);
        assert_eq!(receipt.cleanup_submissions, 1);
        assert_eq!(receipt.calibration_submissions, 5);
        assert_eq!(receipt.calibration_dispatches, 6);
        assert_eq!(receipt.calibrated_decisions, 0);
        assert_eq!(receipt.process_cache_hits, 0);
        assert_eq!(receipt.fallback_decisions, 1);
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        TEST_LAST_CLEANUP_LEDGER.with(|ledger| assert_eq!(ledger.get(), (5, 6)));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn cached_grouped_failure_falls_back_without_poisoning_timing_metadata() -> Result<()> {
        let device = MlxDevice::new()?;
        let (seed_weight, params) = injected_failure_case(&device, 16, 37, 53, 104)?;
        let shape = shape_from_call(&seed_weight, &params);
        let mut seed_registry = KernelRegistry::new();
        let (_, seed_receipt) = calibrate_dense_matmul_id_routes(
            &mut seed_registry,
            &device,
            73_004,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &seed_weight,
                params,
            }],
        )?;
        let key = ProcessKey {
            build_fingerprint: seed_receipt.build_fingerprint.clone(),
            device_name: device.name(),
            device_registry_id: device.registry_id(),
            pipeline_set_fingerprint: seed_receipt.pipeline_set_fingerprint.clone(),
            shape,
        };
        let forced_cell = Arc::new(OnceLock::new());
        forced_cell
            .set(CachedRouteEntry::Ready(CachedRouteMetadata {
                shape,
                selected_route: DenseMatmulIdRoute::GroupedPrefill,
                status: DenseMatmulIdSelectionStatus::CalibratedWinner,
                timings: seed_receipt.decisions[0].timings.clone(),
                timing_submissions: 20,
                timing_dispatches: 30,
                fallback_reason: None,
            }))
            .map_err(|_| MlxError::InvalidArgument("failed to seed cached route".into()))?;
        process_cache()
            .lock()
            .map_err(|_| MlxError::InvalidArgument("test cache mutex poisoned".into()))?
            .insert(key.clone(), forced_cell.clone());

        let _failure = set_test_failures(&[TestFailurePoint::GroupedProofAfterCommit]);
        let (current_weight, _) = injected_failure_case(&device, 16, 37, 53, 105)?;
        let mut registry = KernelRegistry::new();
        let (plan, receipt) = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_005,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &current_weight,
                params,
            }],
        )?;
        assert_eq!(plan.route_for(shape), Some(DenseMatmulIdRoute::Direct));
        assert_eq!(receipt.process_cache_hits, 1);
        assert_eq!(
            receipt.decisions[0].status,
            DenseMatmulIdSelectionStatus::ErrorFallback
        );
        assert_eq!(receipt.empirical_shape_proof_submissions, 2);
        assert_eq!(receipt.empirical_shape_proof_dispatches, 3);
        assert_eq!(receipt.cached_timing_submissions, 20);
        assert_eq!(receipt.cleanup_submissions, 1);
        assert_eq!(receipt.calibration_submissions, 3);
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        let retained = process_cache()
            .lock()
            .map_err(|_| MlxError::InvalidArgument("test cache mutex poisoned".into()))?
            .get(&key)
            .cloned()
            .ok_or_else(|| MlxError::InvalidArgument("cached timing metadata vanished".into()))?;
        assert!(Arc::ptr_eq(&retained, &forced_cell));
        assert!(matches!(
            retained.get(),
            Some(CachedRouteEntry::Ready(metadata))
                if metadata.selected_route == DenseMatmulIdRoute::GroupedPrefill
        ));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn deferred_eviction_failure_occurs_after_exactly_one_cleanup_boundary() -> Result<()> {
        let _failure = set_test_failures(&[
            TestFailurePoint::DirectProofAfterCommit,
            TestFailurePoint::CacheEviction,
        ]);
        let device = MlxDevice::new()?;
        let (weight, params) = injected_failure_case(&device, 15, 35, 51, 103)?;
        let mut registry = KernelRegistry::new();
        let error = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            73_003,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: 1,
                max_submissions: 64,
            },
            &[DenseMatmulIdCalibrationCase {
                weight: &weight,
                params,
            }],
        )
        .expect_err("deferred eviction failure must remain fail-closed");
        assert!(error.to_string().contains("cache eviction"));
        assert!(registry.dense_matmul_id_plan().is_none());
        TEST_CLEANUP_BOUNDARIES.with(|count| assert_eq!(count.get(), 1));
        Ok(())
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn cached_grouped_a_b_a_reuses_pointer_free_timing_and_reproves_exact_shape() -> Result<()> {
        fn weight(device: &MlxDevice, seed: u32) -> Result<MlxBuffer> {
            let mut weight = device.alloc_buffer(596, DType::F32, vec![149])?;
            for (index, value) in weight.as_mut_slice::<f32>()?.iter_mut().enumerate() {
                *value = (((index as u32 * 17 + seed * 43) % 251) as f32 - 125.0) / 257.0;
            }
            Ok(weight)
        }

        let device = MlxDevice::new()?;
        let params = DenseMatmulIdParams {
            m: 3,
            n: 5,
            k: 7,
            top_k: 3,
            n_experts: 4,
            expert_stride_bytes: 152,
            input_layout: DenseMatmulIdInputLayout::SharedPerToken,
            id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
            route: DenseMatmulIdRoute::Direct,
        };
        let limits = DenseMatmulIdCalibrationLimits {
            max_elapsed_ms: 5_000,
            max_cases: 1,
            max_submissions: 128,
        };
        let weight_a = weight(&device, 1)?;
        let bytes_a = weight_a.as_slice::<u8>()?.to_vec();
        let mut registry_a = KernelRegistry::new();
        let (_, receipt_a) = calibrate_dense_matmul_id_routes(
            &mut registry_a,
            &device,
            71_001,
            limits,
            &[DenseMatmulIdCalibrationCase {
                weight: &weight_a,
                params,
            }],
        )?;
        let shape = shape_from_call(&weight_a, &params);
        let key = ProcessKey {
            build_fingerprint: receipt_a.build_fingerprint.clone(),
            device_name: device.name(),
            device_registry_id: device.registry_id(),
            pipeline_set_fingerprint: receipt_a.pipeline_set_fingerprint.clone(),
            shape,
        };
        let forced_grouped = CachedRouteMetadata {
            shape,
            selected_route: DenseMatmulIdRoute::GroupedPrefill,
            status: DenseMatmulIdSelectionStatus::CalibratedWinner,
            timings: receipt_a.decisions[0].timings.clone(),
            timing_submissions: 20,
            timing_dispatches: 30,
            fallback_reason: None,
        };
        let cell = Arc::new(OnceLock::new());
        cell.set(CachedRouteEntry::Ready(forced_grouped))
            .map_err(|_| {
                MlxError::InvalidArgument("failed to seed private cached-Grouped test cell".into())
            })?;
        process_cache()
            .lock()
            .map_err(|_| MlxError::InvalidArgument("test cache mutex poisoned".into()))?
            .insert(key, cell);

        let weight_b = weight(&device, 2)?;
        let weight_b_other = weight(&device, 22)?;
        let mut registry_b = KernelRegistry::new();
        let (_, receipt_b) = calibrate_dense_matmul_id_routes(
            &mut registry_b,
            &device,
            71_002,
            limits,
            &[
                DenseMatmulIdCalibrationCase {
                    weight: &weight_b,
                    params,
                },
                DenseMatmulIdCalibrationCase {
                    weight: &weight_b_other,
                    params,
                },
            ],
        )?;
        assert_ne!(bytes_a, weight_b.as_slice::<u8>()?);
        assert_ne!(weight_b.as_slice::<u8>()?, weight_b_other.as_slice::<u8>()?);
        assert_eq!(receipt_b.process_cache_hits, 1);
        assert_eq!(receipt_b.declared_weight_identities, 2);
        assert_eq!(receipt_b.theorem_authorized_weight_identities, 2);
        assert_eq!(receipt_b.decisions[0].declared_weight_identities, 2);
        assert_eq!(
            receipt_b.decisions[0].theorem_authorized_weight_identities,
            2
        );
        assert_eq!(receipt_b.empirical_shape_proof_submissions, 4);
        assert_eq!(receipt_b.empirical_shape_proof_dispatches, 6);
        assert_eq!(receipt_b.current_timing_submissions, 0);
        assert_eq!(receipt_b.cached_timing_submissions, 20);
        assert_eq!(receipt_b.calibration_submissions, 5);
        assert_eq!(
            receipt_b.decisions[0].selected_route,
            DenseMatmulIdRoute::GroupedPrefill
        );

        let weight_a2 = weight(&device, 3)?;
        let mut registry_a2 = KernelRegistry::new();
        let (_, receipt_a2) = calibrate_dense_matmul_id_routes(
            &mut registry_a2,
            &device,
            71_003,
            limits,
            &[DenseMatmulIdCalibrationCase {
                weight: &weight_a2,
                params,
            }],
        )?;
        assert_ne!(weight_b.as_slice::<u8>()?, weight_a2.as_slice::<u8>()?);
        assert_eq!(receipt_a2.process_cache_hits, 1);
        assert_eq!(receipt_a2.theorem_authorized_weight_identities, 1);
        assert_eq!(receipt_a2.empirical_shape_proof_submissions, 4);
        assert_eq!(receipt_a2.empirical_shape_proof_dispatches, 6);
        assert_eq!(receipt_a2.current_timing_submissions, 0);
        assert_eq!(receipt_a2.cached_timing_submissions, 20);
        assert_eq!(receipt_a2.calibration_submissions, 5);
        assert_eq!(
            receipt_a2.decisions[0].selected_route,
            DenseMatmulIdRoute::GroupedPrefill
        );
        Ok(())
    }

    #[test]
    #[ignore = "activation-cost benchmark; run on an idle Apple GPU"]
    #[cfg(target_vendor = "apple")]
    fn many_layer_cached_grouped_activation_cost_receipt() -> Result<()> {
        fn params(m: u32, n: u32, k: u32) -> DenseMatmulIdParams {
            DenseMatmulIdParams {
                m,
                n,
                k,
                top_k: 3,
                n_experts: 4,
                expert_stride_bytes: u64::from(n) * u64::from(k) * 4,
                input_layout: DenseMatmulIdInputLayout::SharedPerToken,
                id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
                route: DenseMatmulIdRoute::Direct,
            }
        }

        fn weight(device: &MlxDevice, params: DenseMatmulIdParams, seed: u32) -> Result<MlxBuffer> {
            let bytes = usize::try_from(u64::from(params.n_experts) * params.expert_stride_bytes)
                .map_err(|_| {
                MlxError::InvalidArgument("benchmark weight extent overflow".into())
            })?;
            let mut weight = device.alloc_buffer(bytes, DType::F32, vec![bytes / 4])?;
            for (index, value) in weight.as_mut_slice::<f32>()?.iter_mut().enumerate() {
                *value = (((index as u32 * 17 + seed * 43) % 251) as f32 - 125.0) / 257.0;
            }
            Ok(weight)
        }

        const LAYERS: usize = 48;
        const SCHEDULER_WIDTHS: [u32; 7] = [1, 2, 3, 4, 8, 9, 33];
        const PROJECTION_DIMS: [(u32, u32); 3] = [(6, 7), (9, 7), (5, 9)];
        const EXACT_SHAPES: usize = SCHEDULER_WIDTHS.len() * PROJECTION_DIMS.len();
        const SHAPE_WEIGHT_AUTHORITIES: usize = LAYERS * EXACT_SHAPES;
        let device = MlxDevice::new()?;
        let representatives = PROJECTION_DIMS
            .iter()
            .enumerate()
            .map(|(index, (n, k))| weight(&device, params(1, *n, *k), index as u32 + 1))
            .collect::<Result<Vec<_>>>()?;
        let mut representative_cases = Vec::with_capacity(EXACT_SHAPES);
        for m in SCHEDULER_WIDTHS {
            for (projection, (n, k)) in PROJECTION_DIMS.iter().copied().enumerate() {
                representative_cases.push(DenseMatmulIdCalibrationCase {
                    weight: &representatives[projection],
                    params: params(m, n, k),
                });
            }
        }
        let mut seed_registry = KernelRegistry::new();
        let (_, seed_receipt) = calibrate_dense_matmul_id_routes(
            &mut seed_registry,
            &device,
            72_001,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: EXACT_SHAPES as u32,
                max_submissions: 1_024,
            },
            &representative_cases,
        )?;
        for decision in &seed_receipt.decisions {
            let cell = Arc::new(OnceLock::new());
            cell.set(CachedRouteEntry::Ready(CachedRouteMetadata {
                shape: decision.shape,
                selected_route: DenseMatmulIdRoute::GroupedPrefill,
                status: DenseMatmulIdSelectionStatus::CalibratedWinner,
                timings: decision.timings.clone(),
                timing_submissions: 20,
                timing_dispatches: 30,
                fallback_reason: None,
            }))
            .map_err(|_| {
                MlxError::InvalidArgument("failed to seed activation-cost cache cell".into())
            })?;
            process_cache()
                .lock()
                .map_err(|_| MlxError::InvalidArgument("benchmark cache mutex poisoned".into()))?
                .insert(
                    ProcessKey {
                        build_fingerprint: seed_receipt.build_fingerprint.clone(),
                        device_name: device.name(),
                        device_registry_id: device.registry_id(),
                        pipeline_set_fingerprint: seed_receipt.pipeline_set_fingerprint.clone(),
                        shape: decision.shape,
                    },
                    cell,
                );
        }

        let mut stacks = Vec::with_capacity(LAYERS * PROJECTION_DIMS.len());
        for layer in 0..LAYERS {
            for (projection, (n, k)) in PROJECTION_DIMS.iter().copied().enumerate() {
                stacks.push(weight(
                    &device,
                    params(1, n, k),
                    100 + (layer * 3 + projection) as u32,
                )?);
            }
        }
        let mut cases = Vec::with_capacity(SHAPE_WEIGHT_AUTHORITIES);
        for m in SCHEDULER_WIDTHS {
            for layer in 0..LAYERS {
                for (projection, (n, k)) in PROJECTION_DIMS.iter().copied().enumerate() {
                    cases.push(DenseMatmulIdCalibrationCase {
                        weight: &stacks[layer * PROJECTION_DIMS.len() + projection],
                        params: params(m, n, k),
                    });
                }
            }
        }
        let mut registry = KernelRegistry::new();
        let wall_started = Instant::now();
        let (_, receipt) = calibrate_dense_matmul_id_routes(
            &mut registry,
            &device,
            72_002,
            DenseMatmulIdCalibrationLimits {
                max_elapsed_ms: 60_000,
                max_cases: EXACT_SHAPES as u32,
                max_submissions: 5_000,
            },
            &cases,
        )?;
        let wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "dense_matmul_id activation-cost spike: layers={LAYERS} projections_per_layer={} scheduler_widths={:?} exact_shapes={} shape_weight_authorities={} proven_authorities={} proof_submissions={} proof_dispatches={} receipt_elapsed_ms={:.3} wall_ms={wall_ms:.3}",
            PROJECTION_DIMS.len(),
            SCHEDULER_WIDTHS,
            receipt.declared_cases,
            receipt.declared_weight_identities,
            receipt.theorem_authorized_weight_identities,
            receipt.empirical_shape_proof_submissions,
            receipt.empirical_shape_proof_dispatches,
            receipt.elapsed_ms,
        );
        assert_eq!(receipt.declared_cases, EXACT_SHAPES as u32);
        assert_eq!(
            receipt.declared_weight_identities,
            SHAPE_WEIGHT_AUTHORITIES as u32
        );
        assert_eq!(
            receipt.theorem_authorized_weight_identities,
            SHAPE_WEIGHT_AUTHORITIES as u32
        );
        assert_eq!(receipt.process_cache_hits, EXACT_SHAPES as u32);
        assert_eq!(
            receipt.empirical_shape_proof_submissions,
            (EXACT_SHAPES * 4) as u32
        );
        assert_eq!(
            receipt.empirical_shape_proof_dispatches,
            (EXACT_SHAPES * 6) as u32
        );
        assert_eq!(receipt.current_timing_submissions, 0);
        assert_eq!(
            receipt.calibration_submissions,
            (EXACT_SHAPES * 4 + 1) as u32
        );
        assert_eq!(receipt.fallback_decisions, 0);
        Ok(())
    }
}
