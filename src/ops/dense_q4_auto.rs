//! Frozen, device-calibrated routing for native Q4_0 dense projections.
//!
//! The 64x32 tensor candidate is never selected from a static model-family or
//! row-count threshold.  A caller declares exact reachable shapes while a
//! model is activated; calibration borrows every distinct native Q4_0 weight
//! reachable by each shape, proves every exact shape/current-weight pair, and
//! freezes an immutable pointer-free plan in the kernel registry. Proof work
//! may share a command buffer, but no weight×shape cross-term is omitted.
//! Missing or late shapes retain the compatibility route.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, DispatchKind, EncodedKernelDispatch};
use crate::error::{MlxError, Result};
use crate::ggml_capability::{GgmlRoutingPolicy, GgmlTensorMmPreference};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml_with_policy, GgmlQuantizedMatmulParams, GgmlType,
};

pub use crate::ops::dense_q4_acceptance::{
    validate_dense_q4_cartesian_acceptance, DenseQ4CartesianAcceptanceRequirements,
};
pub use crate::ops::dense_q4_calibration::calibrate_dense_q4_routes;

pub const DENSE_Q4_ROUTE_SCHEMA_VERSION: u32 = 4;

fn next_registry_authority_id() -> u64 {
    static NEXT_REGISTRY_AUTHORITY_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_REGISTRY_AUTHORITY_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("dense Q4 registry activation authority exhausted")
}

pub(super) const Q4_MM_PIPELINE_INT_CONSTANTS: &[(usize, i32)] = &[(700, 1), (701, 1), (702, 1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4Route {
    CompatibilitySimdgroup,
    CompatibilityTensorV1,
    CompatibilityV2,
    Tensor64x32,
}

impl DenseQ4Route {
    pub(super) fn kernel_name(self) -> &'static str {
        match self {
            Self::CompatibilitySimdgroup => "kernel_mul_mm_q4_0_f32",
            Self::CompatibilityTensorV1 => "kernel_mul_mm_q4_0_tensor_f32",
            Self::CompatibilityV2 => "kernel_mul_mm_q4_0_tensor_v2_f32",
            Self::Tensor64x32 => "kernel_mul_mm_q4_0_tensor_64x32_f32",
        }
    }

    pub(super) fn pipeline_label(self) -> String {
        format!("{}|700:i1|701:i1|702:i1", self.kernel_name())
    }

    fn from_kernel_name(kernel_name: &str) -> Option<Self> {
        [
            Self::CompatibilitySimdgroup,
            Self::CompatibilityTensorV1,
            Self::CompatibilityV2,
            Self::Tensor64x32,
        ]
        .into_iter()
        .find(|route| route.kernel_name() == kernel_name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4InputLayout {
    Contiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4BaseShape {
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub input_layout: DenseQ4InputLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub input_layout: DenseQ4InputLayout,
}

impl DenseQ4BaseShape {
    pub(super) fn with_m(self, m: u32) -> DenseQ4Shape {
        DenseQ4Shape {
            m,
            n: self.n,
            k: self.k,
            batch: self.batch,
            input_layout: self.input_layout,
        }
    }
}

impl DenseQ4Shape {
    pub(super) fn params(self) -> GgmlQuantizedMatmulParams {
        GgmlQuantizedMatmulParams {
            m: self.m,
            n: self.n,
            k: self.k,
            ggml_type: GgmlType::Q4_0,
        }
    }
}

/// One native Q4_0 weight and every row count that can reach it.
///
/// Declare every distinct loaded weight buffer, including multiple layer-local
/// buffers with the same base shape. Activation calibration times one
/// representative per base shape but proves candidate coherence against every
/// distinct current buffer before publishing candidate authority.
pub struct DenseQ4CalibrationCase<'a> {
    pub weight: &'a MlxBuffer,
    pub shape: DenseQ4BaseShape,
    pub reachable_m: &'a [u32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseQ4CalibrationLimits {
    pub max_elapsed_ms: u64,
    pub max_shapes: u32,
}

impl Default for DenseQ4CalibrationLimits {
    fn default() -> Self {
        Self {
            max_elapsed_ms: 15_000,
            max_shapes: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4DecisionSource {
    FrozenPlan,
    NoPlanCompatibilityFallback,
    UndeclaredCompatibilityFallback,
    IneligibleCompatibilityFallback,
    ForcedTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4DispatchDecision {
    pub route: DenseQ4Route,
    pub source: DenseQ4DecisionSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4TimingDistribution {
    pub p25_us: f64,
    pub median_us: f64,
    pub p75_us: f64,
    pub samples: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4RouteTiming {
    pub route: DenseQ4Route,
    pub wall: DenseQ4TimingDistribution,
    pub gpu: DenseQ4TimingDistribution,
    pub encoded: EncodedKernelDispatch,
    pub pipeline: KernelPipelineIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ4SelectionStatus {
    CalibratedWinner,
    CompatibilityFastest,
    NoStableWinner,
    CandidateUnavailable,
    IncoherentCandidate,
    CalibrationErrorFallback,
    BudgetFallback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4CalibrationDecision {
    pub shape: DenseQ4Shape,
    pub selected_route: DenseQ4Route,
    pub status: DenseQ4SelectionStatus,
    pub diagnostic: Option<String>,
    pub timings: Vec<DenseQ4RouteTiming>,
    pub process_cache_hit: bool,
    /// Distinct current activation buffers authorized for this exact shape.
    pub authorized_weight_buffers: u32,
    /// Proof command-buffer attempts, counted immediately before commit.
    pub proof_submissions: u32,
    /// Production-route dispatches encoded by the proof batch.
    pub proof_route_dispatches: u32,
    /// Poison and comparison dispatches encoded by the proof batch.
    pub proof_auxiliary_dispatches: u32,
    /// Caller-owned scratch bytes live for this exact-shape proof batch.
    pub proof_scratch_bytes: u64,
    /// GPU interval spanning the complete proof command buffer.
    pub proof_gpu_us: f64,
    /// Timing command-buffer attempts, counted immediately before commit.
    pub timing_submissions: u32,
    pub calibration_submissions: u32,
}

#[derive(Clone, Debug)]
pub struct DenseQ4RoutePlan {
    pub(super) plan_id: String,
    pub(super) build_fingerprint: String,
    pub(super) device_name: String,
    pub(super) device_registry_id: u64,
    pub(super) registry_authority_id: u64,
    pub(super) activation_epoch: u64,
    pub(super) decisions: HashMap<DenseQ4Shape, DenseQ4Route>,
}

impl DenseQ4RoutePlan {
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    pub fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn registry_authority_id(&self) -> u64 {
        self.registry_authority_id
    }

    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4CalibrationBatchReceipt {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub build_fingerprint: String,
    pub plan_id: String,
    pub activation_epoch: u64,
    pub device_name: String,
    pub device_registry_id: u64,
    pub registry_authority_id: u64,
    pub declared_shapes: u32,
    pub calibrated_decisions: u32,
    pub process_cache_hits: u32,
    pub compatibility_route_decisions: u32,
    /// Total exact-shape/current-weight route authorities actually submitted
    /// and proved. Pre-proof compatibility fallback decisions contribute zero.
    pub authorized_shape_weight_pairs: u32,
    pub proof_submissions: u32,
    pub proof_route_dispatches: u32,
    pub proof_auxiliary_dispatches: u32,
    pub peak_proof_scratch_bytes: u64,
    pub proof_gpu_us: f64,
    pub timing_submissions: u32,
    pub cleanup_submissions: u32,
    pub calibration_submissions: u32,
    pub elapsed_ms: f64,
    pub deadline_overrun_ms: f64,
    pub decisions: Vec<DenseQ4CalibrationDecision>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseQ4DispatchTrace {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub build_fingerprint: String,
    pub plan_id: Option<String>,
    pub shape: DenseQ4Shape,
    pub decision: DenseQ4DispatchDecision,
    pub encoded: EncodedKernelDispatch,
    pub pipeline: KernelPipelineIdentity,
}

pub(crate) struct DenseQ4AutoState {
    pub(super) frozen_plan: Option<Arc<DenseQ4RoutePlan>>,
    registry_authority_id: u64,
}

impl Default for DenseQ4AutoState {
    fn default() -> Self {
        Self {
            frozen_plan: None,
            registry_authority_id: next_registry_authority_id(),
        }
    }
}

impl DenseQ4AutoState {
    pub(super) fn registry_authority_id(&self) -> u64 {
        self.registry_authority_id
    }

    pub(super) fn validate_plan_authority(&self, plan: &DenseQ4RoutePlan) -> Result<()> {
        if plan.registry_authority_id != self.registry_authority_id {
            return Err(MlxError::InvalidArgument(
                "dense Q4 route plan belongs to a different registry activation authority".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn exact_shape(
    params: &GgmlQuantizedMatmulParams,
    batch: u32,
    contiguous_input: bool,
) -> Option<DenseQ4Shape> {
    (params.ggml_type == GgmlType::Q4_0
        && params.m > 8
        && params.n > 0
        && params.k >= 32
        && params.k % 32 == 0
        && batch == 1
        && contiguous_input)
        .then_some(DenseQ4Shape {
            m: params.m,
            n: params.n,
            k: params.k,
            batch,
            input_layout: DenseQ4InputLayout::Contiguous,
        })
}

pub(crate) fn select_route(
    registry: &KernelRegistry,
    device: &MlxDevice,
    params: &GgmlQuantizedMatmulParams,
    batch: u32,
    contiguous_input: bool,
    routing: &GgmlRoutingPolicy,
) -> DenseQ4DispatchDecision {
    let Some(shape) = exact_shape(params, batch, contiguous_input) else {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityV2,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        };
    };
    if routing.dense_tensor_mm != GgmlTensorMmPreference::AutoProbe {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilitySimdgroup,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        };
    }
    if !routing.allow_dense_large_tile_mm {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityTensorV1,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        };
    }
    let Some(plan) = registry.dense_q4_auto.frozen_plan.as_ref() else {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityV2,
            source: DenseQ4DecisionSource::NoPlanCompatibilityFallback,
        };
    };
    if registry
        .dense_q4_auto
        .validate_plan_authority(plan)
        .is_err()
    {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityV2,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        };
    }
    if plan.device_registry_id != device.registry_id() || plan.device_name != device.name() {
        return DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityV2,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        };
    }
    plan.decisions.get(&shape).copied().map_or(
        DenseQ4DispatchDecision {
            route: DenseQ4Route::CompatibilityV2,
            source: DenseQ4DecisionSource::UndeclaredCompatibilityFallback,
        },
        |route| DenseQ4DispatchDecision {
            route,
            source: DenseQ4DecisionSource::FrozenPlan,
        },
    )
}

impl KernelRegistry {
    pub(super) fn validate_dense_q4_plan(
        &mut self,
        device: &MlxDevice,
        plan: &DenseQ4RoutePlan,
    ) -> Result<()> {
        self.dense_q4_auto.validate_plan_authority(plan)?;
        if plan.device_registry_id != device.registry_id() || plan.device_name != device.name() {
            return Err(MlxError::InvalidArgument(
                "dense Q4 route plan belongs to a different Metal device".into(),
            ));
        }
        if let Some(existing) = self.dense_q4_auto.frozen_plan.as_ref() {
            if existing.plan_id == plan.plan_id {
                return Ok(());
            }
            return Err(MlxError::InvalidArgument(format!(
                "dense Q4 route plan is already frozen as {}",
                existing.plan_id
            )));
        }
        let build_fingerprint =
            crate::ops::dense_q4_calibration::current_build_fingerprint(self, device)?;
        if build_fingerprint != plan.build_fingerprint {
            return Err(MlxError::InvalidArgument(
                "dense Q4 route plan build/pipeline identity mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn install_validated_dense_q4_plan(
        &mut self,
        plan: Arc<DenseQ4RoutePlan>,
    ) -> Result<()> {
        if let Some(existing) = self.dense_q4_auto.frozen_plan.as_ref() {
            if existing.plan_id == plan.plan_id {
                return Ok(());
            }
            return Err(MlxError::InvalidArgument(format!(
                "dense Q4 route plan is already frozen as {}",
                existing.plan_id
            )));
        }
        self.dense_q4_auto.frozen_plan = Some(plan);
        Ok(())
    }

    pub(super) fn install_prevalidated_dense_q4_plan(
        &mut self,
        plan: Arc<DenseQ4RoutePlan>,
    ) -> Result<()> {
        // "Prevalidated" skips only the expensive build/pipeline fingerprint
        // recomputation. Registry-local activation authority and one-shot
        // freeze remain release-mode checks at the mutation boundary.
        self.dense_q4_auto.validate_plan_authority(&plan)?;
        self.install_validated_dense_q4_plan(plan)
    }

    /// Freeze one immutable, pointer-free Q4 route plan into this registry.
    pub fn freeze_dense_q4_plan(
        &mut self,
        device: &MlxDevice,
        plan: Arc<DenseQ4RoutePlan>,
    ) -> Result<()> {
        self.validate_dense_q4_plan(device, &plan)?;
        self.install_validated_dense_q4_plan(plan)
    }

    pub fn dense_q4_plan(&self) -> Option<Arc<DenseQ4RoutePlan>> {
        self.dense_q4_auto.frozen_plan.clone()
    }
}

pub(super) fn expected_dispatch(route: DenseQ4Route, shape: DenseQ4Shape) -> EncodedKernelDispatch {
    let (token_tile, shmem_bytes) = match route {
        DenseQ4Route::CompatibilitySimdgroup | DenseQ4Route::CompatibilityTensorV1 => (32, 8192),
        DenseQ4Route::CompatibilityV2 => (128, 4096),
        DenseQ4Route::Tensor64x32 => (32, 4096),
    };
    EncodedKernelDispatch {
        pipeline_label: route.pipeline_label(),
        dispatch_kind: DispatchKind::ThreadGroups,
        grid: [
            u64::from(shape.m).div_ceil(token_tile),
            u64::from(shape.n).div_ceil(64),
            u64::from(shape.batch),
        ],
        threads_per_threadgroup: [128, 1, 1],
        threadgroup_memory: vec![(0, shmem_bytes)],
    }
}

/// Trace one ordinary Q4 dispatch together with its frozen-plan authority.
/// `routing` must be the exact policy that the loaded model will use for
/// production dispatch; the trace never substitutes process defaults.
#[allow(clippy::too_many_arguments)]
pub fn trace_dense_q4_auto(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    routing: &GgmlRoutingPolicy,
) -> Result<DenseQ4DispatchTrace> {
    let shape = exact_shape(params, 1, true).ok_or_else(|| {
        MlxError::InvalidArgument("dense Q4 auto trace requires eligible Q4_0 MM shape".into())
    })?;
    let planned_decision = select_route(registry, device, params, 1, true, routing);
    if encoder.device_registry_id() != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense Q4 trace encoder/device mismatch".into(),
        ));
    }
    encoder.start_encoded_dispatch_receipt(1)?;
    let operation = quantized_matmul_ggml_with_policy(
        encoder, registry, device, input, weight, output, params, routing,
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    operation?;
    let mut encoded = encoded?;
    if encoded.len() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 auto route encoded {} dispatches, expected one",
            encoded.len()
        )));
    }
    let encoded = encoded.remove(0);
    let kernel_name = encoded.pipeline_label.split('|').next().unwrap_or_default();
    let actual_route = DenseQ4Route::from_kernel_name(kernel_name).ok_or_else(|| {
        MlxError::InvalidArgument(format!(
            "dense Q4 auto trace encoded unexpected pipeline {}",
            encoded.pipeline_label
        ))
    })?;
    let decision = if actual_route == planned_decision.route {
        planned_decision
    } else {
        DenseQ4DispatchDecision {
            route: actual_route,
            source: DenseQ4DecisionSource::IneligibleCompatibilityFallback,
        }
    };
    if encoded != expected_dispatch(decision.route, shape) {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 auto route {:?} encoded unexpected geometry: {encoded:?}",
            decision.route
        )));
    }
    let pipeline = registry.pipeline_identity(&encoded.pipeline_label)?;
    if pipeline.pipeline_label != encoded.pipeline_label
        || pipeline.kernel_name != decision.route.kernel_name()
    {
        return Err(MlxError::InvalidArgument(
            "dense Q4 auto route encoded an unexpected pipeline".into(),
        ));
    }
    let plan = registry.dense_q4_auto.frozen_plan.as_ref();
    if decision.route == DenseQ4Route::Tensor64x32
        && plan.and_then(|plan| plan.decisions.get(&shape)).copied()
            != Some(DenseQ4Route::Tensor64x32)
    {
        return Err(MlxError::InvalidArgument(
            "dense Q4 candidate trace lacks a frozen exact-shape decision".into(),
        ));
    }
    Ok(DenseQ4DispatchTrace {
        schema_version: DENSE_Q4_ROUTE_SCHEMA_VERSION,
        mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
        build_fingerprint: plan
            .map(|plan| plan.build_fingerprint.clone())
            .unwrap_or_else(|| "compatibility-unplanned".into()),
        plan_id: plan.map(|plan| plan.plan_id.clone()),
        shape,
        decision,
        encoded,
        pipeline,
    })
}

#[cfg(test)]
#[path = "dense_q4_auto_tests.rs"]
mod tests;
