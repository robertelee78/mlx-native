//! Frozen, device-calibrated routing for native BF16 dense projections.
//!
//! Calibration is an explicit pre-serve operation over borrowed model
//! weights. The process cache and frozen plans retain metadata only. Normal
//! route selection performs one registry-local shape lookup and never
//! benchmarks, waits, reads environment state, allocates, or locks global
//! state. Encoding then uses the registry's ordinary pipeline lookup.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, DispatchKind, EncodedKernelDispatch};
use crate::error::{MlxError, Result};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::dense_gemv_bf16::{dense_gemv_bf16_f32, dense_gemv_bf16_f32_tiled4};
use crate::ops::dense_mm_bf16::{
    dense_matmul_bf16_f32_with_tile, DenseMmBf16F32Params, DenseMmBf16TensorTile,
};
use crate::ops::dense_mm_capability::DenseMmBackend;

pub use crate::ops::dense_bf16_calibration::calibrate_dense_bf16_routes;

pub const DENSE_BF16_ROUTE_SCHEMA_VERSION: u32 = 1;
pub const DENSE_BF16_CALIBRATION_MAX_M: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseBf16Route {
    Row,
    Tiled4,
    TensorV1,
    Simdgroup,
}

impl DenseBf16Route {
    pub(super) fn kernel_name(self) -> &'static str {
        match self {
            Self::Row => "hf2q_dense_gemv_bf16_f32_4",
            Self::Tiled4 => "hf2q_dense_gemv_bf16_f32_r1_4",
            Self::TensorV1 => "hf2q_dense_mm_bf16_f32_tensor",
            Self::Simdgroup => "hf2q_dense_mm_bf16_f32_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16BaseShape {
    pub n: u32,
    pub k: u32,
    pub src0_batch: u32,
    pub src1_batch: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub src0_batch: u32,
    pub src1_batch: u32,
}

impl DenseBf16Shape {
    pub(super) fn params(self) -> DenseMmBf16F32Params {
        DenseMmBf16F32Params {
            m: self.m,
            n: self.n,
            k: self.k,
            src0_batch: self.src0_batch,
            src1_batch: self.src1_batch,
        }
    }
}

impl DenseBf16BaseShape {
    pub(super) fn with_m(self, m: u32) -> DenseBf16Shape {
        DenseBf16Shape {
            m,
            n: self.n,
            k: self.k,
            src0_batch: self.src0_batch,
            src1_batch: self.src1_batch,
        }
    }
}

pub struct DenseBf16CalibrationCase<'a> {
    pub weight: &'a MlxBuffer,
    pub shape: DenseBf16BaseShape,
    pub reachable_m: &'a [u32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseBf16CalibrationLimits {
    pub max_elapsed_ms: u64,
    pub max_shapes: u32,
}

impl Default for DenseBf16CalibrationLimits {
    fn default() -> Self {
        Self {
            max_elapsed_ms: 15_000,
            max_shapes: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseBf16DecisionSource {
    FrozenPlan,
    FrozenShapeFallback,
    ForcedTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16DispatchDecision {
    pub route: DenseBf16Route,
    pub source: DenseBf16DecisionSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16TimingDistribution {
    pub p25_us: f64,
    pub median_us: f64,
    pub p75_us: f64,
    pub samples: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16RouteTiming {
    pub route: DenseBf16Route,
    pub wall: DenseBf16TimingDistribution,
    pub gpu: DenseBf16TimingDistribution,
    pub encoded: EncodedKernelDispatch,
    pub pipeline: KernelPipelineIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseBf16SelectionStatus {
    CalibratedWinner,
    CompatibilityFastest,
    NoStableWinner,
    BudgetFallback,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16CalibrationDecision {
    pub shape: DenseBf16Shape,
    pub compatibility_route: DenseBf16Route,
    pub selected_route: DenseBf16Route,
    pub status: DenseBf16SelectionStatus,
    pub unavailable_routes: Vec<DenseBf16Route>,
    pub incoherent_routes: Vec<DenseBf16Route>,
    pub timings: Vec<DenseBf16RouteTiming>,
    pub process_cache_hit: bool,
    pub calibration_submissions: u32,
}

#[derive(Clone, Debug)]
pub struct DenseBf16RoutePlan {
    pub(super) plan_id: String,
    pub(super) build_fingerprint: String,
    pub(super) device_name: String,
    pub(super) device_registry_id: u64,
    pub(super) activation_epoch: u64,
    pub(super) default_dense_mm_route: DenseBf16Route,
    pub(super) decisions: HashMap<DenseBf16Shape, DenseBf16Route>,
}

impl DenseBf16RoutePlan {
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    pub fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16CalibrationBatchReceipt {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub build_fingerprint: String,
    pub plan_id: String,
    pub activation_epoch: u64,
    pub device_name: String,
    pub device_registry_id: u64,
    pub declared_shapes: u32,
    pub calibrated_decisions: u32,
    pub process_cache_hits: u32,
    pub budget_fallback_decisions: u32,
    pub calibration_submissions: u32,
    pub elapsed_ms: f64,
    pub deadline_overrun_ms: f64,
    pub decisions: Vec<DenseBf16CalibrationDecision>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseBf16DispatchTrace {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub build_fingerprint: String,
    pub shape: DenseBf16Shape,
    pub decision: DenseBf16DispatchDecision,
    pub encoded: EncodedKernelDispatch,
    pub pipeline: KernelPipelineIdentity,
}

#[derive(Default)]
pub(crate) struct DenseBf16AutoState {
    pub(super) frozen_plan: Option<Arc<DenseBf16RoutePlan>>,
}

pub(super) fn encode_route(
    route: DenseBf16Route,
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    match route {
        DenseBf16Route::Row => {
            dense_gemv_bf16_f32(encoder, registry, device, weight, input, output, params)
        }
        DenseBf16Route::Tiled4 => {
            dense_gemv_bf16_f32_tiled4(encoder, registry, device, weight, input, output, params)
        }
        DenseBf16Route::TensorV1 => dense_matmul_bf16_f32_with_tile(
            encoder,
            registry,
            device,
            weight,
            input,
            output,
            params,
            DenseMmBackend::TensorRequired,
            DenseMmBf16TensorTile::V1,
        ),
        DenseBf16Route::Simdgroup => dense_matmul_bf16_f32_with_tile(
            encoder,
            registry,
            device,
            weight,
            input,
            output,
            params,
            DenseMmBackend::FallbackRequired,
            DenseMmBf16TensorTile::V1,
        ),
    }
}

fn shape_from_params(params: &DenseMmBf16F32Params) -> DenseBf16Shape {
    DenseBf16Shape {
        m: params.m,
        n: params.n,
        k: params.k,
        src0_batch: params.src0_batch,
        src1_batch: params.src1_batch,
    }
}

impl KernelRegistry {
    /// Freeze one immutable BF16 route plan into this registry.
    ///
    /// Reinstalling the same plan is idempotent. A different plan is rejected
    /// so a live engine cannot change numerical or performance behavior.
    pub fn freeze_dense_bf16_plan(
        &mut self,
        device: &MlxDevice,
        plan: Arc<DenseBf16RoutePlan>,
    ) -> Result<()> {
        if plan.device_registry_id != device.registry_id() || plan.device_name != device.name() {
            return Err(MlxError::InvalidArgument(
                "dense BF16 route plan belongs to a different Metal device".into(),
            ));
        }
        if let Some(existing) = self.dense_bf16_auto.frozen_plan.as_ref() {
            if existing.plan_id == plan.plan_id {
                return Ok(());
            }
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 route plan is already frozen as {}",
                existing.plan_id
            )));
        }
        let build_fingerprint =
            crate::ops::dense_bf16_calibration::current_build_fingerprint(self, device)?;
        if build_fingerprint != plan.build_fingerprint {
            return Err(MlxError::InvalidArgument(
                "dense BF16 route plan build/pipeline identity mismatch".into(),
            ));
        }
        self.dense_bf16_auto.frozen_plan = Some(plan);
        Ok(())
    }

    pub fn dense_bf16_plan(&self) -> Option<Arc<DenseBf16RoutePlan>> {
        self.dense_bf16_auto.frozen_plan.clone()
    }
}

/// Encode one native BF16 projection through a frozen, registry-local plan.
pub fn dense_matmul_bf16_f32_auto(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<DenseBf16DispatchDecision> {
    let shape = shape_from_params(params);
    let plan = registry
        .dense_bf16_auto
        .frozen_plan
        .as_ref()
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "dense BF16 auto routing requires a frozen pre-serve plan".into(),
            )
        })?;
    if plan.device_registry_id != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense BF16 route plan/device mismatch".into(),
        ));
    }
    let (route, source) = plan.decisions.get(&shape).copied().map_or_else(
        || {
            let route = if shape.m <= DENSE_BF16_CALIBRATION_MAX_M && shape.k % 4 == 0 {
                DenseBf16Route::Row
            } else if shape.k < 32 && shape.k % 4 == 0 {
                DenseBf16Route::Tiled4
            } else if shape.k % 4 != 0 {
                DenseBf16Route::Simdgroup
            } else {
                plan.default_dense_mm_route
            };
            (route, DenseBf16DecisionSource::FrozenShapeFallback)
        },
        |route| (route, DenseBf16DecisionSource::FrozenPlan),
    );
    encode_route(
        route, encoder, registry, device, weight, input, output, params,
    )?;
    Ok(DenseBf16DispatchDecision { route, source })
}

/// Encode an explicit route for focused tests and calibration tooling.
pub fn dense_matmul_bf16_f32_forced(
    route: DenseBf16Route,
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<DenseBf16DispatchDecision> {
    encode_route(
        route, encoder, registry, device, weight, input, output, params,
    )?;
    Ok(DenseBf16DispatchDecision {
        route,
        source: DenseBf16DecisionSource::ForcedTest,
    })
}

pub(super) fn expected_dispatch(
    route: DenseBf16Route,
    shape: DenseBf16Shape,
) -> EncodedKernelDispatch {
    let nsg = ((u64::from(shape.k) + 127) / 128).min(4);
    let (grid, threads, threadgroup_memory) = match route {
        DenseBf16Route::Row => (
            [
                (u64::from(shape.n) + 1) / 2,
                u64::from(shape.m),
                u64::from(shape.src1_batch),
            ],
            [32, nsg, 1],
            vec![(0, 256)],
        ),
        DenseBf16Route::Tiled4 => (
            [
                (u64::from(shape.n) + 1) / 2,
                (u64::from(shape.m) + 3) / 4,
                u64::from(shape.src1_batch),
            ],
            [32, 4, 1],
            vec![(0, 1024)],
        ),
        DenseBf16Route::TensorV1 | DenseBf16Route::Simdgroup => (
            [
                (u64::from(shape.m) + 31) / 32,
                (u64::from(shape.n) + 63) / 64,
                u64::from(shape.src1_batch),
            ],
            [128, 1, 1],
            vec![(0, 8192)],
        ),
    };
    EncodedKernelDispatch {
        pipeline_label: route.kernel_name().to_string(),
        dispatch_kind: DispatchKind::ThreadGroups,
        grid,
        threads_per_threadgroup: threads,
        threadgroup_memory,
    }
}

/// Trace the exact pipeline and geometry encoded by one frozen auto call.
/// This evidence call allocates strings; the ordinary auto route does not.
pub fn trace_dense_matmul_bf16_f32_auto(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<DenseBf16DispatchTrace> {
    if encoder.device_registry_id() != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense BF16 trace encoder/device mismatch".into(),
        ));
    }
    encoder.start_encoded_dispatch_receipt(1)?;
    let operation =
        dense_matmul_bf16_f32_auto(encoder, registry, device, weight, input, output, params);
    let encoded = encoder.take_encoded_dispatch_receipt();
    let decision = operation?;
    let mut encoded = encoded?;
    if encoded.len() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 auto route encoded {} dispatches, expected one",
            encoded.len()
        )));
    }
    let encoded = encoded.remove(0);
    let shape = shape_from_params(params);
    if encoded != expected_dispatch(decision.route, shape) {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 auto route {:?} encoded unexpected geometry: {encoded:?}",
            decision.route
        )));
    }
    let pipeline = registry.pipeline_identity(&encoded.pipeline_label)?;
    if pipeline.pipeline_label != encoded.pipeline_label
        || pipeline.kernel_name != decision.route.kernel_name()
    {
        return Err(MlxError::InvalidArgument(
            "dense BF16 auto route encoded an unexpected pipeline".into(),
        ));
    }
    let build_fingerprint = registry
        .dense_bf16_auto
        .frozen_plan
        .as_ref()
        .map(|plan| plan.build_fingerprint.clone())
        .ok_or_else(|| MlxError::InvalidArgument("dense BF16 plan disappeared".into()))?;
    Ok(DenseBf16DispatchTrace {
        schema_version: DENSE_BF16_ROUTE_SCHEMA_VERSION,
        mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
        build_fingerprint,
        shape,
        decision,
        encoded,
        pipeline,
    })
}
