//! Pre-serve calibration for frozen native Q4_0 dense routing.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use metal::foreign_types::ForeignType;
use metal::MTLSize;
use sha2::{Digest, Sha256};

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, EncodedKernelDispatch, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::dense_q4_auto::{
    expected_dispatch, DenseQ4BaseShape, DenseQ4CalibrationBatchReceipt, DenseQ4CalibrationCase,
    DenseQ4CalibrationDecision, DenseQ4CalibrationLimits, DenseQ4Route, DenseQ4RoutePlan,
    DenseQ4RouteTiming, DenseQ4SelectionStatus, DenseQ4Shape, DenseQ4TimingDistribution,
    DENSE_Q4_ROUTE_SCHEMA_VERSION, Q4_MM_PIPELINE_INT_CONSTANTS,
};
use crate::ops::quantized_matmul_ggml::dispatch_mm_q4_route_internal;

pub(super) const CALIBRATION_SAMPLES: usize = 5;
const MATERIAL_WIN_FRACTION: f64 = 0.05;
const GPU_CONTRARY_TOLERANCE: f64 = 0.02;
const OUTPUT_GUARD_ELEMENTS: usize = 16;
const Q4_0_BLOCK_VALUES: u32 = 32;
const Q4_0_BLOCK_BYTES: usize = 18;
const PROOF_POISON_KERNEL: &str = "hf2q_dense_q4_proof_poison";
const PROOF_COMPARE_KERNEL: &str = "hf2q_dense_q4_proof_compare";

const PROOF_CONTROL_UNWRITTEN: u32 = 1 << 0;
const PROOF_CONTROL_NONFINITE: u32 = 1 << 1;
const PROOF_CANDIDATE_UNWRITTEN: u32 = 1 << 2;
const PROOF_CANDIDATE_NONFINITE: u32 = 1 << 3;
const PROOF_MISMATCH: u32 = 1 << 4;
const PROOF_CONTROL_GUARD: u32 = 1 << 5;
const PROOF_CANDIDATE_GUARD: u32 = 1 << 6;
const PROOF_CONTROL_FAILURES: u32 =
    PROOF_CONTROL_UNWRITTEN | PROOF_CONTROL_NONFINITE | PROOF_CONTROL_GUARD;
const PROOF_CANDIDATE_FAILURES: u32 =
    PROOF_CANDIDATE_UNWRITTEN | PROOF_CANDIDATE_NONFINITE | PROOF_MISMATCH | PROOF_CANDIDATE_GUARD;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenseQ4ProofAuxParams {
    logical_elements: u64,
    guarded_elements: u64,
    status_index: u32,
    _padding: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DenseQ4ProcessKey {
    build_fingerprint: String,
    device_name: String,
    device_registry_id: u64,
    pipeline_set_fingerprint: String,
    shape: DenseQ4Shape,
}

#[derive(Clone, Debug)]
struct DenseQ4CalibrationFailure {
    status: DenseQ4SelectionStatus,
    message: String,
    timing_submissions: u32,
}

#[derive(Clone, Debug)]
struct DenseQ4CachedRouteTiming {
    route: DenseQ4Route,
    wall: DenseQ4TimingDistribution,
    gpu: DenseQ4TimingDistribution,
}

#[derive(Clone, Debug)]
struct DenseQ4CachedTimingDecision {
    selected_route: DenseQ4Route,
    status: DenseQ4SelectionStatus,
    timings: Vec<DenseQ4CachedRouteTiming>,
    timing_submissions: u32,
}

type CalibrationCell =
    OnceLock<std::result::Result<DenseQ4CachedTimingDecision, DenseQ4CalibrationFailure>>;

fn process_cache() -> &'static Mutex<HashMap<DenseQ4ProcessKey, Arc<CalibrationCell>>> {
    static CACHE: OnceLock<Mutex<HashMap<DenseQ4ProcessKey, Arc<CalibrationCell>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn process_cell(key: DenseQ4ProcessKey) -> Result<Arc<CalibrationCell>> {
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense Q4 calibration cache mutex is poisoned".into())
    })?;
    Ok(cache
        .entry(key)
        .or_insert_with(|| Arc::new(OnceLock::new()))
        .clone())
}

fn evict_process_cell_if_same(key: &DenseQ4ProcessKey, cell: &Arc<CalibrationCell>) -> Result<()> {
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense Q4 calibration cache mutex is poisoned".into())
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
            digest.update(DENSE_Q4_ROUTE_SCHEMA_VERSION.to_le_bytes());
            digest.update(include_bytes!(
                "../shaders/quantized_matmul_mm_tensor.metal"
            ));
            digest.update(include_bytes!("../shaders/dense_q4_calibration.metal"));
            hex::encode(digest.finalize())
        })
        .as_str()
}

struct PreparedDenseQ4Routes {
    candidate_available: bool,
    candidate_diagnostic: Option<String>,
    pipeline_set_fingerprint: String,
    build_fingerprint: String,
    identities: HashMap<DenseQ4Route, KernelPipelineIdentity>,
    proof_identities: Option<[KernelPipelineIdentity; 2]>,
}

fn prepare_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<PreparedDenseQ4Routes> {
    let mut identities = HashMap::new();
    let compatibility = DenseQ4Route::CompatibilityV2;
    registry
        .get_pipeline_with_constants(
            compatibility.kernel_name(),
            device.metal_device(),
            &[],
            Q4_MM_PIPELINE_INT_CONSTANTS,
        )
        .map_err(|error| {
            MlxError::InvalidArgument(format!(
                "required dense Q4 compatibility V2 pipeline is unavailable: {error}"
            ))
        })?;
    identities.insert(
        compatibility,
        registry.pipeline_identity(&compatibility.pipeline_label())?,
    );

    let candidate = DenseQ4Route::Tensor64x32;
    let candidate_result = registry
        .get_pipeline_with_constants(
            candidate.kernel_name(),
            device.metal_device(),
            &[],
            Q4_MM_PIPELINE_INT_CONSTANTS,
        )
        .map(|_| ())
        .map_err(|error| error.to_string());
    let mut candidate_available = candidate_result.is_ok();
    let mut candidate_diagnostic = candidate_result
        .err()
        .map(|error| format!("64x32 candidate pipeline unavailable: {error}"));
    let mut proof_identities = None;
    if candidate_available {
        identities.insert(
            candidate,
            registry.pipeline_identity(&candidate.pipeline_label())?,
        );
        let proof_result = (|| -> Result<[KernelPipelineIdentity; 2]> {
            registry.get_pipeline(PROOF_POISON_KERNEL, device.metal_device())?;
            registry.get_pipeline(PROOF_COMPARE_KERNEL, device.metal_device())?;
            Ok([
                registry.pipeline_identity(PROOF_POISON_KERNEL)?,
                registry.pipeline_identity(PROOF_COMPARE_KERNEL)?,
            ])
        })();
        match proof_result {
            Ok(identities) => proof_identities = Some(identities),
            Err(error) => {
                candidate_available = false;
                candidate_diagnostic = Some(format!(
                    "64x32 candidate proof pipelines unavailable: {error}"
                ));
                identities.remove(&candidate);
            }
        }
    }

    let mut ordered: Vec<_> = identities.values().cloned().collect();
    if let Some(proof) = &proof_identities {
        ordered.extend(proof.iter().cloned());
    }
    ordered.sort_by(|left, right| left.pipeline_label.cmp(&right.pipeline_label));
    let encoded = serde_json::to_vec(&ordered).map_err(|error| {
        MlxError::InvalidArgument(format!("serialize dense Q4 pipeline identities: {error}"))
    })?;
    let pipeline_set_fingerprint = hex::encode(Sha256::digest(encoded));
    let mut build_digest = Sha256::new();
    build_digest.update(source_build_fingerprint().as_bytes());
    build_digest.update(pipeline_set_fingerprint.as_bytes());
    Ok(PreparedDenseQ4Routes {
        candidate_available,
        candidate_diagnostic,
        pipeline_set_fingerprint,
        build_fingerprint: hex::encode(build_digest.finalize()),
        identities,
        proof_identities,
    })
}

pub(super) fn current_build_fingerprint(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<String> {
    Ok(prepare_routes(registry, device)?.build_fingerprint)
}

fn checked_elements(dimensions: &[u32]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension as usize).ok_or_else(|| {
            MlxError::InvalidArgument("dense Q4 calibration size overflows usize".into())
        })
    })
}

fn trace_route_encoding(
    route: DenseQ4Route,
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    shape: DenseQ4Shape,
    prepared: &PreparedDenseQ4Routes,
) -> Result<(EncodedKernelDispatch, KernelPipelineIdentity)> {
    encoder.start_encoded_dispatch_receipt(1)?;
    let operation = dispatch_mm_q4_route_internal(
        encoder,
        registry,
        device,
        input,
        weight,
        output,
        &shape.params(),
        route,
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    operation?;
    let mut encoded = encoded?;
    if encoded.len() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 route {route:?} encoded {} dispatches, expected one",
            encoded.len()
        )));
    }
    let encoded = encoded.remove(0);
    if encoded != expected_dispatch(route, shape) {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 route {route:?} encoded unexpected geometry: {encoded:?}"
        )));
    }
    let pipeline = registry.pipeline_identity(&encoded.pipeline_label)?;
    let expected_pipeline = prepared.identities.get(&route).ok_or_else(|| {
        MlxError::InvalidArgument(format!("dense Q4 route {route:?} was not prepared"))
    })?;
    if &pipeline != expected_pipeline
        || pipeline.pipeline_label != encoded.pipeline_label
        || pipeline.kernel_name != route.kernel_name()
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 route {route:?} pipeline identity changed during activation"
        )));
    }
    Ok((encoded, pipeline))
}

struct DenseQ4TimedSample {
    wall_us: f64,
    gpu_us: f64,
}

fn time_route(
    route: DenseQ4Route,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    shape: DenseQ4Shape,
    timing_submissions: &mut u32,
) -> Result<DenseQ4TimedSample> {
    let started = Instant::now();
    let mut encoder = device.command_encoder()?;
    dispatch_mm_q4_route_internal(
        &mut encoder,
        registry,
        device,
        input,
        weight,
        output,
        &shape.params(),
        route,
    )?;
    *timing_submissions = (*timing_submissions).checked_add(1).ok_or_else(|| {
        MlxError::InvalidArgument("dense Q4 timing submission count overflow".into())
    })?;
    let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time()?;
    let wall_us = started.elapsed().as_secs_f64() * 1e6;
    if !gpu_start.is_finite()
        || !gpu_end.is_finite()
        || gpu_end <= gpu_start
        || !wall_us.is_finite()
        || wall_us <= 0.0
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 route {route:?} returned an invalid timing interval"
        )));
    }
    Ok(DenseQ4TimedSample {
        wall_us,
        gpu_us: (gpu_end - gpu_start) * 1e6,
    })
}

fn distribution(mut samples: Vec<f64>) -> Result<DenseQ4TimingDistribution> {
    if samples.len() != CALIBRATION_SAMPLES {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 timing has {} samples, expected {CALIBRATION_SAMPLES}",
            samples.len()
        )));
    }
    samples.sort_by(f64::total_cmp);
    Ok(DenseQ4TimingDistribution {
        p25_us: samples[(samples.len() - 1) / 4],
        median_us: samples[samples.len() / 2],
        p75_us: samples[(samples.len() - 1) * 3 / 4],
        samples: samples.len() as u32,
    })
}

fn select_route(timings: &[DenseQ4RouteTiming]) -> Result<(DenseQ4Route, DenseQ4SelectionStatus)> {
    let baseline = timings
        .iter()
        .find(|timing| timing.route == DenseQ4Route::CompatibilityV2)
        .ok_or_else(|| MlxError::InvalidArgument("dense Q4 baseline was not timed".into()))?;
    let candidate = timings
        .iter()
        .find(|timing| timing.route == DenseQ4Route::Tensor64x32)
        .ok_or_else(|| MlxError::InvalidArgument("dense Q4 candidate was not timed".into()))?;
    if candidate.wall.median_us >= baseline.wall.median_us {
        return Ok((
            DenseQ4Route::CompatibilityV2,
            DenseQ4SelectionStatus::CompatibilityFastest,
        ));
    }
    let material =
        candidate.wall.median_us <= baseline.wall.median_us * (1.0 - MATERIAL_WIN_FRACTION);
    let stable = candidate.wall.p75_us < baseline.wall.p25_us;
    let no_contrary_gpu =
        candidate.gpu.median_us <= baseline.gpu.median_us * (1.0 + GPU_CONTRARY_TOLERANCE);
    if material && stable && no_contrary_gpu {
        Ok((
            DenseQ4Route::Tensor64x32,
            DenseQ4SelectionStatus::CalibratedWinner,
        ))
    } else {
        Ok((
            DenseQ4Route::CompatibilityV2,
            DenseQ4SelectionStatus::NoStableWinner,
        ))
    }
}

fn compatibility_decision(
    shape: DenseQ4Shape,
    status: DenseQ4SelectionStatus,
    proof: DenseQ4ProofMetrics,
    timing_submissions: u32,
    diagnostic: Option<String>,
) -> DenseQ4CalibrationDecision {
    DenseQ4CalibrationDecision {
        shape,
        selected_route: DenseQ4Route::CompatibilityV2,
        status,
        diagnostic,
        timings: Vec::new(),
        process_cache_hit: false,
        authorized_weight_buffers: 1,
        proof_submissions: proof.submissions,
        proof_route_dispatches: proof.route_dispatches,
        proof_auxiliary_dispatches: proof.auxiliary_dispatches,
        proof_scratch_bytes: proof.scratch_bytes,
        proof_gpu_us: proof.gpu_us,
        timing_submissions,
        calibration_submissions: proof.submissions + timing_submissions,
    }
}

fn apply_base_weight_failure(
    decisions: &mut [DenseQ4CalibrationDecision],
    status: DenseQ4SelectionStatus,
    diagnostic: &str,
) {
    for decision in decisions {
        decision.selected_route = DenseQ4Route::CompatibilityV2;
        decision.status = status;
        decision.diagnostic = Some(match decision.diagnostic.take() {
            Some(existing) => {
                format!("{existing}; base-shape Cartesian proof failure: {diagnostic}")
            }
            None => format!("base-shape Cartesian proof failure: {diagnostic}"),
        });
    }
}

fn finalize_base_shape_decisions(
    base_outcomes: Vec<(DenseQ4CalibrationDecision, bool)>,
) -> (
    Vec<DenseQ4CalibrationDecision>,
    HashMap<DenseQ4Shape, DenseQ4Route>,
) {
    let base_weight_failure = base_outcomes.iter().find_map(|(decision, candidate_proven)| {
        (!*candidate_proven).then(|| {
            (
                decision.status,
                decision.diagnostic.clone().unwrap_or_else(|| {
                    format!(
                        "exact M={} Cartesian current-weight batch did not authorize the optional candidate",
                        decision.shape.m
                    )
                }),
            )
        })
    });
    let mut base_decisions = base_outcomes
        .into_iter()
        .map(|(decision, _)| decision)
        .collect::<Vec<_>>();
    if let Some((status, diagnostic)) = base_weight_failure {
        apply_base_weight_failure(&mut base_decisions, status, &diagnostic);
    }
    let plan_decisions = base_decisions
        .iter()
        .map(|decision| (decision.shape, decision.selected_route))
        .collect();
    (base_decisions, plan_decisions)
}

fn deadline_reached(started: Instant, max_elapsed_ms: u64) -> bool {
    started.elapsed().as_secs_f64() * 1000.0 >= max_elapsed_ms as f64
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DenseQ4ProofMetrics {
    submissions: u32,
    route_dispatches: u32,
    auxiliary_dispatches: u32,
    scratch_bytes: u64,
    gpu_us: f64,
}

struct DenseQ4CurrentProof {
    input: MlxBuffer,
    output: MlxBuffer,
    evidence: HashMap<DenseQ4Route, (EncodedKernelDispatch, KernelPipelineIdentity)>,
    metrics: DenseQ4ProofMetrics,
}

enum DenseQ4ProofOutcome {
    Coherent(DenseQ4CurrentProof),
    Fallback(DenseQ4CalibrationDecision),
}

fn encode_proof_auxiliary(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kernel_name: &str,
    control: &MlxBuffer,
    candidate: &MlxBuffer,
    statuses: Option<&MlxBuffer>,
    params: &DenseQ4ProofAuxParams,
    expected_identity: &KernelPipelineIdentity,
) -> Result<()> {
    let pipeline = registry
        .get_pipeline(kernel_name, device.metal_device())?
        .to_owned();
    let actual_identity = registry.pipeline_identity(kernel_name)?;
    if &actual_identity != expected_identity {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 proof pipeline {kernel_name} identity changed during activation"
        )));
    }
    let threads = MTLSize::new(params.guarded_elements, 1, 1);
    let threadgroup = MTLSize::new(params.guarded_elements.min(256), 1, 1);
    if let Some(statuses) = statuses {
        encoder.dispatch_tracked_threads_with_args(
            &pipeline,
            &[
                (0, KernelArg::Buffer(control)),
                (1, KernelArg::Buffer(candidate)),
                (2, KernelArg::Buffer(statuses)),
                (3, KernelArg::Bytes(bytemuck::bytes_of(params))),
            ],
            &[control, candidate],
            &[statuses],
            threads,
            threadgroup,
        );
    } else {
        encoder.dispatch_tracked_threads_with_args(
            &pipeline,
            &[
                (0, KernelArg::Buffer(control)),
                (1, KernelArg::Buffer(candidate)),
                (2, KernelArg::Bytes(bytemuck::bytes_of(params))),
            ],
            &[],
            &[control, candidate],
            threads,
            threadgroup,
        );
    }
    Ok(())
}

fn proof_status_diagnostic(weight_index: usize, status: u32) -> Result<Option<String>> {
    let unknown = status & !(PROOF_CONTROL_FAILURES | PROOF_CANDIDATE_FAILURES);
    if unknown != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 proof weight {weight_index} returned unknown status bits {unknown:#x}"
        )));
    }
    if status & PROOF_CONTROL_FAILURES != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "required dense Q4 compatibility V2 proof failed for weight {weight_index}: \
             {} (status={status:#x})",
            proof_status_labels(status & PROOF_CONTROL_FAILURES)
        )));
    }
    if status & PROOF_CANDIDATE_FAILURES != 0 {
        return Ok(Some(format!(
            "optional dense Q4 candidate proof failed for weight {weight_index}: {} \
             (status={status:#x})",
            proof_status_labels(status & PROOF_CANDIDATE_FAILURES)
        )));
    }
    Ok(None)
}

fn proof_status_labels(status: u32) -> String {
    let mut labels = Vec::new();
    for (bit, label) in [
        (PROOF_CONTROL_UNWRITTEN, "compatibility output unwritten"),
        (PROOF_CONTROL_NONFINITE, "compatibility output non-finite"),
        (PROOF_CANDIDATE_UNWRITTEN, "candidate output unwritten"),
        (PROOF_CANDIDATE_NONFINITE, "candidate output non-finite"),
        (PROOF_MISMATCH, "candidate output bitwise mismatch"),
        (
            PROOF_CONTROL_GUARD,
            "compatibility output guard overwritten",
        ),
        (PROOF_CANDIDATE_GUARD, "candidate output guard overwritten"),
    ] {
        if status & bit != 0 {
            labels.push(label);
        }
    }
    labels.join(", ")
}

type DenseQ4ProofStatusTransform = fn(DenseQ4Shape, usize, u32) -> u32;

fn identity_proof_status(_shape: DenseQ4Shape, _weight_index: usize, status: u32) -> u32 {
    status
}

fn prove_current_weights(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &[&MlxBuffer],
    shape: DenseQ4Shape,
    prepared: &PreparedDenseQ4Routes,
    started: Instant,
    limits: DenseQ4CalibrationLimits,
    metrics: &mut DenseQ4ProofMetrics,
    status_transform: DenseQ4ProofStatusTransform,
) -> Result<DenseQ4ProofOutcome> {
    if weights.is_empty() {
        return Err(MlxError::InvalidArgument(
            "dense Q4 proof batch requires at least one current weight".into(),
        ));
    }
    if !prepared.candidate_available {
        return Ok(DenseQ4ProofOutcome::Fallback(compatibility_decision(
            shape,
            DenseQ4SelectionStatus::CandidateUnavailable,
            DenseQ4ProofMetrics::default(),
            0,
            prepared.candidate_diagnostic.clone(),
        )));
    }
    if deadline_reached(started, limits.max_elapsed_ms) {
        return Ok(DenseQ4ProofOutcome::Fallback(compatibility_decision(
            shape,
            DenseQ4SelectionStatus::BudgetFallback,
            DenseQ4ProofMetrics::default(),
            0,
            Some("calibration deadline reached before shape proof".into()),
        )));
    }
    let input_elements = checked_elements(&[shape.m, shape.k])?;
    let output_elements = checked_elements(&[shape.m, shape.n])?;
    let mut input = device.alloc_buffer(
        input_elements
            .checked_mul(DType::F32.size_of())
            .ok_or_else(|| MlxError::InvalidArgument("dense Q4 input bytes overflow".into()))?,
        DType::F32,
        vec![shape.m as usize, shape.k as usize],
    )?;
    for (index, value) in input.as_mut_slice::<f32>()?.iter_mut().enumerate() {
        let coarse = (index.wrapping_mul(29) % 257) as f32 - 128.0;
        let fine = (index.wrapping_mul(43) % 31) as f32 - 15.0;
        *value = coarse / 251.0 + fine / 16_381.0;
    }
    let guarded_output_elements = output_elements
        .checked_add(OUTPUT_GUARD_ELEMENTS)
        .ok_or_else(|| MlxError::InvalidArgument("dense Q4 output guard overflow".into()))?;
    if guarded_output_elements > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "dense Q4 proof output exceeds auxiliary shader indexing".into(),
        ));
    }
    let output_bytes = guarded_output_elements
        .checked_mul(DType::F32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("dense Q4 output bytes overflow".into()))?;
    let output = device.alloc_buffer(output_bytes, DType::F32, vec![guarded_output_elements])?;
    let candidate_output =
        device.alloc_buffer(output_bytes, DType::F32, vec![guarded_output_elements])?;
    let mut statuses = device.alloc_buffer(
        weights
            .len()
            .checked_mul(DType::U32.size_of())
            .ok_or_else(|| MlxError::InvalidArgument("dense Q4 status bytes overflow".into()))?,
        DType::U32,
        vec![weights.len()],
    )?;
    statuses.as_mut_slice::<u32>()?.fill(0);
    let scratch_bytes = input
        .data_byte_len()
        .checked_add(output.data_byte_len())
        .and_then(|bytes| bytes.checked_add(candidate_output.data_byte_len()))
        .and_then(|bytes| bytes.checked_add(statuses.data_byte_len()))
        .ok_or_else(|| MlxError::InvalidArgument("dense Q4 proof scratch bytes overflow".into()))?
        as u64;

    metrics.scratch_bytes = scratch_bytes;
    let mut evidence = HashMap::new();
    let proof_identities = prepared.proof_identities.as_ref().ok_or_else(|| {
        MlxError::InvalidArgument("dense Q4 candidate has no prepared proof pipelines".into())
    })?;
    let mut encoder = device.command_encoder()?;
    for (weight_index, weight) in weights.iter().enumerate() {
        if deadline_reached(started, limits.max_elapsed_ms) {
            return Ok(DenseQ4ProofOutcome::Fallback(compatibility_decision(
                shape,
                DenseQ4SelectionStatus::BudgetFallback,
                *metrics,
                0,
                Some("calibration deadline reached while encoding shape proof batch".into()),
            )));
        }
        let params = DenseQ4ProofAuxParams {
            logical_elements: output_elements as u64,
            guarded_elements: guarded_output_elements as u64,
            status_index: weight_index as u32,
            _padding: 0,
        };
        encode_proof_auxiliary(
            &mut encoder,
            registry,
            device,
            PROOF_POISON_KERNEL,
            &output,
            &candidate_output,
            None,
            &params,
            &proof_identities[0],
        )?;
        metrics.auxiliary_dispatches = metrics
            .auxiliary_dispatches
            .checked_add(1)
            .ok_or_else(|| MlxError::InvalidArgument("dense Q4 proof dispatch overflow".into()))?;
        encoder.memory_barrier();

        for (route, route_output) in [
            (DenseQ4Route::CompatibilityV2, &output),
            (DenseQ4Route::Tensor64x32, &candidate_output),
        ] {
            let route_evidence = trace_route_encoding(
                route,
                &mut encoder,
                registry,
                device,
                weight,
                &input,
                route_output,
                shape,
                prepared,
            )?;
            metrics.route_dispatches =
                metrics.route_dispatches.checked_add(1).ok_or_else(|| {
                    MlxError::InvalidArgument("dense Q4 proof route dispatch overflow".into())
                })?;
            evidence.entry(route).or_insert(route_evidence);
        }
        encoder.memory_barrier();
        encode_proof_auxiliary(
            &mut encoder,
            registry,
            device,
            PROOF_COMPARE_KERNEL,
            &output,
            &candidate_output,
            Some(&statuses),
            &params,
            &proof_identities[1],
        )?;
        metrics.auxiliary_dispatches = metrics
            .auxiliary_dispatches
            .checked_add(1)
            .ok_or_else(|| MlxError::InvalidArgument("dense Q4 proof dispatch overflow".into()))?;
        encoder.memory_barrier();
    }
    metrics.submissions = 1;
    let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time()?;
    if !gpu_start.is_finite() || !gpu_end.is_finite() || gpu_end <= gpu_start {
        return Err(MlxError::InvalidArgument(
            "dense Q4 proof batch returned an invalid GPU interval".into(),
        ));
    }
    metrics.gpu_us = (gpu_end - gpu_start) * 1e6;

    for (weight_index, &status) in statuses.as_slice::<u32>()?.iter().enumerate() {
        let status = status_transform(shape, weight_index, status);
        if let Some(diagnostic) = proof_status_diagnostic(weight_index, status)? {
            return Ok(DenseQ4ProofOutcome::Fallback(compatibility_decision(
                shape,
                DenseQ4SelectionStatus::IncoherentCandidate,
                *metrics,
                0,
                Some(diagnostic),
            )));
        }
    }

    Ok(DenseQ4ProofOutcome::Coherent(DenseQ4CurrentProof {
        input,
        output,
        evidence,
        metrics: *metrics,
    }))
}

fn measure_timing_decision(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    shape: DenseQ4Shape,
    proof: &DenseQ4CurrentProof,
    started: Instant,
    limits: DenseQ4CalibrationLimits,
) -> std::result::Result<DenseQ4CachedTimingDecision, DenseQ4CalibrationFailure> {
    let mut timing_submissions = 0u32;
    let failure = |status, message: String, timing_submissions| DenseQ4CalibrationFailure {
        status,
        message,
        timing_submissions,
    };

    let mut wall_samples: HashMap<DenseQ4Route, Vec<f64>> =
        [DenseQ4Route::CompatibilityV2, DenseQ4Route::Tensor64x32]
            .into_iter()
            .map(|route| (route, Vec::with_capacity(CALIBRATION_SAMPLES)))
            .collect();
    let mut gpu_samples = wall_samples.clone();
    for round in 0..CALIBRATION_SAMPLES {
        let routes = if round % 2 == 0 {
            [DenseQ4Route::CompatibilityV2, DenseQ4Route::Tensor64x32]
        } else {
            [DenseQ4Route::Tensor64x32, DenseQ4Route::CompatibilityV2]
        };
        for route in routes {
            if deadline_reached(started, limits.max_elapsed_ms) {
                return Err(failure(
                    DenseQ4SelectionStatus::BudgetFallback,
                    "calibration deadline reached during timing".into(),
                    timing_submissions,
                ));
            }
            let sample = time_route(
                route,
                registry,
                device,
                weight,
                &proof.input,
                &proof.output,
                shape,
                &mut timing_submissions,
            )
            .map_err(|error| {
                failure(
                    DenseQ4SelectionStatus::CalibrationErrorFallback,
                    error.to_string(),
                    timing_submissions,
                )
            })?;
            wall_samples
                .get_mut(&route)
                .ok_or_else(|| {
                    failure(
                        DenseQ4SelectionStatus::CalibrationErrorFallback,
                        "dense Q4 wall route vanished".into(),
                        timing_submissions,
                    )
                })?
                .push(sample.wall_us);
            gpu_samples
                .get_mut(&route)
                .ok_or_else(|| {
                    failure(
                        DenseQ4SelectionStatus::CalibrationErrorFallback,
                        "dense Q4 GPU route vanished".into(),
                        timing_submissions,
                    )
                })?
                .push(sample.gpu_us);
        }
    }

    let mut timings = Vec::with_capacity(2);
    for route in [DenseQ4Route::CompatibilityV2, DenseQ4Route::Tensor64x32] {
        let (encoded, pipeline) = proof.evidence.get(&route).cloned().ok_or_else(|| {
            failure(
                DenseQ4SelectionStatus::CalibrationErrorFallback,
                format!("dense Q4 route {route:?} has no current proof"),
                timing_submissions,
            )
        })?;
        timings.push(DenseQ4RouteTiming {
            route,
            wall: distribution(wall_samples.remove(&route).ok_or_else(|| {
                failure(
                    DenseQ4SelectionStatus::CalibrationErrorFallback,
                    "dense Q4 wall samples vanished".into(),
                    timing_submissions,
                )
            })?)
            .map_err(|error| {
                failure(
                    DenseQ4SelectionStatus::CalibrationErrorFallback,
                    error.to_string(),
                    timing_submissions,
                )
            })?,
            gpu: distribution(gpu_samples.remove(&route).ok_or_else(|| {
                failure(
                    DenseQ4SelectionStatus::CalibrationErrorFallback,
                    "dense Q4 GPU samples vanished".into(),
                    timing_submissions,
                )
            })?)
            .map_err(|error| {
                failure(
                    DenseQ4SelectionStatus::CalibrationErrorFallback,
                    error.to_string(),
                    timing_submissions,
                )
            })?,
            encoded,
            pipeline,
        });
    }
    let (selected_route, status) = select_route(&timings).map_err(|error| {
        failure(
            DenseQ4SelectionStatus::CalibrationErrorFallback,
            error.to_string(),
            timing_submissions,
        )
    })?;
    Ok(DenseQ4CachedTimingDecision {
        selected_route,
        status,
        timings: timings
            .into_iter()
            .map(|timing| DenseQ4CachedRouteTiming {
                route: timing.route,
                wall: timing.wall,
                gpu: timing.gpu,
            })
            .collect(),
        timing_submissions,
    })
}

fn materialize_timing_decision(
    shape: DenseQ4Shape,
    proof: &DenseQ4CurrentProof,
    cached: &DenseQ4CachedTimingDecision,
    process_cache_hit: bool,
) -> Result<DenseQ4CalibrationDecision> {
    let mut timings = Vec::with_capacity(cached.timings.len());
    for timing in &cached.timings {
        let (encoded, pipeline) = proof.evidence.get(&timing.route).cloned().ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "dense Q4 cached timing route {:?} has no current proof",
                timing.route
            ))
        })?;
        timings.push(DenseQ4RouteTiming {
            route: timing.route,
            wall: timing.wall,
            gpu: timing.gpu,
            encoded,
            pipeline,
        });
    }
    Ok(DenseQ4CalibrationDecision {
        shape,
        selected_route: cached.selected_route,
        status: cached.status,
        diagnostic: None,
        timings,
        process_cache_hit,
        authorized_weight_buffers: 1,
        proof_submissions: proof.metrics.submissions,
        proof_route_dispatches: proof.metrics.route_dispatches,
        proof_auxiliary_dispatches: proof.metrics.auxiliary_dispatches,
        proof_scratch_bytes: proof.metrics.scratch_bytes,
        proof_gpu_us: proof.metrics.gpu_us,
        timing_submissions: if process_cache_hit {
            0
        } else {
            cached.timing_submissions
        },
        calibration_submissions: proof.metrics.submissions
            + if process_cache_hit {
                0
            } else {
                cached.timing_submissions
            },
    })
}

struct ValidatedCalibrationCase<'a> {
    weights: Vec<&'a MlxBuffer>,
    shape: DenseQ4BaseShape,
    reachable_m: BTreeSet<u32>,
}

fn same_logical_weight(left: &MlxBuffer, right: &MlxBuffer) -> bool {
    left.metal_buffer().as_ptr() == right.metal_buffer().as_ptr()
        && left.byte_offset() == right.byte_offset()
        && left.data_byte_len() == right.data_byte_len()
}

fn validate_cases<'a>(
    cases: &'a [DenseQ4CalibrationCase<'a>],
    limits: DenseQ4CalibrationLimits,
) -> Result<Vec<ValidatedCalibrationCase<'a>>> {
    if limits.max_elapsed_ms == 0 || limits.max_shapes == 0 {
        return Err(MlxError::InvalidArgument(
            "dense Q4 calibration limits must be nonzero".into(),
        ));
    }
    if cases.is_empty() {
        return Err(MlxError::InvalidArgument(
            "dense Q4 calibration requires at least one borrowed weight case".into(),
        ));
    }
    let mut validated: Vec<ValidatedCalibrationCase<'a>> = Vec::new();
    for case in cases {
        let shape = case.shape;
        if shape.n == 0
            || shape.k < Q4_0_BLOCK_VALUES
            || shape.k % Q4_0_BLOCK_VALUES != 0
            || shape.batch != 1
            || shape.input_layout != super::dense_q4_auto::DenseQ4InputLayout::Contiguous
        {
            return Err(MlxError::InvalidArgument(format!(
                "invalid dense Q4 calibration base shape {shape:?}"
            )));
        }
        if shape.n > i32::MAX as u32 || shape.k > i32::MAX as u32 {
            return Err(MlxError::InvalidArgument(
                "dense Q4 calibration dimensions exceed shader indexing".into(),
            ));
        }
        if case.weight.dtype() != DType::U8 || case.weight.byte_offset() % 2 != 0 {
            return Err(MlxError::InvalidArgument(
                "dense Q4 calibration weight must be aligned native U8 Q4_0 storage".into(),
            ));
        }
        let weight_bytes = checked_elements(&[shape.n, shape.k / Q4_0_BLOCK_VALUES])?
            .checked_mul(Q4_0_BLOCK_BYTES)
            .ok_or_else(|| MlxError::InvalidArgument("dense Q4 weight bytes overflow".into()))?;
        if case.weight.data_byte_len() != weight_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "dense Q4 calibration weight must be exactly {weight_bytes} bytes, got {}",
                case.weight.data_byte_len()
            )));
        }
        if case.reachable_m.is_empty()
            || case
                .reachable_m
                .iter()
                .any(|&m| m <= 8 || m > i32::MAX as u32)
        {
            return Err(MlxError::InvalidArgument(
                "dense Q4 reachable rows must be in 9..=i32::MAX".into(),
            ));
        }
        if let Some(existing) = validated.iter_mut().find(|entry| entry.shape == shape) {
            existing
                .reachable_m
                .extend(case.reachable_m.iter().copied());
            if !existing
                .weights
                .iter()
                .any(|weight| same_logical_weight(weight, case.weight))
            {
                existing.weights.push(case.weight);
            }
        } else {
            validated.push(ValidatedCalibrationCase {
                weights: vec![case.weight],
                shape,
                reachable_m: case.reachable_m.iter().copied().collect(),
            });
        }
    }
    let declared_shapes: usize = validated.iter().map(|case| case.reachable_m.len()).sum();
    if declared_shapes == 0 || declared_shapes > limits.max_shapes as usize {
        return Err(MlxError::InvalidArgument(format!(
            "dense Q4 calibration declares {declared_shapes} shapes, limit is {}",
            limits.max_shapes
        )));
    }
    validated.sort_by_key(|case| case.shape);
    Ok(validated)
}

fn plan_id(
    build_fingerprint: &str,
    device_registry_id: u64,
    registry_authority_id: u64,
    activation_epoch: u64,
    decisions: &HashMap<DenseQ4Shape, DenseQ4Route>,
) -> Result<String> {
    let mut ordered: Vec<_> = decisions
        .iter()
        .map(|(shape, route)| (*shape, *route))
        .collect();
    ordered.sort_by_key(|(shape, _)| *shape);
    let encoded = serde_json::to_vec(&ordered).map_err(|error| {
        MlxError::InvalidArgument(format!("serialize dense Q4 route plan: {error}"))
    })?;
    let mut digest = Sha256::new();
    digest.update(build_fingerprint.as_bytes());
    digest.update(device_registry_id.to_le_bytes());
    digest.update(registry_authority_id.to_le_bytes());
    digest.update(activation_epoch.to_le_bytes());
    digest.update(encoded);
    Ok(hex::encode(digest.finalize()))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DenseQ4FinalBudget {
    compatibility_route_decisions: u32,
    candidate_downgrades: u32,
    elapsed_ms: f64,
    deadline_overrun_ms: f64,
}

fn validate_receipt_plan_routes(
    decisions: &[DenseQ4CalibrationDecision],
    plan_decisions: &HashMap<DenseQ4Shape, DenseQ4Route>,
) -> Result<u32> {
    if plan_decisions.len() != decisions.len() {
        return Err(MlxError::InvalidArgument(
            "dense Q4 receipt/plan decision count mismatch".into(),
        ));
    }
    for decision in decisions {
        if plan_decisions.get(&decision.shape).copied() != Some(decision.selected_route) {
            return Err(MlxError::InvalidArgument(format!(
                "dense Q4 receipt/plan route mismatch for {:?}",
                decision.shape
            )));
        }
    }
    decisions
        .iter()
        .filter(|decision| decision.selected_route == DenseQ4Route::CompatibilityV2)
        .count()
        .try_into()
        .map_err(|_| {
            MlxError::InvalidArgument(
                "dense Q4 compatibility route decision count overflows u32".into(),
            )
        })
}

fn prepare_budget_fallback(
    decisions: &mut [DenseQ4CalibrationDecision],
    plan_decisions: &mut HashMap<DenseQ4Shape, DenseQ4Route>,
    diagnostic: &str,
) -> Result<(u32, u32)> {
    let mut candidate_downgrades = 0u32;
    for decision in decisions.iter_mut() {
        if decision.selected_route == DenseQ4Route::Tensor64x32 {
            candidate_downgrades = candidate_downgrades.checked_add(1).ok_or_else(|| {
                MlxError::InvalidArgument(
                    "dense Q4 final budget downgrade count overflows u32".into(),
                )
            })?;
            decision.selected_route = DenseQ4Route::CompatibilityV2;
            decision.status = DenseQ4SelectionStatus::BudgetFallback;
            decision.diagnostic = Some(match decision.diagnostic.take() {
                Some(existing) => format!("{existing}; {diagnostic}"),
                None => diagnostic.into(),
            });
        }
    }
    for route in plan_decisions.values_mut() {
        if *route == DenseQ4Route::Tensor64x32 {
            *route = DenseQ4Route::CompatibilityV2;
        }
    }
    let compatibility_route_decisions = validate_receipt_plan_routes(decisions, plan_decisions)?;
    if decisions
        .iter()
        .any(|decision| decision.selected_route == DenseQ4Route::Tensor64x32)
        || plan_decisions
            .values()
            .any(|route| *route == DenseQ4Route::Tensor64x32)
    {
        return Err(MlxError::InvalidArgument(
            "dense Q4 candidate survived prepared deadline fallback".into(),
        ));
    }
    Ok((compatibility_route_decisions, candidate_downgrades))
}

#[cfg(test)]
fn enforce_final_budget(
    decisions: &mut [DenseQ4CalibrationDecision],
    plan_decisions: &mut HashMap<DenseQ4Shape, DenseQ4Route>,
    elapsed_ms: f64,
    max_elapsed_ms: u64,
) -> Result<DenseQ4FinalBudget> {
    if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
        return Err(MlxError::InvalidArgument(
            "dense Q4 final calibration elapsed time must be finite and nonnegative".into(),
        ));
    }
    let deadline_overrun_ms = (elapsed_ms - max_elapsed_ms as f64).max(0.0);
    let mut candidate_downgrades = 0u32;
    if deadline_overrun_ms > 0.0 {
        let diagnostic = format!(
            "final dense Q4 calibration deadline exceeded: {elapsed_ms:.3} ms elapsed > \
             {max_elapsed_ms} ms budget; candidate disabled"
        );
        (_, candidate_downgrades) =
            prepare_budget_fallback(decisions, plan_decisions, &diagnostic)?;
    }
    if deadline_overrun_ms > 0.0
        && (decisions
            .iter()
            .any(|decision| decision.selected_route == DenseQ4Route::Tensor64x32)
            || plan_decisions
                .values()
                .any(|route| *route == DenseQ4Route::Tensor64x32))
    {
        return Err(MlxError::InvalidArgument(
            "dense Q4 candidate survived the final calibration deadline gate".into(),
        ));
    }
    let compatibility_route_decisions = validate_receipt_plan_routes(decisions, plan_decisions)?;
    Ok(DenseQ4FinalBudget {
        compatibility_route_decisions,
        candidate_downgrades,
        elapsed_ms,
        deadline_overrun_ms,
    })
}

fn build_route_plan(
    prepared: &PreparedDenseQ4Routes,
    device: &MlxDevice,
    registry_authority_id: u64,
    activation_epoch: u64,
    decisions: HashMap<DenseQ4Shape, DenseQ4Route>,
) -> Result<Arc<DenseQ4RoutePlan>> {
    let plan_id = plan_id(
        &prepared.build_fingerprint,
        device.registry_id(),
        registry_authority_id,
        activation_epoch,
        &decisions,
    )?;
    Ok(Arc::new(DenseQ4RoutePlan {
        plan_id,
        build_fingerprint: prepared.build_fingerprint.clone(),
        device_name: device.name(),
        device_registry_id: device.registry_id(),
        registry_authority_id,
        activation_epoch,
        decisions,
    }))
}

fn commit_cleanup_boundary(device: &MlxDevice, submissions: &mut u32) -> Result<()> {
    let next = (*submissions).checked_add(1).ok_or_else(|| {
        MlxError::InvalidArgument("dense Q4 calibration submission count overflow".into())
    })?;
    let mut cleanup = device.command_encoder()?;
    *submissions = next;
    cleanup.commit_and_wait()
}

/// Calibrate exact declared Q4_0 shapes and freeze one immutable plan.
///
/// Borrowed weights are used only during this call. Callers must declare every
/// distinct weight buffer and exact row count reachable by a base shape. Every
/// activation proves every exact row-count/current-weight pair. Proof route
/// dispatches share one command buffer per exact shape, while GPU poison and
/// comparison kernels preserve full-overwrite, finite, guard, and bitwise
/// authority for each pair. Any candidate proof failure disables the candidate
/// for the whole base shape. Timing runs once per exact row count on the
/// representative. The process cache retains only timing distributions and
/// their derived route; plans and receipts contain pointer-free shape and
/// pipeline metadata.
fn calibrate_dense_q4_routes_impl(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    limits: DenseQ4CalibrationLimits,
    cases: &[DenseQ4CalibrationCase<'_>],
    status_transform: DenseQ4ProofStatusTransform,
) -> Result<(Arc<DenseQ4RoutePlan>, DenseQ4CalibrationBatchReceipt)> {
    if activation_epoch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense Q4 activation epoch must be nonzero".into(),
        ));
    }
    if registry.dense_q4_plan().is_some() {
        return Err(MlxError::InvalidArgument(
            "dense Q4 registry is already frozen; calibration is one-shot".into(),
        ));
    }
    let validated = validate_cases(cases, limits)?;
    let started = Instant::now();
    let prepared = prepare_routes(registry, device)?;
    let declared_shapes: usize = validated.iter().map(|case| case.reachable_m.len()).sum();
    let mut decisions = Vec::with_capacity(declared_shapes);
    let mut plan_decisions = HashMap::with_capacity(declared_shapes);
    let mut calibrated_decisions = 0u32;
    let mut process_cache_hits = 0u32;
    let mut calibration_submissions = 0u32;

    for case in validated {
        let authorized_weight_buffers = u32::try_from(case.weights.len()).map_err(|_| {
            MlxError::InvalidArgument("dense Q4 authorized weight count overflow".into())
        })?;
        let mut base_outcomes = Vec::with_capacity(case.reachable_m.len());
        for m in case.reachable_m {
            let shape = case.shape.with_m(m);
            let representative_weight = case.weights[0];
            let mut proof_metrics = DenseQ4ProofMetrics::default();
            let proof = match prove_current_weights(
                registry,
                device,
                &case.weights,
                shape,
                &prepared,
                started,
                limits,
                &mut proof_metrics,
                status_transform,
            ) {
                Ok(proof) => proof,
                Err(error) => {
                    calibration_submissions = calibration_submissions
                        .checked_add(proof_metrics.submissions)
                        .ok_or_else(|| {
                            MlxError::InvalidArgument(
                                "dense Q4 calibration submission count overflow".into(),
                            )
                        })?;
                    commit_cleanup_boundary(device, &mut calibration_submissions)?;
                    return Err(MlxError::InvalidArgument(format!(
                        "required dense Q4 compatibility V2 proof failed: {error}"
                    )));
                }
            };
            let candidate_proven = matches!(&proof, DenseQ4ProofOutcome::Coherent(_));
            let decision = match proof {
                DenseQ4ProofOutcome::Fallback(decision) => decision,
                DenseQ4ProofOutcome::Coherent(proof) => {
                    let key = DenseQ4ProcessKey {
                        build_fingerprint: prepared.build_fingerprint.clone(),
                        device_name: device.name(),
                        device_registry_id: device.registry_id(),
                        pipeline_set_fingerprint: prepared.pipeline_set_fingerprint.clone(),
                        shape,
                    };
                    let cell = match process_cell(key.clone()) {
                        Ok(cell) => cell,
                        Err(error) => {
                            calibration_submissions = calibration_submissions
                                .checked_add(proof.metrics.submissions)
                                .ok_or_else(|| {
                                    MlxError::InvalidArgument(
                                        "dense Q4 calibration submission count overflow".into(),
                                    )
                                })?;
                            drop(proof);
                            commit_cleanup_boundary(device, &mut calibration_submissions)?;
                            return Err(error);
                        }
                    };
                    let initialized_here = Cell::new(false);
                    let cached = cell.get_or_init(|| {
                        initialized_here.set(true);
                        measure_timing_decision(
                            registry,
                            device,
                            representative_weight,
                            shape,
                            &proof,
                            started,
                            limits,
                        )
                    });
                    match cached {
                        Ok(cached) => {
                            let cache_hit = !initialized_here.get();
                            if cache_hit {
                                process_cache_hits += 1;
                            } else {
                                calibrated_decisions += 1;
                            }
                            match materialize_timing_decision(shape, &proof, cached, cache_hit) {
                                Ok(decision) => decision,
                                Err(error) => {
                                    let current_submissions = proof
                                        .metrics
                                        .submissions
                                        .checked_add(if cache_hit {
                                            0
                                        } else {
                                            cached.timing_submissions
                                        })
                                        .ok_or_else(|| {
                                            MlxError::InvalidArgument(
                                                "dense Q4 calibration submission count overflow"
                                                    .into(),
                                            )
                                        })?;
                                    calibration_submissions = calibration_submissions
                                        .checked_add(current_submissions)
                                        .ok_or_else(|| {
                                            MlxError::InvalidArgument(
                                                "dense Q4 calibration submission count overflow"
                                                    .into(),
                                            )
                                        })?;
                                    drop(proof);
                                    commit_cleanup_boundary(device, &mut calibration_submissions)?;
                                    return Err(error);
                                }
                            }
                        }
                        Err(failure) => {
                            let timing_submissions = if initialized_here.get() {
                                failure.timing_submissions
                            } else {
                                0
                            };
                            let eviction = evict_process_cell_if_same(&key, &cell);
                            if failure.status == DenseQ4SelectionStatus::BudgetFallback {
                                if let Err(error) = eviction {
                                    let current_submissions = proof
                                        .metrics
                                        .submissions
                                        .checked_add(timing_submissions)
                                        .ok_or_else(|| {
                                            MlxError::InvalidArgument(
                                                "dense Q4 calibration submission count overflow"
                                                    .into(),
                                            )
                                        })?;
                                    calibration_submissions = calibration_submissions
                                        .checked_add(current_submissions)
                                        .ok_or_else(|| {
                                            MlxError::InvalidArgument(
                                                "dense Q4 calibration submission count overflow"
                                                    .into(),
                                            )
                                        })?;
                                    drop(proof);
                                    commit_cleanup_boundary(device, &mut calibration_submissions)?;
                                    return Err(error);
                                }
                                compatibility_decision(
                                    shape,
                                    failure.status,
                                    proof.metrics,
                                    timing_submissions,
                                    Some(failure.message.clone()),
                                )
                            } else {
                                let current_submissions = proof
                                    .metrics
                                    .submissions
                                    .checked_add(timing_submissions)
                                    .ok_or_else(|| {
                                        MlxError::InvalidArgument(
                                            "dense Q4 calibration submission count overflow".into(),
                                        )
                                    })?;
                                calibration_submissions = calibration_submissions
                                    .checked_add(current_submissions)
                                    .ok_or_else(|| {
                                        MlxError::InvalidArgument(
                                            "dense Q4 calibration submission count overflow".into(),
                                        )
                                    })?;
                                drop(proof);
                                commit_cleanup_boundary(device, &mut calibration_submissions)?;
                                eviction?;
                                return Err(MlxError::InvalidArgument(format!(
                                    "dense Q4 timing calibration failed after current-weight proof: {}",
                                    failure.message
                                )));
                            }
                        }
                    }
                }
            };
            let mut decision = decision;
            // A fallback selected before a proof submission is executable but
            // owns no activation-time proof authority. Keep that distinction
            // visible in the receipt so an all-fallback run cannot publish
            // declared pairs as proved pairs.
            decision.authorized_weight_buffers = if decision.proof_submissions == 0 {
                0
            } else {
                authorized_weight_buffers
            };
            calibration_submissions = calibration_submissions
                .checked_add(decision.calibration_submissions)
                .ok_or_else(|| {
                    MlxError::InvalidArgument(
                        "dense Q4 calibration submission count overflow".into(),
                    )
                })?;
            base_outcomes.push((decision, candidate_proven));
        }
        let (base_decisions, base_plan_decisions) = finalize_base_shape_decisions(base_outcomes);
        plan_decisions.extend(base_plan_decisions);
        decisions.extend(base_decisions);
    }
    decisions.sort_by_key(|decision| decision.shape);
    let authorized_shape_weight_pairs = decisions.iter().try_fold(0u32, |total, decision| {
        total
            .checked_add(decision.authorized_weight_buffers)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense Q4 authorized weight count overflow".into())
            })
    })?;
    let proof_submissions = decisions.iter().try_fold(0u32, |total, decision| {
        total
            .checked_add(decision.proof_submissions)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense Q4 proof submission count overflow".into())
            })
    })?;
    let proof_route_dispatches = decisions.iter().try_fold(0u32, |total, decision| {
        total
            .checked_add(decision.proof_route_dispatches)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense Q4 proof route dispatch count overflow".into())
            })
    })?;
    let proof_auxiliary_dispatches = decisions.iter().try_fold(0u32, |total, decision| {
        total
            .checked_add(decision.proof_auxiliary_dispatches)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense Q4 proof auxiliary dispatch count overflow".into())
            })
    })?;
    let peak_proof_scratch_bytes = decisions
        .iter()
        .map(|decision| decision.proof_scratch_bytes)
        .max()
        .unwrap_or(0);
    let proof_gpu_us = decisions
        .iter()
        .map(|decision| decision.proof_gpu_us)
        .sum::<f64>();
    let timing_submissions = decisions.iter().try_fold(0u32, |total, decision| {
        total
            .checked_add(decision.timing_submissions)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense Q4 timing submission count overflow".into())
            })
    })?;
    if proof_submissions
        .checked_add(timing_submissions)
        .ok_or_else(|| {
            MlxError::InvalidArgument("dense Q4 calibration submission count overflow".into())
        })?
        != calibration_submissions
    {
        return Err(MlxError::InvalidArgument(
            "dense Q4 aggregate submission receipt is inconsistent".into(),
        ));
    }
    // The last proof/timing scratch buffers have dropped. Commit one empty
    // boundary so deferred residency removals cannot leak into the first
    // request or the next model activation.
    let cleanup_submissions = 1;
    commit_cleanup_boundary(device, &mut calibration_submissions)?;

    // Build and fully validate both possible outcomes before taking the
    // authoritative deadline sample. The selected plan can then be installed
    // without compilation, hashing, rebuilding, or another fallible check.
    let candidate_compatibility_route_decisions =
        validate_receipt_plan_routes(&decisions, &plan_decisions)?;
    let candidate_plan = build_route_plan(
        &prepared,
        device,
        registry.dense_q4_auto.registry_authority_id(),
        activation_epoch,
        plan_decisions.clone(),
    )?;
    registry.validate_dense_q4_plan(device, &candidate_plan)?;
    let mut compatibility_decisions = decisions.clone();
    let mut compatibility_plan_decisions = plan_decisions.clone();
    let (compatibility_route_decisions, candidate_downgrades) = prepare_budget_fallback(
        &mut compatibility_decisions,
        &mut compatibility_plan_decisions,
        "final dense Q4 calibration deadline exceeded; candidate disabled",
    )?;
    let compatibility_plan = build_route_plan(
        &prepared,
        device,
        registry.dense_q4_auto.registry_authority_id(),
        activation_epoch,
        compatibility_plan_decisions.clone(),
    )?;
    registry.validate_dense_q4_plan(device, &compatibility_plan)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let deadline_overrun_ms = (elapsed_ms - limits.max_elapsed_ms as f64).max(0.0);
    let (plan, final_budget) = if deadline_overrun_ms > 0.0 {
        decisions = compatibility_decisions;
        (
            compatibility_plan,
            DenseQ4FinalBudget {
                compatibility_route_decisions,
                candidate_downgrades,
                elapsed_ms,
                deadline_overrun_ms,
            },
        )
    } else {
        (
            candidate_plan,
            DenseQ4FinalBudget {
                compatibility_route_decisions: candidate_compatibility_route_decisions,
                candidate_downgrades: 0,
                elapsed_ms,
                deadline_overrun_ms,
            },
        )
    };
    registry.install_prevalidated_dense_q4_plan(plan.clone())?;
    let plan_id = plan.plan_id.clone();
    Ok((
        plan,
        DenseQ4CalibrationBatchReceipt {
            schema_version: DENSE_Q4_ROUTE_SCHEMA_VERSION,
            mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: prepared.build_fingerprint,
            plan_id,
            activation_epoch,
            device_name: device.name(),
            device_registry_id: device.registry_id(),
            registry_authority_id: registry.dense_q4_auto.registry_authority_id(),
            declared_shapes: declared_shapes as u32,
            calibrated_decisions,
            process_cache_hits,
            compatibility_route_decisions: final_budget.compatibility_route_decisions,
            authorized_shape_weight_pairs,
            proof_submissions,
            proof_route_dispatches,
            proof_auxiliary_dispatches,
            peak_proof_scratch_bytes,
            proof_gpu_us,
            timing_submissions,
            cleanup_submissions,
            calibration_submissions,
            elapsed_ms: final_budget.elapsed_ms,
            deadline_overrun_ms: final_budget.deadline_overrun_ms,
            decisions,
        },
    ))
}

/// Calibrate every declared exact Q4_0 shape/current-weight pair and freeze
/// one immutable, pointer-free route plan.
pub fn calibrate_dense_q4_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    limits: DenseQ4CalibrationLimits,
    cases: &[DenseQ4CalibrationCase<'_>],
) -> Result<(Arc<DenseQ4RoutePlan>, DenseQ4CalibrationBatchReceipt)> {
    calibrate_dense_q4_routes_impl(
        registry,
        device,
        activation_epoch,
        limits,
        cases,
        identity_proof_status,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::DispatchKind;
    use crate::kernel_registry::{KernelPipelineOrigin, KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION};
    use half::f16;

    fn test_q4_weight(device: &MlxDevice, n: usize, k: usize, salt: u8) -> MlxBuffer {
        let mut weight = device
            .alloc_buffer(n * (k / 32) * 18, DType::U8, vec![n * (k / 32) * 18])
            .expect("allocate test Q4 weight");
        for (block_index, block) in weight
            .as_mut_slice::<u8>()
            .expect("map test Q4 weight")
            .chunks_exact_mut(18)
            .enumerate()
        {
            block[..2].copy_from_slice(
                &f16::from_f32(0.015625 + f32::from(salt) / 8192.0)
                    .to_bits()
                    .to_le_bytes(),
            );
            for (byte_index, packed) in block[2..].iter_mut().enumerate() {
                let low = (block_index + byte_index + usize::from(salt)) % 15 + 1;
                let high = (block_index * 3 + byte_index * 5 + usize::from(salt)) % 15 + 1;
                *packed = low as u8 | ((high as u8) << 4);
            }
        }
        weight
    }

    fn inject_additional_weight_candidate_failure(
        shape: DenseQ4Shape,
        weight_index: usize,
        status: u32,
    ) -> u32 {
        if shape.m == 9 && weight_index == 1 {
            status | PROOF_MISMATCH
        } else {
            status
        }
    }

    fn timing(
        route: DenseQ4Route,
        wall: (f64, f64, f64),
        gpu: (f64, f64, f64),
    ) -> DenseQ4RouteTiming {
        let pipeline_label = route.pipeline_label();
        DenseQ4RouteTiming {
            route,
            wall: DenseQ4TimingDistribution {
                p25_us: wall.0,
                median_us: wall.1,
                p75_us: wall.2,
                samples: CALIBRATION_SAMPLES as u32,
            },
            gpu: DenseQ4TimingDistribution {
                p25_us: gpu.0,
                median_us: gpu.1,
                p75_us: gpu.2,
                samples: CALIBRATION_SAMPLES as u32,
            },
            encoded: EncodedKernelDispatch {
                pipeline_label: pipeline_label.clone(),
                dispatch_kind: DispatchKind::ThreadGroups,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
                threadgroup_memory: Vec::new(),
            },
            pipeline: KernelPipelineIdentity {
                schema_version: KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION,
                pipeline_label,
                kernel_name: route.kernel_name().into(),
                origin: KernelPipelineOrigin::RuntimeSource,
                runtime_source_sha256: Some("test".into()),
                embedded_metallib_sha256: None,
                precise_fp32_math: true,
                threadgroup_size_multiple_hint: true,
            },
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn candidate_only_additional_weight_failure_downgrades_base_and_cleans_up() {
        let device = MlxDevice::new().expect("Metal device");
        let first = test_q4_weight(&device, 75, 192, 43);
        let second = test_q4_weight(&device, 75, 192, 47);
        let mut registry = KernelRegistry::new();
        let (_plan, receipt) = calibrate_dense_q4_routes_impl(
            &mut registry,
            &device,
            901,
            DenseQ4CalibrationLimits {
                max_elapsed_ms: 20_000,
                max_shapes: 2,
            },
            &[
                DenseQ4CalibrationCase {
                    weight: &first,
                    shape: DenseQ4BaseShape {
                        n: 75,
                        k: 192,
                        batch: 1,
                        input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
                    },
                    reachable_m: &[9, 37],
                },
                DenseQ4CalibrationCase {
                    weight: &second,
                    shape: DenseQ4BaseShape {
                        n: 75,
                        k: 192,
                        batch: 1,
                        input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
                    },
                    reachable_m: &[9, 37],
                },
            ],
            inject_additional_weight_candidate_failure,
        )
        .expect("candidate-only proof failure must fail closed to V2");

        assert_eq!(receipt.authorized_shape_weight_pairs, 4);
        assert_eq!(receipt.proof_submissions, 2);
        assert_eq!(receipt.proof_route_dispatches, 8);
        assert_eq!(receipt.proof_auxiliary_dispatches, 8);
        assert_eq!(receipt.timing_submissions, 10);
        assert_eq!(receipt.cleanup_submissions, 1);
        assert_eq!(receipt.calibration_submissions, 13);
        assert_eq!(receipt.compatibility_route_decisions, 2);
        assert!(receipt.proof_gpu_us > 0.0);
        assert!(receipt.decisions.iter().all(|decision| {
            decision.selected_route == DenseQ4Route::CompatibilityV2
                && decision.status == DenseQ4SelectionStatus::IncoherentCandidate
                && decision
                    .diagnostic
                    .as_deref()
                    .is_some_and(|diagnostic| diagnostic.contains("weight 1"))
        }));
    }

    #[test]
    fn base_shape_finalizer_downgrades_plan_and_conserves_receipt_metrics_without_hardware() {
        let shape = |m| DenseQ4Shape {
            m,
            n: 75,
            k: 192,
            batch: 1,
            input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
        };
        let decision = |m, status, diagnostic, salt: u32| DenseQ4CalibrationDecision {
            shape: shape(m),
            selected_route: DenseQ4Route::Tensor64x32,
            status,
            diagnostic,
            timings: vec![timing(
                DenseQ4Route::Tensor64x32,
                (1.0 + f64::from(salt), 2.0, 3.0),
                (4.0, 5.0 + f64::from(salt), 6.0),
            )],
            process_cache_hit: salt % 2 == 0,
            authorized_weight_buffers: 2 + salt,
            proof_submissions: 1 + salt,
            proof_route_dispatches: 4 + salt,
            proof_auxiliary_dispatches: 5 + salt,
            proof_scratch_bytes: 1024 + u64::from(salt),
            proof_gpu_us: 1.0 + f64::from(salt),
            timing_submissions: 10 + salt,
            calibration_submissions: 11 + salt * 2,
        };
        let base_outcomes = vec![
            (
                decision(
                    9,
                    DenseQ4SelectionStatus::CalibratedWinner,
                    Some("existing".into()),
                    1,
                ),
                true,
            ),
            (
                decision(37, DenseQ4SelectionStatus::IncoherentCandidate, None, 2),
                false,
            ),
            (
                decision(
                    65,
                    DenseQ4SelectionStatus::CandidateUnavailable,
                    Some("later failure must not win".into()),
                    3,
                ),
                false,
            ),
        ];
        let originals = base_outcomes
            .iter()
            .map(|(decision, _)| decision.clone())
            .collect::<Vec<_>>();
        let (decisions, plan_decisions) = finalize_base_shape_decisions(base_outcomes);

        assert_eq!(decisions.len(), 3);
        assert_eq!(plan_decisions.len(), 3);
        assert!(decisions.iter().all(|decision| {
            decision.selected_route == DenseQ4Route::CompatibilityV2
                && decision.status == DenseQ4SelectionStatus::IncoherentCandidate
                && decision
                    .diagnostic
                    .as_deref()
                    .is_some_and(|diagnostic| diagnostic.contains("exact M=37"))
        }));
        assert!(decisions[0]
            .diagnostic
            .as_deref()
            .is_some_and(|diagnostic| diagnostic.starts_with("existing;")));
        assert!(plan_decisions
            .values()
            .all(|route| *route == DenseQ4Route::CompatibilityV2));

        for (before, after) in originals.iter().zip(&decisions) {
            assert_eq!(after.shape, before.shape);
            assert_eq!(after.timings, before.timings);
            assert_eq!(after.process_cache_hit, before.process_cache_hit);
            assert_eq!(
                after.authorized_weight_buffers,
                before.authorized_weight_buffers
            );
            assert_eq!(after.proof_submissions, before.proof_submissions);
            assert_eq!(after.proof_route_dispatches, before.proof_route_dispatches);
            assert_eq!(
                after.proof_auxiliary_dispatches,
                before.proof_auxiliary_dispatches
            );
            assert_eq!(after.proof_scratch_bytes, before.proof_scratch_bytes);
            assert_eq!(after.proof_gpu_us, before.proof_gpu_us);
            assert_eq!(after.timing_submissions, before.timing_submissions);
            assert_eq!(
                after.calibration_submissions,
                before.calibration_submissions
            );
        }
    }

    #[test]
    fn selector_requires_material_stable_wall_win_and_no_contrary_gpu() {
        let clear = [
            timing(
                DenseQ4Route::CompatibilityV2,
                (99.0, 100.0, 101.0),
                (89.0, 90.0, 91.0),
            ),
            timing(
                DenseQ4Route::Tensor64x32,
                (69.0, 70.0, 71.0),
                (59.0, 60.0, 61.0),
            ),
        ];
        assert_eq!(
            select_route(&clear).expect("clear winner"),
            (
                DenseQ4Route::Tensor64x32,
                DenseQ4SelectionStatus::CalibratedWinner
            )
        );

        let overlapping = [
            timing(
                DenseQ4Route::CompatibilityV2,
                (95.0, 100.0, 105.0),
                (89.0, 90.0, 91.0),
            ),
            timing(
                DenseQ4Route::Tensor64x32,
                (90.0, 94.0, 101.0),
                (79.0, 80.0, 81.0),
            ),
        ];
        assert_eq!(
            select_route(&overlapping).expect("overlapping winner"),
            (
                DenseQ4Route::CompatibilityV2,
                DenseQ4SelectionStatus::NoStableWinner
            )
        );

        let contrary_gpu = [
            timing(
                DenseQ4Route::CompatibilityV2,
                (99.0, 100.0, 101.0),
                (89.0, 90.0, 91.0),
            ),
            timing(
                DenseQ4Route::Tensor64x32,
                (69.0, 70.0, 71.0),
                (99.0, 100.0, 101.0),
            ),
        ];
        assert_eq!(
            select_route(&contrary_gpu).expect("contrary GPU"),
            (
                DenseQ4Route::CompatibilityV2,
                DenseQ4SelectionStatus::NoStableWinner
            )
        );
    }

    #[test]
    fn final_budget_gate_deterministically_removes_every_candidate_authority() {
        let candidate_shape = DenseQ4Shape {
            m: 32,
            n: 768,
            k: 768,
            batch: 1,
            input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
        };
        let compatibility_shape = DenseQ4Shape {
            m: 129,
            ..candidate_shape
        };
        let mut decisions = vec![
            DenseQ4CalibrationDecision {
                shape: candidate_shape,
                selected_route: DenseQ4Route::Tensor64x32,
                status: DenseQ4SelectionStatus::CalibratedWinner,
                diagnostic: None,
                timings: Vec::new(),
                process_cache_hit: false,
                authorized_weight_buffers: 1,
                proof_submissions: 2,
                proof_route_dispatches: 2,
                proof_auxiliary_dispatches: 2,
                proof_scratch_bytes: 1024,
                proof_gpu_us: 100.0,
                timing_submissions: 10,
                calibration_submissions: 12,
            },
            DenseQ4CalibrationDecision {
                shape: compatibility_shape,
                selected_route: DenseQ4Route::CompatibilityV2,
                status: DenseQ4SelectionStatus::CompatibilityFastest,
                diagnostic: Some("measured compatibility winner".into()),
                timings: Vec::new(),
                process_cache_hit: true,
                authorized_weight_buffers: 1,
                proof_submissions: 2,
                proof_route_dispatches: 2,
                proof_auxiliary_dispatches: 2,
                proof_scratch_bytes: 1024,
                proof_gpu_us: 100.0,
                timing_submissions: 0,
                calibration_submissions: 2,
            },
        ];
        let mut plan_decisions = HashMap::from([
            (candidate_shape, DenseQ4Route::Tensor64x32),
            (compatibility_shape, DenseQ4Route::CompatibilityV2),
        ]);

        let final_budget = enforce_final_budget(&mut decisions, &mut plan_decisions, 10.25, 10)
            .expect("final budget gate");

        assert_eq!(final_budget.deadline_overrun_ms, 0.25);
        assert_eq!(final_budget.compatibility_route_decisions, 2);
        assert_eq!(final_budget.candidate_downgrades, 1);
        assert!(decisions
            .iter()
            .all(|decision| decision.selected_route == DenseQ4Route::CompatibilityV2));
        assert!(plan_decisions
            .values()
            .all(|route| *route == DenseQ4Route::CompatibilityV2));
        assert_eq!(decisions[0].status, DenseQ4SelectionStatus::BudgetFallback);
        assert!(decisions[0]
            .diagnostic
            .as_deref()
            .is_some_and(|diagnostic| diagnostic.contains("candidate disabled")));
        assert_eq!(
            decisions[1].status,
            DenseQ4SelectionStatus::CompatibilityFastest
        );
        assert_eq!(
            decisions[1].diagnostic.as_deref(),
            Some("measured compatibility winner")
        );
    }

    #[test]
    fn final_budget_gate_preserves_candidate_only_without_overrun() {
        let shape = DenseQ4Shape {
            m: 32,
            n: 768,
            k: 768,
            batch: 1,
            input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
        };
        let mut decisions = vec![DenseQ4CalibrationDecision {
            shape,
            selected_route: DenseQ4Route::Tensor64x32,
            status: DenseQ4SelectionStatus::CalibratedWinner,
            diagnostic: None,
            timings: Vec::new(),
            process_cache_hit: false,
            authorized_weight_buffers: 1,
            proof_submissions: 2,
            proof_route_dispatches: 2,
            proof_auxiliary_dispatches: 2,
            proof_scratch_bytes: 1024,
            proof_gpu_us: 100.0,
            timing_submissions: 10,
            calibration_submissions: 12,
        }];
        let mut plan_decisions = HashMap::from([(shape, DenseQ4Route::Tensor64x32)]);

        let final_budget = enforce_final_budget(&mut decisions, &mut plan_decisions, 10.0, 10)
            .expect("within-budget final gate");

        assert_eq!(final_budget.deadline_overrun_ms, 0.0);
        assert_eq!(final_budget.compatibility_route_decisions, 0);
        assert_eq!(final_budget.candidate_downgrades, 0);
        assert_eq!(decisions[0].selected_route, DenseQ4Route::Tensor64x32);
        assert_eq!(plan_decisions.get(&shape), Some(&DenseQ4Route::Tensor64x32));
    }

    #[test]
    fn activation_epoch_changes_pointer_free_plan_identity() {
        let shape = DenseQ4Shape {
            m: 32,
            n: 768,
            k: 768,
            batch: 1,
            input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
        };
        let decisions = HashMap::from([(shape, DenseQ4Route::Tensor64x32)]);
        let first = plan_id("build", 17, 31, 1, &decisions).expect("first plan id");
        let second = plan_id("build", 17, 31, 2, &decisions).expect("second plan id");
        assert_ne!(first, second);
    }

    #[test]
    fn retryable_process_cells_are_evicted_without_weight_identity() {
        let key = DenseQ4ProcessKey {
            build_fingerprint: "test-build".into(),
            device_name: "test-device".into(),
            device_registry_id: u64::MAX,
            pipeline_set_fingerprint: "test-pipelines".into(),
            shape: DenseQ4Shape {
                m: 9,
                n: 65,
                k: 96,
                batch: 1,
                input_layout: super::super::dense_q4_auto::DenseQ4InputLayout::Contiguous,
            },
        };
        let failed = process_cell(key.clone()).expect("failed cell");
        failed
            .set(Err(DenseQ4CalibrationFailure {
                status: DenseQ4SelectionStatus::CalibrationErrorFallback,
                message: "transient".into(),
                timing_submissions: 0,
            }))
            .expect("initialize failed cell");
        evict_process_cell_if_same(&key, &failed).expect("evict failed cell");
        let retry = process_cell(key.clone()).expect("retry cell");
        assert!(!Arc::ptr_eq(&failed, &retry));
        evict_process_cell_if_same(&key, &retry).expect("clean retry cell");
    }
}
