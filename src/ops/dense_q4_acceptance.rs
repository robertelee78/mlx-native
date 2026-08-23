//! Fail-closed acceptance checks for dense Q4 Cartesian qualification receipts.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{MlxError, Result};
use crate::kernel_registry::{KernelPipelineOrigin, KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION};
use crate::ops::dense_q4_auto::{
    expected_dispatch, DenseQ4BaseShape, DenseQ4CalibrationBatchReceipt,
    DenseQ4CalibrationDecision, DenseQ4Route, DenseQ4SelectionStatus, DenseQ4Shape,
    DENSE_Q4_ROUTE_SCHEMA_VERSION,
};
use crate::ops::dense_q4_calibration::CALIBRATION_SAMPLES;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseQ4CartesianAcceptanceRequirements {
    pub expected_base_shapes: u32,
    pub expected_weight_buffers_per_base: u32,
    pub reachable_m: Vec<u32>,
    pub required_compatibility_m: Vec<u32>,
    pub minimum_candidate_decisions: u32,
    pub maximum_elapsed_ms: u64,
}

fn acceptance_error(message: impl Into<String>) -> MlxError {
    MlxError::InvalidArgument(format!(
        "dense Q4 Cartesian acceptance failed: {}",
        message.into()
    ))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_product(label: &str, values: &[u32]) -> Result<u32> {
    values.iter().try_fold(1u32, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| acceptance_error(format!("{label} count overflow")))
    })
}

fn validate_requirements(
    requirements: &DenseQ4CartesianAcceptanceRequirements,
) -> Result<(u32, u32, BTreeSet<u32>, BTreeSet<u32>)> {
    if requirements.expected_base_shapes == 0
        || requirements.expected_weight_buffers_per_base == 0
        || requirements.reachable_m.is_empty()
        || requirements.minimum_candidate_decisions == 0
        || requirements.maximum_elapsed_ms == 0
    {
        return Err(acceptance_error(
            "base shapes, current weights, reachable rows, candidate minimum, and wall-time ceiling must be nonzero",
        ));
    }
    let reachable_m: BTreeSet<_> = requirements.reachable_m.iter().copied().collect();
    if reachable_m.len() != requirements.reachable_m.len() || reachable_m.iter().any(|&m| m <= 8) {
        return Err(acceptance_error(
            "reachable rows must be unique eligible MM widths",
        ));
    }
    let required_compatibility_m: BTreeSet<_> = requirements
        .required_compatibility_m
        .iter()
        .copied()
        .collect();
    if required_compatibility_m.len() != requirements.required_compatibility_m.len()
        || !required_compatibility_m.is_subset(&reachable_m)
    {
        return Err(acceptance_error(
            "required compatibility rows must be a unique subset of reachable rows",
        ));
    }
    let reachable_count = u32::try_from(reachable_m.len())
        .map_err(|_| acceptance_error("reachable row count overflow"))?;
    let declared_shapes = checked_product(
        "declared shape",
        &[requirements.expected_base_shapes, reachable_count],
    )?;
    if requirements.minimum_candidate_decisions > declared_shapes {
        return Err(acceptance_error(
            "minimum candidate decisions exceeds declared shapes",
        ));
    }
    let authorized_pairs = checked_product(
        "authorized shape/current-weight pair",
        &[
            declared_shapes,
            requirements.expected_weight_buffers_per_base,
        ],
    )?;
    Ok((
        declared_shapes,
        authorized_pairs,
        reachable_m,
        required_compatibility_m,
    ))
}

fn validate_distribution(
    label: &str,
    distribution: &super::dense_q4_auto::DenseQ4TimingDistribution,
) -> Result<()> {
    if distribution.samples != CALIBRATION_SAMPLES as u32
        || !distribution.p25_us.is_finite()
        || !distribution.median_us.is_finite()
        || !distribution.p75_us.is_finite()
        || distribution.p25_us <= 0.0
        || distribution.p25_us > distribution.median_us
        || distribution.median_us > distribution.p75_us
    {
        return Err(acceptance_error(format!(
            "{label} timing distribution is incomplete or invalid"
        )));
    }
    Ok(())
}

fn validate_decision(
    decision: &DenseQ4CalibrationDecision,
    expected_weight_buffers: u32,
    expect_cache_hit: bool,
) -> Result<()> {
    let expected_route_dispatches = expected_weight_buffers
        .checked_mul(2)
        .ok_or_else(|| acceptance_error("per-shape route dispatch count overflow"))?;
    let expected_timing_submissions = if expect_cache_hit {
        0
    } else {
        (CALIBRATION_SAMPLES as u32) * 2
    };
    if decision.authorized_weight_buffers != expected_weight_buffers
        || decision.proof_submissions != 1
        || decision.proof_route_dispatches != expected_route_dispatches
        || decision.proof_auxiliary_dispatches != expected_route_dispatches
        || decision.timing_submissions != expected_timing_submissions
        || decision.calibration_submissions
            != decision.proof_submissions + decision.timing_submissions
        || decision.process_cache_hit != expect_cache_hit
        || decision.proof_scratch_bytes == 0
        || !decision.proof_gpu_us.is_finite()
        || decision.proof_gpu_us <= 0.0
    {
        return Err(acceptance_error(format!(
            "shape {:?} has incomplete or inconsistent Cartesian proof counts",
            decision.shape
        )));
    }
    if !matches!(
        (decision.selected_route, decision.status),
        (
            DenseQ4Route::Tensor64x32,
            DenseQ4SelectionStatus::CalibratedWinner
        ) | (
            DenseQ4Route::CompatibilityV2,
            DenseQ4SelectionStatus::CompatibilityFastest | DenseQ4SelectionStatus::NoStableWinner
        )
    ) || decision.diagnostic.is_some()
    {
        return Err(acceptance_error(format!(
            "shape {:?} contains a fallback or inconsistent selection status",
            decision.shape
        )));
    }
    if decision.timings.len() != 2 {
        return Err(acceptance_error(format!(
            "shape {:?} does not carry both route timings",
            decision.shape
        )));
    }
    let timing_routes: BTreeSet<_> = decision.timings.iter().map(|timing| timing.route).collect();
    if timing_routes != BTreeSet::from([DenseQ4Route::CompatibilityV2, DenseQ4Route::Tensor64x32]) {
        return Err(acceptance_error(format!(
            "shape {:?} timing routes are incomplete",
            decision.shape
        )));
    }
    for timing in &decision.timings {
        validate_distribution("wall", &timing.wall)?;
        validate_distribution("GPU", &timing.gpu)?;
        let pipeline_hashes_are_bound = match timing.pipeline.origin {
            KernelPipelineOrigin::RuntimeSource => {
                timing
                    .pipeline
                    .runtime_source_sha256
                    .as_deref()
                    .is_some_and(is_sha256)
                    && timing.pipeline.embedded_metallib_sha256.is_none()
            }
            KernelPipelineOrigin::PrecompiledMetallib => {
                timing
                    .pipeline
                    .embedded_metallib_sha256
                    .as_deref()
                    .is_some_and(is_sha256)
                    && timing.pipeline.runtime_source_sha256.is_none()
            }
        };
        if timing.encoded != expected_dispatch(timing.route, decision.shape)
            || timing.pipeline.schema_version != KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION
            || timing.pipeline.pipeline_label != timing.encoded.pipeline_label
            || timing.pipeline.kernel_name != timing.route.kernel_name()
            || !pipeline_hashes_are_bound
        {
            return Err(acceptance_error(format!(
                "shape {:?} has an unbound route dispatch or pipeline identity",
                decision.shape
            )));
        }
    }
    Ok(())
}

fn validate_receipt(
    label: &str,
    receipt: &DenseQ4CalibrationBatchReceipt,
    requirements: &DenseQ4CartesianAcceptanceRequirements,
    declared_shapes: u32,
    authorized_pairs: u32,
    reachable_m: &BTreeSet<u32>,
    required_compatibility_m: &BTreeSet<u32>,
    expect_cache_hit: bool,
) -> Result<BTreeMap<DenseQ4Shape, (DenseQ4Route, DenseQ4SelectionStatus)>> {
    let expected_route_dispatches = authorized_pairs
        .checked_mul(2)
        .ok_or_else(|| acceptance_error("aggregate route dispatch count overflow"))?;
    let expected_timing_submissions = if expect_cache_hit {
        0
    } else {
        declared_shapes
            .checked_mul((CALIBRATION_SAMPLES as u32) * 2)
            .ok_or_else(|| acceptance_error("aggregate timing submission count overflow"))?
    };
    let expected_cache_hits = if expect_cache_hit { declared_shapes } else { 0 };
    let expected_calibrated = if expect_cache_hit { 0 } else { declared_shapes };
    let expected_calibration_submissions = declared_shapes
        .checked_add(expected_timing_submissions)
        .and_then(|submissions| submissions.checked_add(1))
        .ok_or_else(|| acceptance_error("aggregate calibration submission count overflow"))?;
    if receipt.schema_version != DENSE_Q4_ROUTE_SCHEMA_VERSION
        || receipt.mlx_native_version.is_empty()
        || !is_sha256(&receipt.build_fingerprint)
        || !is_sha256(&receipt.plan_id)
        || receipt.device_name.is_empty()
        || receipt.activation_epoch == 0
        || receipt.registry_authority_id == 0
        || receipt.declared_shapes != declared_shapes
        || receipt.decisions.len() != declared_shapes as usize
        || receipt.calibrated_decisions != expected_calibrated
        || receipt.process_cache_hits != expected_cache_hits
        || receipt.authorized_shape_weight_pairs != authorized_pairs
        || receipt.proof_submissions != declared_shapes
        || receipt.proof_route_dispatches != expected_route_dispatches
        || receipt.proof_auxiliary_dispatches != expected_route_dispatches
        || receipt.timing_submissions != expected_timing_submissions
        || receipt.cleanup_submissions != 1
        || receipt.calibration_submissions != expected_calibration_submissions
        || receipt.peak_proof_scratch_bytes == 0
        || !receipt.proof_gpu_us.is_finite()
        || receipt.proof_gpu_us <= 0.0
        || !receipt.elapsed_ms.is_finite()
        || receipt.elapsed_ms <= 0.0
        || receipt.elapsed_ms > requirements.maximum_elapsed_ms as f64
        || receipt.deadline_overrun_ms != 0.0
    {
        return Err(acceptance_error(format!(
            "{label} receipt is incomplete, over budget, or has inconsistent aggregate counts"
        )));
    }

    let mut decisions = BTreeMap::new();
    let mut base_shapes = BTreeSet::new();
    let mut candidates = 0u32;
    let mut compatibility = 0u32;
    for decision in &receipt.decisions {
        if !reachable_m.contains(&decision.shape.m) {
            return Err(acceptance_error(format!(
                "{label} contains undeclared physical row width {}",
                decision.shape.m
            )));
        }
        validate_decision(
            decision,
            requirements.expected_weight_buffers_per_base,
            expect_cache_hit,
        )?;
        base_shapes.insert(DenseQ4BaseShape {
            n: decision.shape.n,
            k: decision.shape.k,
            batch: decision.shape.batch,
            input_layout: decision.shape.input_layout,
        });
        match decision.selected_route {
            DenseQ4Route::Tensor64x32 => candidates += 1,
            DenseQ4Route::CompatibilityV2 => compatibility += 1,
            other => {
                return Err(acceptance_error(format!(
                    "{label} selected unsupported qualification route {other:?}"
                )))
            }
        }
        if required_compatibility_m.contains(&decision.shape.m)
            && decision.selected_route != DenseQ4Route::CompatibilityV2
        {
            return Err(acceptance_error(format!(
                "{label} selected the short-row candidate at required compatibility M={}",
                decision.shape.m
            )));
        }
        if decisions
            .insert(decision.shape, (decision.selected_route, decision.status))
            .is_some()
        {
            return Err(acceptance_error(format!(
                "{label} contains duplicate exact shape {:?}",
                decision.shape
            )));
        }
    }
    if base_shapes.len() != requirements.expected_base_shapes as usize
        || base_shapes.iter().any(|base| {
            reachable_m
                .iter()
                .any(|&m| !decisions.contains_key(&base.with_m(m)))
        })
    {
        return Err(acceptance_error(format!(
            "{label} does not cover the exact base-shape by physical-row Cartesian product"
        )));
    }
    if candidates < requirements.minimum_candidate_decisions {
        return Err(acceptance_error(format!(
            "{label} retained {candidates} candidate decisions, expected at least {}",
            requirements.minimum_candidate_decisions
        )));
    }
    if compatibility != receipt.compatibility_route_decisions {
        return Err(acceptance_error(format!(
            "{label} compatibility decision count is inconsistent"
        )));
    }
    Ok(decisions)
}

/// Validate cold and timing-cache-reactivation receipts as publishable
/// Cartesian qualification evidence. Runtime fallback remains permissive;
/// only evidence publication is fail-closed.
pub fn validate_dense_q4_cartesian_acceptance(
    cold: &DenseQ4CalibrationBatchReceipt,
    reactivation: &DenseQ4CalibrationBatchReceipt,
    requirements: &DenseQ4CartesianAcceptanceRequirements,
) -> Result<()> {
    let (declared_shapes, authorized_pairs, reachable_m, required_compatibility_m) =
        validate_requirements(requirements)?;
    let cold_decisions = validate_receipt(
        "cold",
        cold,
        requirements,
        declared_shapes,
        authorized_pairs,
        &reachable_m,
        &required_compatibility_m,
        false,
    )?;
    let reactivation_decisions = validate_receipt(
        "reactivation",
        reactivation,
        requirements,
        declared_shapes,
        authorized_pairs,
        &reachable_m,
        &required_compatibility_m,
        true,
    )?;
    if cold.build_fingerprint != reactivation.build_fingerprint
        || cold.mlx_native_version != reactivation.mlx_native_version
        || cold.device_name != reactivation.device_name
        || cold.device_registry_id != reactivation.device_registry_id
        || cold.activation_epoch == reactivation.activation_epoch
        || cold.registry_authority_id == reactivation.registry_authority_id
        || cold.plan_id == reactivation.plan_id
        || cold_decisions != reactivation_decisions
    {
        return Err(acceptance_error(
            "cold/reactivation identity or exact-shape route decisions disagree",
        ));
    }
    Ok(())
}
