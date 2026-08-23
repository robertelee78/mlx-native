//! Pre-serve calibration for the frozen BF16 dense routing plan.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::EncodedKernelDispatch;
use crate::error::{MlxError, Result};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::dense_bf16_auto::{
    encode_route, expected_dispatch, DenseBf16BaseShape, DenseBf16CalibrationBatchReceipt,
    DenseBf16CalibrationCase, DenseBf16CalibrationDecision, DenseBf16CalibrationLimits,
    DenseBf16Route, DenseBf16RoutePlan, DenseBf16RouteTiming, DenseBf16SelectionStatus,
    DenseBf16Shape, DenseBf16TimingDistribution, DENSE_BF16_CALIBRATION_MAX_M,
    DENSE_BF16_ROUTE_SCHEMA_VERSION,
};
use crate::ops::dense_mm_capability::{is_unavailable_tensor_header, tensor_disabled_from_env};

const CALIBRATION_SAMPLES: usize = 5;
const MATERIAL_WIN_FRACTION: f64 = 0.05;
const GPU_CONTRARY_TOLERANCE: f64 = 0.02;
const OUTPUT_GUARD_ELEMENTS: usize = 16;
const OUTPUT_POISON_BITS: u32 = 0x7fc0_00a5;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DenseBf16ProcessKey {
    build_fingerprint: String,
    device_name: String,
    device_registry_id: u64,
    pipeline_set_fingerprint: String,
    shape: DenseBf16Shape,
}

#[derive(Clone, Debug)]
struct DenseBf16CalibrationFailure {
    message: String,
}

type CalibrationCell =
    OnceLock<std::result::Result<DenseBf16CalibrationDecision, DenseBf16CalibrationFailure>>;

fn process_cache() -> &'static Mutex<HashMap<DenseBf16ProcessKey, Arc<CalibrationCell>>> {
    static CACHE: OnceLock<Mutex<HashMap<DenseBf16ProcessKey, Arc<CalibrationCell>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn process_cell(key: DenseBf16ProcessKey) -> Result<Arc<CalibrationCell>> {
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense BF16 calibration cache mutex is poisoned".into())
    })?;
    Ok(cache
        .entry(key)
        .or_insert_with(|| Arc::new(OnceLock::new()))
        .clone())
}

fn evict_process_cell_if_same(
    key: &DenseBf16ProcessKey,
    cell: &Arc<CalibrationCell>,
) -> Result<()> {
    let mut cache = process_cache().lock().map_err(|_| {
        MlxError::InvalidArgument("dense BF16 calibration cache mutex is poisoned".into())
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
            digest.update(DENSE_BF16_ROUTE_SCHEMA_VERSION.to_le_bytes());
            digest.update(include_bytes!("../shaders/dense_gemv_bf16.metal"));
            digest.update(include_bytes!("../shaders/dense_mm_bf16_tensor.metal"));
            digest.update(include_bytes!("../shaders/dense_mm_fallback.metal"));
            hex::encode(digest.finalize())
        })
        .as_str()
}

struct PreparedDenseBf16Routes {
    tensor_available: bool,
    pipeline_set_fingerprint: String,
    build_fingerprint: String,
    identities: HashMap<DenseBf16Route, KernelPipelineIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseBf16EquivalenceClass {
    F32RowReduction,
    TensorV1,
    Simdgroup,
}

fn equivalence_class(route: DenseBf16Route) -> DenseBf16EquivalenceClass {
    match route {
        DenseBf16Route::Row | DenseBf16Route::Tiled4 => DenseBf16EquivalenceClass::F32RowReduction,
        DenseBf16Route::TensorV1 => DenseBf16EquivalenceClass::TensorV1,
        DenseBf16Route::Simdgroup => DenseBf16EquivalenceClass::Simdgroup,
    }
}

fn prepare_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<PreparedDenseBf16Routes> {
    let mut identities = HashMap::new();
    for route in [
        DenseBf16Route::Row,
        DenseBf16Route::Tiled4,
        DenseBf16Route::Simdgroup,
    ] {
        registry.get_pipeline(route.kernel_name(), device.metal_device())?;
        identities.insert(route, registry.pipeline_identity(route.kernel_name())?);
    }
    let tensor_available = if tensor_disabled_from_env() {
        false
    } else {
        match registry.get_pipeline(
            DenseBf16Route::TensorV1.kernel_name(),
            device.metal_device(),
        ) {
            Ok(_) => true,
            Err(error) if is_unavailable_tensor_header(&error) => false,
            Err(error) => return Err(error),
        }
    };
    if tensor_available {
        identities.insert(
            DenseBf16Route::TensorV1,
            registry.pipeline_identity(DenseBf16Route::TensorV1.kernel_name())?,
        );
    }
    let mut ordered: Vec<_> = identities.iter().collect();
    ordered.sort_by_key(|(route, _)| **route);
    let encoded = serde_json::to_vec(&ordered).map_err(|error| {
        MlxError::InvalidArgument(format!("serialize dense BF16 pipeline identities: {error}"))
    })?;
    let pipeline_set_fingerprint = hex::encode(Sha256::digest(encoded));
    let mut build_digest = Sha256::new();
    build_digest.update(source_build_fingerprint().as_bytes());
    build_digest.update(pipeline_set_fingerprint.as_bytes());
    Ok(PreparedDenseBf16Routes {
        tensor_available,
        pipeline_set_fingerprint,
        build_fingerprint: hex::encode(build_digest.finalize()),
        identities,
    })
}

pub(super) fn current_build_fingerprint(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<String> {
    Ok(prepare_routes(registry, device)?.build_fingerprint)
}

fn legal_routes(shape: DenseBf16Shape) -> Vec<DenseBf16Route> {
    let mut routes = Vec::with_capacity(2);
    if shape.k % 4 == 0 {
        routes.push(DenseBf16Route::Row);
        routes.push(DenseBf16Route::Tiled4);
    } else if shape.k >= 32 {
        routes.push(DenseBf16Route::Simdgroup);
    }
    routes
}

fn compatibility_route(
    shape: DenseBf16Shape,
    prepared: &PreparedDenseBf16Routes,
) -> Result<DenseBf16Route> {
    if shape.m <= DENSE_BF16_CALIBRATION_MAX_M && shape.k % 4 == 0 {
        Ok(DenseBf16Route::Row)
    } else if prepared.tensor_available && shape.k >= 32 && shape.k % 4 == 0 {
        Ok(DenseBf16Route::TensorV1)
    } else if shape.k >= 32 {
        Ok(DenseBf16Route::Simdgroup)
    } else if shape.k % 4 == 0 {
        Ok(DenseBf16Route::Tiled4)
    } else {
        Err(MlxError::InvalidArgument(format!(
            "dense BF16 shape {shape:?} has no executable native route"
        )))
    }
}

fn trace_route_and_execute(
    route: DenseBf16Route,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    shape: DenseBf16Shape,
    prepared: &PreparedDenseBf16Routes,
) -> Result<(EncodedKernelDispatch, KernelPipelineIdentity)> {
    let mut encoder = device.command_encoder()?;
    if encoder.device_registry_id() != device.registry_id() {
        return Err(MlxError::InvalidArgument(
            "dense BF16 trace encoder/device mismatch".into(),
        ));
    }
    encoder.start_encoded_dispatch_receipt(1)?;
    let operation = encode_route(
        route,
        &mut encoder,
        registry,
        device,
        weight,
        input,
        output,
        &shape.params(),
    );
    let encoded = encoder.take_encoded_dispatch_receipt();
    operation?;
    let mut encoded = encoded?;
    if encoded.len() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 route {route:?} encoded {} dispatches, expected one",
            encoded.len()
        )));
    }
    let encoded = encoded.remove(0);
    if encoded != expected_dispatch(route, shape) {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 route {route:?} encoded unexpected geometry: {encoded:?}"
        )));
    }
    let pipeline = registry.pipeline_identity(&encoded.pipeline_label)?;
    let expected_pipeline = prepared.identities.get(&route).ok_or_else(|| {
        MlxError::InvalidArgument(format!("dense BF16 route {route:?} was not prepared"))
    })?;
    if &pipeline != expected_pipeline
        || pipeline.pipeline_label != encoded.pipeline_label
        || pipeline.kernel_name != route.kernel_name()
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 route {route:?} pipeline identity changed during activation"
        )));
    }
    encoder.commit_and_wait()?;
    Ok((encoded, pipeline))
}

struct DenseBf16TimedSample {
    wall_us: f64,
    gpu_us: f64,
}

fn time_route(
    route: DenseBf16Route,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    shape: DenseBf16Shape,
) -> Result<DenseBf16TimedSample> {
    let started = Instant::now();
    let mut encoder = device.command_encoder()?;
    encode_route(
        route,
        &mut encoder,
        registry,
        device,
        weight,
        input,
        output,
        &shape.params(),
    )?;
    let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time()?;
    let wall_us = started.elapsed().as_secs_f64() * 1e6;
    if !gpu_start.is_finite()
        || !gpu_end.is_finite()
        || gpu_end <= gpu_start
        || !wall_us.is_finite()
        || wall_us <= 0.0
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 route {route:?} returned an invalid timing interval"
        )));
    }
    Ok(DenseBf16TimedSample {
        wall_us,
        gpu_us: (gpu_end - gpu_start) * 1e6,
    })
}

fn distribution(mut samples: Vec<f64>) -> Result<DenseBf16TimingDistribution> {
    if samples.len() != CALIBRATION_SAMPLES {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 timing has {} samples, expected {CALIBRATION_SAMPLES}",
            samples.len()
        )));
    }
    samples.sort_by(f64::total_cmp);
    Ok(DenseBf16TimingDistribution {
        p25_us: samples[(samples.len() - 1) / 4],
        median_us: samples[samples.len() / 2],
        p75_us: samples[(samples.len() - 1) * 3 / 4],
        samples: samples.len() as u32,
    })
}

fn select_route(
    compatibility: DenseBf16Route,
    timings: &[DenseBf16RouteTiming],
) -> Result<(DenseBf16Route, DenseBf16SelectionStatus)> {
    let baseline = timings
        .iter()
        .find(|timing| timing.route == compatibility)
        .ok_or_else(|| {
            MlxError::InvalidArgument("dense BF16 compatibility route was not timed".into())
        })?;
    let mut ranked: Vec<_> = timings.iter().collect();
    ranked.sort_by(|left, right| left.wall.median_us.total_cmp(&right.wall.median_us));
    let winner = ranked.first().copied().ok_or_else(|| {
        MlxError::InvalidArgument("dense BF16 calibration produced no timings".into())
    })?;
    if winner.route == compatibility {
        return Ok((
            compatibility,
            DenseBf16SelectionStatus::CompatibilityFastest,
        ));
    }
    let material = winner.wall.median_us <= baseline.wall.median_us * (1.0 - MATERIAL_WIN_FRACTION);
    let second = ranked.get(1).copied().unwrap_or(baseline);
    let stable = winner.wall.p75_us < second.wall.p25_us;
    let no_contrary_gpu =
        winner.gpu.median_us <= baseline.gpu.median_us * (1.0 + GPU_CONTRARY_TOLERANCE);
    if material && stable && no_contrary_gpu {
        Ok((winner.route, DenseBf16SelectionStatus::CalibratedWinner))
    } else {
        Ok((compatibility, DenseBf16SelectionStatus::NoStableWinner))
    }
}

fn poison_output(output: &mut MlxBuffer) -> Result<()> {
    output
        .as_mut_slice::<f32>()?
        .fill(f32::from_bits(OUTPUT_POISON_BITS));
    Ok(())
}

fn verified_output_bits(output: &MlxBuffer, logical_elements: usize) -> Result<Vec<u32>> {
    let values = output.as_slice::<f32>()?;
    let expected = logical_elements
        .checked_add(OUTPUT_GUARD_ELEMENTS)
        .ok_or_else(|| MlxError::InvalidArgument("dense BF16 guard size overflow".into()))?;
    if values.len() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 proof buffer has {} elements, expected {expected}",
            values.len()
        )));
    }
    let mut bits = Vec::with_capacity(logical_elements);
    for (index, value) in values[..logical_elements].iter().enumerate() {
        if value.to_bits() == OUTPUT_POISON_BITS {
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 route left logical output element {index} unwritten"
            )));
        }
        if !value.is_finite() {
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 route wrote non-finite logical output element {index}"
            )));
        }
        bits.push(value.to_bits());
    }
    if let Some((guard_index, _)) = values[logical_elements..]
        .iter()
        .enumerate()
        .find(|(_, value)| value.to_bits() != OUTPUT_POISON_BITS)
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 route overwrote output guard element {guard_index}"
        )));
    }
    Ok(bits)
}

fn checked_elements(dimensions: &[u32]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension as usize).ok_or_else(|| {
            MlxError::InvalidArgument("dense BF16 calibration size overflows usize".into())
        })
    })
}

fn budget_fallback(
    shape: DenseBf16Shape,
    compatibility_route: DenseBf16Route,
    unavailable_routes: Vec<DenseBf16Route>,
    incoherent_routes: Vec<DenseBf16Route>,
    submissions: u32,
) -> DenseBf16CalibrationDecision {
    DenseBf16CalibrationDecision {
        shape,
        compatibility_route,
        selected_route: compatibility_route,
        status: DenseBf16SelectionStatus::BudgetFallback,
        unavailable_routes,
        incoherent_routes,
        timings: Vec::new(),
        process_cache_hit: false,
        calibration_submissions: submissions,
    }
}

fn deadline_reached(started: Instant, max_elapsed_ms: u64) -> bool {
    started.elapsed().as_secs_f64() * 1000.0 >= max_elapsed_ms as f64
}

fn calibrate_one_decision(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    shape: DenseBf16Shape,
    prepared: &PreparedDenseBf16Routes,
    started: Instant,
    limits: DenseBf16CalibrationLimits,
) -> Result<DenseBf16CalibrationDecision> {
    let compatibility = compatibility_route(shape, prepared)?;
    let unavailable_routes = if shape.k % 4 != 0 && shape.k >= 32 {
        vec![DenseBf16Route::TensorV1]
    } else {
        Vec::new()
    };
    if deadline_reached(started, limits.max_elapsed_ms) {
        return Ok(budget_fallback(
            shape,
            compatibility,
            unavailable_routes,
            Vec::new(),
            0,
        ));
    }
    let input_elements = checked_elements(&[shape.src1_batch, shape.m, shape.k])?;
    let output_elements = checked_elements(&[shape.src1_batch, shape.m, shape.n])?;
    let mut input = device.alloc_buffer(
        input_elements
            .checked_mul(DType::F32.size_of())
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense BF16 calibration input bytes overflow".into())
            })?,
        DType::F32,
        vec![
            shape.src1_batch as usize,
            shape.m as usize,
            shape.k as usize,
        ],
    )?;
    for (index, value) in input.as_mut_slice::<f32>()?.iter_mut().enumerate() {
        *value = ((index.wrapping_mul(29) % 251) as f32 - 125.0) / 1003.0;
    }
    let guarded_output_elements = output_elements
        .checked_add(OUTPUT_GUARD_ELEMENTS)
        .ok_or_else(|| MlxError::InvalidArgument("dense BF16 output guard overflow".into()))?;
    let mut output = device.alloc_buffer(
        guarded_output_elements
            .checked_mul(DType::F32.size_of())
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense BF16 calibration output bytes overflow".into())
            })?,
        DType::F32,
        vec![guarded_output_elements],
    )?;

    let mut routes = legal_routes(shape);
    routes.retain(|route| equivalence_class(*route) == equivalence_class(compatibility));
    routes.sort_by_key(|route| if *route == compatibility { 0 } else { 1 });
    let mut proofs = HashMap::with_capacity(routes.len());
    let mut coherent_routes = Vec::with_capacity(routes.len());
    let mut incoherent_routes = Vec::new();
    let mut reference_bits = None;
    let mut submissions = 0u32;
    for route in routes {
        if deadline_reached(started, limits.max_elapsed_ms) {
            return Ok(budget_fallback(
                shape,
                compatibility,
                unavailable_routes,
                incoherent_routes,
                submissions,
            ));
        }
        poison_output(&mut output)?;
        let proof = trace_route_and_execute(
            route, registry, device, weight, &input, &output, shape, prepared,
        )?;
        submissions += 1;
        let bits = verified_output_bits(&output, output_elements)?;
        if deadline_reached(started, limits.max_elapsed_ms) {
            return Ok(budget_fallback(
                shape,
                compatibility,
                unavailable_routes,
                incoherent_routes,
                submissions,
            ));
        }
        if route == compatibility {
            reference_bits = Some(bits);
            coherent_routes.push(route);
            proofs.insert(route, proof);
        } else if reference_bits.as_ref() == Some(&bits) {
            coherent_routes.push(route);
            proofs.insert(route, proof);
        } else {
            incoherent_routes.push(route);
        }
    }
    if !coherent_routes.contains(&compatibility) {
        return Err(MlxError::InvalidArgument(
            "dense BF16 calibration did not execute its compatibility route".into(),
        ));
    }

    let mut wall_samples: HashMap<DenseBf16Route, Vec<f64>> = coherent_routes
        .iter()
        .copied()
        .map(|route| (route, Vec::with_capacity(CALIBRATION_SAMPLES)))
        .collect();
    let mut gpu_samples = wall_samples.clone();
    for round in 0..CALIBRATION_SAMPLES {
        let ordered_routes: Vec<_> = if round % 2 == 0 {
            coherent_routes.clone()
        } else {
            coherent_routes.iter().rev().copied().collect()
        };
        for route in ordered_routes {
            if deadline_reached(started, limits.max_elapsed_ms) {
                return Ok(budget_fallback(
                    shape,
                    compatibility,
                    unavailable_routes,
                    incoherent_routes,
                    submissions,
                ));
            }
            let sample = time_route(route, registry, device, weight, &input, &output, shape)?;
            submissions += 1;
            if deadline_reached(started, limits.max_elapsed_ms) {
                return Ok(budget_fallback(
                    shape,
                    compatibility,
                    unavailable_routes,
                    incoherent_routes,
                    submissions,
                ));
            }
            wall_samples
                .get_mut(&route)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("dense BF16 wall sample route disappeared".into())
                })?
                .push(sample.wall_us);
            gpu_samples
                .get_mut(&route)
                .ok_or_else(|| {
                    MlxError::InvalidArgument("dense BF16 GPU sample route disappeared".into())
                })?
                .push(sample.gpu_us);
        }
    }

    let mut timings = Vec::with_capacity(coherent_routes.len());
    for route in coherent_routes {
        let (encoded, pipeline) = proofs.remove(&route).ok_or_else(|| {
            MlxError::InvalidArgument(format!("dense BF16 route {route:?} has no trace proof"))
        })?;
        let wall = wall_samples.remove(&route).ok_or_else(|| {
            MlxError::InvalidArgument("dense BF16 wall timing route disappeared".into())
        })?;
        let gpu = gpu_samples.remove(&route).ok_or_else(|| {
            MlxError::InvalidArgument("dense BF16 GPU timing route disappeared".into())
        })?;
        timings.push(DenseBf16RouteTiming {
            route,
            wall: distribution(wall)?,
            gpu: distribution(gpu)?,
            encoded,
            pipeline,
        });
    }
    if deadline_reached(started, limits.max_elapsed_ms) {
        return Ok(budget_fallback(
            shape,
            compatibility,
            unavailable_routes,
            incoherent_routes,
            submissions,
        ));
    }
    let (selected_route, status) = select_route(compatibility, &timings)?;
    Ok(DenseBf16CalibrationDecision {
        shape,
        compatibility_route: compatibility,
        selected_route,
        status,
        unavailable_routes,
        incoherent_routes,
        timings,
        process_cache_hit: false,
        calibration_submissions: submissions,
    })
}

struct ValidatedCalibrationCase<'a> {
    weight: &'a MlxBuffer,
    shape: DenseBf16BaseShape,
    reachable_m: BTreeSet<u32>,
}

fn validate_cases<'a>(
    cases: &'a [DenseBf16CalibrationCase<'a>],
    limits: DenseBf16CalibrationLimits,
) -> Result<Vec<ValidatedCalibrationCase<'a>>> {
    if limits.max_elapsed_ms == 0 || limits.max_shapes == 0 {
        return Err(MlxError::InvalidArgument(
            "dense BF16 calibration limits must be nonzero".into(),
        ));
    }
    if cases.is_empty() {
        return Err(MlxError::InvalidArgument(
            "dense BF16 calibration requires at least one borrowed weight case".into(),
        ));
    }
    let mut validated: Vec<ValidatedCalibrationCase<'a>> = Vec::new();
    for case in cases {
        let shape = case.shape;
        if shape.n == 0
            || shape.k == 0
            || shape.src0_batch == 0
            || shape.src1_batch == 0
            || shape.src1_batch % shape.src0_batch != 0
            || shape.src1_batch / shape.src0_batch > i16::MAX as u32
        {
            return Err(MlxError::InvalidArgument(format!(
                "invalid dense BF16 calibration base shape {shape:?}"
            )));
        }
        for (name, dimension) in [
            ("N", shape.n),
            ("K", shape.k),
            ("src0_batch", shape.src0_batch),
            ("src1_batch", shape.src1_batch),
        ] {
            if dimension > i32::MAX as u32 {
                return Err(MlxError::InvalidArgument(format!(
                    "dense BF16 calibration {name} ({dimension}) exceeds i32 shader indexing"
                )));
            }
        }
        if shape.k < 32 && shape.k % 4 != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 calibration K ({}) has no executable native route",
                shape.k
            )));
        }
        if case.weight.dtype() != DType::BF16 || case.weight.byte_offset() % 8 != 0 {
            return Err(MlxError::InvalidArgument(
                "dense BF16 calibration weight must be aligned native BF16 storage".into(),
            ));
        }
        let weight_elements = checked_elements(&[shape.src0_batch, shape.n, shape.k])?;
        let weight_bytes = weight_elements
            .checked_mul(DType::BF16.size_of())
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense BF16 calibration weight bytes overflow".into())
            })?;
        if case.weight.data_byte_len() < weight_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 calibration weight needs {weight_bytes} bytes, got {}",
                case.weight.data_byte_len()
            )));
        }
        if case.reachable_m.is_empty()
            || case
                .reachable_m
                .iter()
                .any(|&m| m == 0 || m > DENSE_BF16_CALIBRATION_MAX_M)
        {
            return Err(MlxError::InvalidArgument(format!(
                "dense BF16 reachable rows must be in 1..={DENSE_BF16_CALIBRATION_MAX_M}"
            )));
        }
        if let Some(existing) = validated.iter_mut().find(|entry| entry.shape == shape) {
            existing
                .reachable_m
                .extend(case.reachable_m.iter().copied());
        } else {
            validated.push(ValidatedCalibrationCase {
                weight: case.weight,
                shape,
                reachable_m: case.reachable_m.iter().copied().collect(),
            });
        }
    }
    let declared_shapes: usize = validated.iter().map(|case| case.reachable_m.len()).sum();
    if declared_shapes == 0 || declared_shapes > limits.max_shapes as usize {
        return Err(MlxError::InvalidArgument(format!(
            "dense BF16 calibration declares {declared_shapes} shapes, limit is {}",
            limits.max_shapes
        )));
    }
    validated.sort_by_key(|case| case.shape);
    Ok(validated)
}

fn plan_id(
    build_fingerprint: &str,
    device_registry_id: u64,
    activation_epoch: u64,
    decisions: &HashMap<DenseBf16Shape, DenseBf16Route>,
) -> Result<String> {
    let mut ordered: Vec<_> = decisions
        .iter()
        .map(|(shape, route)| (*shape, *route))
        .collect();
    ordered.sort_by_key(|(shape, _)| *shape);
    let encoded = serde_json::to_vec(&ordered).map_err(|error| {
        MlxError::InvalidArgument(format!("serialize dense BF16 route plan: {error}"))
    })?;
    let mut digest = Sha256::new();
    digest.update(build_fingerprint.as_bytes());
    digest.update(device_registry_id.to_le_bytes());
    digest.update(activation_epoch.to_le_bytes());
    digest.update(encoded);
    Ok(hex::encode(digest.finalize()))
}

/// Calibrate declared reachable BF16 shapes and freeze the resulting plan.
///
/// `cases` borrow model weights only for this call. No buffer, closure, model
/// identity, or artifact identity enters the process cache or returned plan.
/// A process-cache hit performs zero timing submissions.
pub fn calibrate_dense_bf16_routes(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    limits: DenseBf16CalibrationLimits,
    cases: &[DenseBf16CalibrationCase<'_>],
) -> Result<(Arc<DenseBf16RoutePlan>, DenseBf16CalibrationBatchReceipt)> {
    if activation_epoch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense BF16 activation epoch must be nonzero".into(),
        ));
    }
    if registry.dense_bf16_plan().is_some() {
        return Err(MlxError::InvalidArgument(
            "dense BF16 registry is already frozen; calibration is one-shot".into(),
        ));
    }
    let validated = validate_cases(cases, limits)?;
    let started = Instant::now();
    let prepared = prepare_routes(registry, device)?;
    let default_dense_mm_route = if prepared.tensor_available {
        DenseBf16Route::TensorV1
    } else {
        DenseBf16Route::Simdgroup
    };
    let declared_shapes: usize = validated.iter().map(|case| case.reachable_m.len()).sum();
    let mut decisions = Vec::with_capacity(declared_shapes);
    let mut plan_decisions = HashMap::with_capacity(declared_shapes);
    let mut calibrated_decisions = 0u32;
    let mut process_cache_hits = 0u32;
    let mut budget_fallback_decisions = 0u32;
    let mut calibration_submissions = 0u32;

    for case in validated {
        for m in case.reachable_m {
            let shape = case.shape.with_m(m);
            let key = DenseBf16ProcessKey {
                build_fingerprint: prepared.build_fingerprint.clone(),
                device_name: device.name(),
                device_registry_id: device.registry_id(),
                pipeline_set_fingerprint: prepared.pipeline_set_fingerprint.clone(),
                shape,
            };
            let cell = process_cell(key.clone())?;
            let initialized_here = Cell::new(false);
            let cached = cell.get_or_init(|| {
                initialized_here.set(true);
                calibrate_one_decision(
                    registry,
                    device,
                    case.weight,
                    shape,
                    &prepared,
                    started,
                    limits,
                )
                .map_err(|error| DenseBf16CalibrationFailure {
                    message: error.to_string(),
                })
            });
            let cached_result = cached.clone();
            if matches!(
                &cached_result,
                Err(_)
                    | Ok(DenseBf16CalibrationDecision {
                        status: DenseBf16SelectionStatus::BudgetFallback,
                        ..
                    })
            ) {
                // A caller budget or transient GPU failure is not an intrinsic
                // device/build/shape result. Let later model activations retry
                // instead of making the first load order permanent.
                evict_process_cell_if_same(&key, &cell)?;
            }
            let mut decision = cached_result.map_err(|failure| {
                MlxError::InvalidArgument(format!(
                    "dense BF16 calibration failed: {}",
                    failure.message
                ))
            })?;
            if initialized_here.get() {
                calibration_submissions = calibration_submissions
                    .checked_add(decision.calibration_submissions)
                    .ok_or_else(|| {
                        MlxError::InvalidArgument(
                            "dense BF16 calibration submission count overflow".into(),
                        )
                    })?;
                if decision.status != DenseBf16SelectionStatus::BudgetFallback {
                    calibrated_decisions += 1;
                }
            } else {
                process_cache_hits += 1;
                decision.process_cache_hit = true;
                decision.calibration_submissions = 0;
            }
            if decision.status == DenseBf16SelectionStatus::BudgetFallback {
                budget_fallback_decisions += 1;
            }
            plan_decisions.insert(shape, decision.selected_route);
            decisions.push(decision);
        }
    }
    decisions.sort_by_key(|decision| decision.shape);
    if calibration_submissions > 0 {
        // The final decision's scratch buffers have just dropped. Commit one
        // empty boundary so residency-set removals are not carried into the
        // first request or a model swap.
        let mut cleanup = device.command_encoder()?;
        cleanup.commit_and_wait()?;
        calibration_submissions = calibration_submissions.checked_add(1).ok_or_else(|| {
            MlxError::InvalidArgument("dense BF16 calibration submission count overflow".into())
        })?;
    }
    let plan_id = plan_id(
        &prepared.build_fingerprint,
        device.registry_id(),
        activation_epoch,
        &plan_decisions,
    )?;
    let plan = Arc::new(DenseBf16RoutePlan {
        plan_id: plan_id.clone(),
        build_fingerprint: prepared.build_fingerprint.clone(),
        device_name: device.name(),
        device_registry_id: device.registry_id(),
        activation_epoch,
        default_dense_mm_route,
        decisions: plan_decisions,
    });
    registry.freeze_dense_bf16_plan(device, plan.clone())?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((
        plan,
        DenseBf16CalibrationBatchReceipt {
            schema_version: DENSE_BF16_ROUTE_SCHEMA_VERSION,
            mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
            build_fingerprint: prepared.build_fingerprint,
            plan_id,
            activation_epoch,
            device_name: device.name(),
            device_registry_id: device.registry_id(),
            declared_shapes: declared_shapes as u32,
            calibrated_decisions,
            process_cache_hits,
            budget_fallback_decisions,
            calibration_submissions,
            elapsed_ms,
            deadline_overrun_ms: (elapsed_ms - limits.max_elapsed_ms as f64).max(0.0),
            decisions,
        },
    ))
}

#[cfg(test)]
#[path = "dense_bf16_auto_tests.rs"]
mod tests;
