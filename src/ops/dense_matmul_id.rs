//! Native F32/F16/BF16 expert-ID matrix multiplication.
//!
//! The weight buffer is an explicitly-strided stack of row-major matrices.
//! No conversion, shadow allocation, or dequantization is performed. Decode
//! and repeat-allowed routing use one direct dispatch. Larger distinct-routing
//! prefills use one map dispatch and one grouped matrix dispatch with
//! caller-owned scratch.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use serde::{Deserialize, Serialize};

/// Schema for native scalar expert-ID capability, calibration, and trace
/// receipts. This schema is intentionally independent of the block-quantized
/// GGML capability schema.
pub const DENSE_MATMUL_ID_SCHEMA_VERSION: u32 = 3;

/// Complete shader-name surface owned by this primitive. Registration and
/// post-freeze mutation protection consume this same list.
pub(crate) const DENSE_MATMUL_ID_PIPELINE_NAMES: [&str; 7] = [
    "dense_matmul_id_direct_f32_f32",
    "dense_matmul_id_direct_f16_f32",
    "dense_matmul_id_direct_bf16_f32",
    "dense_matmul_id_map_distinct",
    "dense_matmul_id_grouped_f32_f32",
    "dense_matmul_id_grouped_f16_f32",
    "dense_matmul_id_grouped_bf16_f32",
];

const GROUPED_TILE_M: u64 = 8;
const GROUPED_TILE_N: u64 = 8;
const DIRECT_OUTPUTS_PER_THREADGROUP: u64 = 8;
const DIRECT_THREADS: u64 = 64;
const GROUPED_THREADS: u64 = 256;
const GROUPED_MAX_EXPERTS: u32 = 1024;

/// F32 activation layout for one expert-ID product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdInputLayout {
    /// One `[K]` activation row is shared by every selected expert slot.
    SharedPerToken,
    /// Every `(token, slot)` pair owns its own `[K]` activation row.
    Slotted,
}

/// Multiplicity contract for expert IDs within one token's top-k row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdMultiplicity {
    /// The caller guarantees every expert ID in a token row is distinct.
    DistinctPerToken,
    /// A token row may select the same expert in more than one slot.
    MayRepeat,
}

/// Host dimensions and explicit strides for native scalar expert-ID matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub top_k: u32,
    pub n_experts: u32,
    /// Byte stride from expert `e` to expert `e + 1` in the weight stack.
    pub expert_stride_bytes: u64,
    pub input_layout: DenseMatmulIdInputLayout,
    pub id_multiplicity: DenseMatmulIdMultiplicity,
    /// Explicit execution route. Route selection/calibration belongs to the
    /// caller; capability validation never guesses performance from M alone.
    pub route: DenseMatmulIdRoute,
}

/// Selected execution route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseMatmulIdRoute {
    /// One expert-indexed matrix-vector dispatch over every routed row.
    Direct,
    /// One distinct-route map dispatch plus one grouped matrix dispatch.
    GroupedPrefill,
}

/// Required caller-owned scratch for grouped prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdScratchRequirement {
    pub expert_counts_bytes: usize,
    pub routed_rows_bytes: usize,
}

/// Fail-closed capability and exact byte-extent result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdCapability {
    pub schema_version: u32,
    pub weight_dtype: DType,
    pub route: DenseMatmulIdRoute,
    pub required_weight_bytes: usize,
    pub required_input_bytes: usize,
    pub required_ids_bytes: usize,
    pub required_output_bytes: usize,
    pub scratch: Option<DenseMatmulIdScratchRequirement>,
}

/// Dispatch receipt for profiling and route assertions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseMatmulIdDispatchReceipt {
    pub weight_dtype: DType,
    pub route: DenseMatmulIdRoute,
    pub dispatch_count: u32,
}

/// Caller-owned grouped-prefill scratch.
///
/// The buffers must outlive every command buffer that references them. Reuse
/// across command encoders is legal only after the prior submission has
/// completed. Same-encoder reuse is ordered by the grouped primitive's
/// mandatory pre-map memory barrier.
pub struct DenseMatmulIdScratch {
    expert_counts: MlxBuffer,
    routed_rows: MlxBuffer,
    max_experts: u32,
    max_tokens: u32,
}

impl DenseMatmulIdScratch {
    pub fn new(device: &MlxDevice, max_experts: u32, max_tokens: u32) -> Result<Self> {
        if max_experts == 0 || max_tokens == 0 {
            return Err(MlxError::InvalidArgument(
                "dense_matmul_id scratch capacities must be nonzero".into(),
            ));
        }
        let counts_bytes = checked_usize_product(
            "scratch expert counts",
            &[u64::from(max_experts), DType::U32.size_of() as u64],
        )?;
        let routed_bytes = checked_usize_product(
            "scratch routed rows",
            &[
                u64::from(max_experts),
                u64::from(max_tokens),
                DType::U32.size_of() as u64,
            ],
        )?;
        Ok(Self {
            expert_counts: device.alloc_buffer(
                counts_bytes,
                DType::U32,
                vec![max_experts as usize],
            )?,
            routed_rows: device.alloc_buffer(
                routed_bytes,
                DType::U32,
                vec![max_experts as usize, max_tokens as usize],
            )?,
            max_experts,
            max_tokens,
        })
    }

    pub fn max_experts(&self) -> u32 {
        self.max_experts
    }

    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    pub(crate) fn expert_counts(&self) -> &MlxBuffer {
        &self.expert_counts
    }

    pub(crate) fn routed_rows(&self) -> &MlxBuffer {
        &self.routed_rows
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMatmulIdGpuParams {
    m: u32,
    n: u32,
    k: u32,
    top_k: u32,
    n_experts: u32,
    input_layout: u32,
    expert_stride_bytes: u64,
    input_token_stride_bytes: u64,
    input_slot_stride_bytes: u64,
}

fn checked_product(label: &str, factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1u64, |value, factor| {
        value.checked_mul(*factor).ok_or_else(|| {
            MlxError::InvalidArgument(format!("dense_matmul_id {label} byte extent overflow"))
        })
    })
}

fn checked_usize_product(label: &str, factors: &[u64]) -> Result<usize> {
    usize::try_from(checked_product(label, factors)?).map_err(|_| {
        MlxError::InvalidArgument(format!("dense_matmul_id {label} byte extent exceeds usize"))
    })
}

fn scalar_dtype_size(dtype: DType) -> Result<u64> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(dtype.size_of() as u64),
        other => Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id weights must be F32, F16, or BF16, got {other}"
        ))),
    }
}

/// Resolve the exact native route and byte contract without allocating or
/// compiling a pipeline.
pub fn dense_matmul_id_capability(
    weight_dtype: DType,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdCapability> {
    if params.m == 0 || params.n == 0 || params.k == 0 || params.top_k == 0 || params.n_experts == 0
    {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id M, N, K, top_k, and n_experts must be nonzero".into(),
        ));
    }
    if params.top_k > params.n_experts
        && params.id_multiplicity == DenseMatmulIdMultiplicity::DistinctPerToken
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id distinct top_k {} exceeds n_experts {}",
            params.top_k, params.n_experts
        )));
    }

    let scalar_bytes = scalar_dtype_size(weight_dtype)?;
    let matrix_bytes = checked_product(
        "expert matrix",
        &[u64::from(params.n), u64::from(params.k), scalar_bytes],
    )?;
    if params.expert_stride_bytes < matrix_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id expert stride {} is smaller than one {matrix_bytes}-byte matrix",
            params.expert_stride_bytes
        )));
    }
    if params.expert_stride_bytes % scalar_bytes != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id expert stride {} is not aligned to {weight_dtype}",
            params.expert_stride_bytes
        )));
    }
    let last_expert_offset = u64::from(params.n_experts - 1)
        .checked_mul(params.expert_stride_bytes)
        .ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id expert weight offset overflow".into())
        })?;
    let required_weight_bytes = usize::try_from(
        last_expert_offset
            .checked_add(matrix_bytes)
            .ok_or_else(|| {
                MlxError::InvalidArgument("dense_matmul_id weight extent overflow".into())
            })?,
    )
    .map_err(|_| MlxError::InvalidArgument("dense_matmul_id weight extent exceeds usize".into()))?;

    let total_rows = u64::from(params.m)
        .checked_mul(u64::from(params.top_k))
        .ok_or_else(|| MlxError::InvalidArgument("dense_matmul_id row count overflow".into()))?;
    if total_rows > u64::from(u32::MAX) {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id flattened routed rows exceed the Metal u32 ABI".into(),
        ));
    }
    let input_rows = match params.input_layout {
        DenseMatmulIdInputLayout::SharedPerToken => u64::from(params.m),
        DenseMatmulIdInputLayout::Slotted => total_rows,
    };
    let required_input_bytes = checked_usize_product(
        "input",
        &[input_rows, u64::from(params.k), DType::F32.size_of() as u64],
    )?;
    let required_ids_bytes =
        checked_usize_product("expert IDs", &[total_rows, DType::U32.size_of() as u64])?;
    let required_output_bytes = checked_usize_product(
        "output",
        &[total_rows, u64::from(params.n), DType::F32.size_of() as u64],
    )?;

    if params.route == DenseMatmulIdRoute::GroupedPrefill
        && params.id_multiplicity != DenseMatmulIdMultiplicity::DistinctPerToken
    {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_id grouped prefill requires distinct expert IDs per token".into(),
        ));
    }
    if params.route == DenseMatmulIdRoute::GroupedPrefill && params.n_experts > GROUPED_MAX_EXPERTS
    {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id grouped prefill supports at most {GROUPED_MAX_EXPERTS} experts"
        )));
    }
    let route = params.route;
    let scratch = (route == DenseMatmulIdRoute::GroupedPrefill)
        .then(|| {
            Ok(DenseMatmulIdScratchRequirement {
                expert_counts_bytes: checked_usize_product(
                    "scratch expert counts",
                    &[u64::from(params.n_experts), DType::U32.size_of() as u64],
                )?,
                routed_rows_bytes: checked_usize_product(
                    "scratch routed rows",
                    &[
                        u64::from(params.n_experts),
                        u64::from(params.m),
                        DType::U32.size_of() as u64,
                    ],
                )?,
            })
        })
        .transpose()?;

    Ok(DenseMatmulIdCapability {
        schema_version: DENSE_MATMUL_ID_SCHEMA_VERSION,
        weight_dtype,
        route,
        required_weight_bytes,
        required_input_bytes,
        required_ids_bytes,
        required_output_bytes,
        scratch,
    })
}

fn validate_buffer(
    label: &str,
    buffer: &MlxBuffer,
    dtype: DType,
    required_bytes: usize,
) -> Result<()> {
    if buffer.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id {label} must be {dtype}, got {}",
            buffer.dtype()
        )));
    }
    if buffer.data_byte_len() < required_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id {label} requires {required_bytes} logical bytes, got {}",
            buffer.data_byte_len()
        )));
    }
    if buffer.byte_offset() % dtype.size_of() as u64 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id {label} byte offset {} is not aligned to {dtype}",
            buffer.byte_offset()
        )));
    }
    Ok(())
}

pub(super) fn pipeline_names(dtype: DType) -> Result<(&'static str, &'static str)> {
    match dtype {
        DType::BF16 => Ok((
            "dense_matmul_id_direct_bf16_f32",
            "dense_matmul_id_grouped_bf16_f32",
        )),
        DType::F16 => Ok((
            "dense_matmul_id_direct_f16_f32",
            "dense_matmul_id_grouped_f16_f32",
        )),
        DType::F32 => Ok((
            "dense_matmul_id_direct_f32_f32",
            "dense_matmul_id_grouped_f32_f32",
        )),
        other => Err(MlxError::InvalidArgument(format!(
            "dense_matmul_id has no scalar pipeline for {other}"
        ))),
    }
}

fn logical_range(buffer: &MlxBuffer) -> (usize, usize) {
    let start = (buffer.contents_ptr() as usize).saturating_add(buffer.byte_offset() as usize);
    (start, start.saturating_add(buffer.data_byte_len()))
}

fn required_range(buffer: &MlxBuffer, required_bytes: usize) -> Result<(usize, usize)> {
    let start = (buffer.contents_ptr() as usize)
        .checked_add(buffer.byte_offset() as usize)
        .ok_or_else(|| {
            MlxError::InvalidArgument("dense_matmul_id logical buffer address overflow".into())
        })?;
    let end = start.checked_add(required_bytes).ok_or_else(|| {
        MlxError::InvalidArgument("dense_matmul_id logical buffer range overflow".into())
    })?;
    Ok((start, end))
}

fn ranges_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_dense_matmul_id_call(
    weights: &MlxBuffer,
    input: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: Option<&DenseMatmulIdScratch>,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdCapability> {
    let capability = dense_matmul_id_capability(weights.dtype(), params)?;
    validate_buffer(
        "weights",
        weights,
        capability.weight_dtype,
        capability.required_weight_bytes,
    )?;
    validate_buffer("input", input, DType::F32, capability.required_input_bytes)?;
    validate_buffer(
        "expert IDs",
        expert_ids,
        DType::U32,
        capability.required_ids_bytes,
    )?;
    validate_buffer(
        "output",
        output,
        DType::F32,
        capability.required_output_bytes,
    )?;
    let output_range = required_range(output, capability.required_output_bytes)?;
    for (label, read_range) in [
        (
            "weights",
            required_range(weights, capability.required_weight_bytes)?,
        ),
        (
            "input",
            required_range(input, capability.required_input_bytes)?,
        ),
        (
            "expert IDs",
            required_range(expert_ids, capability.required_ids_bytes)?,
        ),
    ] {
        if ranges_overlap(output_range, read_range) {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id output must not overlap {label}"
            )));
        }
    }
    if capability.route == DenseMatmulIdRoute::GroupedPrefill {
        let scratch = scratch.ok_or_else(|| {
            MlxError::InvalidArgument(
                "dense_matmul_id grouped prefill requires caller-owned scratch".into(),
            )
        })?;
        if scratch.max_experts < params.n_experts || scratch.max_tokens < params.m {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matmul_id scratch capacity experts/tokens={}/{} is smaller than {}/{}",
                scratch.max_experts, scratch.max_tokens, params.n_experts, params.m
            )));
        }
        let requirement = capability.scratch.ok_or_else(|| {
            MlxError::InvalidArgument(
                "dense_matmul_id grouped route omitted scratch requirement".into(),
            )
        })?;
        validate_buffer(
            "scratch expert counts",
            &scratch.expert_counts,
            DType::U32,
            requirement.expert_counts_bytes,
        )?;
        validate_buffer(
            "scratch routed rows",
            &scratch.routed_rows,
            DType::U32,
            requirement.routed_rows_bytes,
        )?;
    }
    Ok(capability)
}

/// Encode native scalar expert-ID matrix multiplication into the caller's
/// command encoder. Expert IDs outside `n_experts` produce a fully overwritten
/// zero output row without forming a weight address. `DistinctPerToken` is an
/// explicit caller promise required by the grouped route. Use `MayRepeat` and
/// `Direct` when that promise is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn dense_matmul_id(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weights: &MlxBuffer,
    input: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: Option<&DenseMatmulIdScratch>,
    params: &DenseMatmulIdParams,
) -> Result<DenseMatmulIdDispatchReceipt> {
    let capability =
        validate_dense_matmul_id_call(weights, input, expert_ids, output, scratch, params)?;

    let input_row_bytes = checked_product(
        "input row",
        &[u64::from(params.k), DType::F32.size_of() as u64],
    )?;
    let (input_token_stride_bytes, input_slot_stride_bytes) = match params.input_layout {
        DenseMatmulIdInputLayout::SharedPerToken => (input_row_bytes, 0),
        DenseMatmulIdInputLayout::Slotted => (
            input_row_bytes
                .checked_mul(u64::from(params.top_k))
                .ok_or_else(|| {
                    MlxError::InvalidArgument(
                        "dense_matmul_id slotted input token stride overflow".into(),
                    )
                })?,
            input_row_bytes,
        ),
    };
    let gpu_params = DenseMatmulIdGpuParams {
        m: params.m,
        n: params.n,
        k: params.k,
        top_k: params.top_k,
        n_experts: params.n_experts,
        input_layout: match params.input_layout {
            DenseMatmulIdInputLayout::SharedPerToken => 0,
            DenseMatmulIdInputLayout::Slotted => 1,
        },
        expert_stride_bytes: params.expert_stride_bytes,
        input_token_stride_bytes,
        input_slot_stride_bytes,
    };
    let (direct_name, grouped_name) = pipeline_names(weights.dtype())?;

    match capability.route {
        DenseMatmulIdRoute::Direct => {
            let pipeline = registry.get_pipeline(direct_name, device.metal_device())?;
            if encoder.is_capturing() {
                encoder.set_pending_buffer_ranges(
                    vec![
                        logical_range(weights),
                        logical_range(input),
                        logical_range(expert_ids),
                    ],
                    vec![logical_range(output)],
                );
            }
            encoder.encode_threadgroups_with_args(
                pipeline,
                &[
                    (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                    (1, KernelArg::Buffer(weights)),
                    (2, KernelArg::Buffer(input)),
                    (3, KernelArg::Buffer(expert_ids)),
                    (4, KernelArg::Buffer(output)),
                ],
                metal::MTLSize::new(
                    (u64::from(params.n) + DIRECT_OUTPUTS_PER_THREADGROUP - 1)
                        / DIRECT_OUTPUTS_PER_THREADGROUP,
                    u64::from(params.m) * u64::from(params.top_k),
                    1,
                ),
                metal::MTLSize::new(DIRECT_THREADS, 1, 1),
            );
            Ok(DenseMatmulIdDispatchReceipt {
                weight_dtype: weights.dtype(),
                route: capability.route,
                dispatch_count: 1,
            })
        }
        DenseMatmulIdRoute::GroupedPrefill => {
            let scratch = scratch.ok_or_else(|| {
                MlxError::InvalidArgument(
                    "dense_matmul_id grouped prefill requires caller-owned scratch".into(),
                )
            })?;

            // Resolve and validate every pipeline before touching the caller's
            // encoder. A missing grouped pipeline must not leave a map
            // dispatch, barrier, or pending capture ranges behind.
            let map_pipeline = registry
                .get_pipeline("dense_matmul_id_map_distinct", device.metal_device())?
                .clone();
            let grouped_pipeline = registry
                .get_pipeline(grouped_name, device.metal_device())?
                .clone();
            if u64::from(params.n_experts) > map_pipeline.max_total_threads_per_threadgroup() {
                return Err(MlxError::InvalidArgument(format!(
                    "dense_matmul_id n_experts {} exceeds map threadgroup limit {}",
                    params.n_experts,
                    map_pipeline.max_total_threads_per_threadgroup()
                )));
            }
            if GROUPED_THREADS > grouped_pipeline.max_total_threads_per_threadgroup() {
                return Err(MlxError::InvalidArgument(format!(
                    "dense_matmul_id grouped threads {GROUPED_THREADS} exceed pipeline limit {}",
                    grouped_pipeline.max_total_threads_per_threadgroup()
                )));
            }

            // A prior grouped call may still be reading the caller-owned
            // counts/row map while this call starts overwriting them. Own the
            // inter-call scratch edge here so bare CommandEncoder callers get
            // the same ordering guarantee as graph/session callers. The
            // following map->grouped barrier remains a separate RAW edge.
            encoder.memory_barrier();
            if encoder.is_capturing() {
                encoder.set_pending_buffer_ranges(
                    vec![logical_range(expert_ids)],
                    vec![
                        logical_range(&scratch.expert_counts),
                        logical_range(&scratch.routed_rows),
                        logical_range(output),
                    ],
                );
            }
            encoder.encode_threadgroups_with_args(
                &map_pipeline,
                &[
                    (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                    (1, KernelArg::Buffer(expert_ids)),
                    (2, KernelArg::Buffer(&scratch.expert_counts)),
                    (3, KernelArg::Buffer(&scratch.routed_rows)),
                    (4, KernelArg::Buffer(output)),
                ],
                metal::MTLSize::new(1, 1, 1),
                metal::MTLSize::new(u64::from(params.n_experts), 1, 1),
            );
            encoder.memory_barrier();

            if encoder.is_capturing() {
                encoder.set_pending_buffer_ranges(
                    vec![
                        logical_range(weights),
                        logical_range(input),
                        logical_range(&scratch.expert_counts),
                        logical_range(&scratch.routed_rows),
                    ],
                    vec![logical_range(output)],
                );
            }
            let shmem_bytes = 8 * 128 * DType::F32.size_of() as u64;
            encoder.encode_threadgroups_with_args_and_shared(
                &grouped_pipeline,
                &[
                    (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                    (1, KernelArg::Buffer(weights)),
                    (2, KernelArg::Buffer(input)),
                    (3, KernelArg::Buffer(&scratch.expert_counts)),
                    (4, KernelArg::Buffer(&scratch.routed_rows)),
                    (5, KernelArg::Buffer(output)),
                ],
                &[(0, shmem_bytes)],
                metal::MTLSize::new(
                    (u64::from(params.m) + GROUPED_TILE_M - 1) / GROUPED_TILE_M,
                    (u64::from(params.n) + GROUPED_TILE_N - 1) / GROUPED_TILE_N,
                    u64::from(params.n_experts),
                ),
                metal::MTLSize::new(GROUPED_THREADS, 1, 1),
            );
            Ok(DenseMatmulIdDispatchReceipt {
                weight_dtype: weights.dtype(),
                route: capability.route,
                dispatch_count: 2,
            })
        }
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;

    #[test]
    fn grouped_pipeline_resolution_failure_does_not_mutate_encoder() -> Result<()> {
        let device = MlxDevice::new()?;
        let m = 9u32;
        let n = 7u32;
        let k = 35u32;
        let top_k = 2u32;
        let n_experts = 4u32;
        let expert_stride_bytes = u64::from(n) * u64::from(k) * DType::F32.size_of() as u64;
        let weight_bytes = usize::try_from(u64::from(n_experts) * expert_stride_bytes)
            .map_err(|_| MlxError::InvalidArgument("test weight extent overflow".into()))?;
        let weights = device.alloc_buffer(weight_bytes, DType::F32, vec![weight_bytes / 4])?;
        let input = device.alloc_buffer(
            usize::try_from(u64::from(m) * u64::from(k) * 4)
                .map_err(|_| MlxError::InvalidArgument("test input extent overflow".into()))?,
            DType::F32,
            vec![m as usize, k as usize],
        )?;
        let expert_ids = device.alloc_buffer(
            usize::try_from(u64::from(m) * u64::from(top_k) * 4)
                .map_err(|_| MlxError::InvalidArgument("test ID extent overflow".into()))?,
            DType::U32,
            vec![m as usize, top_k as usize],
        )?;
        let output = device.alloc_buffer(
            usize::try_from(u64::from(m) * u64::from(top_k) * u64::from(n) * 4)
                .map_err(|_| MlxError::InvalidArgument("test output extent overflow".into()))?,
            DType::F32,
            vec![m as usize, top_k as usize, n as usize],
        )?;
        let scratch = DenseMatmulIdScratch::new(&device, n_experts, m)?;
        let params = DenseMatmulIdParams {
            m,
            n,
            k,
            top_k,
            n_experts,
            expert_stride_bytes,
            input_layout: DenseMatmulIdInputLayout::SharedPerToken,
            id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
            route: DenseMatmulIdRoute::GroupedPrefill,
        };
        let mut registry = KernelRegistry::new();
        registry.inject_next_pipeline_lookup_failure("dense_matmul_id_grouped_f32_f32");
        let mut encoder = device.command_encoder()?;
        encoder.start_capture();
        let error = dense_matmul_id(
            &mut encoder,
            &mut registry,
            &device,
            &weights,
            &input,
            &expert_ids,
            &output,
            Some(&scratch),
            &params,
        )
        .expect_err("grouped pipeline resolution must fail before encode");
        assert!(error
            .to_string()
            .contains("dense_matmul_id_grouped_f32_f32"));
        let captured = encoder.take_capture().ok_or_else(|| {
            MlxError::InvalidArgument("test capture unexpectedly disappeared".into())
        })?;
        assert!(captured.is_empty());

        registry.inject_next_pipeline_lookup_failure("dense_matmul_id_grouped_f32_f32");
        let mut receipt_encoder = device.command_encoder()?;
        receipt_encoder.start_encoded_dispatch_receipt(2)?;
        dense_matmul_id(
            &mut receipt_encoder,
            &mut registry,
            &device,
            &weights,
            &input,
            &expert_ids,
            &output,
            Some(&scratch),
            &params,
        )
        .expect_err("grouped pipeline resolution must not encode a dispatch");
        assert!(receipt_encoder.take_encoded_dispatch_receipt()?.is_empty());
        Ok(())
    }
}
