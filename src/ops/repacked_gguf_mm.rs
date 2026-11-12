//! Dense MM over load-once repacked Q4_K/Q6_K weights.
//!
//! A weight is prepared once and then multiplies any number of activation
//! rows. Rows are processed in 32-row tiles; the last tile may be partial.
//! The bits of an output row depend only on that row and the weight: not on
//! the row count, and not on the row's position among the rows.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
use crate::ops::quantized_matmul_ggml::{GgmlQuantizedMatmulParams, GgmlType};

pub const REPACK_KERNEL: &str = "hf2q_gguf_repack_q4k_q6k";
pub const Q4K_MM_KERNEL: &str = "hf2q_repacked_mm_q4k_f32";
pub const Q6K_MM_KERNEL: &str = "hf2q_repacked_mm_q6k_f32";
pub const Q4K_MM_BF16_INPUT_KERNEL: &str = "hf2q_repacked_mm_q4k_bf16_input_f32";
pub const Q6K_MM_BF16_INPUT_KERNEL: &str = "hf2q_repacked_mm_q6k_bf16_input_f32";
pub const REDUCE_KERNEL: &str = "hf2q_repacked_mm_reduce_f32";
pub static SOURCE: &str = include_str!("../shaders/gguf_repacked_mm.metal");

/// Kernel variants for one weight type and activation dtype, indexed by
/// `[partial tile][serial splits]`.
type MmKernels = [[&'static str; 2]; 2];

const Q4K_MM_KERNELS: MmKernels = [
    [Q4K_MM_KERNEL, "hf2q_repacked_mm_q4k_f32_serial"],
    [
        "hf2q_repacked_mm_q4k_f32_edge",
        "hf2q_repacked_mm_q4k_f32_edge_serial",
    ],
];
const Q6K_MM_KERNELS: MmKernels = [
    [Q6K_MM_KERNEL, "hf2q_repacked_mm_q6k_f32_serial"],
    [
        "hf2q_repacked_mm_q6k_f32_edge",
        "hf2q_repacked_mm_q6k_f32_edge_serial",
    ],
];
const Q4K_MM_BF16_INPUT_KERNELS: MmKernels = [
    [
        Q4K_MM_BF16_INPUT_KERNEL,
        "hf2q_repacked_mm_q4k_bf16_input_f32_serial",
    ],
    [
        "hf2q_repacked_mm_q4k_bf16_input_f32_edge",
        "hf2q_repacked_mm_q4k_bf16_input_f32_edge_serial",
    ],
];
const Q6K_MM_BF16_INPUT_KERNELS: MmKernels = [
    [
        Q6K_MM_BF16_INPUT_KERNEL,
        "hf2q_repacked_mm_q6k_bf16_input_f32_serial",
    ],
    [
        "hf2q_repacked_mm_q6k_bf16_input_f32_edge",
        "hf2q_repacked_mm_q6k_bf16_input_f32_edge_serial",
    ],
];

/// Activation rows per tile.
const TILE_ROWS: u32 = 32;

pub fn register(registry: &mut KernelRegistry) {
    let mm_kernels = [
        Q4K_MM_KERNELS,
        Q6K_MM_KERNELS,
        Q4K_MM_BF16_INPUT_KERNELS,
        Q6K_MM_BF16_INPUT_KERNELS,
    ];
    for name in [REPACK_KERNEL, REDUCE_KERNEL]
        .into_iter()
        .chain(mm_kernels.into_iter().flatten().flatten())
    {
        registry.register_source(name, SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RepackParams {
    rows: u32,
    input_size: u32,
    format: u32,
    plane1_offset: u32,
    meta_offset: u32,
    src_row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MmParams {
    k: u32,
    n: u32,
    m: u32,
    splits: u32,
}

/// Who adds the K splits of a prepared weight. Both schedules add the same
/// partial sums in the same order, so they produce identical bits; only the
/// threadgroup count differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitSchedule {
    /// One pass per row tile with one threadgroup per split. Partial sums go
    /// through the prepared scratch and the reduce kernel. Keeps a few row
    /// tiles wide enough to fill the GPU.
    Grid,
    /// One pass over every row tile. A threadgroup walks every split and adds
    /// each finished split into the output; no scratch and no reduce pass.
    Serial,
}

/// A Q4_K or Q6_K weight repacked for the prepared MM. One prepared weight
/// serves every row count.
///
/// The split-grid schedule passes through the weight's scratch, so two
/// dispatches of one prepared weight need a memory barrier between them.
#[derive(Clone)]
pub struct PreparedGgufWeight {
    _source: MlxBuffer,
    bytes: MlxBuffer,
    plane1_offset: u32,
    /// Split-grid scratch for one row tile, `[splits, TILE_ROWS, n]`; absent
    /// for one split.
    partials: Option<MlxBuffer>,
    meta_offset: u32,
    n: u32,
    k: u32,
    ggml_type: GgmlType,
    /// K-split count. It fixes the summation order of every output element,
    /// so it is part of the weight's numerics and never follows the row count.
    splits: u32,
    /// Most row tiles served by the split-grid schedule.
    split_grid_row_tiles: u32,
}

impl PreparedGgufWeight {
    fn split_schedule(&self, rows: u32) -> SplitSchedule {
        if rows.div_ceil(TILE_ROWS) <= self.split_grid_row_tiles {
            SplitSchedule::Grid
        } else {
            SplitSchedule::Serial
        }
    }
}

/// Weight bits one threadgroup decodes per K step, 64 rows of 32 values, or
/// `None` for a type the prepared MM does not serve.
fn step_bits(ggml_type: GgmlType) -> Option<u64> {
    let bits_sixteenths = match ggml_type {
        GgmlType::Q4_K => 72_u64,
        GgmlType::Q6_K => 105_u64,
        _ => return None,
    };
    Some(64 * 32 * bits_sixteenths / 16)
}

/// Cost of one pass in the staged-wave model of the M5 Max: `groups`
/// threadgroups that each walk `steps` K steps and, with `partials`, store a
/// row tile of partial sums that the reduce pass reads back.
fn wave_cost(groups: u64, steps: u64, step_bits: u64, partials: bool) -> u64 {
    const M5_MAX_CORES: u64 = 40;
    const RESIDENT_GROUPS: u64 = 10;
    const SATURATING_GROUPS: u64 = 6;
    const FOLLOWING_GROUPS: u64 = 2;
    let per_core = groups.div_ceil(M5_MAX_CORES);
    let waves = per_core.div_ceil(RESIDENT_GROUPS);
    let full = (waves - 1) * RESIDENT_GROUPS * M5_MAX_CORES;
    let floor = if waves > 1 {
        FOLLOWING_GROUPS
    } else {
        SATURATING_GROUPS
    } * M5_MAX_CORES;
    let partial_bits = if partials {
        2 * u64::from(TILE_ROWS) * 64 * 32
    } else {
        0
    };
    steps * step_bits * (full + (groups - full).max(floor)) + groups * partial_bits
}

/// K-split count of a weight shape: the cheapest split-grid pass over one row
/// tile. Changing the result for a shape changes that weight's output bits.
fn target_splits(n: u32, k: u32, ggml_type: GgmlType) -> u32 {
    let Some(step_bits) = step_bits(ggml_type) else {
        return 1;
    };
    let mut best_splits = 1;
    let mut best_cost = u64::MAX;
    for splits in [1_u32, 2, 4, 8] {
        if (k / 32) % splits != 0 || (splits > 1 && k / splits < 512) {
            continue;
        }
        let groups = u64::from(n / 64) * u64::from(splits);
        let steps = u64::from(k / 32 / splits);
        let cost = wave_cost(groups, steps, step_bits, splits > 1);
        if cost < best_cost {
            best_cost = cost;
            best_splits = splits;
        }
    }
    best_splits
}

/// Most row tiles for which one split-grid pass per tile is modelled cheaper
/// than a single serial pass over all of them. Few row tiles leave a serial
/// pass short of threadgroups; many fill the GPU without splitting K.
fn split_grid_row_tiles(n: u32, k: u32, ggml_type: GgmlType, splits: u32) -> u32 {
    let Some(step_bits) = step_bits(ggml_type).filter(|_| splits > 1) else {
        return 1;
    };
    let column_tiles = u64::from(n / 64);
    let steps = u64::from(k / 32);
    let grid_pass = wave_cost(
        column_tiles * u64::from(splits),
        steps / u64::from(splits),
        step_bits,
        true,
    );
    let mut row_tiles = 1_u32;
    while u64::from(row_tiles + 1) * grid_pass
        < wave_cost(
            column_tiles * u64::from(row_tiles + 1),
            steps,
            step_bits,
            false,
        )
    {
        row_tiles += 1;
    }
    row_tiles
}

/// Repack a Q4_K or Q6_K weight `[N, K]` for the prepared MM.
///
/// The result does not depend on a row count; `params.m` is ignored.
pub fn prepare_repacked_gguf_weight(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<PreparedGgufWeight> {
    if weight.dtype() != DType::U8 {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF preparation requires native U8 blocks, got {:?}",
            weight.dtype()
        )));
    }
    let format: u8 = match params.ggml_type {
        GgmlType::Q4_K => 0,
        GgmlType::Q6_K => 1,
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "repacked GGUF MM does not support {other:?}"
            )))
        }
    };
    if params.n % 64 != 0 || params.k % 256 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM requires N % 64 == 0 and K % 256 == 0, got N={} K={}",
            params.n, params.k
        )));
    }
    let rows = params.n as usize;
    let source_bytes = rows
        .checked_mul(params.k as usize / 256)
        .and_then(|value| value.checked_mul(params.ggml_type.block_bytes() as usize))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF source size overflow".into()))?;
    if weight.data_byte_len() < source_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF source needs {source_bytes} bytes, got {}",
            weight.data_byte_len()
        )));
    }

    let groups = params.k as usize / 32;
    let plane0 = rows
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF plane size overflow".into()))?;
    let plane1 = if format == 1 {
        rows.checked_mul(groups)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                MlxError::InvalidArgument("repacked GGUF high-bit plane size overflow".into())
            })?
    } else {
        0
    };
    let meta = rows
        .checked_mul(groups / 8)
        .and_then(|v| v.checked_mul(if format == 1 { 20 } else { 16 }))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF metadata size overflow".into()))?;
    let total = plane0
        .checked_add(plane1)
        .and_then(|v| v.checked_add(meta))
        .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF size overflow".into()))?;
    let bytes = device.alloc_buffer(total, DType::U8, vec![total])?;
    let splits = target_splits(params.n, params.k, params.ggml_type);
    let partials = if splits > 1 {
        let shape = vec![splits as usize, TILE_ROWS as usize, params.n as usize];
        let bytes = shape
            .iter()
            .try_fold(std::mem::size_of::<f32>(), |bytes, &extent| {
                bytes.checked_mul(extent)
            })
            .ok_or_else(|| {
                MlxError::InvalidArgument("repacked GGUF partial size overflow".into())
            })?;
        Some(device.alloc_buffer(bytes, DType::F32, shape)?)
    } else {
        None
    };
    let p = RepackParams {
        rows: params.n,
        input_size: params.k,
        format: u32::from(format),
        plane1_offset: u32::try_from(plane0).map_err(|_| {
            MlxError::InvalidArgument("repacked GGUF plane offset exceeds u32".into())
        })?,
        meta_offset: u32::try_from(plane0 + plane1).map_err(|_| {
            MlxError::InvalidArgument("repacked GGUF meta offset exceeds u32".into())
        })?,
        src_row_bytes: (params.k / 256) * params.ggml_type.block_bytes(),
    };
    let pipeline = registry.get_pipeline(REPACK_KERNEL, device.metal_device())?;
    let threads = rows * groups;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(&bytes)),
            (2, KernelArg::Bytes(as_bytes(&p))),
        ],
        MTLSize::new(threads as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    Ok(PreparedGgufWeight {
        _source: weight.clone(),
        bytes,
        partials,
        plane1_offset: p.plane1_offset,
        meta_offset: p.meta_offset,
        n: params.n,
        k: params.k,
        ggml_type: params.ggml_type,
        splits,
        split_grid_row_tiles: split_grid_row_tiles(params.n, params.k, params.ggml_type, splits),
    })
}

/// Multiply `params.m >= 1` F32 activation rows `[M, K]` by a prepared weight,
/// writing F32 `[M, N]`.
///
/// Rows at or beyond `M` are neither read nor written.
///
/// Every activation is recast to half before it is multiplied: it keeps 11
/// significant bits, and a magnitude beyond the largest finite half, 65504,
/// overflows to infinity. Stock `quantized_matmul_ggml` does the same at 32
/// rows but multiplies F32 activations unrounded at other row counts, so for
/// values that are not exact in half this entry agrees with stock to about
/// 1e-3 (relative RMSE) there, against 3e-4 at 32 rows and on the BF16 route.
/// The recast is repeated once per output-column tile; prefer
/// [`dispatch_repacked_gguf_mm_bf16_input`] for BF16 activations.
pub fn dispatch_repacked_gguf_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    dispatch_repacked_gguf_mm_typed(
        encoder,
        registry,
        device,
        input,
        prepared,
        output,
        params,
        DType::F32,
    )
}

/// Multiply `params.m >= 1` BF16 activation rows `[M, K]` by a prepared
/// weight, writing F32 `[M, N]`.
///
/// Rows at or beyond `M` are neither read nor written. Unlike
/// [`dispatch_repacked_gguf_mm`], this route feeds the device BF16 matrix
/// directly to MPP. It does not restage and recast the same activation once
/// per output-column tile.
pub fn dispatch_repacked_gguf_mm_bf16_input(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    dispatch_repacked_gguf_mm_typed(
        encoder,
        registry,
        device,
        input,
        prepared,
        output,
        params,
        DType::BF16,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_repacked_gguf_mm_typed(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    input_dtype: DType,
) -> Result<()> {
    if params.m == 0 || input.dtype() != input_dtype || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM requires M >= 1, {input_dtype:?} input, and F32 output buffers"
        )));
    }
    if prepared.n != params.n || prepared.k != params.k || prepared.ggml_type != params.ggml_type {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM parameters do not match the prepared weight: prepared N={} K={} {:?}, got N={} K={} {:?}",
            prepared.n, prepared.k, prepared.ggml_type, params.n, params.k, params.ggml_type
        )));
    }
    encode_repacked_gguf_mm(
        encoder,
        registry,
        device,
        input,
        prepared,
        output,
        params.m,
        input_dtype,
        prepared.split_schedule(params.m),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_repacked_gguf_mm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    rows: u32,
    input_dtype: DType,
    schedule: SplitSchedule,
) -> Result<()> {
    let row_bytes = |width: u32, element: usize| {
        (rows as usize)
            .checked_mul(width as usize)
            .and_then(|elements| elements.checked_mul(element))
            .ok_or_else(|| MlxError::InvalidArgument("repacked GGUF MM size overflow".into()))
    };
    let input_bytes = row_bytes(prepared.k, input_dtype.size_of())?;
    let output_bytes = row_bytes(prepared.n, std::mem::size_of::<f32>())?;
    if input.data_byte_len() < input_bytes || output.data_byte_len() < output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repacked GGUF MM buffer is too small: input {}/{input_bytes}, output {}/{output_bytes}",
            input.data_byte_len(),
            output.data_byte_len()
        )));
    }
    match schedule {
        SplitSchedule::Grid if rows <= TILE_ROWS => encode_split_grid_tile(
            encoder,
            registry,
            device,
            input,
            prepared,
            output,
            rows,
            input_dtype,
        ),
        SplitSchedule::Grid => {
            let input_row = prepared.k as usize * input_dtype.size_of();
            let output_row = prepared.n as usize * std::mem::size_of::<f32>();
            for first_row in (0..rows).step_by(TILE_ROWS as usize) {
                let tile_rows = (rows - first_row).min(TILE_ROWS);
                if first_row > 0 && prepared.partials.is_some() {
                    // The next tile reuses the partial-sum scratch.
                    encoder.memory_barrier();
                }
                let tile_input = input.slice_view(
                    (first_row as usize * input_row) as u64,
                    tile_rows as usize * prepared.k as usize,
                );
                let tile_output = output.slice_view(
                    (first_row as usize * output_row) as u64,
                    tile_rows as usize * prepared.n as usize,
                );
                encode_split_grid_tile(
                    encoder,
                    registry,
                    device,
                    &tile_input,
                    prepared,
                    &tile_output,
                    tile_rows,
                    input_dtype,
                )?;
            }
            Ok(())
        }
        SplitSchedule::Serial => {
            let column_tiles = u64::from(prepared.n / 64);
            let full_tiles = u64::from(rows / TILE_ROWS);
            let edge_tiles = u64::from(rows % TILE_ROWS != 0);
            for (edge, row_tiles) in [(false, full_tiles), (true, edge_tiles)] {
                if row_tiles > 0 {
                    encode_mm_pass(
                        encoder,
                        registry,
                        device,
                        input,
                        prepared,
                        output,
                        output,
                        rows,
                        input_dtype,
                        edge,
                        true,
                        MTLSize::new(row_tiles, column_tiles, 1),
                    )?;
                }
            }
            Ok(())
        }
    }
}

/// Encode one row tile (`rows <= TILE_ROWS`) with one threadgroup per K
/// split, then the reduce pass when the weight has several splits.
#[allow(clippy::too_many_arguments)]
fn encode_split_grid_tile(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    rows: u32,
    input_dtype: DType,
) -> Result<()> {
    let partials = prepared.partials.as_ref();
    encode_mm_pass(
        encoder,
        registry,
        device,
        input,
        prepared,
        output,
        partials.unwrap_or(output),
        rows,
        input_dtype,
        rows < TILE_ROWS,
        false,
        MTLSize::new(u64::from(prepared.splits), u64::from(prepared.n / 64), 1),
    )?;
    if let Some(partials) = partials {
        encoder.memory_barrier();
        let reduce = registry.get_pipeline(REDUCE_KERNEL, device.metal_device())?;
        let elements = rows.checked_mul(prepared.n).ok_or_else(|| {
            MlxError::InvalidArgument("repacked GGUF MM tile exceeds u32 elements".into())
        })?;
        let dims = [elements, prepared.splits];
        encode_with_args(
            encoder,
            reduce,
            &[
                (0, KernelArg::Buffer(partials)),
                (1, KernelArg::Buffer(output)),
                (2, KernelArg::Bytes(as_bytes(&dims))),
            ],
            MTLSize::new(u64::from(elements), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    Ok(())
}

/// Encode one MM kernel pass over the `rows` activation rows bound to it.
#[allow(clippy::too_many_arguments)]
fn encode_mm_pass(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &PreparedGgufWeight,
    output: &MlxBuffer,
    partials: &MlxBuffer,
    rows: u32,
    input_dtype: DType,
    edge: bool,
    serial: bool,
    threadgroups: MTLSize,
) -> Result<()> {
    let kernels = match (prepared.ggml_type, input_dtype) {
        (GgmlType::Q4_K, DType::F32) => Q4K_MM_KERNELS,
        (GgmlType::Q6_K, DType::F32) => Q6K_MM_KERNELS,
        (GgmlType::Q4_K, DType::BF16) => Q4K_MM_BF16_INPUT_KERNELS,
        (GgmlType::Q6_K, DType::BF16) => Q6K_MM_BF16_INPUT_KERNELS,
        _ => unreachable!(),
    };
    let kernel = kernels[usize::from(edge)][usize::from(serial)];
    let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
    let p = MmParams {
        k: prepared.k,
        n: prepared.n,
        m: rows,
        splits: prepared.splits,
    };
    let offsets = [prepared.plane1_offset, prepared.meta_offset];
    let shared_bytes = if input_dtype == DType::BF16 {
        8_192
    } else {
        12_288
    };
    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&p))),
            (1, KernelArg::Buffer(&prepared.bytes)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&offsets))),
            (5, KernelArg::Buffer(partials)),
        ],
        &[(0, shared_bytes)],
        threadgroups,
        MTLSize::new(64, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    use crate::ops::quantized_matmul_ggml::quantized_matmul_ggml;

    /// Row counts the row-tiling invariants are checked at: one row, one short
    /// of a tile, one tile, one past a tile, two whole tiles, and two
    /// prefill-sized batches whose last tile is partial.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    const ROW_COUNTS: [usize; 7] = [1, 31, 32, 33, 64, 157, 720];

    /// Small `(N, K)` shapes with their K-split counts. Together they cover
    /// every split count the policy can choose; the last has an odd number of
    /// K steps per split, so consecutive splits of a serial pass start on the
    /// weight staging buffer the previous one ended on.
    const SPLIT_SHAPES: [(usize, usize, u32); 5] = [
        (192, 512, 1),
        (128, 1024, 2),
        (128, 2048, 4),
        (64, 4096, 8),
        (64, 4352, 8),
    ];

    /// Qwen3.8-27B projections: name, N, K, weight type, K-split count.
    const QWEN_SHAPES: [(&str, usize, usize, GgmlType, u32); 8] = [
        ("ffn-gate", 17_408, 5_120, GgmlType::Q4_K, 1),
        ("ffn-down", 5_120, 17_408, GgmlType::Q6_K, 4),
        ("attn-q-gate", 12_288, 5_120, GgmlType::Q4_K, 2),
        ("attn-kv", 1_024, 5_120, GgmlType::Q4_K, 8),
        ("attn-out", 5_120, 6_144, GgmlType::Q4_K, 4),
        ("gdn-qkv", 8_192, 5_120, GgmlType::Q4_K, 2),
        ("gdn-gate", 4_096, 5_120, GgmlType::Q4_K, 4),
        ("gdn-out", 5_120, 4_096, GgmlType::Q4_K, 4),
    ];

    fn buffer(device: &MlxDevice, bytes: usize, dtype: DType, shape: Vec<usize>) -> MlxBuffer {
        device.alloc_buffer(bytes, dtype, shape).unwrap()
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn fill_weight(ggml_type: GgmlType, salt: usize, weight: &mut MlxBuffer) {
        let block_bytes = ggml_type.block_bytes() as usize;
        for (block, bytes) in weight
            .as_mut_slice::<u8>()
            .unwrap()
            .chunks_exact_mut(block_bytes)
            .enumerate()
        {
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = ((block * 31 + i * 13 + salt) & 255) as u8;
            }
            match ggml_type {
                GgmlType::Q4_K => {
                    bytes[0..2].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                    bytes[2..4].copy_from_slice(&half::f16::from_f32(0.0078125).to_le_bytes());
                    for scale in &mut bytes[4..16] {
                        *scale &= 0x3f;
                    }
                }
                GgmlType::Q6_K => {
                    for scale in &mut bytes[192..208] {
                        *scale = (*scale as i8).clamp(-31, 31) as u8;
                    }
                    bytes[208..210].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }

    /// Element `col` of synthetic activation row `row`, in [-1, 1). A row's
    /// content does not depend on the batch it is placed in.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn activation(row: usize, col: usize) -> f32 {
        let mut h = (row as u32).wrapping_mul(0x9E37_79B1) ^ (col as u32).wrapping_mul(0x85EB_CA77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        (h & 0xFFFF) as f32 / 32768.0 - 1.0
    }

    /// One synthetic weight, prepared once and reused for every row count.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    struct Harness {
        device: MlxDevice,
        registry: KernelRegistry,
        weight: MlxBuffer,
        prepared: PreparedGgufWeight,
        ggml_type: GgmlType,
        n: usize,
        k: usize,
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    impl Harness {
        fn new(ggml_type: GgmlType, n: usize, k: usize) -> Self {
            let device = MlxDevice::new().unwrap();
            let mut registry = KernelRegistry::new();
            let mut weight = buffer(
                &device,
                n * (k / ggml_type.block_values() as usize) * ggml_type.block_bytes() as usize,
                DType::U8,
                vec![n, k],
            );
            fill_weight(ggml_type, 7, &mut weight);
            let params = GgmlQuantizedMatmulParams {
                m: 1,
                n: n as u32,
                k: k as u32,
                ggml_type,
            };
            let mut encoder = device.command_encoder().unwrap();
            let prepared = prepare_repacked_gguf_weight(
                &mut encoder,
                &mut registry,
                &device,
                &weight,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            Self {
                device,
                registry,
                weight,
                prepared,
                ggml_type,
                n,
                k,
            }
        }

        fn params(&self, m: usize) -> GgmlQuantizedMatmulParams {
            GgmlQuantizedMatmulParams {
                m: m as u32,
                n: self.n as u32,
                k: self.k as u32,
                ggml_type: self.ggml_type,
            }
        }

        /// Activation `[rows.len(), K]` holding the given synthetic rows in
        /// order. `bf16_values` rounds F32 elements to BF16 precision.
        fn activations(&self, rows: &[usize], dtype: DType, bf16_values: bool) -> MlxBuffer {
            let k = self.k;
            let values = rows
                .iter()
                .flat_map(|&row| (0..k).map(move |col| activation(row, col)));
            let mut input = buffer(
                &self.device,
                rows.len() * k * dtype.size_of(),
                dtype,
                vec![rows.len(), k],
            );
            match dtype {
                DType::BF16 => {
                    for (slot, value) in input
                        .as_mut_slice::<half::bf16>()
                        .unwrap()
                        .iter_mut()
                        .zip(values)
                    {
                        *slot = half::bf16::from_f32(value);
                    }
                }
                DType::F32 => {
                    for (slot, value) in input.as_mut_slice::<f32>().unwrap().iter_mut().zip(values)
                    {
                        *slot = if bf16_values {
                            half::bf16::from_f32(value).to_f32()
                        } else {
                            value
                        };
                    }
                }
                _ => unreachable!(),
            }
            input
        }

        /// Activation of the given dtype: BF16-rounded, or F32 as generated.
        fn input(&self, rows: &[usize], dtype: DType) -> MlxBuffer {
            self.activations(rows, dtype, false)
        }

        /// F32 activation holding the BF16-rounded values, which are exact in
        /// BF16, half and F32 alike.
        fn bf16_exact_f32(&self, rows: &[usize]) -> MlxBuffer {
            self.activations(rows, DType::F32, true)
        }

        fn output(&self, m: usize) -> MlxBuffer {
            buffer(&self.device, m * self.n * 4, DType::F32, vec![m, self.n])
        }

        /// Encode the public prepared dispatch matching the input's dtype.
        fn encode(&mut self, encoder: &mut CommandEncoder, input: &MlxBuffer, output: &MlxBuffer) {
            let params = self.params(input.shape()[0]);
            if input.dtype() == DType::BF16 {
                dispatch_repacked_gguf_mm_bf16_input(
                    encoder,
                    &mut self.registry,
                    &self.device,
                    input,
                    &self.prepared,
                    output,
                    &params,
                )
            } else {
                dispatch_repacked_gguf_mm(
                    encoder,
                    &mut self.registry,
                    &self.device,
                    input,
                    &self.prepared,
                    output,
                    &params,
                )
            }
            .unwrap();
        }

        fn encode_stock(
            &mut self,
            encoder: &mut CommandEncoder,
            input: &MlxBuffer,
            output: &MlxBuffer,
        ) {
            let params = self.params(input.shape()[0]);
            quantized_matmul_ggml(
                encoder,
                &mut self.registry,
                &self.device,
                input,
                &self.weight,
                output,
                &params,
            )
            .unwrap();
        }

        /// Output bits of the prepared route for an activation buffer.
        fn run_input(&mut self, input: &MlxBuffer) -> Vec<u32> {
            let output = self.output(input.shape()[0]);
            let mut encoder = self.device.command_encoder().unwrap();
            self.encode(&mut encoder, input, &output);
            encoder.commit_and_wait().unwrap();
            output.as_slice::<u32>().unwrap().to_vec()
        }

        /// Output bits of the prepared route for the given activation rows.
        fn run(&mut self, rows: &[usize], dtype: DType) -> Vec<u32> {
            self.run_input(&self.input(rows, dtype))
        }

        /// Output bits for an activation buffer under an explicit split
        /// schedule.
        fn run_input_scheduled(&mut self, input: &MlxBuffer, schedule: SplitSchedule) -> Vec<u32> {
            let rows = input.shape()[0];
            let output = self.output(rows);
            let mut encoder = self.device.command_encoder().unwrap();
            encode_repacked_gguf_mm(
                &mut encoder,
                &mut self.registry,
                &self.device,
                input,
                &self.prepared,
                &output,
                rows as u32,
                input.dtype(),
                schedule,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            output.as_slice::<u32>().unwrap().to_vec()
        }

        /// Like [`Self::run`], under an explicit split schedule.
        fn run_scheduled(
            &mut self,
            rows: &[usize],
            dtype: DType,
            schedule: SplitSchedule,
        ) -> Vec<u32> {
            self.run_input_scheduled(&self.input(rows, dtype), schedule)
        }

        /// Stock output for an F32 activation buffer.
        fn run_stock(&mut self, input: &MlxBuffer) -> Vec<f32> {
            let output = self.output(input.shape()[0]);
            let mut encoder = self.device.command_encoder().unwrap();
            self.encode_stock(&mut encoder, input, &output);
            encoder.commit_and_wait().unwrap();
            output.as_slice::<f32>().unwrap().to_vec()
        }
    }

    /// The width-32 kernels exactly as shipped in 0.21.7, compiled from their
    /// original source. The served decode route runs at 32 rows and must keep
    /// these bits.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    struct Width32Reference {
        registry: KernelRegistry,
    }

    /// `MmParams` of the 0.21.7 kernels.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Width32ReferenceParams {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        splits: u32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
        _pad1: u32,
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    impl Width32Reference {
        const ROWS: usize = 32;

        fn new() -> Self {
            static SOURCE: &str =
                include_str!("../../tests/fixtures/gguf_repacked_mm_width32_reference.metal");
            let mut registry = KernelRegistry::new();
            for name in [
                Q4K_MM_KERNEL,
                Q6K_MM_KERNEL,
                Q4K_MM_BF16_INPUT_KERNEL,
                Q6K_MM_BF16_INPUT_KERNEL,
                REDUCE_KERNEL,
            ] {
                registry.register_source(name, SOURCE);
            }
            Self { registry }
        }

        /// The 0.21.7 dispatch, encoded verbatim.
        fn encode(
            &mut self,
            encoder: &mut CommandEncoder,
            harness: &Harness,
            input: &MlxBuffer,
            output: &MlxBuffer,
        ) {
            let prepared = &harness.prepared;
            let element = input.dtype().size_of() as u64;
            let k = harness.k as u64;
            let p = Width32ReferenceParams {
                ne00: harness.k as i32,
                ne02: 1,
                nb01: k / 256 * u64::from(harness.ggml_type.block_bytes()),
                nb02: 0,
                nb03: 0,
                ne12: 1,
                splits: prepared.splits,
                nb10: element,
                nb11: k * element,
                nb12: Self::ROWS as u64 * k * element,
                nb13: 0,
                ne0: harness.n as i32,
                ne1: Self::ROWS as i32,
                r2: 1,
                r3: 1,
                _pad1: 0,
            };
            let kernel = match (harness.ggml_type, input.dtype()) {
                (GgmlType::Q4_K, DType::F32) => Q4K_MM_KERNEL,
                (GgmlType::Q6_K, DType::F32) => Q6K_MM_KERNEL,
                (GgmlType::Q4_K, DType::BF16) => Q4K_MM_BF16_INPUT_KERNEL,
                (GgmlType::Q6_K, DType::BF16) => Q6K_MM_BF16_INPUT_KERNEL,
                _ => unreachable!(),
            };
            let device = harness.device.metal_device();
            let pipeline = self.registry.get_pipeline(kernel, device).unwrap();
            let offsets = [prepared.plane1_offset, prepared.meta_offset];
            let partials = prepared.partials.as_ref().unwrap_or(output);
            let shared_bytes = if input.dtype() == DType::BF16 {
                8_192
            } else {
                12_288
            };
            encoder.encode_threadgroups_with_args_and_shared(
                &pipeline,
                &[
                    (0, KernelArg::Bytes(as_bytes(&p))),
                    (1, KernelArg::Buffer(&prepared.bytes)),
                    (2, KernelArg::Buffer(input)),
                    (3, KernelArg::Buffer(output)),
                    (4, KernelArg::Bytes(as_bytes(&offsets))),
                    (5, KernelArg::Buffer(partials)),
                ],
                &[(0, shared_bytes)],
                MTLSize::new(u64::from(prepared.splits), harness.n as u64 / 64, 1),
                MTLSize::new(64, 1, 1),
            );
            if prepared.splits > 1 {
                encoder.memory_barrier();
                let reduce = self.registry.get_pipeline(REDUCE_KERNEL, device).unwrap();
                let elements = (Self::ROWS * harness.n) as u32;
                let dims = [elements, prepared.splits];
                encode_with_args(
                    encoder,
                    reduce,
                    &[
                        (0, KernelArg::Buffer(partials)),
                        (1, KernelArg::Buffer(output)),
                        (2, KernelArg::Bytes(as_bytes(&dims))),
                    ],
                    MTLSize::new(u64::from(elements), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
        }

        /// Output bits of the 0.21.7 kernels for 32 activation rows.
        fn run(&mut self, harness: &Harness, rows: &[usize], dtype: DType) -> Vec<u32> {
            assert_eq!(rows.len(), Self::ROWS);
            let input = harness.input(rows, dtype);
            let output = harness.output(Self::ROWS);
            let mut encoder = harness.device.command_encoder().unwrap();
            self.encode(&mut encoder, harness, &input, &output);
            encoder.commit_and_wait().unwrap();
            output.as_slice::<u32>().unwrap().to_vec()
        }
    }

    /// Invariant: at 32 rows the output is bitwise identical to the 0.21.7
    /// kernels, for every split count and for the served Qwen3.8 shapes.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn width_32_output_is_bitwise_identical_to_the_0_21_7_kernels() {
        let mut reference = Width32Reference::new();
        let rows: Vec<usize> = (0..Width32Reference::ROWS).collect();
        let small = SPLIT_SHAPES
            .into_iter()
            .flat_map(|(n, k, _)| [(n, k, GgmlType::Q4_K), (n, k, GgmlType::Q6_K)]);
        let served = QWEN_SHAPES
            .into_iter()
            .map(|(_, n, k, ggml_type, _)| (n, k, ggml_type));
        for (n, k, ggml_type) in small.chain(served) {
            let mut harness = Harness::new(ggml_type, n, k);
            for dtype in [DType::BF16, DType::F32] {
                let expected = reference.run(&harness, &rows, dtype);
                assert!(expected.iter().any(|&bits| bits != 0));
                assert!(
                    harness.run(&rows, dtype) == expected,
                    "{ggml_type:?} N={n} K={k} {dtype:?} input differs from the 0.21.7 kernel"
                );
            }
        }
    }

    /// Invariant: an output row has the same bits whatever the row count and
    /// wherever the row sits among the rows.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn rows_are_independent_of_their_batch(ggml_type: GgmlType) {
        const POOL: usize = 720;
        for (n, k, _) in SPLIT_SHAPES {
            let mut harness = Harness::new(ggml_type, n, k);
            for dtype in [DType::BF16, DType::F32] {
                let in_order: Vec<usize> = (0..POOL).collect();
                let reference = harness.run(&in_order, dtype);
                let reversed: Vec<usize> = (0..POOL).rev().collect();
                let mut batches = vec![reversed];
                for m in ROW_COUNTS {
                    // Start off a tile boundary so that every row changes its
                    // position within its tile.
                    let first = (POOL - m) * 5 / 7;
                    assert!(first % 32 != 0 || m == POOL);
                    batches.push((first..first + m).collect());
                }
                for batch in batches {
                    let bits = harness.run(&batch, dtype);
                    for (position, &row) in batch.iter().enumerate() {
                        assert!(
                            bits[position * n..(position + 1) * n]
                                == reference[row * n..(row + 1) * n],
                            "{ggml_type:?} N={n} K={k} {dtype:?}: row {row} at position {position} of {} rows differs from its bits in the {POOL}-row batch",
                            batch.len()
                        );
                    }
                }
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q4k_rows_are_independent_of_their_batch() {
        rows_are_independent_of_their_batch(GgmlType::Q4_K);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q6k_rows_are_independent_of_their_batch() {
        rows_are_independent_of_their_batch(GgmlType::Q6_K);
    }

    /// The split schedule is a scheduling choice only: one serial pass that
    /// adds each finished split into the output gives the bits of per-tile
    /// split grids and their reduce passes, for one tile and for several.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn split_schedules_give_identical_bits() {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            for (n, k, _) in SPLIT_SHAPES {
                let mut harness = Harness::new(ggml_type, n, k);
                for dtype in [DType::BF16, DType::F32] {
                    for m in [1, 31, 32, 33, 64, 157] {
                        let rows: Vec<usize> = (0..m).collect();
                        let grid = harness.run_scheduled(&rows, dtype, SplitSchedule::Grid);
                        let serial = harness.run_scheduled(&rows, dtype, SplitSchedule::Serial);
                        assert!(
                            grid == serial,
                            "{ggml_type:?} N={n} K={k} {dtype:?} M={m}: schedules differ"
                        );
                    }
                }
            }
        }
    }

    /// Row independence and schedule equivalence on the Qwen3.8 shapes, whose
    /// outputs are wide enough to run many threadgroups at once.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn qwen_shapes_keep_row_bits_across_row_counts_and_schedules() {
        const POOL: usize = 200;
        let pool: Vec<usize> = (0..POOL).collect();
        for (name, n, k, ggml_type, _) in QWEN_SHAPES {
            let mut harness = Harness::new(ggml_type, n, k);
            for dtype in [DType::BF16, DType::F32] {
                let reference = harness.run_scheduled(&pool, dtype, SplitSchedule::Serial);
                assert!(reference.iter().any(|&bits| bits != 0));
                assert!(
                    harness.run_scheduled(&pool, dtype, SplitSchedule::Grid) == reference,
                    "{name} {dtype:?} M={POOL}: schedules differ"
                );
                for m in [1, 31, 33, 97, 160] {
                    // Off a tile boundary, so every row changes its position
                    // within its tile.
                    let first = (POOL - m) * 5 / 7;
                    assert!(first % 32 != 0);
                    let batch: Vec<usize> = (first..first + m).collect();
                    assert!(
                        harness.run(&batch, dtype) == reference[first * n..(first + m) * n],
                        "{name} {dtype:?}: rows {first}..{} computed alone differ from their bits in the {POOL}-row batch",
                        first + m
                    );
                }
            }
        }
    }

    /// A split weight's serial schedule in pipelines built with Metal shader
    /// validation, whose instrumentation does not preserve thread-private
    /// state across the K loop the way plain pipelines do: a serial kernel
    /// that kept its running total there stored blocks of zeros in a tenth to
    /// a half of such F32 runs. Runs in a child process, because validation is
    /// fixed when the Metal device is created.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn serial_schedule_matches_split_grid_under_shader_validation() {
        const CHILD: &str = "MLX_REPACKED_MM_SHADER_VALIDATION_CHILD";
        const ROWS: usize = 230;
        const REPEATS: usize = 40;
        if std::env::var_os(CHILD).is_none() {
            let name = concat!(
                module_path!(),
                "::serial_schedule_matches_split_grid_under_shader_validation"
            );
            let (_crate, name) = name.split_once("::").unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env("MTL_SHADER_VALIDATION", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "validated child failed:\n{stdout}\n{stderr}"
            );
            assert!(
                stderr.contains("Metal GPU Validation Enabled"),
                "child ran without shader validation:\n{stderr}"
            );
            return;
        }
        let rows: Vec<usize> = (0..ROWS).collect();
        for (n, k, ggml_type) in [
            (5_120, 17_408, GgmlType::Q6_K),
            (5_120, 17_408, GgmlType::Q4_K),
            (5_120, 6_144, GgmlType::Q4_K),
        ] {
            let mut harness = Harness::new(ggml_type, n, k);
            for dtype in [DType::F32, DType::BF16] {
                let input = harness.input(&rows, dtype);
                let grid = harness.run_input_scheduled(&input, SplitSchedule::Grid);
                assert!(grid.iter().any(|&bits| bits != 0));
                for repeat in 0..REPEATS {
                    let serial = harness.run_input_scheduled(&input, SplitSchedule::Serial);
                    let differing = serial.iter().zip(&grid).filter(|(a, b)| a != b).count();
                    assert!(
                        differing == 0,
                        "{ggml_type:?} N={n} K={k} {dtype:?} repeat {repeat}: {differing} elements differ from the split grid"
                    );
                }
            }
        }
    }

    /// Invariant: two runs give identical bits.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn repeated_runs_give_identical_bits() {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            for (n, k, _) in SPLIT_SHAPES {
                let mut harness = Harness::new(ggml_type, n, k);
                for dtype in [DType::BF16, DType::F32] {
                    for m in ROW_COUNTS {
                        let rows: Vec<usize> = (0..m).collect();
                        let first = harness.run(&rows, dtype);
                        assert!(first.iter().any(|&bits| bits != 0));
                        assert!(
                            harness.run(&rows, dtype) == first,
                            "{ggml_type:?} N={n} K={k} {dtype:?} M={m}: second run differs"
                        );
                    }
                }
            }
        }
    }

    /// A partial last tile writes exactly M rows: the output buffer holds M
    /// rows and nothing around it is touched.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn partial_tiles_write_exactly_m_rows() {
        const SENTINEL: u32 = 0x7fc0_dead;
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            for (n, k, _) in SPLIT_SHAPES {
                let mut harness = Harness::new(ggml_type, n, k);
                let guard = TILE_ROWS as usize * n;
                for dtype in [DType::BF16, DType::F32] {
                    for m in [1, 31, 33, 157] {
                        let rows: Vec<usize> = (0..m).collect();
                        let elements = m * n;
                        let mut parent = buffer(
                            &harness.device,
                            (elements + 2 * guard) * 4,
                            DType::F32,
                            vec![elements + 2 * guard],
                        );
                        parent.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
                        let output = parent.slice_view(guard as u64 * 4, elements);
                        let input = harness.input(&rows, dtype);
                        let mut encoder = harness.device.command_encoder().unwrap();
                        harness.encode(&mut encoder, &input, &output);
                        encoder.commit_and_wait().unwrap();
                        let bits = parent.as_slice::<u32>().unwrap();
                        let (before, rest) = bits.split_at(guard);
                        let (written, after) = rest.split_at(elements);
                        assert!(
                            before.iter().chain(after).all(|&bits| bits == SENTINEL),
                            "{ggml_type:?} N={n} K={k} {dtype:?} M={m}: wrote outside the M output rows"
                        );
                        assert!(
                            written == harness.run(&rows, dtype),
                            "{ggml_type:?} N={n} K={k} {dtype:?} M={m}: output rows differ"
                        );
                    }
                }
            }
        }
    }

    /// Relative RMSE and reference-normalized maximum error of the prepared
    /// route on `input` against stock on `stock_input`.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn stock_errors(
        harness: &mut Harness,
        stock_input: &MlxBuffer,
        input: &MlxBuffer,
    ) -> (f64, f32) {
        let stock = harness.run_stock(stock_input);
        let candidate = harness.run_input(input);
        let mut worst_absolute = 0.0_f32;
        let mut reference_max = 0.0_f32;
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for (&x, &bits) in stock.iter().zip(&candidate) {
            let error = (x - f32::from_bits(bits)).abs();
            worst_absolute = worst_absolute.max(error);
            reference_max = reference_max.max(x.abs());
            squared_error += f64::from(error) * f64::from(error);
            squared_reference += f64::from(x) * f64::from(x);
        }
        (
            (squared_error / squared_reference).sqrt(),
            worst_absolute / reference_max,
        )
    }

    /// Invariant: parity against stock at one tile and at prefill row counts,
    /// with and without K splits. Both routes multiply the same activation
    /// values: BF16-exact ones, which neither a half nor an F32 cast changes.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn parity(ggml_type: GgmlType, dtype: DType) {
        for (n, k) in [(128, 2048), (192, 512)] {
            let mut harness = Harness::new(ggml_type, n, k);
            for m in [32, 157, 720] {
                let rows: Vec<usize> = (0..m).collect();
                let exact = harness.bf16_exact_f32(&rows);
                let input = match dtype {
                    DType::BF16 => harness.input(&rows, DType::BF16),
                    _ => exact.clone(),
                };
                let (relative_rmse, normalized_max_error) =
                    stock_errors(&mut harness, &exact, &input);
                assert!(
                    relative_rmse < 3e-4 && normalized_max_error < 5e-4,
                    "{ggml_type:?} N={n} K={k} {dtype:?} M={m}: normalized max {normalized_max_error}, relative RMSE {relative_rmse}"
                );
            }
        }
    }

    /// The F32 entry recasts every activation to half. Stock does the same at
    /// 32 rows but multiplies F32 activations as they are at other row counts.
    /// Arbitrary F32 values therefore match stock within the parity tolerance
    /// at 32 rows and within the recast, 11 significant bits, elsewhere.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn f32_recast_parity(ggml_type: GgmlType) {
        for (n, k) in [(128, 2048), (192, 512)] {
            let mut harness = Harness::new(ggml_type, n, k);
            for m in [32, 157, 720] {
                let rows: Vec<usize> = (0..m).collect();
                let input = harness.input(&rows, DType::F32);
                let (relative_rmse, normalized_max_error) =
                    stock_errors(&mut harness, &input, &input);
                let (rmse_bound, max_bound) = if m == 32 { (3e-4, 5e-4) } else { (1e-3, 1e-3) };
                assert!(
                    relative_rmse < rmse_bound && normalized_max_error < max_bound,
                    "{ggml_type:?} N={n} K={k} M={m}: normalized max {normalized_max_error}, relative RMSE {relative_rmse}"
                );
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q4k_f32_input_recast_stays_within_half_precision_of_stock() {
        f32_recast_parity(GgmlType::Q4_K);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q6k_f32_input_recast_stays_within_half_precision_of_stock() {
        f32_recast_parity(GgmlType::Q6_K);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q4k_matches_stock_tensor_mm() {
        parity(GgmlType::Q4_K, DType::F32);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q6k_matches_stock_tensor_mm() {
        parity(GgmlType::Q6_K, DType::F32);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q4k_bf16_input_matches_stock_tensor_mm() {
        parity(GgmlType::Q4_K, DType::BF16);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn q6k_bf16_input_matches_stock_tensor_mm() {
        parity(GgmlType::Q6_K, DType::BF16);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn dispatch_rejects_zero_rows() {
        let mut harness = Harness::new(GgmlType::Q4_K, 64, 256);
        let input = harness.input(&[0], DType::BF16);
        let output = harness.output(1);
        let params = harness.params(0);
        let mut encoder = harness.device.command_encoder().unwrap();
        let result = dispatch_repacked_gguf_mm_bf16_input(
            &mut encoder,
            &mut harness.registry,
            &harness.device,
            &input,
            &harness.prepared,
            &output,
            &params,
        );
        assert!(matches!(
            result,
            Err(MlxError::InvalidArgument(message)) if message.contains("M >= 1")
        ));
    }

    #[test]
    fn preparation_rejects_untiled_weight_shapes() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let weight = buffer(&device, 96 * 144, DType::U8, vec![96, 256]);
        let params = GgmlQuantizedMatmulParams {
            m: 32,
            n: 96,
            k: 256,
            ggml_type: GgmlType::Q4_K,
        };
        let mut encoder = device.command_encoder().unwrap();
        let result =
            prepare_repacked_gguf_weight(&mut encoder, &mut registry, &device, &weight, &params);
        assert!(matches!(
            result,
            Err(MlxError::InvalidArgument(message)) if message.contains("N % 64 == 0")
        ));
    }

    #[test]
    fn m5_max_split_policy_matches_staged_wave_model() {
        assert_eq!(target_splits(5_120, 17_408, GgmlType::Q4_K), 4);
        assert_eq!(target_splits(12_288, 5_120, GgmlType::Q4_K), 2);
        assert_eq!(target_splits(17_408, 5_120, GgmlType::Q4_K), 1);
        assert_eq!(target_splits(248_320, 5_120, GgmlType::Q6_K), 1);
    }

    /// The split count defines a weight's summation order, so a change here
    /// changes served output bits.
    #[test]
    fn split_counts_are_pinned_per_weight_shape() {
        for (n, k, splits) in SPLIT_SHAPES {
            for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
                assert_eq!(target_splits(n as u32, k as u32, ggml_type), splits);
            }
        }
        for (name, n, k, ggml_type, splits) in QWEN_SHAPES {
            assert_eq!(
                target_splits(n as u32, k as u32, ggml_type),
                splits,
                "{name}"
            );
        }
    }

    /// The split-grid schedule serves the few row tiles that would leave a
    /// serial pass short of threadgroups; a single split never needs it.
    #[test]
    fn split_grid_serves_few_row_tiles() {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            assert_eq!(split_grid_row_tiles(192, 512, ggml_type, 1), 1);
        }
        let row_tiles = QWEN_SHAPES.map(|(_, n, k, ggml_type, splits)| {
            split_grid_row_tiles(n as u32, k as u32, ggml_type, splits)
        });
        assert_eq!(row_tiles, [1, 2, 1, 5, 2, 1, 2, 2]);
    }

    /// Prepared BF16-input route against stock on the Qwen3.8-27B shapes.
    ///
    /// Serialized protocol: a sample is one command buffer of several
    /// dispatches with a memory barrier after each and a distinct output per
    /// dispatch; a figure is the median GPU time per dispatch after the
    /// warm-ups. The two routes alternate sample by sample, so both see the
    /// same GPU state. Run it alone, with nothing else on the GPU:
    /// `cargo test --release --lib repacked_gguf_mm::tests::bench -- --ignored --nocapture`
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    #[ignore = "GPU benchmark; run alone with --nocapture"]
    fn bench_prepared_against_stock_on_qwen_shapes() {
        const ROW_COUNTS: [usize; 8] = [5, 32, 64, 96, 160, 256, 720, 2048];
        const DISPATCHES: usize = 4;
        const WARMUPS: usize = 2;
        const SAMPLES: usize = 9;

        /// Median (stock, prepared) GPU milliseconds per dispatch at `m` rows.
        fn measure(harness: &mut Harness, m: usize) -> (f64, f64) {
            let rows: Vec<usize> = (0..m).collect();
            let prepared_input = harness.input(&rows, DType::BF16);
            let stock_input = harness.bf16_exact_f32(&rows);
            let outputs: Vec<MlxBuffer> = (0..DISPATCHES).map(|_| harness.output(m)).collect();
            let mut samples = [Vec::with_capacity(SAMPLES), Vec::with_capacity(SAMPLES)];
            for sample in 0..WARMUPS + SAMPLES {
                for (route, samples) in samples.iter_mut().enumerate() {
                    let mut encoder = harness.device.command_encoder().unwrap();
                    for output in &outputs {
                        if route == 0 {
                            harness.encode_stock(&mut encoder, &stock_input, output);
                        } else {
                            harness.encode(&mut encoder, &prepared_input, output);
                        }
                        encoder.memory_barrier();
                    }
                    let (start, end) = encoder.commit_wait_with_gpu_time().unwrap();
                    if sample >= WARMUPS {
                        samples.push((end - start) * 1e3 / DISPATCHES as f64);
                    }
                }
            }
            let [stock, prepared] = samples.map(|mut samples| {
                samples.sort_by(f64::total_cmp);
                samples[SAMPLES / 2]
            });
            (stock, prepared)
        }

        println!("| shape | N x K | type | M | stock ms | prepared ms | prepared/stock | split schedule |");
        println!("|---|---|---|---:|---:|---:|---:|---|");
        for (name, n, k, ggml_type, _) in QWEN_SHAPES {
            let mut harness = Harness::new(ggml_type, n, k);
            // Build the pipelines and page the weight in before the first figure.
            measure(&mut harness, 32);
            for m in ROW_COUNTS {
                let (stock, prepared) = measure(&mut harness, m);
                println!(
                    "| {name} | {n} x {k} | {ggml_type:?} | {m} | {stock:.3} | {prepared:.3} | {:.2} | {:?} |",
                    prepared / stock,
                    harness.prepared.split_schedule(m as u32)
                );
            }
        }
    }
}
