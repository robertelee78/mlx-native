//! Staged AF4G64 draft projections over BF16 activation rows.
//!
//! The weights are the prepared planes of
//! [`dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16`](super::qmm_affine::dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16);
//! only the tile differs. A weight is dequantized once to half, a step's
//! activation slice is multiplied by a 32-column half stage with `matmul2d`,
//! and K is cut into the runs the wave law below selects.
//!
//! Portions derived from Splash commit 031a25a6 (Apache-2.0): the tile, the
//! wave law of `runtime/ops/LinearGguf.cpp` and the one-pass gate/up pair.
//! See `src/shaders/dflash2_af4g64_staged.metal` for full attribution.
//!
//! Numerics contract: for a fixed phase and precision, an output's bits depend
//! on its activation row, its weight row and the run count. The run count is a
//! function of the weight shape alone, never of the row count, and rows are
//! covered by tiles of 32, 16 and 8 rows independently of tile height.
//! Context-prefill BF16 projections use reference32 splits and an unshifted K
//! walk; BF16 staged/decode projections use reference8 splits and tile spread.
//! Legacy/F32 projections and gate/up retain reference32 splits.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
use crate::ops::tensor_kernels::{require_tensor_kernel_family, TensorKernelFamily};
use crate::ops::ActivationPrecision;

pub const PROJECTION_KERNELS: [&str; 3] = [
    "dflash2_af4g64_staged_m8",
    "dflash2_af4g64_staged_m16",
    "dflash2_af4g64_staged_m32",
];
pub const GATE_UP_KERNELS: [&str; 3] = [
    "dflash2_af4g64_staged_gate_up_m8",
    "dflash2_af4g64_staged_gate_up_m16",
    "dflash2_af4g64_staged_gate_up_m32",
];
pub const REDUCE_KERNEL: &str = "dflash2_af4g64_staged_reduce";
pub const GATE_UP_REDUCE_KERNEL: &str = "dflash2_af4g64_staged_gate_up_reduce";
const BF16_PROJECTION_KERNELS: [&str; 3] = [
    "dflash2_af4g64_staged_m8_precision_bf16",
    "dflash2_af4g64_staged_m16_precision_bf16",
    "dflash2_af4g64_staged_m32_precision_bf16",
];
const BF16_GATE_UP_KERNELS: [&str; 3] = [
    "dflash2_af4g64_staged_gate_up_m8_precision_bf16",
    "dflash2_af4g64_staged_gate_up_m16_precision_bf16",
    "dflash2_af4g64_staged_gate_up_m32_precision_bf16",
];
const BF16_REDUCE_KERNEL: &str = "dflash2_af4g64_staged_reduce_precision_bf16";
const BF16_GATE_UP_REDUCE_KERNEL: &str = "dflash2_af4g64_staged_gate_up_reduce_precision_bf16";
const PREFILL_PROJECTION_KERNELS: [&str; 3] = [
    "dflash2_af4g64_prefill_m8",
    "dflash2_af4g64_prefill_m16",
    "dflash2_af4g64_prefill_m32",
];
const BF16_PREFILL_PROJECTION_KERNELS: [&str; 3] = [
    "dflash2_af4g64_prefill_m8_precision_bf16",
    "dflash2_af4g64_prefill_m16_precision_bf16",
    "dflash2_af4g64_prefill_m32_precision_bf16",
];
pub static SOURCE: &str = include_str!("../shaders/dflash2_af4g64_staged.metal");

pub fn register(registry: &mut KernelRegistry) {
    for name in PROJECTION_KERNELS
        .into_iter()
        .chain(GATE_UP_KERNELS)
        .chain([REDUCE_KERNEL, GATE_UP_REDUCE_KERNEL])
        .chain(PREFILL_PROJECTION_KERNELS)
        .chain(BF16_PREFILL_PROJECTION_KERNELS)
    {
        registry.register_source(name, SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StagedParams {
    k: u32,
    n: u32,
    splits: u32,
    spread: u32,
}

/// Prefill-only ABI: a tile may contain fewer real rows than its MMA height.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefillParams {
    k: u32,
    n: u32,
    splits: u32,
    spread: u32,
    rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    rows: u32,
    n: u32,
    splits: u32,
}

/// Threadgroups a core keeps resident in a full wave, the least a grid's only
/// wave costs, and the least a following wave costs (Splash
/// `LinearGguf.cpp:128-136`).
struct Occupancy {
    resident: u64,
    saturating: u64,
    following: u64,
}

const STAGED_TILE: Occupancy = Occupancy {
    resident: 10,
    saturating: 6,
    following: 2,
};
const GATE_UP_PAIR_TILE: Occupancy = Occupancy {
    resident: 16,
    saturating: 10,
    following: 3,
};

/// The law is evaluated for one reference device and one row tile so that the
/// run count, and with it every output bit, is the same on every host and at
/// every row count.
pub(super) const LAW_CORES: u64 = 40;
const LEGACY_LAW_ROWS: u64 = 32;
const BF16_DECODE_LAW_ROWS: u64 = 8;
/// AF4G64 stores 36 bytes per 64 weights: 4.5 bits, in sixteenths.
const WEIGHT_BITS_16: u64 = 72;

fn staged_time(groups: u64, steps: u64, splits: u32, tile: &Occupancy, reference_rows: u64) -> u64 {
    let per_core = groups.div_ceil(LAW_CORES);
    let waves = per_core.div_ceil(tile.resident);
    let full = (waves - 1) * tile.resident * LAW_CORES;
    let floor = if waves > 1 {
        tile.following
    } else {
        tile.saturating
    } * LAW_CORES;
    let step_bits = 64 * 32 * WEIGHT_BITS_16 / 16;
    let partial_bits = if splits > 1 {
        2 * reference_rows * 64 * 32
    } else {
        0
    };
    steps * step_bits * (full + (groups - full).max(floor)) + groups * partial_bits
}

fn law_splits(n: u32, k: u32, pair: bool, reference_rows: u64) -> u32 {
    let tile = if pair {
        &GATE_UP_PAIR_TILE
    } else {
        &STAGED_TILE
    };
    let mut best = 1;
    let mut least = u64::MAX;
    for splits in [1_u32, 2, 4, 8] {
        if (k / 32) % splits != 0 || (splits > 1 && k / splits < 512) {
            break;
        }
        let tiles = u64::from(n / 64) * u64::from(splits);
        let groups = if pair { 2 * tiles } else { tiles };
        let time = staged_time(groups, u64::from(k / 32 / splits), splits, tile, reference_rows);
        if time < least {
            least = time;
            best = splits;
        }
    }
    best
}

/// K runs of a legacy/F32 staged or any context-prefill `n x k` projection.
pub fn dflash2_af4g64_staged_splits(n: u32, k: u32) -> u32 {
    law_splits(n, k, false, LEGACY_LAW_ROWS)
}

/// K runs of a staged/decode projection with explicit logical output precision.
/// F32 retains reference32; BF16 uses the reference8 wave law. Prefill must use
/// [`dflash2_af4g64_staged_splits`] instead, including for BF16.
pub fn dflash2_af4g64_staged_splits_with_precision(
    n: u32,
    k: u32,
    precision: ActivationPrecision,
) -> u32 {
    match precision {
        ActivationPrecision::F32 => dflash2_af4g64_staged_splits(n, k),
        ActivationPrecision::Bf16 => law_splits(n, k, false, BF16_DECODE_LAW_ROWS),
    }
}

/// K runs of a one-pass staged gate/up pair of two `n x k` AF4G64 tensors.
pub fn dflash2_af4g64_staged_gate_up_splits(n: u32, k: u32) -> u32 {
    law_splits(n, k, true, LEGACY_LAW_ROWS)
}

/// F32 elements of partial scratch a legacy/F32 or context-prefill projection needs.
/// Prefill accepts any positive `m` and uses the same compact real-row count;
/// there is no padded-row scratch requirement.
pub fn dflash2_af4g64_staged_partial_elements(m: u32, n: u32, k: u32) -> usize {
    match dflash2_af4g64_staged_splits(n, k) {
        1 => 0,
        splits => splits as usize * m as usize * n as usize,
    }
}

/// F32 scratch elements for staged/decode with explicit output precision.
/// Context prefill uses [`dflash2_af4g64_staged_partial_elements`] instead.
pub fn dflash2_af4g64_staged_partial_elements_with_precision(
    m: u32,
    n: u32,
    k: u32,
    precision: ActivationPrecision,
) -> usize {
    match dflash2_af4g64_staged_splits_with_precision(n, k, precision) {
        1 => 0,
        splits => splits as usize * m as usize * n as usize,
    }
}

/// F32 elements of partial scratch an `m`-row gate/up pair needs.
pub fn dflash2_af4g64_staged_gate_up_partial_elements(m: u32, n: u32, k: u32) -> usize {
    match dflash2_af4g64_staged_gate_up_splits(n, k) {
        1 => 0,
        splits => splits as usize * m as usize * 2 * n as usize,
    }
}

/// Row tiles covering `m` rows: `(first row, kernel index, rows)`.
fn row_tiles(m: u32) -> impl Iterator<Item = (u32, usize, u32)> {
    projection_tiles(m, false)
}

/// Prefill rounds only the final MMA height up, never the logical row count.
/// A 17-row tail uses one 32-row MMA rather than extra 16- and 8-row dispatches.
fn projection_tiles(m: u32, prefill: bool) -> impl Iterator<Item = (u32, usize, u32)> {
    let mut first = 0;
    std::iter::from_fn(move || {
        let remaining = m - first;
        let kernel = match remaining {
            0 => return None,
            17.. if prefill => 2,
            9.. if prefill => 1,
            32.. => 2,
            16.. => 1,
            _ => 0,
        };
        let rows = remaining.min([8, 16, 32][kernel]);
        let tile = (first, kernel, rows);
        first += rows;
        Some(tile)
    })
}

fn bytes(elements: &[usize], dtype: DType) -> Result<usize> {
    elements
        .iter()
        .try_fold(dtype.size_of(), |total, count| total.checked_mul(*count))
        .ok_or_else(|| MlxError::InvalidArgument("staged AF4G64 size overflow".into()))
}

fn require(buffer: &MlxBuffer, dtype: DType, need: usize, label: &str) -> Result<()> {
    if buffer.dtype() != dtype || buffer.data_byte_len() < need {
        return Err(MlxError::InvalidArgument(format!(
            "staged AF4G64 {label} must be {dtype} with at least {need} bytes, got {} with {}",
            buffer.dtype(),
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

fn require_geometry(m: u32, n: u32, k: u32) -> Result<()> {
    if m == 0 || m % 8 != 0 || n == 0 || n % 64 != 0 || k == 0 || k % 64 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "staged AF4G64 requires M % 8 == 0, N % 64 == 0 and K % 64 == 0, got M={m} N={n} K={k}"
        )));
    }
    Ok(())
}

fn require_prefill_geometry(m: u32, n: u32, k: u32) -> Result<()> {
    if m == 0 || n == 0 || n % 64 != 0 || k == 0 || k % 64 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "prefill AF4G64 requires M > 0, N % 64 == 0 and K % 64 == 0, got M={m} N={n} K={k}"
        )));
    }
    Ok(())
}

fn require_weight(weights: &MlxBuffer, metadata: &MlxBuffer, n: u32, k: u32) -> Result<()> {
    let (n, k) = (n as usize, k as usize);
    require(
        weights,
        DType::U32,
        bytes(&[n, k / 8], DType::U32)?,
        "weights",
    )?;
    require(
        metadata,
        DType::U32,
        bytes(&[n, k / 64], DType::U32)?,
        "metadata",
    )
}

fn offset(buffer: &MlxBuffer, elements: &[usize]) -> Result<u64> {
    Ok(buffer.byte_offset() + bytes(elements, buffer.dtype())? as u64)
}

/// Where a row tile's runs start in the partial scratch: `[split][row][width]`
/// blocks in row-tile order. An unsplit projection binds the scratch unread.
fn partial_offset(partials: &MlxBuffer, splits: u32, first: usize, width: usize) -> Result<u64> {
    if splits == 1 {
        return Ok(partials.byte_offset());
    }
    offset(partials, &[splits as usize, first, width])
}

/// Encode `output[m, n] = input[m, k] x weight[n, k]^T`.
///
/// `input` is BF16, `output` F32. `partials` is caller-owned F32 scratch of at
/// least [`dflash2_af4g64_staged_partial_elements`] elements; it is unread
/// when the projection runs unsplit. The caller places the barriers around
/// the operation; the one between a split projection and its reduce is
/// encoded here.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_af4g64_staged(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MlxBuffer,
    metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    dispatch_dflash2_af4g64_staged_with_precision(
        encoder, registry, device, input, weights, metadata, partials, output, m, n, k,
        ActivationPrecision::F32,
    )
}

/// Staged projection with explicit logical output precision.
///
/// `Bf16` rounds only the complete F32 reduction to nearest-even BF16, then
/// widens exactly into F32 output storage. Split partials remain F32.
/// Geometry and barriers match [`dispatch_dflash2_af4g64_staged`]. Scratch needs
/// [`dflash2_af4g64_staged_partial_elements_with_precision`] elements for the same
/// precision. BF16 uses reference8 splits; F32 retains reference32. Both modes
/// use the tile-spread K walk.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_af4g64_staged_with_precision(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MlxBuffer,
    metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    precision: ActivationPrecision,
) -> Result<()> {
    dispatch_projection(
        encoder, registry, device, input, weights, metadata, partials, output, m, n, k,
        precision, false,
    )
}

/// Context-prefill projection with explicit logical output precision.
///
/// BF16 context prefill walks each K run from its beginning (`spread = 0`);
/// [`dispatch_dflash2_af4g64_staged_with_precision`] retains the staged/decode
/// tile-spread walk. `F32` preserves the legacy staged walk and output bits.
/// Input remains BF16; output and split partial storage remain F32. Any positive
/// real `m` is supported without padding input/output: the final MMA tile masks
/// absent input rows and stores only real rows. Shape-only splits, rounding and
/// barriers retain the reference32 profile. Scratch needs
/// [`dflash2_af4g64_staged_partial_elements`] with the real `m`: tile blocks are
/// `[split][real tile row][n]`, with no space reserved for masked rows.
///
/// Scratch must not overlap the input, weights, metadata or output during this
/// operation. Reusing another phase's storage requires the caller to order its
/// last access before prefill and its next access after prefill completes.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_af4g64_prefill_with_precision(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MlxBuffer,
    metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    precision: ActivationPrecision,
) -> Result<()> {
    dispatch_projection(
        encoder, registry, device, input, weights, metadata, partials, output, m, n, k,
        precision, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_projection(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MlxBuffer,
    metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    precision: ActivationPrecision,
    prefill: bool,
) -> Result<()> {
    if prefill {
        require_prefill_geometry(m, n, k)?;
    } else {
        require_geometry(m, n, k)?;
    }
    let splits = if prefill {
        dflash2_af4g64_staged_splits(n, k)
    } else {
        dflash2_af4g64_staged_splits_with_precision(n, k, precision)
    };
    let (rows, columns, inputs) = (m as usize, n as usize, k as usize);
    require(
        input,
        DType::BF16,
        bytes(&[rows, inputs], DType::BF16)?,
        "input",
    )?;
    require(
        output,
        DType::F32,
        bytes(&[rows, columns], DType::F32)?,
        "output",
    )?;
    require_weight(weights, metadata, n, k)?;
    require(
        partials,
        DType::F32,
        bytes(&[if splits == 1 { 0 } else { splits as usize }, rows, columns], DType::F32)?,
        "partials",
    )?;
    require_tensor_kernel_family(registry, device, TensorKernelFamily::Dflash2Af4g64Staged)?;

    let params = StagedParams {
        k,
        n,
        splits,
        spread: if prefill && precision == ActivationPrecision::Bf16 { 0 } else { 1 },
    };
    let (kernels, masked_kernels, reduce_kernel) = match precision {
        ActivationPrecision::F32 =>
            (&PROJECTION_KERNELS, &PREFILL_PROJECTION_KERNELS, REDUCE_KERNEL),
        ActivationPrecision::Bf16 =>
            (&BF16_PROJECTION_KERNELS, &BF16_PREFILL_PROJECTION_KERNELS, BF16_REDUCE_KERNEL),
    };
    if precision == ActivationPrecision::Bf16 {
        // The captured pipeline retains the specialized rounding boundary.
        encoder.set_op_kind(crate::encoder::CapturedOpKind::Other);
    }
    for (first, kernel, tile_rows) in projection_tiles(m, prefill) {
        let masked = prefill && tile_rows != [8, 16, 32][kernel];
        let prefill_params = PrefillParams {
            k,
            n,
            splits,
            spread: params.spread,
            rows: tile_rows,
        };
        let params_bytes = if masked { as_bytes(&prefill_params) } else { as_bytes(&params) };
        let first = first as usize;
        let pipeline = registry.get_pipeline(
            if masked { masked_kernels[kernel] } else { kernels[kernel] },
            device.metal_device(),
        )?;
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Bytes(params_bytes)),
                (1, KernelArg::Buffer(weights)),
                (2, KernelArg::Buffer(metadata)),
                (
                    3,
                    KernelArg::BufferWithOffset(input, offset(input, &[first, inputs])?),
                ),
                (
                    4,
                    KernelArg::BufferWithOffset(output, offset(output, &[first, columns])?),
                ),
                (
                    5,
                    KernelArg::BufferWithOffset(
                        partials,
                        partial_offset(partials, splits, first, columns)?,
                    ),
                ),
            ],
            &[(0, 8_192)],
            MTLSize::new(u64::from(splits), u64::from(n / 64), 1),
            MTLSize::new(64, 1, 1),
        );
    }
    if splits == 1 {
        return Ok(());
    }
    encoder.memory_barrier();
    for (first, _, tile_rows) in projection_tiles(m, prefill) {
        let first = first as usize;
        let reduce = registry.get_pipeline(reduce_kernel, device.metal_device())?;
        let dims = ReduceParams {
            rows: tile_rows,
            n,
            splits,
        };
        encode_with_args(
            encoder,
            reduce,
            &[
                (
                    0,
                    KernelArg::BufferWithOffset(
                        partials,
                        partial_offset(partials, splits, first, columns)?,
                    ),
                ),
                (
                    1,
                    KernelArg::BufferWithOffset(output, offset(output, &[first, columns])?),
                ),
                (2, KernelArg::Bytes(as_bytes(&dims))),
            ],
            MTLSize::new(u64::from(tile_rows) * u64::from(n), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    Ok(())
}

/// Encode `output[m, n] = BF16(SiLU(input x gate^T) * (input x up^T))` in one
/// pass over both `n x k` tensors.
///
/// `partials` needs at least
/// [`dflash2_af4g64_staged_gate_up_partial_elements`] F32 elements.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_af4g64_staged_gate_up(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gate_weights: &MlxBuffer,
    gate_metadata: &MlxBuffer,
    up_weights: &MlxBuffer,
    up_metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    dispatch_dflash2_af4g64_staged_gate_up_with_precision(
        encoder, registry, device, input, gate_weights, gate_metadata, up_weights, up_metadata,
        partials, output, m, n, k, ActivationPrecision::F32,
    )
}

/// Fused staged gate/up with explicit logical producer precision.
///
/// Both modes accumulate in F32 and round the activation at its BF16 store.
/// `Bf16` additionally rounds each completed gate and up sum to nearest-even
/// BF16 before SiLU(gate) * up, after all split partials have been reduced.
/// `F32` preserves unrounded gate/up sums. Scratch and barriers match the F32 API.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_af4g64_staged_gate_up_with_precision(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gate_weights: &MlxBuffer,
    gate_metadata: &MlxBuffer,
    up_weights: &MlxBuffer,
    up_metadata: &MlxBuffer,
    partials: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    precision: ActivationPrecision,
) -> Result<()> {
    require_geometry(m, n, k)?;
    let splits = dflash2_af4g64_staged_gate_up_splits(n, k);
    let (rows, columns, inputs) = (m as usize, n as usize, k as usize);
    require(
        input,
        DType::BF16,
        bytes(&[rows, inputs], DType::BF16)?,
        "input",
    )?;
    require(
        output,
        DType::BF16,
        bytes(&[rows, columns], DType::BF16)?,
        "gated output",
    )?;
    require_weight(gate_weights, gate_metadata, n, k)?;
    require_weight(up_weights, up_metadata, n, k)?;
    require(
        partials,
        DType::F32,
        bytes(
            &[dflash2_af4g64_staged_gate_up_partial_elements(m, n, k)],
            DType::F32,
        )?,
        "partials",
    )?;
    require_tensor_kernel_family(registry, device, TensorKernelFamily::Dflash2Af4g64Staged)?;

    let params = StagedParams {
        k,
        n,
        splits,
        spread: 1,
    };
    let (kernels, reduce_kernel) = match precision {
        ActivationPrecision::F32 => (&GATE_UP_KERNELS, GATE_UP_REDUCE_KERNEL),
        ActivationPrecision::Bf16 => (&BF16_GATE_UP_KERNELS, BF16_GATE_UP_REDUCE_KERNEL),
    };
    if precision == ActivationPrecision::Bf16 {
        // Do not expose rounded producers as a generic F32 fusion candidate.
        encoder.set_op_kind(crate::encoder::CapturedOpKind::Other);
    }
    for (first, kernel, _) in row_tiles(m) {
        let first = first as usize;
        let pipeline = registry.get_pipeline(kernels[kernel], device.metal_device())?;
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&params))),
                (1, KernelArg::Buffer(gate_weights)),
                (2, KernelArg::Buffer(gate_metadata)),
                (3, KernelArg::Buffer(up_weights)),
                (4, KernelArg::Buffer(up_metadata)),
                (
                    5,
                    KernelArg::BufferWithOffset(input, offset(input, &[first, inputs])?),
                ),
                (
                    6,
                    KernelArg::BufferWithOffset(output, offset(output, &[first, columns])?),
                ),
                (
                    7,
                    KernelArg::BufferWithOffset(
                        partials,
                        partial_offset(partials, splits, first, 2 * columns)?,
                    ),
                ),
            ],
            &[(0, 4_096)],
            MTLSize::new(u64::from(splits), u64::from(n / 32), 1),
            MTLSize::new(32, 1, 1),
        );
    }
    if splits == 1 {
        return Ok(());
    }
    encoder.memory_barrier();
    for (first, _, tile_rows) in row_tiles(m) {
        let first = first as usize;
        let reduce = registry.get_pipeline(reduce_kernel, device.metal_device())?;
        let dims = ReduceParams {
            rows: tile_rows,
            n,
            splits,
        };
        encode_with_args(
            encoder,
            reduce,
            &[
                (
                    0,
                    KernelArg::BufferWithOffset(
                        partials,
                        partial_offset(partials, splits, first, 2 * columns)?,
                    ),
                ),
                (
                    1,
                    KernelArg::BufferWithOffset(output, offset(output, &[first, columns])?),
                ),
                (2, KernelArg::Bytes(as_bytes(&dims))),
            ],
            MTLSize::new(u64::from(tile_rows) * u64::from(n), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn split_law_matches_the_staged_wave_model() {
        // Qwen3.8-27B draft: dynamic, q|k|v, attention out, down, context, k|v.
        assert_eq!(dflash2_af4g64_staged_splits(1_280, 5_120), 8);
        assert_eq!(dflash2_af4g64_staged_splits(6_144, 5_120), 4);
        assert_eq!(dflash2_af4g64_staged_splits(5_120, 4_096), 4);
        assert_eq!(dflash2_af4g64_staged_splits(5_120, 17_408), 4);
        assert_eq!(dflash2_af4g64_staged_splits(5_120, 25_600), 4);
        assert_eq!(dflash2_af4g64_staged_splits(2_048, 5_120), 8);
        assert_eq!(dflash2_af4g64_staged_gate_up_splits(17_408, 5_120), 1);
        // Qwen3.6-35B draft pair; partitions keep at least 512 inputs.
        assert_eq!(dflash2_af4g64_staged_gate_up_splits(6_144, 2_048), 2);
        assert_eq!(dflash2_af4g64_staged_splits(64, 256), 1);
    }


    #[test]
    fn row_tiles_cover_every_multiple_of_eight() {
        let tiles = |m| row_tiles(m).collect::<Vec<_>>();
        assert_eq!(tiles(8), [(0, 0, 8)]);
        assert_eq!(tiles(16), [(0, 1, 16)]);
        assert_eq!(tiles(24), [(0, 1, 16), (16, 0, 8)]);
        assert_eq!(tiles(32), [(0, 2, 32)]);
        assert_eq!(tiles(72), [(0, 2, 32), (32, 2, 32), (64, 0, 8)]);
    }

    #[test]
    fn geometry_is_validated_before_encoding() {
        assert!(require_geometry(12, 64, 64).is_err());
        assert!(require_geometry(8, 96, 64).is_err());
        assert!(require_geometry(8, 64, 96).is_err());
        assert!(require_geometry(24, 128, 128).is_ok());
    }


    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn prefill_real_row_tails_match_padded_rows_and_preserve_guards() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES {
            let weights = weight(&device, n, k, 403);
            for m in [1_usize, 3, 7, 9, 15, 17, 31, 33, 49] {
                let x = activations(m, k, 809);
                // No padded rows are available through this input handle.
                let input = upload(&device, &x, vec![m, k]);
                let padded_m = m.div_ceil(32) * 32;
                let mut padded_x = x.clone();
                padded_x.resize(padded_m * k, bf16::ZERO);
                let padded_input = upload(&device, &padded_x, vec![padded_m, k]);
                let elements =
                    dflash2_af4g64_staged_partial_elements(m as u32, n as u32, k as u32);
                let padded_elements =
                    dflash2_af4g64_staged_partial_elements(padded_m as u32, n as u32, k as u32);
                for precision in [ActivationPrecision::F32, ActivationPrecision::Bf16] {
                    let output_root = scratch(&device, (m + 2) * n);
                    let output = output_root.slice_view((n * 4) as u64, m * n);
                    let partial_root = scratch(&device, elements + 2 * n);
                    let partials = partial_root.slice_view((n * 4) as u64, elements);
                    let padded_output = scratch(&device, padded_m * n);
                    let padded_partials = scratch(&device, padded_elements);
                    let mut encoder = device.command_encoder().unwrap();
                    dispatch_dflash2_af4g64_prefill_with_precision(
                        &mut encoder, &mut registry, &device, &input, &weights.planes,
                        &weights.metadata, &partials, &output,
                        m as u32, n as u32, k as u32, precision,
                    ).unwrap();
                    encoder.memory_barrier();
                    // F32 must retain the legacy arithmetic, even for new tails.
                    let reference_dispatch = if precision == ActivationPrecision::F32 {
                        dispatch_dflash2_af4g64_staged_with_precision
                    } else {
                        dispatch_dflash2_af4g64_prefill_with_precision
                    };
                    reference_dispatch(
                        &mut encoder, &mut registry, &device, &padded_input, &weights.planes,
                        &weights.metadata, &padded_partials, &padded_output,
                        padded_m as u32, n as u32, k as u32, precision,
                    ).unwrap();
                    encoder.commit_and_wait().unwrap();
                    let actual = output.as_slice::<f32>().unwrap();
                    let expected = &padded_output.as_slice::<f32>().unwrap()[..m * n];
                    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                        assert!(actual.is_finite());
                        assert_eq!(
                            actual.to_bits(), expected.to_bits(),
                            "{n}x{k} real M={m}, index={index}, {precision:?}"
                        );
                    }
                    let output_bytes = output_root.as_slice::<f32>().unwrap();
                    assert!(output_bytes[..n].iter().all(|v| v.is_nan()));
                    assert!(output_bytes[(m + 1) * n..].iter().all(|v| v.is_nan()));
                    let partial_bytes = partial_root.as_slice::<f32>().unwrap();
                    assert!(partial_bytes[..n].iter().all(|v| v.is_nan()));
                    assert!(partial_bytes[n + elements..].iter().all(|v| v.is_nan()));
                    assert!(partial_bytes[n..n + elements].iter().all(|v| v.is_finite()));
                }
            }
        }
    }

    struct Weight {
        codes: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        planes: MlxBuffer,
        metadata: MlxBuffer,
    }

    fn random(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 40) as f32) / ((1_u64 << 24) as f32) - 0.5
    }

    fn weight(device: &MlxDevice, n: usize, k: usize, seed: u64) -> Weight {
        let mut state = seed;
        let groups64 = k / 64;
        let groups32 = k / 32;
        let codes: Vec<u8> = (0..n * k)
            .map(|_| ((random(&mut state) + 0.5) * 16.0) as u8 & 15)
            .collect();
        let scale_bits: Vec<u16> = (0..n * groups64)
            .map(|_| bf16::from_f32(0.004 + 0.01 * (random(&mut state) + 0.5)).to_bits())
            .collect();
        let bias_bits: Vec<u16> = (0..n * groups64)
            .map(|_| bf16::from_f32(-0.08 + 0.05 * random(&mut state)).to_bits())
            .collect();
        let mut plane = vec![0_u32; n * groups32 * 4];
        let mut packed_metadata = vec![0_u32; n * groups64];
        for row in 0..n {
            for group64 in 0..groups64 {
                let source = row * groups64 + group64;
                packed_metadata[(row / 64 * groups64 + group64) * 64 + row % 64] =
                    u32::from(scale_bits[source]) | (u32::from(bias_bits[source]) << 16);
            }
            for group32 in 0..groups32 {
                for chunk in 0..4 {
                    let mut word = 0_u32;
                    for pair in 0..4 {
                        let first = group32 * 32 + 16 * (pair >> 1) + 4 * chunk + 2 * (pair & 1);
                        word |= u32::from(codes[row * k + first]) << (4 * pair);
                        word |= u32::from(codes[row * k + first + 1]) << (16 + 4 * pair);
                    }
                    plane[((row / 64 * groups32 + group32) * 64 + row % 64) * 4 + chunk] = word;
                }
            }
        }
        let mut planes = device
            .alloc_buffer(plane.len() * 4, DType::U32, vec![n / 64, groups32, 64, 4])
            .unwrap();
        planes
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&plane);
        let mut metadata = device
            .alloc_buffer(
                packed_metadata.len() * 4,
                DType::U32,
                vec![n / 64, groups64, 64],
            )
            .unwrap();
        metadata
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&packed_metadata);
        Weight {
            codes,
            scales: scale_bits
                .iter()
                .map(|bits| bf16::from_bits(*bits).to_f32())
                .collect(),
            biases: bias_bits
                .iter()
                .map(|bits| bf16::from_bits(*bits).to_f32())
                .collect(),
            planes,
            metadata,
        }
    }

    fn activations(rows: usize, k: usize, seed: u64) -> Vec<bf16> {
        let mut state = seed;
        (0..rows * k)
            .map(|_| bf16::from_f32(3.0 * random(&mut state)))
            .collect()
    }

    fn upload(device: &MlxDevice, values: &[bf16], shape: Vec<usize>) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(values.len() * 2, DType::BF16, shape)
            .unwrap();
        buffer
            .as_mut_slice::<bf16>()
            .unwrap()
            .copy_from_slice(values);
        buffer
    }

    /// F32 scratch that holds NaN, as a reused workspace may: a result that
    /// reads an element its command did not write is not finite.
    fn scratch(device: &MlxDevice, elements: usize) -> MlxBuffer {
        let elements = elements.max(1);
        let mut buffer = device
            .alloc_buffer(elements * 4, DType::F32, vec![elements])
            .unwrap();
        buffer.as_mut_slice::<f32>().unwrap().fill(f32::NAN);
        buffer
    }

    /// Every column of a dense activation row.
    fn dense(x: &[bf16]) -> impl Iterator<Item = (usize, f32)> + Clone + '_ {
        x.iter().map(|value| value.to_f32()).enumerate()
    }

    /// F64 dot product of activation entries `(column, value)` with one exact
    /// affine weight row, and the bound on a staged result's distance from it:
    /// every weight is rounded once to F32 and once to half (relative 2^-24
    /// and 2^-11, or 2^-25 absolute below the smallest normal half), and the
    /// exact half-by-BF16 products of the entries are added in F32. The
    /// factor 1.001 covers the products of those relative terms.
    fn reference(
        weight: &Weight,
        entries: impl Iterator<Item = (usize, f32)>,
        row: usize,
        k: usize,
    ) -> (f64, f64) {
        let mut sum = 0.0_f64;
        let mut magnitude = 0.0_f64;
        let mut activation_magnitude = 0.0_f64;
        let mut terms = 0.0_f64;
        for (column, activation) in entries {
            let group = row * (k / 64) + column / 64;
            let value = f64::from(weight.scales[group]) * f64::from(weight.codes[row * k + column])
                + f64::from(weight.biases[group]);
            let activation = f64::from(activation);
            sum += value * activation;
            magnitude += (value * activation).abs();
            activation_magnitude += activation.abs();
            terms += 1.0;
        }
        let bound = (2.0_f64.powi(-11) + (terms + 2.0) * 2.0_f64.powi(-24)) * 1.001 * magnitude
            + 2.0_f64.powi(-25) * activation_magnitude;
        (sum, bound)
    }

    fn project(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        weight: &Weight,
        x: &[bf16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let input = upload(device, x, vec![m, k]);
        let output = scratch(device, m * n);
        let partials = scratch(
            device,
            dflash2_af4g64_staged_partial_elements(m as u32, n as u32, k as u32),
        );
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_af4g64_staged(
            &mut encoder,
            registry,
            device,
            &input,
            &weight.planes,
            &weight.metadata,
            &partials,
            &output,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let result = output.as_slice::<f32>().unwrap()[..m * n].to_vec();
        assert!(
            result.iter().all(|value| value.is_finite()),
            "{n}x{k} M={m}: a result read scratch its command did not write"
        );
        result
    }

    fn gate_up(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        gate: &Weight,
        up: &Weight,
        x: &[bf16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<bf16> {
        let input = upload(device, x, vec![m, k]);
        let mut output = device
            .alloc_buffer(m * n * 2, DType::BF16, vec![m, n])
            .unwrap();
        output.as_mut_slice::<bf16>().unwrap().fill(bf16::NAN);
        let partials = scratch(
            device,
            dflash2_af4g64_staged_gate_up_partial_elements(m as u32, n as u32, k as u32),
        );
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_af4g64_staged_gate_up(
            &mut encoder,
            registry,
            device,
            &input,
            &gate.planes,
            &gate.metadata,
            &up.planes,
            &up.metadata,
            &partials,
            &output,
            m as u32,
            n as u32,
            k as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let result = output.as_slice::<bf16>().unwrap().to_vec();
        assert!(
            result.iter().all(|value| value.is_finite()),
            "{n}x{k} M={m}: a gated result read scratch its command did not write"
        );
        result
    }

    /// Unsplit and split shapes: `(n, k)`.
    const SHAPES: [(usize, usize); 2] = [(192, 256), (128, 2_048)];

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn projection_is_within_the_k_scaled_bound_of_the_f64_reference() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES {
            let weight = weight(&device, n, k, 11);
            let x = activations(32, k, 23);
            let actual = project(&device, &mut registry, &weight, &x, 32, n, k);
            let mut worst = 0.0_f64;
            for row in 0..32 {
                for column in 0..n {
                    let (expected, bound) =
                        reference(&weight, dense(&x[row * k..(row + 1) * k]), column, k);
                    let error = (f64::from(actual[row * n + column]) - expected).abs();
                    assert!(
                        error <= bound,
                        "{n}x{k} row {row} column {column}: error {error} exceeds {bound}"
                    );
                    worst = worst.max(error / bound);
                }
            }
            assert!(worst > 0.0, "{n}x{k} reference comparison is vacuous");
        }
        assert_eq!(dflash2_af4g64_staged_splits(192, 256), 1);
        assert!(dflash2_af4g64_staged_splits(128, 2_048) > 1);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn projection_rows_are_deterministic_and_independent_of_their_cohort() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES {
            let weight = weight(&device, n, k, 31);
            let x = activations(32, k, 41);
            let base = project(&device, &mut registry, &weight, &x, 32, n, k);
            assert_eq!(
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                project(&device, &mut registry, &weight, &x, 32, n, k)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{n}x{k} repeat differs"
            );
            // Every row count and tile height, each row at another position
            // beside other rows.
            for m in [8_usize, 16, 24, 32] {
                let mut moved = activations(m, k, 53);
                let kept: Vec<usize> = (0..m).filter(|row| row % 3 != 1).collect();
                for &row in &kept {
                    let source = 31 - row;
                    moved[row * k..(row + 1) * k].copy_from_slice(&x[source * k..(source + 1) * k]);
                }
                let actual = project(&device, &mut registry, &weight, &moved, m, n, k);
                for &row in &kept {
                    let source = 31 - row;
                    assert_eq!(
                        actual[row * n..(row + 1) * n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        base[source * n..(source + 1) * n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "{n}x{k} M={m} row {row} depends on its cohort"
                    );
                }
            }
        }
    }

    fn silu(gate: f64) -> f64 {
        gate / (1.0 + (-gate).exp())
    }

    /// F64 `SiLU(gate) * up` of activation entries and the bound on a staged
    /// pair's distance from it: |SiLU'| <= 1.1, the product and SiLU are
    /// evaluated in F32 and the result is rounded once to BF16.
    fn gated_reference(
        gate: &Weight,
        up: &Weight,
        entries: impl Iterator<Item = (usize, f32)> + Clone,
        row: usize,
        k: usize,
    ) -> (f64, f64) {
        let (g, g_bound) = reference(gate, entries.clone(), row, k);
        let (u, u_bound) = reference(up, entries, row, k);
        let expected = silu(g) * u;
        let bound = 1.1 * g_bound * (u.abs() + u_bound)
            + silu(g).abs() * u_bound
            + expected.abs() * (2.0_f64.powi(-8) + 2.0_f64.powi(-20))
            + 2.0_f64.powi(-100);
        (expected, bound)
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn gate_up_is_within_the_k_scaled_bound_of_the_f64_reference() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES {
            let gate = weight(&device, n, k, 61);
            let up = weight(&device, n, k, 67);
            let x = activations(32, k, 71);
            let actual = gate_up(&device, &mut registry, &gate, &up, &x, 32, n, k);
            let mut worst = 0.0_f64;
            for row in 0..32 {
                let x = &x[row * k..(row + 1) * k];
                for column in 0..n {
                    let (expected, bound) = gated_reference(&gate, &up, dense(x), column, k);
                    let error = (f64::from(actual[row * n + column].to_f32()) - expected).abs();
                    assert!(
                        error <= bound,
                        "{n}x{k} row {row} column {column}: error {error} exceeds {bound}"
                    );
                    worst = worst.max(error / bound);
                }
            }
            assert!(
                worst > 0.0,
                "{n}x{k} gate/up reference comparison is vacuous"
            );
        }
        assert_eq!(dflash2_af4g64_staged_gate_up_splits(192, 256), 1);
        assert!(dflash2_af4g64_staged_gate_up_splits(128, 2_048) > 1);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn gate_up_rows_are_deterministic_and_independent_of_their_cohort() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (n, k) in SHAPES {
            let gate = weight(&device, n, k, 73);
            let up = weight(&device, n, k, 79);
            let x = activations(32, k, 83);
            let base = gate_up(&device, &mut registry, &gate, &up, &x, 32, n, k);
            assert_eq!(
                bits(&base),
                bits(&gate_up(&device, &mut registry, &gate, &up, &x, 32, n, k)),
                "{n}x{k} repeat differs"
            );
            for m in [8_usize, 16, 24, 32] {
                let mut moved = activations(m, k, 89);
                let kept: Vec<usize> = (0..m).filter(|row| row % 3 != 1).collect();
                for &row in &kept {
                    let source = 31 - row;
                    moved[row * k..(row + 1) * k].copy_from_slice(&x[source * k..(source + 1) * k]);
                }
                let actual = gate_up(&device, &mut registry, &gate, &up, &moved, m, n, k);
                for &row in &kept {
                    let source = 31 - row;
                    assert_eq!(
                        bits(&actual[row * n..(row + 1) * n]),
                        bits(&base[source * n..(source + 1) * n]),
                        "{n}x{k} M={m} row {row} depends on its cohort"
                    );
                }
            }
        }
    }

    /// `(n, k)` of every staged projection of the Qwen3.8-27B and
    /// Qwen3.6-35B drafts: dynamic, q|k|v, attention out, down, context and
    /// context k|v (the Qwen3.6 dynamic projection keeps the register tile).
    const SERVED_PROJECTIONS: [(usize, usize); 11] = [
        (1_280, 5_120),
        (6_144, 5_120),
        (5_120, 4_096),
        (5_120, 17_408),
        (5_120, 25_600),
        (2_048, 5_120),
        (6_144, 2_048),
        (2_048, 4_096),
        (2_048, 6_144),
        (2_048, 16_384),
        (2_048, 2_048),
    ];
    const SERVED_GATE_UP: [(usize, usize); 2] = [(17_408, 5_120), (6_144, 2_048)];

    /// 32 activation rows that together probe every 32-input step of K: row
    /// `r` owns steps `r, r + 32, ...` and is nonzero at each one's first
    /// input, its last, and one between. An output is then a sum of
    /// `3 K / 1024` products, so a step that is skipped, repeated or read
    /// from another place moves it by far more than the bound. The K-scaled
    /// bound of dense rows cannot show that at K = 17,408 or 25,600.
    fn step_probes(k: usize) -> Vec<Vec<(usize, f32)>> {
        const VALUES: [f32; 7] = [1.0, -0.75, 0.5, 1.5, -1.25, 2.0, -0.375];
        (0..32_usize)
            .map(|row| {
                (row..k / 32)
                    .step_by(32)
                    .flat_map(|step| {
                        [0, 1 + (step * 7 + row * 13) % 30, 31]
                            .into_iter()
                            .enumerate()
                            .map(move |(index, offset)| {
                                (step * 32 + offset, VALUES[(step + row + index) % 7])
                            })
                    })
                    .collect()
            })
            .collect()
    }

    fn dense_rows(probes: &[Vec<(usize, f32)>], k: usize) -> Vec<bf16> {
        let mut x = vec![bf16::from_bits(0); probes.len() * k];
        for (row, entries) in probes.iter().enumerate() {
            for (column, value) in entries {
                x[row * k + column] = bf16::from_f32(*value);
            }
        }
        x
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn served_projections_sum_every_input_step_once() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SERVED_PROJECTIONS {
            let weight = weight(&device, n, k, 3_000 + (n + k) as u64);
            let probes = step_probes(k);
            assert_eq!(probes.iter().map(Vec::len).sum::<usize>(), 3 * k / 32);
            let x = dense_rows(&probes, k);
            let actual = project(&device, &mut registry, &weight, &x, 32, n, k);
            for (row, entries) in probes.iter().enumerate() {
                for column in 0..n {
                    let (expected, bound) = reference(&weight, entries.iter().copied(), column, k);
                    let error = (f64::from(actual[row * n + column]) - expected).abs();
                    assert!(
                        error <= bound,
                        "{n}x{k} row {row} column {column}: error {error} exceeds {bound}"
                    );
                }
            }
            // The same rows through the 8- and 16-row tiles.
            for m in [8_usize, 16, 24] {
                let part = project(&device, &mut registry, &weight, &x[..m * k], m, n, k);
                assert!(
                    part.iter()
                        .map(|v| v.to_bits())
                        .eq(actual[..m * n].iter().map(|v| v.to_bits())),
                    "{n}x{k} M={m} differs from the 32-row tile"
                );
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn served_gate_up_pairs_sum_every_input_step_once() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SERVED_GATE_UP {
            let gate = weight(&device, n, k, 4_000 + n as u64);
            let up = weight(&device, n, k, 5_000 + n as u64);
            let probes = step_probes(k);
            let x = dense_rows(&probes, k);
            let actual = gate_up(&device, &mut registry, &gate, &up, &x, 32, n, k);
            for (row, entries) in probes.iter().enumerate() {
                for column in 0..n {
                    let (expected, bound) =
                        gated_reference(&gate, &up, entries.iter().copied(), column, k);
                    let error = (f64::from(actual[row * n + column].to_f32()) - expected).abs();
                    assert!(
                        error <= bound,
                        "{n}x{k} row {row} column {column}: error {error} exceeds {bound}"
                    );
                }
            }
            for m in [8_usize, 16, 24] {
                let part = gate_up(&device, &mut registry, &gate, &up, &x[..m * k], m, n, k);
                assert!(
                    part.iter()
                        .map(|v| v.to_bits())
                        .eq(actual[..m * n].iter().map(|v| v.to_bits())),
                    "{n}x{k} M={m} differs from the 32-row tile"
                );
            }
        }
    }

    /// Unit weights at input 0; optional unit weights at input 1 and K/2.
    /// These exact binary probes distinguish final rounding from rounding runs.
    fn precision_weights(
        device: &MlxDevice,
        n: usize,
        k: usize,
        fractional: impl Fn(usize) -> bool,
    ) -> (MlxBuffer, MlxBuffer) {
        let mut planes = device
            .alloc_buffer(n * k / 8 * 4, DType::U32, vec![n / 64, k / 32, 64, 4])
            .unwrap();
        let words = planes.as_mut_slice::<u32>().unwrap();
        for column in 0..n {
            let at = |step| ((column / 64 * (k / 32) + step) * 64 + column % 64) * 4;
            words[at(0)] = 1 | if fractional(column) { 1 << 16 } else { 0 };
            if fractional(column) {
                words[at(k / 64)] = 1;
            }
        }
        let mut metadata = device
            .alloc_buffer(n * k / 64 * 4, DType::U32, vec![n / 64, k / 64, 64])
            .unwrap();
        metadata.as_mut_slice::<u32>().unwrap().fill(u32::from(bf16::ONE.to_bits()));
        (planes, metadata)
    }

    fn precision_input(device: &MlxDevice, m: usize, k: usize, signed: bool) -> MlxBuffer {
        let mut values = vec![bf16::ZERO; m * k];
        for row in 0..m {
            let sign = if signed && row % 2 == 1 { -1.0 } else { 1.0 };
            values[row * k] = bf16::from_f32(sign);
            values[row * k + 1] = bf16::from_f32(sign / 256.0);
            values[row * k + k / 2] = bf16::from_f32(sign / 65_536.0);
        }
        upload(device, &values, vec![m, k])
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn staged_projection_precision_rounds_after_complete_split_reduction() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES.into_iter().chain([(6_144, 2_048)]) {
            let splits = dflash2_af4g64_staged_splits(n as u32, k as u32);
            assert_eq!(splits == 1, k == 256);
            let (weights, metadata) = precision_weights(&device, n, k, |_| true);
            for m in [8_usize, 16, 24, 32] {
                let input = precision_input(&device, m, k, true);
                // Each precision owns its profile's exact scratch capacity.
                let output = scratch(&device, m * n);
                let mut f32_partials = Vec::new();
                for mode in [None, Some(ActivationPrecision::F32), Some(ActivationPrecision::Bf16)] {
                    let precision = mode.unwrap_or(ActivationPrecision::F32);
                    let profile_splits = dflash2_af4g64_staged_splits_with_precision(
                        n as u32, k as u32, precision,
                    );
                    let elements = dflash2_af4g64_staged_partial_elements_with_precision(
                        m as u32, n as u32, k as u32, precision,
                    );
                    let partials = scratch(&device, elements);
                    let mut encoder = device.command_encoder().unwrap();
                    if precision == ActivationPrecision::Bf16 && profile_splits > splits {
                        let legacy_elements = dflash2_af4g64_staged_partial_elements(
                            m as u32, n as u32, k as u32,
                        );
                        let undersized = partials.slice_view(0, legacy_elements);
                        let error = dispatch_dflash2_af4g64_staged_with_precision(
                            &mut encoder, &mut registry, &device, &input, &weights, &metadata,
                            &undersized, &output, m as u32, n as u32, k as u32, precision,
                        ).unwrap_err();
                        assert!(matches!(error, MlxError::InvalidArgument(_)));
                    }
                    if mode == Some(ActivationPrecision::Bf16) {
                        encoder.start_capture();
                        encoder.set_op_kind(crate::encoder::CapturedOpKind::RmsNorm);
                    }
                    if let Some(precision) = mode {
                        dispatch_dflash2_af4g64_staged_with_precision(
                            &mut encoder, &mut registry, &device, &input, &weights, &metadata,
                            &partials, &output, m as u32, n as u32, k as u32, precision,
                        ).unwrap();
                    } else {
                        dispatch_dflash2_af4g64_staged(
                            &mut encoder, &mut registry, &device, &input, &weights, &metadata,
                            &partials, &output, m as u32, n as u32, k as u32,
                        ).unwrap();
                    }
                    if mode == Some(ActivationPrecision::Bf16) {
                        replay_precision_capture(&mut encoder);
                    }
                    encoder.commit_and_wait().unwrap();
                    for row in 0..m {
                        let sign = if row % 2 == 1 { -1.0 } else { 1.0 };
                        let sum = sign * (1.0 + 1.0 / 256.0 + 1.0 / 65_536.0);
                        let want = if mode == Some(ActivationPrecision::Bf16) {
                            bf16::from_f32(sum).to_f32()
                        } else {
                            sum
                        };
                        for &actual in &output.as_slice::<f32>().unwrap()[row * n..(row + 1) * n] {
                            assert_eq!(actual.to_bits(), want.to_bits(), "{n}x{k} M={m} row={row} {mode:?}");
                        }
                    }
                    let bits: Vec<u32> = partials.as_slice::<f32>().unwrap()[..elements]
                        .iter().map(|v| v.to_bits()).collect();
                    if profile_splits > 1 {
                        let values = partials.as_slice::<f32>().unwrap();
                        for row in 0..m {
                            let sign = if row % 2 == 1 { -1.0 } else { 1.0 };
                            let want: f32 = sign * (1.0 + 1.0 / 256.0 + 1.0 / 65_536.0);
                            let (first, _, height) = row_tiles(m as u32)
                                .find(|&(first, _, height)| row >= first as usize && row < (first + height) as usize)
                                .unwrap();
                            for column in 0..n {
                                let sum: f32 = (0..profile_splits as usize)
                                    .map(|split| values[
                                        first as usize * profile_splits as usize * n
                                            + split * height as usize * n
                                            + (row - first as usize) * n + column
                                    ]).sum();
                                assert_eq!(sum.to_bits(), want.to_bits(), "partial reduction {mode:?}");
                            }
                        }
                    }
                    if mode.is_none() {
                        f32_partials = bits;
                    } else if profile_splits == splits {
                        assert_eq!(bits, f32_partials, "same-profile partials must be unchanged");
                    }
                }
            }
        }
        // Rounding the first run's tie before adding the small second run
        // produces 1.0; rounding the complete sum produces 1.0078125.
        let early = bf16::from_f32(1.0 + 1.0 / 256.0).to_f32() + 1.0 / 65_536.0;
        assert_eq!(bf16::from_f32(early).to_f32(), 1.0);
        assert_eq!(bf16::from_f32(1.0 + 1.0 / 256.0 + 1.0 / 65_536.0).to_f32(), 1.0078125);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn staged_gate_up_precision_rounds_each_producer_before_silu() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (n, k) in SHAPES {
            let splits = dflash2_af4g64_staged_gate_up_splits(n as u32, k as u32);
            assert_eq!(splits == 1, k == 256);
            // Separate columns falsify omitting gate rounding, up rounding, or both.
            let (gate_weights, gate_metadata) = precision_weights(&device, n, k, |c| c % 3 != 1);
            let (up_weights, up_metadata) = precision_weights(&device, n, k, |c| c % 3 != 0);
            for m in [8_usize, 16, 24, 32] {
                let input = precision_input(&device, m, k, false);
                let elements = dflash2_af4g64_staged_gate_up_partial_elements(m as u32, n as u32, k as u32);
                let partials = scratch(&device, elements);
                let output = device.alloc_buffer(m * n * 2, DType::BF16, vec![m, n]).unwrap();
                let mut f32_partials = Vec::new();
                for mode in [None, Some(ActivationPrecision::F32), Some(ActivationPrecision::Bf16)] {
                    let mut encoder = device.command_encoder().unwrap();
                    if mode == Some(ActivationPrecision::Bf16) {
                        encoder.start_capture();
                        encoder.set_op_kind(crate::encoder::CapturedOpKind::RmsNorm);
                    }
                    if let Some(precision) = mode {
                        dispatch_dflash2_af4g64_staged_gate_up_with_precision(
                            &mut encoder, &mut registry, &device, &input, &gate_weights, &gate_metadata,
                            &up_weights, &up_metadata, &partials, &output, m as u32, n as u32, k as u32, precision,
                        ).unwrap();
                    } else {
                        dispatch_dflash2_af4g64_staged_gate_up(
                            &mut encoder, &mut registry, &device, &input, &gate_weights, &gate_metadata,
                            &up_weights, &up_metadata, &partials, &output, m as u32, n as u32, k as u32,
                        ).unwrap();
                    }
                    if mode == Some(ActivationPrecision::Bf16) {
                        replay_precision_capture(&mut encoder);
                    }
                    encoder.commit_and_wait().unwrap();
                    for column in 0..n {
                        let sum = 1.0_f32 + 1.0 / 256.0 + 1.0 / 65_536.0;
                        let gate = if column % 3 != 1 { sum } else { 1.0 };
                        let up = if column % 3 != 0 { sum } else { 1.0 };
                        let oracle = |g: f32, u: f32| {
                            bf16::from_f64(silu(f64::from(g)) * f64::from(u))
                        };
                        let default = oracle(gate, up);
                        let rounded = oracle(bf16::from_f32(gate).to_f32(), bf16::from_f32(up).to_f32());
                        assert_ne!(default.to_bits(), rounded.to_bits(), "probe must expose each boundary");
                        let want = if mode == Some(ActivationPrecision::Bf16) { rounded } else { default };
                        for row in 0..m {
                            assert_eq!(output.as_slice::<bf16>().unwrap()[row * n + column].to_bits(),
                                want.to_bits(), "{n}x{k} M={m} column={column} {mode:?}");
                        }
                    }
                    let bits: Vec<u32> = partials.as_slice::<f32>().unwrap()[..elements]
                        .iter().map(|v| v.to_bits()).collect();
                    if mode.is_none() {
                        f32_partials = bits;
                    } else {
                        assert_eq!(bits, f32_partials, "gate/up partials must stay F32");
                    }
                }
            }
        }
    }

    fn replay_precision_capture(encoder: &mut CommandEncoder) {
        use crate::encoder::{CapturedNode, CapturedOpKind};
        for node in encoder.take_capture().unwrap() {
            match node {
                CapturedNode::Dispatch {
                    pipeline, bindings, threads_per_grid, threads_per_threadgroup,
                    threadgroup_memory, dispatch_kind, op_kind, ..
                } => {
                    assert_eq!(op_kind, CapturedOpKind::Other);
                    encoder.set_op_kind(op_kind);
                    encoder.replay_dispatch(
                        &pipeline, &bindings, &threadgroup_memory,
                        threads_per_grid, threads_per_threadgroup, dispatch_kind,
                    );
                }
                CapturedNode::Barrier => encoder.memory_barrier(),
            }
        }
    }
}
