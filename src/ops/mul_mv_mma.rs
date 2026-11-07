//! Few-row MMA mat-vec (`mul_mv_mma`) for GGUF block weights at decode
//! widths 1..=8: the reference implementation's technique, ported in
//! `shaders/mul_mv_mma.metal`.
//!
//! Kernels: `kernel_mul_mv_mma_<q>_f32_nt<NT>_rt1`, NT ∈ {1, 2, 4}, for
//! Q4_K / Q6_K / Q5_0 (generic kind, 64-weight chunks) and Q8_0 (32-weight
//! block kind). Function constants: nsg (i32 @ 2300) and the block kind's
//! compile-time K (i32 @ 2304, 0 for the generic kind).
//!
//! Tiling (the reference's rule, `rt = 1` because widths are ≤ 8):
//!   - `nsg` = 32 for ≤ 64 rows, 16 for ≤ 6144 rows, else 8, halved while it
//!     exceeds the number of K steps (K/32 for the block kind, K/64 else);
//!   - `nt` = 4 halved while it exceeds
//!     `min(16384 / (nsg·256), N / 1024)`;
//!   - the pipeline's thread limit may halve `nsg` further (the reference's
//!     `_auto` step), and threadgroup memory is `nsg·nt·256` bytes;
//!   - grid `(⌈N / 8·nt⌉, 1, 1)`, threadgroup `(32, nsg, 1)`.
//!
//! Every input above depends on (type, N, K) only, never on the batch
//! width, so every width runs the same pipeline.
//!
//! Callers reach this module only through the validated
//! `quantized_matmul_ggml_column_invariant_mma` entry.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::mul_mv_ext::MulMvExtGpuArgs;
use crate::ops::quantized_matmul_ggml::GgmlType;

/// Function-constant index of the simdgroup count.
const FC_NSG: usize = 2300;
/// Function-constant index of the block kind's compile-time K.
const FC_NE00: usize = 2304;

/// Rows per 8×8 MMA tile.
const TILE: u32 = 8;
/// Most 8-row weight tiles per threadgroup.
const NT_MAX: u32 = 4;
/// Largest cross-simdgroup reduction buffer.
const SMEM_MAX: u64 = 16384;
/// Threadgroups needed to fill the GPU (sizes `nt`).
const MIN_THREADGROUPS: u64 = 128;
/// Bytes of one f32 8×8 output tile.
const TILE_BYTES: u64 = 8 * 8 * 4;

/// Kernel kind, as the reference selects it at ≤ 8 columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MulMvMmaKind {
    /// 64-weight chunks dequantized to f32 (Q4_K, Q6_K, Q5_0).
    Generic,
    /// 32-weight blocks as exact-integer halves, scale per block (Q8_0).
    Block,
}

/// Launch tiling before the pipeline thread-limit step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MulMvMmaTiling {
    pub(super) kind: MulMvMmaKind,
    /// Simdgroups per threadgroup, each over a round-robin slice of K.
    pub(super) nsg: u32,
    /// 8-row weight tiles per threadgroup.
    pub(super) nt: u32,
}

/// The kind for an admitted type, `None` otherwise.
pub(super) fn kind(ggml_type: GgmlType) -> Option<MulMvMmaKind> {
    match ggml_type {
        GgmlType::Q8_0 => Some(MulMvMmaKind::Block),
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q5_0 => Some(MulMvMmaKind::Generic),
        _ => None,
    }
}

/// K elements per loop step of a kind; K must be a multiple of it.
pub(super) fn k_step(kind: MulMvMmaKind) -> u32 {
    match kind {
        MulMvMmaKind::Block => 32,
        MulMvMmaKind::Generic => 64,
    }
}

fn halve_to_limit(mut n: u32, limit: u64) -> u32 {
    while n > 1 && u64::from(n) > limit {
        n /= 2;
    }
    n
}

/// The reference's tiling for `rt = 1` from (type, N, K) alone.
pub(super) fn tiling(ggml_type: GgmlType, n: u32, k: u32) -> Option<MulMvMmaTiling> {
    let kind = kind(ggml_type)?;
    let n_steps = u64::from(k / k_step(kind));
    let nsg0 = if n <= 64 {
        32
    } else if n <= 6144 {
        16
    } else {
        8
    };
    let nsg = halve_to_limit(nsg0, n_steps);
    let nt_smem = SMEM_MAX / (u64::from(nsg) * TILE_BYTES);
    let nt_rows = u64::from(n) / (MIN_THREADGROUPS * u64::from(TILE));
    let nt = halve_to_limit(NT_MAX, nt_smem.min(nt_rows));
    Some(MulMvMmaTiling { kind, nsg, nt })
}

fn kernel_name(ggml_type: GgmlType, nt: u32) -> Result<&'static str> {
    Ok(match (ggml_type, nt) {
        (GgmlType::Q4_K, 1) => "kernel_mul_mv_mma_q4_K_f32_nt1_rt1",
        (GgmlType::Q4_K, 2) => "kernel_mul_mv_mma_q4_K_f32_nt2_rt1",
        (GgmlType::Q4_K, 4) => "kernel_mul_mv_mma_q4_K_f32_nt4_rt1",
        (GgmlType::Q6_K, 1) => "kernel_mul_mv_mma_q6_K_f32_nt1_rt1",
        (GgmlType::Q6_K, 2) => "kernel_mul_mv_mma_q6_K_f32_nt2_rt1",
        (GgmlType::Q6_K, 4) => "kernel_mul_mv_mma_q6_K_f32_nt4_rt1",
        (GgmlType::Q5_0, 1) => "kernel_mul_mv_mma_q5_0_f32_nt1_rt1",
        (GgmlType::Q5_0, 2) => "kernel_mul_mv_mma_q5_0_f32_nt2_rt1",
        (GgmlType::Q5_0, 4) => "kernel_mul_mv_mma_q5_0_f32_nt4_rt1",
        (GgmlType::Q8_0, 1) => "kernel_mul_mv_mma_q8_0_f32_nt1_rt1",
        (GgmlType::Q8_0, 2) => "kernel_mul_mv_mma_q8_0_f32_nt2_rt1",
        (GgmlType::Q8_0, 4) => "kernel_mul_mv_mma_q8_0_f32_nt4_rt1",
        (other, nt) => {
            return Err(MlxError::InvalidArgument(format!(
                "mul_mv_mma: no kernel for {other:?} at nt {nt}"
            )))
        }
    })
}

/// Shape of one validated dispatch (single weight matrix, contiguous
/// `[m, k]` f32 input, `[m, n]` f32 output).
pub(super) struct MulMvMmaParams {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) ggml_type: GgmlType,
}

/// Encode one `mul_mv_mma` dispatch. The caller has validated the type,
/// `m ∈ 1..=8`, K alignment, buffer extents, and aliasing.
pub(super) fn mul_mv_mma_dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &MulMvMmaParams,
) -> Result<()> {
    let plan = tiling(params.ggml_type, params.n, params.k).ok_or_else(|| {
        MlxError::InvalidArgument(format!(
            "mul_mv_mma: unsupported type {:?}",
            params.ggml_type
        ))
    })?;
    let name = kernel_name(params.ggml_type, plan.nt)?;
    let ne00_fc = match plan.kind {
        MulMvMmaKind::Block => params.k as i32,
        MulMvMmaKind::Generic => 0,
    };

    // The reference's `_auto` step: fewer simdgroups only when the compiled
    // pipeline cannot run 32·nsg threads. Depends on (type, N, K) only.
    let mut nsg = plan.nsg;
    let pipeline = loop {
        let pipeline = registry
            .get_pipeline_with_constants(
                name,
                device.metal_device(),
                &[],
                &[(FC_NSG, nsg as i32), (FC_NE00, ne00_fc)],
            )?
            .clone();
        if nsg > 1 && pipeline.max_total_threads_per_threadgroup() < u64::from(nsg) * 32 {
            nsg /= 2;
            continue;
        }
        break pipeline;
    };
    let smem = u64::from(nsg) * u64::from(plan.nt) * TILE_BYTES;

    let block_qk = params.ggml_type.block_values();
    let nb00 = u64::from(params.ggml_type.block_bytes());
    let nb01 = u64::from(params.k / block_qk) * nb00;
    let nb02 = nb01 * u64::from(params.n);
    let nb11 = u64::from(params.k) * 4;
    let nb12 = nb11 * u64::from(params.m);
    let args = MulMvExtGpuArgs {
        ne00: params.k as i32,
        ne01: params.n as i32,
        ne02: 1,
        _pad0: 0,
        nb00,
        nb01,
        nb02,
        nb03: nb02,
        ne10: params.k as i32,
        ne11: params.m as i32,
        ne12: 1,
        _pad1: 0,
        nb10: 4,
        nb11,
        nb12,
        nb13: nb12,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2: 1,
        r3: 1,
        _pad2: 0,
    };

    let rows_per_group = u64::from(TILE * plan.nt);
    encoder.dispatch_tracked_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&args))),
            (1, KernelArg::Buffer(weight)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
        ],
        &[(0, smem)],
        &[weight, input],
        &[output],
        crate::MTLSize::new(u64::from(params.n).div_ceil(rows_per_group), 1, 1),
        crate::MTLSize::new(32, u64::from(nsg), 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(ggml_type: GgmlType, n: u32, k: u32) -> (MulMvMmaKind, u32, u32) {
        let t = tiling(ggml_type, n, k).expect("admitted type");
        (t.kind, t.nsg, t.nt)
    }

    #[test]
    fn gemma_shapes_match_the_reference_tiling() {
        use MulMvMmaKind::{Block, Generic};
        for (n, k, nsg, nt) in [
            (4096, 2816, 16, 4),
            (8192, 2816, 8, 4),
            (2048, 2816, 16, 2),
            (1024, 2816, 16, 1),
            (2816, 4096, 16, 2),
            (2816, 8192, 16, 2),
            (2112, 2816, 16, 2),
        ] {
            assert_eq!(plan(GgmlType::Q4_K, n, k), (Generic, nsg, nt), "Q4_K {n}x{k}");
        }
        assert_eq!(plan(GgmlType::Q6_K, 262_144, 2816), (Generic, 8, 4));
        assert_eq!(plan(GgmlType::Q6_K, 2048, 2816), (Generic, 16, 2));
        assert_eq!(plan(GgmlType::Q5_0, 2816, 2112), (Generic, 16, 2));
        assert_eq!(plan(GgmlType::Q8_0, 2816, 2112), (Block, 16, 2));
        // Few rows: nsg 32 halved to the K steps, nt 1.
        assert_eq!(plan(GgmlType::Q4_K, 37, 512), (Generic, 8, 1));
        assert_eq!(plan(GgmlType::Q8_0, 37, 320), (Block, 8, 1));
        assert!(tiling(GgmlType::Q5_K, 4096, 2816).is_none());
    }
}
