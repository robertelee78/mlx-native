//! Q4-GEMMA-DECODE dense decode routes (ADR-063): D1 row-tiled Q4_K/Q6_K and
//! D2 multi-column Q5_0/Q8_0 mat-vec.
//!
//! Exactness contract: every output (column c, row r) is bitwise equal to the
//! reference single-column kernel on column c — `kernel_mul_mv_q4_K_f32`,
//! `kernel_mul_mv_q6_K_f32_nr2`, `kernel_mul_mv_q5_0_f32` and
//! `kernel_mul_mv_q8_0_f32_nr2`. The kernels live in
//! `quantized_matmul_ggml.metal` next to their references, so both share one
//! compile-flag set on both metallib paths. Geometry (R0 rows per SIMD group,
//! NSG SIMD groups per threadgroup, column tile width) never changes bits; it
//! is chosen from a static per-shape table, never from timing or run-time
//! data.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::ggml_capability::GgmlRoutingPolicy;
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::{
    checked_mn_tile_byte_offset, GgmlMatvecGpuParams, GgmlQuantizedMatmulParams, GgmlType,
    MM_ROUTING_THRESHOLD,
};

/// D1 geometry: `r0` weight rows per SIMD group and `nsg` SIMD groups per
/// threadgroup. Instantiated cells: `r0 ∈ {2, 4}`, `nsg ∈ {2, 4, 8}`. One
/// row per SIMD group is the stock geometry and stays on the stock kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RowTiledGeometry {
    pub r0: u32,
    pub nsg: u32,
}

impl RowTiledGeometry {
    pub const R0_CELLS: [u32; 2] = [2, 4];
    pub const NSG_CELLS: [u32; 3] = [2, 4, 8];

    fn validate(self) -> Result<()> {
        if Self::R0_CELLS.contains(&self.r0) && Self::NSG_CELLS.contains(&self.nsg) {
            Ok(())
        } else {
            Err(MlxError::InvalidArgument(format!(
                "row-tiled geometry {self:?} is not an instantiated cell (r0 ∈ {{2,4}}, nsg ∈ {{2,4,8}})"
            )))
        }
    }

    pub(crate) fn rows_per_threadgroup(self) -> u32 {
        self.r0 * self.nsg
    }
}

/// The frozen per-shape D1 geometry table: `(type, n, k)` for every Gemma 4
/// dense Q4_K/Q6_K projection (ADR-063 Q4-GEMMA-DECODE §2.5) and the lm head.
/// A shape outside this table keeps its stock route even when the D1 policy
/// field is set. The geometry is replaced only by the M0 freeze (§7.5).
pub fn row_tiled_geometry(ggml_type: GgmlType, n: u32, k: u32) -> Option<RowTiledGeometry> {
    const G: RowTiledGeometry = RowTiledGeometry { r0: 2, nsg: 2 };
    match (ggml_type, n, k) {
        // q 2816→4096 / 8192, k 2816→2048 / 1024, v (Q4_K) 2816→2048,
        // o 4096 / 8192→2816, dense gate/up 2816→2112.
        (GgmlType::Q4_K, 4096 | 8192 | 2048 | 1024 | 2112, 2816)
        | (GgmlType::Q4_K, 2816, 4096 | 8192) => Some(G),
        // v (Q6_K) 2816→2048 and the tied lm head 2816→262,144.
        (GgmlType::Q6_K, 2048 | 262_144, 2816) => Some(G),
        _ => None,
    }
}

/// The frozen D2 shape table: dense `ffn_down` 2112→2816 (Q5_0 / Q8_0).
pub fn dense_mn_shape_admitted(ggml_type: GgmlType, n: u32, k: u32) -> bool {
    matches!(
        (ggml_type, n, k),
        (GgmlType::Q5_0 | GgmlType::Q8_0, 2816, 2112)
    )
}

/// Which Q4-GEMMA-DECODE dense route the policy admits for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmaDenseRoute {
    RowTiled(RowTiledGeometry),
    MnQ5_0,
    MnQ8_0,
}

/// Route predicate shared by the planner and the capability API. A lever is
/// honoured only when its field is set, the reference flags of the contract
/// hold (`dense_decode_mv_ext == false`; Q6_K needs `dense_q6k_mv_nr2`, Q8_0
/// needs `dense_q8_0_mv_nr2`), and the shape is in the frozen table.
pub(crate) fn plan_gemma_dense_route(
    ggml_type: GgmlType,
    m: u32,
    n: u32,
    k: u32,
    routing: &GgmlRoutingPolicy,
) -> Option<GemmaDenseRoute> {
    if routing.dense_decode_mv_ext || m == 0 || m > MM_ROUTING_THRESHOLD {
        return None;
    }
    let width_ok = m >= 2 || routing.dense_row_tiled_m1;
    match ggml_type {
        GgmlType::Q4_K if routing.dense_row_tiled_q4k && width_ok => {
            row_tiled_geometry(ggml_type, n, k).map(GemmaDenseRoute::RowTiled)
        }
        GgmlType::Q6_K
            if routing.dense_row_tiled_q6k && routing.dense_q6k_mv_nr2 && width_ok =>
        {
            row_tiled_geometry(ggml_type, n, k).map(GemmaDenseRoute::RowTiled)
        }
        GgmlType::Q5_0 if routing.dense_mn_q5_0 && m >= 2 => {
            dense_mn_shape_admitted(ggml_type, n, k).then_some(GemmaDenseRoute::MnQ5_0)
        }
        GgmlType::Q8_0 if routing.dense_mn_q8_0 && routing.dense_q8_0_mv_nr2 && m >= 2 => {
            dense_mn_shape_admitted(ggml_type, n, k).then_some(GemmaDenseRoute::MnQ8_0)
        }
        _ => None,
    }
}

/// Column tiles of at most four columns: `(col0, width)`.
pub(crate) fn gemma_column_tiling(m: u32) -> Vec<(u32, u32)> {
    match m {
        1..=4 => vec![(0, m)],
        5 => vec![(0, 3), (3, 2)],
        6 => vec![(0, 3), (3, 3)],
        7 => vec![(0, 4), (4, 3)],
        8 => vec![(0, 4), (4, 4)],
        _ => Vec::new(),
    }
}

pub(crate) fn gemma_dense_dispatch_count(m: u32) -> u32 {
    gemma_column_tiling(m).len() as u32
}

/// Kernel name of one D1 tile.
pub(crate) fn row_tiled_kernel_name(
    ggml_type: GgmlType,
    geometry: RowTiledGeometry,
    width: u32,
) -> Result<String> {
    let ty = match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "row-tiled mat-vec supports Q4_K and Q6_K, got {other:?}"
            )))
        }
    };
    Ok(format!(
        "kernel_mul_mv_{ty}_f32_rt{}_r1_{width}_nsg{}",
        geometry.r0, geometry.nsg
    ))
}

/// Kernel name of one D2 tile.
pub(crate) fn dense_mn_kernel_name(ggml_type: GgmlType, width: u32) -> Result<String> {
    match ggml_type {
        GgmlType::Q5_0 => Ok(format!("kernel_mul_mv_q5_0_f32_mN_r1_{width}")),
        GgmlType::Q8_0 => Ok(format!("kernel_mul_mv_q8_0_f32_nr2_mN_r1_{width}")),
        other => Err(MlxError::InvalidArgument(format!(
            "dense mN supports Q5_0 and Q8_0, got {other:?}"
        ))),
    }
}

/// Every D1 and D2 kernel name, for runtime-source registration.
pub(crate) fn all_kernel_names() -> Vec<String> {
    let mut names = Vec::new();
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for r0 in RowTiledGeometry::R0_CELLS {
            for nsg in RowTiledGeometry::NSG_CELLS {
                for width in 1..=4 {
                    if let Ok(name) =
                        row_tiled_kernel_name(ty, RowTiledGeometry { r0, nsg }, width)
                    {
                        names.push(name);
                    }
                }
            }
        }
    }
    for ty in [GgmlType::Q5_0, GgmlType::Q8_0] {
        for width in 2..=4 {
            if let Ok(name) = dense_mn_kernel_name(ty, width) {
                names.push(name);
            }
        }
    }
    for nsg in RowTiledGeometry::NSG_CELLS {
        for width in 1..=4 {
            names.push(dual_gate_up_kernel_name(nsg, width));
        }
    }
    names
}

fn div_ceil(a: u32, b: u32) -> u64 {
    u64::from(a).div_ceil(u64::from(b))
}

/// F1 kernel name (dual-weight Q4_K gate+up).
pub(crate) fn dual_gate_up_kernel_name(nsg: u32, width: u32) -> String {
    format!("kernel_mul_mv_q4_K_f32_rt_dual_r1_{width}_nsg{nsg}")
}

/// Encode F1: dense gate and up (same Q4_K shape) in one launch per column
/// tile. Each output equals the reference single-column kernel on its own
/// tensor. Validation is the caller's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_dual_gate_up(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    gate_out: &MlxBuffer,
    up_out: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    nsg: u32,
) -> Result<()> {
    if params.ggml_type != GgmlType::Q4_K || !RowTiledGeometry::NSG_CELLS.contains(&nsg) {
        return Err(MlxError::InvalidArgument(format!(
            "dual gate+up requires Q4_K and nsg in {{2,4,8}}, got {:?} nsg={nsg}",
            params.ggml_type
        )));
    }
    for (col0, width) in gemma_column_tiling(params.m) {
        let kernel = dual_gate_up_kernel_name(nsg, width);
        let input_off =
            checked_mn_tile_byte_offset(input.byte_offset(), col0 as usize, params.k, "dual input")?;
        let gate_off =
            checked_mn_tile_byte_offset(gate_out.byte_offset(), col0 as usize, params.n, "gate out")?;
        let up_off =
            checked_mn_tile_byte_offset(up_out.byte_offset(), col0 as usize, params.n, "up out")?;
        let pipeline = registry.get_pipeline_with_constants(
            &kernel,
            device.metal_device(),
            &[],
            &[(700, 1), (701, 1), (702, 1)],
        )?;
        let gpu_params = GgmlMatvecGpuParams {
            ne00: i64::from(params.k),
            ne01: i64::from(params.n),
            ne02: 1,
            ne10: i64::from(params.k),
            ne12: 1,
            ne0: i64::from(params.n),
            ne1: i64::from(width),
            r2: 1,
            r3: 1,
        };
        encoder.dispatch_tracked_threadgroups_with_args(
            pipeline,
            &[
                (0, KernelArg::Buffer(gate)),
                (1, KernelArg::BufferWithOffset(input, input_off)),
                (2, KernelArg::BufferWithOffset(gate_out, gate_off)),
                (3, KernelArg::Bytes(as_bytes(&gpu_params))),
                (4, KernelArg::Buffer(up)),
                (5, KernelArg::BufferWithOffset(up_out, up_off)),
            ],
            &[gate, up, input],
            &[gate_out, up_out],
            metal::MTLSize::new(div_ceil(params.n, nsg), 1, 1),
            metal::MTLSize::new(32, u64::from(nsg), 1),
        );
    }
    Ok(())
}

/// Encode the D1 route at an explicit geometry. Columns `0..m` are tiled in
/// chunks of at most four; every column's output equals the reference
/// single-column kernel. Validation (dtypes, buffer extents) is the caller's;
/// public callers use
/// [`crate::ops::quantized_matmul_ggml::quantized_matmul_ggml_row_tiled_with_geometry`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_row_tiled(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    geometry: RowTiledGeometry,
) -> Result<()> {
    geometry.validate()?;
    for (col0, width) in gemma_column_tiling(params.m) {
        let kernel = row_tiled_kernel_name(params.ggml_type, geometry, width)?;
        dispatch_tile(
            encoder,
            registry,
            device,
            input,
            weight,
            output,
            params,
            &kernel,
            col0,
            width,
            div_ceil(params.n, geometry.rows_per_threadgroup()),
            metal::MTLSize::new(32, u64::from(geometry.nsg), 1),
            None,
        )?;
    }
    Ok(())
}

/// Encode the D2 route (Q5_0 / Q8_0 multi-column mat-vec).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_dense_mn(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> Result<()> {
    if params.m < 2 {
        return Err(MlxError::InvalidArgument(
            "dense mN requires m >= 2; width 1 uses the reference kernel".into(),
        ));
    }
    for (col0, width) in gemma_column_tiling(params.m) {
        let kernel = dense_mn_kernel_name(params.ggml_type, width)?;
        let (tgx, threads, shmem) = match params.ggml_type {
            GgmlType::Q5_0 => (div_ceil(params.n, 8), metal::MTLSize::new(8, 8, 1), None),
            _ => (
                div_ceil(params.n, 2),
                metal::MTLSize::new(32, 4, 1),
                Some(u64::from(width) * 2 * 32 * 4),
            ),
        };
        dispatch_tile(
            encoder, registry, device, input, weight, output, params, &kernel, col0, width, tgx,
            threads, shmem,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_tile(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    kernel: &str,
    col0: u32,
    width: u32,
    threadgroups_x: u64,
    threads_per_tg: metal::MTLSize,
    shmem_bytes: Option<u64>,
) -> Result<()> {
    let label = params.ggml_type;
    let input_off =
        checked_mn_tile_byte_offset(input.byte_offset(), col0 as usize, params.k, &format!("{label:?} input"))?;
    let output_off = checked_mn_tile_byte_offset(
        output.byte_offset(),
        col0 as usize,
        params.n,
        &format!("{label:?} output"),
    )?;
    let pipeline = registry.get_pipeline_with_constants(
        kernel,
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;
    let gpu_params = GgmlMatvecGpuParams {
        ne00: i64::from(params.k),
        ne01: i64::from(params.n),
        ne02: 1,
        ne10: i64::from(params.k),
        ne12: 1,
        ne0: i64::from(params.n),
        ne1: i64::from(width),
        r2: 1,
        r3: 1,
    };
    let threadgroups = metal::MTLSize::new(threadgroups_x, 1, 1);
    let args = [
        (0, KernelArg::Buffer(weight)),
        (1, KernelArg::BufferWithOffset(input, input_off)),
        (2, KernelArg::BufferWithOffset(output, output_off)),
        (3, KernelArg::Bytes(as_bytes(&gpu_params))),
    ];
    match shmem_bytes {
        Some(bytes) => encoder.dispatch_tracked_threadgroups_with_args_and_shared(
            pipeline,
            &args,
            &[(0, bytes)],
            &[weight, input],
            &[output],
            threadgroups,
            threads_per_tg,
        ),
        None => encoder.dispatch_tracked_threadgroups_with_args(
            pipeline,
            &args,
            &[weight, input],
            &[output],
            threadgroups,
            threads_per_tg,
        ),
    }
    Ok(())
}
