//! Explicit diagnostics from the SAME kernel that produces selected IDs/weights.
//!
//! No observer/global opt-in and no waits. `None` calls the ordinary API. A
//! witness dispatch preserves the selected variant and exports its actual F32
//! selection-domain kth/kth+1 values before top-K renorm/expert scaling. This is
//! not CPU re-softmax, a logits-gap proxy, a model parity claim or a new policy.
use crate::{
    buffer::MlxBuffer,
    device::MlxDevice,
    dtypes::DType,
    encoder::{as_bytes, CommandEncoder, KernelArg},
    error::{MlxError, Result},
    kernel_registry::KernelRegistry,
};
use metal::MTLSize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingSelectionVariant {
    QwenSoftmax,
    GemmaV1,
    GemmaV2,
    GemmaV3,
    DeepSeekV4SqrtSoftplusBias,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionDomain {
    SoftmaxBeforeTopKRenorm,
    SqrtSoftplusPlusBiasBeforeUnbiasedWeightNorm,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutingSelectionDispatch {
    /// Actual chosen pipeline name, not an environment-value inference.
    pub kernel: &'static str,
    pub variant: RoutingSelectionVariant,
    pub domain: SelectionDomain,
    pub witness_encoded: bool,
}
/// Caller-owned, distinct writable outputs: F32 `[rows,2]`, U32 `[rows,2]`,
/// U32 `[rows]`. Status1=scored,2=all experts selected (no kth+1),3=invalid.
/// Every selected row is written. Never inspect any output before the existing
/// owning command's successful checked completion. No method here waits.
#[derive(Clone, Copy)]
pub struct SelectionWitness<'a> {
    pub scores: &'a MlxBuffer,
    pub ids: &'a MlxBuffer,
    pub status: &'a MlxBuffer,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionBoundary {
    Scored {
        kth_id: u32,
        kth_score: f32,
        next_id: u32,
        next_score: f32,
    },
    AllExpertsSelected {
        kth_id: u32,
        kth_score: f32,
    },
    /// Nonfinite producer input/domain or invalid native route. No margin.
    Invalid,
}
fn invalid(message: &str) -> MlxError {
    MlxError::InvalidArgument(message.into())
}
fn elements(rows: u32, width: u32) -> Result<usize> {
    // The diagnostic output/input indexing includes u32 products in Metal.
    rows.checked_mul(width)
        .map(|n| n as usize)
        .ok_or_else(|| invalid("routing witness u32 index overflow"))
}
fn buffer(b: &MlxBuffer, dtype: DType, count: usize, writable: bool) -> Result<()> {
    if b.contents_ptr().is_null()
        || b.dtype() != dtype
        || b.element_count() != count
        || b.byte_offset() % 4 != 0
        || b.data_byte_len()
            != count
                .checked_mul(4)
                .ok_or_else(|| invalid("routing witness byte overflow"))?
        || (writable && !b.is_cpu_writable())
    {
        return Err(invalid(
            "routing witness dtype/extent/alignment/writability",
        ));
    }
    Ok(())
}
fn physical_range(base: usize, offset: u64, len: usize) -> Result<(usize, usize)> {
    let begin = base
        .checked_add(usize::try_from(offset).map_err(|_| invalid("routing witness offset"))?)
        .ok_or_else(|| invalid("routing witness range overflow"))?;
    Ok((
        begin,
        begin
            .checked_add(len)
            .ok_or_else(|| invalid("routing witness range overflow"))?,
    ))
}
fn range(b: &MlxBuffer) -> Result<(usize, usize)> {
    physical_range(
        b.contents_ptr() as usize,
        b.byte_offset(),
        b.data_byte_len(),
    )
}
fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}
fn disjoint(outputs: &[&MlxBuffer], inputs: &[&MlxBuffer]) -> Result<()> {
    for (i, a) in outputs.iter().enumerate() {
        let (a0, a1) = range(a)?;
        for b in outputs.iter().skip(i + 1).chain(inputs.iter()) {
            let (b0, b1) = range(b)?;
            if overlaps((a0, a1), (b0, b1)) {
                return Err(invalid("routing witness output aliases input/output"));
            }
        }
    }
    Ok(())
}
fn validate(
    w: SelectionWitness<'_>,
    rows: u32,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    k: u32,
    id_dtype: DType,
    inputs: &[&MlxBuffer],
) -> Result<()> {
    if rows == 0 {
        return Err(invalid("routing witness requires positive rows"));
    }
    buffer(ids, id_dtype, elements(rows, k)?, true)?;
    buffer(weights, DType::F32, elements(rows, k)?, true)?;
    buffer(w.scores, DType::F32, elements(rows, 2)?, true)?;
    buffer(w.ids, DType::U32, elements(rows, 2)?, true)?;
    buffer(w.status, DType::U32, rows as usize, true)?;
    disjoint(&[ids, weights, w.scores, w.ids, w.status], inputs)
}
// Captured/reordered graphs must see ALL new witness writes. Metadata only:
// this neither inserts a barrier nor changes direct-dispatch ordering.
fn annotate(enc: &mut CommandEncoder, inputs: &[&MlxBuffer], outputs: &[&MlxBuffer]) -> Result<()> {
    if enc.is_capturing() {
        let reads = inputs
            .iter()
            .map(|b| range(b))
            .collect::<Result<Vec<_>>>()?;
        let writes = outputs
            .iter()
            .map(|b| range(b))
            .collect::<Result<Vec<_>>>()?;
        enc.set_pending_buffer_ranges(reads, writes);
    }
    Ok(())
}
/// Decode only AFTER the caller's existing successful checked GPU drain. Invalid
/// rows are explicit, and K==E never supplies an invented zero selection margin.
/// This reads actual witness values; it does not recompute any routing scores.
pub fn read_after_checked_drain(
    w: SelectionWitness<'_>,
    rows: u32,
    experts: u32,
    k: u32,
) -> Result<Vec<SelectionBoundary>> {
    if rows == 0 || k == 0 || k > experts {
        return Err(invalid("invalid witness decode geometry"));
    }
    buffer(w.scores, DType::F32, elements(rows, 2)?, true)?;
    buffer(w.ids, DType::U32, elements(rows, 2)?, true)?;
    buffer(w.status, DType::U32, rows as usize, true)?;
    let s = w.scores.as_logical_slice::<f32>()?;
    let ids = w.ids.as_logical_slice::<u32>()?;
    let status = w.status.as_logical_slice::<u32>()?;
    (0..rows as usize)
        .map(|r| {
            decode(
                status[r],
                [ids[2 * r], ids[2 * r + 1]],
                [s[2 * r], s[2 * r + 1]],
                experts,
                k,
            )
        })
        .collect()
}
fn decode(
    status: u32,
    ids: [u32; 2],
    s: [f32; 2],
    experts: u32,
    k: u32,
) -> Result<SelectionBoundary> {
    match status {
        1 if k < experts
            && ids[0] < experts
            && ids[1] < experts
            && ids[0] != ids[1]
            && s[0].is_finite()
            && s[1].is_finite()
            && s[0] >= s[1] =>
        {
            Ok(SelectionBoundary::Scored {
                kth_id: ids[0],
                kth_score: s[0],
                next_id: ids[1],
                next_score: s[1],
            })
        }
        2 if k == experts
            && ids[0] < experts
            && s[0].is_finite()
            && ids[1] == u32::MAX
            && s[1].is_nan() =>
        {
            Ok(SelectionBoundary::AllExpertsSelected {
                kth_id: ids[0],
                kth_score: s[0],
            })
        }
        3 => Ok(SelectionBoundary::Invalid),
        _ => Err(invalid(
            "routing witness unwritten/corrupt native status/boundary",
        )),
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QwenParams {
    rows: u32,
    experts: u32,
    k: u32,
    pad: f32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmaParams {
    experts: u32,
    k: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekParams {
    rows: u32,
    vocab: u32,
}
/// Selected Qwen GPU softmax route. Shapes follow the ordinary API; diagnostics
/// explicitly refuse non-power-of-two expert counts, rather than approximate a
/// different reduction. The old API's accepted inputs are not changed.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qwen_with_witness(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    rows: u32,
    experts: u32,
    k: u32,
    witness: Option<SelectionWitness<'_>>,
) -> Result<RoutingSelectionDispatch> {
    let mut out = RoutingSelectionDispatch {
        kernel: "moe_softmax_topk_f32",
        variant: RoutingSelectionVariant::QwenSoftmax,
        domain: SelectionDomain::SoftmaxBeforeTopKRenorm,
        witness_encoded: false,
    };
    let Some(w) = witness else {
        super::moe_softmax_topk::dispatch_moe_softmax_topk(
            enc, registry, device, logits, ids, weights, rows, experts, k,
        )?;
        return Ok(out);
    };
    if rows == 0 || !experts.is_power_of_two() || experts > 256 || k == 0 || k > 64 || k > experts {
        return Err(invalid("Qwen diagnostic rows/experts/top-K unsupported"));
    }
    buffer(logits, DType::F32, elements(rows, experts)?, false)?;
    validate(w, rows, ids, weights, k, DType::U32, &[logits])?;
    out.kernel = "moe_softmax_topk_selection_witness_f32";
    out.witness_encoded = true;
    let pipeline = registry.get_pipeline(out.kernel, device.metal_device())?;
    let p = QwenParams {
        rows,
        experts,
        k,
        pad: 0.0,
    };
    let tg = experts.min(128) as u64;
    annotate(enc, &[logits], &[ids, weights, w.scores, w.ids, w.status])?;
    enc.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&p))),
            (1, KernelArg::Buffer(logits)),
            (2, KernelArg::Buffer(ids)),
            (3, KernelArg::Buffer(weights)),
            (4, KernelArg::Buffer(w.scores)),
            (5, KernelArg::Buffer(w.ids)),
            (6, KernelArg::Buffer(w.status)),
        ],
        &[(0, (2 * tg + experts as u64) * 4)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    Ok(out)
}
/// Actual cached scalar/per-slot Gemma V1/V2/V3 producer; no serial substitution
/// for the selected tournament route. Existing HF batched body dispatches this
/// same per-slot producer. The distinct native batched API is not covered here.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemma_with_witness(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    scales: &MlxBuffer,
    experts: u32,
    k: u32,
    witness: Option<SelectionWitness<'_>>,
) -> Result<RoutingSelectionDispatch> {
    let kernel = super::fused_norm_add::selected_fused_moe_routing_f32_kernel();
    let (variant, diagnostic) = match kernel {
        "fused_moe_routing_f32" => (
            RoutingSelectionVariant::GemmaV1,
            "fused_moe_routing_f32_selection_witness",
        ),
        "fused_moe_routing_f32_v2" => (
            RoutingSelectionVariant::GemmaV2,
            "fused_moe_routing_f32_v2_selection_witness",
        ),
        "fused_moe_routing_f32_v3" => (
            RoutingSelectionVariant::GemmaV3,
            "fused_moe_routing_f32_v3_selection_witness",
        ),
        _ => return Err(invalid("unrecognized actual cached Gemma routing variant")),
    };
    let mut out = RoutingSelectionDispatch {
        kernel,
        variant,
        domain: SelectionDomain::SoftmaxBeforeTopKRenorm,
        witness_encoded: false,
    };
    let Some(w) = witness else {
        super::fused_norm_add::dispatch_fused_moe_routing_f32(
            enc,
            registry,
            device.metal_device(),
            logits,
            ids,
            weights,
            scales,
            experts,
            k,
        )?;
        return Ok(out);
    };
    if experts < 32
        || !experts.is_power_of_two()
        || experts > 256
        || k == 0
        || k > 64
        || k > experts
    {
        return Err(invalid("Gemma diagnostic experts/top-K unsupported"));
    }
    buffer(logits, DType::F32, experts as usize, false)?;
    buffer(scales, DType::F32, experts as usize, false)?;
    validate(w, 1, ids, weights, k, DType::U32, &[logits, scales])?;
    out.kernel = diagnostic;
    out.witness_encoded = true;
    let p = GemmaParams { experts, k };
    let tg = 64.min(experts.next_power_of_two()) as u64;
    let nsg = (tg + 31) / 32;
    let slots = 2 * experts as u64
        + match variant {
            RoutingSelectionVariant::GemmaV1 => tg,
            RoutingSelectionVariant::GemmaV2 => nsg,
            RoutingSelectionVariant::GemmaV3 => 2 * nsg,
            _ => unreachable!(),
        };
    let pipeline = registry.get_pipeline(out.kernel, device.metal_device())?;
    annotate(
        enc,
        &[logits, scales],
        &[ids, weights, w.scores, w.ids, w.status],
    )?;
    super::encode_helpers::encode_threadgroups_with_args_and_shared(
        enc,
        pipeline,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Buffer(ids)),
            (2, KernelArg::Buffer(weights)),
            (3, KernelArg::Buffer(scales)),
            (4, KernelArg::Bytes(as_bytes(&p))),
            (5, KernelArg::Buffer(w.scores)),
            (6, KernelArg::Buffer(w.ids)),
            (7, KernelArg::Buffer(w.status)),
        ],
        &[(0, slots * 4)],
        MTLSize::new(1, 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    Ok(out)
}
/// DeepSeek-V4 scored layers ONLY: sqrt(softplus)+bias over all256 experts,
/// top6, then unbiased gathered weights normalized*1.5. Hash layers use the
/// existing hash API and actual checkpoint IDs; their score margin is N/A.
pub fn dispatch_deepseek_score_with_witness(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    bias: &MlxBuffer,
    ids: &MlxBuffer,
    weights: &MlxBuffer,
    rows: u32,
    witness: Option<SelectionWitness<'_>>,
) -> Result<RoutingSelectionDispatch> {
    let mut out = RoutingSelectionDispatch {
        kernel: "deepseek_moe_score_route_f32",
        variant: RoutingSelectionVariant::DeepSeekV4SqrtSoftplusBias,
        domain: SelectionDomain::SqrtSoftplusPlusBiasBeforeUnbiasedWeightNorm,
        witness_encoded: false,
    };
    let Some(w) = witness else {
        super::deepseek_moe_routing::dispatch_deepseek_moe_score_route(
            enc,
            registry,
            device,
            logits,
            bias,
            ids,
            weights,
            rows as usize,
        )?;
        return Ok(out);
    };
    buffer(logits, DType::F32, elements(rows, 256)?, false)?;
    buffer(bias, DType::F32, 256, false)?;
    validate(w, rows, ids, weights, 6, DType::I32, &[logits, bias])?;
    out.kernel = "deepseek_moe_score_selection_witness_f32";
    out.witness_encoded = true;
    let p = DeepSeekParams { rows, vocab: 0 };
    let pipeline = registry.get_pipeline(out.kernel, device.metal_device())?;
    annotate(
        enc,
        &[logits, bias],
        &[ids, weights, w.scores, w.ids, w.status],
    )?;
    enc.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&p))),
            (1, KernelArg::Buffer(logits)),
            (2, KernelArg::Buffer(bias)),
            (3, KernelArg::Buffer(ids)),
            (4, KernelArg::Buffer(weights)),
            (5, KernelArg::Buffer(w.scores)),
            (6, KernelArg::Buffer(w.ids)),
            (7, KernelArg::Buffer(w.status)),
        ],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(out)
}
#[cfg(test)]
#[path = "routing_selection_witness/tests.rs"]
mod tests;

mod batch;
pub use batch::dispatch_gemma_batch_with_witness;
