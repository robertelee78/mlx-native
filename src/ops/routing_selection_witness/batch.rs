//! Optional witness from the real batch V1/V3 producer, never scalar replay.
use super::*;
fn dimensions(rows:u32,experts:u32,k:u32)->Result<()> {
    if rows==0 || experts<32 || !experts.is_power_of_two() || experts>256 || k==0 || k>64 || k>experts {
        return Err(invalid("Gemma batch witness dimensions unsupported"));
    }
    elements(rows,experts)?;elements(rows,k)?;elements(rows,2)?;Ok(())
}
/// Gemma prefill batch selection. The returned full kernel name distinguishes
/// batch V1/V3 from scalar variants. `None` retains the exact ordinary batch API.
/// No allocation, submission, wait, scalar replay or CPU softmax occurs here.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemma_batch_with_witness(
    enc: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    logits: &MlxBuffer, ids: &MlxBuffer, weights: &MlxBuffer, scales: &MlxBuffer,
    experts: u32, k: u32, rows: u32, witness: Option<SelectionWitness<'_>>,
) -> Result<RoutingSelectionDispatch> {
    let kernel=super::super::fused_norm_add::selected_fused_moe_routing_batch_f32_kernel();
    let (variant,diagnostic)=match kernel {
        "fused_moe_routing_batch_f32" => (RoutingSelectionVariant::GemmaV1,
            "fused_moe_routing_batch_f32_selection_witness"),
        "fused_moe_routing_batch_f32_v3" => (RoutingSelectionVariant::GemmaV3,
            "fused_moe_routing_batch_f32_v3_selection_witness"),
        _ => return Err(invalid("unrecognized actual Gemma batch routing selector")),
    };
    let mut out=RoutingSelectionDispatch {kernel,variant,
        domain:SelectionDomain::SoftmaxBeforeTopKRenorm,witness_encoded:false};
    let Some(w)=witness else {
        super::super::fused_norm_add::dispatch_fused_moe_routing_batch_f32(enc,registry,
            device.metal_device(),logits,ids,weights,scales,experts,k,rows)?;
        return Ok(out);
    };
    dimensions(rows,experts,k)?;
    buffer(logits,DType::F32,elements(rows,experts)?,false)?;
    buffer(scales,DType::F32,experts as usize,false)?;
    validate(w,rows,ids,weights,k,DType::U32,&[logits,scales])?;
    out.kernel=diagnostic;out.witness_encoded=true;
    let p=GemmaParams {experts,k};
    let tg=64.min(experts.next_power_of_two()) as u64;
    let nsg=(tg+31)/32;
    let slots=2*experts as u64+if variant==RoutingSelectionVariant::GemmaV3 {2*nsg}else{tg};
    let pipeline=registry.get_pipeline(out.kernel,device.metal_device())?;
    annotate(enc,&[logits,scales],&[ids,weights,w.scores,w.ids,w.status])?;
    super::super::encode_helpers::encode_threadgroups_with_args_and_shared(enc,pipeline,&[
        (0,KernelArg::Buffer(logits)),(1,KernelArg::Buffer(ids)),
        (2,KernelArg::Buffer(weights)),(3,KernelArg::Buffer(scales)),
        (4,KernelArg::Bytes(as_bytes(&p))),(5,KernelArg::Buffer(w.scores)),
        (6,KernelArg::Buffer(w.ids)),(7,KernelArg::Buffer(w.status))],
        &[(0,slots*4)],MTLSize::new(rows as u64,1,1),MTLSize::new(tg,1,1));
    Ok(out)
}

#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn actual_batch_dimension_admission_is_bounded_before_buffers() {
        for rows in [1,4,128,512] {assert!(dimensions(rows,128,8).is_ok());}
        assert!(dimensions(4,64,64).is_ok());
        for (rows,e,k) in [(0,128,8),(1,0,8),(1,129,8),(1,512,8),(1,128,0),(1,32,33),(1,128,65),(u32::MAX,128,8)] {
            assert!(dimensions(rows,e,k).is_err());
        }
    }
}
