//! Actual bounded Metal proof of each cached batch producer; fresh process per V3 flag.
use super::super::*;
use super::fixture::*;
fn outputs(d:&MlxDevice,rows:usize,k:usize)->Outputs {
    let mut o=Outputs::new(d,rows,k,DType::U32);
    o.ids.view=o.ids.view.with_shape(vec![rows,k]).unwrap();
    o.weights.view=o.weights.view.with_shape(vec![rows,k]).unwrap();
    o.scores.view=o.scores.view.with_shape(vec![rows,2]).unwrap();
    o.boundary_ids.view=o.boundary_ids.view.with_shape(vec![rows,2]).unwrap();
    o
}
// Same actual graph seam as the existing scalar fixture, with an explicit
// inter-call WAW barrier AFTER capture extraction and before replay.
fn replay_batch(enc:&mut CommandEncoder,inputs:&[&MlxBuffer],o:&Outputs) {
    use crate::{encoder::CapturedNode,graph::ComputeGraph};
    let nodes=enc.take_capture().expect("actual batch witness capture");
    assert_eq!(nodes.len(),1);
    let reads=inputs.iter().map(|b|range(b).unwrap()).collect::<Vec<_>>();
    let writes=[&o.ids.view,&o.weights.view,&o.scores.view,&o.boundary_ids.view,&o.status.view]
        .iter().map(|b|range(b).unwrap()).collect::<Vec<_>>();
    match &nodes[0] {CapturedNode::Dispatch {reads:r,writes:w,..}=>{
        assert_eq!(*r,reads);assert_eq!(*w,writes);
    },_=>panic!("one actual batch producer required")}
    enc.memory_barrier();
    let mut graph=ComputeGraph::from_nodes(nodes);
    assert_eq!(graph.unannotated_dispatch_count(),0);graph.reorder();graph.encode_with_barriers(enc);
}
#[test]
#[ignore="bounded actual Metal batch routing; fresh processes V3=0/1; no model"]
fn actual_gemma_batch_selection_witness_matches_actual_batch_producer() {
    let d=MlxDevice::new().expect("actual Metal required; no no-device PASS");
    let mut registry=KernelRegistry::new();let mut actual=None;let mut cases=0;
    for rows in [1usize,4,128,512] {
        for case in ["ties","near","all"] {
            let (e,k)=if case=="all" {(64usize,64usize)}else{(128,8)};
            let values=(0..rows*e).map(|n|if case=="near" {
                ((n%e) as f32-64.0)*0.000001+(n/e) as f32*0.0000001
            }else{0.0}).collect::<Vec<_>>();
            let l=input(&d,&values).with_shape(vec![rows,e]).unwrap();
            let scales=input(&d,&(0..e).map(|i|1.0+i as f32/32.0).collect::<Vec<_>>());
            let old=outputs(&d,rows,k);let new=outputs(&d,rows,k);
            let mut enc=d.command_encoder().unwrap();
            let old_dispatch=dispatch_gemma_batch_with_witness(&mut enc,&mut registry,&d,
                &l,&old.ids.view,&old.weights.view,&scales,e as u32,k as u32,rows as u32,None).unwrap();
            assert!(!old_dispatch.witness_encoded);
            let capture=rows==4 && case=="near";
            if capture {enc.start_capture();}
            let selected=dispatch_gemma_batch_with_witness(&mut enc,&mut registry,&d,
                &l,&new.ids.view,&new.weights.view,&scales,e as u32,k as u32,rows as u32,Some(new.witness())).unwrap();
            if capture {replay_batch(&mut enc,&[&l,&scales],&new);}
            assert_eq!(selected.variant,old_dispatch.variant);
            assert_eq!(selected.domain,SelectionDomain::SoftmaxBeforeTopKRenorm);
            assert!(selected.witness_encoded);
            let next=if k<64 {Some(outputs(&d,rows,k+1))}else{None};
            if let Some(n)=&next {
                super::super::super::fused_norm_add::dispatch_fused_moe_routing_batch_f32(
                    &mut enc,&mut registry,d.metal_device(),&l,&n.ids.view,&n.weights.view,
                    &scales,e as u32,(k+1) as u32,rows as u32).unwrap();
            }
            enc.commit_and_wait().unwrap();parity(&old,&new);
            assert_selected_boundary(&new,rows as u32,k as u32,e as u32);
            let boundaries=read_after_checked_drain(new.witness(),rows as u32,e as u32,k as u32).unwrap();
            for (row,b) in boundaries.iter().enumerate() {
                match b {
                    SelectionBoundary::Scored {kth_score,next_score,next_id,..} => {
                        assert!(*kth_score>=*next_score && kth_score.is_finite() && next_score.is_finite());
                        assert_eq!(*next_id,next.as_ref().unwrap().ids.ids()[row*(k+1)+k]);
                        if case=="ties" {assert_eq!(kth_score.to_bits(),next_score.to_bits());}
                    }
                    SelectionBoundary::AllExpertsSelected {..} => assert_eq!(k,e),
                    SelectionBoundary::Invalid => panic!("valid batch row invalid"),
                }
            }
            if let Some(n)=next {n.check();}
            if let Some(v)=actual {assert_eq!(v,selected.variant);}else{actual=Some(selected.variant);}
            let expected=match selected.variant {
                RoutingSelectionVariant::GemmaV1=>"fused_moe_routing_batch_f32_selection_witness",
                RoutingSelectionVariant::GemmaV3=>"fused_moe_routing_batch_f32_v3_selection_witness",
                _=>panic!("no scalar V2 batch fallback"),
            };assert_eq!(selected.kernel,expected);
            cases+=1;println!("GEMMA_BATCH_WITNESS rows={rows} case={case} kernel={} exact_ids=true exact_weight_bits=true views_guarded=true",selected.kernel);
        }
    }
    for kind in 0..3 {
        let rows=4usize;let e=128usize;let k=8usize;
        let mut values=vec![0.0;rows*e];let mut scales=vec![1.0;e];
        if kind==0 {values[e+1]=f32::NAN;}else if kind==1 {values[3*e+33]=f32::NAN;}else{scales[33]=f32::NAN;}
        let l=input(&d,&values).with_shape(vec![rows,e]).unwrap();let scales=input(&d,&scales);
        let o=outputs(&d,rows,k);let mut enc=d.command_encoder().unwrap();
        dispatch_gemma_batch_with_witness(&mut enc,&mut registry,&d,&l,&o.ids.view,&o.weights.view,&scales,
            e as u32,k as u32,rows as u32,Some(o.witness())).unwrap();enc.commit_and_wait().unwrap();
        let states=read_after_checked_drain(o.witness(),rows as u32,e as u32,k as u32).unwrap();
        for (row,state) in states.iter().enumerate() {
            let invalid=kind==2 || row==if kind==0 {1}else{3};
            if invalid {
                assert_eq!(*state,SelectionBoundary::Invalid);
                assert!(o.scores.bits()[row*2..row*2+2].iter().all(|&x|f32::from_bits(x).is_nan() && x!=0x7fc12345));
                assert!(o.boundary_ids.ids()[row*2..row*2+2].iter().all(|&x|x==u32::MAX));
                assert!(o.ids.ids()[row*k..(row+1)*k].iter().all(|&x|x==u32::MAX));
                assert!(o.weights.bits()[row*k..(row+1)*k].iter().all(|&x|f32::from_bits(x).is_nan() && x!=0x7fc12345));
            }else {assert!(matches!(state,SelectionBoundary::Scored {..}));}
        }
        o.check();println!("GEMMA_BATCH_WITNESS_INVALID case={kind} rows=4 nonzero_lane_and_row=true views_guarded=true");
    }
    let l=input(&d,&vec![0.0;4*128]).with_shape(vec![4,128]).unwrap();let scales=input(&d,&vec![1.0;128]);
    let o=outputs(&d,4,8);let mut enc=d.command_encoder().unwrap();
    assert!(dispatch_gemma_batch_with_witness(&mut enc,&mut registry,&d,&l,&o.ids.view,&o.weights.view,
        &scales,128,8,0,Some(o.witness())).is_err());
    let alias=o.weights.view.slice_view(0,8).with_shape(vec![4,2]).unwrap();
    let bad=SelectionWitness {scores:&alias,ids:&o.boundary_ids.view,status:&o.status.view};
    assert!(dispatch_gemma_batch_with_witness(&mut enc,&mut registry,&d,&l,&o.ids.view,&o.weights.view,
        &scales,128,8,4,Some(bad)).is_err());
    println!("GEMMA_BATCH_SELECTION_WITNESS_COMPLETE cases={cases} invalid_cases=3 variant={:?} guards=true model_parity=false",actual.unwrap());
}
