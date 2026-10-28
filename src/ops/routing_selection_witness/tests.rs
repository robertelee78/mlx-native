use super::*;
#[test]
fn actual_decode_never_turns_missing_or_nonfinite_margin_into_zero() {
    assert!(matches!(
        decode(1, [1, 2], [0.5, 0.5], 4, 2).unwrap(),
        SelectionBoundary::Scored { .. }
    ));
    assert!(matches!(
        decode(2, [3, u32::MAX], [0.25, f32::NAN], 4, 4).unwrap(),
        SelectionBoundary::AllExpertsSelected { .. }
    ));
    assert_eq!(
        decode(3, [u32::MAX; 2], [f32::NAN; 2], 256, 6).unwrap(),
        SelectionBoundary::Invalid
    );
    for (status, ids, s, e, k) in [
        (0, [0, 1], [1.0, 0.0], 4, 2),
        (1, [0, 0], [1.0, 0.0], 4, 2),
        (1, [0, 4], [1.0, 0.0], 4, 2),
        (1, [0, 1], [f32::INFINITY, 0.0], 4, 2),
        (1, [0, 1], [0.0, 1.0], 4, 2),
        (2, [0, u32::MAX], [1.0, f32::NAN], 4, 2),
        (1, [0, 1], [1.0, 0.0], 4, 4),
    ] {
        assert!(decode(status, ids, s, e, k).is_err());
    }
}
#[test]
fn diagnostic_parameter_layouts_match_existing_kernel_abis() {
    assert_eq!(std::mem::size_of::<QwenParams>(), 16);
    assert_eq!(std::mem::size_of::<GemmaParams>(), 8);
    assert_eq!(std::mem::size_of::<DeepSeekParams>(), 8);
    assert_eq!(elements(4, 256).unwrap(), 1024);
    assert!(elements(u32::MAX, 2).is_err());
}
#[test]
fn actual_view_range_arithmetic_and_alias_predicate_are_checked() {
    assert_eq!(physical_range(4096, 16, 8).unwrap(), (4112, 4120));
    assert!(physical_range(usize::MAX, 1, 0).is_err());
    assert!(physical_range(1, 0, usize::MAX).is_err());
    assert!(overlaps((16, 24), (20, 28)));
    assert!(!overlaps((16, 24), (24, 32)));
}
#[path = "fixture.rs"]
mod fixture;
use fixture::*;

#[test]
#[ignore = "actual bounded Metal witness parity; fresh process per Gemma V1/V2/V3 flags, no model"]
fn actual_selection_witness_preserves_ids_weights_and_domains() {
    let d = MlxDevice::new().expect("Metal required; no device-unavailable PASS");
    let mut registry = KernelRegistry::new();
    // No cumulative held fixtures: each case drops after an existing checked
    // drain. All inputs+outputs below 64KiB per case; no weights/model files.
    for (label, e, k, logits) in [
        ("ties", 256, 8, vec![0.0; 512]),
        (
            "near",
            256,
            8,
            (0..512)
                .map(|i| f32::from_bits(0x3f800000 + (i % 11) as u32))
                .collect(),
        ),
        (
            "underflow",
            256,
            64,
            (0..512)
                .map(|i| {
                    if i % 256 < 63 {
                        0.0
                    } else {
                        -1000.0 - (i % 256) as f32
                    }
                })
                .collect(),
        ),
        ("all", 8, 8, vec![0.0; 16]),
    ] {
        let l = input(&d, &logits);
        let a = Outputs::new(&d, 2, k as usize, DType::U32);
        let b = Outputs::new(&d, 2, k as usize, DType::U32);
        let mut enc = d.command_encoder().unwrap();
        super::super::moe_softmax_topk::dispatch_moe_softmax_topk(
            &mut enc,
            &mut registry,
            &d,
            &l,
            &a.ids.view,
            &a.weights.view,
            2,
            e,
            k,
        )
        .unwrap();
        if label == "near" {
            enc.start_capture();
        }
        let chosen = dispatch_qwen_with_witness(
            &mut enc,
            &mut registry,
            &d,
            &l,
            &b.ids.view,
            &b.weights.view,
            2,
            e,
            k,
            Some(b.witness()),
        )
        .unwrap();
        if label == "near" {
            replay(&mut enc, &[&l], &b);
        }
        let next = if k < 64 && k < e {
            Some(Outputs::new(&d, 2, (k + 1) as usize, DType::U32))
        } else {
            None
        };
        if let Some(n) = &next {
            super::super::moe_softmax_topk::dispatch_moe_softmax_topk(
                &mut enc,
                &mut registry,
                &d,
                &l,
                &n.ids.view,
                &n.weights.view,
                2,
                e,
                k + 1,
            )
            .unwrap();
        }
        enc.commit_and_wait().unwrap();
        parity(&a, &b);
        assert_selected_boundary(&b, 2, k, e);
        if let Some(n) = &next {
            let reference = n.ids.ids();
            let observed = b.boundary_ids.ids();
            for row in 0..2 {
                assert_eq!(
                    observed[row * 2 + 1],
                    reference[row * (k + 1) as usize + k as usize]
                );
            }
            n.check();
        }
        assert_eq!(chosen.kernel, "moe_softmax_topk_selection_witness_f32");
        if label == "underflow" {
            match read_after_checked_drain(b.witness(), 2, e, k).unwrap()[0] {
                SelectionBoundary::Scored {
                    kth_id,
                    next_id,
                    kth_score,
                    next_score,
                } => {
                    assert_eq!((kth_id, next_id), (63, 64));
                    assert_eq!((kth_score, next_score), (0.0, 0.0));
                }
                _ => panic!("underflow selection boundary absent"),
            }
        }
        println!("ROUTING_WITNESS_QWEN case={label} kernel={} rows=2 exact_ids=true exact_weight_bits=true",chosen.kernel);
    }
    let mut actual_gemma = Vec::new();
    for (label, k, values) in [
        ("ties", 8, vec![0.0; 128]),
        (
            "near",
            8,
            (0..128)
                .map(|i| f32::from_bits(0x3f800000 + (i % 11) as u32))
                .collect(),
        ),
        ("all", 64, vec![0.0; 64]),
    ] {
        let e = values.len();
        let l = input(&d, &values);
        let scales = input(
            &d,
            &(0..e).map(|i| 1.0 + i as f32 / 32.0).collect::<Vec<_>>(),
        );
        let a = Outputs::new(&d, 1, k, DType::U32);
        let b = Outputs::new(&d, 1, k, DType::U32);
        let mut enc = d.command_encoder().unwrap();
        super::super::fused_norm_add::dispatch_fused_moe_routing_f32(
            &mut enc,
            &mut registry,
            d.metal_device(),
            &l,
            &a.ids.view,
            &a.weights.view,
            &scales,
            e as u32,
            k as u32,
        )
        .unwrap();
        if label == "near" {
            enc.start_capture();
        }
        let chosen = dispatch_gemma_with_witness(
            &mut enc,
            &mut registry,
            &d,
            &l,
            &b.ids.view,
            &b.weights.view,
            &scales,
            e as u32,
            k as u32,
            Some(b.witness()),
        )
        .unwrap();
        if label == "near" {
            replay(&mut enc, &[&l, &scales], &b);
        }
        let next = if k < 64 {
            Some(Outputs::new(&d, 1, k + 1, DType::U32))
        } else {
            None
        };
        if let Some(n) = &next {
            super::super::fused_norm_add::dispatch_fused_moe_routing_f32(
                &mut enc,
                &mut registry,
                d.metal_device(),
                &l,
                &n.ids.view,
                &n.weights.view,
                &scales,
                e as u32,
                (k + 1) as u32,
            )
            .unwrap();
        }
        enc.commit_and_wait().unwrap();
        parity(&a, &b);
        assert_selected_boundary(&b, 1, k as u32, e as u32);
        if let Some(n) = &next {
            assert_eq!(b.boundary_ids.ids()[1], n.ids.ids()[k]);
            n.check();
        }
        if label != "near" {
            assert_eq!(
                b.scores.view.as_logical_slice::<f32>().unwrap()[0].to_bits(),
                (1.0f32 / e as f32).to_bits()
            );
        }
        actual_gemma.push(chosen.variant);
        println!("ROUTING_WITNESS_GEMMA case={label} variant={:?} kernel={} exact_ids=true exact_weight_bits=true",chosen.variant,chosen.kernel);
    }
    let rows = 2;
    let l = input(&d, &vec![0.0; rows * 256]);
    let bias = input(
        &d,
        &(0..256)
            .map(|i| if i < 6 { 6.0 - i as f32 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    let a = Outputs::new(&d, rows, 6, DType::I32);
    let b = Outputs::new(&d, rows, 6, DType::I32);
    let mut enc = d.command_encoder().unwrap();
    super::super::deepseek_moe_routing::dispatch_deepseek_moe_score_route(
        &mut enc,
        &mut registry,
        &d,
        &l,
        &bias,
        &a.ids.view,
        &a.weights.view,
        rows,
    )
    .unwrap();
    enc.start_capture();
    let chosen = dispatch_deepseek_score_with_witness(
        &mut enc,
        &mut registry,
        &d,
        &l,
        &bias,
        &b.ids.view,
        &b.weights.view,
        rows as u32,
        Some(b.witness()),
    )
    .unwrap();
    replay(&mut enc, &[&l, &bias], &b);
    enc.commit_and_wait().unwrap();
    parity(&a, &b);
    assert_selected_boundary(&b, rows as u32, 6, 256);
    assert_eq!(b.boundary_ids.ids(), vec![5, 6, 5, 6]);
    assert_eq!(
        chosen.domain,
        SelectionDomain::SqrtSoftplusPlusBiasBeforeUnbiasedWeightNorm
    );
    println!("ROUTING_WITNESS_DS rows=2 kernel={} biased_selection=true unbiased_weights=true exact_ids=true exact_weight_bits=true",chosen.kernel);
    // Invalid inputs must produce explicit Invalid, never a fabricated zero gap.
    for kind in [0, 1, 2, 3, 4, 5] {
        let mut values = vec![0.0; 256];
        if kind < 2 {
            values[0] = f32::NAN;
        }
        if kind == 4 {
            values[1] = f32::NAN;
        }
        let l = input(&d, &values);
        let mut scale_values = vec![
            if kind == 2 {
                f32::INFINITY
            } else if kind == 3 {
                f32::NAN
            } else {
                1.0
            };
            256
        ];
        if kind == 5 {
            scale_values[33] = f32::NAN;
        }
        let scales = input(&d, &scale_values);
        let o = Outputs::new(
            &d,
            1,
            6,
            if matches!(kind, 1 | 3 | 4 | 5) {
                DType::I32
            } else {
                DType::U32
            },
        );
        let mut enc = d.command_encoder().unwrap();
        match kind {
            0 => {
                dispatch_qwen_with_witness(
                    &mut enc,
                    &mut registry,
                    &d,
                    &l,
                    &o.ids.view,
                    &o.weights.view,
                    1,
                    256,
                    6,
                    Some(o.witness()),
                )
                .unwrap();
            }
            1 | 3 | 4 | 5 => {
                dispatch_deepseek_score_with_witness(
                    &mut enc,
                    &mut registry,
                    &d,
                    &l,
                    &scales,
                    &o.ids.view,
                    &o.weights.view,
                    1,
                    Some(o.witness()),
                )
                .unwrap();
            }
            _ => {
                dispatch_gemma_with_witness(
                    &mut enc,
                    &mut registry,
                    &d,
                    &l,
                    &o.ids.view,
                    &o.weights.view,
                    &scales,
                    256,
                    6,
                    Some(o.witness()),
                )
                .unwrap();
            }
        }
        enc.commit_and_wait().unwrap();
        assert_eq!(
            read_after_checked_drain(o.witness(), 1, 256, 6).unwrap(),
            vec![SelectionBoundary::Invalid]
        );
        o.check();
    }
    // Actual dispatcher refuses geometry/alias before encoding a diagnostic.
    let l = input(&d, &vec![0.0; 256]);
    let o = Outputs::new(&d, 1, 8, DType::U32);
    let mut enc = d.command_encoder().unwrap();
    assert!(dispatch_qwen_with_witness(
        &mut enc,
        &mut registry,
        &d,
        &l,
        &o.ids.view,
        &o.weights.view,
        0,
        256,
        8,
        Some(o.witness())
    )
    .is_err());
    let alias_scores = o.weights.view.slice_view(0, 2);
    let bad = SelectionWitness {
        scores: &alias_scores,
        ids: &o.boundary_ids.view,
        status: &o.status.view,
    };
    assert!(dispatch_qwen_with_witness(
        &mut enc,
        &mut registry,
        &d,
        &l,
        &o.ids.view,
        &o.weights.view,
        1,
        256,
        8,
        Some(bad)
    )
    .is_err());
    assert!(actual_gemma.iter().all(|v| *v == actual_gemma[0]));
    println!("ROUTING_SELECTION_WITNESS_COMPLETE qwen_cases=4 gemma_cases=3 ds_cases=1 invalid_cases=6 views_guarded=true gemma_variant={:?} model_parity=false",actual_gemma[0]);
}
