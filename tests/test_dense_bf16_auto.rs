#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::dense_bf16_auto::{
    calibrate_dense_bf16_routes, dense_matmul_bf16_f32_auto, trace_dense_matmul_bf16_f32_auto,
    DenseBf16BaseShape, DenseBf16CalibrationCase, DenseBf16CalibrationLimits,
    DenseBf16DecisionSource, DenseBf16Route,
};
use mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params;
use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn calibration_rejects_impossible_shapes_before_scratch_allocation() {
    let device = MlxDevice::new().expect("Metal device");
    let weight = device
        .alloc_buffer(8, DType::BF16, vec![4])
        .expect("tiny weight");
    let reachable = [1u32];
    for shape in [
        DenseBf16BaseShape {
            n: i32::MAX as u32 + 1,
            k: 4,
            src0_batch: 1,
            src1_batch: 1,
        },
        DenseBf16BaseShape {
            n: 1,
            k: 3,
            src0_batch: 1,
            src1_batch: 1,
        },
    ] {
        let cases = [DenseBf16CalibrationCase {
            weight: &weight,
            shape,
            reachable_m: &reachable,
        }];
        let mut registry = KernelRegistry::new();
        assert!(calibrate_dense_bf16_routes(
            &mut registry,
            &device,
            99,
            DenseBf16CalibrationLimits {
                max_elapsed_ms: 100,
                max_shapes: 1,
            },
            &cases,
        )
        .is_err());
        assert_eq!(registry.cached_count(), 0);
    }
}

#[test]
fn calibration_freezes_routes_and_process_cache_reuses_without_submissions() {
    let device = MlxDevice::new().expect("Metal device");
    let shape = DenseBf16BaseShape {
        n: 257,
        k: 512,
        src0_batch: 1,
        src1_batch: 1,
    };
    let mut weight = device
        .alloc_buffer(
            (shape.n * shape.k * 2) as usize,
            DType::BF16,
            vec![shape.n as usize, shape.k as usize],
        )
        .expect("weight allocation");
    for (index, value) in weight
        .as_mut_slice::<u16>()
        .expect("map BF16 weight")
        .iter_mut()
        .enumerate()
    {
        *value = bf16::from_f32(((index * 37 % 251) as f32 - 125.0) / 509.0).to_bits();
    }
    let reachable_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let cases = [DenseBf16CalibrationCase {
        weight: &weight,
        shape,
        reachable_m: &reachable_m,
    }];
    let limits = DenseBf16CalibrationLimits {
        max_elapsed_ms: 30_000,
        max_shapes: 32,
    };

    let mut unfrozen_registry = KernelRegistry::new();
    let unfrozen_params = DenseMmBf16F32Params {
        m: 1,
        n: shape.n,
        k: shape.k,
        src0_batch: shape.src0_batch,
        src1_batch: shape.src1_batch,
    };
    let unfrozen_input = device
        .alloc_buffer((shape.k * 4) as usize, DType::F32, vec![shape.k as usize])
        .expect("unfrozen input allocation");
    let unfrozen_output = device
        .alloc_buffer((shape.n * 4) as usize, DType::F32, vec![shape.n as usize])
        .expect("unfrozen output allocation");
    let mut unfrozen_encoder = device.command_encoder().expect("unfrozen encoder");
    assert!(dense_matmul_bf16_f32_auto(
        &mut unfrozen_encoder,
        &mut unfrozen_registry,
        &device,
        &weight,
        &unfrozen_input,
        &unfrozen_output,
        &unfrozen_params,
    )
    .is_err());

    let mut registry_a = KernelRegistry::new();
    let (plan_a, receipt_a) =
        calibrate_dense_bf16_routes(&mut registry_a, &device, 1, limits, &cases)
            .expect("first calibration");
    assert_eq!(receipt_a.declared_shapes, 16);
    assert_eq!(receipt_a.process_cache_hits, 0);
    assert_eq!(receipt_a.calibrated_decisions, 16);
    assert_eq!(receipt_a.budget_fallback_decisions, 0);
    assert!(receipt_a.calibration_submissions > 0);
    assert!(receipt_a.decisions.iter().all(|decision| {
        !decision.process_cache_hit
            && matches!(decision.compatibility_route, DenseBf16Route::Row)
            && matches!(
                decision.selected_route,
                DenseBf16Route::Row | DenseBf16Route::Tiled4
            )
            && decision.incoherent_routes.is_empty()
            && !decision.timings.is_empty()
            && decision
                .timings
                .iter()
                .any(|timing| timing.route == decision.compatibility_route)
            && decision
                .timings
                .iter()
                .any(|timing| timing.route == decision.selected_route)
    }));
    assert_eq!(plan_a.decision_count(), 16);
    registry_a
        .freeze_dense_bf16_plan(&device, plan_a.clone())
        .expect("same-plan freeze is idempotent");

    for &m in &reachable_m {
        let params = DenseMmBf16F32Params {
            m,
            n: shape.n,
            k: shape.k,
            src0_batch: shape.src0_batch,
            src1_batch: shape.src1_batch,
        };
        let mut input = device
            .alloc_buffer(
                (m * shape.k * 4) as usize,
                DType::F32,
                vec![m as usize, shape.k as usize],
            )
            .expect("input allocation");
        for (index, value) in input
            .as_mut_slice::<f32>()
            .expect("map input")
            .iter_mut()
            .enumerate()
        {
            *value = ((index * 29 % 241) as f32 - 120.0) / 1003.0;
        }
        let output = device
            .alloc_buffer(
                (m * shape.n * 4) as usize,
                DType::F32,
                vec![m as usize, shape.n as usize],
            )
            .expect("output allocation");
        let mut encoder = device.command_encoder().expect("command encoder");
        let trace = trace_dense_matmul_bf16_f32_auto(
            &mut encoder,
            &mut registry_a,
            &device,
            &weight,
            &input,
            &output,
            &params,
        )
        .expect("auto trace");
        assert_eq!(trace.decision.source, DenseBf16DecisionSource::FrozenPlan);
        assert_eq!(trace.encoded.pipeline_label, trace.pipeline.pipeline_label);
        encoder.commit_and_wait().expect("auto completion");
        let actual = output.as_slice::<f32>().expect("read output");
        assert!(actual.iter().any(|value| value.abs() > 1e-6));
        for row in 0..m as usize {
            let mut row_input = device
                .alloc_buffer((shape.k * 4) as usize, DType::F32, vec![shape.k as usize])
                .expect("independent row input");
            row_input
                .as_mut_slice::<f32>()
                .expect("map independent row input")
                .copy_from_slice(
                    &input.as_slice::<f32>().expect("read batched input")
                        [row * shape.k as usize..(row + 1) * shape.k as usize],
                );
            let row_output = device
                .alloc_buffer((shape.n * 4) as usize, DType::F32, vec![shape.n as usize])
                .expect("independent row output");
            let row_params = DenseMmBf16F32Params { m: 1, ..params };
            let mut row_encoder = device.command_encoder().expect("independent row encoder");
            mlx_native::ops::dense_bf16_auto::dense_matmul_bf16_f32_forced(
                DenseBf16Route::Row,
                &mut row_encoder,
                &mut registry_a,
                &device,
                &weight,
                &row_input,
                &row_output,
                &row_params,
            )
            .expect("independent row dispatch");
            row_encoder
                .commit_and_wait()
                .expect("independent row completion");
            let expected = row_output.as_slice::<f32>().expect("read independent row");
            assert_eq!(
                &actual[row * shape.n as usize..(row + 1) * shape.n as usize],
                expected,
                "physical M={m} row {row} diverged from independent M=1"
            );
        }
    }

    assert!(registry_a
        .try_register_source(
            "hf2q_dense_gemv_bf16_f32_r1_4",
            include_str!("../src/shaders/dense_gemv_bf16.metal"),
        )
        .is_ok());
    assert!(registry_a
        .try_register_source(
            "hf2q_dense_gemv_bf16_f32_r1_4",
            "kernel void hf2q_dense_gemv_bf16_f32_r1_4() {}",
        )
        .is_err());

    let missing_m = 17u32;
    let missing_params = DenseMmBf16F32Params {
        m: missing_m,
        n: shape.n,
        k: shape.k,
        src0_batch: shape.src0_batch,
        src1_batch: shape.src1_batch,
    };
    let missing_input = device
        .alloc_buffer(
            (missing_m * shape.k * 4) as usize,
            DType::F32,
            vec![(missing_m * shape.k) as usize],
        )
        .expect("missing-shape input");
    let missing_output = device
        .alloc_buffer(
            (missing_m * shape.n * 4) as usize,
            DType::F32,
            vec![(missing_m * shape.n) as usize],
        )
        .expect("missing-shape output");
    let mut missing_encoder = device.command_encoder().expect("missing-shape encoder");
    let missing_decision = dense_matmul_bf16_f32_auto(
        &mut missing_encoder,
        &mut registry_a,
        &device,
        &weight,
        &missing_input,
        &missing_output,
        &missing_params,
    )
    .expect("missing-shape dispatch");
    assert!(matches!(
        missing_decision.route,
        DenseBf16Route::TensorV1 | DenseBf16Route::Simdgroup
    ));
    if std::env::var("MLX_NATIVE_DISABLE_METAL_TENSOR").as_deref() == Ok("1") {
        assert_eq!(missing_decision.route, DenseBf16Route::Simdgroup);
    }
    missing_encoder
        .commit_and_wait()
        .expect("missing-shape completion");

    let mut registry_b = KernelRegistry::new();
    let (plan_b, receipt_b) =
        calibrate_dense_bf16_routes(&mut registry_b, &device, 2, limits, &cases)
            .expect("process-cache reuse");
    assert_eq!(receipt_b.process_cache_hits, 16);
    assert_eq!(receipt_b.calibrated_decisions, 0);
    assert_eq!(receipt_b.calibration_submissions, 0);
    assert!(receipt_b
        .decisions
        .iter()
        .all(|decision| { decision.process_cache_hit && decision.calibration_submissions == 0 }));
    assert_ne!(plan_a.plan_id(), plan_b.plan_id());
    assert!(registry_a.freeze_dense_bf16_plan(&device, plan_b).is_err());
}

#[test]
fn non_vector_aligned_k_uses_simdgroup_for_calibrated_and_missing_shapes() {
    let device = MlxDevice::new().expect("Metal device");
    let shape = DenseBf16BaseShape {
        n: 65,
        k: 33,
        src0_batch: 1,
        src1_batch: 1,
    };
    let mut weight = device
        .alloc_buffer(
            (shape.n * shape.k * 2) as usize,
            DType::BF16,
            vec![shape.n as usize, shape.k as usize],
        )
        .expect("weight");
    for (index, value) in weight
        .as_mut_slice::<u16>()
        .expect("map weight")
        .iter_mut()
        .enumerate()
    {
        *value = bf16::from_f32(((index * 11 % 97) as f32 - 48.0) / 211.0).to_bits();
    }
    let reachable = [1u32];
    let cases = [DenseBf16CalibrationCase {
        weight: &weight,
        shape,
        reachable_m: &reachable,
    }];
    let mut registry = KernelRegistry::new();
    let (_, receipt) = calibrate_dense_bf16_routes(
        &mut registry,
        &device,
        10,
        DenseBf16CalibrationLimits {
            max_elapsed_ms: 30_000,
            max_shapes: 2,
        },
        &cases,
    )
    .expect("K=33 calibration");
    assert_eq!(
        receipt.decisions[0].selected_route,
        DenseBf16Route::Simdgroup
    );
    assert_eq!(
        receipt.decisions[0].unavailable_routes,
        vec![DenseBf16Route::TensorV1]
    );

    for m in [1u32, 17] {
        let params = DenseMmBf16F32Params {
            m,
            n: shape.n,
            k: shape.k,
            src0_batch: 1,
            src1_batch: 1,
        };
        let input = device
            .alloc_buffer(
                (m * shape.k * 4) as usize,
                DType::F32,
                vec![(m * shape.k) as usize],
            )
            .expect("input");
        let output = device
            .alloc_buffer(
                (m * shape.n * 4) as usize,
                DType::F32,
                vec![(m * shape.n) as usize],
            )
            .expect("output");
        let mut encoder = device.command_encoder().expect("encoder");
        let decision = dense_matmul_bf16_f32_auto(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &input,
            &output,
            &params,
        )
        .expect("K=33 auto dispatch");
        assert_eq!(decision.route, DenseBf16Route::Simdgroup);
        encoder.commit_and_wait().expect("K=33 completion");
    }
}

#[test]
fn budget_fallback_does_not_poison_later_model_activation() {
    let device = MlxDevice::new().expect("Metal device");
    let shape = DenseBf16BaseShape {
        n: 259,
        k: 508,
        src0_batch: 1,
        src1_batch: 1,
    };
    let weight = device
        .alloc_buffer(
            (shape.n * shape.k * 2) as usize,
            DType::BF16,
            vec![shape.n as usize, shape.k as usize],
        )
        .expect("weight");
    let reachable = [1u32];
    let cases = [DenseBf16CalibrationCase {
        weight: &weight,
        shape,
        reachable_m: &reachable,
    }];
    let mut short_registry = KernelRegistry::new();
    let (_, short_receipt) = calibrate_dense_bf16_routes(
        &mut short_registry,
        &device,
        20,
        DenseBf16CalibrationLimits {
            max_elapsed_ms: 1,
            max_shapes: 1,
        },
        &cases,
    )
    .expect("short-budget activation");
    assert_eq!(short_receipt.budget_fallback_decisions, 1);

    let mut retry_registry = KernelRegistry::new();
    let (_, retry_receipt) = calibrate_dense_bf16_routes(
        &mut retry_registry,
        &device,
        21,
        DenseBf16CalibrationLimits {
            max_elapsed_ms: 30_000,
            max_shapes: 1,
        },
        &cases,
    )
    .expect("later activation retry");
    assert_eq!(retry_receipt.process_cache_hits, 0);
    assert_eq!(retry_receipt.calibrated_decisions, 1);
    assert_eq!(retry_receipt.budget_fallback_decisions, 0);
}
