use super::*;
use crate::encoder::DispatchKind;

fn timing(
    route: DenseBf16Route,
    wall: (f64, f64, f64),
    gpu: (f64, f64, f64),
) -> DenseBf16RouteTiming {
    let pipeline_label = route.kernel_name().to_string();
    DenseBf16RouteTiming {
        route,
        wall: DenseBf16TimingDistribution {
            p25_us: wall.0,
            median_us: wall.1,
            p75_us: wall.2,
            samples: CALIBRATION_SAMPLES as u32,
        },
        gpu: DenseBf16TimingDistribution {
            p25_us: gpu.0,
            median_us: gpu.1,
            p75_us: gpu.2,
            samples: CALIBRATION_SAMPLES as u32,
        },
        encoded: EncodedKernelDispatch {
            pipeline_label: pipeline_label.clone(),
            dispatch_kind: DispatchKind::ThreadGroups,
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            threadgroup_memory: Vec::new(),
        },
        pipeline: KernelPipelineIdentity {
            schema_version: crate::kernel_registry::KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION,
            pipeline_label,
            kernel_name: route.kernel_name().to_string(),
            origin: crate::kernel_registry::KernelPipelineOrigin::RuntimeSource,
            runtime_source_sha256: Some("test".into()),
            embedded_metallib_sha256: None,
            precise_fp32_math: true,
            threadgroup_size_multiple_hint: true,
        },
    }
}

#[test]
fn selector_requires_stable_wall_win_without_contrary_gpu_signal() {
    let clear = [
        timing(DenseBf16Route::Row, (79.0, 80.0, 81.0), (69.0, 70.0, 71.0)),
        timing(
            DenseBf16Route::Tiled4,
            (49.0, 50.0, 51.0),
            (44.0, 45.0, 46.0),
        ),
        timing(
            DenseBf16Route::TensorV1,
            (99.0, 100.0, 101.0),
            (89.0, 90.0, 91.0),
        ),
    ];
    assert_eq!(
        select_route(DenseBf16Route::Row, &clear[..2]).expect("clear selection"),
        (
            DenseBf16Route::Tiled4,
            DenseBf16SelectionStatus::CalibratedWinner
        )
    );

    let contrary_gpu = [
        timing(
            DenseBf16Route::Tiled4,
            (49.0, 50.0, 51.0),
            (104.0, 105.0, 106.0),
        ),
        timing(
            DenseBf16Route::Row,
            (99.0, 100.0, 101.0),
            (89.0, 90.0, 91.0),
        ),
    ];
    assert_eq!(
        select_route(DenseBf16Route::Row, &contrary_gpu).expect("contrary GPU selection"),
        (
            DenseBf16Route::Row,
            DenseBf16SelectionStatus::NoStableWinner
        )
    );

    let overlapping_iqr = [
        timing(
            DenseBf16Route::Row,
            (95.0, 100.0, 105.0),
            (90.0, 95.0, 100.0),
        ),
        timing(
            DenseBf16Route::Tiled4,
            (90.0, 94.0, 101.0),
            (80.0, 85.0, 90.0),
        ),
    ];
    assert_eq!(
        select_route(DenseBf16Route::Row, &overlapping_iqr).expect("overlap selection"),
        (
            DenseBf16Route::Row,
            DenseBf16SelectionStatus::NoStableWinner
        )
    );
}

#[test]
fn output_proof_rejects_unwritten_cells_and_guard_damage() {
    let device = MlxDevice::new().expect("Metal device");
    let logical = 4usize;
    let mut output = device
        .alloc_buffer(
            (logical + OUTPUT_GUARD_ELEMENTS) * DType::F32.size_of(),
            DType::F32,
            vec![logical + OUTPUT_GUARD_ELEMENTS],
        )
        .expect("proof output");
    poison_output(&mut output).expect("poison output");
    assert!(verified_output_bits(&output, logical).is_err());

    output.as_mut_slice::<f32>().expect("write logical")[..logical]
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        verified_output_bits(&output, logical).expect("valid proof"),
        [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );

    output.as_mut_slice::<f32>().expect("damage guard")[logical] = 0.0;
    assert!(verified_output_bits(&output, logical).is_err());
}

#[test]
fn retryable_process_cells_are_evicted_by_identity() {
    let key = DenseBf16ProcessKey {
        build_fingerprint: "retry-test-build".into(),
        device_name: "retry-test-device".into(),
        device_registry_id: u64::MAX,
        pipeline_set_fingerprint: "retry-test-pipelines".into(),
        shape: DenseBf16Shape {
            m: 16,
            n: 17,
            k: 20,
            src0_batch: 1,
            src1_batch: 1,
        },
    };
    let failed = process_cell(key.clone()).expect("failed cell");
    failed
        .set(Err(DenseBf16CalibrationFailure {
            message: "transient".into(),
        }))
        .expect("initialize failed cell");
    evict_process_cell_if_same(&key, &failed).expect("evict failed cell");
    let retry = process_cell(key.clone()).expect("retry cell");
    assert!(!Arc::ptr_eq(&failed, &retry));
    evict_process_cell_if_same(&key, &retry).expect("clean retry cell");
}

#[test]
fn expected_geometry_covers_all_exact_routes() {
    let shape = DenseBf16Shape {
        m: 5,
        n: 513,
        k: 516,
        src0_batch: 2,
        src1_batch: 8,
    };
    let row = expected_dispatch(DenseBf16Route::Row, shape);
    assert_eq!(row.grid, [257, 5, 8]);
    assert_eq!(row.threads_per_threadgroup, [32, 4, 1]);
    assert_eq!(row.threadgroup_memory, vec![(0, 256)]);

    let tiled = expected_dispatch(DenseBf16Route::Tiled4, shape);
    assert_eq!(tiled.grid, [257, 2, 8]);
    assert_eq!(tiled.threads_per_threadgroup, [32, 4, 1]);
    assert_eq!(tiled.threadgroup_memory, vec![(0, 1024)]);

    let tensor = expected_dispatch(DenseBf16Route::TensorV1, shape);
    assert_eq!(tensor.grid, [1, 9, 8]);
    assert_eq!(tensor.threads_per_threadgroup, [128, 1, 1]);
    assert_eq!(tensor.threadgroup_memory, vec![(0, 8192)]);
    let simdgroup = expected_dispatch(DenseBf16Route::Simdgroup, shape);
    assert_ne!(tensor.pipeline_label, simdgroup.pipeline_label);
    assert_eq!(tensor.grid, simdgroup.grid);
    assert_eq!(
        tensor.threads_per_threadgroup,
        simdgroup.threads_per_threadgroup
    );
    assert_eq!(tensor.threadgroup_memory, simdgroup.threadgroup_memory);
}
