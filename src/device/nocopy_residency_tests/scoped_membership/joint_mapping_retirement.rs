//! Separate finite joint-retirement case; historical single-parent result unchanged.
use super::{
    class, cpu_positive, gpu_read, host_page_size, membership, query, Fixture, SecondFile,
    FILE_BYTES, LIMIT, LIVE_METADATA_CAP, LOG_CAP, PAGE, PAGES, USERDATA, WIRED,
};
use crate::{DType, MlxBuffer, MlxDevice};
use memmap2::Mmap;
use serde::Serialize;
use serde_json::json;
use std::{
    sync::{Arc, Weak},
    time::Instant,
};
#[path = "mapping_retirement/region.rs"]
mod region;
#[path = "joint_mapping_retirement/state.rs"]
mod state;
use state::*;
fn device_identity(device: &MlxDevice) -> [u64; 3] {
    use metal::foreign_types::{ForeignType, ForeignTypeRef};
    [
        device.registry_id(),
        device.metal_device().as_ptr() as usize as u64,
        device.metal_queue().as_ptr() as usize as u64,
    ]
}
fn output_identity(output: &MlxBuffer) -> [u64; 4] {
    [
        output.metal_buffer().as_ref() as *const metal::BufferRef as usize as u64,
        output.contents_ptr() as usize as u64,
        output.byte_len() as u64,
        output.data_byte_len() as u64,
    ]
}
#[test]
#[ignore = "fresh20s supervised Metal process,<=256MiB requested data; no model credit"]
fn joint_original_backing_retirement_with_live_device_output() {
    assert_eq!(std::env::var("HF2Q_ENCODER_SESSION").as_deref(), Ok("1"));
    assert_eq!(host_page_size().unwrap(), PAGE);
    assert!(USERDATA <= LIMIT);
    assert!(LIVE_METADATA_CAP >= 8 * 1024 * 1024);
    assert!(
        !std::env::vars().any(
            |(k, _)| (k.starts_with("HF2Q_") && k != "HF2Q_ENCODER_SESSION")
                || k.starts_with("MLX_")
                || k.starts_with("MTL_")
                || k.starts_with("METAL_")
        ),
        "ambient activation"
    );
    cpu_positive();
    let mut first = Fixture::new();
    let mut second = SecondFile::new(&first);
    let mut a0 = Some(first.mapping());
    let a1 = first.mapping();
    let mut b0 = Some(second.mapping());
    let b1 = second.mapping();
    let weak_a = Arc::downgrade(a0.as_ref().unwrap());
    let weak_b = Arc::downgrade(b0.as_ref().unwrap());
    let a_base = a0.as_ref().unwrap().as_ptr() as usize;
    let b_base = b0.as_ref().unwrap().as_ptr() as usize;
    assert_ne!(a_base, a1.as_ptr() as usize);
    assert_ne!(b_base, b1.as_ptr() as usize);
    assert_ne!(first.file_id, second.id);
    println!(
        "JOINT_MAPPING_RETIREMENT_BEGIN {}",
        json!({"schema":"mlx.native.joint-mapping-retirement.v1","base":"f14926ed4b483fb92e63c5e524e2360f0d2db35e",
        "unique_file_payload_bytes":2*FILE_BYTES,"mapped_root_bytes":2*FILE_BYTES,"alias_virtual_bytes_nonadditive":4*FILE_BYTES,
        "userdata_bound":USERDATA,"userdata_limit":LIMIT,"log_cap":LOG_CAP,"metadata_cap":LIVE_METADATA_CAP,
        "driver_overhead":"Unknown/outside requested data","page_bytes":PAGE,"files":[first.dir.join("owned.bin"),second.path],
        "file_ids":[first.file_id,second.id],"A0_saved_address":a_base,"B0_saved_address":b_base,
        "default_heartbeat_source_seconds":180,"old_completed_CB_retirement":"Unknown","model_credit_bytes":0})
    );
    let mut observer = Observer {
        origin: Instant::now(),
        last: 0,
        phases: 0,
        queries: 0,
        regions: 0,
        bytes: 0,
    };
    let mut phases = Vec::new();
    let baseline_a = region::observe("A0_live_baseline", a_base, FILE_BYTES, &observer.origin);
    let baseline_b = region::observe("B0_live_baseline", b_base, FILE_BYTES, &observer.origin);
    let baseline = baseline_a.kind == region::Kind::LiveComplete
        && baseline_b.kind == region::Kind::LiveComplete;
    phases.push(observer.phase(
        "P0_before_device",
        [a0.as_deref(), Some(&a1), b0.as_deref(), Some(&b1)],
        None,
        &weak_a,
        &weak_b,
        vec![baseline_a, baseline_b],
    ));
    let device = MlxDevice::new().expect("actual Metal device required; no skip");
    assert!(device.residency_sets_enabled());
    let a = device
        .map_file_buffer(
            Arc::clone(a0.as_ref().unwrap()),
            0,
            FILE_BYTES,
            DType::F32,
            vec![FILE_BYTES / 4],
        )
        .unwrap();
    let b = device
        .map_file_buffer(
            Arc::clone(b0.as_ref().unwrap()),
            0,
            FILE_BYTES,
            DType::F32,
            vec![FILE_BYTES / 4],
        )
        .unwrap();
    println!(
        "JOINT_MAPPING_RETIREMENT_RESOURCES {}",
        json!({"classes":[class(&a),class(&b)],"device":device.metal_device().name(),
        "A_resource":a.metal_buffer().as_ref()as *const metal::BufferRef as usize,
        "B_resource":b.metal_buffer().as_ref()as *const metal::BufferRef as usize,"source_backings":["A0","B0"]})
    );
    let out = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    membership(&device, &a, &b, [true, true]);
    gpu_read(&device, &a, &out, 0);
    gpu_read(&device, &b, &out, 97);
    assert!(!out.is_file_backed() && out.is_cpu_writable());
    assert_eq!(
        out.metal_buffer().as_ref().storage_mode(),
        metal::MTLStorageMode::Shared
    );
    assert_eq!(
        (out.byte_len(), out.data_byte_len(), out.byte_offset()),
        (64, 64, 0)
    );
    assert_eq!(out.shape(), [16]);
    assert_eq!(out.dtype(), DType::F32);
    let output_bytes = out.as_logical_slice::<u8>().unwrap().to_vec();
    assert_eq!(output_bytes.len(), 64);
    let device_id = device_identity(&device);
    let output_id = output_identity(&out);

    phases.push(observer.phase(
        "P1_both_GPU_checked",
        [a0.as_deref(), Some(&a1), b0.as_deref(), Some(&b1)],
        Some(&device),
        &weak_a,
        &weak_b,
        vec![],
    ));
    membership(&device, &a, &b, [false, true]);
    phases.push(observer.phase(
        "P2_A_demoted_owners_alive",
        [a0.as_deref(), Some(&a1), b0.as_deref(), Some(&b1)],
        Some(&device),
        &weak_a,
        &weak_b,
        vec![],
    ));
    objc::rc::autoreleasepool(|| drop(a));
    let live_a = region::observe(
        "A0_caller_owned_after_Metal_drop",
        a_base,
        FILE_BYTES,
        &observer.origin,
    );
    // A0 is still strongly owned here; an absent region contradicts lineage.
    let live_a_absent = live_a.kind == region::Kind::Absent;
    phases.push(observer.phase(
        "P3_A_Metal_wrapper_dropped_A0_alive",
        [a0.as_deref(), Some(&a1), b0.as_deref(), Some(&b1)],
        Some(&device),
        &weak_a,
        &weak_b,
        vec![live_a],
    ));
    assert!(!live_a_absent, "live A0 unexpectedly absent");
    objc::rc::autoreleasepool(|| drop(a0.take()));
    let region_a4 = region::observe(
        "A0_after_caller_mapping_drop",
        a_base,
        FILE_BYTES,
        &observer.origin,
    );
    let absent4 = region_a4.kind == region::Kind::Absent;
    phases.push(observer.phase(
        "P4_A0_dropped_B_live",
        [None, Some(&a1), b0.as_deref(), Some(&b1)],
        Some(&device),
        &weak_a,
        &weak_b,
        vec![region_a4],
    ));
    let expired4 = phases.last().unwrap().expired[0];
    assert!(
        owner_region_consistent(expired4, absent4),
        "Weak-alive A0 unexpectedly absent"
    );
    // Last sidecar mutation/GPU commit BEFORE joint retirement. Fresh session.
    {
        let mut session = device
            .encoder_session()
            .unwrap()
            .expect("session capability");
        assert!(session.remove_from_residency_set(&b));
        session.commit_and_wait().unwrap();
    }
    objc::rc::autoreleasepool(|| {
        drop(b);
        drop(b0.take());
    });
    let a5 = region::observe(
        "A0_joint_retired_device_live",
        a_base,
        FILE_BYTES,
        &observer.origin,
    );
    let b5 = region::observe(
        "B0_joint_retired_device_live",
        b_base,
        FILE_BYTES,
        &observer.origin,
    );
    let absent5 = [
        a5.kind == region::Kind::Absent,
        b5.kind == region::Kind::Absent,
    ];
    let join = OutputJoin {
        same_device: device_identity(&device) == device_id,
        same_output: output_identity(&out) == output_id
            && out.byte_offset() == 0
            && out.shape() == [16],
        ordinary_shared: !out.is_file_backed()
            && out.is_cpu_writable()
            && out.metal_buffer().as_ref().storage_mode() == metal::MTLStorageMode::Shared,
        contents_match: out.as_logical_slice::<u8>().unwrap() == output_bytes.as_slice(),
    };
    println!(
        "JOINT_MAPPING_RETIREMENT_LIVE_OUTPUT {}",
        json!({"phase":"P5","device_before":device_id,"device_now":device_identity(&device),"output_before":output_id,"output_now":output_identity(&out),"output_class":class(&out),"source":"MlxDevice::alloc_buffer Shared, not map_file_buffer","is_file_backed":out.is_file_backed(),"cpu_writable":out.is_cpu_writable(),"storage_mode":out.metal_buffer().as_ref().storage_mode() as u64,"logical_bytes":out.data_byte_len(),"byte_offset":out.byte_offset(),"shape":out.shape(),"dtype":format!("{:?}",out.dtype()),"output_bytes_before":output_bytes,"output_bytes_now":out.as_logical_slice::<u8>().unwrap()})
    );
    assert!(
        join.same_device && join.same_output && join.ordinary_shared && join.contents_match,
        "actual live device/output changed"
    );
    phases.push(observer.phase(
        "P5_both_originals_retired_device_output_live",
        [None, Some(&a1), None, Some(&b1)],
        Some(&device),
        &weak_a,
        &weak_b,
        vec![a5, b5],
    ));
    for index in 0..2 {
        assert!(
            owner_region_consistent(phases[5].expired[index], absent5[index]),
            "Weak-live original unexpectedly absent"
        );
    }
    assert_eq!(
        device_identity(&device),
        device_id,
        "same device/queue after P5 observation"
    );
    assert_eq!(
        output_identity(&out),
        output_id,
        "same output after P5 observation"
    );
    assert_eq!(
        out.as_logical_slice::<u8>().unwrap(),
        output_bytes.as_slice(),
        "all64 known output bytes remain unchanged after P5"
    );

    // Only now may the SAME ordinary output and device/queue retire.
    objc::rc::autoreleasepool(|| {
        drop(out);
        drop(device);
    });
    let a6 = region::observe(
        "A0_after_device_output_drop",
        a_base,
        FILE_BYTES,
        &observer.origin,
    );
    let b6 = region::observe(
        "B0_after_device_output_drop",
        b_base,
        FILE_BYTES,
        &observer.origin,
    );
    let absent6 = [
        a6.kind == region::Kind::Absent,
        b6.kind == region::Kind::Absent,
    ];
    phases.push(observer.phase(
        "P6_device_output_retired_observers_only",
        [None, Some(&a1), None, Some(&b1)],
        None,
        &weak_a,
        &weak_b,
        vec![a6, b6],
    ));
    for index in 0..2 {
        assert!(
            owner_region_consistent(phases[6].expired[index], absent6[index]),
            "Weak-live original unexpectedly absent"
        );
    }
    let after_device_zero = phases[6].expired == [true; 2]
        && absent6 == [true; 2]
        && phases[6].wired[1] == Some(0)
        && phases[6].wired[3] == Some(0);
    drop(a1);
    drop(b1);
    assert!(second.cleanup(), "second file cleanup");
    assert!(first.cleanup(), "owned fixture cleanup");
    assert_eq!(
        (observer.phases, observer.queries, observer.regions),
        (7, 23, 8)
    );
    let classification = classify(&phases, absent5, join, baseline, true);
    println!(
        "JOINT_MAPPING_RETIREMENT_COMPLETE {}",
        json!({"protocol":"PASS","classification":classification,"phases":7,"live_alias_queries":23,"dispositions":23*PAGES,"region_calls":8,"cleanup":true,"sentinel_GPU_reads":2,"checked_drain":true,"P4_A_expired":expired4,"P4_A_absent":absent4,"P5_expired":phases[5].expired,"P5_absent":absent5,"P6_expired":phases[6].expired,"P6_absent":absent6,"secondary_after_device_both_zero":after_device_zero,"primary_not_promoted_by_P6":true,"elapsed_ns":observer.origin.elapsed().as_nanos(),"model_credit_bytes":0,"physical_admission":false,"whole_model_qualified":false})
    );
}
