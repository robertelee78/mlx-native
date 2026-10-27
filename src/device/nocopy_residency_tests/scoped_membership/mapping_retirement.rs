//! Distinguish original NoCopy mapping lifetime from a surviving observer.
//! One fresh process; no sleep/poll/retry or source/physical model credit.
use super::{
    class, cpu_positive, gpu_read, host_page_size, membership, query, Fixture, SecondFile,
    FILE_BYTES, LIMIT, LIVE_METADATA_CAP, LOG_CAP, PAGE, PAGES, USERDATA,
};
use crate::{DType, MlxDevice};
use memmap2::Mmap;
use serde::Serialize;
use serde_json::json;
use std::{
    sync::{Arc, Weak},
    time::Instant,
};
mod region;
#[derive(Clone, Copy, Debug)]
struct Summary {
    wired: [Option<usize>; 4],
    agree_a: bool,
    agree_b: bool,
    expired: [bool; 2],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Classification {
    ObserverExclusionObservedAfterMappingRetirement,
    MappingRetiredObserverStillWired,
    ObserverExclusionOnlyAfterDeviceRetirement,
    RetirementUnknown,
}
fn owner_region_consistent(expired: bool, absent: bool) -> bool {
    expired || !absent
}
fn classify(
    phases: &[Summary],
    weak: &[bool; 3],
    absent: &[bool; 3],
    baseline: bool,
    cleanup: bool,
) -> Classification {
    if phases.len() != 7 || !baseline || !cleanup {
        return Classification::RetirementUnknown;
    }
    let prior = phases[1].agree_a
        && phases[1].agree_b
        && phases[1].wired[0].is_some_and(|n| n > 0)
        && phases[1].wired[2].is_some_and(|n| n > 0);
    if !prior || !phases[..4].iter().all(|p| p.agree_a && p.agree_b) {
        return Classification::RetirementUnknown;
    }
    if weak[0]
        && absent[0]
        && phases[4].wired[1] == Some(0)
        && phases[4].agree_b
        && phases[4].wired[2].is_some_and(|n| n > 0)
    {
        return Classification::ObserverExclusionObservedAfterMappingRetirement;
    }
    if (0..3).any(|i| weak[i] && absent[i] && phases[i + 4].wired[1].is_some_and(|n| n > 0)) {
        return Classification::MappingRetiredObserverStillWired;
    }
    if (1..3).any(|i| weak[i] && absent[i] && phases[i + 4].wired[1] == Some(0)) {
        return Classification::ObserverExclusionOnlyAfterDeviceRetirement;
    }
    Classification::RetirementUnknown
}
struct Observer {
    origin: Instant,
    last: u128,
    phases: usize,
    queries: usize,
    regions: usize,
    bytes: usize,
}
impl Observer {
    fn phase(
        &mut self,
        name: &'static str,
        aliases: [Option<&Mmap>; 4],
        device: Option<&MlxDevice>,
        weak_a: &Weak<Mmap>,
        weak_b: &Weak<Mmap>,
        regions: Vec<region::Record>,
    ) -> Summary {
        let begin = self.origin.elapsed().as_nanos();
        assert!(begin >= self.last);
        let mut vectors: [Option<Vec<i32>>; 4] = std::array::from_fn(|_| None);
        let mut identities = Vec::new();
        let mut wired = [None; 4];
        for (index, map) in aliases.iter().enumerate() {
            if let Some(map) = map {
                let bits = query(map.as_ptr(), FILE_BYTES);
                assert_eq!(bits.len(), PAGES);
                wired[index] = Some(
                    bits.iter()
                        .filter(|b| **b & super::super::WIRED != 0)
                        .count(),
                );
                vectors[index] = Some(bits);
                self.queries += 1;
                identities.push(json!({"alias":["A0","A1","B0","B1"][index],"base":map.as_ptr()as usize,"bytes":map.len()}));
            }
        }
        let mut summary = Summary {
            expired: [false; 2],
            wired,
            agree_a: vectors[0]
                .as_ref()
                .zip(vectors[1].as_ref())
                .is_some_and(|(a, b)| a == b),
            agree_b: vectors[2]
                .as_ref()
                .zip(vectors[3].as_ref())
                .is_some_and(|(a, b)| a == b),
        };
        self.regions += regions.len();
        let strong_a = weak_a.strong_count();
        let strong_b = weak_b.strong_count();
        summary.expired = [strong_a == 0, strong_b == 0];
        let actual_device = device.map(|d| d.metal_device().name().to_owned());
        let current_allocated = device.map(|d| d.metal_device().current_allocated_size());
        let commits = crate::residency::residency_commit_call_count_for_test();
        let end = self.origin.elapsed().as_nanos();
        self.last = end;
        let protocol = regions.iter().all(|r| r.protocol_valid);
        let record = json!({"phase_index":self.phases,"phase":name,"begin_ns":begin,"end_ns":end,"clock":"Instant origin ns",
            "live_aliases":identities,"a0":vectors[0],"a1":vectors[1],"b0":vectors[2],"b1":vectors[3],
            "wired_counts":wired,"a0_strong_count":strong_a,"b0_strong_count":strong_b,"a0_weak_expired":strong_a==0,"b0_weak_expired":strong_b==0,
            "region_observations":regions,"region_protocol_valid":protocol,
            "actual_device":actual_device,
            "metal_current_allocated_size":current_allocated,
            "residency_commits_observed":commits,
            "model_credit_bytes":0,"physical_admission":false});
        let text = serde_json::to_string(&record).unwrap();
        assert!(text.len() <= 1024 * 1024);
        self.bytes = self.bytes.checked_add(text.len()).unwrap();
        assert!(self.bytes <= LOG_CAP);
        println!("MAPPING_RETIREMENT_PHASE {text}");
        self.phases += 1;
        assert!(
            protocol,
            "invalid region ABI/API/port cleanup; no qualification"
        );
        summary
    }
}
#[test]
fn mapping_retirement_classification_requires_full_live_device_join() {
    assert!(!owner_region_consistent(false, true));
    assert!(owner_region_consistent(true, true));
    assert!(owner_region_consistent(false, false));
    let live = Summary {
        wired: [Some(3); 4],
        agree_a: true,
        agree_b: true,
        expired: [false; 2],
    };
    let mut phases = vec![live; 7];
    phases[4].wired = [None, Some(0), Some(3), Some(3)];
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], true, true),
        Classification::ObserverExclusionObservedAfterMappingRetirement
    );
    assert_eq!(
        classify(&phases, &[false; 3], &[true; 3], true, true),
        Classification::RetirementUnknown
    );
    assert_eq!(
        classify(&phases, &[true; 3], &[false; 3], true, true),
        Classification::RetirementUnknown
    );
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], false, true),
        Classification::RetirementUnknown
    );
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], true, false),
        Classification::RetirementUnknown
    );
    assert_eq!(
        classify(&phases[..6], &[true; 3], &[true; 3], true, true),
        Classification::RetirementUnknown
    );
    phases[4].wired[1] = Some(1);
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], true, true),
        Classification::MappingRetiredObserverStillWired
    );
    phases[4].wired[1] = Some(0);
    phases[4].wired[2] = Some(0);
    phases[4].wired[3] = Some(0);
    phases[5].wired[1] = Some(0);
    phases[6].wired[1] = Some(0);
    assert_eq!(
        classify(
            &phases,
            &[false, true, true],
            &[false, true, true],
            true,
            true
        ),
        Classification::ObserverExclusionOnlyAfterDeviceRetirement
    );
    phases[1].wired = [Some(0); 4];
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], true, true),
        Classification::RetirementUnknown
    );
    phases[1] = live;
    phases[1].agree_a = false;
    assert_eq!(
        classify(&phases, &[true; 3], &[true; 3], true, true),
        Classification::RetirementUnknown
    );
}
#[test]
#[ignore = "fresh20s supervised Metal process,<=256MiB requested data; no model credit"]
fn original_backing_mapping_retirement_observation() {
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
        "MAPPING_RETIREMENT_BEGIN {}",
        json!({"schema":"mlx.native.mapping-retirement.v1","base":"65ec3f518da867f8d9af6f173451eae424463e3a",
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
        "MAPPING_RETIREMENT_RESOURCES {}",
        json!({"classes":[class(&a),class(&b)],"device":device.metal_device().name(),
        "A_resource":a.metal_buffer().as_ref()as *const metal::BufferRef as usize,
        "B_resource":b.metal_buffer().as_ref()as *const metal::BufferRef as usize,"source_backings":["A0","B0"]})
    );
    let out = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    membership(&device, &a, &b, [true, true]);
    gpu_read(&device, &a, &out, 0);
    gpu_read(&device, &b, &out, 97);
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
    // No A resource reference is recreated. Fresh undrained sidecar removes B.
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
        drop(out);
        device.residency_set().unwrap().flush_pending();
        drop(device);
    });
    let region_a5 = region::observe("A0_after_device_drop", a_base, FILE_BYTES, &observer.origin);
    let absent5 = region_a5.kind == region::Kind::Absent;
    phases.push(observer.phase(
        "P5_device_queue_dropped",
        [None, Some(&a1), b0.as_deref(), Some(&b1)],
        None,
        &weak_a,
        &weak_b,
        vec![region_a5],
    ));
    let expired5 = phases.last().unwrap().expired[0];
    assert!(
        owner_region_consistent(expired5, absent5),
        "Weak-alive A0 unexpectedly absent"
    );
    objc::rc::autoreleasepool(|| drop(b0.take()));
    let region_a6 = region::observe(
        "A0_final_observer_only",
        a_base,
        FILE_BYTES,
        &observer.origin,
    );
    let region_b6 = region::observe(
        "B0_final_observer_only",
        b_base,
        FILE_BYTES,
        &observer.origin,
    );
    let absent6 = region_a6.kind == region::Kind::Absent;
    let absent_b6 = region_b6.kind == region::Kind::Absent;
    phases.push(observer.phase(
        "P6_B0_dropped_observers_only",
        [None, Some(&a1), None, Some(&b1)],
        None,
        &weak_a,
        &weak_b,
        vec![region_a6, region_b6],
    ));
    let expired6 = phases.last().unwrap().expired[0];
    assert!(
        owner_region_consistent(expired6, absent6),
        "Weak-alive A0 unexpectedly absent"
    );
    let expired_b6 = phases.last().unwrap().expired[1];
    assert!(
        owner_region_consistent(expired_b6, absent_b6),
        "Weak-alive B0 unexpectedly absent"
    );
    drop(a1);
    drop(b1);
    assert!(second.cleanup(), "second file cleanup");
    assert!(first.cleanup(), "owned fixture cleanup");
    assert_eq!(
        (observer.phases, observer.queries, observer.regions),
        (7, 24, 7)
    );
    let classification = classify(
        &phases,
        &[expired4, expired5, expired6],
        &[absent4, absent5, absent6],
        baseline,
        true,
    );
    println!(
        "MAPPING_RETIREMENT_COMPLETE {}",
        json!({"protocol":"PASS","classification":classification,"phases":7,"live_alias_queries":24,
        "dispositions":24*PAGES,"region_calls":7,"cleanup":true,"sentinel_GPU_reads":2,"checked_drain":true,
        "a0_expired":[expired4,expired5,expired6],"a0_range_absent":[absent4,absent5,absent6],"baseline_complete":baseline,
        "elapsed_ns":observer.origin.elapsed().as_nanos(),"model_credit_bytes":0,"physical_admission":false,"whole_model_qualified":false})
    );
}
