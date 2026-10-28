//! Actual observer and fail-closed primary classifier for the separate joint case.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct Summary {
    pub wired: [Option<usize>; 4],
    pub agree_a: bool,
    pub agree_b: bool,
    pub expired: [bool; 2],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) enum Classification {
    BothOriginalMappingsRetiredObserversUnwiredWithLiveDeviceOutput,
    BothMappingsRetiredObserverStillWired,
    JointRetirementUnknown,
}
#[derive(Clone, Copy)]
pub(super) struct OutputJoin {
    pub same_device: bool,
    pub same_output: bool,
    pub ordinary_shared: bool,
    pub contents_match: bool,
}
pub(super) fn owner_region_consistent(expired: bool, absent: bool) -> bool {
    expired || !absent
}
pub(super) fn classify(
    phases: &[Summary],
    absent5: [bool; 2],
    output: OutputJoin,
    baseline: bool,
    cleanup: bool,
) -> Classification {
    if phases.len() != 7
        || !baseline
        || !cleanup
        || !output.same_device
        || !output.same_output
        || !output.ordinary_shared
        || !output.contents_match
    {
        return Classification::JointRetirementUnknown;
    }
    if !phases[..4].iter().all(|p| p.agree_a && p.agree_b)
        || !phases[4].agree_b
        || phases[1].wired != [Some(PAGES); 4]
    {
        return Classification::JointRetirementUnknown;
    }
    let deciding = phases[5];
    if deciding.expired != [true; 2]
        || absent5 != [true; 2]
        || deciding.wired[0].is_some()
        || deciding.wired[2].is_some()
    {
        return Classification::JointRetirementUnknown;
    }
    match (deciding.wired[1], deciding.wired[3]) {
        (Some(0), Some(0)) => {
            Classification::BothOriginalMappingsRetiredObserversUnwiredWithLiveDeviceOutput
        }
        (Some(_), Some(_)) => Classification::BothMappingsRetiredObserverStillWired,
        _ => Classification::JointRetirementUnknown,
    }
}
pub(super) struct Observer {
    pub(super) origin: Instant,
    pub(super) last: u128,
    pub(super) phases: usize,
    pub(super) queries: usize,
    pub(super) regions: usize,
    pub(super) bytes: usize,
}
impl Observer {
    pub(super) fn phase(
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
                wired[index] = Some(bits.iter().filter(|b| **b & super::WIRED != 0).count());
                vectors[index] = Some(bits);
                self.queries += 1;
                identities.push(json!({"alias":(["A0","A1","B0","B1"][index]),"base":map.as_ptr()as usize,"bytes":map.len()}));
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
        println!("JOINT_MAPPING_RETIREMENT_PHASE {text}");
        self.phases += 1;
        assert!(
            protocol,
            "invalid region ABI/API/port cleanup; no qualification"
        );
        summary
    }
}

#[test]
fn joint_mapping_retirement_primary_requires_exact_live_device_join() {
    let live = Summary {
        wired: [Some(PAGES); 4],
        agree_a: true,
        agree_b: true,
        expired: [false; 2],
    };
    let mut p = vec![live; 7];
    p[4].wired = [None, Some(PAGES), Some(PAGES), Some(PAGES)];
    p[5].wired = [None, Some(0), None, Some(0)];
    p[5].expired = [true; 2];
    let good = OutputJoin {
        same_device: true,
        same_output: true,
        ordinary_shared: true,
        contents_match: true,
    };
    assert_eq!(
        classify(&p, [true; 2], good, true, true),
        Classification::BothOriginalMappingsRetiredObserversUnwiredWithLiveDeviceOutput
    );
    for output in [
        OutputJoin {
            same_device: false,
            ..good
        },
        OutputJoin {
            same_output: false,
            ..good
        },
        OutputJoin {
            ordinary_shared: false,
            ..good
        },
        OutputJoin {
            contents_match: false,
            ..good
        },
    ] {
        assert_eq!(
            classify(&p, [true; 2], output, true, true),
            Classification::JointRetirementUnknown
        );
    }
    let mut bad = p.clone();
    bad[5].expired[0] = false;
    assert_eq!(
        classify(&bad, [true; 2], good, true, true),
        Classification::JointRetirementUnknown
    );
    assert_eq!(
        classify(&p, [true, false], good, true, true),
        Classification::JointRetirementUnknown
    );
    assert_eq!(
        classify(&p, [true; 2], good, false, true),
        Classification::JointRetirementUnknown
    );
    assert_eq!(
        classify(&p, [true; 2], good, true, false),
        Classification::JointRetirementUnknown
    );
    assert_eq!(
        classify(&p[..6], [true; 2], good, true, true),
        Classification::JointRetirementUnknown
    );
    let mut bad = p.clone();
    bad[1].wired[0] = Some(PAGES - 1);
    assert_eq!(
        classify(&bad, [true; 2], good, true, true),
        Classification::JointRetirementUnknown
    );
    let mut bad = p.clone();
    bad[2].agree_b = false;
    assert_eq!(
        classify(&bad, [true; 2], good, true, true),
        Classification::JointRetirementUnknown
    );
    let mut bad = p.clone();
    bad[4].agree_b = false;
    assert_eq!(
        classify(&bad, [true; 2], good, true, true),
        Classification::JointRetirementUnknown,
        "P4 still retains both B aliases; contradictory flags cannot qualify"
    );
    let mut bad = p.clone();
    bad[5].wired[1] = Some(1);
    bad[6].wired = [None, Some(0), None, Some(0)];
    bad[6].expired = [true; 2];
    assert_eq!(
        classify(&bad, [true; 2], good, true, true),
        Classification::BothMappingsRetiredObserverStillWired,
        "P6 zero cannot promote P5"
    );
    assert!(!owner_region_consistent(false, true));
    assert!(owner_region_consistent(true, true));
}
