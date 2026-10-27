//! Fresh-process startup membership cancellation plus actual recorded graph use.
use super::{
    class, cpu_positive, host_page_size, Fixture, Flags, Observer, SecondFile, FILE_BYTES, LIMIT,
    LIVE_METADATA_CAP, PAGE, PHASE_METADATA, SENTINELS, USERDATA,
};
use crate::graph::GraphExecutor;
use crate::{DType, KernelRegistry, MlxDevice};
use memmap2::Mmap;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const OBSERVATIONS: usize = 7;
const EXTRA_ROOT_BYTES: usize = 80; // output64 +addend16, separate from mapped roots.

fn checked_range(base: usize, offset: u64, len: usize, root_len: usize) -> Option<(usize, usize)> {
    let offset = usize::try_from(offset).ok()?;
    if base == 0 || len == 0 || offset.checked_add(len)? > root_len {
        return None;
    }
    let start = base.checked_add(offset)?;
    Some((start, start.checked_add(len)?))
}

fn logical_range(buffer: &crate::MlxBuffer) -> (usize, usize) {
    checked_range(
        buffer.contents_ptr() as usize,
        buffer.byte_offset(),
        buffer.data_byte_len(),
        buffer.byte_len(),
    )
    .expect("bounded nonnull logical annotation")
}

#[test]
fn startup_logical_annotation_range_uses_offset_and_rejects_overflow() {
    assert_eq!(checked_range(4096, 16384, 16, 65536), Some((20480, 20496)));
    assert_eq!(checked_range(4096, 16, 16, 64), Some((4112, 4128)));
    assert_eq!(checked_range(4096, 48, 16, 64), Some((4144, 4160)));
    assert_eq!(checked_range(0, 0, 16, 64), None);
    assert_eq!(checked_range(4096, 49, 16, 64), None);
    assert_eq!(checked_range(usize::MAX - 7, 0, 16, 64), None);
    assert_eq!(checked_range(4096, u64::MAX, 16, usize::MAX), None);
}

fn observed_startup(phases: &[[Flags; 2]], commits: &[usize], baseline: usize) -> bool {
    phases.len() == OBSERVATIONS
        && commits.len() == OBSERVATIONS
        && phases
            .iter()
            .all(|p| p[0].aliases_equal && p[1].aliases_equal && p[1].zero_both)
        && phases.last().is_some_and(|p| p[0].any_both)
        && commits.first().is_some_and(|&c| c <= baseline)
        && commits[1..OBSERVATIONS - 1].iter().all(|&c| c == baseline)
        && baseline.checked_add(1) == commits.last().copied()
}

/// Replace the existing byte fixture BEFORE creating any mapping/resource.
/// Both initialization passes write the same bounded128MiB unique file pages;
/// there is no extra file or full-file host Vec. Final bytes are known F32.
fn initialize_floats(path: &Path, identity: (u64, u64), seed: i32) {
    let before = fs::symlink_metadata(path).unwrap();
    assert!(before.is_file() && !before.file_type().is_symlink());
    assert_eq!((before.dev(), before.ino()), identity);
    assert_eq!(before.len(), FILE_BYTES as u64);
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    let actual = file.metadata().unwrap();
    assert_eq!((actual.dev(), actual.ino()), identity);
    let mut page = [0_u8; PAGE];
    for index in 0..FILE_BYTES / PAGE {
        let value = ((index % 17) as i32 - 8 + seed) as f32;
        for chunk in page.chunks_exact_mut(4) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        file.write_all(&page).unwrap();
    }
    file.sync_all().unwrap();
    assert_eq!(file.metadata().unwrap().len(), FILE_BYTES as u64);
}

fn sample(
    observer: &mut Observer,
    name: &str,
    aliases: [&Mmap; 4],
    device: Option<&MlxDevice>,
    membership: [bool; 2],
    classes: Option<&[String; 2]>,
    phases: &mut Vec<[Flags; 2]>,
    commits: &mut Vec<usize>,
) {
    let flags = observer.phase(name, aliases, device, membership, classes);
    let counter = crate::residency::residency_commit_call_count_for_test();
    println!(
        "STARTUP_GRAPH_COUNTER {}",
        json!({"phase":name,"residency_set_commits":counter})
    );
    phases.push(flags);
    commits.push(counter);
}

#[test]
fn startup_classification_requires_every_retained_phase_and_actual_flush() {
    let absent = Flags {
        zero_both: true,
        any_both: false,
        aliases_equal: true,
    };
    let live = Flags {
        zero_both: false,
        any_both: true,
        aliases_equal: true,
    };
    let mut phases = vec![[absent, absent]; OBSERVATIONS];
    phases[OBSERVATIONS - 1][0] = live;
    let mut commits = vec![9; OBSERVATIONS];
    commits[OBSERVATIONS - 1] = 10;
    assert!(observed_startup(&phases, &commits, 9));
    let mut device_initial_commit = commits.clone();
    device_initial_commit[0] = 8;
    assert!(observed_startup(&phases, &device_initial_commit, 9));
    let mut early_wired = phases.clone();
    early_wired[3][1] = live;
    assert!(!observed_startup(&early_wired, &commits, 9));
    let mut wrong_alias = phases.clone();
    wrong_alias[2][1].aliases_equal = false;
    assert!(!observed_startup(&wrong_alias, &commits, 9));
    assert!(!observed_startup(&phases[..6], &commits[..6], 9));
    let mut premature = commits.clone();
    premature[3] = 10;
    assert!(!observed_startup(&phases, &premature, 9));
    let mut no_flush = commits.clone();
    no_flush[OBSERVATIONS - 1] = 9;
    assert!(!observed_startup(&phases, &no_flush, 9));
    phases[OBSERVATIONS - 1][0] = absent;
    assert!(!observed_startup(&phases, &commits, 9));
}

#[test]
#[ignore = "startup two64MiB mapped roots with captured compute; isolated root-supervised fresh process"]
fn startup_pending_remove_then_recorded_graph_flush_observation() {
    assert!(USERDATA + EXTRA_ROOT_BYTES <= LIMIT);
    assert!(PHASE_METADATA <= LIVE_METADATA_CAP);
    assert_eq!(host_page_size().unwrap(), PAGE);
    assert_eq!(std::env::var("HF2Q_ENCODER_SESSION").as_deref(), Ok("1"));
    assert!(!std::env::vars().any(|(k, _)| (k.starts_with("HF2Q_")
        && k != "HF2Q_ENCODER_SESSION")
        || k.starts_with("MLX_")
        || k.starts_with("MTL_")
        || k.starts_with("METAL_")));
    cpu_positive();
    let mut first = Fixture::new();
    let mut second = SecondFile::new(&first);
    initialize_floats(&first.dir.join("owned.bin"), first.file_id.unwrap(), 0);
    initialize_floats(&second.path, second.id.unwrap(), 32);
    let a0 = first.mapping();
    let a1 = first.mapping();
    let b0 = second.mapping();
    let b1 = second.mapping();
    assert_ne!(a0.as_ptr(), a1.as_ptr());
    assert_ne!(b0.as_ptr(), b1.as_ptr());
    assert_ne!(first.file_id, second.id);
    println!(
        "STARTUP_GRAPH_BEGIN {}",
        json!({
        "file_payload_union_bytes":2*FILE_BYTES,"requested_weight_roots":2,
        "requested_weight_root_bytes":2*FILE_BYTES,"extra_root_bytes":EXTRA_ROOT_BYTES,
        "alias_VA_bytes_nonadditive":4*FILE_BYTES,"userdata_upper_bytes":USERDATA+EXTRA_ROOT_BYTES,
        "userdata_limit":LIMIT,"initialization_passes_per_file":2,
        "files":[first.dir.join("owned.bin"),second.path],
        "file_ids":[first.file_id,second.id],"page_bytes":PAGE,
        "inactive_wire_scope":"retained observations, not continuous peak or model credit",
        "model_loaded":false,"physical_admission":false})
    );
    let mut observer = Observer {
        bytes: 0,
        phases: 0,
        origin: Instant::now(),
        last_end_ns: 0,
    };
    let mut phases = Vec::with_capacity(OBSERVATIONS);
    let mut commits = Vec::with_capacity(OBSERVATIONS);
    sample(
        &mut observer,
        "plain_aliases_pre_device",
        [&a0, &a1, &b0, &b1],
        None,
        [false, false],
        None,
        &mut phases,
        &mut commits,
    );
    let device = MlxDevice::new().expect("Metal required, no skip");
    assert!(device.residency_sets_enabled());
    // OBSERVED baseline: no assumption that device creation commits an empty set.
    let c0 = crate::residency::residency_commit_call_count_for_test();
    // A possible device-side initial commit is recorded relative to the pre-device
    // counter, not assumed zero or confused with a later weight membership flush.
    sample(
        &mut observer,
        "fresh_device_before_roots",
        [&a0, &a1, &b0, &b1],
        Some(&device),
        [false, false],
        None,
        &mut phases,
        &mut commits,
    );
    let a = device
        .map_file_buffer(
            Arc::clone(&a0),
            0,
            FILE_BYTES,
            DType::F32,
            vec![FILE_BYTES / 4],
        )
        .unwrap();
    sample(
        &mut observer,
        "A_auto_added_before_B",
        [&a0, &a1, &b0, &b1],
        Some(&device),
        [true, false],
        None,
        &mut phases,
        &mut commits,
    );
    let b = device
        .map_file_buffer(
            Arc::clone(&b0),
            0,
            FILE_BYTES,
            DType::F32,
            vec![FILE_BYTES / 4],
        )
        .unwrap();
    assert!(
        a.is_file_backed() && b.is_file_backed() && !a.is_cpu_writable() && !b.is_cpu_writable()
    );
    let classes = [class(&a), class(&b)];
    sample(
        &mut observer,
        "B_auto_added_no_flush",
        [&a0, &a1, &b0, &b1],
        Some(&device),
        [true, true],
        Some(&classes),
        &mut phases,
        &mut commits,
    );
    let mut output = device.alloc_buffer(64, DType::F32, vec![16]).unwrap();
    output.as_mut_slice::<f32>().unwrap().fill(-999.0);
    let mut addend = device.alloc_buffer(16, DType::F32, vec![4]).unwrap();
    addend.as_mut_slice::<f32>().unwrap().fill(2.0);
    let sidecar = device
        .encoder_session()
        .unwrap()
        .expect("public sidecar capability required");
    assert!(
        sidecar.remove_from_residency_set(&b),
        "actual membership capability unavailable"
    );
    sample(
        &mut observer,
        "pending_B_removed_empty_sidecar_alive",
        [&a0, &a1, &b0, &b1],
        Some(&device),
        [true, false],
        Some(&classes),
        &mut phases,
        &mut commits,
    );
    let executor = GraphExecutor::new(device.clone());
    assert_eq!(
        executor.device().metal_queue().as_ref() as *const metal::CommandQueueRef,
        device.metal_queue().as_ref() as *const metal::CommandQueueRef
    );
    let mut registry = KernelRegistry::new();
    println!(
        "STARTUP_GRAPH_QUEUE {}",
        json!({"registry_id":device.registry_id(),
        "queue_address":device.metal_queue().as_ref() as *const metal::CommandQueueRef as usize,
        "roots":[a.metal_buffer().as_ref() as *const metal::BufferRef as usize,
            b.metal_buffer().as_ref() as *const metal::BufferRef as usize],"classes":classes})
    );
    let (reordered, barriers) = {
        let mut graph = executor.begin_recorded().unwrap();
        let mut owners = Vec::with_capacity(4);
        for (index, page) in SENTINELS.iter().enumerate() {
            let input = a
                .slice_view((page * PAGE) as u64, 4)
                .with_shape(vec![4])
                .unwrap();
            let result = output
                .slice_view((index * 16) as u64, 4)
                .with_shape(vec![4])
                .unwrap();
            // These are true logical view ranges, including byte offsets.
            // GraphSession::barrier_between uses whole-parent bounds; do not use
            // it to describe four disjoint16B page/output slices in this control.
            graph.encoder_mut().set_pending_buffer_ranges(
                vec![logical_range(&input), logical_range(&addend)],
                vec![logical_range(&result)],
            );
            graph
                .elementwise_add(
                    &mut registry,
                    device.metal_device(),
                    &input,
                    &addend,
                    &result,
                    4,
                    DType::F32,
                )
                .unwrap();
            owners.push((input, result));
        }
        sample(
            &mut observer,
            "four_dispatches_captured_before_finish",
            [&a0, &a1, &b0, &b1],
            Some(&device),
            [true, false],
            Some(&classes),
            &mut phases,
            &mut commits,
        );
        assert!(
            commits.iter().skip(1).all(|&n| n == c0),
            "unexpected pre-graph set commit"
        );
        let counts = graph
            .finish_with_reorder()
            .expect("actual annotated recorded graph checked finish");
        drop(owners); // Graph/recorded bindings consumed; views drop before classification.
        counts
    };
    let values = output.as_logical_slice::<f32>().unwrap();
    for (index, page) in SENTINELS.iter().enumerate() {
        let expected = ((*page % 17) as i32 - 8) as f32 + 2.0;
        assert!(expected.is_finite() && expected != 0.0);
        for lane in 0..4 {
            assert_eq!(
                values[index * 4 + lane].to_bits(),
                expected.to_bits(),
                "independent initialized-page captured GPU add oracle"
            );
        }
    }
    sample(
        &mut observer,
        "recorded_graph_checked_drained_owners_alive",
        [&a0, &a1, &b0, &b1],
        Some(&device),
        [true, false],
        Some(&classes),
        &mut phases,
        &mut commits,
    );
    assert_eq!(
        commits.last().copied(),
        c0.checked_add(1),
        "real GraphSession must flush once"
    );
    let observed = observed_startup(&phases, &commits, c0);
    println!(
        "STARTUP_GRAPH_COMPUTE {}",
        json!({"captured_add_dispatches":4,
        "output_elements":16,"cpu_oracle_bits_equal":true,"checked_finish":true,
        "reordered_nodes":reordered,"barriers":barriers,"observed_commit_baseline":c0,
        "observed_commit_after_graph":commits.last(),"sidecar_commit_invocations":0})
    );
    drop(registry);
    drop(executor);
    drop(sidecar);
    drop(addend);
    drop(output);
    drop(b);
    drop(a);
    drop(device);
    drop(b1);
    drop(b0);
    drop(a1);
    drop(a0);
    assert!(second.cleanup());
    assert!(first.cleanup());
    assert_eq!(observer.phases, OBSERVATIONS);
    println!(
        "STARTUP_GRAPH_COMPLETE {}",
        json!({"protocol":"PASS","phases":observer.phases,
        "classification":if observed{"StartupGraphExclusionObservedAtRetainedPhases"}
            else{"NoObservedStartupGraphExclusion"},"cleanup":true,
        "continuous_peak_proved":false,"model_credit_bytes":0,"model_scale_qualified":false,
        "physical_admission":false})
    );
}
