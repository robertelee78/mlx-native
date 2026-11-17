use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const CORPUS_BYTES: usize = 14_705_760;
const CORPUS_SHA256: &str = "413c816611ea65276469a58cfd56d4663540e532b287e176c997c1f33c058835";
const SELECTED: [usize; 5] = [0, 107, 214, 321, 428];

#[derive(Clone, Copy)]
struct Record { rows: u32, ids_offset: usize }

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes.get(offset..offset + 4).expect("complete routing word").try_into().unwrap())
}

pub(super) struct Corpus {
    bytes: Vec<u8>,
    records: Vec<Record>,
    sha256: String,
    path: std::path::PathBuf,
}

impl Corpus {
    pub(super) fn load() -> Self {
        let path = std::path::PathBuf::from(std::env::var_os("MLX_LIVE_M8_ROUTES")
            .expect("explicit MLX_LIVE_M8_ROUTES frozen historical corpus required; no fallback"));
        assert_eq!(std::fs::metadata(&path).expect("routing metadata").len(), CORPUS_BYTES as u64);
        let bytes = std::fs::read(&path).expect("frozen historical routing corpus");
        assert_eq!(bytes.len(), CORPUS_BYTES);
        let sha256 = hex::encode(Sha256::digest(&bytes));
        assert_eq!(sha256, CORPUS_SHA256, "frozen corpus checksum mismatch");
        let mut records = Vec::with_capacity(17_160);
        let mut offset = 0;
        let mut row_counts = BTreeMap::new();
        while offset < bytes.len() {
            let layer = word(&bytes, offset);
            let rows = word(&bytes, offset + 4);
            let top_k = word(&bytes, offset + 8);
            assert_eq!(layer as usize, records.len() % BANKS, "ordered complete forty-layer cycles");
            assert!([8, 16, 24, 32].contains(&rows), "historical row geometry");
            assert_eq!(top_k, TOP_K);
            offset += 12;
            let end = offset + (rows * TOP_K) as usize * 4;
            assert!(end <= bytes.len(), "truncated routing record");
            for entry in (offset..end).step_by(4) { assert!(word(&bytes, entry) <= EXPERTS); }
            records.push(Record { rows, ids_offset: offset });
            *row_counts.entry(rows).or_insert(0usize) += 1;
            offset = end;
        }
        assert_eq!(offset, CORPUS_BYTES);
        assert_eq!(records.len(), 17_160);
        assert_eq!(records.len() / BANKS, 429);
        assert_eq!(row_counts, BTreeMap::from([(8, 960), (16, 1680), (24, 5760), (32, 8760)]));
        Self { bytes, records, sha256, path }
    }
}

struct Workspace {
    p: GgmlQuantizedMatmulIdParams,
    input: MlxBuffer,
    ids: MlxBuffer,
    gathered: MlxBuffer,
    scratch: IdMmScratch,
    output: [MlxBuffer; 2],
}

impl Workspace {
    fn new(device: &MlxDevice, rows: u32, layer: usize) -> Self {
        let mut input = alloc(device, (rows * K) as usize, DType::F32);
        let mut rng = Rng(0x6210_0000 + u64::from(rows) * 40 + layer as u64);
        for value in input.as_mut_slice::<f32>().expect("recorded-route inputs") { *value = rng.value(); }
        Self {
            p: production_params(rows), input,
            ids: alloc(device, (rows * TOP_K) as usize, DType::U32),
            gathered: alloc(device, grouped_m8_gathered_rows(rows, TOP_K, EXPERTS)
                .expect("gather capacity") * K as usize, DType::BF16),
            scratch: IdMmScratch::alloc(device, EXPERTS, rows).expect("recorded-route schedule"),
            output: std::array::from_fn(|_| alloc(device, (rows * TOP_K * N) as usize, DType::F32)),
        }
    }

    fn load(&mut self, bytes: &[u8], record: Record) {
        assert_eq!(self.p.n_tokens, record.rows);
        for (index, value) in self.ids.as_mut_slice::<u32>().expect("actual IDs").iter_mut().enumerate() {
            *value = word(bytes, record.ids_offset + index * 4);
        }
        for output in &mut self.output {
            output.as_mut_slice::<f32>().expect("output poison").fill(f32::from_bits(F32_POISON));
        }
    }

    fn encode(&mut self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, bank: &Bank, root: &MlxBuffer, arm: usize) {
        let live;
        let weight = if arm == 0 { &bank.prepared } else {
            // Includes map0c AND live expert repacking. The unchanged projection
            // consumes precisely this encoder-owned schedule, gathering afresh.
            live = prepare_quantized_matmul_id_ggml_grouped_m8_live_into(encoder, registry,
                device, &bank.source, root, &self.ids, &mut self.scratch, &self.p, true, false)
                .expect("actual-route live repack into the one caller plane");
            &live
        };
        quantized_matmul_id_ggml_grouped_m8_epilogue_masked(encoder, registry, device,
            &self.input, &self.gathered, weight, &self.ids, &self.output[arm], None,
            &mut self.scratch, &self.p, false, arm == 1, false,
            GroupedM8Input::F32, GroupedM8Epilogue::F32).expect("unchanged masked projection");
    }
}

fn burst(device: &MlxDevice, registry: &mut KernelRegistry, work: &mut [Workspace],
    banks: &[Bank], root: &MlxBuffer, arm: usize, start: Instant) -> f64 {
    budget(start);
    assert_eq!(work.len(), BANKS);
    assert_eq!(banks.len(), BANKS);
    let mut encoder = device.command_encoder().expect("forty serialized layer command");
    for (work, bank) in work.iter_mut().zip(banks) {
        work.encode(&mut encoder, registry, device, bank, root, arm);
        encoder.memory_barrier();
    }
    let (begin, end) = encoder.commit_wait_with_gpu_time().expect("checked forty-layer timestamp wait");
    assert!(begin.is_finite() && end.is_finite() && begin > 0.0 && end > begin,
        "invalid GPU timestamps: {begin}, {end}");
    budget(start);
    (end - begin) * 1.0e3
}

pub(super) fn run(device: &MlxDevice, registry: &mut KernelRegistry, corpus: Corpus,
    parity: Value, start: Instant) {
    let p = production_params(1);
    let banks: Vec<_> = (0..BANKS).map(|layer|
        bank(device, registry, &p, 0x7150_0000 + layer as u64, start)).collect();
    let address = |buffer: &MlxBuffer| buffer.contents_ptr() as usize + buffer.byte_offset() as usize;
    let source_addresses: BTreeSet<_> = banks.iter().map(|bank| address(&bank.source)).collect();
    assert_eq!(source_addresses.len(), BANKS, "forty distinct live production stacks");
    assert_eq!(banks.iter().map(|bank| bank.seed).collect::<BTreeSet<_>>().len(), BANKS);
    let comparator_weight_bytes = BANKS * (source_bytes(&p) + plane_bytes(&p));
    assert!(comparator_weight_bytes < 12 * 1024 * 1024 * 1024usize);
    // ONE root, allocated once outside every cycle, layer, parity and timing
    // command. Never allocate a separate prepared candidate plane per layer.
    let root = alloc(device, plane_bytes(&p), DType::U8);
    let root_address = address(&root);
    assert!(!source_addresses.contains(&root_address));
    let mut workspaces: BTreeMap<u32, Vec<Workspace>> = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut observations = Vec::with_capacity(80);
    let mut warm_observations = Vec::with_capacity(10);
    let mut compared_elements = 0usize;
    let mut parity_cases = 0usize;
    let mut all_reference = Vec::with_capacity(80);
    let mut all_candidate = Vec::with_capacity(80);
    let mut all_ratios = Vec::with_capacity(80);
    for (sample, cycle) in SELECTED.into_iter().enumerate() {
        budget(start);
        let selected_records = &corpus.records[cycle * BANKS..(cycle + 1) * BANKS];
        let rows = selected_records[0].rows;
        assert!(selected_records.iter().all(|record| record.rows == rows));
        let work = workspaces.entry(rows).or_insert_with(||
            (0..BANKS).map(|layer| Workspace::new(device, rows, layer)).collect());
        for (work, record) in work.iter_mut().zip(selected_records) { work.load(&corpus.bytes, *record); }
        let mut encoder = device.command_encoder().expect("recorded-route paired parity");
        for (layer, (work, bank)) in work.iter_mut().zip(&banks).enumerate() {
            let first = layer % 2;
            for arm in [first, 1 - first] {
                work.encode(&mut encoder, registry, device, bank, &root, arm);
            }
            encoder.memory_barrier();
        }
        encoder.commit_and_wait().expect("checked full recorded-route parity");
        let mut live_routes = 0usize;
        let mut live_expert_total = 0usize;
        for (layer, work) in work.iter().enumerate() {
            let ids = work.ids.as_slice::<u32>().expect("completed actual IDs");
            let a = work.output[0].as_slice::<f32>().expect("immutable output");
            let b = work.output[1].as_slice::<f32>().expect("live output");
            let [mismatch, masked_writes, nonfinite_live, nonzero_live] =
                compare(a, b, ids, &work.p, f32::to_bits, |v| v, F32_POISON);
            let live = ids.iter().filter(|&&id| id < EXPERTS).count();
            let experts = ids.iter().copied().filter(|&id| id < EXPERTS).collect::<BTreeSet<_>>().len();
            parity_cases += 1;
            compared_elements += a.len();
            live_routes += live;
            live_expert_total += experts;
            eprintln!("{}", json!({"kind":"live_m8_recorded_parity_case","cycle":cycle,"layer":layer,
                "rows":rows,"live_routes":live,"live_experts":experts,"compared_elements":a.len(),
                "mismatch_elements":mismatch,"masked_writes":masked_writes,"nonfinite_live":nonfinite_live,
                "nonzero_live":nonzero_live,"caller_root_address":format!("0x{root_address:x}")}));
            assert_eq!((mismatch, masked_writes, nonfinite_live), (0, 0, 0));
            assert!(live == 0 || nonzero_live > 0, "no trivial live result");
        }
        // Two warm commands per sample: one reference and one candidate.
        for arm in [0, 1] {
            let gpu_ms = burst(device, registry, work, &banks, &root, arm, start);
            let observation = json!({"sample":sample,"cycle":cycle,"rows":rows,"layers":BANKS,
                "arm":if arm == 0 {"immutable_prepared"} else {"live_prepare"},"gpu_ms":gpu_ms});
            eprintln!("{}", json!({"kind":"live_m8_recorded_warm","observation":observation}));
            warm_observations.push(observation);
        }
        let mut reference = Vec::with_capacity(16);
        let mut candidate = Vec::with_capacity(16);
        let mut ratios = Vec::with_capacity(16);
        for local_pair in 0..16 {
            let first = observations.len() % 2;
            let mut times = [0.0; 2];
            for arm in [first, 1 - first] {
                times[arm] = burst(device, registry, work, &banks, &root, arm, start);
            }
            let ratio = times[1] / times[0];
            reference.push(times[0]); candidate.push(times[1]); ratios.push(ratio);
            all_reference.push(times[0]); all_candidate.push(times[1]); all_ratios.push(ratio);
            let observation = json!({"pair":observations.len(),"sample":sample,"cycle":cycle,
                "local_pair":local_pair,"rows":rows,"layers":BANKS,
                "first":if first == 0 {"immutable_prepared"} else {"live_prepare"},
                "prepared_gpu_ms":times[0],"live_prepare_gpu_ms":times[1],"candidate_over_reference":ratio});
            eprintln!("{}", json!({"kind":"live_m8_recorded_pair","observation":observation}));
            observations.push(observation);
        }
        let reference_median = median(&mut reference);
        let candidate_median = median(&mut candidate);
        summaries.push(json!({"cycle":cycle,"rows":rows,"layers":BANKS,"live_routes":live_routes,
            "live_experts_across_layers":live_expert_total,"paired_observations":16,
            "prepared_median_gpu_ms":reference_median,"live_prepare_median_gpu_ms":candidate_median,
            "ratio_of_medians":candidate_median/reference_median,"median_pair_ratio":median(&mut ratios),
            "prepared_min_gpu_ms":reference[0],"prepared_max_gpu_ms":reference[15],
            "live_prepare_min_gpu_ms":candidate[0],"live_prepare_max_gpu_ms":candidate[15]}));
    }
    assert_eq!(parity_cases, 200);
    assert_eq!(observations.len(), 80);
    assert_eq!(warm_observations.len(), 10);
    assert_eq!(address(&root), root_address);
    budget(start);
    let workspace_extents: Vec<_> = workspaces.iter().map(|(&rows, work)| json!({"rows":rows,
        "workspaces":work.len(),"input_bytes":work.iter().map(|w| w.input.data_byte_len()).sum::<usize>(),
        "ids_bytes":work.iter().map(|w| w.ids.data_byte_len()).sum::<usize>(),
        "gathered_bytes":work.iter().map(|w| w.gathered.data_byte_len()).sum::<usize>(),
        "schedule_descriptor_bytes":work.iter().map(|w| w.scratch.htpe().data_byte_len()).sum::<usize>(),
        "schedule_route_list_bytes":work.iter().map(|w| w.scratch.hids().data_byte_len()).sum::<usize>(),
        "output_bytes_both_arms":work.iter().map(|w| w.output.iter().map(MlxBuffer::data_byte_len).sum::<usize>()).sum::<usize>()})).collect();
    let reference_median = median(&mut all_reference);
    let candidate_median = median(&mut all_candidate);
    let sources: Vec<_> = banks.iter().enumerate().map(|(layer, bank)| json!({
        "layer":layer,"seed":bank.seed,
        "logical_source_address":format!("0x{:x}",address(&bank.source))
    })).collect();
    let allocation_extents = json!({
        "measurement":"requested logical buffer extents and exact native prepared layout only; not allocator/physical-memory telemetry",
        "format":"Q4_K","shape":[EXPERTS,N,K],"top_k":TOP_K,"weight_banks":BANKS,
        "distinct_source_allocations":source_addresses.len(),"source_bytes_per_stack":source_bytes(&p),
        "prepared_bytes_per_stack":plane_bytes(&p),"retained_source_bytes":BANKS*source_bytes(&p),
        "immutable_comparator_prepared_bytes":BANKS*plane_bytes(&p),
        "comparator_raw_plus_prepared_payload_bytes":comparator_weight_bytes,
        "candidate_caller_roots":1,"candidate_caller_plane_bytes":root.data_byte_len(),
        "caller_root_address":format!("0x{root_address:x}"),
        "combined_weight_payload_in_this_comparison_bytes":comparator_weight_bytes+root.data_byte_len(),
        "candidate_source_plus_one_plane_requested_bytes":BANKS*source_bytes(&p)+root.data_byte_len(),
        "private_prepared_allocation_bytes_observed":null,"allocator_bytes_observed":null,"physical_saving_observed":null,
        "workspaces":workspace_extents,"sources":sources
    });
    let timing = json!({
        "clock":"MTLCommandBuffer GPUEndTime - GPUStartTime; checked completion",
        "reference":"immutable preparation outside timing; ordinary compact schedule + BF16 gather + unchanged F32 projection",
        "candidate":"compact schedule + live expert GPU repack into ONE reused root + BF16 gather + unchanged F32 projection",
        "scope":"forty serialized Q4_K production gate layers per command, barriers included; original historical IDs/masks; deterministic synthetic weights/inputs; not full MoE or model wall time",
        "paired_observations":80,"warm_observations":warm_observations,
        "prepared_median_gpu_ms":reference_median,"live_prepare_median_gpu_ms":candidate_median,
        "ratio_of_medians":candidate_median/reference_median,"median_pair_ratio":median(&mut all_ratios),
        "summaries":summaries,"observations":observations
    });
    eprintln!("{}", json!({"kind":"unpublished_live_m8_recorded_routes_spike","release_qualification":false,
        "hf_cutover":false,"model_fit_claim":false,"corpus_path":corpus.path,
        "corpus_bytes":corpus.bytes.len(),"corpus_sha256":corpus.sha256,
        "corpus_status":"historical real routes, current retained bytes frozen now; not current qualification",
        "historical_records":corpus.records.len(),"historical_cycles":429,"selected_cycles":SELECTED,
        "parity":parity,"actual_route_parity_cases":parity_cases,"compared_elements":compared_elements,
        "mismatch_elements":0,"allocation_extents":allocation_extents,"timing":timing,
        "elapsed_wall_seconds":start.elapsed().as_secs_f64()}));
}
