use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Record { rows: u32, ids_offset: usize }

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes.get(offset..offset + 4).expect("complete routing word").try_into().unwrap())
}

fn records(bytes: &[u8]) -> Vec<Record> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let layer = word(bytes, offset);
        let rows = word(bytes, offset + 4);
        let top_k = word(bytes, offset + 8);
        assert_eq!(layer as usize, result.len() % BANKS, "ordered complete forty-layer cycles");
        assert!((1..=32).contains(&rows));
        assert_eq!(top_k, TOP_K);
        offset += 12;
        let end = offset + rows as usize * TOP_K as usize * 4;
        assert!(end <= bytes.len(), "truncated actual-routing record");
        for entry in (offset..end).step_by(4) { assert!(word(bytes, entry) <= EXPERTS); }
        result.push(Record { rows, ids_offset: offset });
        offset = end;
    }
    assert_eq!(result.len() % BANKS, 0, "no partial cycle accepted");
    assert!(result.len() >= 5 * BANKS);
    result
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
        let mut input = alloc(device, rows as usize * K as usize, DType::F32);
        let mut rng = Rng(0x6210_0000 + u64::from(rows) * 40 + layer as u64);
        for value in input.as_mut_slice::<f32>().expect("recorded-route inputs") { *value = rng.value(); }
        Self {
            p: params(rows), input,
            ids: alloc(device, (rows * TOP_K) as usize, DType::U32),
            gathered: alloc(device, grouped_m8_gathered_rows(rows, TOP_K, EXPERTS).expect("gather capacity") * K as usize, DType::BF16),
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
        device: &MlxDevice, bank: &Bank, arm: usize, reuse: bool) {
        let weight = if arm == 0 { &bank.prepared } else { &bank.raw };
        quantized_matmul_id_ggml_grouped_m8_epilogue_masked(
            encoder, registry, device, &self.input, &self.gathered, weight, &self.ids,
            &self.output[arm], None, &mut self.scratch, &self.p, false, reuse, reuse,
            GroupedM8Input::F32, GroupedM8Epilogue::F32).expect("same masked projection API");
    }
}

fn burst(device: &MlxDevice, registry: &mut KernelRegistry, work: &mut [Workspace],
    banks: &[Bank], arm: usize, start: Instant) -> f64 {
    budget(start);
    let mut encoder = device.command_encoder().expect("forty-layer gate command");
    for (work, bank) in work.iter_mut().zip(banks) {
        work.encode(&mut encoder, registry, device, bank, arm, false);
        encoder.memory_barrier();
    }
    let (begin, end) = encoder.commit_wait_with_gpu_time().expect("checked forty-layer timestamp wait");
    assert!(begin.is_finite() && end.is_finite() && begin > 0.0 && end > begin);
    budget(start);
    (end - begin) * 1.0e3
}

pub(super) fn run() {
    assert!(std::env::var_os("MLX_PROFILE_CB").is_none());
    let start = Instant::now();
    let path = std::env::var_os("MLX_RAW_M8_ROUTES").expect("explicit frozen historical route corpus required");
    assert!(std::fs::metadata(&path).expect("routing metadata").len() <= 64 * 1024 * 1024);
    let bytes = std::fs::read(&path).expect("frozen routing corpus");
    let corpus_sha256 = hex::encode(Sha256::digest(&bytes));
    let records = records(&bytes);
    let cycles = records.len() / BANKS;
    let selected = [0, (cycles - 1) / 4, (cycles - 1) / 2, 3 * (cycles - 1) / 4, cycles - 1];
    assert_eq!(selected, [0, 107, 214, 321, 428], "preregistered historical corpus selection");
    let device = MlxDevice::new().expect("Metal tensor device");
    let mut registry = KernelRegistry::new();
    let banks: Vec<_> = (0..BANKS).map(|layer| bank(&device, &mut registry, 0x7150_0000 + layer as u64, start)).collect();
    assert_eq!(banks.iter().map(|bank| bank.source_address).collect::<BTreeSet<_>>().len(), BANKS);
    assert!(BANKS * (source_bytes() + prepared_bytes()) < 12 * 1024 * 1024 * 1024usize);
    let mut workspaces: BTreeMap<u32, Vec<Workspace>> = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut observations = Vec::with_capacity(80);
    let mut compared_elements = 0usize;
    let mut parity_cases = 0usize;
    for (sample, cycle) in selected.into_iter().enumerate() {
        let selected_records = &records[cycle * BANKS..(cycle + 1) * BANKS];
        let rows = selected_records[0].rows;
        assert!(selected_records.iter().all(|record| record.rows == rows));
        let work = workspaces.entry(rows).or_insert_with(||
            (0..BANKS).map(|layer| Workspace::new(&device, rows, layer)).collect());
        for (work, record) in work.iter_mut().zip(selected_records) { work.load(&bytes, *record); }
        let mut encoder = device.command_encoder().expect("recorded-route parity");
        for (layer, (work, bank)) in work.iter_mut().zip(&banks).enumerate() {
            let first = layer % 2;
            work.encode(&mut encoder, &mut registry, &device, bank, first, false);
            work.encode(&mut encoder, &mut registry, &device, bank, 1 - first, true);
            encoder.memory_barrier();
        }
        encoder.commit_and_wait().expect("checked full recorded-route parity");
        let mut live_routes = 0usize;
        for (layer, work) in work.iter().enumerate() {
            let ids = work.ids.as_slice::<u32>().expect("checked actual IDs");
            let a = work.output[0].as_slice::<f32>().expect("prepared output");
            let b = work.output[1].as_slice::<f32>().expect("raw output");
            let [mismatch, masked_writes, nonfinite_live, nonzero_live] =
                compare(a, b, ids, f32::to_bits, |value| value, F32_POISON);
            let live = ids.iter().filter(|&&id| id < EXPERTS).count();
            assert_eq!((mismatch, masked_writes, nonfinite_live), (0, 0, 0));
            assert!(live == 0 || nonzero_live > 0, "no trivial live result");
            parity_cases += 1;
            compared_elements += a.len();
            live_routes += live;
            eprintln!("{}", json!({"kind":"raw_m8_recorded_parity_case","cycle":cycle,"layer":layer,
                "rows":rows,"live_routes":live,"compared_elements":a.len(),"mismatch_elements":mismatch,
                "masked_writes":masked_writes,"nonfinite_live":nonfinite_live}));
        }
        for arm in [0, 1] { let _ = burst(&device, &mut registry, work, &banks, arm, start); }
        let mut prepared = Vec::with_capacity(16);
        let mut raw = Vec::with_capacity(16);
        for local_pair in 0..16 {
            let first = observations.len() % 2;
            let mut times = [0.0; 2];
            for arm in [first, 1 - first] {
                times[arm] = burst(&device, &mut registry, work, &banks, arm, start);
            }
            prepared.push(times[0]); raw.push(times[1]);
            let observation = json!({"pair":observations.len(),"sample":sample,"cycle":cycle,
                "local_pair":local_pair,"rows":rows,"layers":BANKS,
                "first":if first == 0 {"prepared"} else {"raw"},
                "prepared_gpu_ms":times[0],"raw_gpu_ms":times[1]});
            eprintln!("{}", json!({"kind":"raw_m8_recorded_pair","observation":observation}));
            observations.push(observation);
        }
        summaries.push(json!({"cycle":cycle,"rows":rows,"layers":BANKS,"live_routes":live_routes,
            "paired_observations":16,"prepared_median_gpu_ms":median(&mut prepared),
            "raw_median_gpu_ms":median(&mut raw)}));
    }
    assert_eq!(parity_cases, 200);
    assert_eq!(observations.len(), 80);
    budget(start);
    eprintln!("{}", json!({"kind":"unpublished_raw_m8_recorded_routes_spike","release_qualification":false,
        "corpus_sha256":corpus_sha256,"historical_cycles":cycles,"selected_cycles":selected,
        "parity_cases":parity_cases,"compared_elements":compared_elements,"mismatch_elements":0,
        "weight_banks":BANKS,"paired_observations":80,"warm_observations":10,
        "timing_clock":"MTLCommandBuffer GPUEndTime - GPUStartTime; checked completion",
        "scope":"forty serialized Q4_K F32 gate projections per command, including map/gather and interlayer barriers; actual historical IDs/masks, deterministic synthetic weights/inputs; not full MoE or model wall time",
        "summaries":summaries,"observations":observations,"elapsed_wall_seconds":start.elapsed().as_secs_f64()}));
}
