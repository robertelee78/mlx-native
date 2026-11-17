//! Isolated cost experiment, not a full-model or physical-memory qualification.
use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_pooled_with_options_for_test, IdMmScheduleOptions,
};
use mlx_native::{
    dispatch_prepared_id_mm_pooled_with_options_for_test,
    prepare_quantized_matmul_id_ggml_grouped_m8, CapturedNode, ComputeGraph, DType,
    ExpertMmSchedule, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, IdMmScratch, KernelRegistry, MlxBuffer,
    MlxDevice, PreparedGroupedM8Experts,
};
use serde_json::{json, Value};
use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::hash::Hasher;

const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const BANKS: usize = 40;
const MAX_TOKENS: usize = 256;
const GUARD: usize = 16;
const GUARD_WORD: u32 = 0xa55a_936c;
const PAIRS: usize = 16;
const LIMIT: usize = 20 * 1024 * 1024 * 1024;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
fn checksum(bytes: &[u8]) -> u64 {
    // In-process mutation detection, not a cryptographic provenance claim.
    let mut h = DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
fn buffer_checksum(buffer: &MlxBuffer) -> u64 {
    checksum(buffer.as_slice::<u8>().expect("settled CPU bytes"))
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
}

#[derive(Clone, Copy)]
struct Projection { down: bool }
impl Projection {
    fn name(self) -> &'static str { if self.down { "mixed_down" } else { "gate_up_shape" } }
    fn n(self) -> usize { if self.down { 2048 } else { 512 } }
    fn k(self) -> usize { if self.down { 512 } else { 2048 } }
    fn ty(self, layer: usize) -> GgmlType {
        // The artifact's 20+20 mix; order is deliberately synthetic.
        if self.down && layer % 2 != 0 { GgmlType::Q6_K } else { GgmlType::Q4_K }
    }
    fn params(self, layer: usize, tokens: usize) -> GgmlQuantizedMatmulIdParams {
        GgmlQuantizedMatmulIdParams {
            n_tokens: tokens as u32, top_k: TOP_K as u32, n: self.n() as u32,
            k: self.k() as u32, n_experts: EXPERTS as u32,
            expert_stride: (self.n() * (self.k() / 256)
                * self.ty(layer).block_bytes() as usize) as u64,
            ggml_type: self.ty(layer),
        }
    }
}
#[derive(Clone, Copy)]
struct Case { tokens: usize, concentrated: bool, tensor: bool, compact: bool }
impl Case {
    fn schedule(self) -> ExpertMmSchedule {
        if self.compact { ExpertMmSchedule::CompactTiles } else { ExpertMmSchedule::UniformGrid }
    }
    fn policy(self) -> GgmlRoutingPolicy {
        GgmlRoutingPolicy { expert_tensor_mm: if self.tensor { GgmlTensorMmPreference::AutoProbe }
            else { GgmlTensorMmPreference::ForceSimd }, expert_mm_schedule: self.schedule(),
            ..GgmlRoutingPolicy::default() }
    }
    fn id(self, layer: usize, token: usize, slot: usize) -> u32 {
        (if self.concentrated { (slot + layer * 7) % EXPERTS }
            else { (token * 7 + slot * 13 + layer * 11) % EXPERTS }) as u32
    }
    fn metadata(self) -> Value {
        json!({"tokens": self.tokens, "top_k": TOP_K, "tensor": self.tensor,
            "compact": self.compact, "half_tile_skip": "Production",
            "routing": if self.concentrated { "concentrated_eight" } else { "balanced" }})
    }
}

struct Guarded { root: MlxBuffer, capacity: usize }
impl Guarded {
    fn new(device: &MlxDevice, capacity: usize, dtype: DType) -> Self {
        let words = capacity + 2 * GUARD;
        let root = device.alloc_buffer(words * 4, dtype, vec![words]).expect("guarded words");
        let mut value = Self { root, capacity };
        value.reset(capacity, GUARD_WORD);
        value
    }
    fn reset(&mut self, words: usize, sentinel: u32) {
        assert!(words <= self.capacity);
        let all = self.root.as_mut_slice::<u32>().expect("settled mutable words");
        all.fill(GUARD_WORD);
        all[GUARD..GUARD + words].fill(sentinel);
    }
    fn view(&self, words: usize) -> MlxBuffer {
        assert!(words <= self.capacity);
        self.root.slice_view((GUARD * 4) as u64, words)
    }
    fn payload(&self, words: usize) -> &[u32] {
        &self.root.as_slice::<u32>().expect("completed words")[GUARD..GUARD + words]
    }
    fn payload_mut(&mut self, words: usize) -> &mut [u32] {
        &mut self.root.as_mut_slice::<u32>().expect("completed mutable words")[GUARD..GUARD + words]
    }
    fn check_guards(&self, words: usize) {
        let all = self.root.as_slice::<u32>().expect("completed guards");
        assert!(all[..GUARD].iter().chain(&all[GUARD + words..]).all(|v| *v == GUARD_WORD));
    }
}
struct Bank {
    raw_root: MlxBuffer, raw: MlxBuffer, prepared: PreparedGroupedM8Experts,
    raw_checksum: u64, input: Guarded, input_checksum: u64, ids: Guarded,
    reference: Guarded, candidate: Guarded,
    reference_scratch: IdMmScratch, candidate_scratch: IdMmScratch,
}
struct Views { input: MlxBuffer, ids: MlxBuffer, reference: MlxBuffer, candidate: MlxBuffer }
struct Packet { banks: Vec<Bank>, projection: Projection, requested_bytes: usize }
impl Packet {
    fn new(device: &MlxDevice, registry: &mut KernelRegistry, projection: Projection) -> Self {
        let mut banks = Vec::with_capacity(BANKS);
        let mut requested_bytes = 0;
        let mut encoder = device.command_encoder().expect("batch preparation encoder");
        for layer in 0..BANKS {
            let params = projection.params(layer, MAX_TOKENS);
            let raw_bytes = params.expert_stride as usize * EXPERTS;
            let root_bytes = raw_bytes + 2 * GUARD * 4;
            let mut raw_root = device.alloc_buffer(root_bytes, DType::U8, vec![root_bytes])
                .expect("distinct raw bank");
            let all = raw_root.as_mut_slice::<u8>().expect("raw initialization");
            all.fill(0xa7);
            let mut rng = Rng(0xc057_4d4d_0000_0001 ^ layer as u64 ^ ((projection.down as u64) << 40));
            for block in all[GUARD * 4..GUARD * 4 + raw_bytes]
                .chunks_exact_mut(params.ggml_type.block_bytes() as usize) {
                // Fill directly into shared Metal storage; no second weight-sized host copy.
                let mut chunks = block.chunks_exact_mut(8);
                for chunk in &mut chunks { chunk.copy_from_slice(&rng.next().to_le_bytes()); }
                let tail = chunks.into_remainder();
                let last = rng.next().to_le_bytes();
                tail.copy_from_slice(&last[..tail.len()]);
                let scale = ((4 + rng.next() % 5) as u16) << 10
                    | (rng.next() as u16 & 0x3ff) | ((rng.next() as u16 & 1) << 15);
                if params.ggml_type == GgmlType::Q4_K {
                    block[..2].copy_from_slice(&scale.to_le_bytes());
                    let min = (5u16 << 10) | (rng.next() as u16 & 0x3ff);
                    block[2..4].copy_from_slice(&min.to_le_bytes());
                } else { block[208..210].copy_from_slice(&scale.to_le_bytes()); }
            }
            let raw_checksum = buffer_checksum(&raw_root);
            let raw = raw_root.slice_view((GUARD * 4) as u64, raw_bytes);
            let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(&mut encoder, registry,
                device, &raw, &params).expect("prepare retained bank");
            let input_words = MAX_TOKENS * if projection.down { TOP_K } else { 1 } * projection.k();
            let mut input = Guarded::new(device, input_words, DType::F32);
            for value in input.payload_mut(input_words) {
                *value = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
            }
            let input_checksum = buffer_checksum(&input.root);
            let ids = Guarded::new(device, MAX_TOKENS * TOP_K, DType::U32);
            let output_words = MAX_TOKENS * TOP_K * projection.n();
            let reference = Guarded::new(device, output_words, DType::F32);
            let candidate = Guarded::new(device, output_words, DType::F32);
            let reference_scratch = IdMmScratch::alloc(device, EXPERTS as u32, MAX_TOKENS as u32)
                .expect("raw routing scratch");
            let candidate_scratch = IdMmScratch::alloc(device, EXPERTS as u32, MAX_TOKENS as u32)
                .expect("prepared routing scratch");
            let prepared_bytes = EXPERTS * projection.n() * (projection.k() / 256)
                * if params.ggml_type == GgmlType::Q4_K { 144 } else { 212 };
            requested_bytes += root_bytes + prepared_bytes + input.root.data_byte_len()
                + ids.root.data_byte_len() + reference.root.data_byte_len() + candidate.root.data_byte_len()
                + reference_scratch.htpe().data_byte_len() + reference_scratch.hids().data_byte_len()
                + candidate_scratch.htpe().data_byte_len() + candidate_scratch.hids().data_byte_len();
            assert!(requested_bytes <= LIMIT, "preregistered fixture packet exceeded");
            banks.push(Bank { raw_root, raw, prepared, raw_checksum, input, input_checksum, ids,
                reference, candidate, reference_scratch, candidate_scratch });
        }
        encoder.commit_and_wait_labeled("test.prepared_mm_cost.prepare")
            .expect("all preparation completed before any fixture access");
        Self { banks, projection, requested_bytes }
    }
    fn views(&mut self, case: Case) -> Vec<Views> {
        let inputs = case.tokens * if self.projection.down { TOP_K } else { 1 } * self.projection.k();
        let ids = case.tokens * TOP_K;
        let outputs = ids * self.projection.n();
        let mut views = Vec::with_capacity(BANKS);
        for (layer, bank) in self.banks.iter_mut().enumerate() {
            bank.ids.reset(ids, 0);
            for (i, value) in bank.ids.payload_mut(ids).iter_mut().enumerate() {
                *value = case.id(layer, i / TOP_K, i % TOP_K);
            }
            bank.reference.reset(outputs, 0x7fc1_1111);
            bank.candidate.reset(outputs, 0x7fc2_2222);
            views.push(Views { input: bank.input.view(inputs), ids: bank.ids.view(ids),
                reference: bank.reference.view(outputs), candidate: bank.candidate.view(outputs) });
        }
        views
    }
    fn encode(&mut self, encoder: &mut mlx_native::CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, case: Case, views: &[Views], prepared: bool) {
        let policy = case.policy();
        assert!(case.tokens as u32 > policy.expert_mm_threshold);
        let options = IdMmScheduleOptions::production(case.schedule());
        for (layer, (bank, view)) in self.banks.iter_mut().zip(views).enumerate() {
            let params = self.projection.params(layer, case.tokens);
            if prepared {
                dispatch_prepared_id_mm_pooled_with_options_for_test(encoder, registry, device,
                    &view.input, &bank.prepared, &view.ids, &view.candidate,
                    &mut bank.candidate_scratch, &params, self.projection.down, false, &policy, options)
                    .expect("prepared MM dispatch");
            } else {
                dispatch_id_mm_pooled_with_options_for_test(encoder, registry, device,
                    &view.input, &bank.raw, &view.ids, &view.reference, &mut bank.reference_scratch,
                    &params, self.projection.down, false, &policy, options).expect("actual raw MM dispatch");
            }
        }
    }
    fn parity(&mut self, device: &MlxDevice, registry: &mut KernelRegistry, case: Case,
        views: &[Views], routes: &mut BTreeMap<String, usize>) -> u64 {
        let mut encoder = device.command_encoder().expect("parity encoder");
        encoder.start_capture();
        self.encode(&mut encoder, registry, device, case, views, false);
        let mut raw_nodes = encoder.take_capture().expect("actual raw commands");
        encoder.start_capture();
        self.encode(&mut encoder, registry, device, case, views, true);
        let candidate_nodes = encoder.take_capture().expect("actual prepared commands");
        let raw_signatures = signatures(&raw_nodes, self.projection, case, false, routes);
        let candidate_signatures = signatures(&candidate_nodes, self.projection, case, true, routes);
        assert_eq!(raw_signatures, candidate_signatures, "actual launch geometry changed");
        raw_nodes.push(CapturedNode::Barrier);
        raw_nodes.extend(candidate_nodes);
        ComputeGraph::from_nodes(raw_nodes).encode_sequential(&mut encoder);
        encoder.commit_and_wait_labeled("test.prepared_mm_cost.parity")
            .expect("checked actual parity execution");
        let words = case.tokens * TOP_K * self.projection.n();
        for (layer, bank) in self.banks.iter().enumerate() {
            for (row, (a, b)) in bank.reference.payload(words).chunks_exact(self.projection.n())
                .zip(bank.candidate.payload(words).chunks_exact(self.projection.n())).enumerate() {
                assert_eq!(a, b, "F32 bit difference layer={layer} row={row}");
                assert!(a.iter().all(|x| f32::from_bits(*x).is_finite()), "nonfinite output");
                assert!(a.iter().any(|x| f32::from_bits(*x) != 0.0), "unwritten/zero output row");
            }
            bank.reference.check_guards(words);
            bank.candidate.check_guards(words);
            bank.ids.check_guards(case.tokens * TOP_K);
            for (i, id) in bank.ids.payload(case.tokens * TOP_K).iter().enumerate() {
                assert_eq!(*id, case.id(layer, i / TOP_K, i % TOP_K));
            }
        }
        (words * BANKS) as u64
    }
    fn timed(&mut self, device: &MlxDevice, registry: &mut KernelRegistry, case: Case,
        views: &[Views], prepared: bool) -> f64 {
        let mut encoder = device.command_encoder().expect("timed command");
        self.encode(&mut encoder, registry, device, case, views, prepared);
        let (start, end) = encoder.commit_wait_with_gpu_time().expect("checked timed completion");
        assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
        (end - start) * 1000.0
    }
    fn unchanged(&self) {
        for bank in &self.banks {
            assert_eq!(buffer_checksum(&bank.raw_root), bank.raw_checksum, "raw source/guard mutation");
            assert_eq!(buffer_checksum(&bank.input.root), bank.input_checksum, "input/guard mutation");
        }
    }
}

fn signatures(nodes: &[CapturedNode], projection: Projection, case: Case, prepared: bool,
    routes: &mut BTreeMap<String, usize>) -> Vec<Value> {
    let mut signatures = Vec::with_capacity(BANKS * 2);
    let mut mm = 0;
    let mut maps = 0;
    for node in nodes {
        if let CapturedNode::Dispatch { pipeline, dispatch_kind, threads_per_grid,
            threads_per_threadgroup, threadgroup_memory, .. } = node {
            let label = pipeline.label().to_owned();
            if label.starts_with("kernel_mul_mm_id_map0") { maps += 1; }
            else {
                assert!(mm < BANKS);
                let format = if projection.ty(mm) == GgmlType::Q4_K { "q4_K" } else { "q6_K" };
                let expected = format!("kernel_mul_mm_id_{}{format}{}_f32{}",
                    if prepared { "prepared_" } else { "" },
                    if case.tensor { "_tensor" } else { "" }, if case.compact { "_c" } else { "" });
                assert_eq!(label, expected, "requested arithmetic path did not execute");
                mm += 1;
            }
            *routes.entry(label.clone()).or_default() += 1;
            signatures.push(json!({"pipeline": label.replacen("kernel_mul_mm_id_prepared_", "kernel_mul_mm_id_", 1),
                "kind": format!("{dispatch_kind:?}"),
                "grid": [threads_per_grid.width, threads_per_grid.height, threads_per_grid.depth],
                "threads": [threads_per_threadgroup.width, threads_per_threadgroup.height, threads_per_threadgroup.depth],
                "threadgroup_memory": threadgroup_memory}));
        }
    }
    assert_eq!((maps, mm), (BANKS, BANKS));
    signatures
}

fn run(projection: Projection) {
    assert!(std::env::var_os("MTL_SHADER_VALIDATION").is_none());
    assert!(std::env::var_os("MTL_DEBUG_LAYER").is_none());
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut packet = Packet::new(&device, &mut registry, projection);
    let mut scenarios = Vec::with_capacity(16);
    let mut routes = BTreeMap::new();
    let mut compared = 0u64;
    for tokens in [128, 256] {
        for concentrated in [false, true] {
            for tensor in [false, true] {
                for compact in [false, true] {
                    let case = Case { tokens, concentrated, tensor, compact };
                    let views = packet.views(case);
                    compared += packet.parity(&device, &mut registry, case, &views, &mut routes);
                    for _ in 0..2 {
                        packet.timed(&device, &mut registry, case, &views, false);
                        packet.timed(&device, &mut registry, case, &views, true);
                    }
                    let mut raw_ms = Vec::with_capacity(PAIRS);
                    let mut prepared_ms = Vec::with_capacity(PAIRS);
                    let mut ratios = Vec::with_capacity(PAIRS);
                    let mut samples = Vec::with_capacity(PAIRS);
                    for pair in 0..PAIRS {
                        let (raw, candidate) = if pair % 2 == 0 {
                            (packet.timed(&device, &mut registry, case, &views, false),
                                packet.timed(&device, &mut registry, case, &views, true))
                        } else {
                            let candidate = packet.timed(&device, &mut registry, case, &views, true);
                            (packet.timed(&device, &mut registry, case, &views, false), candidate)
                        };
                        raw_ms.push(raw);
                        prepared_ms.push(candidate);
                        ratios.push(candidate / raw);
                        samples.push(json!({"pair": pair, "raw_first": pair % 2 == 0,
                            "raw_gpu_ms": raw, "prepared_gpu_ms": candidate}));
                    }
                    scenarios.push(json!({"case": case.metadata(), "samples": samples,
                        "raw_median_gpu_ms": median(&raw_ms), "prepared_median_gpu_ms": median(&prepared_ms),
                        "median_paired_ratio": median(&ratios)}));
                    let words = tokens * TOP_K * projection.n();
                    for bank in &packet.banks {
                        bank.reference.check_guards(words);
                        bank.candidate.check_guards(words);
                    }
                }
            }
        }
    }
    packet.unchanged();
    assert_eq!(scenarios.len(), 16);
    eprintln!("GROUPED_M8_PREPARED_MM_COST_SPIKE_JSON={}", json!({
        "projection": projection.name(), "qualification": false, "physical_savings_claimed": false,
        "experts": EXPERTS, "banks": BANKS, "top_k": TOP_K, "n": projection.n(), "k": projection.k(),
        "slotted": projection.down, "requested_fixture_bytes": packet.requested_bytes,
        "compared_f32_words": compared, "bit_differences": 0, "guard_or_source_failures": 0,
        "routes": routes, "scenarios": scenarios,
        "timing_scope": "fresh map+MM commands over40distinct banks; GPU end-start; normal mode; no capture in timed commands"
    }));
}
#[test]
#[ignore = "bounded isolated production-shape prepared MM cost experiment"]
fn grouped_m8_prepared_mm_gate_cost_spike() { run(Projection { down: false }); }
#[test]
#[ignore = "bounded isolated production-shape prepared MM cost experiment"]
fn grouped_m8_prepared_mm_down_cost_spike() { run(Projection { down: true }); }
