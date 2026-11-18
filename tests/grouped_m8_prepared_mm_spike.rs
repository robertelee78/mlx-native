//! Isolated native-only full-MM arithmetic proof; no cutover or timing claim.
use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_pooled_with_options_for_test, HalfTileSkip, IdMmScheduleOptions,
};
use mlx_native::{
    dispatch_prepared_id_mm_pooled_with_options_for_test,
    prepare_quantized_matmul_id_ggml_grouped_m8, CapturedNode, ComputeGraph, DType,
    DispatchKind, ExpertMmSchedule, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, IdMmScratch, KernelRegistry, MlxBuffer,
    MlxDevice, PreparedGroupedM8Experts,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const E: usize = 32;
const N: usize = 128;
const K: usize = 512;
const MAX_TOKENS: usize = 257;
const MAX_TOP_K: usize = 8;
const MAX_ROWS: usize = MAX_TOKENS * MAX_TOP_K;
const GUARD: usize = 16;
const PAD: usize = 32;
const LIMIT: usize = 64 * 1024 * 1024;
const RAW_GUARD: u8 = 0xa7;
const INPUT_GUARD: u32 = 0x7fc1_1111;
const IDS_GUARD: u32 = 0xa1d5_a1d5;
const REF_GUARD: u32 = 0x7fc2_2222;
const CAND_GUARD: u32 = 0x7fc3_3333;
const SCRATCH_GUARD: u32 = 0xa5ca_7c4e;

#[derive(Clone, Copy)]
struct WeightPlan { ty: GgmlType }
impl WeightPlan {
    fn format(self) -> &'static str {
        match self.ty { GgmlType::Q4_K => "Q4_K", GgmlType::Q6_K => "Q6_K", _ => unreachable!() }
    }
    fn matrix(self) -> usize { N * (K / 256) * self.ty.block_bytes() as usize }
    fn stride(self) -> usize { self.matrix() + PAD }
    fn view_bytes(self) -> usize { (E - 1) * self.stride() + self.matrix() }
    fn root_bytes(self) -> usize { 2 * GUARD * 4 + E * self.stride() }
    fn prepared_bytes(self) -> usize {
        E * N * (K / 256) * if self.ty == GgmlType::Q4_K { 144 } else { 212 }
    }
    fn params(self, tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
        GgmlQuantizedMatmulIdParams {
            n_tokens: tokens as u32, top_k: top_k as u32, n: N as u32, k: K as u32,
            n_experts: E as u32, expert_stride: self.stride() as u64, ggml_type: self.ty,
        }
    }
    fn kernel(self, tensor: bool, compact: bool, prepared: bool) -> String {
        let ty = if self.ty == GgmlType::Q4_K { "q4_K" } else { "q6_K" };
        format!("kernel_mul_mm_id_{}{ty}{}_f32{}", if prepared { "prepared_" } else { "" },
            if tensor { "_tensor" } else { "" }, if compact { "_c" } else { "" })
    }
}

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

struct Weight {
    plan: WeightPlan,
    root: MlxBuffer,
    raw: MlxBuffer,
    expected: Vec<u8>,
    prepared: PreparedGroupedM8Experts,
}
impl Weight {
    fn new(device: &MlxDevice, registry: &mut KernelRegistry, plan: WeightPlan) -> Self {
        let mut root = device.alloc_buffer(plan.root_bytes(), DType::U8,
            vec![plan.root_bytes()]).expect("raw guarded root");
        let bytes = root.as_mut_slice::<u8>().expect("raw CPU bytes");
        bytes.fill(RAW_GUARD);
        let mut rng = Rng(0xf16f_011a_4b6b_0001 ^ plan.ty.block_bytes() as u64);
        for expert in 0..E {
            let base = GUARD * 4 + expert * plan.stride();
            for block in bytes[base..base + plan.matrix()].chunks_exact_mut(plan.ty.block_bytes() as usize) {
                for b in block.iter_mut() { *b = rng.next() as u8; }
                // Fixture encoding only, not host dequantization. Finite normal,
                // signed, non-power-of-two scales exercise F16 rounding.
                let scale = ((4 + rng.next() % 5) as u16) << 10
                    | (rng.next() as u16 & 0x3ff) | ((rng.next() as u16 & 1) << 15);
                if plan.ty == GgmlType::Q4_K {
                    block[..2].copy_from_slice(&scale.to_le_bytes());
                    let min = (5u16 << 10) | (rng.next() as u16 & 0x3ff);
                    block[2..4].copy_from_slice(&min.to_le_bytes());
                } else {
                    block[208..210].copy_from_slice(&scale.to_le_bytes());
                }
            }
        }
        let expected = bytes.to_vec();
        let raw = root.slice_view((GUARD * 4) as u64, plan.view_bytes());
        let mut encoder = device.command_encoder().expect("preparation encoder");
        let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
            &mut encoder, registry, device, &raw, &plan.params(MAX_TOKENS, MAX_TOP_K),
        ).expect("prepare once per format");
        encoder.commit_and_wait_labeled("test.prepared_mm.prepare")
            .expect("checked preparation completion before CPU access or consumers");
        assert_eq!(root.as_slice::<u8>().expect("settled raw bytes"), expected);
        Self { plan, root, raw, expected, prepared }
    }
    fn unchanged(&self) -> usize {
        self.root.as_slice::<u8>().expect("raw CPU bytes").iter().zip(&self.expected)
            .filter(|(a, b)| a != b).count()
    }
}

/// All views are exact logical extents in permanently retained guarded roots.
struct Words { root: MlxBuffer, capacity: usize, dtype: DType }
impl Words {
    fn root_bytes(capacity: usize) -> usize { (capacity + 2 * GUARD) * 4 }
    fn new(device: &MlxDevice, capacity: usize, dtype: DType) -> Self {
        let root = device.alloc_buffer(Self::root_bytes(capacity), dtype,
            vec![capacity + 2 * GUARD]).expect("guarded fixture root");
        Self { root, capacity, dtype }
    }
    fn reset(&mut self, tag: u32) {
        self.root.as_mut_slice::<u32>().expect("root CPU words").fill(tag);
    }
    fn view(&self, len: usize) -> MlxBuffer {
        assert!(len <= self.capacity);
        assert!(self.dtype == DType::F32 || self.dtype == DType::U32);
        self.root.slice_view((GUARD * 4) as u64, len)
    }
    fn payload_mut(&mut self, len: usize) -> &mut [u32] {
        &mut self.root.as_mut_slice::<u32>().expect("payload words")[GUARD..GUARD + len]
    }
    fn payload(&self, len: usize) -> &[u32] {
        &self.root.as_slice::<u32>().expect("settled payload words")[GUARD..GUARD + len]
    }
    fn guard_differences(&self, len: usize, tag: u32) -> usize {
        let words = self.root.as_slice::<u32>().expect("settled root words");
        words[..GUARD].iter().chain(&words[GUARD + len..]).filter(|&&x| x != tag).count()
    }
}

// Mirrors the public constructor's documented sizing, solely for the packet
// preregistration and red zones; never calls a crate-private planning helper.
fn scratch_words(tokens: usize, top_k: usize, compact: bool) -> (usize, usize) {
    if compact {
        let rows = tokens * top_k;
        let tiles = rows.min(rows.div_ceil(32) + E.min(rows) - 1);
        (1 + 3 * tiles, rows + E)
    } else { (E, E * tokens) }
}
fn scratch_capacity() -> (usize, usize) {
    let (a, b) = scratch_words(MAX_TOKENS, 1, false);
    let (c, d) = scratch_words(MAX_TOKENS, MAX_TOP_K, true);
    (a.max(c), b.max(d))
}
struct Scratch {
    value: IdMmScratch,
    htpe: Words,
    hids: Words,
}
impl Scratch {
    fn new(device: &MlxDevice) -> Self {
        let mut value = IdMmScratch::alloc(device, E as u32, MAX_TOKENS as u32).expect("scratch metadata");
        let (a, b) = scratch_capacity();
        assert_eq!(value.htpe().data_byte_len(), a * 4);
        assert_eq!(value.hids().data_byte_len(), b * 4);
        let htpe = Words::new(device, a, DType::U32);
        let hids = Words::new(device, b, DType::U32);
        // The constructor's two temporary roots are accounted for in the
        // packet too. Replacement happens only here, never in the case loop.
        let (x, y) = value.buffers_mut();
        *x = htpe.view(a);
        *y = hids.view(b);
        Self { value, htpe, hids }
    }
    fn reset(&mut self) {
        let _ = self.value.buffers_mut(); // invalidate old routing identity
        self.htpe.reset(SCRATCH_GUARD);
        self.hids.reset(SCRATCH_GUARD);
    }
    fn guard_differences(&self, tokens: usize, top_k: usize, compact: bool, used: bool) -> usize {
        let (a, b) = if used { scratch_words(tokens, top_k, compact) } else { (0, 0) };
        self.htpe.guard_differences(a, SCRATCH_GUARD) + self.hids.guard_differences(b, SCRATCH_GUARD)
    }
}

#[derive(Clone, Copy, Debug)]
enum Pattern { Balanced, SparseTail, Invalid, Duplicate, Sentinel256 }
impl Pattern {
    fn name(self) -> &'static str {
        match self { Self::Balanced => "balanced", Self::SparseTail => "sparse_tail",
            Self::Invalid => "invalid_out_of_range", Self::Duplicate => "duplicate", Self::Sentinel256 => "all_sentinel256" }
    }
    fn poison(self) -> bool { !matches!(self, Self::Balanced | Self::SparseTail) }
    fn id(self, token: usize, slot: usize, tokens: usize, top_k: usize) -> u32 {
        match self {
            Self::Sentinel256 => 256,
            Self::Invalid if token + 1 == tokens && slot + 1 == top_k => E as u32 + 7,
            Self::Duplicate if token + 1 == tokens && slot == 1 => ((token * 7) % E) as u32,
            // Concentrate on a small expert set, but send the last token to
            // the far expert tail. Includes empty experts and partial tiles.
            Self::SparseTail => if token + 1 == tokens { (E - top_k + slot) as u32 }
                else { ((token % 3 + slot) % (top_k + 3)) as u32 },
            _ => ((token * 7 + slot * 3) % E) as u32,
        }
    }
}
#[derive(Clone, Copy)]
struct Case {
    tokens: usize, top_k: usize, tensor: bool, compact: bool, slotted: bool,
    forced: bool, pattern: Pattern, reuse: bool,
}
impl Case {
    fn schedule(self) -> ExpertMmSchedule {
        if self.compact { ExpertMmSchedule::CompactTiles } else { ExpertMmSchedule::UniformGrid }
    }
    fn options(self) -> IdMmScheduleOptions {
        IdMmScheduleOptions { schedule: self.schedule(),
            half_tile_skip: if self.forced { HalfTileSkip::Forced } else { HalfTileSkip::Disabled } }
    }
    fn policy(self) -> GgmlRoutingPolicy {
        GgmlRoutingPolicy { expert_tensor_mm: if self.tensor { GgmlTensorMmPreference::AutoProbe }
            else { GgmlTensorMmPreference::ForceSimd }, expert_mm_schedule: self.schedule(),
            ..GgmlRoutingPolicy::default() }
    }
    fn evidence(self, plan: WeightPlan) -> Value {
        json!({"format": plan.format(), "tokens": self.tokens, "top_k": self.top_k,
            "tensor": self.tensor, "compact": self.compact, "slotted": self.slotted,
            "half_tile_skip": if self.forced { "Forced" } else { "Disabled" },
            "pattern": self.pattern.name(), "reuse_reference_schedule": self.reuse})
    }
}

#[derive(Default)]
struct Stats {
    cases: usize, core: usize, forced: usize, poison: usize,
    compared: u64, differences: u64, boundary_words: u64, zeros: u64,
    negative_zeros: u64, nans: u64, failures: usize, first_failures: Vec<Value>,
    raw_routes: BTreeMap<String, usize>, candidate_routes: BTreeMap<String, usize>,
    coverage: BTreeMap<String, usize>,
}
impl Stats {
    fn bump(&mut self, key: String) { *self.coverage.entry(key).or_default() += 1; }
    fn fail(&mut self, case: Case, plan: WeightPlan, kind: &str, details: Value) {
        self.failures += 1;
        if self.first_failures.len() < 12 {
            self.first_failures.push(json!({"case": case.evidence(plan), "kind": kind, "details": details}));
        }
    }
}

/// Inspect exactly the nodes subsequently replayed, not a separately requested
/// pipeline or an inferred selector. Capture is not execution; replay + checked
/// completion below is indispensable to this numerical proof.
fn inspect_route(nodes: &[CapturedNode], expected: &str, case: Case, expect_map: bool,
    routes: &mut BTreeMap<String, usize>) -> Value {
    let mut mm = 0;
    let mut maps = 0;
    let mut bad = Vec::new();
    let expected_map = format!("kernel_mul_mm_id_map0{}_ne20_{}",
        if case.compact { "c" } else { "" }, case.top_k);
    for node in nodes {
        if let CapturedNode::Dispatch { pipeline, dispatch_kind, threads_per_grid,
            threads_per_threadgroup, threadgroup_memory, .. } = node {
            let label = pipeline.label().to_owned();
            if label == expected {
                mm += 1;
                *routes.entry(label.clone()).or_default() += 1;
                let rows = case.tokens * case.top_k;
                let grid = if case.compact { [rows.min(rows.div_ceil(32) + E.min(rows) - 1) as u64, 2, 1] }
                    else { [case.tokens.div_ceil(32) as u64, 2, E as u64] };
                if *dispatch_kind != DispatchKind::ThreadGroups
                    || [threads_per_grid.width, threads_per_grid.height, threads_per_grid.depth] != grid
                    || [threads_per_threadgroup.width, threads_per_threadgroup.height, threads_per_threadgroup.depth] != [128, 1, 1]
                    || threadgroup_memory.as_slice() != [(0, 8192)] {
                    bad.push(format!("{label}: reduction launch geometry differs"));
                }
            } else if label == expected_map {
                maps += 1;
            } else { bad.push(label); }
        }
    }
    if mm == 1 && maps == usize::from(expect_map) && bad.is_empty() { Value::Null }
    else { json!({"expected_mm": expected, "matching_mm": mm, "maps": maps,
        "expected_maps": usize::from(expect_map), "unexpected": bad}) }
}

struct Packet {
    input: Words, ids: Words, reference: Words, candidate: Words,
    reference_scratch: Scratch, candidate_scratch: Scratch,
    input_template: Vec<u32>,
}
impl Packet {
    fn new(device: &MlxDevice) -> Self {
        let mut rng = Rng(0xc1a5_51cf_16f3_2001);
        let input_template = (0..MAX_ROWS * K).map(|_| {
            // Not all F16-representable: classic and tensor retain their own
            // original activation conversion, never compare across modes.
            (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits()
        }).collect();
        Self { input: Words::new(device, MAX_ROWS * K, DType::F32),
            ids: Words::new(device, MAX_ROWS, DType::U32),
            reference: Words::new(device, MAX_ROWS * N, DType::F32),
            candidate: Words::new(device, MAX_ROWS * N, DType::F32),
            reference_scratch: Scratch::new(device), candidate_scratch: Scratch::new(device), input_template }
    }
    fn run(&mut self, device: &MlxDevice, registry: &mut KernelRegistry, weight: &Weight,
        case: Case, stats: &mut Stats) {
        let rows = case.tokens * case.top_k;
        let inputs = if case.slotted { rows } else { case.tokens } * K;
        let outputs = rows * N;
        self.input.reset(INPUT_GUARD);
        self.input.payload_mut(inputs).copy_from_slice(&self.input_template[..inputs]);
        self.ids.reset(IDS_GUARD);
        for (i, id) in self.ids.payload_mut(rows).iter_mut().enumerate() {
            *id = case.pattern.id(i / case.top_k, i % case.top_k, case.tokens, case.top_k);
        }
        self.reference.reset(REF_GUARD);
        self.candidate.reset(CAND_GUARD);
        self.reference_scratch.reset();
        self.candidate_scratch.reset();
        let input = self.input.view(inputs);
        let ids = self.ids.view(rows);
        let reference = self.reference.view(outputs);
        let candidate = self.candidate.view(outputs);
        let params = weight.plan.params(case.tokens, case.top_k);
        let policy = case.policy();
        assert!(params.n_tokens > policy.expert_mm_threshold);
        let mut encoder = device.command_encoder().expect("case encoder");
        encoder.start_capture();
        dispatch_id_mm_pooled_with_options_for_test(&mut encoder, registry, device,
            &input, &weight.raw, &ids, &reference, &mut self.reference_scratch.value,
            &params, case.slotted, false, &policy, case.options()).expect("actual raw MM oracle");
        let mut nodes = encoder.take_capture().expect("raw captured commands");
        let route = inspect_route(&nodes, &weight.plan.kernel(case.tensor, case.compact, false),
            case, true, &mut stats.raw_routes);
        if !route.is_null() { stats.fail(case, weight.plan, "raw_route", route); }
        encoder.start_capture();
        let scratch = if case.reuse { &mut self.reference_scratch.value } else { &mut self.candidate_scratch.value };
        dispatch_prepared_id_mm_pooled_with_options_for_test(&mut encoder, registry, device,
            &input, &weight.prepared, &ids, &candidate, scratch, &params,
            case.slotted, case.reuse, &policy, case.options()).expect("opaque prepared MM candidate");
        let candidate_nodes = encoder.take_capture().expect("prepared captured commands");
        let route = inspect_route(&candidate_nodes, &weight.plan.kernel(case.tensor, case.compact, true),
            case, !case.reuse, &mut stats.candidate_routes);
        if !route.is_null() { stats.fail(case, weight.plan, "candidate_route", route); }
        nodes.push(CapturedNode::Barrier);
        nodes.extend(candidate_nodes);
        let graph = ComputeGraph::from_nodes(nodes);
        graph.encode_sequential(&mut encoder);
        encoder.commit_and_wait_labeled("test.grouped_m8.prepared_mm")
            .expect("checked real GPU execution before CPU reads or any fixture reuse");

        let a = self.reference.payload(outputs);
        let b = self.candidate.payload(outputs);
        let mut differences = 0u64;
        let mut first = None;
        let mut invalid_numerics = 0;
        let mut unwritten = 0;
        let mut boundary_words = 0u64;
        for row in 0..rows {
            let mut nonzero_a = false;
            let mut nonzero_b = false;
            let token = row / case.top_k;
            let boundary = [0, 31, 32, 63, 64, 127, 128, 255, 256].contains(&token)
                || token + 1 == case.tokens;
            for col in 0..N {
                let i = row * N + col;
                let (x, y) = (a[i], b[i]);
                if x != y {
                    differences += 1;
                    if first.is_none() { first = Some(json!({"index": i, "token": token,
                        "slot": row % case.top_k, "column": col, "raw_bits": x, "prepared_bits": y})); }
                }
                unwritten += usize::from(x == REF_GUARD || y == CAND_GUARD);
                let (xf, yf) = (f32::from_bits(x), f32::from_bits(y));
                if case.pattern.poison() { invalid_numerics += usize::from(!xf.is_nan() || !yf.is_nan()); }
                else { invalid_numerics += usize::from(!xf.is_finite() || !yf.is_finite()); }
                nonzero_a |= xf.is_finite() && xf != 0.0;
                nonzero_b |= yf.is_finite() && yf != 0.0;
                stats.zeros += u64::from(xf == 0.0);
                stats.negative_zeros += u64::from(x == 0x8000_0000);
                stats.nans += u64::from(xf.is_nan());
                boundary_words += u64::from(boundary);
            }
            if !case.pattern.poison() { invalid_numerics += usize::from(!nonzero_a || !nonzero_b); }
        }
        stats.compared += outputs as u64;
        stats.differences += differences;
        stats.boundary_words += boundary_words;
        if differences != 0 { stats.fail(case, weight.plan, "exact_f32_bits",
            json!({"differences": differences, "first": first})); }
        if invalid_numerics != 0 || unwritten != 0 { stats.fail(case, weight.plan, "numerics",
            json!({"invalid_or_zero_rows": invalid_numerics, "unwritten": unwritten,
                "expected": if case.pattern.poison() { "every word NaN (legacy poison)" } else { "all finite; every row nonzero" }})); }
        let input_differences = self.input.payload(inputs).iter().zip(&self.input_template[..inputs])
            .filter(|(x, y)| x != y).count();
        let id_differences = self.ids.payload(rows).iter().enumerate().filter(|(i, id)|
            **id != case.pattern.id(i / case.top_k, i % case.top_k, case.tokens, case.top_k)).count();
        let guards = self.input.guard_differences(inputs, INPUT_GUARD)
            + self.ids.guard_differences(rows, IDS_GUARD)
            + self.reference.guard_differences(outputs, REF_GUARD)
            + self.candidate.guard_differences(outputs, CAND_GUARD)
            + self.reference_scratch.guard_differences(case.tokens, case.top_k, case.compact, true)
            + self.candidate_scratch.guard_differences(case.tokens, case.top_k, case.compact, !case.reuse);
        let weights = weight.unchanged();
        if input_differences + id_differences + guards + weights != 0 { stats.fail(case, weight.plan,
            "source_or_guard_mutation", json!({"input": input_differences, "ids": id_differences,
                "guards": guards, "raw_weight_payload_and_guards": weights})); }
        stats.cases += 1;
        if case.pattern.poison() { stats.poison += 1; } else if case.forced { stats.forced += 1; } else { stats.core += 1; }
        for key in [format!("format/{}", weight.plan.format()), format!("tokens/{}", case.tokens),
            format!("top_k/{}", case.top_k), format!("tensor/{}", case.tensor),
            format!("compact/{}", case.compact), format!("slotted/{}", case.slotted),
            format!("forced/{}", case.forced), format!("pattern/{}", case.pattern.name()),
            format!("reuse/{}", case.reuse)] { stats.bump(key); }
    }
}

#[test]
#[ignore = "preregistered bounded native-only full prepared MM arithmetic spike"]
fn grouped_m8_prepared_mm_spike() {
    let plans = [WeightPlan { ty: GgmlType::Q4_K }, WeightPlan { ty: GgmlType::Q6_K }];
    // Preplan TOTAL requested GPU fixture extents before any fixture/device
    // allocation. Include both retained raw/prepared owners and even scratch
    // constructor roots replaced during setup; no per-case GPU allocation.
    let raw_bytes: usize = plans.iter().map(|p| p.root_bytes()).sum();
    let prepared_bytes: usize = plans.iter().map(|p| p.prepared_bytes()).sum();
    let input_bytes = Words::root_bytes(MAX_ROWS * K);
    let id_bytes = Words::root_bytes(MAX_ROWS);
    let output_bytes = 2 * Words::root_bytes(MAX_ROWS * N);
    let (a, b) = scratch_capacity();
    let scratch_constructor_bytes = 2 * (a + b) * 4;
    let scratch_guarded_bytes = 2 * (Words::root_bytes(a) + Words::root_bytes(b));
    let total = raw_bytes + prepared_bytes + input_bytes + id_bytes + output_bytes
        + scratch_constructor_bytes + scratch_guarded_bytes;
    assert!(total <= LIMIT, "complete requested fixture packet exceeds 64 MiB");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weights = plans.map(|p| Weight::new(&device, &mut registry, p));
    let mut packet = Packet::new(&device);
    let mut stats = Stats::default();
    for weight in &weights {
        for tensor in [false, true] {
            for compact in [false, true] {
                for slotted in [false, true] {
                    for top_k in [1, 6, 8] {
                        for tokens in [33, 65, 128, 257] {
                            for (pattern, reuse) in [(Pattern::Balanced, false), (Pattern::SparseTail, true)] {
                                for forced in [false, true] {
                                    if forced && !compact { continue; }
                                    packet.run(&device, &mut registry, weight,
                                        Case { tokens, top_k, tensor, compact, slotted, forced, pattern, reuse }, &mut stats);
                                }
                            }
                        }
                    }
                    for pattern in [Pattern::Invalid, Pattern::Duplicate, Pattern::Sentinel256] {
                        for reuse in [false, true] {
                            packet.run(&device, &mut registry, weight,
                                Case { tokens: 65, top_k: 8, tensor, compact, slotted,
                                    forced: false, pattern, reuse }, &mut stats);
                        }
                    }
                }
            }
        }
    }
    // Retain every source and prepared owner through the final checked GPU
    // completion. Verify even the inactive format's raw root remained intact.
    let final_weight_mutations: usize = weights.iter().map(Weight::unchanged).sum();
    let counts = json!({"cases": stats.cases, "core_disabled": stats.core,
        "compact_forced": stats.forced, "poison": stats.poison,
        "compared_f32_words": stats.compared, "exact_differences": stats.differences,
        "boundary_f32_words": stats.boundary_words, "raw_zero_words": stats.zeros,
        "raw_negative_zero_words": stats.negative_zeros, "raw_nan_words": stats.nans,
        "gate_failures": stats.failures, "final_weight_mutations": final_weight_mutations});
    let allocation = json!({"fixture_packet_upper_bytes": LIMIT,
        "total_requested_bytes": total, "raw_roots": raw_bytes,
        "retained_prepared_planes": prepared_bytes, "input_root": input_bytes,
        "id_root": id_bytes, "both_output_roots": output_bytes,
        "scratch_setup_constructor_roots": scratch_constructor_bytes,
        "both_guarded_scratch_pairs": scratch_guarded_bytes,
        "per_case_gpu_allocations": 0, "preparations": 2,
        "basis": "requested GPU fixture extents; excludes shader/runtime infrastructure; no physical-memory claim",
        "all_source_and_prepared_owners_retained": true});
    let summary = json!({"schema": "mlx-native.prepared-full-mm-spike.v1",
        "test": "grouped_m8_prepared_mm_spike", "device": device.name(),
        "geometry": {"experts": E, "n": N, "k": K},
        "counts": counts, "allocation": allocation, "coverage": stats.coverage,
        "raw_actual_replayed_pipelines": stats.raw_routes,
        "prepared_actual_replayed_pipelines": stats.candidate_routes,
        "first_failures": stats.first_failures,
        "route_evidence": "exact captured pipelines and launch geometry, sequentially replayed and GPU-completed for every numerical case",
        "opaque_prepared_storage": "exact-size preparation allocation; no public plane mutation or fabricated guard access",
        "instrumentation": {"MTL_SHADER_VALIDATION": std::env::var("MTL_SHADER_VALIDATION").ok(),
            "MTL_DEBUG_LAYER": std::env::var("MTL_DEBUG_LAYER").ok(),
            "MLX_PRECOMPILED_METALLIB": std::env::var("MLX_PRECOMPILED_METALLIB").ok()},
        "source_owner_release": false, "performance_claim": false,
        "production_cutover": false, "release_qualification": false});
    let line = serde_json::to_string(&summary).expect("bounded summary JSON");
    assert!(line.len() < 64 * 1024, "summary exceeds preregistered bound");
    eprintln!("{line}"); // Leave stdout's exact test-name/result line intact.
    assert_eq!((stats.core, stats.forced, stats.poison, stats.cases), (384, 192, 96, 672));
    assert_eq!(final_weight_mutations, 0);
    assert_eq!(stats.failures, 0, "exact full-MM gate failed; retain STDERR evidence without tolerance or rerun");
}
