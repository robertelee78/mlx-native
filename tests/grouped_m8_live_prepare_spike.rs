#![cfg(mlx_native_has_metal_tensor_artifact)]

// Unpublished native-only spike, not release qualification or a model-fit claim.
use half::bf16;
use mlx_native::{
    ggml_expert_bytes, grouped_m8_gathered_rows, grouped_m8_prepared_stack_bytes,
    prepare_quantized_matmul_id_ggml_grouped_m8,
    prepare_quantized_matmul_id_ggml_grouped_m8_live_into,
    quantized_matmul_id_ggml_grouped_m8_epilogue,
    quantized_matmul_id_ggml_grouped_m8_epilogue_masked,
    quantized_matmul_id_ggml_grouped_m8_gate_up, CommandEncoder, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, GroupedM8Epilogue, GroupedM8Input,
    IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice, PreparedGroupedM8Experts,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[path = "support/live_m8_recorded.rs"]
mod live_m8_recorded;

const EXPERTS: u32 = 256;
const N: u32 = 512;
const K: u32 = 2048;
const TOP_K: u32 = 8;
const BANKS: usize = 40;
const F32_POISON: u32 = 0x7fc1_2345;
const BF16_POISON: u16 = 0x7fc1;
const PLANE_POISON: u8 = 0xa5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn value(&mut self) -> f32 {
        (((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.125
    }
}

fn budget(start: Instant) {
    assert!(start.elapsed() < Duration::from_secs(390), "spike exceeded its 390-second work budget");
}

fn params(rows: u32, experts: u32, n: u32, k: u32, ggml_type: GgmlType) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: rows, top_k: TOP_K, n, k, n_experts: experts,
        expert_stride: u64::from(n) * u64::from(k / 256) * u64::from(ggml_type.block_bytes()),
        ggml_type,
    }
}

fn production_params(rows: u32) -> GgmlQuantizedMatmulIdParams {
    params(rows, EXPERTS, N, K, GgmlType::Q4_K)
}

fn alloc(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    device.alloc_buffer(elements * dtype.size_of(), dtype, vec![elements]).expect("allocate probe buffer")
}

fn source_bytes(p: &GgmlQuantizedMatmulIdParams) -> usize {
    ggml_expert_bytes(p.ggml_type, p.n_experts, p.n, p.k, p.expert_stride).expect("source extent") as usize
}

fn plane_bytes(p: &GgmlQuantizedMatmulIdParams) -> usize {
    grouped_m8_prepared_stack_bytes(p.n_experts, p.n, p.k, p.ggml_type).expect("exact prepared extent")
}

struct Bank {
    source: MlxBuffer,
    prepared: PreparedGroupedM8Experts,
    seed: u64,
}

fn bank(device: &MlxDevice, registry: &mut KernelRegistry, p: &GgmlQuantizedMatmulIdParams,
    seed: u64, start: Instant) -> Bank {
    budget(start);
    let mut source = alloc(device, source_bytes(p), DType::U8);
    let mut rng = Rng(seed);
    for (index, block) in source.as_mut_slice::<u8>().expect("packed source")
        .chunks_exact_mut(p.ggml_type.block_bytes() as usize).enumerate() {
        for chunk in block.chunks_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes()[..chunk.len()]);
        }
        let offsets: &[usize] = match p.ggml_type {
            GgmlType::Q4_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            _ => unreachable!(),
        };
        // Preserve the rejected probe's deterministic, nontrivial Q4 fixture:
        // packed scales/minima/nibbles and finite normal/subnormal half scales.
        // Q6 keeps signed scales and both quant planes nontrivial as well.
        for (scale, &offset) in offsets.iter().enumerate() {
            let mantissa = (rng.next() as u16 & 0x03ff).max(1);
            let magnitude = match (index + scale) % 6 {
                0 => 1,
                1 => mantissa,
                2 => 0x0400,
                3 => (2 << 10) | mantissa,
                _ => (7 << 10) | mantissa,
            };
            let sign = if scale == 0 && rng.next() & 1 != 0 { 0x8000 } else { 0 };
            block[offset..offset + 2].copy_from_slice(&(magnitude | sign).to_le_bytes());
        }
    }
    if matches!(p.ggml_type, GgmlType::Q4_K) {
        for block in source.as_mut_slice::<u8>().expect("established Q4 fixture").chunks_exact_mut(144) {
            block[16] ^= 0x5a;
        }
    }
    let mut encoder = device.command_encoder().expect("immutable preparation encoder");
    let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(&mut encoder, registry, device, &source, p)
        .expect("immutable comparator preparation");
    encoder.commit_and_wait().expect("checked immutable preparation wait");
    budget(start);
    Bank { source, prepared, seed }
}

#[derive(Clone, Copy)]
enum Mode { F32, Bf16, Bf16Input, SiluMul, FusedGateUp }

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32_projection",
            Self::Bf16 => "bf16_projection",
            Self::Bf16Input => "slotted_bf16_input_f32_projection",
            Self::SiluMul => "bf16_gate_consumed_by_silu_mul",
            Self::FusedGateUp => "fused_gate_up_bf16",
        }
    }

    fn f32_output(self) -> bool { matches!(self, Self::F32 | Self::Bf16Input) }
}

struct Work {
    p: GgmlQuantizedMatmulIdParams,
    input: MlxBuffer,
    slotted: MlxBuffer,
    ids: MlxBuffer,
    gathered: MlxBuffer,
    scratch: IdMmScratch,
    f32_output: [MlxBuffer; 2],
    bf16_output: [MlxBuffer; 2],
    gate: [MlxBuffer; 2],
}

impl Work {
    fn new(device: &MlxDevice, p: GgmlQuantizedMatmulIdParams, seed: u64) -> Self {
        let pairs = (p.n_tokens * p.top_k) as usize;
        let elements = pairs * p.n as usize;
        let mut input = alloc(device, (p.n_tokens * p.k) as usize, DType::F32);
        let mut slotted = alloc(device, pairs * p.k as usize, DType::BF16);
        let mut rng = Rng(seed);
        for value in input.as_mut_slice::<f32>().expect("F32 inputs") { *value = rng.value(); }
        for value in slotted.as_mut_slice::<u16>().expect("BF16 inputs") {
            *value = bf16::from_f32(rng.value()).to_bits();
        }
        Self {
            input, slotted,
            ids: alloc(device, pairs, DType::U32),
            gathered: alloc(device, grouped_m8_gathered_rows(p.n_tokens, p.top_k, p.n_experts)
                .expect("guarded gather rows") * p.k as usize, DType::BF16),
            scratch: IdMmScratch::alloc(device, p.n_experts, p.n_tokens).expect("schedule scratch"),
            f32_output: std::array::from_fn(|_| alloc(device, elements, DType::F32)),
            bf16_output: std::array::from_fn(|_| alloc(device, elements, DType::BF16)),
            gate: std::array::from_fn(|_| alloc(device, elements, DType::BF16)),
            p,
        }
    }

    fn routes(&mut self, phase: usize, masked: bool) {
        for (pair, id) in self.ids.as_mut_slice::<u32>().expect("route IDs").iter_mut().enumerate() {
            let token = pair / TOP_K as usize;
            let slot = pair % TOP_K as usize;
            *id = if masked && (token + slot) % 5 == 0 { self.p.n_experts }
                else { (phase * 4 + (slot + token * 3) % 8) as u32 };
        }
    }

    fn poison(&mut self) {
        for output in &mut self.f32_output {
            output.as_mut_slice::<f32>().expect("F32 poison").fill(f32::from_bits(F32_POISON));
        }
        for output in self.bf16_output.iter_mut().chain(self.gate.iter_mut()) {
            output.as_mut_slice::<u16>().expect("BF16 poison").fill(BF16_POISON);
        }
    }

    fn output(&self, mode: Mode, arm: usize) -> &MlxBuffer {
        match mode {
            Mode::F32 | Mode::Bf16Input => &self.f32_output[arm],
            Mode::Bf16 => &self.gate[arm],
            _ => &self.bf16_output[arm],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project(&mut self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, prepared: &PreparedGroupedM8Experts, up: &PreparedGroupedM8Experts,
        arm: usize, mode: Mode, masked: bool, reuse_schedule: bool) {
        if matches!(mode, Mode::FusedGateUp) {
            assert!(!masked, "no public masked fused gate/up API");
            quantized_matmul_id_ggml_grouped_m8_gate_up(encoder, registry, device, &self.input,
                &self.gathered, prepared, up, &self.ids, &self.bf16_output[arm],
                &mut self.scratch, &self.p, GroupedM8Input::F32).expect("unchanged fused consumer");
            return;
        }
        let dispatch = if masked { quantized_matmul_id_ggml_grouped_m8_epilogue_masked }
            else { quantized_matmul_id_ggml_grouped_m8_epilogue };
        if matches!(mode, Mode::SiluMul) {
            dispatch(encoder, registry, device, &self.input, &self.gathered, prepared, &self.ids,
                &self.gate[arm], None, &mut self.scratch, &self.p, false, reuse_schedule, false,
                GroupedM8Input::F32, GroupedM8Epilogue::Bf16).expect("BF16 gate producer");
        }
        let (input, kind, slotted) = if matches!(mode, Mode::Bf16Input) {
            (&self.slotted, GroupedM8Input::Bf16, true)
        } else { (&self.input, GroupedM8Input::F32, false) };
        let (output, epilogue, aux) = match mode {
            Mode::F32 | Mode::Bf16Input => (&self.f32_output[arm], GroupedM8Epilogue::F32, None),
            Mode::Bf16 => (&self.gate[arm], GroupedM8Epilogue::Bf16, None),
            Mode::SiluMul => (&self.bf16_output[arm], GroupedM8Epilogue::SiluMulBf16, Some(&self.gate[arm])),
            Mode::FusedGateUp => unreachable!(),
        };
        let gate_ready = matches!(mode, Mode::SiluMul);
        dispatch(encoder, registry, device, input, &self.gathered, prepared, &self.ids, output,
            aux, &mut self.scratch, &self.p, slotted, reuse_schedule || gate_ready, gate_ready,
            kind, epilogue).expect("unchanged staged-M8 consumer");
    }

    fn live(&mut self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, source: &MlxBuffer, root: &MlxBuffer, masked: bool,
        reuse_schedule: bool) -> PreparedGroupedM8Experts {
        prepare_quantized_matmul_id_ggml_grouped_m8_live_into(encoder, registry, device,
            source, root, &self.ids, &mut self.scratch, &self.p, masked, reuse_schedule)
            .expect("live repack into the caller's one root")
    }
}

fn compare<T: Copy>(a: &[T], b: &[T], ids: &[u32], p: &GgmlQuantizedMatmulIdParams,
    bits: impl Fn(T) -> u32, value: impl Fn(T) -> f32, poison: u32) -> [usize; 4] {
    assert_eq!(a.len(), b.len());
    let mut counts = [0usize; 4];
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        counts[0] += usize::from(bits(a) != bits(b));
        if ids[index / p.n as usize] == p.n_experts {
            counts[1] += usize::from(bits(a) != poison || bits(b) != poison);
        } else {
            let (a, b) = (value(a), value(b));
            counts[2] += usize::from(!a.is_finite() || !b.is_finite());
            counts[3] += usize::from(a != 0.0 && b != 0.0);
        }
    }
    counts
}

fn check_output(work: &Work, mode: Mode) -> [usize; 4] {
    let ids = work.ids.as_slice::<u32>().expect("completed route IDs");
    let a = work.output(mode, 0);
    let b = work.output(mode, 1);
    if mode.f32_output() {
        compare(a.as_slice::<f32>().expect("immutable F32"), b.as_slice::<f32>().expect("live F32"),
            ids, &work.p, f32::to_bits, |v| v, F32_POISON)
    } else {
        compare(a.as_slice::<u16>().expect("immutable BF16"), b.as_slice::<u16>().expect("live BF16"),
            ids, &work.p, u32::from, |v| bf16::from_bits(v).to_f32(), u32::from(BF16_POISON))
    }
}

fn check_unselected(root: &MlxBuffer, work: &[Work], p: &GgmlQuantizedMatmulIdParams) -> usize {
    let selected: BTreeSet<_> = work.iter().flat_map(|w| w.ids.as_slice::<u32>().expect("completed IDs"))
        .copied().filter(|&id| id < p.n_experts).collect();
    let stride = plane_bytes(p) / p.n_experts as usize;
    let bytes = root.as_slice::<u8>().expect("completed caller root");
    let mut untouched = 0;
    for (expert, bytes) in bytes.chunks_exact(stride).enumerate() {
        if selected.contains(&(expert as u32)) {
            assert!(bytes.iter().any(|&b| b != PLANE_POISON), "selected expert {expert} was not repacked");
        } else {
            assert!(bytes.iter().all(|&b| b == PLANE_POISON), "unselected expert {expert} was written");
            untouched += bytes.len();
        }
    }
    assert!(untouched > 0);
    untouched
}

fn validation(device: &MlxDevice, registry: &mut KernelRegistry, start: Instant) -> Value {
    let p = params(3, 32, 64, 256, GgmlType::Q4_K);
    let bank = bank(device, registry, &p, 0x9981, start);
    let mut work = Work::new(device, p, 0x9982);
    work.routes(0, false);
    let root = alloc(device, plane_bytes(&p), DType::U8);
    let source_short = alloc(device, source_bytes(&p) - 1, DType::U8);
    let source_wrong_type = alloc(device, source_bytes(&p) / 4, DType::F32);
    let root_short = alloc(device, plane_bytes(&p) - 1, DType::U8);
    let root_wrong_type = alloc(device, plane_bytes(&p) / 4, DType::F32);
    let ids_short = alloc(device, (p.n_tokens * p.top_k - 1) as usize, DType::U32);
    let ids_wrong_type = alloc(device, (p.n_tokens * p.top_k) as usize, DType::F32);
    let mut rejected = Vec::new();
    for label in ["short_source", "source_dtype", "short_root", "root_dtype", "short_ids", "ids_dtype",
        "short_stride", "bad_n", "bad_k", "zero_experts", "unsupported_type", "source_extent_overflow", "unowned_reuse"] {
        let mut bad = p;
        match label {
            "short_stride" => bad.expert_stride -= 1,
            "bad_n" => bad.n -= 1,
            "bad_k" => bad.k -= 1,
            "zero_experts" => bad.n_experts = 0,
            "unsupported_type" => bad.ggml_type = GgmlType::Q4_0,
            "source_extent_overflow" => bad.expert_stride = u64::MAX,
            _ => (),
        }
        let source = match label {
            "short_source" => &source_short,
            "source_dtype" => &source_wrong_type,
            _ => &bank.source,
        };
        let output = match label {
            "short_root" => &root_short,
            "root_dtype" => &root_wrong_type,
            _ => &root,
        };
        let ids = match label {
            "short_ids" => &ids_short,
            "ids_dtype" => &ids_wrong_type,
            _ => &work.ids,
        };
        let mut encoder = device.command_encoder().expect("rejected call encoder");
        let before = mlx_native::dispatch_count();
        let error = match prepare_quantized_matmul_id_ggml_grouped_m8_live_into(&mut encoder,
            registry, device, source, output, ids, &mut work.scratch, &bad, false, label == "unowned_reuse") {
            Ok(_) => panic!("{label} unexpectedly accepted"),
            Err(error) => error,
        };
        assert_eq!(mlx_native::dispatch_count(), before, "{label} encoded before validation");
        rejected.push(json!({"case":label,"error":error.to_string(),"encoded_dispatches":0}));
    }
    for (experts, n, k, kind) in [(0, 64, 256, GgmlType::Q4_K),
        (32, 63, 256, GgmlType::Q4_K), (32, 64, 255, GgmlType::Q6_K),
        (32, 64, 256, GgmlType::Q4_0), (u32::MAX, u32::MAX - 63, u32::MAX - 255, GgmlType::Q6_K)] {
        assert!(grouped_m8_prepared_stack_bytes(experts, n, k, kind).is_err(), "invalid/overflow layout accepted");
    }
    json!({"rejected_before_encoding":rejected,"invalid_layout_cases":5})
}

fn sparse_boundaries(device: &MlxDevice, registry: &mut KernelRegistry, start: Instant) -> Value {
    let mut observations = Vec::new();
    for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
        let p = params(9, 32, 64, 256, kind);
        let bank = bank(device, registry, &p, 0xbc41, start);
        let mut work = Work::new(device, p, 0xbc42);
        work.poison();
        let ids = work.ids.as_mut_slice::<u32>().expect("sparse boundary IDs");
        ids.fill(p.n_experts);
        ids[0] = 0;
        *ids.last_mut().unwrap() = p.n_experts - 1;
        let mut root = alloc(device, plane_bytes(&p), DType::U8);
        root.as_mut_slice::<u8>().expect("boundary root poison").fill(PLANE_POISON);
        let mut encoder = device.command_encoder().expect("first/last expert sparse proof");
        let live = work.live(&mut encoder, registry, device, &bank.source, &root, true, false);
        work.project(&mut encoder, registry, device, &live, &bank.prepared, 1, Mode::F32, true, true);
        work.project(&mut encoder, registry, device, &bank.prepared, &bank.prepared,
            0, Mode::F32, true, false);
        encoder.commit_and_wait().expect("checked sparse boundary proof");
        let [mismatches, masked_writes, nonfinite_live, nonzero_live] = check_output(&work, Mode::F32);
        let untouched = check_unselected(&root, std::slice::from_ref(&work), &p);
        assert_eq!((mismatches, masked_writes, nonfinite_live), (0, 0, 0));
        assert!(nonzero_live > 0);
        assert_eq!(untouched, plane_bytes(&p) / 32 * 30);
        let observation = json!({"format":format!("{kind:?}"),"selected_experts":[0,31],
            "live_routes":2,"masked_routes":p.n_tokens*p.top_k-2,"mismatch_elements":mismatches,
            "masked_writes":masked_writes,"nonfinite_live":nonfinite_live,"nonzero_live":nonzero_live,
            "unselected_experts":30,"untouched_unselected_bytes":untouched});
        eprintln!("{}", json!({"kind":"live_m8_sparse_boundary_case","observation":observation}));
        observations.push(observation);
        budget(start);
    }
    json!({"cases":observations})
}

fn no_write_schedules(device: &MlxDevice, registry: &mut KernelRegistry, start: Instant) -> usize {
    let mut cases = 0;
    for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
        let p = params(9, 32, 64, 256, kind);
        let bank = bank(device, registry, &p, 0xa911, start);
        for label in ["all_masked_empty", "out_of_range", "duplicate", "unmasked_sentinel"] {
            budget(start);
            let masked = label != "unmasked_sentinel";
            let mut work = Work::new(device, p, 0xa912);
            work.routes(0, masked);
            work.poison();
            let ids = work.ids.as_mut_slice::<u32>().expect("invalid route setup");
            match label {
                "all_masked_empty" => ids.fill(p.n_experts),
                "out_of_range" => ids[0] = p.n_experts + 1,
                "duplicate" => { ids[0] = 1; ids[1] = 1; },
                "unmasked_sentinel" => ids[0] = p.n_experts,
                _ => unreachable!(),
            }
            let mut root = alloc(device, plane_bytes(&p), DType::U8);
            root.as_mut_slice::<u8>().expect("caller root poison").fill(PLANE_POISON);
            let mut encoder = device.command_encoder().expect("empty/poison schedule command");
            work.project(&mut encoder, registry, device, &bank.prepared, &bank.prepared,
                0, Mode::F32, masked, false);
            let prepared = work.live(&mut encoder, registry, device, &bank.source, &root, masked, true);
            work.project(&mut encoder, registry, device, &prepared, &bank.prepared,
                1, Mode::F32, masked, true);
            encoder.commit_and_wait().expect("checked empty/poison wait");
            assert!(root.as_slice::<u8>().expect("root after poison schedule").iter().all(|&b| b == PLANE_POISON));
            for output in &work.f32_output {
                assert!(output.as_slice::<f32>().expect("output after poison schedule").iter()
                    .all(|v| v.to_bits() == F32_POISON), "{label} wrote output");
            }
            let state = work.scratch.htpe().as_slice::<u32>().expect("completed compact state")[0];
            assert_eq!(state, if label == "all_masked_empty" { 0 } else { 0x8000_0000 });
            eprintln!("{}", json!({"kind":"live_m8_no_write_case","format":format!("{kind:?}"),
                "schedule":label,"state":state,"root_unchanged_bytes":plane_bytes(&p),"output_writes":0}));
            cases += 1;
        }
    }
    cases
}

fn parity(device: &MlxDevice, registry: &mut KernelRegistry, start: Instant) -> Value {
    let validation = validation(device, registry, start);
    let no_write_cases = no_write_schedules(device, registry, start);
    let sparse_boundaries = sparse_boundaries(device, registry, start);
    let mut cases = 0;
    let mut compared_elements = 0;
    let mut poison_bytes_checked = 0;
    let mut split_descriptor_cases = 0;
    for kind in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for (n, k) in [(64, 256), (128, 512)] {
            let base = params(1, 32, n, k, kind);
            let banks = [bank(device, registry, &base, 0x9102_3491, start),
                bank(device, registry, &base, 0x7359_1023, start)];
            assert_ne!(banks[0].source.contents_ptr(), banks[1].source.contents_ptr());
            let mut root = alloc(device, plane_bytes(&base), DType::U8);
            for rows in [1, 3, 7, 8, 9, 17, 31, 32, 33, 65] {
                let p = params(rows, 32, n, k, kind);
                for masked in [false, true] {
                    let mut work: [Work; 2] = std::array::from_fn(|phase| {
                        let mut w = Work::new(device, p, 0x5a91_8230 + u64::from(rows));
                        w.routes(phase, masked);
                        w
                    });
                    for mode in [Mode::F32, Mode::Bf16, Mode::Bf16Input, Mode::SiluMul, Mode::FusedGateUp] {
                        if masked && matches!(mode, Mode::FusedGateUp) { continue; }
                        budget(start);
                        root.as_mut_slice::<u8>().expect("poison caller plane").fill(PLANE_POISON);
                        for w in &mut work { w.poison(); }
                        let mut encoder = device.command_encoder().expect("back-to-back live parity command");
                        // Comparator schedules are encoder-owned and remain valid for
                        // both later live preparations, which exercise reuse=true.
                        for (phase, w) in work.iter_mut().enumerate() {
                            w.project(&mut encoder, registry, device, &banks[phase].prepared,
                                &banks[1 - phase].prepared, 0, mode, masked, false);
                        }
                        let reuse = (cases / 2) % 2 == 0;
                        // No caller barrier or host wait between these two overwrite/read
                        // phases. Distinct source stacks and overlapping expert sets share
                        // precisely the same supplied root; the live API owns WAR/RAW.
                        for (phase, w) in work.iter_mut().enumerate() {
                            let prepared = w.live(&mut encoder, registry, device, &banks[phase].source,
                                &root, masked, reuse);
                            w.project(&mut encoder, registry, device, &prepared,
                                &banks[1 - phase].prepared, 1, mode, masked, true);
                        }
                        encoder.commit_and_wait().expect("checked two-phase parity wait");
                        for (phase, w) in work.iter_mut().enumerate() {
                            let [mismatches, masked_writes, nonfinite_live, nonzero_live] = check_output(w, mode);
                            let ids = w.ids.as_slice::<u32>().expect("completed parity IDs");
                            let mut counts = vec![0usize; p.n_experts as usize];
                            for &id in ids { if id < p.n_experts { counts[id as usize] += 1; } }
                            let max_routes = *counts.iter().max().unwrap();
                            let split = counts.iter().filter(|&&count| count > 32).count();
                            if rows == 65 { assert!(split > 0, "must exercise multiple descriptors per expert"); }
                            split_descriptor_cases += usize::from(split > 0);
                            cases += 1;
                            compared_elements += w.output(mode, 0).data_byte_len()
                                / if mode.f32_output() { 4 } else { 2 };
                            eprintln!("{}", json!({"kind":"live_m8_parity_case","case":cases,
                                "format":format!("{kind:?}"),"n":n,"k":k,"rows":rows,"phase":phase,
                                "masked":masked,"operation":mode.name(),"reuse_prepare_schedule":reuse,
                                "max_routes_per_expert":max_routes,"experts_with_split_descriptors":split,
                                "mismatch_elements":mismatches,"masked_writes":masked_writes,
                                "nonfinite_live_elements":nonfinite_live,"nonzero_live_elements":nonzero_live}));
                            assert_eq!((mismatches, masked_writes, nonfinite_live), (0, 0, 0));
                            assert!(nonzero_live > 0, "no trivial all-zero live result");
                        }
                        // Both phases use the same inputs but different weights and IDs.
                        assert!(work[0].output(mode, 1).as_slice::<u8>().expect("phase A bits")
                            != work[1].output(mode, 1).as_slice::<u8>().expect("phase B bits"),
                            "source-switch witness must be nontrivial");
                        poison_bytes_checked += check_unselected(&root, &work, &p);
                        budget(start);
                    }
                }
            }
        }
    }
    assert_eq!(cases, 720);
    assert!(split_descriptor_cases > 0);
    json!({"exact_case_count":cases,"compared_elements":compared_elements,"mismatch_elements":0,
        "untouched_unselected_bytes_checked":poison_bytes_checked,"split_descriptor_cases":split_descriptor_cases,
        "no_write_schedule_cases":no_write_cases,"validation":validation,
        "sparse_boundaries":sparse_boundaries,
        "caller_roots_per_geometry":1,"back_to_back_distinct_sources":2,"host_waits_between_phases":0})
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    (values[middle - 1] + values[middle]) * 0.5
}

#[test]
#[ignore = "unpublished live prepared-plane parity and frozen actual-route paired GPU timing spike"]
fn grouped_m8_live_prepare_spike() {
    assert!(std::env::var_os("MLX_PROFILE_CB").is_none(), "unset MLX_PROFILE_CB for this timing probe");
    let start = Instant::now();
    // Require and authenticate the historical corpus before any GPU work.
    let corpus = live_m8_recorded::Corpus::load();
    let device = MlxDevice::new().expect("Metal tensor device");
    let mut registry = KernelRegistry::new();
    let parity = parity(&device, &mut registry, start);
    eprintln!("{}", json!({"kind":"live_m8_parity_summary","parity":parity}));
    // Small fixtures and their immutable comparators have dropped before the
    // forty production stacks are allocated. Exactly one candidate plane lives.
    live_m8_recorded::run(&device, &mut registry, corpus, parity, start);
}
