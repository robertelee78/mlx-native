#![cfg(mlx_native_has_metal_tensor_artifact)]

// Unpublished, model-free spike. This is neither release qualification nor a fit claim.
use half::bf16;
use mlx_native::ops::quantized_matmul_id_ggml::bind_quantized_matmul_id_ggml_grouped_m8_raw;
use mlx_native::{
    ggml_expert_bytes, grouped_m8_gathered_rows, prepare_quantized_matmul_id_ggml_grouped_m8,
    quantized_matmul_id_ggml_grouped_m8_epilogue,
    quantized_matmul_id_ggml_grouped_m8_epilogue_masked,
    quantized_matmul_id_ggml_grouped_m8_gate_up, CommandEncoder, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, GroupedM8Epilogue, GroupedM8Input,
    IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice, PreparedGroupedM8Experts,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

const EXPERTS: u32 = 256;
const N: u32 = 512;
const K: u32 = 2048;
const TOP_K: u32 = 8;
const BANKS: usize = 40;
const TIMING_ROWS: u32 = 17;
const ROWS: [u32; 7] = [1, 2, 3, 8, 9, 17, 32];
const F32_POISON: u32 = 0x7fc1_2345;
const BF16_POISON: u16 = 0x7fc1;

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

fn params(rows: u32) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: rows,
        top_k: TOP_K,
        n: N,
        k: K,
        n_experts: EXPERTS,
        expert_stride: u64::from(N) * u64::from(K / 256) * 144,
        ggml_type: GgmlType::Q4_K,
    }
}

fn alloc(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    device.alloc_buffer(elements * dtype.size_of(), dtype, vec![elements]).expect("allocate probe buffer")
}

fn source_bytes() -> usize {
    let p = params(1);
    ggml_expert_bytes(p.ggml_type, EXPERTS, N, K, p.expert_stride).expect("source extent") as usize
}

fn prepared_bytes() -> usize {
    // Existing Q4 staged-M8 low plane plus metadata; not an allocator measurement.
    EXPERTS as usize * N as usize * ((K / 32) as usize * 16 + (K / 256) as usize * 16)
}

struct Bank {
    prepared: PreparedGroupedM8Experts,
    raw: PreparedGroupedM8Experts,
    source_address: usize,
    seed: u64,
}

fn bank(device: &MlxDevice, registry: &mut KernelRegistry, seed: u64, start: Instant) -> Bank {
    budget(start);
    let mut source = alloc(device, source_bytes(), DType::U8);
    let mut rng = Rng(seed);
    for (index, block) in source.as_mut_slice::<u8>().expect("Q4 source").chunks_exact_mut(144).enumerate() {
        for chunk in block.chunks_exact_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes());
        }
        // Finite, nonzero half scales: model-like magnitudes, small normals,
        // minimum normal, and both tiny and large half subnormals. All packed
        // 6-bit scale/min fields and nibbles remain nontrivial valid Q4_K data.
        for offset in [0usize, 2] {
            let mantissa = (rng.next() as u16 & 0x03ff).max(1);
            let magnitude = match (index + offset / 2) % 6 {
                0 => 1,
                1 => mantissa,
                2 => 0x0400,
                3 => (2 << 10) | mantissa,
                _ => (7 << 10) | mantissa,
            };
            let sign = if offset == 0 && rng.next() & 1 != 0 { 0x8000 } else { 0 };
            block[offset..offset + 2].copy_from_slice(&(magnitude | sign).to_le_bytes());
        }
    }
    let p = params(1);
    let raw = bind_quantized_matmul_id_ggml_grouped_m8_raw(&source, &p).expect("bind actual raw Q4 source");
    // No CPU slice or GPU access remains live here. Change the source after
    // binding, before preparation: parity also exercises aliasing rather than
    // a snapshot taken by the binder. Each block changes a decoded nibble.
    for block in source.as_mut_slice::<u8>().expect("source alias witness").chunks_exact_mut(144) {
        block[16] ^= 0x5a;
    }
    let source_address = source.contents_ptr() as usize + source.byte_offset() as usize;
    let mut encoder = device.command_encoder().expect("preparation encoder");
    let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(&mut encoder, registry, device, &source, &p)
        .expect("prepare comparator from same mutated source");
    encoder.commit_and_wait().expect("checked preparation wait");
    budget(start);
    // Both handles must keep their source alive after this local owner drops.
    Bank { prepared, raw, source_address, seed }
}

#[derive(Clone, Copy)]
enum Mode {
    F32,
    Bf16,
    Bf16Input,
    SiluMul,
    FusedGateUp,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32_projection",
            Self::Bf16 => "bf16_gate",
            Self::Bf16Input => "slotted_bf16_input_f32_projection",
            Self::SiluMul => "bf16_gate_consumed_by_silu_mul",
            Self::FusedGateUp => "fused_gate_up_bf16",
        }
    }

    fn f32_output(self) -> bool {
        matches!(self, Self::F32 | Self::Bf16Input)
    }
}

struct Work {
    p: GgmlQuantizedMatmulIdParams,
    input: MlxBuffer,
    slotted_bf16: MlxBuffer,
    ids: MlxBuffer,
    gathered: MlxBuffer,
    scratch: IdMmScratch,
    f32_output: [MlxBuffer; 2],
    bf16_output: [MlxBuffer; 2],
    gate: [MlxBuffer; 2],
}

impl Work {
    fn new(device: &MlxDevice, rows: u32) -> Self {
        let p = params(rows);
        let pairs = (rows * TOP_K) as usize;
        let elements = pairs * N as usize;
        let mut input = alloc(device, rows as usize * K as usize, DType::F32);
        let mut slotted_bf16 = alloc(device, pairs * K as usize, DType::BF16);
        let mut rng = Rng(0x5a91_8230 + u64::from(rows));
        for value in input.as_mut_slice::<f32>().expect("F32 inputs") {
            *value = rng.value();
        }
        for value in slotted_bf16.as_mut_slice::<u16>().expect("BF16 inputs") {
            *value = bf16::from_f32(rng.value()).to_bits();
        }
        Self {
            p,
            input,
            slotted_bf16,
            ids: alloc(device, pairs, DType::U32),
            gathered: alloc(device, grouped_m8_gathered_rows(rows, TOP_K, EXPERTS).expect("guarded gather rows") * K as usize, DType::BF16),
            scratch: IdMmScratch::alloc(device, EXPERTS, rows).expect("schedule scratch"),
            f32_output: std::array::from_fn(|_| alloc(device, elements, DType::F32)),
            bf16_output: std::array::from_fn(|_| alloc(device, elements, DType::BF16)),
            gate: std::array::from_fn(|_| alloc(device, elements, DType::BF16)),
        }
    }

    fn routes(&mut self, repeated: bool, masked: bool, layer: usize) {
        for (pair, id) in self.ids.as_mut_slice::<u32>().expect("route ids").iter_mut().enumerate() {
            let token = pair / TOP_K as usize;
            let slot = pair % TOP_K as usize;
            *id = if masked && (token + slot) % 5 == 0 {
                EXPERTS
            } else {
                let route = if repeated { slot * 29 } else { pair * 73 };
                ((route + layer * 19) % EXPERTS as usize) as u32
            };
        }
    }

    fn output(&self, mode: Mode, arm: usize) -> &MlxBuffer {
        match mode {
            Mode::F32 | Mode::Bf16Input => &self.f32_output[arm],
            Mode::Bf16 => &self.gate[arm],
            _ => &self.bf16_output[arm],
        }
    }

    fn poison(&mut self, mode: Mode) {
        let outputs = match mode {
            Mode::F32 | Mode::Bf16Input => &mut self.f32_output,
            Mode::Bf16 => &mut self.gate,
            _ => &mut self.bf16_output,
        };
        for output in outputs {
            if mode.f32_output() {
                output.as_mut_slice::<f32>().expect("F32 poison").fill(f32::from_bits(F32_POISON));
            } else {
                output.as_mut_slice::<u16>().expect("BF16 poison").fill(BF16_POISON);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(&mut self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
        gate: &Bank, up: &Bank, arm: usize, mode: Mode, masked: bool, reuse: bool) {
        let projection = if matches!(mode, Mode::SiluMul) { up } else { gate };
        let weight = if arm == 0 { &projection.prepared } else { &projection.raw };
        if matches!(mode, Mode::FusedGateUp) {
            assert!(!masked, "there is no public masked fused gate/up API");
            let up = if arm == 0 { &up.prepared } else { &up.raw };
            quantized_matmul_id_ggml_grouped_m8_gate_up(encoder, registry, device, &self.input,
                &self.gathered, weight, up, &self.ids, &self.bf16_output[arm], &mut self.scratch,
                &self.p, GroupedM8Input::F32).expect("same fused gate/up dispatch API");
            return;
        }
        let (input, input_kind, slotted) = if matches!(mode, Mode::Bf16Input) {
            (&self.slotted_bf16, GroupedM8Input::Bf16, true)
        } else {
            (&self.input, GroupedM8Input::F32, false)
        };
        let (output, epilogue, aux) = match mode {
            Mode::F32 | Mode::Bf16Input => (&self.f32_output[arm], GroupedM8Epilogue::F32, None),
            Mode::Bf16 => (&self.gate[arm], GroupedM8Epilogue::Bf16, None),
            Mode::SiluMul => (&self.bf16_output[arm], GroupedM8Epilogue::SiluMulBf16, Some(&self.gate[arm])),
            Mode::FusedGateUp => unreachable!(),
        };
        let dispatch = if masked { quantized_matmul_id_ggml_grouped_m8_epilogue_masked }
            else { quantized_matmul_id_ggml_grouped_m8_epilogue };
        dispatch(encoder, registry, device, input, &self.gathered, weight, &self.ids, output,
            aux, &mut self.scratch, &self.p, slotted, reuse, reuse, input_kind, epilogue)
            .expect("same staged-M8 epilogue dispatch API");
    }
}

fn compare<T: Copy>(
    a: &[T], b: &[T], ids: &[u32], bits: impl Fn(T) -> u32,
    value: impl Fn(T) -> f32, poison: u32,
) -> [usize; 4] {
    assert_eq!(a.len(), b.len());
    let mut counts = [0usize; 4];
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        let (abits, bbits) = (bits(a), bits(b));
        counts[0] += usize::from(abits != bbits);
        if ids[index / N as usize] == EXPERTS {
            counts[1] += usize::from(abits != poison || bbits != poison);
        } else {
            let (a, b) = (value(a), value(b));
            counts[2] += usize::from(!a.is_finite() || !b.is_finite());
            counts[3] += usize::from(a != 0.0 && b != 0.0);
        }
    }
    counts
}

fn rejection(result: Result<(), mlx_native::MlxError>, before: u64, label: &str, fragment: &str) -> Value {
    let message = match result {
        Err(mlx_native::MlxError::InvalidArgument(message)) => message,
        other => panic!("{label}: expected InvalidArgument, got {other:?}"),
    };
    assert!(message.contains(fragment), "{label}: wrong rejection: {message}");
    let dispatches = mlx_native::dispatch_count() - before;
    assert_eq!(dispatches, 0, "{label}: rejection encoded GPU work");
    json!({"case":label, "error":message, "encoded_dispatches":dispatches})
}

fn rejected_consumers(device: &MlxDevice, registry: &mut KernelRegistry, gate: &Bank, up: &Bank) -> Value {
    let mut work = Work::new(device, 1);
    let mut encoder = device.command_encoder().expect("rejection encoder");
    let mut results = Vec::new();
    for (a, b, label) in [
        (&gate.prepared, &up.raw, "fused_prepared_gate_raw_up"),
        (&gate.raw, &up.prepared, "fused_raw_gate_prepared_up"),
    ] {
        let before = mlx_native::dispatch_count();
        let result = quantized_matmul_id_ggml_grouped_m8_gate_up(&mut encoder, registry, device,
            &work.input, &work.gathered, a, b, &work.ids, &work.bf16_output[0],
            &mut work.scratch, &work.p, GroupedM8Input::F32);
        results.push(rejection(result, before, label, "layout"));
    }
    // Minimal legal source geometries isolate stride incompatibility without
    // allocating another production stack. Neither handle is dispatched.
    let small = GgmlQuantizedMatmulIdParams {
        n_tokens: 1, top_k: 1, n: 256, k: 256, n_experts: 2,
        expert_stride: 256 * 144, ggml_type: GgmlType::Q4_K,
    };
    let padded = GgmlQuantizedMatmulIdParams { expert_stride: small.expert_stride + 16, ..small };
    let mut source = alloc(device, (2 * small.expert_stride + 16) as usize, DType::U8);
    source.as_mut_slice::<u8>().expect("small valid Q4 storage").fill(0x5a);
    let dense = bind_quantized_matmul_id_ggml_grouped_m8_raw(&source, &small).expect("dense raw binding");
    let padded = bind_quantized_matmul_id_ggml_grouped_m8_raw(&source, &padded).expect("padded raw binding");
    let before = mlx_native::dispatch_count();
    let result = quantized_matmul_id_ggml_grouped_m8_gate_up(&mut encoder, registry, device,
        &work.input, &work.gathered, &dense, &padded, &work.ids, &work.bf16_output[0],
        &mut work.scratch, &small, GroupedM8Input::F32);
    results.push(rejection(result, before, "fused_unequal_raw_strides", "stride"));

    let grids = alloc(device, mlx_native::MOE_TILES_GRID_WORDS, DType::U32);
    let buffers = mlx_native::MoeTilesBuffers {
        gathered: &work.gathered, intermediate: &work.gathered, grids: &grids,
    };
    let tiles = mlx_native::MoeTilesParams {
        n_tokens: 1, top_k: TOP_K, n_experts: EXPERTS, hidden: K, intermediate: N,
        masked_routes: false, column_tile: false, shared_planes: false,
    };
    let before = mlx_native::dispatch_count();
    let result = mlx_native::moe_tiles_gate_up(&mut encoder, registry, device, &gate.raw,
        &up.raw, &work.ids, &buffers, &work.scratch, &tiles);
    results.push(rejection(result, before, "prepared_only_moe_tiles_gate_up", "raw"));
    let down_tiles = mlx_native::MoeTilesParams { hidden: N, intermediate: K, ..tiles };
    let before = mlx_native::dispatch_count();
    let result = mlx_native::moe_tiles_down(&mut encoder, registry, device, &gate.raw,
        &work.ids, &buffers, &work.scratch, &work.f32_output[0], &down_tiles);
    results.push(rejection(result, before, "prepared_only_moe_tiles_down", "raw"));
    let weights = mlx_native::MoeTilesFusedWeights {
        gate: &gate.raw, up: &up.raw, down: &gate.raw,
        shared_gate: &gate.raw, shared_up: &up.raw, shared_down: &gate.raw,
    };
    let fused = mlx_native::MoeTilesFusedParams {
        n_tokens: 1, top_k: TOP_K, n_experts: EXPERTS, hidden: K, intermediate: N, column_tile: false,
    };
    let before = mlx_native::dispatch_count();
    let result = mlx_native::moe_tiles_fused_admits(&weights, &fused);
    results.push(rejection(result, before, "prepared_only_moe_tiles_fused_admits", "raw"));
    json!({"exact_case_count":results.len(), "cases":results})
}

fn parity(device: &MlxDevice, registry: &mut KernelRegistry, start: Instant) -> Value {
    let gate = bank(device, registry, 0x9102_3491, start);
    let up = bank(device, registry, 0x7359_1023, start);
    let rejected_consumers = rejected_consumers(device, registry, &gate, &up);
    let mut cases = 0usize;
    let mut compared_elements = 0usize;
    let mut mismatch_elements = 0usize;
    for rows in ROWS {
        let mut work = Work::new(device, rows);
        for repeated in [false, true] {
            for masked in [false, true] {
                work.routes(repeated, masked, 0);
                for mode in [Mode::F32, Mode::Bf16, Mode::Bf16Input, Mode::SiluMul, Mode::FusedGateUp] {
                    if masked && matches!(mode, Mode::FusedGateUp) { continue; }
                    budget(start);
                    work.poison(mode);
                    let mut encoder = device.command_encoder().expect("parity encoder");
                    let first = cases % 2;
                    work.encode(&mut encoder, registry, device, &gate, &up, first, mode, masked, false);
                    // Alternate which representation builds/gathers and which
                    // reuses the SAME encoder-owned schedule and gathered input.
                    work.encode(&mut encoder, registry, device, &gate, &up, 1 - first, mode, masked,
                        !matches!(mode, Mode::FusedGateUp));
                    encoder.commit_and_wait().expect("checked parity wait");
                    let a = work.output(mode, 0);
                    let b = work.output(mode, 1);
                    let ids = work.ids.as_slice::<u32>().expect("checked route ids");
                    let elements = (rows * TOP_K * N) as usize;
                    let [mismatches, masked_writes, nonfinite_live, nonzero_live] = if mode.f32_output() {
                        compare(a.as_slice::<f32>().expect("prepared F32"),
                            b.as_slice::<f32>().expect("raw F32"), ids,
                            f32::to_bits, |v| v, F32_POISON)
                    } else {
                        compare(a.as_slice::<u16>().expect("prepared BF16"),
                            b.as_slice::<u16>().expect("raw BF16"), ids,
                            u32::from, |v| bf16::from_bits(v).to_f32(), u32::from(BF16_POISON))
                    };
                    cases += 1;
                    compared_elements += elements;
                    mismatch_elements += mismatches;
                    println!("{}", json!({"kind":"raw_m8_spike_parity_case", "case":cases,
                        "rows":rows, "schedule":if repeated { "repeated_eight_experts" } else { "dispersed" },
                        "masked":masked, "operation":mode.name(), "compared_elements":elements,
                        "mismatch_elements":mismatches, "masked_writes":masked_writes,
                        "nonfinite_live_elements":nonfinite_live, "nonzero_live_elements":nonzero_live}));
                    assert_eq!(mismatches, 0, "bitwise parity failed in case {cases}");
                    assert_eq!(masked_writes, 0, "masked route written in case {cases}");
                    assert_eq!(nonfinite_live, 0, "unwritten/nonfinite live output in case {cases}");
                    assert!(nonzero_live > 0, "trivial all-zero result in case {cases}");
                    budget(start);
                }
            }
        }
    }
    assert_eq!(cases, 126);
    json!({"exact_case_count":cases, "compared_elements":compared_elements,
        "rejected_consumers":rejected_consumers,
        "mismatch_elements":mismatch_elements, "masked_route_cases":56,
        "raw_alias_evidence":"bind before source-nibble mutation; prepare after mutation; bitwise parity after local source owner drops",
        "source_stacks":2, "source_bytes_per_stack":source_bytes(),
        "prepared_layout_bytes_per_stack":prepared_bytes()})
}

fn timed_projection(device: &MlxDevice, registry: &mut KernelRegistry, work: &mut Work,
    bank: &Bank, arm: usize, start: Instant) -> f64 {
    budget(start);
    let mut encoder = device.command_encoder().expect("timing encoder");
    // One ordinary F32 projection INCLUDING its compact schedule and BF16
    // gather. Preparation, input generation and CPU waits are not GPU timing.
    work.encode(&mut encoder, registry, device, bank, bank, arm, Mode::F32, false, false);
    let (begin, end) = encoder.commit_wait_with_gpu_time().expect("checked GPU timestamp wait");
    assert!(begin.is_finite() && end.is_finite() && begin > 0.0 && end > begin,
        "missing/invalid command-buffer GPU timestamps: {begin}, {end}");
    budget(start);
    (end - begin) * 1.0e3
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    (values[middle - 1] + values[middle]) * 0.5
}

#[test]
#[ignore = "unpublished raw Q4 staged-M8 parity and bounded 40-stack paired GPU timing spike"]
fn grouped_m8_raw_q4_production_gate_spike() {
    assert!(std::env::var_os("MLX_PROFILE_CB").is_none(), "unset MLX_PROFILE_CB for this timing probe");
    let start = Instant::now();
    let device = MlxDevice::new().expect("Metal device with grouped-M8 tensor support");
    let mut registry = KernelRegistry::new();
    let parity = parity(&device, &mut registry, start);
    println!("{}", json!({"kind":"raw_m8_spike_parity_summary", "parity":parity}));

    // Parity's two stacks are dropped before allocating the timing banks.
    let mut banks = Vec::with_capacity(BANKS);
    for layer in 0..BANKS {
        banks.push(bank(&device, &mut registry, 0x8410_0000 + layer as u64, start));
    }
    let unique_sources: BTreeSet<_> = banks.iter().map(|bank| bank.source_address).collect();
    assert_eq!(unique_sources.len(), BANKS, "timing banks must have distinct live source allocations");
    let live_weight_bytes = BANKS * (source_bytes() + prepared_bytes());
    assert!(live_weight_bytes < 12 * 1024 * 1024 * 1024usize);
    let mut work = Work::new(&device, TIMING_ROWS);
    for (layer, bank) in banks.iter().enumerate() {
        work.routes(true, false, layer);
        for arm in [0, 1] {
            let _ = timed_projection(&device, &mut registry, &mut work, bank, arm, start);
        }
    }

    let mut prepared_ms = Vec::with_capacity(2 * BANKS);
    let mut raw_ms = Vec::with_capacity(2 * BANKS);
    let mut observations = Vec::with_capacity(2 * BANKS);
    for rotation in 0..2 {
        for (layer, bank) in banks.iter().enumerate() {
            work.routes(true, false, layer);
            // Alternate the first representation across all eighty pairs,
            // including the boundary between the two full bank rotations.
            let first = observations.len() % 2;
            let mut elapsed = [0.0; 2];
            for arm in [first, 1 - first] {
                elapsed[arm] = timed_projection(&device, &mut registry, &mut work, bank, arm, start);
            }
            prepared_ms.push(elapsed[0]);
            raw_ms.push(elapsed[1]);
            let observation = json!({"pair":observations.len(), "rotation":rotation, "layer":layer,
                "first":if first == 0 { "prepared" } else { "raw" },
                "prepared_gpu_ms":elapsed[0], "raw_gpu_ms":elapsed[1]});
            println!("{}", json!({"kind":"raw_m8_spike_pair", "observation":observation}));
            observations.push(observation);
        }
    }
    budget(start);
    assert_eq!(observations.len(), 80);
    println!("{}", json!({
        "kind":"unpublished_raw_m8_spike", "release_qualification":false,
        "source":{"format":"Q4_K", "shape":[EXPERTS,N,K], "top_k":TOP_K,
            "expert_stride_bytes":params(TIMING_ROWS).expert_stride, "timed_rows":TIMING_ROWS},
        "parity":parity,
        "allocation_extents":{"measurement":"requested source allocation and existing prepared-layout formula; not allocator telemetry",
            "timing_source_stacks":BANKS, "distinct_source_allocations":unique_sources.len(),
            "source_bytes_per_stack":source_bytes(), "prepared_bytes_per_stack":prepared_bytes(),
            "raw_handle_full_weight_bytes_expected_by_contract":0,
            "private_handle_allocation_bytes_observed":null, "timing_live_weight_bytes_by_contract":live_weight_bytes,
            "scratch_reused":true, "gathered_bytes":work.gathered.data_byte_len(),
            "input_f32_bytes":work.input.data_byte_len(), "output_f32_bytes_per_arm":work.f32_output[0].data_byte_len(),
            "sources":banks.iter().enumerate().map(|(layer, bank)| json!({"layer":layer,
                "seed":bank.seed, "logical_source_address":format!("0x{:x}", bank.source_address),
                "prepared_and_raw_bind_same_source":true})).collect::<Vec<_>>()},
        "timing":{"clock":"MTLCommandBuffer GPUEndTime - GPUStartTime; checked completion waits",
            "measured_operation":"one F32-input/F32-output Q4_K grouped-M8 gate projection including compact schedule and BF16 gather; no preparation or fused gate/up",
            "schedule":"same repeated eight experts within each layer; expert set rotates across 40 layers",
            "banks":BANKS, "warm_observations":2*BANKS, "rotations":2, "paired_observations":observations.len(),
            "prepared_median_gpu_ms":median(&mut prepared_ms), "raw_median_gpu_ms":median(&mut raw_ms),
            "observations":observations},
        "elapsed_wall_seconds":start.elapsed().as_secs_f64()
    }));
}
