//! ADR-063 Q4-GEMMA-PREFILL-MOE §5.1 — compact `mm_id` tile schedule (P2),
//! schedule reuse (P2b) and half-tile skip (P2c), model-free and bitwise.
//!
//! Every comparison here is on output bytes. There is no tolerance: the
//! compact schedule, the reusing entry point and every shipped half-tile
//! skip are Class A against the uniform grid.
//!
//! * N1 — slotted == flattened (and both schedules agree), every `mm_id`
//!   type × {tensor, simdgroup} × `n_tokens` × four routings × the Gemma 4
//!   down shape (704 → 2816, 128 experts, top-8; 32-value-block types only,
//!   since 704 is not a multiple of 256) and a Qwen-like shape (512 → 2048,
//!   256 experts, top-8). N7 (row independence) runs on the same batches.
//! * N2 — `CompactTiles` == `UniformGrid` for the auto, pooled shared,
//!   pooled pair and pooled slotted entry points × top-k {1, 6, 8} × the
//!   Gemma gate/up, Gemma down and a DeepSeek-V4-like shape.
//! * N3 — invalid metadata poisons the whole output on both schedules and
//!   schedule writes stay inside their extent.
//! * N4 — the P2b schedule key: every mismatch fails before encoding; a
//!   valid reuse (also across `commit_wait_and_rotate`) equals a fresh
//!   schedule.
//! * N5 — half-tile skip against the 32-row body on tiles with `nr1` ∈ {1,
//!   15, 16, 17, 31, 32}; full tiles must always match, production must
//!   match the 32-row body, and the partial-tile result is recorded per type
//!   and path.
//! * N6 — capacity: the `max_tiles` bound is attained and the device
//!   schedule equals a host-built one; undersized scratch fails; the
//!   `alloc` constructor serves the compact schedule.
//! * N7 — row independence (inside N1).
//! * N9 — scratch compatibility and accessor contracts.
//!
//! N8 is the existing suites, run unchanged.
//!
//! The DeepSeek-V4-like shape keeps the routing geometry (256 experts,
//! top-6, K = 4096) with N reduced to 128 so its weight stack fits a test
//! process; N only changes the N-tile count, which both schedules share.
//!
//! Run on both metallib paths, one test at a time:
//!
//! ```text
//! MLX_PRECOMPILED_METALLIB=1 cargo test --release --locked --test test_mm_id_compact_schedule -- --test-threads=1 --nocapture
//! MLX_PRECOMPILED_METALLIB=0 cargo test --release --locked --test test_mm_id_compact_schedule -- --test-threads=1 --nocapture
//! ```
//!
//! `MTL_SHADER_VALIDATION=1` additionally turns any out-of-bounds weight
//! read in N3 (the invalid id addresses one slab past the exact-size weight
//! buffer) into a command-buffer error.

use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_map0_for_test, dispatch_id_mm_pooled_with_options_for_test,
    tensor_half_tile_skip_validated, GgmlIdMmDispatchParams, HalfTileSkip, IdMmScheduleOptions,
};
use mlx_native::{
    ggml_expert_bytes, quantized_matmul_id_ggml_pooled_pair_with_policy,
    quantized_matmul_id_ggml_pooled_slotted_reusing_schedule_with_policy,
    quantized_matmul_id_ggml_pooled_slotted_with_policy,
    quantized_matmul_id_ggml_pooled_with_policy, quantized_matmul_id_ggml_with_policy,
    CommandEncoder, DType, ExpertMmSchedule, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, IdMmScratch, KernelPipelineOrigin, KernelRegistry, MlxBuffer,
    MlxDevice, MlxError,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next_u64() % u64::from(bound)) as u32
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// f16 bits with magnitude in [2^-12, 2^-4).
    fn scale_f16_bits(&mut self, allow_negative: bool) -> u16 {
        let r = self.next_u64();
        let exponent = 3 + (r % 8) as u16;
        let mantissa = ((r >> 8) & 0x3ff) as u16;
        let sign = if allow_negative && (r >> 20) & 1 == 1 {
            0x8000
        } else {
            0
        };
        sign | (exponent << 10) | mantissa
    }

    fn input_value(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 8.0 - 4.0
    }
}

/// Every type with an `mm_id` kernel, with the byte offsets of its f16
/// scale fields (random payloads keep finite, normal scales).
const TYPES: [(GgmlType, &[usize]); 12] = [
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q8_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::IQ4_NL, &[0]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::Q3_K, &[108]),
    (GgmlType::Q4_K, &[0, 2]),
    (GgmlType::Q5_K, &[0, 2]),
    (GgmlType::Q6_K, &[208]),
    (GgmlType::IQ4_XS, &[0]),
    (GgmlType::IQ2_XXS, &[0]),
];

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    n_experts: u32,
    n: u32,
    k: u32,
}

const GEMMA_DOWN: Shape = Shape {
    name: "gemma_down",
    n_experts: 128,
    n: 2816,
    k: 704,
};
const QWEN_DOWN: Shape = Shape {
    name: "qwen_down",
    n_experts: 256,
    n: 2048,
    k: 512,
};
const GEMMA_GATE_UP: Shape = Shape {
    name: "gemma_gate_up",
    n_experts: 128,
    n: 1408,
    k: 2816,
};
const DEEPSEEK_LIKE: Shape = Shape {
    name: "deepseek_v4_like",
    n_experts: 256,
    n: 128,
    k: 4096,
};

const N_TOKENS: [u32; 8] = [33, 40, 64, 232, 355, 528, 1420, 4101];

fn supports(shape: Shape, ggml_type: GgmlType) -> bool {
    shape.k % ggml_type.block_values() == 0
}

fn expert_stride(shape: Shape, ggml_type: GgmlType) -> u64 {
    u64::from(shape.n)
        * u64::from(shape.k / ggml_type.block_values())
        * u64::from(ggml_type.block_bytes())
}

fn weights(device: &MlxDevice, shape: Shape, ggml_type: GgmlType, scales: &[usize]) -> MlxBuffer {
    let total = ggml_expert_bytes(
        ggml_type,
        shape.n_experts,
        shape.n,
        shape.k,
        expert_stride(shape, ggml_type),
    )
    .expect("expert bytes") as usize;
    let mut buffer = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("weights");
    let bytes = buffer.as_mut_slice::<u8>().expect("weight bytes");
    let mut rng = Rng::new(0x5eed ^ u64::from(ggml_type.block_bytes()) ^ u64::from(shape.k));
    for chunk in bytes.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next_u64().to_le_bytes()[..chunk.len()]);
    }
    for block in bytes.chunks_exact_mut(ggml_type.block_bytes() as usize) {
        for (index, &offset) in scales.iter().enumerate() {
            let bits = rng.scale_f16_bits(index == 0);
            block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    buffer
}

fn f32_buffer(device: &MlxDevice, len: usize, seed: u64) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len.max(1) * 4, DType::F32, vec![len.max(1)])
        .expect("f32 buffer");
    let mut rng = Rng::new(seed);
    for value in buffer.as_mut_slice::<f32>().expect("f32") {
        *value = rng.input_value();
    }
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(
            values.len().max(1) * 4,
            DType::U32,
            vec![values.len().max(1)],
        )
        .expect("u32 buffer");
    buffer.as_mut_slice::<u32>().expect("u32")[..values.len()].copy_from_slice(values);
    buffer
}

fn output_buffer(device: &MlxDevice, len: usize) -> MlxBuffer {
    device
        .alloc_buffer(len.max(1) * 4, DType::F32, vec![len.max(1)])
        .expect("output")
}

/// Fill with a NaN sentinel unique to this buffer role, so an element a
/// kernel failed to write can never compare equal across two outputs.
fn poison(buffer: &mut MlxBuffer, tag: u32) {
    buffer
        .as_mut_slice::<u32>()
        .expect("output words")
        .fill(0x7fc0_0000 | (tag & 0x3f_ffff));
}

fn bytes(buffer: &MlxBuffer, elements: usize) -> &[u8] {
    &buffer.as_slice::<u8>().expect("bytes")[..elements * 4]
}

fn assert_bitwise(label: &str, left: &MlxBuffer, right: &MlxBuffer, elements: usize) {
    let (a, b) = (bytes(left, elements), bytes(right, elements));
    if a != b {
        let first = a
            .chunks_exact(4)
            .zip(b.chunks_exact(4))
            .position(|(x, y)| x != y)
            .expect("differing word");
        let differing = a
            .chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(x, y)| x != y)
            .count();
        panic!(
            "{label}: outputs differ in {differing} of {elements} f32 words; first at {first}: {:?} vs {:?}",
            &a[first * 4..first * 4 + 4],
            &b[first * 4..first * 4 + 4],
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Tensor,
    Simdgroup,
}

fn policy(path: Path, schedule: ExpertMmSchedule) -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_tensor_mm: match path {
            Path::Tensor => GgmlTensorMmPreference::AutoProbe,
            Path::Simdgroup => GgmlTensorMmPreference::ForceSimd,
        },
        expert_mm_schedule: schedule,
        ..GgmlRoutingPolicy::default()
    }
}

const SCHEDULES: [ExpertMmSchedule; 2] = [
    ExpertMmSchedule::UniformGrid,
    ExpertMmSchedule::CompactTiles,
];

struct Gpu {
    device: MlxDevice,
    registry: KernelRegistry,
    tensor_available: bool,
}

impl Gpu {
    fn new() -> Self {
        let device = MlxDevice::new().expect("Metal device");
        let mut registry = KernelRegistry::new();
        let tensor_available = registry
            .get_pipeline("kernel_mul_mm_id_q4_0_tensor_f32", device.metal_device())
            .is_ok();
        eprintln!(
            "device={} tensor_mm_id_available={tensor_available} tensor_half_tile_skip_ships={} MLX_PRECOMPILED_METALLIB={:?}",
            device.name(),
            tensor_half_tile_skip_validated(&device),
            std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        );
        // The matrix covers both kernel paths; a device without the tensor
        // path must opt out explicitly, and the skip is recorded.
        if !tensor_available {
            assert!(
                std::env::var("MLX_MM_ID_TEST_ALLOW_NO_TENSOR").as_deref() == Ok("1"),
                "tensor mm_id path unavailable on {}: set MLX_MM_ID_TEST_ALLOW_NO_TENSOR=1 to run the simdgroup path only",
                device.name()
            );
            eprintln!("RECORDED SKIP: tensor mm_id path unavailable; simdgroup path only");
        }
        Self {
            device,
            registry,
            tensor_available,
        }
    }

    /// Distinct kernel paths for `ggml_type`: the tensor path only where
    /// the auto route selects it on this device.
    fn paths(&self, ggml_type: GgmlType) -> Vec<Path> {
        if self.tensor_available && ggml_type != GgmlType::Q5_0 {
            vec![Path::Tensor, Path::Simdgroup]
        } else {
            vec![Path::Simdgroup]
        }
    }

    fn run(&mut self, encode: impl FnOnce(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice)) {
        let mut encoder = self.device.command_encoder().expect("encoder");
        encode(&mut encoder, &mut self.registry, &self.device);
        encoder.commit_and_wait().expect("command buffer");
    }
}

fn params(
    ggml_type: GgmlType,
    shape: Shape,
    n_tokens: u32,
    top_k: u32,
) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens,
        top_k,
        n: shape.n,
        k: shape.k,
        n_experts: shape.n_experts,
        expert_stride: expert_stride(shape, ggml_type),
        ggml_type,
    }
}

// ---------------------------------------------------------------------------
// Routing tables (`[n_tokens, top_k]`, distinct ids per token)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Routing {
    Uniform,
    /// One hot expert in ~79% of tokens (the witness-proxy skew).
    HotExpert,
    /// Every token routes to one shared expert.
    AllToOne,
    /// Forty experts receive no rows.
    FortyEmpty,
}

const ROUTINGS: [Routing; 4] = [
    Routing::Uniform,
    Routing::HotExpert,
    Routing::AllToOne,
    Routing::FortyEmpty,
];

fn route_table(routing: Routing, n_tokens: u32, top_k: u32, n_experts: u32, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let hot = 7 % n_experts;
    let excluded = |e: u32| e % 3 == 1 && e / 3 < 40;
    let mut ids = Vec::with_capacity((n_tokens * top_k) as usize);
    for token in 0..n_tokens {
        let start = ids.len();
        let forced = match routing {
            Routing::Uniform | Routing::FortyEmpty => None,
            Routing::HotExpert => (rng.unit() < 0.79).then_some(hot),
            Routing::AllToOne => Some(0),
        };
        if let Some(expert) = forced {
            // The shared expert sits in a rotating slot; the other slots are
            // distinct random experts.
            let slot = (token % top_k) as usize;
            let mut row = vec![u32::MAX; top_k as usize];
            row[slot] = expert;
            for index in (0..top_k as usize).filter(|&index| index != slot) {
                loop {
                    let candidate = rng.below(n_experts);
                    if !row.contains(&candidate) {
                        row[index] = candidate;
                        break;
                    }
                }
            }
            ids.extend(row);
            continue;
        }
        while ids.len() - start < top_k as usize {
            let candidate = rng.below(n_experts);
            let allowed = !matches!(routing, Routing::FortyEmpty) || !excluded(candidate);
            if allowed && !ids[start..].contains(&candidate) {
                ids.push(candidate);
            }
        }
    }
    ids
}

// ---------------------------------------------------------------------------
// N1 + N7 — slotted == flattened on both schedules; row independence
// ---------------------------------------------------------------------------

#[test]
fn n1_slotted_equals_flattened_and_n7_row_independence() {
    let mut gpu = Gpu::new();
    let mut cases = 0usize;
    for shape in [GEMMA_DOWN, QWEN_DOWN] {
        let top_k = 8u32;
        let max_tokens = *N_TOKENS.iter().max().expect("sizes");
        let max_rows = (max_tokens * top_k) as usize;
        let input = f32_buffer(
            &gpu.device,
            max_rows * shape.k as usize,
            0x1111 + u64::from(shape.k),
        );
        let pad_input = f32_buffer(&gpu.device, 33 * top_k as usize * shape.k as usize, 0x2222);
        let mut outputs: Vec<MlxBuffer> = (0..4)
            .map(|_| output_buffer(&gpu.device, max_rows * shape.n as usize))
            .collect();
        let mut alone = output_buffer(&gpu.device, 33 * top_k as usize * shape.n as usize);
        for (ggml_type, scales) in TYPES {
            if !supports(shape, ggml_type) {
                eprintln!("N1 skip {ggml_type:?} on {}: K % block != 0", shape.name);
                continue;
            }
            let weight = weights(&gpu.device, shape, ggml_type, scales);
            for path in gpu.paths(ggml_type) {
                for &n_tokens in &N_TOKENS {
                    for (r, routing) in ROUTINGS.into_iter().enumerate() {
                        let table = route_table(
                            routing,
                            n_tokens,
                            top_k,
                            shape.n_experts,
                            u64::from(n_tokens) * 31 + r as u64,
                        );
                        let ids = u32_buffer(&gpu.device, &table);
                        let rows = (n_tokens * top_k) as usize;
                        let elements = rows * shape.n as usize;
                        let mut scratch =
                            IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens * top_k)
                                .expect("scratch");
                        // outputs: [flat grid, flat compact, slotted grid, slotted compact]
                        for (slot, (slotted, schedule)) in [
                            (false, SCHEDULES[0]),
                            (false, SCHEDULES[1]),
                            (true, SCHEDULES[0]),
                            (true, SCHEDULES[1]),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            poison(&mut outputs[slot], slot as u32 + 1);
                            let routing_policy = policy(path, schedule);
                            let output = &outputs[slot];
                            gpu.run(|encoder, registry, device| {
                                if slotted {
                                    quantized_matmul_id_ggml_pooled_slotted_with_policy(
                                        encoder,
                                        registry,
                                        device,
                                        &input,
                                        &weight,
                                        &ids,
                                        output,
                                        &mut scratch,
                                        &params(ggml_type, shape, n_tokens, top_k),
                                        &routing_policy,
                                    )
                                } else {
                                    quantized_matmul_id_ggml_pooled_with_policy(
                                        encoder,
                                        registry,
                                        device,
                                        &input,
                                        &weight,
                                        &ids,
                                        output,
                                        &mut scratch,
                                        &params(ggml_type, shape, n_tokens * top_k, 1),
                                        &routing_policy,
                                    )
                                }
                                .expect("N1 dispatch");
                            });
                        }
                        let label = format!(
                            "N1 {ggml_type:?} {path:?} {} n={n_tokens} {routing:?}",
                            shape.name
                        );
                        assert_bitwise(
                            &format!("{label} flat grid vs flat compact"),
                            &outputs[0],
                            &outputs[1],
                            elements,
                        );
                        assert_bitwise(
                            &format!("{label} flat vs slotted (grid)"),
                            &outputs[0],
                            &outputs[2],
                            elements,
                        );
                        assert_bitwise(
                            &format!("{label} slotted grid vs compact"),
                            &outputs[2],
                            &outputs[3],
                            elements,
                        );

                        // N7: a token alone among 32 fixed padding tokens.
                        let row_elems = top_k as usize * shape.n as usize;
                        for target in [0, n_tokens / 2, n_tokens - 1] {
                            let mut alone_ids: Vec<u32> = table
                                [(target * top_k) as usize..((target + 1) * top_k) as usize]
                                .to_vec();
                            for pad in 1..33u32 {
                                for slot in 0..top_k {
                                    alone_ids.push((pad * top_k + slot) % shape.n_experts);
                                }
                            }
                            let alone_ids = u32_buffer(&gpu.device, &alone_ids);
                            let mut alone_input = pad_input.clone();
                            {
                                let k = shape.k as usize;
                                let src = &input.as_slice::<f32>().expect("input")[target as usize
                                    * top_k as usize
                                    * k
                                    ..(target as usize + 1) * top_k as usize * k];
                                alone_input.as_mut_slice::<f32>().expect("alone input")
                                    [..top_k as usize * k]
                                    .copy_from_slice(src);
                            }
                            for (slot, schedule) in [(2usize, SCHEDULES[0]), (3, SCHEDULES[1])] {
                                poison(&mut alone, 9);
                                let routing_policy = policy(path, schedule);
                                let mut alone_scratch =
                                    IdMmScratch::alloc(&gpu.device, shape.n_experts, 33)
                                        .expect("alone scratch");
                                gpu.run(|encoder, registry, device| {
                                    quantized_matmul_id_ggml_pooled_slotted_with_policy(
                                        encoder,
                                        registry,
                                        device,
                                        &alone_input,
                                        &weight,
                                        &alone_ids,
                                        &alone,
                                        &mut alone_scratch,
                                        &params(ggml_type, shape, 33, top_k),
                                        &routing_policy,
                                    )
                                    .expect("N7 dispatch");
                                });
                                let batch_row = &bytes(&outputs[slot], elements)[target as usize
                                    * row_elems
                                    * 4
                                    ..(target as usize + 1) * row_elems * 4];
                                let alone_row = &bytes(&alone, row_elems)[..row_elems * 4];
                                assert!(
                                    batch_row == alone_row,
                                    "N7 {label} {schedule:?}: token {target} alone differs from the batch"
                                );
                            }
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("N1/N7 cases: {cases}");
    assert!(cases > 0);
}

// ---------------------------------------------------------------------------
// N2 — CompactTiles == UniformGrid on every entry point
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Entry {
    Auto,
    PooledShared,
    PooledPair,
    PooledSlotted,
}

#[test]
fn n2_compact_equals_uniform_grid_on_every_entry_point() {
    let mut gpu = Gpu::new();
    let mut cases = 0usize;
    for shape in [GEMMA_GATE_UP, GEMMA_DOWN, DEEPSEEK_LIKE] {
        let max_tokens = *N_TOKENS.iter().max().expect("sizes");
        let max_rows = (max_tokens * 8) as usize;
        let input = f32_buffer(
            &gpu.device,
            max_rows * shape.k as usize,
            0x3333 + u64::from(shape.k),
        );
        let mut outputs: Vec<MlxBuffer> = (0..4)
            .map(|_| output_buffer(&gpu.device, max_rows * shape.n as usize))
            .collect();
        for (ggml_type, scales) in TYPES {
            if !supports(shape, ggml_type) {
                eprintln!("N2 skip {ggml_type:?} on {}: K % block != 0", shape.name);
                continue;
            }
            let weight = weights(&gpu.device, shape, ggml_type, scales);
            for path in gpu.paths(ggml_type) {
                for top_k in [1u32, 6, 8] {
                    for &n_tokens in &N_TOKENS {
                        for (r, routing) in ROUTINGS.into_iter().enumerate() {
                            let table = route_table(
                                routing,
                                n_tokens,
                                top_k,
                                shape.n_experts,
                                u64::from(n_tokens) * 131 + u64::from(top_k) * 7 + r as u64,
                            );
                            let ids = u32_buffer(&gpu.device, &table);
                            let elements = (n_tokens * top_k * shape.n) as usize;
                            let p = params(ggml_type, shape, n_tokens, top_k);
                            for entry in [
                                Entry::Auto,
                                Entry::PooledShared,
                                Entry::PooledPair,
                                Entry::PooledSlotted,
                            ] {
                                for (index, schedule) in SCHEDULES.into_iter().enumerate() {
                                    let routing_policy = policy(path, schedule);
                                    let (first, rest) = outputs.split_at_mut(2 * index + 1);
                                    let first = &mut first[2 * index];
                                    let second = &mut rest[0];
                                    poison(first, 10 + index as u32);
                                    poison(second, 20 + index as u32);
                                    let mut scratch =
                                        IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens)
                                            .expect("scratch");
                                    let (first, second) = (&*first, &*second);
                                    gpu.run(|encoder, registry, device| {
                                        match entry {
                                            Entry::Auto => quantized_matmul_id_ggml_with_policy(
                                                encoder,
                                                registry,
                                                device,
                                                &input,
                                                &weight,
                                                &ids,
                                                first,
                                                &p,
                                                &routing_policy,
                                            ),
                                            Entry::PooledShared => {
                                                quantized_matmul_id_ggml_pooled_with_policy(
                                                    encoder,
                                                    registry,
                                                    device,
                                                    &input,
                                                    &weight,
                                                    &ids,
                                                    first,
                                                    &mut scratch,
                                                    &p,
                                                    &routing_policy,
                                                )
                                            }
                                            Entry::PooledPair => {
                                                quantized_matmul_id_ggml_pooled_pair_with_policy(
                                                    encoder,
                                                    registry,
                                                    device,
                                                    &input,
                                                    &weight,
                                                    &weight,
                                                    &ids,
                                                    first,
                                                    second,
                                                    &mut scratch,
                                                    &p,
                                                    &routing_policy,
                                                )
                                            }
                                            Entry::PooledSlotted => {
                                                quantized_matmul_id_ggml_pooled_slotted_with_policy(
                                                    encoder,
                                                    registry,
                                                    device,
                                                    &input,
                                                    &weight,
                                                    &ids,
                                                    first,
                                                    &mut scratch,
                                                    &p,
                                                    &routing_policy,
                                                )
                                            }
                                        }
                                        .expect("N2 dispatch");
                                    });
                                }
                                let label = format!(
                                    "N2 {ggml_type:?} {path:?} {} top_k={top_k} n={n_tokens} {routing:?} {entry:?}",
                                    shape.name
                                );
                                assert_bitwise(&label, &outputs[0], &outputs[2], elements);
                                if matches!(entry, Entry::PooledPair) {
                                    assert_bitwise(
                                        &format!("{label} second projection"),
                                        &outputs[1],
                                        &outputs[3],
                                        elements,
                                    );
                                    assert_bitwise(
                                        &format!("{label} pair projections"),
                                        &outputs[2],
                                        &outputs[3],
                                        elements,
                                    );
                                }
                            }
                            cases += 1;
                        }
                    }
                }
                let compact_kernel = match path {
                    Path::Tensor => format!("{}_tensor_f32_c", kernel_stem(ggml_type)),
                    Path::Simdgroup => format!("{}_f32_c", kernel_stem(ggml_type)),
                };
                assert!(
                    gpu.registry.is_cached(&compact_kernel),
                    "N2 {ggml_type:?} {path:?}: compact kernel {compact_kernel} never resolved"
                );
                assert_library_origin(&gpu.registry, &compact_kernel);
            }
        }
    }
    for top_k in [1, 6, 8] {
        assert_library_origin(
            &gpu.registry,
            &format!("kernel_mul_mm_id_map0c_ne20_{top_k}"),
        );
    }
    eprintln!("N2 cases: {cases}");
    assert!(cases > 0);
}

/// The compact kernels must come from the embedded metallib on the default
/// path and from runtime source under `MLX_PRECOMPILED_METALLIB=0`, so a
/// shader that failed to precompile cannot turn the precompiled run into a
/// second runtime-source run.
fn assert_library_origin(registry: &KernelRegistry, kernel: &str) {
    let runtime = matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    );
    let origin = registry
        .pipeline_identity(kernel)
        .unwrap_or_else(|e| panic!("{kernel}: no pipeline identity: {e}"))
        .origin;
    let want = if runtime {
        KernelPipelineOrigin::RuntimeSource
    } else {
        KernelPipelineOrigin::PrecompiledMetallib
    };
    assert_eq!(origin, want, "{kernel}: pipeline origin");
}

fn kernel_stem(ggml_type: GgmlType) -> String {
    let tag = match ggml_type {
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q5_1 => "q5_1",
        GgmlType::IQ4_NL => "iq4_nl",
        GgmlType::Q2_K => "q2_K",
        GgmlType::Q3_K => "q3_K",
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q6_K => "q6_K",
        GgmlType::IQ4_XS => "iq4_xs",
        GgmlType::IQ2_XXS => "iq2_xxs",
        other => panic!("no mm_id kernel for {other:?}"),
    };
    format!("kernel_mul_mm_id_{tag}")
}

// ---------------------------------------------------------------------------
// N3 — invalid metadata
// ---------------------------------------------------------------------------

const GUARD: u32 = 0xa5a5_a5a5;

#[test]
fn n3_invalid_metadata_poisons_output_on_both_schedules() {
    let mut gpu = Gpu::new();
    let shape = QWEN_DOWN;
    let n_tokens = 64u32;
    let input = f32_buffer(&gpu.device, (n_tokens * 8 * shape.k) as usize, 0x4444);
    let mut output = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);
    for (ggml_type, scales) in TYPES {
        let weight = weights(&gpu.device, shape, ggml_type, scales);
        for path in gpu.paths(ggml_type) {
            for top_k in [1u32, 6, 8] {
                let base = route_table(Routing::Uniform, n_tokens, top_k, shape.n_experts, 77);
                let mut cases = vec![("out_of_range", {
                    let mut ids = base.clone();
                    ids[5 * top_k as usize] = shape.n_experts;
                    ids
                })];
                if top_k > 1 {
                    cases.push(("duplicate", {
                        let mut ids = base.clone();
                        ids[9 * top_k as usize + 1] = ids[9 * top_k as usize];
                        ids
                    }));
                    cases.push(("both", {
                        let mut ids = base.clone();
                        ids[2 * top_k as usize] = shape.n_experts + 3;
                        ids[40 * top_k as usize + 1] = ids[40 * top_k as usize];
                        ids
                    }));
                }
                for (case, table) in cases {
                    let ids = u32_buffer(&gpu.device, &table);
                    for schedule in SCHEDULES {
                        let routing_policy = policy(path, schedule);
                        // Oversized scratch with guard words past the extent
                        // this call's schedule may write.
                        let mut scratch =
                            IdMmScratch::alloc(&gpu.device, shape.n_experts, 4 * n_tokens)
                                .expect("scratch");
                        scratch
                            .htpe_mut()
                            .as_mut_slice::<u32>()
                            .expect("htpe")
                            .fill(GUARD);
                        scratch
                            .hids_mut()
                            .as_mut_slice::<u32>()
                            .expect("hids")
                            .fill(GUARD);
                        // A finite prefill, so only the kernel's poison
                        // write can make every element NaN.
                        output.as_mut_slice::<f32>().expect("output").fill(1.0);
                        let p = params(ggml_type, shape, n_tokens, top_k);
                        let out = &output;
                        gpu.run(|encoder, registry, device| {
                            quantized_matmul_id_ggml_pooled_slotted_with_policy(
                                encoder,
                                registry,
                                device,
                                &input,
                                &weight,
                                &ids,
                                out,
                                &mut scratch,
                                &p,
                                &routing_policy,
                            )
                            .expect("N3 dispatch encodes");
                        });
                        let elements = (n_tokens * top_k * shape.n) as usize;
                        let values = &output.as_slice::<f32>().expect("output")[..elements];
                        assert!(
                            values.iter().all(|v| v.is_nan()),
                            "N3 {ggml_type:?} {path:?} top_k={top_k} {case} {schedule:?}: output not fully poisoned"
                        );
                        let (htpe_words, hids_words) =
                            schedule_extent_words(schedule, shape.n_experts, n_tokens, top_k);
                        let htpe = scratch.htpe().as_slice::<u32>().expect("htpe");
                        let hids = scratch.hids().as_slice::<u32>().expect("hids");
                        assert!(
                            htpe[htpe_words..].iter().all(|w| *w == GUARD)
                                && hids[hids_words..].iter().all(|w| *w == GUARD),
                            "N3 {ggml_type:?} {path:?} top_k={top_k} {case} {schedule:?}: schedule wrote past its extent"
                        );
                    }
                }
            }
        }
    }
}

/// Enforces the N3 "no weight read" obligation: the N3 weight stacks are
/// exact-size buffers, so the out-of-range ids (`n_experts`, `n_experts + 3`)
/// address slabs wholly past their end. Shader validation alone only
/// zero-fills an out-of-bounds access and logs it, so the child also sets
/// `MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1` (a fault aborts the process) and
/// `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`, and its stderr must carry no
/// validation report.
#[test]
fn n3_invalid_metadata_reads_no_weight_under_shader_validation() {
    if std::env::var_os("MTL_SHADER_VALIDATION").is_some() {
        return; // Already the child (or an outer validation run).
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test exe"))
        .args([
            "--exact",
            "n3_invalid_metadata_poisons_output_on_both_schedules",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MTL_SHADER_VALIDATION", "1")
        .env("MTL_SHADER_VALIDATION_ABORT_ON_FAULT", "1")
        .env("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1")
        .output()
        .expect("run N3 under shader validation");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "N3 under shader validation failed (status {:?}):\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("running 1 test") && stdout.contains("1 passed"),
        "shader-validation child ran no N3 test:\n{stdout}"
    );
    // Positive control: the child really ran under GPU validation (Metal
    // logs this banner when MTL_SHADER_VALIDATION takes effect).
    assert!(
        stderr.contains("Metal GPU Validation Enabled")
            || stdout.contains("Metal GPU Validation Enabled"),
        "shader validation did not engage in the N3 child:\n{stdout}\n{stderr}"
    );
    let lowered = stderr.to_lowercase();
    for marker in [
        "invalid device load",
        "invalid device store",
        "out of bounds",
        "mtlgpuerror",
        "validation error",
    ] {
        assert!(
            !lowered.contains(marker),
            "shader validation reported a fault ({marker:?}) in N3:\n{stderr}"
        );
    }
}

fn compact_max_tiles(rows: u64, n_experts: u64) -> u64 {
    rows.min(rows.div_ceil(32) + n_experts.min(rows) - 1)
}

fn schedule_extent_words(
    schedule: ExpertMmSchedule,
    n_experts: u32,
    n_tokens: u32,
    top_k: u32,
) -> (usize, usize) {
    let rows = u64::from(n_tokens * top_k);
    match schedule {
        ExpertMmSchedule::UniformGrid => (n_experts as usize, (n_experts * n_tokens) as usize),
        ExpertMmSchedule::CompactTiles => (
            1 + 3 * compact_max_tiles(rows, u64::from(n_experts)) as usize,
            rows as usize + n_experts as usize,
        ),
    }
}

// ---------------------------------------------------------------------------
// N4 — schedule reuse key
// ---------------------------------------------------------------------------

#[test]
fn n4_schedule_reuse_is_keyed_and_bitwise() {
    let mut gpu = Gpu::new();
    let shape = GEMMA_DOWN;
    let gate = GEMMA_GATE_UP;
    let ggml_type = GgmlType::Q8_0;
    let n_tokens = 232u32;
    let top_k = 8u32;
    let down_weight = weights(&gpu.device, shape, ggml_type, &[0]);
    let gate_weight = weights(&gpu.device, gate, ggml_type, &[0]);
    let gate_input = f32_buffer(&gpu.device, (n_tokens * gate.k) as usize, 0x5555);
    let down_input = f32_buffer(&gpu.device, (n_tokens * top_k * shape.k) as usize, 0x6666);
    let table = route_table(Routing::HotExpert, n_tokens, top_k, shape.n_experts, 5);
    let ids = u32_buffer(&gpu.device, &table);
    let other_ids = u32_buffer(&gpu.device, &table);
    let gate_out = output_buffer(&gpu.device, (n_tokens * top_k * gate.n) as usize);
    let mut fresh = output_buffer(&gpu.device, (n_tokens * top_k * shape.n) as usize);
    let mut reused = output_buffer(&gpu.device, (n_tokens * top_k * shape.n) as usize);
    let elements = (n_tokens * top_k * shape.n) as usize;
    let gate_params = params(ggml_type, gate, n_tokens, top_k);
    let down_params = params(ggml_type, shape, n_tokens, top_k);

    for path in gpu.paths(ggml_type) {
        for schedule in SCHEDULES {
            let routing_policy = policy(path, schedule);
            let mut scratch =
                IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).expect("scratch");

            // Fresh schedule reference.
            poison(&mut fresh, 41);
            {
                let out = &fresh;
                gpu.run(|encoder, registry, device| {
                    quantized_matmul_id_ggml_pooled_slotted_with_policy(
                        encoder,
                        registry,
                        device,
                        &down_input,
                        &down_weight,
                        &ids,
                        out,
                        &mut scratch,
                        &down_params,
                        &routing_policy,
                    )
                    .expect("fresh slotted");
                });
            }

            // Valid reuse after gate/up in the same encoder, with and
            // without a commit_wait_and_rotate in between.
            for rotate in [false, true] {
                poison(&mut reused, 42);
                let mut encoder = gpu.device.command_encoder().expect("encoder");
                quantized_matmul_id_ggml_pooled_with_policy(
                    &mut encoder,
                    &mut gpu.registry,
                    &gpu.device,
                    &gate_input,
                    &gate_weight,
                    &ids,
                    &gate_out,
                    &mut scratch,
                    &gate_params,
                    &routing_policy,
                )
                .expect("gate/up builds the schedule");
                if rotate {
                    encoder.commit_wait_and_rotate().expect("rotate");
                }
                quantized_matmul_id_ggml_pooled_slotted_reusing_schedule_with_policy(
                    &mut encoder,
                    &mut gpu.registry,
                    &gpu.device,
                    &down_input,
                    &down_weight,
                    &ids,
                    &reused,
                    &mut scratch,
                    &down_params,
                    &routing_policy,
                )
                .expect("valid reuse");
                encoder.commit_and_wait().expect("reuse command buffer");
                assert_bitwise(
                    &format!("N4 {path:?} {schedule:?} rotate={rotate} reuse vs fresh"),
                    &fresh,
                    &reused,
                    elements,
                );
            }

            // Every mismatch fails before encoding.
            let expect_reject = |label: &str,
                                 gpu: &mut Gpu,
                                 scratch: &mut IdMmScratch,
                                 encoder: &mut CommandEncoder,
                                 reuse_ids: &MlxBuffer,
                                 reuse_params: &GgmlQuantizedMatmulIdParams,
                                 reuse_policy: &GgmlRoutingPolicy| {
                let before = encoder.take_capture().expect("capturing").len();
                encoder.start_capture();
                let result = quantized_matmul_id_ggml_pooled_slotted_reusing_schedule_with_policy(
                    encoder,
                    &mut gpu.registry,
                    &gpu.device,
                    &down_input,
                    &down_weight,
                    reuse_ids,
                    &reused,
                    scratch,
                    reuse_params,
                    reuse_policy,
                );
                let after = encoder.take_capture().expect("capturing").len();
                assert!(
                    matches!(result, Err(MlxError::InvalidArgument(_))),
                    "N4 {label}: expected InvalidArgument, got {result:?}"
                );
                assert_eq!(after, 0, "N4 {label}: rejected reuse encoded {after} nodes");
                let _ = before;
            };
            let build = |gpu: &mut Gpu, scratch: &mut IdMmScratch, encoder: &mut CommandEncoder| {
                encoder.start_capture();
                quantized_matmul_id_ggml_pooled_with_policy(
                    encoder,
                    &mut gpu.registry,
                    &gpu.device,
                    &gate_input,
                    &gate_weight,
                    &ids,
                    &gate_out,
                    scratch,
                    &gate_params,
                    &routing_policy,
                )
                .expect("schedule build");
            };

            // Different ids buffer (same contents).
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            expect_reject(
                "other ids buffer",
                &mut gpu,
                &mut scratch,
                &mut encoder,
                &other_ids,
                &down_params,
                &routing_policy,
            );

            // Different n_tokens over the same buffer.
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            let shorter = GgmlQuantizedMatmulIdParams {
                n_tokens: n_tokens - 1,
                ..down_params
            };
            expect_reject(
                "other n_tokens",
                &mut gpu,
                &mut scratch,
                &mut encoder,
                &ids,
                &shorter,
                &routing_policy,
            );

            // Different top_k.
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            let top6 = GgmlQuantizedMatmulIdParams {
                top_k: 6,
                ..down_params
            };
            expect_reject(
                "other top_k",
                &mut gpu,
                &mut scratch,
                &mut encoder,
                &ids,
                &top6,
                &routing_policy,
            );

            // Other schedule kind.
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            let other_schedule = policy(
                path,
                match schedule {
                    ExpertMmSchedule::UniformGrid => ExpertMmSchedule::CompactTiles,
                    ExpertMmSchedule::CompactTiles => ExpertMmSchedule::UniformGrid,
                },
            );
            expect_reject(
                "other schedule kind",
                &mut gpu,
                &mut scratch,
                &mut encoder,
                &ids,
                &down_params,
                &other_schedule,
            );

            // Another map0 since (a different table into the same scratch).
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            quantized_matmul_id_ggml_pooled_with_policy(
                &mut encoder,
                &mut gpu.registry,
                &gpu.device,
                &gate_input,
                &gate_weight,
                &other_ids,
                &gate_out,
                &mut scratch,
                &gate_params,
                &routing_policy,
            )
            .expect("second map0");
            expect_reject(
                "map0 since",
                &mut gpu,
                &mut scratch,
                &mut encoder,
                &ids,
                &down_params,
                &routing_policy,
            );

            // Another encoder.
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            build(&mut gpu, &mut scratch, &mut encoder);
            let mut other_encoder = gpu.device.command_encoder().expect("other encoder");
            other_encoder.start_capture();
            expect_reject(
                "other encoder",
                &mut gpu,
                &mut scratch,
                &mut other_encoder,
                &ids,
                &down_params,
                &routing_policy,
            );
            drop(encoder);

            // Mutable buffer access through each accessor (N9).
            for accessor in ["hids_mut", "htpe_mut", "buffers_mut"] {
                let mut encoder = gpu.device.command_encoder().expect("encoder");
                build(&mut gpu, &mut scratch, &mut encoder);
                match accessor {
                    "hids_mut" => {
                        let _ = scratch.hids_mut();
                    }
                    "htpe_mut" => {
                        let _ = scratch.htpe_mut();
                    }
                    _ => {
                        let _ = scratch.buffers_mut();
                    }
                }
                expect_reject(
                    &format!("mutable access via {accessor}"),
                    &mut gpu,
                    &mut scratch,
                    &mut encoder,
                    &ids,
                    &down_params,
                    &routing_policy,
                );
            }

            // No schedule at all.
            let mut never =
                IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).expect("scratch");
            let mut encoder = gpu.device.command_encoder().expect("encoder");
            encoder.start_capture();
            expect_reject(
                "no schedule",
                &mut gpu,
                &mut never,
                &mut encoder,
                &ids,
                &down_params,
                &routing_policy,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// N5 — half-tile skip against the 32-row body
// ---------------------------------------------------------------------------

/// Per-expert routed-row counts whose tiles hold `nr1` ∈ {1, 15, 16, 17,
/// 31, 32} rows, including partial tiles after full ones.
const N5_COUNTS: [u32; 12] = [1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64];

#[test]
fn n5_half_tile_skip_against_the_32_row_body() {
    let mut gpu = Gpu::new();
    let shape = QWEN_DOWN;
    // top_k = 1: token t routes to the expert owning row t.
    let mut table = Vec::new();
    let mut tile_rows: Vec<(u32, u32)> = Vec::new(); // (first row, nr1) per tile
    for (expert, &count) in N5_COUNTS.iter().enumerate() {
        let mut remaining = count;
        while remaining > 0 {
            let nr1 = remaining.min(32);
            tile_rows.push((table.len() as u32 + (count - remaining), nr1));
            remaining -= nr1;
        }
        table.extend(std::iter::repeat(expert as u32 * 5).take(count as usize));
    }
    // Interleave tokens so each expert's rows are not contiguous tokens.
    let n_tokens = table.len() as u32;
    let mut rng = Rng::new(0x5a5a);
    let mut order: Vec<u32> = (0..n_tokens).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i as u32 + 1) as usize);
    }
    let shuffled: Vec<u32> = order.iter().map(|&row| table[row as usize]).collect();
    // Row-to-tile fill for every token: token i is the j-th entry of its
    // expert, so its tile holds nr1 rows.
    let mut seen = std::collections::HashMap::<u32, u32>::new();
    let token_tile_rows: Vec<u32> = shuffled
        .iter()
        .map(|&expert| {
            let index = seen.entry(expert).or_insert(0);
            let position = *index;
            *index += 1;
            let count = N5_COUNTS[(expert / 5) as usize];
            (count - (position / 32) * 32).min(32)
        })
        .collect();
    let _ = tile_rows;
    let ids = u32_buffer(&gpu.device, &shuffled);
    let input = f32_buffer(&gpu.device, (n_tokens * shape.k) as usize, 0x7777);
    let elements = (n_tokens * shape.n) as usize;
    let mut outputs: Vec<MlxBuffer> = (0..3)
        .map(|_| output_buffer(&gpu.device, elements))
        .collect();
    let mut record = Vec::new();
    for (ggml_type, scales) in TYPES {
        let weight = weights(&gpu.device, shape, ggml_type, scales);
        for path in gpu.paths(ggml_type) {
            let routing_policy = policy(path, ExpertMmSchedule::CompactTiles);
            for (slot, half_tile_skip) in [
                HalfTileSkip::Disabled,
                HalfTileSkip::Forced,
                HalfTileSkip::Production,
            ]
            .into_iter()
            .enumerate()
            {
                poison(&mut outputs[slot], 50 + slot as u32);
                let options = IdMmScheduleOptions {
                    schedule: ExpertMmSchedule::CompactTiles,
                    half_tile_skip,
                };
                let mut scratch =
                    IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).expect("scratch");
                let out = &outputs[slot];
                gpu.run(|encoder, registry, device| {
                    dispatch_id_mm_pooled_with_options_for_test(
                        encoder,
                        registry,
                        device,
                        &input,
                        &weight,
                        &ids,
                        out,
                        &mut scratch,
                        &params(ggml_type, shape, n_tokens, 1),
                        false,
                        false,
                        &routing_policy,
                        options,
                    )
                    .expect("N5 dispatch");
                });
            }
            let label = format!("N5 {ggml_type:?} {path:?}");
            // Production ships only bitwise variants.
            assert_bitwise(
                &format!("{label} production vs 32-row body"),
                &outputs[0],
                &outputs[2],
                elements,
            );
            let row = shape.n as usize * 4;
            let disabled = bytes(&outputs[0], elements);
            let forced = bytes(&outputs[1], elements);
            let mut partial_ok = true;
            for (token, &nr1) in token_tile_rows.iter().enumerate() {
                let same = disabled[token * row..(token + 1) * row]
                    == forced[token * row..(token + 1) * row];
                if nr1 > 16 {
                    assert!(same, "{label}: full-tile row {token} (nr1={nr1}) changed under the half-tile skip");
                } else if !same {
                    partial_ok = false;
                }
            }
            record.push(format!(
                "{label}: partial tiles {}",
                if partial_ok {
                    "bitwise"
                } else {
                    "DIFFER (falsified)"
                }
            ));
            if path == Path::Simdgroup {
                assert!(
                    partial_ok,
                    "{label}: simdgroup half-tile skip must be bitwise by construction"
                );
            }
        }
    }
    eprintln!(
        "N5 device={} tensor half-tile skip ships={}",
        gpu.device.name(),
        tensor_half_tile_skip_validated(&gpu.device)
    );
    for line in &record {
        eprintln!("{line}");
    }
}

// ---------------------------------------------------------------------------
// N6 — capacity and schedule contents
// ---------------------------------------------------------------------------

/// Host model of map0c: (tile_count, descriptors, list).
fn host_compact_schedule(
    table: &[u32],
    top_k: u32,
    n_experts: u32,
) -> (u32, Vec<[u32; 3]>, Vec<u32>) {
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_experts as usize];
    for (entry, &expert) in table.iter().enumerate() {
        lists[expert as usize].push(entry as u32);
    }
    let _ = top_k;
    let mut descriptors = Vec::new();
    let mut list = Vec::new();
    for (expert, entries) in lists.iter().enumerate() {
        let first = list.len() as u32;
        for start in (0..entries.len()).step_by(32) {
            let rows = (entries.len() - start).min(32) as u32;
            descriptors.push([expert as u32, first + start as u32, rows]);
        }
        list.extend(entries.iter().copied());
        list.push(u32::MAX); // padding slot (not written by the kernel)
    }
    (descriptors.len() as u32, descriptors, list)
}

#[test]
fn n6_capacity_bound_and_schedule_contents() {
    let mut gpu = Gpu::new();
    let shape = GEMMA_DOWN;
    let n_experts = shape.n_experts;

    // (label, ids, top_k, experts, bound attained)
    let mut tables: Vec<(String, Vec<u32>, u32, u32, bool)> = Vec::new();
    // Bound attained by the `rows` term: one row per expert.
    tables.push((
        "one row per expert".into(),
        (0..64).collect(),
        1,
        n_experts,
        true,
    ));
    // Bound attained by the `⌈rows/32⌉ + min(experts, rows) − 1` term: with
    // 32 experts, expert 0 holds 32k + 1 rows (k full tiles plus a one-row
    // tile) and the other 31 one row each.
    for k in [1u32, 4] {
        let mut table: Vec<u32> = std::iter::repeat(0).take((32 * k + 1) as usize).collect();
        table.extend(1..32);
        tables.push((format!("full tiles + partial, k={k}"), table, 1, 32, true));
    }
    for routing in ROUTINGS {
        for top_k in [1u32, 6, 8] {
            tables.push((
                format!("{routing:?} top_k={top_k}"),
                route_table(routing, 355, top_k, n_experts, 99),
                top_k,
                n_experts,
                false,
            ));
        }
    }
    for (label, table, top_k, n_experts, attained) in tables {
        let n_tokens = table.len() as u32 / top_k;
        let rows = u64::from(n_tokens * top_k);
        let ids = u32_buffer(&gpu.device, &table);
        let dispatch = GgmlIdMmDispatchParams {
            n_tokens,
            top_k,
            n: shape.n,
            k: shape.k,
            n_experts,
            expert_stride: expert_stride(shape, GgmlType::Q8_0),
            ggml_type: GgmlType::Q8_0,
        };
        let mut scratch = IdMmScratch::alloc(&gpu.device, n_experts, n_tokens).expect("scratch");
        scratch
            .htpe_mut()
            .as_mut_slice::<u32>()
            .expect("htpe")
            .fill(GUARD);
        scratch
            .hids_mut()
            .as_mut_slice::<u32>()
            .expect("hids")
            .fill(u32::MAX);
        let (htpe, hids) = scratch.buffers_mut();
        gpu.run(|encoder, registry, device| {
            dispatch_id_mm_map0_for_test(
                encoder,
                registry,
                device,
                &ids,
                &mut *htpe,
                &mut *hids,
                &dispatch,
                ExpertMmSchedule::CompactTiles,
            )
            .expect("map0c");
        });
        let (count, descriptors, list) = host_compact_schedule(&table, top_k, n_experts);
        let words = htpe.as_slice::<u32>().expect("htpe");
        assert_eq!(words[0], count, "N6 {label}: tile count");
        let bound = compact_max_tiles(rows, u64::from(n_experts));
        assert!(
            u64::from(count) <= bound,
            "N6 {label}: {count} tiles exceed max_tiles {bound}"
        );
        if attained {
            assert_eq!(
                u64::from(count),
                bound,
                "N6 {label}: max_tiles bound not attained"
            );
        }
        for (tile, descriptor) in descriptors.iter().enumerate() {
            assert_eq!(
                &words[1 + 3 * tile..4 + 3 * tile],
                descriptor,
                "N6 {label}: descriptor {tile}"
            );
        }
        assert!(
            words[1 + 3 * count as usize..].iter().all(|w| *w == GUARD),
            "N6 {label}: descriptor words past tile_count were written"
        );
        let device_list = &hids.as_slice::<u32>().expect("hids")[..list.len()];
        assert_eq!(
            device_list,
            &list[..],
            "N6 {label}: compact list (R2 order)"
        );
    }

    // Undersized scratch fails before encoding.
    let ggml_type = GgmlType::Q8_0;
    let weight = weights(&gpu.device, shape, ggml_type, &[0]);
    let n_tokens = 64u32;
    let input = f32_buffer(&gpu.device, (n_tokens * 8 * shape.k) as usize, 0x8888);
    let output = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);
    let ids = u32_buffer(
        &gpu.device,
        &route_table(Routing::Uniform, n_tokens, 8, n_experts, 3),
    );
    let compact = policy(Path::Simdgroup, ExpertMmSchedule::CompactTiles);
    for (label, mut scratch) in [
        (
            "token cap",
            IdMmScratch::alloc(&gpu.device, n_experts, n_tokens - 1).unwrap(),
        ),
        (
            "row cap",
            IdMmScratch::alloc_with_top_k(&gpu.device, n_experts, n_tokens, 1).unwrap(),
        ),
        (
            "expert cap",
            IdMmScratch::alloc(&gpu.device, n_experts - 1, n_tokens).unwrap(),
        ),
    ] {
        let mut encoder = gpu.device.command_encoder().expect("encoder");
        encoder.start_capture();
        let result = quantized_matmul_id_ggml_pooled_slotted_with_policy(
            &mut encoder,
            &mut gpu.registry,
            &gpu.device,
            &input,
            &weight,
            &ids,
            &output,
            &mut scratch,
            &params(ggml_type, shape, n_tokens, 8),
            &compact,
        );
        assert!(
            matches!(result, Err(MlxError::InvalidArgument(_))),
            "N6 {label}: {result:?}"
        );
        assert_eq!(
            encoder.take_capture().expect("capture").len(),
            0,
            "N6 {label}: encoded work"
        );
    }
    // A replaced buffer one word short of the compact extent fails too.
    let mut short = IdMmScratch::alloc(&gpu.device, n_experts, n_tokens).expect("scratch");
    let (htpe_words, _) =
        schedule_extent_words(ExpertMmSchedule::CompactTiles, n_experts, n_tokens, 8);
    *short.htpe_mut() = gpu
        .device
        .alloc_buffer((htpe_words - 1) * 4, DType::U32, vec![htpe_words - 1])
        .expect("short htpe");
    let mut encoder = gpu.device.command_encoder().expect("encoder");
    encoder.start_capture();
    let result = quantized_matmul_id_ggml_pooled_slotted_with_policy(
        &mut encoder,
        &mut gpu.registry,
        &gpu.device,
        &input,
        &weight,
        &ids,
        &output,
        &mut short,
        &params(ggml_type, shape, n_tokens, 8),
        &compact,
    );
    assert!(
        matches!(result, Err(MlxError::InvalidArgument(_))),
        "N6 short htpe: {result:?}"
    );
    assert_eq!(encoder.take_capture().expect("capture").len(), 0);

    // The legacy-signature constructor fits the compact schedule's worst
    // case at every map0 top-k.
    for top_k in [1u32, 6, 8] {
        for n_tokens in [33u32, 1420, 4101] {
            let scratch = IdMmScratch::alloc(&gpu.device, n_experts, n_tokens).expect("scratch");
            let (htpe_words, hids_words) =
                schedule_extent_words(ExpertMmSchedule::CompactTiles, n_experts, n_tokens, top_k);
            assert!(scratch.htpe().element_count() >= htpe_words);
            assert!(scratch.hids().element_count() >= hids_words);
            let (grid_htpe, grid_hids) =
                schedule_extent_words(ExpertMmSchedule::UniformGrid, n_experts, n_tokens, top_k);
            assert!(scratch.htpe().element_count() >= grid_htpe);
            assert!(scratch.hids().element_count() >= grid_hids);
        }
    }
}

// ---------------------------------------------------------------------------
// N9 — scratch compatibility
// ---------------------------------------------------------------------------

#[test]
fn n9_scratch_constructors_serve_every_entry_point() {
    let mut gpu = Gpu::new();
    let shape = QWEN_DOWN;
    let ggml_type = GgmlType::Q4_0;
    let weight = weights(&gpu.device, shape, ggml_type, &[0]);
    let n_tokens = 96u32;
    let input = f32_buffer(&gpu.device, (n_tokens * 8 * shape.k) as usize, 0x9999);
    let mut reference = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);
    let mut candidate = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);
    let second = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);
    let mut slotted_reference = output_buffer(&gpu.device, (n_tokens * 8 * shape.n) as usize);

    for (label, scratch) in [
        (
            "alloc",
            IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).unwrap(),
        ),
        (
            "alloc_with_top_k(8)",
            IdMmScratch::alloc_with_top_k(&gpu.device, shape.n_experts, n_tokens, 8).unwrap(),
        ),
    ] {
        assert_eq!(scratch.capacity_n_experts(), shape.n_experts, "{label}");
        assert_eq!(scratch.capacity_n_tokens(), n_tokens, "{label}");
        assert_eq!(scratch.capacity_rows(), n_tokens * 8, "{label}");
        assert!(
            scratch.htpe().element_count() >= shape.n_experts as usize,
            "{label}: htpe words"
        );
    }
    let six = IdMmScratch::alloc_with_top_k(&gpu.device, shape.n_experts, n_tokens, 6).unwrap();
    assert_eq!(six.capacity_rows(), n_tokens * 6);
    assert!(six.htpe().element_count() >= shape.n_experts as usize);
    assert!(six.check_capacity(shape.n_experts, n_tokens, 6).is_ok());
    assert!(six.check_capacity(shape.n_experts, n_tokens, 8).is_err());
    assert!(six
        .check_capacity(shape.n_experts + 1, n_tokens, 1)
        .is_err());
    assert!(six
        .check_capacity(shape.n_experts, n_tokens + 1, 1)
        .is_err());

    for top_k in [1u32, 6, 8] {
        let ids = u32_buffer(
            &gpu.device,
            &route_table(
                Routing::Uniform,
                n_tokens,
                top_k,
                shape.n_experts,
                u64::from(top_k),
            ),
        );
        let p = params(ggml_type, shape, n_tokens, top_k);
        let elements = (n_tokens * top_k * shape.n) as usize;
        for path in gpu.paths(ggml_type) {
            let routing_policy = policy(path, ExpertMmSchedule::CompactTiles);
            poison(&mut reference, 60);
            {
                let out = &reference;
                gpu.run(|encoder, registry, device| {
                    quantized_matmul_id_ggml_with_policy(
                        encoder,
                        registry,
                        device,
                        &input,
                        &weight,
                        &ids,
                        out,
                        &p,
                        &routing_policy,
                    )
                    .expect("auto reference");
                });
            }
            // Slotted reference: uniform grid on an independent scratch.
            poison(&mut slotted_reference, 62);
            {
                let out = &slotted_reference;
                let grid_policy = policy(path, ExpertMmSchedule::UniformGrid);
                let mut grid_scratch =
                    IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).expect("scratch");
                gpu.run(|encoder, registry, device| {
                    quantized_matmul_id_ggml_pooled_slotted_with_policy(
                        encoder,
                        registry,
                        device,
                        &input,
                        &weight,
                        &ids,
                        out,
                        &mut grid_scratch,
                        &p,
                        &grid_policy,
                    )
                    .expect("slotted reference");
                });
            }
            for constructor in ["alloc", "alloc_with_top_k"] {
                for entry in [Entry::PooledShared, Entry::PooledPair, Entry::PooledSlotted] {
                    let mut scratch = match constructor {
                        "alloc" => IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens),
                        _ => IdMmScratch::alloc_with_top_k(
                            &gpu.device,
                            shape.n_experts,
                            n_tokens,
                            top_k,
                        ),
                    }
                    .expect("scratch");
                    poison(&mut candidate, 61);
                    let (out, out2) = (&candidate, &second);
                    gpu.run(|encoder, registry, device| {
                        match entry {
                            Entry::PooledShared => quantized_matmul_id_ggml_pooled_with_policy(
                                encoder,
                                registry,
                                device,
                                &input,
                                &weight,
                                &ids,
                                out,
                                &mut scratch,
                                &p,
                                &routing_policy,
                            ),
                            Entry::PooledPair => quantized_matmul_id_ggml_pooled_pair_with_policy(
                                encoder,
                                registry,
                                device,
                                &input,
                                &weight,
                                &weight,
                                &ids,
                                out,
                                out2,
                                &mut scratch,
                                &p,
                                &routing_policy,
                            ),
                            Entry::PooledSlotted => {
                                quantized_matmul_id_ggml_pooled_slotted_with_policy(
                                    encoder,
                                    registry,
                                    device,
                                    &input,
                                    &weight,
                                    &ids,
                                    out,
                                    &mut scratch,
                                    &p,
                                    &routing_policy,
                                )
                            }
                            Entry::Auto => unreachable!(),
                        }
                        .expect("N9 dispatch");
                    });
                    let compared = if matches!(entry, Entry::PooledSlotted) {
                        &slotted_reference
                    } else {
                        &reference
                    };
                    assert_bitwise(
                        &format!("N9 {constructor} {entry:?} top_k={top_k} {path:?}"),
                        compared,
                        &candidate,
                        elements,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P2c coverage of the production default
// ---------------------------------------------------------------------------

/// Every half-tile setting of the compact (tile-fastest) launch agrees with
/// the uniform grid on served-like routing (complements N5's constructed
/// tiles).
#[test]
fn compact_half_tile_settings_equal_uniform_grid() {
    let mut gpu = Gpu::new();
    let shape = GEMMA_DOWN;
    for (ggml_type, scales) in [TYPES[1], TYPES[2]] {
        let weight = weights(&gpu.device, shape, ggml_type, scales);
        for &n_tokens in &[33u32, 232, 1420] {
            let table = route_table(Routing::HotExpert, n_tokens, 8, shape.n_experts, 1234);
            let ids = u32_buffer(&gpu.device, &table);
            let input = f32_buffer(&gpu.device, (n_tokens * 8 * shape.k) as usize, 0xaaaa);
            let elements = (n_tokens * 8 * shape.n) as usize;
            let mut grid = output_buffer(&gpu.device, elements);
            let mut compact = output_buffer(&gpu.device, elements);
            for path in gpu.paths(ggml_type) {
                let grid_policy = policy(path, ExpertMmSchedule::UniformGrid);
                poison(&mut grid, 70);
                let mut scratch =
                    IdMmScratch::alloc(&gpu.device, shape.n_experts, n_tokens).unwrap();
                {
                    let out = &grid;
                    gpu.run(|encoder, registry, device| {
                        quantized_matmul_id_ggml_pooled_slotted_with_policy(
                            encoder,
                            registry,
                            device,
                            &input,
                            &weight,
                            &ids,
                            out,
                            &mut scratch,
                            &params(ggml_type, shape, n_tokens, 8),
                            &grid_policy,
                        )
                        .expect("grid");
                    });
                }
                for half_tile_skip in [HalfTileSkip::Disabled, HalfTileSkip::Production] {
                    poison(&mut compact, 71);
                    let out = &compact;
                    let options = IdMmScheduleOptions {
                        schedule: ExpertMmSchedule::CompactTiles,
                        half_tile_skip,
                    };
                    gpu.run(|encoder, registry, device| {
                        dispatch_id_mm_pooled_with_options_for_test(
                            encoder,
                            registry,
                            device,
                            &input,
                            &weight,
                            &ids,
                            out,
                            &mut scratch,
                            &params(ggml_type, shape, n_tokens, 8),
                            true,
                            false,
                            &grid_policy,
                            options,
                        )
                        .expect("compact");
                    });
                    assert_bitwise(
                        &format!("{ggml_type:?} {path:?} n={n_tokens} {half_tile_skip:?}"),
                        &grid,
                        &compact,
                        elements,
                    );
                }
            }
        }
    }
}
