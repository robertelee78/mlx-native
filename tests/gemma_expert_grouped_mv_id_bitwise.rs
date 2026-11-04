//! ADR-063 Q4-GEMMA-DECODE T-N3 (and the E1 part of T-N4/T-N7): the
//! expert-grouped `mv_id` kernels (leader pairs) are bitwise equal, on the
//! raw u32 of every F32 output, to the stock per-pair kernels
//! `kernel_mul_mv_id_{q4_K,q5_0,q8_0}_f32` on the same metallib path.
//!
//! Covered: the Gemma 4 gate_up (Q4_K, N = 1408, K = 2816, top-8, 1..=16
//! tokens) and down (Q5_0/Q8_0, N = 2816, K = 704, top-1, up to 128 pairs)
//! shapes; every Q4_K R0; synthetic row tails and a 66-block K; all-distinct,
//! all-one-expert, random, pooled, skewed and invalid-id patterns; zero and
//! subnormal half scales; ±0, subnormal, large and NaN/∞ activations;
//! non-zero-offset views for every buffer and sentinel bytes outside the
//! output view. The default routing policy must plan the stock routes.
//!
//! Run on both metallib paths:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_expert_grouped_mv_id_bitwise -- --test-threads=1`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

use mlx_native::{
    expert_grouped_mv_rows_per_simdgroup, ggml_capability, ggml_routing_policy_from_environment,
    quantized_matmul_id_ggml_mv_grouped_with_geometry, quantized_matmul_id_ggml_mv_with_policy,
    quantized_matmul_id_ggml_with_policy, CapturedNode, DType, GgmlCapabilityRequest,
    GgmlExpertShape, GgmlInvocation, GgmlKernelRoute, GgmlQuantizedMatmulIdParams,
    GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, KernelRegistry, MlxBuffer, MlxDevice,
    GGML_CAPABILITY_SCHEMA_VERSION,
};

const SENTINEL: u32 = 0xDEAD_BEEF;
/// Leading/trailing guard elements around every view (256 bytes).
const GUARD: usize = 64;
const INVALID_MAX: u32 = u32::MAX;

// ---- deterministic data ----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Half scale bits: mostly small normal values of either sign, with zero,
/// negative zero and subnormal halves mixed in.
fn half_scale_bits(rng: &mut Rng, block: usize) -> u16 {
    match block % 29 {
        0 => 0x0000,
        1 => 0x8000,
        2 => 1 + rng.below(0x3ff) as u16,
        3 => 0x8000 | (1 + rng.below(0x3ff) as u16),
        _ => {
            let magnitude = 0.000_5 + rng.unit() * 0.004;
            let value = if rng.below(5) == 0 { -magnitude } else { magnitude };
            half::f16::from_f32(value).to_bits()
        }
    }
}

fn block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_K => 144,
        GgmlType::Q5_0 => 22,
        GgmlType::Q8_0 => 34,
        other => panic!("unsupported test type {other:?}"),
    }
}

fn block_values(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_K => 256,
        GgmlType::Q5_0 | GgmlType::Q8_0 => 32,
        other => panic!("unsupported test type {other:?}"),
    }
}

fn fill_block(ty: GgmlType, rng: &mut Rng, index: usize, out: &mut [u8]) {
    let d = half_scale_bits(rng, index).to_le_bytes();
    out[0..2].copy_from_slice(&d);
    let start = match ty {
        GgmlType::Q4_K => {
            let dmin = half_scale_bits(rng, index / 3 + 7).to_le_bytes();
            out[2..4].copy_from_slice(&dmin);
            4
        }
        _ => 2,
    };
    for chunk in out[start..].chunks_mut(8) {
        let bytes = rng.next().to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

/// Weight view at a non-zero offset into a parent buffer. One padding expert
/// follows the last one because the stock kernels read whole row tiles past
/// a row tail.
struct Weights {
    _parent: MlxBuffer,
    view: MlxBuffer,
    ty: GgmlType,
    n: usize,
    k: usize,
    n_experts: usize,
    stride: usize,
}

fn make_weights(
    device: &MlxDevice,
    ty: GgmlType,
    n: usize,
    k: usize,
    n_experts: usize,
    seed: u64,
) -> Weights {
    let block = block_bytes(ty);
    let blocks_per_expert = n * (k / block_values(ty));
    let stride = blocks_per_expert * block;
    let lead = 4 * GUARD;
    let total = lead + (n_experts + 1) * stride;
    let mut parent = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("weight buffer");
    {
        let bytes = parent.as_mut_slice::<u8>().expect("weight slice");
        let mut rng = Rng(seed);
        bytes[..lead].fill(0xA5);
        for (index, out) in bytes[lead..].chunks_mut(block).enumerate() {
            fill_block(ty, &mut rng, index, out);
        }
    }
    let view = parent.slice_view(lead as u64, n_experts * stride);
    Weights {
        _parent: parent,
        view,
        ty,
        n,
        k,
        n_experts,
        stride,
    }
}

fn activation(rng: &mut Rng) -> f32 {
    match rng.below(256) {
        0 => 0.0,
        1 => -0.0,
        2 => {
            let bits = 1 + rng.below(0x007f_ffff) as u32;
            f32::from_bits(bits | if rng.below(2) == 0 { 0 } else { 0x8000_0000 })
        }
        3 => {
            let value = 1.0e3 + rng.unit() * 3.0e4;
            if rng.below(2) == 0 {
                value
            } else {
                -value
            }
        }
        _ => rng.unit() * 2.0 - 1.0,
    }
}

fn make_input(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * k).map(|_| activation(&mut rng)).collect()
}

fn poison_row(input: &mut [f32], row: usize, k: usize) {
    input[row * k + 3] = f32::NAN;
    input[row * k + k / 2] = f32::INFINITY;
    input[row * k + k - 1] = f32::NEG_INFINITY;
}

// ---- routing patterns ----

#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Every pair a different expert.
    Distinct,
    /// Every token routes to the same experts (one expert at top-1).
    AllOne,
    /// Uniform random, distinct within each token.
    Random,
    /// Random from a small pool: heavy sharing.
    Pooled,
    /// One hot expert taken by (almost) every token.
    Skewed,
    InvalidFirst,
    InvalidMiddle,
    InvalidLast,
    /// Invalid ids at first, middle and last inside one big group.
    InvalidInGroup,
}

const PATTERNS: [Pattern; 9] = [
    Pattern::Distinct,
    Pattern::AllOne,
    Pattern::Random,
    Pattern::Pooled,
    Pattern::Skewed,
    Pattern::InvalidFirst,
    Pattern::InvalidMiddle,
    Pattern::InvalidLast,
    Pattern::InvalidInGroup,
];

fn pick_distinct(rng: &mut Rng, pool: &[u32], count: usize) -> Vec<u32> {
    let mut pool = pool.to_vec();
    let mut picked = Vec::with_capacity(count);
    for _ in 0..count {
        let index = rng.below(pool.len() as u64) as usize;
        picked.push(pool.swap_remove(index));
    }
    picked
}

fn make_ids(pattern: Pattern, n_tokens: usize, top_k: usize, n_experts: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    let pairs = n_tokens * top_k;
    let all: Vec<u32> = (0..n_experts as u32).collect();
    let random = |rng: &mut Rng, pool: &[u32]| -> Vec<u32> {
        (0..n_tokens)
            .flat_map(|_| pick_distinct(rng, pool, top_k))
            .collect()
    };
    let pooled_pool: Vec<u32> = (0..(top_k + 3).min(n_experts) as u32)
        .map(|i| (i * 7 + 3) % n_experts as u32)
        .collect();
    let all_one: Vec<u32> = (0..pairs)
        .map(|p| ((p % top_k) as u32 * 5 + 2) % n_experts as u32)
        .collect();
    let mut ids = match pattern {
        Pattern::Distinct => {
            assert!(pairs <= n_experts, "distinct needs pairs <= experts");
            (0..pairs).map(|p| ((p * 37 + 11) % n_experts) as u32).collect()
        }
        Pattern::AllOne | Pattern::InvalidInGroup => all_one,
        Pattern::Random | Pattern::InvalidFirst | Pattern::InvalidMiddle => random(&mut rng, &all),
        Pattern::Pooled | Pattern::InvalidLast => random(&mut rng, &pooled_pool),
        Pattern::Skewed => {
            let hot = 7 % n_experts as u32;
            let rest: Vec<u32> = all.iter().copied().filter(|&e| e != hot).collect();
            let mut ids = Vec::with_capacity(pairs);
            for _ in 0..n_tokens {
                if top_k == 1 {
                    ids.push(if rng.below(4) != 0 {
                        hot
                    } else {
                        rest[rng.below(rest.len() as u64) as usize]
                    });
                } else {
                    ids.push(hot);
                    ids.extend(pick_distinct(&mut rng, &rest, top_k - 1));
                }
            }
            ids
        }
    };
    match pattern {
        Pattern::InvalidFirst => ids[0] = INVALID_MAX,
        Pattern::InvalidMiddle => ids[pairs / 2] = n_experts as u32,
        Pattern::InvalidLast => ids[pairs - 1] = INVALID_MAX,
        Pattern::InvalidInGroup => {
            ids[0] = INVALID_MAX;
            ids[pairs / 2] = n_experts as u32;
            ids[pairs - 1] = INVALID_MAX;
        }
        _ => {}
    }
    ids
}

// ---- execution ----

#[derive(Clone, Copy, Debug)]
enum Route {
    Stock,
    Geometry(u32),
    Policy,
}

fn grouped_policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_grouped_q4k: true,
        expert_grouped_q5_0: true,
        expert_grouped_q8_0: true,
        ..GgmlRoutingPolicy::default()
    }
}

fn params(w: &Weights, n_tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: w.n as u32,
        k: w.k as u32,
        n_experts: w.n_experts as u32,
        expert_stride: w.stride as u64,
        ggml_type: w.ty,
    }
}

fn guarded_f32(device: &MlxDevice, data: &[f32]) -> (MlxBuffer, MlxBuffer) {
    let total = GUARD + data.len() + GUARD;
    let mut parent = device
        .alloc_buffer(total * 4, DType::F32, vec![total])
        .expect("guarded f32");
    {
        let slice = parent.as_mut_slice::<f32>().expect("guarded slice");
        slice.fill(f32::from_bits(SENTINEL));
        slice[GUARD..GUARD + data.len()].copy_from_slice(data);
    }
    let view = parent.slice_view((GUARD * 4) as u64, data.len());
    (parent, view)
}

fn guarded_u32(device: &MlxDevice, data: &[u32]) -> (MlxBuffer, MlxBuffer) {
    let total = GUARD + data.len() + GUARD;
    let mut parent = device
        .alloc_buffer(total * 4, DType::U32, vec![total])
        .expect("guarded u32");
    {
        let slice = parent.as_mut_slice::<u32>().expect("guarded slice");
        slice.fill(SENTINEL);
        slice[GUARD..GUARD + data.len()].copy_from_slice(data);
    }
    let view = parent.slice_view((GUARD * 4) as u64, data.len());
    (parent, view)
}

struct Harness {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Harness {
    fn new() -> Self {
        Self {
            device: MlxDevice::new().expect("Metal device"),
            registry: KernelRegistry::new(),
        }
    }

    /// Runs one route and returns the raw u32 of the whole output parent
    /// buffer (guards included).
    fn run(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
        route: Route,
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * w.n]);
        let p = params(w, n_tokens, top_k);
        let mut encoder = self.device.command_encoder().expect("encoder");
        match route {
            Route::Stock => quantized_matmul_id_ggml_mv_with_policy(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input_view,
                &w.view,
                &ids_view,
                &output_view,
                &p,
                &GgmlRoutingPolicy::default(),
            ),
            Route::Geometry(r0) => quantized_matmul_id_ggml_mv_grouped_with_geometry(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input_view,
                &w.view,
                &ids_view,
                &output_view,
                &p,
                r0,
            ),
            Route::Policy => quantized_matmul_id_ggml_mv_with_policy(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input_view,
                &w.view,
                &ids_view,
                &output_view,
                &p,
                &grouped_policy(),
            ),
        }
        .expect("encode");
        encoder.commit_and_wait().expect("GPU execution");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }

    /// The kernel label and grid a route encodes, without executing it.
    fn captured_kernel(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        routing: &GgmlRoutingPolicy,
        force_mv: bool,
    ) -> (String, [u64; 3]) {
        let pairs = n_tokens * top_k;
        let input = self
            .device
            .alloc_buffer(n_tokens * w.k * 4, DType::F32, vec![n_tokens * w.k])
            .expect("input");
        let ids = self
            .device
            .alloc_buffer(pairs * 4, DType::U32, vec![pairs])
            .expect("ids");
        let output = self
            .device
            .alloc_buffer(pairs * w.n * 4, DType::F32, vec![pairs * w.n])
            .expect("output");
        let p = params(w, n_tokens, top_k);
        let mut encoder = self.device.command_encoder().expect("encoder");
        encoder.start_capture();
        if force_mv {
            quantized_matmul_id_ggml_mv_with_policy(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input,
                &w.view,
                &ids,
                &output,
                &p,
                routing,
            )
        } else {
            quantized_matmul_id_ggml_with_policy(
                &mut encoder,
                &mut self.registry,
                &self.device,
                &input,
                &w.view,
                &ids,
                &output,
                &p,
                routing,
            )
        }
        .expect("capture encode");
        let captured = encoder.take_capture().expect("capture");
        assert_eq!(captured.len(), 1, "expected exactly one dispatch");
        match &captured[0] {
            CapturedNode::Dispatch {
                pipeline,
                threads_per_grid,
                ..
            } => (
                pipeline.label().to_string(),
                [
                    threads_per_grid.width,
                    threads_per_grid.height,
                    threads_per_grid.depth,
                ],
            ),
            CapturedNode::Barrier => panic!("expected a dispatch"),
        }
    }
}

fn fnv(digest: &mut u64, words: &[u32]) {
    for word in words {
        for byte in word.to_le_bytes() {
            *digest ^= u64::from(byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn assert_stock_frame(stock: &[u32], pairs: usize, n: usize, label: &str) {
    assert!(
        stock[..GUARD].iter().all(|&w| w == SENTINEL)
            && stock[GUARD + pairs * n..].iter().all(|&w| w == SENTINEL),
        "{label}: stock wrote outside the output view"
    );
    assert!(
        stock[GUARD..GUARD + pairs * n].iter().all(|&w| w != SENTINEL),
        "{label}: stock left an output element unwritten"
    );
}

fn assert_bitwise(stock: &[u32], got: &[u32], n: usize, ids: &[u32], label: &str) {
    assert_eq!(stock.len(), got.len());
    if let Some(index) = (0..stock.len()).find(|&i| stock[i] != got[i]) {
        if index < GUARD || index >= stock.len() - GUARD {
            panic!(
                "{label}: guard element {index} changed: stock {:#010x}, grouped {:#010x}",
                stock[index], got[index]
            );
        }
        let flat = index - GUARD;
        let pair = flat / n;
        let mismatches = (0..stock.len()).filter(|&i| stock[i] != got[i]).count();
        // Mismatches by output column modulo 8 (the largest row tile).
        let mut by_column_phase = [0usize; 8];
        for i in (GUARD..stock.len() - GUARD).filter(|&i| stock[i] != got[i]) {
            by_column_phase[((i - GUARD) % n) % 8] += 1;
        }
        panic!(
            "{label}: {mismatches} mismatches (by column % 8: {by_column_phase:?}); first pair={pair} (id {:#x}) col={}: stock {:#010x} ({}), grouped {:#010x} ({})",
            ids[pair],
            flat % n,
            stock[index],
            f32::from_bits(stock[index]),
            got[index],
            f32::from_bits(got[index]),
        );
    }
}

/// Compares the stock route with each grouped route for one cell.
fn check_cell(
    h: &mut Harness,
    w: &Weights,
    n_tokens: usize,
    top_k: usize,
    pattern: Pattern,
    routes: &[Route],
    nan_row: bool,
    seed: u64,
    digest: &mut u64,
) {
    let ids = make_ids(pattern, n_tokens, top_k, w.n_experts, seed);
    let mut input = make_input(n_tokens, w.k, seed ^ 0x5eed);
    if nan_row {
        poison_row(&mut input, n_tokens / 2, w.k);
    }
    let label = format!(
        "{:?} N={} K={} E={} tokens={n_tokens} top_k={top_k} {pattern:?} nan_row={nan_row}",
        w.ty, w.n, w.k, w.n_experts
    );
    let pairs = n_tokens * top_k;
    let stock = h.run(w, n_tokens, top_k, &ids, &input, Route::Stock);
    assert_stock_frame(&stock, pairs, w.n, &label);
    fnv(digest, &stock);
    for &route in routes {
        let got = h.run(w, n_tokens, top_k, &ids, &input, route);
        assert_bitwise(&stock, &got, w.n, &ids, &format!("{label} {route:?}"));
    }
}

fn q4k_routes() -> Vec<Route> {
    vec![Route::Geometry(1), Route::Geometry(2), Route::Geometry(4)]
}

// ---- T-N3: Gemma shapes ----

#[test]
fn q4k_gate_up_grouped_is_bitwise_stock_for_every_width_pattern_and_r0() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, 1408, 2816, 128, 0x9e37_79b9_7f4a_7c15);
    assert_eq!(expert_grouped_mv_rows_per_simdgroup(GgmlType::Q4_K, 1408, 2816), Some(2));
    let mut routes = q4k_routes();
    routes.push(Route::Policy);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for n_tokens in 1..=16 {
        for (index, &pattern) in PATTERNS.iter().enumerate() {
            let nan_row = matches!(pattern, Pattern::Pooled | Pattern::AllOne) && n_tokens > 1;
            check_cell(
                &mut h,
                &w,
                n_tokens,
                8,
                pattern,
                &routes,
                nan_row,
                (n_tokens * 131 + index) as u64 + 1,
                &mut digest,
            );
        }
    }
    eprintln!("q4k gate_up stock digest {digest:#018x}");
}

fn down_widths() -> Vec<usize> {
    let mut widths: Vec<usize> = (1..=16).map(|t| t * 8).collect();
    widths.extend([1, 2, 3, 4, 5, 7, 9, 15, 17, 31, 33, 63, 65, 100, 127]);
    widths
}

fn check_down_shape(ty: GgmlType, seed: u64) {
    let mut h = Harness::new();
    let w = make_weights(&h.device, ty, 2816, 704, 128, seed);
    assert_eq!(expert_grouped_mv_rows_per_simdgroup(ty, 2816, 704), Some(4));
    let routes = [Route::Geometry(4), Route::Policy];
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for pairs in down_widths() {
        for (index, &pattern) in PATTERNS.iter().enumerate() {
            let nan_row = matches!(pattern, Pattern::Skewed | Pattern::AllOne) && pairs > 1;
            check_cell(
                &mut h,
                &w,
                pairs,
                1,
                pattern,
                &routes,
                nan_row,
                (pairs * 977 + index) as u64 + 3,
                &mut digest,
            );
        }
    }
    eprintln!("{ty:?} down stock digest {digest:#018x}");
}

#[test]
fn q5_0_down_grouped_is_bitwise_stock_up_to_128_pairs() {
    check_down_shape(GgmlType::Q5_0, 0x0123_4567_89ab_cdef);
}

#[test]
fn q8_0_down_grouped_is_bitwise_stock_up_to_128_pairs() {
    check_down_shape(GgmlType::Q8_0, 0xfedc_ba98_7654_3210);
}

// ---- T-N3 item 5: synthetic row tails and K tails ----

#[test]
fn grouped_row_and_k_tails_are_bitwise_stock() {
    let mut h = Harness::new();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let tail_patterns = [
        Pattern::AllOne,
        Pattern::Pooled,
        Pattern::Random,
        Pattern::InvalidInGroup,
    ];
    for (n, k) in [(7usize, 2816usize), (1027, 2816), (1029, 512)] {
        let w = make_weights(&h.device, GgmlType::Q4_K, n, k, 16, (n * 31 + k) as u64);
        for n_tokens in [1usize, 2, 5, 8, 16] {
            for (index, &pattern) in tail_patterns.iter().enumerate() {
                check_cell(
                    &mut h,
                    &w,
                    n_tokens,
                    8,
                    pattern,
                    &q4k_routes(),
                    n_tokens == 5,
                    (n_tokens * 17 + index) as u64 + 5,
                    &mut digest,
                );
            }
        }
    }
    for ty in [GgmlType::Q5_0, GgmlType::Q8_0] {
        for (n, k) in [(7usize, 704usize), (2817, 704), (2814, 704), (2818, 2112)] {
            let w = make_weights(&h.device, ty, n, k, 16, (n * 7 + k) as u64);
            for pairs in [1usize, 4, 8, 37, 128] {
                for (index, &pattern) in tail_patterns.iter().enumerate() {
                    check_cell(
                        &mut h,
                        &w,
                        pairs,
                        1,
                        pattern,
                        &[Route::Geometry(4)],
                        pairs == 37,
                        (pairs * 13 + index) as u64 + 9,
                        &mut digest,
                    );
                }
            }
            // Top-8 at the down kernel exercises token_idx = pair / top_k.
            for n_tokens in [1usize, 3, 16] {
                check_cell(
                    &mut h,
                    &w,
                    n_tokens,
                    8,
                    Pattern::Pooled,
                    &[Route::Geometry(4)],
                    false,
                    n_tokens as u64 + 77,
                    &mut digest,
                );
            }
        }
    }
    eprintln!("tail stock digest {digest:#018x}");
}

// ---- T-N4 (E1 part): a NaN/∞ token row does not perturb its neighbours ----

#[test]
fn nan_inf_token_row_does_not_perturb_other_pairs() {
    let mut h = Harness::new();
    let cases = [
        (GgmlType::Q4_K, 1408usize, 2816usize, 16usize, 8usize, Route::Geometry(2)),
        (GgmlType::Q4_K, 1408, 2816, 16, 8, Route::Geometry(4)),
        (GgmlType::Q5_0, 2816, 704, 128, 1, Route::Geometry(4)),
        (GgmlType::Q8_0, 2816, 704, 128, 1, Route::Geometry(4)),
    ];
    for (ty, n, k, n_tokens, top_k, route) in cases {
        let w = make_weights(&h.device, ty, n, k, 32, (n + k) as u64);
        for pattern in [Pattern::AllOne, Pattern::Pooled] {
            let ids = make_ids(pattern, n_tokens, top_k, w.n_experts, 0xabc);
            let clean = make_input(n_tokens, k, 0xdef);
            let poisoned_token = n_tokens / 2;
            let mut poisoned = clean.clone();
            poison_row(&mut poisoned, poisoned_token, k);
            let a = h.run(&w, n_tokens, top_k, &ids, &clean, route);
            let b = h.run(&w, n_tokens, top_k, &ids, &poisoned, route);
            for pair in 0..n_tokens * top_k {
                let range = GUARD + pair * n..GUARD + (pair + 1) * n;
                if pair / top_k == poisoned_token {
                    assert!(
                        b[range].iter().any(|&bits| f32::from_bits(bits).is_nan()),
                        "{ty:?} {pattern:?}: the poisoned token produced no NaN"
                    );
                } else {
                    assert_eq!(
                        a[range.clone()],
                        b[range],
                        "{ty:?} {pattern:?} {route:?}: pair {pair} changed when token {poisoned_token} became NaN/inf"
                    );
                }
            }
        }
    }
}

// ---- T-N7 (E1 part): default policy plans exactly the stock expert routes ----

fn expert_request(
    ty: GgmlType,
    n_tokens: u32,
    top_k: u32,
    n: u32,
    k: u32,
    routing: GgmlRoutingPolicy,
    force_mv: bool,
) -> GgmlCapabilityRequest {
    let block = block_bytes(ty) as u64;
    let stride = u64::from(n) * (u64::from(k) / block_values(ty) as u64) * block;
    let shape = GgmlExpertShape {
        n_tokens,
        n,
        k,
        top_k,
        n_experts: 128,
        expert_stride_bytes: stride,
        ids_are_distinct_per_token: true,
        ids_within_expert_range: true,
    };
    GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: if force_mv {
            GgmlInvocation::ExpertForceMv { shape }
        } else {
            GgmlInvocation::ExpertAutoAllocated { shape }
        },
        ggml_type: ty,
        workload: match n_tokens {
            1 => GgmlWorkloadClass::DecodeSingle,
            2..=8 => GgmlWorkloadClass::ContinuousWidth,
            _ => GgmlWorkloadClass::Prompt,
        },
        routing,
    }
}

#[test]
fn default_policy_plans_stock_expert_routes_and_flags_gate_grouping() {
    let shapes = [
        (GgmlType::Q4_K, 1408u32, 2816u32),
        (GgmlType::Q5_0, 2816, 704),
        (GgmlType::Q8_0, 2816, 704),
    ];
    let environment = ggml_routing_policy_from_environment();
    assert!(!environment.expert_grouped_q4k);
    assert!(!environment.expert_grouped_q5_0);
    assert!(!environment.expert_grouped_q8_0);
    for (ty, n, k) in shapes {
        for top_k in [1u32, 8] {
            for n_tokens in 1..=128u32 {
                for force_mv in [false, true] {
                    for base in [GgmlRoutingPolicy::default(), environment] {
                        let route = ggml_capability(expert_request(ty, n_tokens, top_k, n, k, base, force_mv)).route;
                        assert_ne!(
                            route,
                            Some(GgmlKernelRoute::ExpertMvGrouped),
                            "default policy selected the grouped route for {ty:?} tokens={n_tokens} top_k={top_k}"
                        );
                    }
                    let grouped = ggml_capability(expert_request(
                        ty,
                        n_tokens,
                        top_k,
                        n,
                        k,
                        grouped_policy(),
                        force_mv,
                    ))
                    .route;
                    let admitted = match top_k {
                        8 => n_tokens <= 16,
                        _ => n_tokens <= 128,
                    };
                    let mv_planned = force_mv || n_tokens <= GgmlRoutingPolicy::default().expert_mm_threshold;
                    assert_eq!(
                        grouped == Some(GgmlKernelRoute::ExpertMvGrouped),
                        admitted && mv_planned,
                        "{ty:?} tokens={n_tokens} top_k={top_k} force_mv={force_mv}: route {grouped:?}"
                    );
                }
            }
        }
        // Reference flags refuse the grouped route.
        let mut refused = grouped_policy();
        refused.dense_decode_mv_ext = true;
        assert_ne!(
            ggml_capability(expert_request(ty, 2, 8, n, k, refused, true)).route,
            Some(GgmlKernelRoute::ExpertMvGrouped)
        );
        // Other top-k values and shapes outside the table keep the stock route.
        for (top_k, n2, k2) in [(6u32, n, k), (8, n + 32, k), (8, n, k + 256)] {
            assert_ne!(
                ggml_capability(expert_request(ty, 2, top_k, n2, k2, grouped_policy(), true)).route,
                Some(GgmlKernelRoute::ExpertMvGrouped)
            );
        }
    }
    let mut q8_nr2 = grouped_policy();
    q8_nr2.expert_q8_0_mv_nr2 = true;
    assert_eq!(
        ggml_capability(expert_request(GgmlType::Q8_0, 4, 1, 2816, 704, q8_nr2, true)).route,
        Some(GgmlKernelRoute::ExpertMvNr2)
    );
}

#[test]
fn dispatch_encodes_stock_kernels_by_default_and_grouped_kernels_when_enabled() {
    let mut h = Harness::new();
    let cases = [
        (
            GgmlType::Q4_K,
            1408usize,
            2816usize,
            8usize,
            "kernel_mul_mv_id_q4_K_f32",
            "kernel_mul_mv_id_q4_K_f32_grouped_rt2",
            1408 / 4,
        ),
        (
            GgmlType::Q5_0,
            2816,
            704,
            1,
            "kernel_mul_mv_id_q5_0_f32",
            "kernel_mul_mv_id_q5_0_f32_grouped",
            2816 / 8,
        ),
        (
            GgmlType::Q8_0,
            2816,
            704,
            1,
            "kernel_mul_mv_id_q8_0_f32",
            "kernel_mul_mv_id_q8_0_f32_grouped",
            2816 / 8,
        ),
    ];
    for (ty, n, k, top_k, stock_kernel, grouped_kernel, tiles) in cases {
        let w = make_weights(&h.device, ty, n, k, 8, 1);
        let max_tokens = if top_k == 8 { 16 } else { 128 };
        for n_tokens in [1usize, 4, max_tokens] {
            for force_mv in [false, true] {
                // Above the mm_id threshold the auto entry takes mm_id
                // (several dispatches), which E1 never touches.
                if !force_mv && n_tokens > 32 {
                    continue;
                }
                for base in [GgmlRoutingPolicy::default(), ggml_routing_policy_from_environment()] {
                    let (label, _) = h.captured_kernel(&w, n_tokens, top_k, &base, force_mv);
                    assert_eq!(label, stock_kernel, "{ty:?} tokens={n_tokens} force_mv={force_mv}");
                }
            }
            let (label, grid) =
                h.captured_kernel(&w, n_tokens, top_k, &grouped_policy(), true);
            assert_eq!(label, grouped_kernel, "grouped {ty:?} tokens={n_tokens}");
            assert_eq!(grid, [tiles as u64, (n_tokens * top_k) as u64, 1]);
        }
        // Width above the frozen table keeps the stock kernel.
        let (label, _) = h.captured_kernel(&w, max_tokens + 1, top_k, &grouped_policy(), true);
        assert_eq!(label, stock_kernel, "{ty:?} above the admitted width");
    }
    // Q8_0 with the nr2 reference flag keeps the nr2 kernel.
    let w = make_weights(&h.device, GgmlType::Q8_0, 2816, 704, 8, 2);
    let mut q8_nr2 = grouped_policy();
    q8_nr2.expert_q8_0_mv_nr2 = true;
    let (label, _) = h.captured_kernel(&w, 4, 1, &q8_nr2, true);
    assert_eq!(label, "kernel_mul_mv_id_q8_0_f32_nr2");
    let mut mv_ext = grouped_policy();
    mv_ext.dense_decode_mv_ext = true;
    let (label, _) = h.captured_kernel(&w, 4, 1, &mv_ext, true);
    assert_eq!(label, "kernel_mul_mv_id_q8_0_f32");
}

#[test]
fn explicit_geometry_rejects_unsupported_r0() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q5_0, 64, 64, 2, 3);
    let input = h.device.alloc_buffer(64 * 4, DType::F32, vec![64]).expect("input");
    let ids = h.device.alloc_buffer(4, DType::U32, vec![1]).expect("ids");
    let output = h.device.alloc_buffer(64 * 4, DType::F32, vec![64]).expect("output");
    let mut encoder = h.device.command_encoder().expect("encoder");
    for r0 in [0u32, 1, 2, 3, 8] {
        assert!(quantized_matmul_id_ggml_mv_grouped_with_geometry(
            &mut encoder,
            &mut h.registry,
            &h.device,
            &input,
            &w.view,
            &ids,
            &output,
            &params(&w, 1, 1),
            r0,
        )
        .is_err());
    }
}
