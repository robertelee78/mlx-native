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

#[path = "support/gemma_expert_data.rs"]
mod gemma_expert_data;

use gemma_expert_data::*;
use mlx_native::{
    expert_grouped_mv_rows_per_simdgroup, ggml_capability, ggml_routing_policy_from_environment,
    quantized_matmul_id_ggml_mv_grouped_with_geometry, quantized_matmul_id_ggml_mv_with_policy,
    quantized_matmul_id_ggml_with_policy, CapturedNode, DType, GgmlCapabilityRequest,
    GgmlExpertShape, GgmlInvocation, GgmlKernelRoute, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, KernelRegistry, MlxDevice,
    GGML_CAPABILITY_SCHEMA_VERSION,
};

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
