//! ADR-063 Q4-GEMMA-DECODE T-N6 (F2): the expert-grouped Q4_K gate_up kernel
//! with the GELU-tanh(gate) * up epilogue is bitwise equal, on the raw u32 of
//! every F32 output, to the stock chain on the same metallib path:
//! `kernel_mul_mv_id_q4_K_f32` into `[pairs, 2 * n_ff]`, then
//! `moe_swiglu_batch` into `[pairs, n_ff]`.
//!
//! Covered: the Gemma 4 gate_up shape (N = 1408, K = 2816, top-8, 1..=16
//! tokens) with 128 experts; every R0; synthetic row tails (n_ff = 7, 515) and
//! a short K; top-1 pairs; all E1 routing patterns including out-of-range ids
//! (the stock NaN poison through the epilogue); zero and subnormal half
//! scales; ±0, subnormal, large and NaN/∞ activations; non-zero-offset views
//! and sentinel bytes outside the output view.
//!
//! Run on both metallib paths:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_gelu_epilogue_bitwise -- --test-threads=1`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

#[path = "support/gemma_expert_data.rs"]
mod gemma_expert_data;

use gemma_expert_data::*;
use mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode;
use mlx_native::{
    quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry,
    quantized_matmul_id_ggml_mv_with_policy, DType, GgmlRoutingPolicy, GgmlType, KernelRegistry,
    MlxDevice,
};

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

    /// Stock chain: mv_id gate_up, then moe_swiglu_batch. Returns the raw u32
    /// of the whole guarded `[pairs, n/2]` output parent.
    fn reference(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        let n_ff = w.n / 2;
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (_gate_up_parent, gate_up_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * w.n]);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * n_ff]);

        let mut encoder = self.device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_mv_with_policy(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input_view,
            &w.view,
            &ids_view,
            &gate_up_view,
            &params(w, n_tokens, top_k),
            &GgmlRoutingPolicy::default(),
        )
        .expect("stock gate_up");
        encoder.commit_and_wait().expect("stock gate_up GPU");

        let mut encoder = self.device.command_encoder().expect("encoder");
        moe_swiglu_batch_encode(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            &gate_up_view,
            &output_view,
            n_ff,
            pairs,
        )
        .expect("stock swiglu");
        encoder.commit_and_wait().expect("stock swiglu GPU");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }

    fn fused(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
        r0: u32,
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * (w.n / 2)]);
        let mut encoder = self.device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input_view,
            &w.view,
            &ids_view,
            &output_view,
            &params(w, n_tokens, top_k),
            r0,
        )
        .expect("fused gate_up");
        encoder.commit_and_wait().expect("fused GPU");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }
}

fn check_cell(
    h: &mut Harness,
    w: &Weights,
    n_tokens: usize,
    top_k: usize,
    pattern: Pattern,
    nan_row: bool,
    seed: u64,
    digest: &mut u64,
) {
    let ids = make_ids(pattern, n_tokens, top_k, w.n_experts, seed);
    let mut input = make_input(n_tokens, w.k, seed ^ 0x5eed);
    if nan_row {
        poison_row(&mut input, n_tokens / 2, w.k);
    }
    let n_ff = w.n / 2;
    let pairs = n_tokens * top_k;
    let label = format!(
        "GEGLU N={} K={} E={} tokens={n_tokens} top_k={top_k} {pattern:?} nan_row={nan_row}",
        w.n, w.k, w.n_experts
    );
    let stock = h.reference(w, n_tokens, top_k, &ids, &input);
    assert_stock_frame(&stock, pairs, n_ff, &label);
    for (pair, &id) in ids.iter().enumerate() {
        if id as usize >= w.n_experts {
            let row = &stock[GUARD + pair * n_ff..GUARD + (pair + 1) * n_ff];
            assert!(
                row.iter().all(|&bits| f32::from_bits(bits).is_nan()),
                "{label}: stock poison did not stay NaN through moe_swiglu_batch"
            );
        }
    }
    fnv(digest, &stock);
    for r0 in [1u32, 2, 4] {
        let got = h.fused(w, n_tokens, top_k, &ids, &input, r0);
        assert_bitwise(&stock, &got, n_ff, &ids, &format!("{label} R0={r0}"));
    }
}

#[test]
fn geglu_gate_up_is_bitwise_stock_gate_up_then_swiglu_for_every_width_and_pattern() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, 1408, 2816, 128, 0x9e37_79b9_7f4a_7c15);
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
                nan_row,
                (n_tokens * 131 + index) as u64 + 1,
                &mut digest,
            );
        }
    }
    eprintln!("geglu gate_up stock digest {digest:#018x}");
}

#[test]
fn geglu_row_tails_short_k_and_top1_are_bitwise_stock() {
    let mut h = Harness::new();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let patterns = [
        Pattern::AllOne,
        Pattern::Pooled,
        Pattern::Random,
        Pattern::InvalidInGroup,
    ];
    for (n, k) in [(14usize, 2816usize), (1030, 2816), (1030, 512)] {
        let w = make_weights(&h.device, GgmlType::Q4_K, n, k, 16, (n * 31 + k) as u64);
        for n_tokens in [1usize, 2, 5, 16] {
            for (index, &pattern) in patterns.iter().enumerate() {
                check_cell(
                    &mut h,
                    &w,
                    n_tokens,
                    8,
                    pattern,
                    n_tokens == 5,
                    (n_tokens * 17 + index) as u64 + 5,
                    &mut digest,
                );
            }
        }
        for pairs in [1usize, 37, 128] {
            for (index, &pattern) in patterns.iter().enumerate() {
                check_cell(
                    &mut h,
                    &w,
                    pairs,
                    1,
                    pattern,
                    pairs == 37,
                    (pairs * 13 + index) as u64 + 9,
                    &mut digest,
                );
            }
        }
    }
    eprintln!("geglu tail stock digest {digest:#018x}");
}

#[test]
fn geglu_entry_rejects_unsupported_geometry_and_types() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, 8, 256, 2, 3);
    let input = h.device.alloc_buffer(256 * 4, DType::F32, vec![256]).expect("input");
    let ids = h.device.alloc_buffer(4, DType::U32, vec![1]).expect("ids");
    let output = h.device.alloc_buffer(8 * 4, DType::F32, vec![8]).expect("output");
    let mut encoder = h.device.command_encoder().expect("encoder");
    for r0 in [0u32, 3, 8] {
        assert!(quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry(
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
    let mut odd = params(&w, 1, 1);
    odd.n = 7;
    assert!(quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry(
        &mut encoder,
        &mut h.registry,
        &h.device,
        &input,
        &w.view,
        &ids,
        &output,
        &odd,
        1,
    )
    .is_err());
    let mut q8 = params(&w, 1, 1);
    q8.ggml_type = GgmlType::Q8_0;
    assert!(quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry(
        &mut encoder,
        &mut h.registry,
        &h.device,
        &input,
        &w.view,
        &ids,
        &output,
        &q8,
        1,
    )
    .is_err());
}
