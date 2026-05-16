//! Parity test: `fused_moe_routing_batch_f32_v3` ≡ `fused_moe_routing_batch_f32`
//! (ADR-029 iter-175 Step 1j.2/1j.3).
//!
//! REGRESSION PROTECTION for the Step 1j.2 fix (commit 9496c22).
//!
//! V3 batched processes one token per threadgroup via parallel SG-tournament
//! top-K.  V1 batched (default fallback when V3 off) uses tree-reduce
//! softmax + single-thread serial top-K.
//!
//! Step 1j (original): V3 batched used simd-reduce softmax.  This created
//! a different f32 reduction order than V1 batched's tree-reduce, causing
//! ULP-scale prob differences → top-K boundary swaps → production decode
//! divergence after ~30 identical tokens.
//!
//! Step 1j.2 (fix at 9496c22): switched V3 batched softmax to tree-reduce,
//! matching V1 batched exactly.  V3's parallel top-K (Step 4) unchanged.
//!
//! This test catches any regression that would re-introduce the divergence.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_batch_f32;
use mlx_native::{DType, KernelRegistry, MlxDevice};

struct Xoshiro256 {
    s: [u64; 4],
}
impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *si = x ^ (x >> 31);
        }
        Xoshiro256 { s }
    }
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn randn_pair(rng: &mut Xoshiro256) -> (f64, f64) {
    loop {
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();
        if u1 > 1e-30 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}
fn gaussian_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let (a, b) = randn_pair(rng);
        out.push(a as f32);
        if out.len() < n {
            out.push(b as f32);
        }
    }
    out
}

/// Run batched routing in a specified variant.  V3 fires when
/// HF2Q_FUSED_MOE_ROUTING_V3=1 (default); V1 fires when V3=0.
fn run_once_batched(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    variant: &str, // "v3" or "v1"
    logits_data: &[f32],
    per_expert_scale: &[f32],
    num_experts: u32,
    top_k: u32,
    n_tokens: u32,
) -> (Vec<u32>, Vec<f32>) {
    match variant {
        "v3" => {
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V3", "1");
        }
        "v1" => {
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V3", "0");
        }
        _ => panic!("unknown variant {variant}"),
    }
    let total_logits = (num_experts * n_tokens) as usize;
    let mut logits = device
        .alloc_buffer(total_logits * 4, DType::F32, vec![total_logits])
        .expect("alloc logits");
    logits
        .as_mut_slice::<f32>()
        .expect("w")
        .copy_from_slice(logits_data);
    let total_ids = (top_k * n_tokens) as usize;
    let expert_ids = device
        .alloc_buffer(total_ids * 4, DType::U32, vec![total_ids])
        .expect("alloc ids");
    let routing_weights = device
        .alloc_buffer(total_ids * 4, DType::F32, vec![total_ids])
        .expect("alloc weights");
    let mut scale = device
        .alloc_buffer(
            num_experts as usize * 4,
            DType::F32,
            vec![num_experts as usize],
        )
        .expect("alloc scale");
    scale
        .as_mut_slice::<f32>()
        .expect("w")
        .copy_from_slice(per_expert_scale);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_fused_moe_routing_batch_f32(
        &mut enc,
        registry,
        device.metal_device(),
        &logits,
        &expert_ids,
        &routing_weights,
        &scale,
        num_experts,
        top_k,
        n_tokens,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let ids = expert_ids
        .as_slice::<u32>()
        .expect("read ids")
        .to_vec();
    let weights = routing_weights
        .as_slice::<f32>()
        .expect("read weights")
        .to_vec();
    (ids, weights)
}

/// Verify V3 batched produces same top-K SET as V1 batched on
/// gemma4-prefill-like inputs: 18 tokens × 128 experts, Gaussian logits.
#[test]
fn v3_batched_parity_gemma4_prefill_18_tokens() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32;
    let n_tokens = 18u32;
    let mut rng = Xoshiro256::new(0xBA7CED_18);
    let logits = gaussian_vec(&mut rng, (num_experts * n_tokens) as usize);
    let scale = vec![1.0_f32; num_experts as usize];

    let (ids_v1, w_v1) = run_once_batched(
        &device,
        &mut registry,
        "v1",
        &logits,
        &scale,
        num_experts,
        top_k,
        n_tokens,
    );
    let (ids_v3, w_v3) = run_once_batched(
        &device,
        &mut registry,
        "v3",
        &logits,
        &scale,
        num_experts,
        top_k,
        n_tokens,
    );

    // Per-token: top-K SET should match exactly.  Order may differ
    // (V1 serial scan from i=0; V3 tournament arbitrary on ties).
    for tok in 0..n_tokens as usize {
        let v1_slice = &ids_v1[tok * top_k as usize..(tok + 1) * top_k as usize];
        let v3_slice = &ids_v3[tok * top_k as usize..(tok + 1) * top_k as usize];
        let mut s_v1 = v1_slice.to_vec();
        let mut s_v3 = v3_slice.to_vec();
        s_v1.sort_unstable();
        s_v3.sort_unstable();
        assert_eq!(
            s_v1, s_v3,
            "Token {tok}: V1 ids {:?} != V3 ids {:?}",
            v1_slice, v3_slice
        );
    }

    // Weights should match within rtol 1e-5 (Step 1j.2 fix uses tree-reduce
    // softmax for V3 batched, matching V1 batched byte-identically except
    // for top-K order which renorm-sum is invariant to).
    for (i, (a, b)) in w_v1.iter().zip(w_v3.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 {
            abs / a.abs()
        } else {
            abs
        };
        if rel >= 1e-5 {
            eprintln!("k={i} V1={a:.6e} V3={b:.6e} abs={abs:.4e} rel={rel:.4e}");
        }
        assert!(rel < 1e-5, "k={i} weight rel {rel:.4e} exceeds 1e-5");
    }
}

/// Crafted tied logits across 18 batched tokens — tests Step 1j.2 fix
/// for tied-prob cases in batched prefill.
#[test]
fn v3_batched_parity_crafted_ties_18_tokens() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32;
    let n_tokens = 18u32;

    // Per-token: same crafted tie pattern (16 experts at logit=10.0,
    // rest at -10.0).  Top-K should pick lowest 8 of the 16 tied experts.
    let mut logits = vec![-10.0_f32; (num_experts * n_tokens) as usize];
    for tok in 0..n_tokens as usize {
        for i in 0..16 {
            logits[tok * num_experts as usize + i * 8] = 10.0;
        }
    }
    let scale = vec![1.0_f32; num_experts as usize];

    let (ids_v1, _w_v1) = run_once_batched(
        &device, &mut registry, "v1",
        &logits, &scale, num_experts, top_k, n_tokens,
    );
    let (ids_v3, _w_v3) = run_once_batched(
        &device, &mut registry, "v3",
        &logits, &scale, num_experts, top_k, n_tokens,
    );

    for tok in 0..n_tokens as usize {
        let v1_slice = &ids_v1[tok * top_k as usize..(tok + 1) * top_k as usize];
        let v3_slice = &ids_v3[tok * top_k as usize..(tok + 1) * top_k as usize];
        let mut s_v1 = v1_slice.to_vec();
        let mut s_v3 = v3_slice.to_vec();
        s_v1.sort_unstable();
        s_v3.sort_unstable();
        assert_eq!(
            s_v1, s_v3,
            "Token {tok}: tied-V1 {:?} != tied-V3 {:?}",
            v1_slice, v3_slice
        );
    }
}
