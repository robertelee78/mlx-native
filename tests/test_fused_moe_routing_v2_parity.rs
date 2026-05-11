//! Parity test: `fused_moe_routing_f32_v2` ≡ `fused_moe_routing_f32`
//! (ADR-028 Phase 13.2 / iter-363).
//!
//! V2 uses simd_max + simd_sum + per-simdgroup partial-result staging.
//! V1 uses scalar tree reduction. Same math, different reduction order
//! (so f32 sums can have small rounding deltas).
//!
//! Top-K phase is byte-identical to V1 (single-thread serial).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32;
use mlx_native::{DType, KernelRegistry, MlxDevice};

struct Xoshiro256 { s: [u64; 4] }
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
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        result
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn randn_pair(rng: &mut Xoshiro256) -> (f64, f64) {
    loop {
        let u1 = rng.next_f64(); let u2 = rng.next_f64();
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
        if out.len() < n { out.push(b as f32); }
    }
    out
}

fn run_once(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    use_v2: bool,
    logits_data: &[f32],
    per_expert_scale: &[f32],
    num_experts: u32,
    top_k: u32,
) -> (Vec<u32>, Vec<f32>) {
    if use_v2 {
        std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V2", "1");
    } else {
        std::env::remove_var("HF2Q_FUSED_MOE_ROUTING_V2");
    }
    let mut logits = device.alloc_buffer(num_experts as usize * 4, DType::F32, vec![num_experts as usize]).expect("alloc logits");
    logits.as_mut_slice::<f32>().expect("w").copy_from_slice(logits_data);
    let expert_ids = device.alloc_buffer(top_k as usize * 4, DType::U32, vec![top_k as usize]).expect("alloc ids");
    let routing_weights = device.alloc_buffer(top_k as usize * 4, DType::F32, vec![top_k as usize]).expect("alloc weights");
    let mut scale = device.alloc_buffer(num_experts as usize * 4, DType::F32, vec![num_experts as usize]).expect("alloc scale");
    scale.as_mut_slice::<f32>().expect("w").copy_from_slice(per_expert_scale);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_fused_moe_routing_f32(
        &mut enc, registry, device.metal_device(),
        &logits, &expert_ids, &routing_weights, &scale,
        num_experts, top_k,
    ).expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let ids = expert_ids.as_slice::<u32>().expect("read ids").to_vec();
    let weights = routing_weights.as_slice::<f32>().expect("read weights").to_vec();
    (ids, weights)
}

#[test]
fn v2_parity_gemma4_moe_routing_128_experts_top_8() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32; // gemma4 default
    let mut rng = Xoshiro256::new(0xCAFE);
    let logits = gaussian_vec(&mut rng, num_experts as usize);
    let scale = vec![1.0_f32; num_experts as usize]; // unit scale

    let (ids1, w1) = run_once(&device, &mut registry, false, &logits, &scale, num_experts, top_k);
    let (ids2, w2) = run_once(&device, &mut registry, true,  &logits, &scale, num_experts, top_k);

    // Top-K IDs must match exactly (same selection algorithm).
    assert_eq!(ids1, ids2,
        "expert_ids differ: V1={:?} V2={:?}", ids1, ids2);

    // Weights may differ slightly due to f32 reduction order.  Compare with rtol 1e-5.
    for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        println!("k={i} V1={a:.6e} V2={b:.6e} abs={abs:.4e} rel={rel:.4e}");
        assert!(rel < 1e-5, "k={i} weight rel {rel:.4e} exceeds 1e-5");
    }
}

#[test]
fn v2_parity_with_per_expert_scale() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32;
    let mut rng = Xoshiro256::new(0xFEED);
    let logits = gaussian_vec(&mut rng, num_experts as usize);
    let scale: Vec<f32> = (0..num_experts).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let (ids1, w1) = run_once(&device, &mut registry, false, &logits, &scale, num_experts, top_k);
    let (ids2, w2) = run_once(&device, &mut registry, true,  &logits, &scale, num_experts, top_k);

    assert_eq!(ids1, ids2, "expert_ids must match exactly");
    for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        println!("k={i} V1={a:.6e} V2={b:.6e} abs={abs:.4e} rel={rel:.4e}");
        assert!(rel < 1e-5, "k={i} weight rel {rel:.4e} exceeds 1e-5");
    }
}
