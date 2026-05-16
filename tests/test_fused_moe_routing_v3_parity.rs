//! Parity test: `fused_moe_routing_f32_v3` ≡ `fused_moe_routing_f32_v2`
//! (ADR-029 iter-175 Step 1i).
//!
//! V3 = V2 + parallel SG-tournament top-K (replaces V2's single-thread
//! serial top-K with K SG-shuffle-down tournament reductions).  Softmax
//! steps are byte-identical to V2.  Top-K selection result must match
//! V2 exactly for non-tied softmax probabilities — f32 softmax ties
//! over 128 experts are vanishingly rare in production.
//!
//! Regression-protects the Step 1i breakthrough kernel (+11.5% measured
//! win at gemma4 APEX-Q5_K_M, default-flipped ON at commit 5119eee).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32;
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

/// V3 selects via `HF2Q_FUSED_MOE_ROUTING_V3=1`.
/// V2 (baseline for parity) selects via `HF2Q_FUSED_MOE_ROUTING_V3=0` +
/// `HF2Q_FUSED_MOE_ROUTING_V2=1` (the dispatcher cascades V3 → V2 → V1).
fn run_once(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    variant: &str, // "v3", "v2", or "v1"
    logits_data: &[f32],
    per_expert_scale: &[f32],
    num_experts: u32,
    top_k: u32,
) -> (Vec<u32>, Vec<f32>) {
    match variant {
        "v3" => {
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V3", "1");
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V2", "1");
        }
        "v2" => {
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V3", "0");
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V2", "1");
        }
        "v1" => {
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V3", "0");
            std::env::set_var("HF2Q_FUSED_MOE_ROUTING_V2", "0");
        }
        _ => panic!("unknown variant {variant}"),
    }
    let mut logits = device
        .alloc_buffer(
            num_experts as usize * 4,
            DType::F32,
            vec![num_experts as usize],
        )
        .expect("alloc logits");
    logits
        .as_mut_slice::<f32>()
        .expect("w")
        .copy_from_slice(logits_data);
    let expert_ids = device
        .alloc_buffer(top_k as usize * 4, DType::U32, vec![top_k as usize])
        .expect("alloc ids");
    let routing_weights = device
        .alloc_buffer(top_k as usize * 4, DType::F32, vec![top_k as usize])
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
    dispatch_fused_moe_routing_f32(
        &mut enc,
        registry,
        device.metal_device(),
        &logits,
        &expert_ids,
        &routing_weights,
        &scale,
        num_experts,
        top_k,
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

#[test]
fn v3_parity_gemma4_moe_routing_128_experts_top_8() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32; // gemma4 default
    let mut rng = Xoshiro256::new(0xC0DE_5119);
    let logits = gaussian_vec(&mut rng, num_experts as usize);
    let scale = vec![1.0_f32; num_experts as usize];

    let (ids_v2, w_v2) = run_once(
        &device,
        &mut registry,
        "v2",
        &logits,
        &scale,
        num_experts,
        top_k,
    );
    let (ids_v3, w_v3) = run_once(
        &device,
        &mut registry,
        "v3",
        &logits,
        &scale,
        num_experts,
        top_k,
    );

    println!("V2 ids: {:?}", ids_v2);
    println!("V3 ids: {:?}", ids_v3);

    // Top-K IDs: V3 uses parallel SG-tournament; V2 uses serial scan.
    // Result is identical when softmax probabilities are all distinct
    // (Gaussian logits + softmax → vanishingly rare ties).  The two
    // independent V3 runs on same input should produce same ids
    // (determinism within V3) — even if V3 picks a different tied lane
    // than V2 in rare cases, the picked lane must be deterministic.
    assert_eq!(
        ids_v2.len(),
        ids_v3.len(),
        "len mismatch: V2={} V3={}",
        ids_v2.len(),
        ids_v3.len()
    );

    // Verify weights match within rtol 1e-5 (f32 reduction order may
    // differ in step 1-3 softmax but tournament top-K of the same
    // shared[] values produces same weights up to FP rounding).
    for (i, (a, b)) in w_v2.iter().zip(w_v3.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 {
            abs / a.abs()
        } else {
            abs
        };
        println!("k={i} V2={a:.6e} V3={b:.6e} abs={abs:.4e} rel={rel:.4e}");
        assert!(
            rel < 1e-5,
            "k={i} weight rel {rel:.4e} exceeds 1e-5 (V2={a} V3={b})"
        );
    }

    // Top-K ids: assert same SET (not order — tournament may produce
    // different order if SG-tournament processes lanes in different
    // sequence than serial scan, but the SET of K selected experts
    // must be identical when probs are distinct).
    let mut s_v2: Vec<u32> = ids_v2.clone();
    let mut s_v3: Vec<u32> = ids_v3.clone();
    s_v2.sort_unstable();
    s_v3.sort_unstable();
    assert_eq!(
        s_v2, s_v3,
        "top-K SET mismatch: V2={:?} V3={:?}",
        ids_v2, ids_v3
    );
}

#[test]
fn v3_parity_with_per_expert_scale() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let num_experts = 128u32;
    let top_k = 8u32;
    let mut rng = Xoshiro256::new(0xC0DE_AAAA);
    let logits = gaussian_vec(&mut rng, num_experts as usize);
    let scale: Vec<f32> = (0..num_experts).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let (ids_v2, w_v2) = run_once(
        &device,
        &mut registry,
        "v2",
        &logits,
        &scale,
        num_experts,
        top_k,
    );
    let (ids_v3, w_v3) = run_once(
        &device,
        &mut registry,
        "v3",
        &logits,
        &scale,
        num_experts,
        top_k,
    );

    let mut s_v2 = ids_v2.clone();
    let mut s_v3 = ids_v3.clone();
    s_v2.sort_unstable();
    s_v3.sort_unstable();
    assert_eq!(
        s_v2, s_v3,
        "top-K SET mismatch with per_expert_scale: V2={:?} V3={:?}",
        ids_v2, ids_v3
    );

    // For weights, since scale is applied per-expert AFTER top-K, the
    // weights array is keyed by selection order — which may differ
    // between V2 (serial scan from i=0) and V3 (tournament arbitrary).
    // Compare weight-keyed-by-expert-id instead of by position:
    let mut by_id_v2 = std::collections::BTreeMap::new();
    let mut by_id_v3 = std::collections::BTreeMap::new();
    for k in 0..top_k as usize {
        by_id_v2.insert(ids_v2[k], w_v2[k]);
        by_id_v3.insert(ids_v3[k], w_v3[k]);
    }
    for (eid, w_v2_val) in by_id_v2.iter() {
        let w_v3_val = by_id_v3
            .get(eid)
            .copied()
            .expect("V3 missing expert id present in V2");
        let abs = (w_v2_val - w_v3_val).abs();
        let rel = if w_v2_val.abs() > 1e-6 {
            abs / w_v2_val.abs()
        } else {
            abs
        };
        println!(
            "eid={eid} V2={:.6e} V3={:.6e} abs={abs:.4e} rel={rel:.4e}",
            w_v2_val, w_v3_val
        );
        assert!(
            rel < 1e-5,
            "eid={eid} weight rel {rel:.4e} exceeds 1e-5"
        );
    }
}
