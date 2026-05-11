//! Parity test: `dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2` ≡
//! chain of `moe_weighted_sum_encode` + `dispatch_fused_post_ff_norm2_endlayer_f32`
//! (with V2 default-on).
//!
//! ADR-028 iter-367 — fuses moe_weighted_sum INTO the production-default Path A
//! end-of-layer kernel.  V2 uses simd_sum + per-SG staging for both RMS reductions.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::moe_dispatch::moe_weighted_sum_encode;
use mlx_native::ops::rms_norm::{
    dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2,
    dispatch_fused_post_ff_norm2_endlayer_f32,
};
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
        let r = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        r
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

#[allow(clippy::type_complexity)]
fn run_chain(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    experts: &[f32],
    weights: &[f32],
    attn_out: &[f32],
    residual: &[f32],
    w2: &[f32],
    w3: &[f32],
    scalar: &[f32],
    dim: usize,
    top_k: usize,
    eps: f32,
    scalar_is_vector: bool,
) -> (Vec<f32>, Vec<f32>) {
    // Force V2 on for the end-of-layer kernel (matches production default).
    std::env::set_var("HF2Q_FUSED_POST_FF_NORM2_V2", "1");

    let mut e_b = device.alloc_buffer(experts.len() * 4, DType::F32, vec![top_k, dim]).expect("e");
    e_b.as_mut_slice::<f32>().expect("w").copy_from_slice(experts);
    let mut w_b = device.alloc_buffer(weights.len() * 4, DType::F32, vec![top_k]).expect("w");
    w_b.as_mut_slice::<f32>().expect("w").copy_from_slice(weights);
    let mut a_b = device.alloc_buffer(attn_out.len() * 4, DType::F32, vec![dim]).expect("a");
    a_b.as_mut_slice::<f32>().expect("w").copy_from_slice(attn_out);
    let mut r_b = device.alloc_buffer(residual.len() * 4, DType::F32, vec![dim]).expect("r");
    r_b.as_mut_slice::<f32>().expect("w").copy_from_slice(residual);
    let mut w2_b = device.alloc_buffer(w2.len() * 4, DType::F32, vec![dim]).expect("w2");
    w2_b.as_mut_slice::<f32>().expect("w").copy_from_slice(w2);
    let mut w3_b = device.alloc_buffer(w3.len() * 4, DType::F32, vec![dim]).expect("w3");
    w3_b.as_mut_slice::<f32>().expect("w").copy_from_slice(w3);
    let mut s_b = device.alloc_buffer(scalar.len() * 4, DType::F32, vec![scalar.len()]).expect("s");
    s_b.as_mut_slice::<f32>().expect("w").copy_from_slice(scalar);

    let accum = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("acc");
    let mlp_down = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("mlp");
    let hidden = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("h");

    let mut enc = device.command_encoder().expect("enc");
    moe_weighted_sum_encode(
        &mut enc, registry, device.metal_device(),
        &e_b, &w_b, &accum, dim, top_k,
    ).expect("wsum");
    enc.memory_barrier();
    dispatch_fused_post_ff_norm2_endlayer_f32(
        &mut enc, registry, device.metal_device(),
        &a_b, &accum, &r_b, &w2_b, &w3_b, &s_b,
        &mlp_down, &hidden,
        eps, 1, dim as u32, scalar_is_vector,
    ).expect("endlayer");
    enc.commit_and_wait().expect("commit");

    (
        mlp_down.as_slice::<f32>().expect("o").to_vec(),
        hidden.as_slice::<f32>().expect("o").to_vec(),
    )
}

#[allow(clippy::type_complexity)]
fn run_fused(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    experts: &[f32],
    weights: &[f32],
    attn_out: &[f32],
    residual: &[f32],
    w2: &[f32],
    w3: &[f32],
    scalar: &[f32],
    dim: usize,
    top_k: usize,
    eps: f32,
    scalar_is_vector: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut e_b = device.alloc_buffer(experts.len() * 4, DType::F32, vec![top_k, dim]).expect("e");
    e_b.as_mut_slice::<f32>().expect("w").copy_from_slice(experts);
    let mut w_b = device.alloc_buffer(weights.len() * 4, DType::F32, vec![top_k]).expect("w");
    w_b.as_mut_slice::<f32>().expect("w").copy_from_slice(weights);
    let mut a_b = device.alloc_buffer(attn_out.len() * 4, DType::F32, vec![dim]).expect("a");
    a_b.as_mut_slice::<f32>().expect("w").copy_from_slice(attn_out);
    let mut r_b = device.alloc_buffer(residual.len() * 4, DType::F32, vec![dim]).expect("r");
    r_b.as_mut_slice::<f32>().expect("w").copy_from_slice(residual);
    let mut w2_b = device.alloc_buffer(w2.len() * 4, DType::F32, vec![dim]).expect("w2");
    w2_b.as_mut_slice::<f32>().expect("w").copy_from_slice(w2);
    let mut w3_b = device.alloc_buffer(w3.len() * 4, DType::F32, vec![dim]).expect("w3");
    w3_b.as_mut_slice::<f32>().expect("w").copy_from_slice(w3);
    let mut s_b = device.alloc_buffer(scalar.len() * 4, DType::F32, vec![scalar.len()]).expect("s");
    s_b.as_mut_slice::<f32>().expect("w").copy_from_slice(scalar);

    let mlp_down = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("mlp");
    let hidden = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("h");

    let mut enc = device.command_encoder().expect("enc");
    dispatch_fused_moe_wsum_post_ff_norm2_endlayer_f32_v2(
        &mut enc, registry, device.metal_device(),
        &e_b, &w_b, &a_b, &r_b, &w2_b, &w3_b, &s_b,
        &mlp_down, &hidden,
        eps, 1, dim as u32, top_k as u32, scalar_is_vector,
    ).expect("fused");
    enc.commit_and_wait().expect("commit");

    (
        mlp_down.as_slice::<f32>().expect("o").to_vec(),
        hidden.as_slice::<f32>().expect("o").to_vec(),
    )
}

fn check(label: &str, chain: &[f32], fused: &[f32], tol: f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in chain.iter().zip(fused.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        assert!(rel < tol,
            "{label} i={i} chain={a:.6e} fused={b:.6e} abs={abs:.4e} rel={rel:.4e}");
    }
    println!("{label} max_abs={max_abs:.4e} max_rel={max_rel:.4e}");
}

#[test]
fn parity_gemma4_dim3584_top8_scalar_broadcast() {
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let dim = 3584;
    let top_k = 8;
    let eps = 1e-6_f32;
    let mut rng = Xoshiro256::new(0xCAFE_BEEF);
    let experts = gaussian_vec(&mut rng, top_k * dim);
    let weights = gaussian_vec(&mut rng, top_k);
    let attn = gaussian_vec(&mut rng, dim);
    let res = gaussian_vec(&mut rng, dim);
    let w2 = gaussian_vec(&mut rng, dim);
    let w3 = gaussian_vec(&mut rng, dim);
    let scalar = vec![1.0_f32]; // scalar broadcast

    let (cm, ch) = run_chain(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, false);
    let (fm, fh) = run_fused(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, false);

    check("mlp_down (broadcast)", &cm, &fm, 1e-4);
    check("hidden   (broadcast)", &ch, &fh, 1e-4);
}

#[test]
fn parity_gemma4_dim2816_top8_actual_prod_shape() {
    // Production gemma4-ara-2pass-APEX-Q5_K_M.gguf reports hidden=2816 at
    // model-load (not 3584 as some kernel comments claim).  This is the shape
    // the iter-367 wiring will actually exercise.
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let dim = 2816;
    let top_k = 8;
    let eps = 1e-6_f32;
    let mut rng = Xoshiro256::new(0xFEED);
    let experts = gaussian_vec(&mut rng, top_k * dim);
    let weights = gaussian_vec(&mut rng, top_k);
    let attn = gaussian_vec(&mut rng, dim);
    let res = gaussian_vec(&mut rng, dim);
    let w2 = gaussian_vec(&mut rng, dim);
    let w3 = gaussian_vec(&mut rng, dim);
    let scalar: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let (cm, ch) = run_chain(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, true);
    let (fm, fh) = run_fused(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, true);

    check("mlp_down (prod 2816)", &cm, &fm, 1e-4);
    check("hidden   (prod 2816)", &ch, &fh, 1e-4);
}

#[test]
fn parity_gemma4_dim3584_top8_scalar_vector() {
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let dim = 3584;
    let top_k = 8;
    let eps = 1e-6_f32;
    let mut rng = Xoshiro256::new(0xDEAD_BEEF);
    let experts = gaussian_vec(&mut rng, top_k * dim);
    let weights = gaussian_vec(&mut rng, top_k);
    let attn = gaussian_vec(&mut rng, dim);
    let res = gaussian_vec(&mut rng, dim);
    let w2 = gaussian_vec(&mut rng, dim);
    let w3 = gaussian_vec(&mut rng, dim);
    let scalar: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let (cm, ch) = run_chain(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, true);
    let (fm, fh) = run_fused(&device, &mut registry, &experts, &weights, &attn, &res, &w2, &w3, &scalar, dim, top_k, eps, true);

    check("mlp_down (vector)", &cm, &fm, 1e-4);
    check("hidden   (vector)", &ch, &fh, 1e-4);
}
