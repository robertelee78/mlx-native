//! Parity test: `fused_post_attn_triple_norm_f32_v2` ≡ V1
//! (ADR-028 iter-370 — V2-port of iter-186 kernel).
//!
//! V2 uses float4 + simd_sum + per-SG staging.  V1 uses scalar tree-reduce.
//! Same math, different reduction order — small f32 rounding deltas.
//! Tolerance: rel_error < 1e-4 (matches ADR-028 standing budget).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::rms_norm::dispatch_fused_post_attn_triple_norm_f32;
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

fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    use_v2: bool,
    hidden: &[f32],
    attn: &[f32],
    post_attn_w: &[f32],
    wa: &[f32],
    wb: &[f32],
    wc: &[f32],
    dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if use_v2 {
        std::env::set_var("HF2Q_FUSED_TRIPLE_NORM_V2", "1");
    } else {
        std::env::remove_var("HF2Q_FUSED_TRIPLE_NORM_V2");
    }

    let mk = |data: &[f32], shape| {
        let mut b = device.alloc_buffer(data.len() * 4, DType::F32, shape).expect("alloc");
        b.as_mut_slice::<f32>().expect("w").copy_from_slice(data);
        b
    };
    let h = mk(hidden, vec![dim]);
    let a = mk(attn, vec![dim]);
    let pw = mk(post_attn_w, vec![dim]);
    let wa_b = mk(wa, vec![dim]);
    let wb_b = mk(wb, vec![dim]);
    let wc_b = mk(wc, vec![dim]);
    let res_out = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("res");
    let oa = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("oa");
    let ob = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("ob");
    let oc = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).expect("oc");

    let mut enc = device.command_encoder().expect("enc");
    dispatch_fused_post_attn_triple_norm_f32(
        &mut enc, registry, device.metal_device(),
        &h, &a, &pw, &wa_b, &wb_b, &wc_b,
        &res_out, &oa, &ob, &oc,
        eps, 1, dim as u32,
    ).expect("dispatch");
    enc.commit_and_wait().expect("commit");

    (
        res_out.as_slice::<f32>().expect("o").to_vec(),
        oa.as_slice::<f32>().expect("o").to_vec(),
        ob.as_slice::<f32>().expect("o").to_vec(),
        oc.as_slice::<f32>().expect("o").to_vec(),
    )
}

fn check(label: &str, v1: &[f32], v2: &[f32], tol: f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        assert!(rel < tol,
            "{label} i={i} v1={a:.6e} v2={b:.6e} abs={abs:.4e} rel={rel:.4e}");
    }
    println!("{label} max_abs={max_abs:.4e} max_rel={max_rel:.4e}");
}

#[test]
fn parity_gemma4_dim2816_prod_shape() {
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let dim = 2816;
    let eps = 1e-6_f32;
    let mut rng = Xoshiro256::new(0xCAFE);
    let h = gaussian_vec(&mut rng, dim);
    let a = gaussian_vec(&mut rng, dim);
    let pw = gaussian_vec(&mut rng, dim);
    let wa = gaussian_vec(&mut rng, dim);
    let wb = gaussian_vec(&mut rng, dim);
    let wc = gaussian_vec(&mut rng, dim);

    let (r1, oa1, ob1, oc1) = run(&device, &mut registry, false, &h, &a, &pw, &wa, &wb, &wc, dim, eps);
    let (r2, oa2, ob2, oc2) = run(&device, &mut registry, true,  &h, &a, &pw, &wa, &wb, &wc, dim, eps);

    check("residual_out", &r1, &r2, 1e-4);
    check("output_a", &oa1, &oa2, 1e-4);
    check("output_b", &ob1, &ob2, 1e-4);
    check("output_c", &oc1, &oc2, 1e-4);
}
