//! Parity test: `fused_post_ff_norm2_endlayer_f32_v2` ≡
//! `fused_post_ff_norm2_endlayer_f32` at gemma4 hidden_dim=3584
//! (ADR-028 Phase 13 / iter-362).
//!
//! V2 uses float4 loads + simd_sum reduction (mirrors rms_norm_f32_v2
//! pattern from iter-310).  V1 uses scalar loads + tree reduction.  The
//! sum-of-squares accumulation order differs slightly, so we use rtol
//! 1e-4 / atol 1e-5 (same band as rms_norm V2 parity at iter-310).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::rms_norm::dispatch_fused_post_ff_norm2_endlayer_f32;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---- PRNG ----
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
    attn_out_data: &[f32],
    moe_data: &[f32],
    residual_data: &[f32],
    w2_data: &[f32],
    w3_data: &[f32],
    layer_scalar_data: &[f32],
    rows: u32,
    dim: u32,
    scalar_is_vector: bool,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    // toggle env, dispatch, return (mlp_down, hidden) outputs.
    if use_v2 {
        std::env::set_var("HF2Q_FUSED_POST_FF_NORM2_V2", "1");
    } else {
        std::env::remove_var("HF2Q_FUSED_POST_FF_NORM2_V2");
    }
    let n = (rows as usize) * (dim as usize);
    let mut attn_out = device.alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize]).expect("alloc attn_out");
    attn_out.as_mut_slice::<f32>().expect("w attn_out").copy_from_slice(attn_out_data);
    let mut moe = device.alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize]).expect("alloc moe");
    moe.as_mut_slice::<f32>().expect("w moe").copy_from_slice(moe_data);
    let mut residual = device.alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize]).expect("alloc residual");
    residual.as_mut_slice::<f32>().expect("w residual").copy_from_slice(residual_data);
    let mut w2 = device.alloc_buffer(dim as usize * 4, DType::F32, vec![dim as usize]).expect("alloc w2");
    w2.as_mut_slice::<f32>().expect("w w2").copy_from_slice(w2_data);
    let mut w3 = device.alloc_buffer(dim as usize * 4, DType::F32, vec![dim as usize]).expect("alloc w3");
    w3.as_mut_slice::<f32>().expect("w w3").copy_from_slice(w3_data);
    let scalar_sz = if scalar_is_vector { dim as usize } else { 1 };
    let mut layer_scalar = device.alloc_buffer(scalar_sz * 4, DType::F32, vec![scalar_sz]).expect("alloc layer_scalar");
    layer_scalar.as_mut_slice::<f32>().expect("w layer_scalar").copy_from_slice(&layer_scalar_data[..scalar_sz]);
    let mlp_down = device.alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize]).expect("alloc mlp_down");
    let hidden = device.alloc_buffer(n * 4, DType::F32, vec![rows as usize, dim as usize]).expect("alloc hidden");

    let mut enc = device.command_encoder().expect("enc");
    dispatch_fused_post_ff_norm2_endlayer_f32(
        &mut enc, registry, device.metal_device(),
        &attn_out, &moe, &residual,
        &w2, &w3, &layer_scalar,
        &mlp_down, &hidden,
        eps, rows, dim, scalar_is_vector,
    ).expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let mlp_down_out = mlp_down.as_slice::<f32>().expect("read mlp_down").to_vec();
    let hidden_out = hidden.as_slice::<f32>().expect("read hidden").to_vec();
    (mlp_down_out, hidden_out)
}

fn max_relative_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let abs = (x - y).abs();
        let rel = if x.abs() > 1e-6 { abs / x.abs() } else { abs };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
    }
    (max_abs, max_rel)
}

#[test]
fn v2_parity_gemma4_hidden_dim_scalar_broadcast() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let rows = 1u32;
    let dim = 3584u32; // gemma4 hidden
    let mut rng = Xoshiro256::new(0xCAFEFEED);
    let attn = gaussian_vec(&mut rng, (rows * dim) as usize);
    let moe = gaussian_vec(&mut rng, (rows * dim) as usize);
    let residual = gaussian_vec(&mut rng, (rows * dim) as usize);
    let w2 = gaussian_vec(&mut rng, dim as usize);
    let w3 = gaussian_vec(&mut rng, dim as usize);
    let scalar = vec![0.13_f32]; // broadcast scalar

    let (mlp1, hid1) = run_once(&device, &mut registry, false, &attn, &moe, &residual, &w2, &w3, &scalar, rows, dim, false, 1e-6);
    let (mlp2, hid2) = run_once(&device, &mut registry, true,  &attn, &moe, &residual, &w2, &w3, &scalar, rows, dim, false, 1e-6);
    let (mlp_abs, mlp_rel) = max_relative_diff(&mlp1, &mlp2);
    let (hid_abs, hid_rel) = max_relative_diff(&hid1, &hid2);
    println!("scalar broadcast: mlp max_abs={mlp_abs:.4e} max_rel={mlp_rel:.4e} | hidden max_abs={hid_abs:.4e} max_rel={hid_rel:.4e}");
    assert!(mlp_rel < 1e-4 && mlp_abs < 5e-4, "mlp_down parity exceeded band");
    assert!(hid_rel < 1e-4 && hid_abs < 5e-5, "hidden parity exceeded band");
}

#[test]
fn v2_parity_gemma4_hidden_dim_scalar_vector() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let rows = 1u32;
    let dim = 3584u32;
    let mut rng = Xoshiro256::new(0xFEEDC0DE);
    let attn = gaussian_vec(&mut rng, (rows * dim) as usize);
    let moe = gaussian_vec(&mut rng, (rows * dim) as usize);
    let residual = gaussian_vec(&mut rng, (rows * dim) as usize);
    let w2 = gaussian_vec(&mut rng, dim as usize);
    let w3 = gaussian_vec(&mut rng, dim as usize);
    let scalar = gaussian_vec(&mut rng, dim as usize); // per-channel scalar

    let (mlp1, hid1) = run_once(&device, &mut registry, false, &attn, &moe, &residual, &w2, &w3, &scalar, rows, dim, true, 1e-6);
    let (mlp2, hid2) = run_once(&device, &mut registry, true,  &attn, &moe, &residual, &w2, &w3, &scalar, rows, dim, true, 1e-6);
    let (mlp_abs, mlp_rel) = max_relative_diff(&mlp1, &mlp2);
    let (hid_abs, hid_rel) = max_relative_diff(&hid1, &hid2);
    println!("scalar vector: mlp max_abs={mlp_abs:.4e} max_rel={mlp_rel:.4e} | hidden max_abs={hid_abs:.4e} max_rel={hid_rel:.4e}");
    assert!(mlp_rel < 1e-4 && mlp_abs < 5e-4, "mlp_down parity exceeded band");
    assert!(hid_rel < 1e-4 && hid_abs < 5e-4, "hidden parity exceeded band");
}
