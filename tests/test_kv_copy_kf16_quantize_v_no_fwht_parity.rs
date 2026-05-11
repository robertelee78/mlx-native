//! Byte-identity parity test: `dispatch_kv_copy_kf16_quantize_v_no_fwht` ≡
//! `dispatch_kv_cache_copy_batch_f32_to_f16` + `dispatch_kv_quantize_v_no_fwht`
//! at identical params (ADR-028 Phase 10c.5 / iter-354).
//!
//! Each Z-stream of the fused kernel takes the SAME math path as its
//! stand-alone counterpart; this test enforces that property byte-for-byte
//! so Phase 10c.5 wiring at hf2q can drop the two stand-alone dispatches
//! without any output-shift risk.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::hadamard_quantize_kv::{
    dispatch_kv_copy_kf16_quantize_v_no_fwht,
    dispatch_kv_quantize_v_no_fwht,
};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32_to_f16;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---- PRNG (xoshiro256**) ----
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

#[test]
fn fused_kf16_v_quantize_byte_identical_to_two_dispatches_d256_8bit() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();

    let n_heads = 8u32;
    let hd = 256u32;
    let cap = 1u32;
    let cbits = 8u32;

    let mut rng = Xoshiro256::new(0xFEED1234);
    let src_k = gaussian_vec(&mut rng, n_heads as usize * hd as usize);
    let src_v = gaussian_vec(&mut rng, n_heads as usize * hd as usize);

    // Path A: two stand-alone dispatches.
    let mut srck_a = device.alloc_buffer(src_k.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc srck_a");
    srck_a.as_mut_slice::<f32>().expect("write srck_a").copy_from_slice(&src_k);
    let mut srcv_a = device.alloc_buffer(src_v.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc srcv_a");
    srcv_a.as_mut_slice::<f32>().expect("write srcv_a").copy_from_slice(&src_v);
    let cache_k_a = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize * 2, DType::F16, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc cache_k_a");
    let packed_v_a = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize, DType::U8, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc packed_v_a");
    let norms_v_a = device.alloc_buffer(n_heads as usize * cap as usize * 4, DType::F32, vec![n_heads as usize * cap as usize]).expect("alloc norms_v_a");

    let mut enc_a = device.command_encoder().expect("enc_a");
    dispatch_kv_cache_copy_batch_f32_to_f16(
        &mut enc_a, &mut registry, device.metal_device(),
        &srck_a, &cache_k_a,
        n_heads, hd, cap, 0,
    ).expect("Path A: K copy");
    dispatch_kv_quantize_v_no_fwht(
        &mut enc_a, &mut registry, device.metal_device(),
        &srcv_a, &packed_v_a, &norms_v_a,
        n_heads, hd, cap, 0,
        false, 1.0, cbits,
    ).expect("Path A: V encode");
    enc_a.commit_and_wait().expect("commit_a");

    // Path B: fused dispatch.
    let mut srck_b = device.alloc_buffer(src_k.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc srck_b");
    srck_b.as_mut_slice::<f32>().expect("write srck_b").copy_from_slice(&src_k);
    let mut srcv_b = device.alloc_buffer(src_v.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc srcv_b");
    srcv_b.as_mut_slice::<f32>().expect("write srcv_b").copy_from_slice(&src_v);
    let cache_k_b = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize * 2, DType::F16, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc cache_k_b");
    let packed_v_b = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize, DType::U8, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc packed_v_b");
    let norms_v_b = device.alloc_buffer(n_heads as usize * cap as usize * 4, DType::F32, vec![n_heads as usize * cap as usize]).expect("alloc norms_v_b");

    let mut enc_b = device.command_encoder().expect("enc_b");
    dispatch_kv_copy_kf16_quantize_v_no_fwht(
        &mut enc_b, &mut registry, device.metal_device(),
        &srck_b, &srcv_b,
        &cache_k_b, &packed_v_b, &norms_v_b,
        n_heads, hd, cap, 0,
        false, 1.0, cbits,
    ).expect("Path B: fused");
    enc_b.commit_and_wait().expect("commit_b");

    // Byte-compare K (F16 cache).
    let cache_k_bytes_a = cache_k_a.as_slice::<u8>().expect("read cache_k_a");
    let cache_k_bytes_b = cache_k_b.as_slice::<u8>().expect("read cache_k_b");
    assert_eq!(cache_k_bytes_a.len(), cache_k_bytes_b.len(), "K cache size mismatch");
    for (i, (a, b)) in cache_k_bytes_a.iter().zip(cache_k_bytes_b.iter()).enumerate() {
        assert_eq!(a, b, "K cache byte mismatch at index {} (path A={}, path B={})", i, a, b);
    }

    // Byte-compare V packed.
    let packed_v_bytes_a = packed_v_a.as_slice::<u8>().expect("read packed_v_a");
    let packed_v_bytes_b = packed_v_b.as_slice::<u8>().expect("read packed_v_b");
    assert_eq!(packed_v_bytes_a, packed_v_bytes_b, "V packed bytes differ between fused and stand-alone");

    // Byte-compare V norms.
    let norms_a = norms_v_a.as_slice::<f32>().expect("read norms_v_a");
    let norms_b = norms_v_b.as_slice::<f32>().expect("read norms_v_b");
    for (i, (a, b)) in norms_a.iter().zip(norms_b.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(),
            "V norm bit mismatch at index {} (path A={}, path B={})", i, a, b);
    }
}

#[test]
fn fused_kf16_v_quantize_runs_with_sliding_ring_position() {
    // Sanity: write at non-zero ring position; just make sure no panic, no NaN.
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    let n_heads = 4u32; let hd = 256u32; let cap = 64u32; let cbits = 8u32;

    let mut rng = Xoshiro256::new(0xBADCAFE);
    let src_k = gaussian_vec(&mut rng, n_heads as usize * hd as usize);
    let src_v = gaussian_vec(&mut rng, n_heads as usize * hd as usize);

    let mut srck = device.alloc_buffer(src_k.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc");
    srck.as_mut_slice::<f32>().expect("w").copy_from_slice(&src_k);
    let mut srcv = device.alloc_buffer(src_v.len() * 4, DType::F32, vec![n_heads as usize, hd as usize]).expect("alloc");
    srcv.as_mut_slice::<f32>().expect("w").copy_from_slice(&src_v);
    let cache_k = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize * 2, DType::F16, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc");
    let packed_v = device.alloc_buffer(n_heads as usize * cap as usize * hd as usize, DType::U8, vec![n_heads as usize, cap as usize, hd as usize]).expect("alloc");
    let norms_v = device.alloc_buffer(n_heads as usize * cap as usize * 4, DType::F32, vec![n_heads as usize * cap as usize]).expect("alloc");

    let mut enc = device.command_encoder().expect("enc");
    dispatch_kv_copy_kf16_quantize_v_no_fwht(
        &mut enc, &mut registry, device.metal_device(),
        &srck, &srcv, &cache_k, &packed_v, &norms_v,
        n_heads, hd, cap, 100,         // write_pos = 100 → wraps to 100 % 64 = 36
        true, 1.0, cbits,
    ).expect("dispatch");
    enc.commit_and_wait().expect("commit");

    // Spot-check: norms at position 36 should be finite, non-zero.
    let norms = norms_v.as_slice::<f32>().expect("read norms");
    for h in 0..n_heads as usize {
        let n = norms[h * cap as usize + 36];
        assert!(n.is_finite() && n > 0.0, "head {h} norm at pos 36 is invalid: {n}");
    }
}
