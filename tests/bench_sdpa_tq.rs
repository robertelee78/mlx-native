//! Microbenchmark: TQ SDPA vs F16 SDPA kernel timing.
//!
//! Measures isolated kernel execution time at Gemma-4-27B shapes
//! with realistic KV sequence lengths. No model overhead.
//!
//! Run: cargo test --release --test bench_sdpa_tq -- --nocapture

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use std::time::Instant;

use metal::MTLSize;
use mlx_native::ops::encode_helpers::{as_bytes, KernelArg};
use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::ops::flash_attn_vec_tq::{self, FlashAttnVecTqParams};
use mlx_native::turboquant::{fwht_inplace, CODEBOOK_4BIT};
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

fn random_f32_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let (a, b) = randn_pair(rng);
        v.push(a as f32);
        if v.len() < n { v.push(b as f32); }
    }
    v
}

fn nearest_centroid_4bit(value: f32) -> u8 {
    let mut idx: u8 = 0;
    for i in 0..15 {
        let boundary = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
        if value > boundary { idx = (i + 1) as u8; }
    }
    idx
}

// ---- Setup ----

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    flash_attn_vec_tq::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry);
    // Register v2 tiled kernel for benchmarking.
    let v2_src = include_str!("../src/shaders/flash_attn_vec_tq_v2.metal");
    registry.register_source("flash_attn_vec_tq_v2_dk256", v2_src);
    registry.register_source("flash_attn_vec_tq_v2_dk512", v2_src);
    (device, registry)
}

// ---- TQ SDPA bench ----

fn bench_tq_sdpa(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    nwg_override: Option<u32>,
    warmup: usize,
    iters: usize,
) -> f64 {
    let mut rng = Xoshiro256::new(42);
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let hd = head_dim as usize;
    let kvl = kv_seq_len as usize;

    // Generate random Q
    let q_data = random_f32_vec(&mut rng, nh * hd);

    // Generate random TQ-packed KV cache
    let mut k_packed = vec![0u8; nkv * kvl * (hd / 2)];
    let mut k_norms = vec![0.0f32; nkv * kvl];
    let mut v_packed = vec![0u8; nkv * kvl * (hd / 2)];
    let mut v_norms = vec![0.0f32; nkv * kvl];

    for kv_h in 0..nkv {
        for p in 0..kvl {
            // Random K vector → quantize
            let k_vec = random_f32_vec(&mut rng, hd);
            let mut rotated = k_vec.clone();
            fwht_inplace(&mut rotated).unwrap();
            let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
            k_norms[kv_h * kvl + p] = norm;
            if norm > 1e-30 {
                let inv_norm = 1.0 / norm;
                let scale = (hd as f32).sqrt();
                let offset = (kv_h * kvl + p) * (hd / 2);
                for c in 0..hd {
                    let scaled = rotated[c] * inv_norm * scale;
                    let idx = nearest_centroid_4bit(scaled);
                    let byte_idx = c / 2;
                    if c % 2 == 0 {
                        k_packed[offset + byte_idx] = idx & 0xF;
                    } else {
                        k_packed[offset + byte_idx] |= (idx & 0xF) << 4;
                    }
                }
            }

            // Random V vector → quantize
            let v_vec = random_f32_vec(&mut rng, hd);
            let mut v_rot = v_vec.clone();
            fwht_inplace(&mut v_rot).unwrap();
            let v_norm: f32 = v_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            v_norms[kv_h * kvl + p] = v_norm;
            if v_norm > 1e-30 {
                let inv_norm = 1.0 / v_norm;
                let scale = (hd as f32).sqrt();
                let offset = (kv_h * kvl + p) * (hd / 2);
                for c in 0..hd {
                    let scaled = v_rot[c] * inv_norm * scale;
                    let idx = nearest_centroid_4bit(scaled);
                    let byte_idx = c / 2;
                    if c % 2 == 0 {
                        v_packed[offset + byte_idx] = idx & 0xF;
                    } else {
                        v_packed[offset + byte_idx] |= (idx & 0xF) << 4;
                    }
                }
            }
        }
    }

    // Pre-rotate Q for the kernel
    let mut q_rotated = q_data.clone();
    for h in 0..nh {
        fwht_inplace(&mut q_rotated[h * hd..(h + 1) * hd]).unwrap();
    }

    // Allocate GPU buffers
    let mut q_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();
    q_buf.as_mut_slice::<f32>().unwrap()[..nh * hd].copy_from_slice(&q_rotated);

    let mut k_packed_buf = device.alloc_buffer(k_packed.len(), DType::U8, vec![nkv, kvl, hd / 2]).unwrap();
    k_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&k_packed);

    let mut k_norms_buf = device.alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl]).unwrap();
    k_norms_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&k_norms);

    let mut v_packed_buf = device.alloc_buffer(v_packed.len(), DType::U8, vec![nkv, kvl, hd / 2]).unwrap();
    v_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&v_packed);

    let mut v_norms_buf = device.alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl]).unwrap();
    v_norms_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&v_norms);

    let output_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();
    let tmp_bytes = flash_attn_vec_tq::tmp_buffer_bytes(num_heads, head_dim);
    let tmp_buf = device.alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4]).unwrap();

    let params = FlashAttnVecTqParams {
        num_heads, num_kv_heads, head_dim, kv_seq_len,
        kv_capacity: kv_seq_len,
        scale: 1.0,
        mask_type: 1,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
    };

    // Set NWG override if requested
    if let Some(nwg) = nwg_override {
        std::env::set_var("HF2Q_TQ_NWG", nwg.to_string());
    }

    // Warmup
    for _ in 0..warmup {
        let mut encoder = device.command_encoder().unwrap();
        flash_attn_vec_tq::flash_attn_vec_tq(
            &mut encoder, registry, device,
            &q_buf, &k_packed_buf, &k_norms_buf,
            &v_packed_buf, &v_norms_buf, &output_buf, &tmp_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iters {
        let mut encoder = device.command_encoder().unwrap();
        flash_attn_vec_tq::flash_attn_vec_tq(
            &mut encoder, registry, device,
            &q_buf, &k_packed_buf, &k_norms_buf,
            &v_packed_buf, &v_norms_buf, &output_buf, &tmp_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
    }
    let elapsed = start.elapsed();

    if nwg_override.is_some() {
        std::env::remove_var("HF2Q_TQ_NWG");
    }

    elapsed.as_secs_f64() * 1e6 / iters as f64 // microseconds per call
}

// ---- F16 SDPA bench ----

fn bench_f16_sdpa(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let mut rng = Xoshiro256::new(42);
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let hd = head_dim as usize;
    let kvl = kv_seq_len as usize;

    let q_data = random_f32_vec(&mut rng, nh * hd);
    let k_data = random_f32_vec(&mut rng, nkv * kvl * hd);
    let v_data = random_f32_vec(&mut rng, nkv * kvl * hd);

    let mut q_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();
    q_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&q_data);

    let mut k_buf = device.alloc_buffer(nkv * kvl * hd * 4, DType::F32, vec![nkv, kvl, hd]).unwrap();
    k_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&k_data);

    let mut v_buf = device.alloc_buffer(nkv * kvl * hd * 4, DType::F32, vec![nkv, kvl, hd]).unwrap();
    v_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&v_data);

    let output_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();
    let tmp_bytes = flash_attn_vec::tmp_buffer_bytes(num_heads, head_dim);
    let tmp_buf = device.alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4]).unwrap();

    let params = FlashAttnVecParams {
        num_heads, num_kv_heads, head_dim, kv_seq_len,
        kv_capacity: kv_seq_len,
        scale: 1.0 / (head_dim as f32).sqrt(),
        mask_type: 1,
        sliding_window: 0,
        softcap: 0.0,
    };

    // Warmup
    for _ in 0..warmup {
        let mut encoder = device.command_encoder().unwrap();
        flash_attn_vec::flash_attn_vec(
            &mut encoder, registry, device,
            &q_buf, &k_buf, &v_buf, &output_buf, &tmp_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Timed
    let start = Instant::now();
    for _ in 0..iters {
        let mut encoder = device.command_encoder().unwrap();
        flash_attn_vec::flash_attn_vec(
            &mut encoder, registry, device,
            &q_buf, &k_buf, &v_buf, &output_buf, &tmp_buf, &params,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1e6 / iters as f64
}

// ---- Test entry points ----

#[test]
fn bench_sdpa_sliding_layer() {
    let (device, mut registry) = setup();

    // Gemma-4-27B sliding layer: 16 heads, 8 KV heads, head_dim=256
    let nh = 16u32;
    let nkv = 8u32;
    let hd = 256u32;
    let warmup = 5;
    let iters = 20;

    eprintln!("\n=== SDPA Microbench: Sliding Layer (nh={}, nkv={}, hd={}) ===", nh, nkv, hd);
    eprintln!("{:>8} {:>12} {:>12} {:>8}", "kv_len", "TQ SDPA", "F16 SDPA", "ratio");
    eprintln!("{:>8} {:>12} {:>12} {:>8}", "", "(μs)", "(μs)", "");

    for &kvl in &[32u32, 64, 128, 256, 512, 1024] {
        let tq_us = bench_tq_sdpa(&device, &mut registry, nh, nkv, hd, kvl, None, warmup, iters);
        let f16_us = bench_f16_sdpa(&device, &mut registry, nh, nkv, hd, kvl, warmup, iters);
        let ratio = tq_us / f16_us;
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>8.2}x", kvl, tq_us, f16_us, ratio);
    }
}

#[test]
fn bench_sdpa_global_layer() {
    let (device, mut registry) = setup();

    // Gemma-4-27B global layer: 16 heads, 2 KV heads, head_dim=512
    let nh = 16u32;
    let nkv = 2u32;
    let hd = 512u32;
    let warmup = 5;
    let iters = 20;

    eprintln!("\n=== SDPA Microbench: Global Layer (nh={}, nkv={}, hd={}) ===", nh, nkv, hd);
    eprintln!("{:>8} {:>12} {:>12} {:>8}", "kv_len", "TQ SDPA", "F16 SDPA", "ratio");
    eprintln!("{:>8} {:>12} {:>12} {:>8}", "", "(μs)", "(μs)", "");

    for &kvl in &[32u32, 64, 128, 256, 512, 1024] {
        let tq_us = bench_tq_sdpa(&device, &mut registry, nh, nkv, hd, kvl, None, warmup, iters);
        let f16_us = bench_f16_sdpa(&device, &mut registry, nh, nkv, hd, kvl, warmup, iters);
        let ratio = tq_us / f16_us;
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>8.2}x", kvl, tq_us, f16_us, ratio);
    }
}

#[test]
fn bench_tq_nwg_sweep() {
    let (device, mut registry) = setup();

    // Sliding layer at 1024 KV positions — the most common decode shape
    let nh = 16u32;
    let nkv = 8u32;
    let hd = 256u32;
    let kvl = 1024u32;
    let warmup = 5;
    let iters = 20;

    eprintln!("\n=== TQ SDPA NWG Sweep: nh={}, nkv={}, hd={}, kvl={} ===", nh, nkv, hd, kvl);
    eprintln!("{:>6} {:>12}", "NWG", "μs/call");

    for &nwg in &[1u32, 2, 4, 8, 16, 32] {
        let us = bench_tq_sdpa(&device, &mut registry, nh, nkv, hd, kvl, Some(nwg), warmup, iters);
        eprintln!("{:>6} {:>12.1}", nwg, us);
    }
}

// ---- TQ v2 (tiled dequant) bench ----

/// GPU params — must match FlashAttnVecTqParams in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
    nwg: u32,
}

fn bench_tq_v2_sdpa(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    nwg: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    let mut rng = Xoshiro256::new(42);
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let hd = head_dim as usize;
    let kvl = kv_seq_len as usize;

    // Same data setup as v1 bench
    let q_data = random_f32_vec(&mut rng, nh * hd);
    let mut k_packed = vec![0u8; nkv * kvl * (hd / 2)];
    let mut k_norms = vec![0.0f32; nkv * kvl];
    let mut v_packed = vec![0u8; nkv * kvl * (hd / 2)];
    let mut v_norms = vec![0.0f32; nkv * kvl];

    for kv_h in 0..nkv {
        for p in 0..kvl {
            let k_vec = random_f32_vec(&mut rng, hd);
            let mut rotated = k_vec.clone();
            fwht_inplace(&mut rotated).unwrap();
            let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
            k_norms[kv_h * kvl + p] = norm;
            if norm > 1e-30 {
                let inv_norm = 1.0 / norm;
                let scale = (hd as f32).sqrt();
                let offset = (kv_h * kvl + p) * (hd / 2);
                for c in 0..hd {
                    let scaled = rotated[c] * inv_norm * scale;
                    let idx = nearest_centroid_4bit(scaled);
                    if c % 2 == 0 { k_packed[offset + c / 2] = idx & 0xF; }
                    else { k_packed[offset + c / 2] |= (idx & 0xF) << 4; }
                }
            }
            let v_vec = random_f32_vec(&mut rng, hd);
            let mut v_rot = v_vec.clone();
            fwht_inplace(&mut v_rot).unwrap();
            let v_norm: f32 = v_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            v_norms[kv_h * kvl + p] = v_norm;
            if v_norm > 1e-30 {
                let inv_norm = 1.0 / v_norm;
                let scale = (hd as f32).sqrt();
                let offset = (kv_h * kvl + p) * (hd / 2);
                for c in 0..hd {
                    let scaled = v_rot[c] * inv_norm * scale;
                    let idx = nearest_centroid_4bit(scaled);
                    if c % 2 == 0 { v_packed[offset + c / 2] = idx & 0xF; }
                    else { v_packed[offset + c / 2] |= (idx & 0xF) << 4; }
                }
            }
        }
    }

    let mut q_rotated = q_data.clone();
    for h in 0..nh {
        fwht_inplace(&mut q_rotated[h * hd..(h + 1) * hd]).unwrap();
    }

    let mut q_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();
    q_buf.as_mut_slice::<f32>().unwrap()[..nh * hd].copy_from_slice(&q_rotated);
    let mut k_packed_buf = device.alloc_buffer(k_packed.len(), DType::U8, vec![nkv, kvl, hd / 2]).unwrap();
    k_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&k_packed);
    let mut k_norms_buf = device.alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl]).unwrap();
    k_norms_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&k_norms);
    let mut v_packed_buf = device.alloc_buffer(v_packed.len(), DType::U8, vec![nkv, kvl, hd / 2]).unwrap();
    v_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&v_packed);
    let mut v_norms_buf = device.alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv, kvl]).unwrap();
    v_norms_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&v_norms);

    let output_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).unwrap();

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_tq_v2_dk256",
        512 => "flash_attn_vec_tq_v2_dk512",
        _ => panic!("unsupported head_dim"),
    };

    let gpu_params = TqParamsGpu {
        n_heads: num_heads, n_kv_heads: num_kv_heads,
        head_dim, kv_seq_len, kv_capacity: kv_seq_len,
        scale: 1.0, mask_type: 1, sliding_window: 0, softcap: 0.0,
        nwg,
    };

    // CT (chunk tile size) determines shared memory:
    let ct: usize = if head_dim == 512 { 16 } else { 32 };
    let dk4 = head_dim as usize / 4;
    let dv4 = dk4; // DK == DV for Gemma
    let pk = ((head_dim as usize) + 127) & !127; // PAD2(hd, 128)
    let pv = pk;
    let sh = 4 * ct; // CT * 4 halfs for scores
    let tile_size = ct * dk4.max(dv4); // tile_buf in half4
    let shmem_halfs = pk + sh + 2 * pv + tile_size * 4; // tile_buf is half4 = 4 halfs each
    let shmem_bytes = shmem_halfs * 2;

    // Warmup
    for _ in 0..warmup {
        let mut encoder = device.command_encoder().unwrap();
        let pipeline = registry.get_pipeline(kernel_name, device.metal_device()).unwrap();
        let threadgroups = MTLSize::new(1, num_heads as u64, nwg as u64);
        let threadgroup_size = MTLSize::new(32, 1, 1);
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                (1, KernelArg::Buffer(&q_buf)),
                (2, KernelArg::Buffer(&k_packed_buf)),
                (3, KernelArg::Buffer(&k_norms_buf)),
                (4, KernelArg::Buffer(&v_packed_buf)),
                (5, KernelArg::Buffer(&v_norms_buf)),
                (6, KernelArg::Buffer(&output_buf)),
            ],
            &[(0, shmem_bytes as u64)],
            threadgroups,
            threadgroup_size,
        );
        encoder.commit_and_wait().unwrap();
    }

    // Timed
    let start = Instant::now();
    for _ in 0..iters {
        let mut encoder = device.command_encoder().unwrap();
        let pipeline = registry.get_pipeline(kernel_name, device.metal_device()).unwrap();
        let threadgroups = MTLSize::new(1, num_heads as u64, nwg as u64);
        let threadgroup_size = MTLSize::new(32, 1, 1);
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                (1, KernelArg::Buffer(&q_buf)),
                (2, KernelArg::Buffer(&k_packed_buf)),
                (3, KernelArg::Buffer(&k_norms_buf)),
                (4, KernelArg::Buffer(&v_packed_buf)),
                (5, KernelArg::Buffer(&v_norms_buf)),
                (6, KernelArg::Buffer(&output_buf)),
            ],
            &[(0, shmem_bytes as u64)],
            threadgroups,
            threadgroup_size,
        );
        encoder.commit_and_wait().unwrap();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1e6 / iters as f64
}

#[test]
fn bench_tq_v1_vs_v2() {
    let (device, mut registry) = setup();

    let warmup = 5;
    let iters = 30;

    eprintln!("\n=== TQ SDPA v1 vs v2: Sliding (nh=16, nkv=8, hd=256) ===");
    eprintln!("{:>8} {:>12} {:>12} {:>12} {:>8}", "kv_len", "v1 (μs)", "v2 (μs)", "F16 (μs)", "v2/F16");

    for &kvl in &[32u32, 128, 512, 1024] {
        let v1 = bench_tq_sdpa(&device, &mut registry, 16, 8, 256, kvl, Some(16), warmup, iters);
        let v2 = bench_tq_v2_sdpa(&device, &mut registry, 16, 8, 256, kvl, 16, warmup, iters);
        let f16 = bench_f16_sdpa(&device, &mut registry, 16, 8, 256, kvl, warmup, iters);
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>12.1} {:>8.2}x", kvl, v1, v2, f16, v2 / f16);
    }

    eprintln!("\n=== TQ SDPA v1 vs v2: Global (nh=16, nkv=2, hd=512) ===");
    eprintln!("{:>8} {:>12} {:>12} {:>12} {:>8}", "kv_len", "v1 (μs)", "v2 (μs)", "F16 (μs)", "v2/F16");

    for &kvl in &[32u32, 128, 512, 1024] {
        let v1 = bench_tq_sdpa(&device, &mut registry, 16, 2, 512, kvl, Some(16), warmup, iters);
        let v2 = bench_tq_v2_sdpa(&device, &mut registry, 16, 2, 512, kvl, 16, warmup, iters);
        let f16 = bench_f16_sdpa(&device, &mut registry, 16, 2, 512, kvl, warmup, iters);
        eprintln!("{:>8} {:>12.1} {:>12.1} {:>12.1} {:>8.2}x", kvl, v1, v2, f16, v2 / f16);
    }
}
