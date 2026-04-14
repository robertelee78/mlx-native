//! Correctness tests for the `hadamard_quantize_kv` GPU kernel (ADR-007 Phase 1.1).
//!
//! Validates GPU output against the CPU reference in `turboquant.rs`:
//!   - Nibble-packed 4-bit centroid indices must match exactly.
//!   - L2 norm scalars must match within ε < 1e-4.
//!
//! Tests:
//!   a) head_dim=256, num_kv_heads=2, global cache (non-sliding)
//!   b) head_dim=512, num_kv_heads=8, sliding window cache
//!   c) Sliding modulo: write_pos >= cache_capacity wraps correctly
//!   d) Argument validation: non-power-of-two head_dim returns an error

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::hadamard_quantize_kv;
use mlx_native::turboquant::{fwht_inplace, CODEBOOK_4BIT};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---------------------------------------------------------------------------
// CPU reference implementation matching the GPU kernel exactly
// ---------------------------------------------------------------------------

/// Precomputed decision boundaries for CODEBOOK_4BIT.
/// boundaries[i] = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i+1]) / 2
const BOUNDARIES_4BIT: [f32; 15] = [
    -2.4008034,
    -1.8435318,
    -1.4371388,
    -1.0992859,
    -0.7995498,
    -0.5224037,
    -0.2582217,
    0.0,
    0.2582217,
    0.5224037,
    0.7995498,
    1.0992859,
    1.4371388,
    1.8435318,
    2.4008034,
];

/// Find the nearest 4-bit Lloyd-Max centroid index for a value.
fn nearest_centroid_4bit(value: f32) -> u8 {
    let mut idx = 0u8;
    for (b, &boundary) in BOUNDARIES_4BIT.iter().enumerate() {
        if value > boundary {
            idx = (b + 1) as u8;
        }
    }
    idx
}

/// CPU reference for one KV head vector.
///
/// Returns `(packed_nibbles, norm)` where `packed_nibbles` is a `Vec<u8>` of
/// length `head_dim / 2` with the same nibble layout as the GPU kernel:
///   byte[i] = (index[2*i+1] << 4) | index[2*i]
fn cpu_hadamard_quantize_head(src: &[f32]) -> (Vec<u8>, f32) {
    let d = src.len();
    assert!(d.is_power_of_two() && d >= 2);

    // Step 1+2: FWHT then normalize by 1/sqrt(d).
    let mut rotated = src.to_vec();
    fwht_inplace(&mut rotated).unwrap();
    // fwht_inplace already normalizes by 1/sqrt(d) in turboquant.rs.

    // Step 3: Compute L2 norm of rotated vector.
    let norm_sq: f32 = rotated.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt();

    // Step 4+5: Normalize to unit vector then scale to N(0,1) domain.
    let scale = (d as f32).sqrt();
    let indices: Vec<u8> = if norm > 1.0e-10 {
        rotated
            .iter()
            .map(|&v| {
                let unit_val = v / norm;
                let scaled = unit_val * scale;
                nearest_centroid_4bit(scaled)
            })
            .collect()
    } else {
        vec![0u8; d]
    };

    // Step 6: Nibble pack — even index in low nibble, odd index in high nibble.
    let packed: Vec<u8> = (0..d / 2)
        .map(|i| (indices[2 * i] & 0xF) | ((indices[2 * i + 1] & 0xF) << 4))
        .collect();

    (packed, norm)
}

// ---------------------------------------------------------------------------
// xoshiro256** PRNG (same as test_turboquant.rs)
// ---------------------------------------------------------------------------

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

fn random_f32_vec(rng: &mut Xoshiro256, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let (a, b) = randn_pair(rng);
        v.push(a as f32);
        if v.len() < n {
            v.push(b as f32);
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    hadamard_quantize_kv::register(&mut registry);
    (device, registry)
}

/// Run the GPU kernel for a single `write_pos` and return `(packed_bytes, norms_vec)`.
///
/// `src_flat`: `[num_kv_heads * head_dim]` f32.
/// Returns packed bytes of length `num_kv_heads * cache_capacity * head_dim/2`
/// and norms of length `num_kv_heads * cache_capacity`.
fn run_gpu_quantize(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src_flat: &[f32],
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
) -> (Vec<u8>, Vec<f32>) {
    let src_byte_len = src_flat.len() * 4;
    let packed_byte_len =
        (num_kv_heads as usize) * (cache_capacity as usize) * (head_dim as usize / 2);
    let norms_elem = (num_kv_heads as usize) * (cache_capacity as usize);
    let norms_byte_len = norms_elem * 4;

    // Allocate and fill src buffer.
    let mut src_buf = device
        .alloc_buffer(src_byte_len, DType::F32, vec![src_flat.len()])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(src_flat);

    // Allocate packed buffer (u8), zeroed.
    let mut packed_buf = device
        .alloc_buffer(packed_byte_len, DType::U8, vec![packed_byte_len])
        .expect("alloc packed");
    for b in packed_buf.as_mut_slice::<u8>().expect("zero packed").iter_mut() {
        *b = 0;
    }

    // Allocate norms buffer (f32), zeroed.
    let mut norms_buf = device
        .alloc_buffer(norms_byte_len, DType::F32, vec![norms_elem])
        .expect("alloc norms");
    for v in norms_buf.as_mut_slice::<f32>().expect("zero norms").iter_mut() {
        *v = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
        &mut encoder,
        registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        num_kv_heads,
        head_dim,
        cache_capacity,
        write_pos,
        is_sliding,
    )
    .expect("dispatch_hadamard_quantize_kv");
    encoder.commit_and_wait().expect("commit_and_wait");

    let packed_out: Vec<u8> = packed_buf.as_slice::<u8>().expect("read packed").to_vec();
    let norms_out: Vec<f32> = norms_buf.as_slice::<f32>().expect("read norms").to_vec();
    (packed_out, norms_out)
}

// ---------------------------------------------------------------------------
// Test (a): head_dim=256, num_kv_heads=2, global cache
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_d256_heads2_global() {
    let (device, mut registry) = setup();
    let mut rng = Xoshiro256::new(1234);

    let head_dim: u32 = 256;
    let num_kv_heads: u32 = 2;
    let cache_capacity: u32 = 32;
    let write_pos: u32 = 7;

    let src_flat = random_f32_vec(&mut rng, (num_kv_heads * head_dim) as usize);

    let (gpu_packed, gpu_norms) = run_gpu_quantize(
        &device,
        &mut registry,
        &src_flat,
        num_kv_heads,
        head_dim,
        cache_capacity,
        write_pos,
        false, // global
    );

    // CPU reference — one head at a time.
    for h in 0..num_kv_heads as usize {
        let head_src = &src_flat[h * head_dim as usize..(h + 1) * head_dim as usize];
        let (cpu_packed, cpu_norm) = cpu_hadamard_quantize_head(head_src);

        let packed_stride = head_dim as usize / 2;
        let gpu_packed_offset = h * (cache_capacity as usize) * packed_stride
            + (write_pos as usize) * packed_stride;
        let gpu_packed_slice = &gpu_packed[gpu_packed_offset..gpu_packed_offset + packed_stride];

        // Check each packed byte (two nibbles = two 4-bit indices).
        for (byte_i, (&gpu_b, &cpu_b)) in
            gpu_packed_slice.iter().zip(cpu_packed.iter()).enumerate()
        {
            let gpu_lo = gpu_b & 0xF;
            let gpu_hi = (gpu_b >> 4) & 0xF;
            let cpu_lo = cpu_b & 0xF;
            let cpu_hi = (cpu_b >> 4) & 0xF;
            assert_eq!(
                gpu_lo, cpu_lo,
                "d256 heads2 global: head={h} byte={byte_i} \
                 low-nibble mismatch: gpu={gpu_lo} cpu={cpu_lo}"
            );
            assert_eq!(
                gpu_hi, cpu_hi,
                "d256 heads2 global: head={h} byte={byte_i} \
                 high-nibble mismatch: gpu={gpu_hi} cpu={cpu_hi}"
            );
        }

        // Check norm.
        let norm_offset = h * cache_capacity as usize + write_pos as usize;
        let gpu_norm = gpu_norms[norm_offset];
        let norm_err = (gpu_norm - cpu_norm).abs();
        assert!(
            norm_err < 1.0e-4,
            "d256 heads2 global: head={h} norm mismatch: gpu={gpu_norm} cpu={cpu_norm} \
             err={norm_err}"
        );
    }

    println!("PASS test_quantize_d256_heads2_global");
}

// ---------------------------------------------------------------------------
// Test (b): head_dim=512, num_kv_heads=8, sliding window
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_d512_heads8_sliding() {
    let (device, mut registry) = setup();
    let mut rng = Xoshiro256::new(5678);

    let head_dim: u32 = 512;
    let num_kv_heads: u32 = 8;
    let cache_capacity: u32 = 64;
    let write_pos: u32 = 13;

    let src_flat = random_f32_vec(&mut rng, (num_kv_heads * head_dim) as usize);

    let (gpu_packed, gpu_norms) = run_gpu_quantize(
        &device,
        &mut registry,
        &src_flat,
        num_kv_heads,
        head_dim,
        cache_capacity,
        write_pos,
        true, // sliding
    );

    let actual_pos = write_pos % cache_capacity;

    for h in 0..num_kv_heads as usize {
        let head_src = &src_flat[h * head_dim as usize..(h + 1) * head_dim as usize];
        let (cpu_packed, cpu_norm) = cpu_hadamard_quantize_head(head_src);

        let packed_stride = head_dim as usize / 2;
        let gpu_packed_offset = h * (cache_capacity as usize) * packed_stride
            + (actual_pos as usize) * packed_stride;
        let gpu_packed_slice = &gpu_packed[gpu_packed_offset..gpu_packed_offset + packed_stride];

        for (byte_i, (&gpu_b, &cpu_b)) in
            gpu_packed_slice.iter().zip(cpu_packed.iter()).enumerate()
        {
            let gpu_lo = gpu_b & 0xF;
            let gpu_hi = (gpu_b >> 4) & 0xF;
            let cpu_lo = cpu_b & 0xF;
            let cpu_hi = (cpu_b >> 4) & 0xF;
            assert_eq!(
                gpu_lo, cpu_lo,
                "d512 heads8 sliding: head={h} byte={byte_i} \
                 low-nibble mismatch: gpu={gpu_lo} cpu={cpu_lo}"
            );
            assert_eq!(
                gpu_hi, cpu_hi,
                "d512 heads8 sliding: head={h} byte={byte_i} \
                 high-nibble mismatch: gpu={gpu_hi} cpu={cpu_hi}"
            );
        }

        let norm_offset = h * cache_capacity as usize + actual_pos as usize;
        let gpu_norm = gpu_norms[norm_offset];
        let norm_err = (gpu_norm - cpu_norm).abs();
        assert!(
            norm_err < 1.0e-4,
            "d512 heads8 sliding: head={h} norm mismatch: gpu={gpu_norm} cpu={cpu_norm} \
             err={norm_err}"
        );
    }

    println!("PASS test_quantize_d512_heads8_sliding");
}

// ---------------------------------------------------------------------------
// Test (c): Sliding modulo — write_pos >= cache_capacity wraps correctly
// ---------------------------------------------------------------------------

#[test]
fn test_sliding_modulo_wrap() {
    let (device, mut registry) = setup();
    let mut rng = Xoshiro256::new(9999);

    let head_dim: u32 = 256;
    let num_kv_heads: u32 = 2;
    let cache_capacity: u32 = 8;
    // write_pos=10 should wrap to position 10 % 8 = 2
    let write_pos: u32 = 10;
    let expected_actual_pos = write_pos % cache_capacity;

    let src_flat = random_f32_vec(&mut rng, (num_kv_heads * head_dim) as usize);

    let (gpu_packed, gpu_norms) = run_gpu_quantize(
        &device,
        &mut registry,
        &src_flat,
        num_kv_heads,
        head_dim,
        cache_capacity,
        write_pos,
        true, // sliding
    );

    for h in 0..num_kv_heads as usize {
        let head_src = &src_flat[h * head_dim as usize..(h + 1) * head_dim as usize];
        let (cpu_packed, cpu_norm) = cpu_hadamard_quantize_head(head_src);

        let packed_stride = head_dim as usize / 2;
        let gpu_packed_offset = h * (cache_capacity as usize) * packed_stride
            + (expected_actual_pos as usize) * packed_stride;
        let gpu_packed_slice = &gpu_packed[gpu_packed_offset..gpu_packed_offset + packed_stride];

        for (byte_i, (&gpu_b, &cpu_b)) in
            gpu_packed_slice.iter().zip(cpu_packed.iter()).enumerate()
        {
            assert_eq!(
                gpu_b & 0xF,
                cpu_b & 0xF,
                "sliding wrap: head={h} byte={byte_i} low-nibble mismatch"
            );
            assert_eq!(
                (gpu_b >> 4) & 0xF,
                (cpu_b >> 4) & 0xF,
                "sliding wrap: head={h} byte={byte_i} high-nibble mismatch"
            );
        }

        let norm_offset = h * cache_capacity as usize + expected_actual_pos as usize;
        let gpu_norm = gpu_norms[norm_offset];
        let norm_err = (gpu_norm - cpu_norm).abs();
        assert!(
            norm_err < 1.0e-4,
            "sliding wrap: head={h} norm mismatch: gpu={gpu_norm} cpu={cpu_norm} err={norm_err}"
        );
    }

    println!("PASS test_sliding_modulo_wrap");
}

// ---------------------------------------------------------------------------
// Test (d): Argument validation errors
// ---------------------------------------------------------------------------

#[test]
fn test_validation_non_power_of_two() {
    let (device, mut registry) = setup();
    let src_buf = device
        .alloc_buffer(100 * 4, DType::F32, vec![100])
        .expect("alloc src");
    let packed_buf = device
        .alloc_buffer(100, DType::U8, vec![100])
        .expect("alloc packed");
    let norms_buf = device
        .alloc_buffer(8 * 4, DType::F32, vec![8])
        .expect("alloc norms");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        2,    // num_kv_heads
        100,  // head_dim — not a power of two
        4,    // cache_capacity
        0,    // write_pos
        false,
    );
    assert!(
        result.is_err(),
        "Non-power-of-two head_dim should return an error"
    );
    println!("PASS test_validation_non_power_of_two");
}

#[test]
fn test_validation_global_out_of_bounds() {
    let (device, mut registry) = setup();
    let head_dim = 256u32;
    let num_heads = 1u32;
    let capacity = 4u32;

    let src_buf = device
        .alloc_buffer((num_heads * head_dim) as usize * 4, DType::F32, vec![(num_heads * head_dim) as usize])
        .expect("alloc src");
    let packed_buf = device
        .alloc_buffer((num_heads * capacity * head_dim / 2) as usize, DType::U8, vec![(num_heads * capacity * head_dim / 2) as usize])
        .expect("alloc packed");
    let norms_buf = device
        .alloc_buffer((num_heads * capacity) as usize * 4, DType::F32, vec![(num_heads * capacity) as usize])
        .expect("alloc norms");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        num_heads,
        head_dim,
        capacity,
        capacity, // write_pos == capacity — out of bounds for global
        false,
    );
    assert!(
        result.is_err(),
        "write_pos >= cache_capacity for global cache should return an error"
    );
    println!("PASS test_validation_global_out_of_bounds");
}

#[test]
fn test_noop_on_zero_heads() {
    let (device, mut registry) = setup();
    let src_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc src");
    let packed_buf = device
        .alloc_buffer(1, DType::U8, vec![1])
        .expect("alloc packed");
    let norms_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc norms");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        0,   // num_kv_heads = 0 → no-op
        256,
        8,
        0,
        false,
    );
    assert!(result.is_ok(), "num_kv_heads=0 should be a no-op, not an error");
    encoder.commit_and_wait().expect("commit_and_wait");
    println!("PASS test_noop_on_zero_heads");
}

// ---------------------------------------------------------------------------
// Test (e): Codebook boundary check — verify BOUNDARIES_4BIT are correct midpoints
// ---------------------------------------------------------------------------

#[test]
fn test_boundaries_are_midpoints() {
    for i in 0..15 {
        let expected = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
        let actual = BOUNDARIES_4BIT[i];
        let err = (expected - actual).abs();
        assert!(
            err < 1e-6,
            "BOUNDARIES_4BIT[{i}] = {actual} but midpoint = {expected}, err = {err}"
        );
    }
    println!("PASS test_boundaries_are_midpoints");
}
