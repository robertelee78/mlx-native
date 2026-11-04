//! Shared seeded data for the ADR-063 Q4-GEMMA-DECODE expert parity tests
//! (`gemma_expert_grouped_mv_id_bitwise`, `gemma_gelu_epilogue_bitwise`):
//! GGUF blocks with zero and subnormal half scales, activations with ±0,
//! subnormals and large values, routing patterns, and guarded views.

#![allow(dead_code)]

use mlx_native::{DType, GgmlQuantizedMatmulIdParams, GgmlType, MlxBuffer, MlxDevice};

pub const SENTINEL: u32 = 0xDEAD_BEEF;
/// Leading/trailing guard elements around every view (256 bytes).
pub const GUARD: usize = 64;
pub const INVALID_MAX: u32 = u32::MAX;

// ---- deterministic data ----

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Half scale bits: mostly small normal values of either sign, with zero,
/// negative zero and subnormal halves mixed in.
pub fn half_scale_bits(rng: &mut Rng, block: usize) -> u16 {
    match block % 29 {
        0 => 0x0000,
        1 => 0x8000,
        2 => 1 + rng.below(0x3ff) as u16,
        3 => 0x8000 | (1 + rng.below(0x3ff) as u16),
        _ => {
            let magnitude = 0.000_5 + rng.unit() * 0.004;
            let value = if rng.below(5) == 0 { -magnitude } else { magnitude };
            half::f16::from_f32(value).to_bits()
        }
    }
}

pub fn block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_K => 144,
        GgmlType::Q5_0 => 22,
        GgmlType::Q8_0 => 34,
        other => panic!("unsupported test type {other:?}"),
    }
}

pub fn block_values(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_K => 256,
        GgmlType::Q5_0 | GgmlType::Q8_0 => 32,
        other => panic!("unsupported test type {other:?}"),
    }
}

pub fn fill_block(ty: GgmlType, rng: &mut Rng, index: usize, out: &mut [u8]) {
    let d = half_scale_bits(rng, index).to_le_bytes();
    out[0..2].copy_from_slice(&d);
    let start = match ty {
        GgmlType::Q4_K => {
            let dmin = half_scale_bits(rng, index / 3 + 7).to_le_bytes();
            out[2..4].copy_from_slice(&dmin);
            4
        }
        _ => 2,
    };
    for chunk in out[start..].chunks_mut(8) {
        let bytes = rng.next().to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

/// Weight view at a non-zero offset into a parent buffer. One padding expert
/// follows the last one because the stock kernels read whole row tiles past
/// a row tail.
pub struct Weights {
    pub _parent: MlxBuffer,
    pub view: MlxBuffer,
    pub ty: GgmlType,
    pub n: usize,
    pub k: usize,
    pub n_experts: usize,
    pub stride: usize,
}

pub fn make_weights(
    device: &MlxDevice,
    ty: GgmlType,
    n: usize,
    k: usize,
    n_experts: usize,
    seed: u64,
) -> Weights {
    let block = block_bytes(ty);
    let blocks_per_expert = n * (k / block_values(ty));
    let stride = blocks_per_expert * block;
    let lead = 4 * GUARD;
    let total = lead + (n_experts + 1) * stride;
    let mut parent = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("weight buffer");
    {
        let bytes = parent.as_mut_slice::<u8>().expect("weight slice");
        let mut rng = Rng(seed);
        bytes[..lead].fill(0xA5);
        for (index, out) in bytes[lead..].chunks_mut(block).enumerate() {
            fill_block(ty, &mut rng, index, out);
        }
    }
    let view = parent.slice_view(lead as u64, n_experts * stride);
    Weights {
        _parent: parent,
        view,
        ty,
        n,
        k,
        n_experts,
        stride,
    }
}

pub fn activation(rng: &mut Rng) -> f32 {
    match rng.below(256) {
        0 => 0.0,
        1 => -0.0,
        2 => {
            let bits = 1 + rng.below(0x007f_ffff) as u32;
            f32::from_bits(bits | if rng.below(2) == 0 { 0 } else { 0x8000_0000 })
        }
        3 => {
            let value = 1.0e3 + rng.unit() * 3.0e4;
            if rng.below(2) == 0 {
                value
            } else {
                -value
            }
        }
        _ => rng.unit() * 2.0 - 1.0,
    }
}

pub fn make_input(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * k).map(|_| activation(&mut rng)).collect()
}

pub fn poison_row(input: &mut [f32], row: usize, k: usize) {
    input[row * k + 3] = f32::NAN;
    input[row * k + k / 2] = f32::INFINITY;
    input[row * k + k - 1] = f32::NEG_INFINITY;
}

// ---- routing patterns ----

#[derive(Clone, Copy, Debug)]
pub enum Pattern {
    /// Every pair a different expert.
    Distinct,
    /// Every token routes to the same experts (one expert at top-1).
    AllOne,
    /// Uniform random, distinct within each token.
    Random,
    /// Random from a small pool: heavy sharing.
    Pooled,
    /// One hot expert taken by (almost) every token.
    Skewed,
    InvalidFirst,
    InvalidMiddle,
    InvalidLast,
    /// Invalid ids at first, middle and last inside one big group.
    InvalidInGroup,
}

pub const PATTERNS: [Pattern; 9] = [
    Pattern::Distinct,
    Pattern::AllOne,
    Pattern::Random,
    Pattern::Pooled,
    Pattern::Skewed,
    Pattern::InvalidFirst,
    Pattern::InvalidMiddle,
    Pattern::InvalidLast,
    Pattern::InvalidInGroup,
];

pub fn pick_distinct(rng: &mut Rng, pool: &[u32], count: usize) -> Vec<u32> {
    let mut pool = pool.to_vec();
    let mut picked = Vec::with_capacity(count);
    for _ in 0..count {
        let index = rng.below(pool.len() as u64) as usize;
        picked.push(pool.swap_remove(index));
    }
    picked
}

pub fn make_ids(pattern: Pattern, n_tokens: usize, top_k: usize, n_experts: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    let pairs = n_tokens * top_k;
    let all: Vec<u32> = (0..n_experts as u32).collect();
    let random = |rng: &mut Rng, pool: &[u32]| -> Vec<u32> {
        (0..n_tokens)
            .flat_map(|_| pick_distinct(rng, pool, top_k))
            .collect()
    };
    let pooled_pool: Vec<u32> = (0..(top_k + 3).min(n_experts) as u32)
        .map(|i| (i * 7 + 3) % n_experts as u32)
        .collect();
    let all_one: Vec<u32> = (0..pairs)
        .map(|p| ((p % top_k) as u32 * 5 + 2) % n_experts as u32)
        .collect();
    let mut ids = match pattern {
        Pattern::Distinct => {
            assert!(pairs <= n_experts, "distinct needs pairs <= experts");
            (0..pairs).map(|p| ((p * 37 + 11) % n_experts) as u32).collect()
        }
        Pattern::AllOne | Pattern::InvalidInGroup => all_one,
        Pattern::Random | Pattern::InvalidFirst | Pattern::InvalidMiddle => random(&mut rng, &all),
        Pattern::Pooled | Pattern::InvalidLast => random(&mut rng, &pooled_pool),
        Pattern::Skewed => {
            let hot = 7 % n_experts as u32;
            let rest: Vec<u32> = all.iter().copied().filter(|&e| e != hot).collect();
            let mut ids = Vec::with_capacity(pairs);
            for _ in 0..n_tokens {
                if top_k == 1 {
                    ids.push(if rng.below(4) != 0 {
                        hot
                    } else {
                        rest[rng.below(rest.len() as u64) as usize]
                    });
                } else {
                    ids.push(hot);
                    ids.extend(pick_distinct(&mut rng, &rest, top_k - 1));
                }
            }
            ids
        }
    };
    match pattern {
        Pattern::InvalidFirst => ids[0] = INVALID_MAX,
        Pattern::InvalidMiddle => ids[pairs / 2] = n_experts as u32,
        Pattern::InvalidLast => ids[pairs - 1] = INVALID_MAX,
        Pattern::InvalidInGroup => {
            ids[0] = INVALID_MAX;
            ids[pairs / 2] = n_experts as u32;
            ids[pairs - 1] = INVALID_MAX;
        }
        _ => {}
    }
    ids
}

pub fn guarded_f32(device: &MlxDevice, data: &[f32]) -> (MlxBuffer, MlxBuffer) {
    let total = GUARD + data.len() + GUARD;
    let mut parent = device
        .alloc_buffer(total * 4, DType::F32, vec![total])
        .expect("guarded f32");
    {
        let slice = parent.as_mut_slice::<f32>().expect("guarded slice");
        slice.fill(f32::from_bits(SENTINEL));
        slice[GUARD..GUARD + data.len()].copy_from_slice(data);
    }
    let view = parent.slice_view((GUARD * 4) as u64, data.len());
    (parent, view)
}

pub fn guarded_u32(device: &MlxDevice, data: &[u32]) -> (MlxBuffer, MlxBuffer) {
    let total = GUARD + data.len() + GUARD;
    let mut parent = device
        .alloc_buffer(total * 4, DType::U32, vec![total])
        .expect("guarded u32");
    {
        let slice = parent.as_mut_slice::<u32>().expect("guarded slice");
        slice.fill(SENTINEL);
        slice[GUARD..GUARD + data.len()].copy_from_slice(data);
    }
    let view = parent.slice_view((GUARD * 4) as u64, data.len());
    (parent, view)
}

pub fn fnv(digest: &mut u64, words: &[u32]) {
    for word in words {
        for byte in word.to_le_bytes() {
            *digest ^= u64::from(byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

pub fn params(w: &Weights, n_tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: w.n as u32,
        k: w.k as u32,
        n_experts: w.n_experts as u32,
        expert_stride: w.stride as u64,
        ggml_type: w.ty,
    }
}

pub fn assert_stock_frame(stock: &[u32], pairs: usize, n: usize, label: &str) {
    assert!(
        stock[..GUARD].iter().all(|&w| w == SENTINEL)
            && stock[GUARD + pairs * n..].iter().all(|&w| w == SENTINEL),
        "{label}: stock wrote outside the output view"
    );
    assert!(
        stock[GUARD..GUARD + pairs * n].iter().all(|&w| w != SENTINEL),
        "{label}: stock left an output element unwritten"
    );
}

pub fn assert_bitwise(stock: &[u32], got: &[u32], n: usize, ids: &[u32], label: &str) {
    assert_eq!(stock.len(), got.len());
    if let Some(index) = (0..stock.len()).find(|&i| stock[i] != got[i]) {
        if index < GUARD || index >= stock.len() - GUARD {
            panic!(
                "{label}: guard element {index} changed: stock {:#010x}, grouped {:#010x}",
                stock[index], got[index]
            );
        }
        let flat = index - GUARD;
        let pair = flat / n;
        let mismatches = (0..stock.len()).filter(|&i| stock[i] != got[i]).count();
        // Mismatches by output column modulo 8 (the largest row tile).
        let mut by_column_phase = [0usize; 8];
        for i in (GUARD..stock.len() - GUARD).filter(|&i| stock[i] != got[i]) {
            by_column_phase[((i - GUARD) % n) % 8] += 1;
        }
        panic!(
            "{label}: {mismatches} mismatches (by column % 8: {by_column_phase:?}); first pair={pair} (id {:#x}) col={}: stock {:#010x} ({}), grouped {:#010x} ({})",
            ids[pair],
            flat % n,
            stock[index],
            f32::from_bits(stock[index]),
            got[index],
            f32::from_bits(got[index]),
        );
    }
}
