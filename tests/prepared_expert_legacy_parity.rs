//! Hardware numerical gates for canonical legacy expert consumers.
//! Expected full-MM digests are frozen independently of the shared shader cores;
//! provenance is in fixtures/prepared-legacy-provenance.json. No environment is
//! changed. Each test owns its device, registry, sources, scratch and outputs.
#![cfg(target_vendor = "apple")]
#![allow(clippy::too_many_arguments, clippy::unwrap_used, clippy::expect_used)]

use mlx_native::ops::quantized_matmul_id_ggml::*;
use mlx_native::{
    prepare_quantized_matmul_id_ggml_grouped_m8_batch, CompletedGroupedM8Experts,
    DType, ExpertMmSchedule, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, GroupedM8Preparation, IdMmScratch,
    KernelRegistry, MlxBuffer, MlxDevice,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[path = "support/prepared_legacy_mv_goldens.rs"]
mod mv_goldens;

const GUARD: usize = 16;
const GUARD_WORD: u32 = 0xa55a_936c;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const BANKS: usize = 40;
const MAX_TOKENS: usize = 256;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

// Access every root using its stored scalar type; bitcasts never convert F32.
struct Guarded { root: MlxBuffer, capacity: usize }
impl Guarded {
    fn new(device: &MlxDevice, capacity: usize, dtype: DType) -> Self {
        assert!(matches!(dtype, DType::F32 | DType::U32));
        let words = capacity + 2 * GUARD;
        let root = device.alloc_buffer(words * 4, dtype, vec![words]).unwrap();
        let mut result = Self { root, capacity };
        result.reset(capacity, GUARD_WORD);
        result
    }
    fn words(&self) -> &[u32] {
        match self.root.dtype() {
            DType::F32 => bytemuck::cast_slice(self.root.as_slice::<f32>().unwrap()),
            DType::U32 => self.root.as_slice::<u32>().unwrap(),
            _ => unreachable!(),
        }
    }
    fn words_mut(&mut self) -> &mut [u32] {
        match self.root.dtype() {
            DType::F32 => bytemuck::cast_slice_mut(self.root.as_mut_slice::<f32>().unwrap()),
            DType::U32 => self.root.as_mut_slice::<u32>().unwrap(),
            _ => unreachable!(),
        }
    }
    fn reset(&mut self, words: usize, sentinel: u32) {
        assert!(words <= self.capacity);
        let all = self.words_mut();
        all.fill(GUARD_WORD);
        all[GUARD..GUARD + words].fill(sentinel);
    }
    fn view(&self, words: usize) -> MlxBuffer {
        assert!(words <= self.capacity);
        self.root.slice_view((GUARD * 4) as u64, words)
    }
    fn payload(&self, words: usize) -> &[u32] { &self.words()[GUARD..GUARD + words] }
    fn payload_mut(&mut self, words: usize) -> &mut [u32] { &mut self.words_mut()[GUARD..GUARD + words] }
    fn check(&self, words: usize) {
        let all = self.words();
        assert!(all[..GUARD].iter().chain(&all[GUARD + words..]).all(|v| *v == GUARD_WORD), "guard corruption");
    }
    fn digest(&self, words: usize) -> String {
        format!("{:x}", Sha256::digest(bytemuck::cast_slice::<u32, u8>(self.payload(words))))
    }
}

fn params(ty: GgmlType, n: usize, k: usize, experts: usize, tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams { n_tokens: tokens as u32, top_k: top_k as u32,
        n: n as u32, k: k as u32, n_experts: experts as u32,
        expert_stride: (n * (k / 256) * ty.block_bytes() as usize) as u64, ggml_type: ty }
}

struct Source { root: MlxBuffer, view: MlxBuffer, digest: [u8; 32] }
impl Source {
    fn new(device: &MlxDevice, p: &GgmlQuantizedMatmulIdParams, rng: &mut Rng, exact: bool) -> Self {
        let raw_bytes = p.expert_stride as usize * p.n_experts as usize;
        let root_bytes = raw_bytes + 2 * GUARD * 4;
        let mut root = device.alloc_buffer(root_bytes, DType::U8, vec![root_bytes]).unwrap();
        let all = root.as_mut_slice::<u8>().unwrap();
        all.fill(0xa7);
        for (index, block) in all[GUARD * 4..GUARD * 4 + raw_bytes]
            .chunks_exact_mut(p.ggml_type.block_bytes() as usize).enumerate()
        {
            // Frozen generator: even the apparently redundant final draw when
            // the block's tail is empty is required. Do not change the stream.
            let mut chunks = block.chunks_exact_mut(8);
            for chunk in &mut chunks { chunk.copy_from_slice(&rng.next().to_le_bytes()); }
            let tail = chunks.into_remainder();
            let last = rng.next().to_le_bytes();
            tail.copy_from_slice(&last[..tail.len()]);
            let scale = ((4 + rng.next() % 5) as u16) << 10
                | (rng.next() as u16 & 0x3ff) | ((rng.next() as u16 & 1) << 15);
            if p.ggml_type == GgmlType::Q4_K {
                block[..2].copy_from_slice(&scale.to_le_bytes());
                let min = (5u16 << 10) | (rng.next() as u16 & 0x3ff);
                block[2..4].copy_from_slice(&min.to_le_bytes());
            } else { block[208..210].copy_from_slice(&scale.to_le_bytes()); }
            if exact {
                // Powers of two keep one-hot CPU reference arithmetic exact,
                // including signed Q6 scale extrema and zero sub-scales.
                let d = [0x1400u16, 0x9400, 0x1800, 0x9800][index % 4];
                if p.ggml_type == GgmlType::Q4_K {
                    block[..2].copy_from_slice(&d.to_le_bytes());
                    block[2..4].copy_from_slice(&0x1000u16.to_le_bytes());
                } else {
                    block[208..210].copy_from_slice(&d.to_le_bytes());
                    for (i, value) in block[192..208].iter_mut().enumerate() {
                        *value = [-128i8, 127, -1, 0, 1, -64, 63, 32][(i + index) % 8] as u8;
                    }
                }
            }
        }
        let digest = Sha256::digest(&*all).into();
        let view = root.slice_view((GUARD * 4) as u64, raw_bytes);
        Self { root, view, digest }
    }
    fn unchanged(&self) {
        let digest: [u8; 32] = Sha256::digest(self.root.as_slice::<u8>().unwrap()).into();
        assert_eq!(digest, self.digest, "raw source or guard mutation");
    }
}
fn prepare(device: &MlxDevice, registry: &mut KernelRegistry, source: &Source,
    p: &GgmlQuantizedMatmulIdParams) -> CompletedGroupedM8Experts {
    prepare_quantized_matmul_id_ggml_grouped_m8_batch(device, registry,
        vec![GroupedM8Preparation { source: source.view.clone(), params: *p }]).unwrap().remove(0)
}
fn policy(tensor: bool, compact: bool, q6_nr2: bool) -> GgmlRoutingPolicy {
    GgmlRoutingPolicy { expert_tensor_mm: if tensor { GgmlTensorMmPreference::AutoProbe }
        else { GgmlTensorMmPreference::ForceSimd }, expert_q6k_mv_nr2: q6_nr2,
        expert_mm_schedule: if compact { ExpertMmSchedule::CompactTiles } else { ExpertMmSchedule::UniformGrid },
        ..GgmlRoutingPolicy::default() }
}

fn frozen_mm(down: bool) {
    assert!(cfg!(target_endian = "little"));
    let device = MlxDevice::new().unwrap();
    assert!(device.name().starts_with("Apple M5"), "frozen tensor arithmetic requires its M5 qualification device");
    let mut registry = KernelRegistry::new();
    let (packet, expected_sha, n, k) = if down {
        (include_str!("fixtures/prepared-legacy-down-raw-output-goldens.json"),
         "74e59fc3556a5764e2a0e04ab6311ccccf581fbbb604d6c429a080e92529b150", 2048, 512)
    } else {
        (include_str!("fixtures/prepared-legacy-gate-raw-output-goldens.json"),
         "3964b02aaa3878a653b2526e4eb387374d77424ddb0cb804f29aa038e912f453", 512, 2048)
    };
    assert_eq!(format!("{:x}", Sha256::digest(packet.as_bytes())), expected_sha, "immutable expected packet changed");
    let goldens: Value = serde_json::from_str(packet).unwrap();
    assert_eq!(goldens["records"].as_array().unwrap().len(), 16);
    let mut compared = 0u64;
    // All 40 distinct banks and all original shapes/data/cases are retained;
    // one bank is resident at a time because this is numerical, not a cost gate.
    for layer in 0..BANKS {
        let ty = if down && layer % 2 != 0 { GgmlType::Q6_K } else { GgmlType::Q4_K };
        let p = params(ty, n, k, EXPERTS, MAX_TOKENS, TOP_K);
        let mut rng = Rng(0xc057_4d4d_0000_0001 ^ layer as u64 ^ ((down as u64) << 40));
        let source = Source::new(&device, &p, &mut rng, false);
        let prepared = prepare(&device, &mut registry, &source, &p);
        let input_words = MAX_TOKENS * if down { TOP_K } else { 1 } * k;
        let mut input = Guarded::new(&device, input_words, DType::F32);
        for value in input.payload_mut(input_words) {
            *value = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
        }
        let input_digest = input.digest(input_words);
        let mut ids = Guarded::new(&device, MAX_TOKENS * TOP_K, DType::U32);
        let mut raw_output = Guarded::new(&device, MAX_TOKENS * TOP_K * n, DType::F32);
        let mut prepared_output = Guarded::new(&device, MAX_TOKENS * TOP_K * n, DType::F32);
        let mut raw_scratch = IdMmScratch::alloc(&device, EXPERTS as u32, MAX_TOKENS as u32).unwrap();
        let mut prepared_scratch = IdMmScratch::alloc(&device, EXPERTS as u32, MAX_TOKENS as u32).unwrap();
        let mut record_index = 0;
        for tokens in [128, 256] { for concentrated in [false, true] {
            for tensor in [false, true] { for compact in [false, true] {
                let p = params(ty, n, k, EXPERTS, tokens, TOP_K);
                let rows = tokens * TOP_K;
                let words = rows * n;
                ids.reset(rows, 0);
                for (i, id) in ids.payload_mut(rows).iter_mut().enumerate() {
                    let token = i / TOP_K;
                    let slot = i % TOP_K;
                    *id = (if concentrated { (slot + layer * 7) % EXPERTS }
                        else { (token * 7 + slot * 13 + layer * 11) % EXPERTS }) as u32;
                }
                raw_output.reset(words, 0x7fc1_1111);
                prepared_output.reset(words, 0x7fc2_2222);
                let iv = input.view(tokens * if down { TOP_K } else { 1 } * k);
                let idv = ids.view(rows);
                let rv = raw_output.view(words);
                let pv = prepared_output.view(words);
                let routing = policy(tensor, compact, true);
                let mut encoder = device.command_encoder().unwrap();
                if down {
                    quantized_matmul_id_ggml_pooled_slotted_with_policy(&mut encoder, &mut registry,
                        &device, &iv, &source.view, &idv, &rv, &mut raw_scratch, &p, &routing).unwrap();
                    quantized_matmul_id_ggml_prepared_pooled_slotted_with_policy(&mut encoder, &mut registry,
                        &device, &iv, &prepared, &idv, &pv, &mut prepared_scratch, &p, &routing).unwrap();
                } else {
                    quantized_matmul_id_ggml_pooled_with_policy(&mut encoder, &mut registry,
                        &device, &iv, &source.view, &idv, &rv, &mut raw_scratch, &p, &routing).unwrap();
                    quantized_matmul_id_ggml_prepared_pooled_with_policy(&mut encoder, &mut registry,
                        &device, &iv, &prepared, &idv, &pv, &mut prepared_scratch, &p, &routing).unwrap();
                }
                encoder.commit_and_wait().unwrap();
                let record = &goldens["records"][record_index];
                assert_eq!(record["case"]["tokens"], tokens);
                assert_eq!(record["case"]["tensor"], tensor);
                assert_eq!(record["case"]["compact"], compact);
                assert_eq!(record["case"]["routing"], if concentrated { "concentrated_eight" } else { "balanced" });
                let expected = &record["raw_output_hashes"][layer];
                assert_eq!(expected["layer"], layer);
                assert_eq!(expected["f32_words"], words);
                assert_eq!(expected["format"], format!("{ty:?}"));
                let expected_hash = expected["sha256"].as_str().unwrap();
                assert_eq!(raw_output.digest(words), expected_hash,
                    "raw independent hash down={down} layer={layer} case={record_index}");
                assert_eq!(prepared_output.digest(words), expected_hash,
                    "prepared independent hash down={down} layer={layer} case={record_index}");
                assert_eq!(raw_output.payload(words), prepared_output.payload(words));
                assert!(raw_output.payload(words).iter().all(|&x| f32::from_bits(x).is_finite()));
                raw_output.check(words); prepared_output.check(words); ids.check(rows);
                compared += words as u64;
                record_index += 1;
            }}
        }}
        assert_eq!(record_index, 16);
        source.unchanged();
        assert_eq!(input.digest(input_words), input_digest);
        input.check(input_words);
    }
    assert_eq!(compared, if down { 2_013_265_920 } else { 503_316_480 });
}

#[test]
#[ignore = "M5 hardware; all 640 production-shape raw AND prepared gate MM digests"]
fn prepared_legacy_gate_mm_independent_goldens() { frozen_mm(false); }
#[test]
#[ignore = "M5 hardware; all 640 production-shape raw AND prepared mixed-down MM digests"]
fn prepared_legacy_down_mm_independent_goldens() { frozen_mm(true); }

fn scalar_weight(source: &Source, p: &GgmlQuantizedMatmulIdParams, expert: usize,
    row: usize, column: usize) -> f32 {
    let bytes = source.view.as_slice::<u8>().unwrap();
    let offset = expert * p.expert_stride as usize
        + (row * (p.k as usize / 256) + column / 256) * p.ggml_type.block_bytes() as usize;
    let b = &bytes[offset..offset + p.ggml_type.block_bytes() as usize];
    let e = column % 256;
    let half = |at: usize| half::f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32();
    if p.ggml_type == GgmlType::Q4_K {
        let group = e / 32;
        let scales = &b[4..16];
        let (sc, min) = if group < 4 { (scales[group] & 63, scales[group + 4] & 63) }
            else { ((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                    (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)) };
        let q = (b[16 + (group / 2) * 32 + e % 32] >> (4 * (group & 1))) & 15;
        half(0) * f32::from(sc) * f32::from(q) - half(2) * f32::from(min)
    } else {
        let ip = e / 128;
        let sub = (e % 128) / 32;
        let l = e % 32;
        let low = (b[ip * 64 + (sub & 1) * 32 + l] >> (if sub >= 2 { 4 } else { 0 })) & 15;
        let high = (b[128 + ip * 32 + l] >> (2 * sub)) & 3;
        half(208) * f32::from(b[192 + e / 16] as i8) * (i32::from(low | (high << 4)) - 32) as f32
    }
}

#[derive(Clone, Copy, Debug)]
enum MvKind { Stock, Nr2, Geglu }
fn mv_dispatch(prepared: bool, kind: MvKind, masked: bool, encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry, device: &MlxDevice, input: &MlxBuffer, source: &Source,
    planes: &CompletedGroupedM8Experts, ids: &MlxBuffer, output: &MlxBuffer,
    p: &GgmlQuantizedMatmulIdParams, routing: &GgmlRoutingPolicy) {
    if prepared {
        match (kind, masked) {
            (MvKind::Geglu, false) => quantized_matmul_id_ggml_prepared_mv_grouped_geglu(encoder, registry, device, input, planes, ids, output, p),
            (MvKind::Nr2, false) if p.ggml_type == GgmlType::Q4_K => quantized_matmul_id_ggml_prepared_mv_q4_k_nr2(encoder, registry, device, input, planes, ids, output, p),
            (MvKind::Nr2, true) if p.ggml_type == GgmlType::Q4_K => quantized_matmul_id_ggml_prepared_mv_q4_k_nr2_masked(encoder, registry, device, input, planes, ids, output, p),
            (_, true) => quantized_matmul_id_ggml_prepared_mv_masked_with_policy(encoder, registry, device, input, planes, ids, output, p, routing),
            _ => quantized_matmul_id_ggml_prepared_mv_with_policy(encoder, registry, device, input, planes, ids, output, p, routing),
        }.unwrap();
    } else {
        match (kind, masked) {
            (MvKind::Geglu, false) => quantized_matmul_id_ggml_mv_grouped_geglu(encoder, registry, device, input, &source.view, ids, output, p),
            (MvKind::Nr2, false) if p.ggml_type == GgmlType::Q4_K => quantized_matmul_id_ggml_mv_q4_k_nr2(encoder, registry, device, input, &source.view, ids, output, p),
            (MvKind::Nr2, true) if p.ggml_type == GgmlType::Q4_K => quantized_matmul_id_ggml_mv_q4_k_nr2_masked(encoder, registry, device, input, &source.view, ids, output, p),
            (_, true) => quantized_matmul_id_ggml_mv_masked(encoder, registry, device, input, &source.view, ids, output, p),
            _ => quantized_matmul_id_ggml_mv_with_policy(encoder, registry, device, input, &source.view, ids, output, p, routing),
        }.unwrap();
    }
}

#[test]
#[ignore = "Apple GPU; complete stock/NR2/masked/GEGLU arithmetic and poison edges"]
fn prepared_legacy_mv_masks_poison_and_exact_one_hot() {
    let device = MlxDevice::new().unwrap();
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for (n, k) in [(512, 2048), (2048, 512), (512, 256), (512, 768), (512, 1024)] {
            for kind in [MvKind::Stock, MvKind::Nr2, MvKind::Geglu] {
                if matches!(kind, MvKind::Geglu) && ty != GgmlType::Q4_K { continue; }
                let routing = policy(false, false, matches!(kind, MvKind::Nr2));
                let mut registry = KernelRegistry::new();
                registry.freeze_ggml_routing_policy(routing).unwrap();
                let p = params(ty, n, k, 64, 32, 8);
                let mut rng = Rng(0x619a_56ce_33a2_8891 ^ k as u64 ^ n as u64);
                let source = Source::new(&device, &p, &mut rng, true);
                let planes = prepare(&device, &mut registry, &source, &p);
                let cols = if matches!(kind, MvKind::Geglu) { n / 2 } else { n };
                for tokens in [1, 2, 3, 4, 8, 16, 24, 32] {
                    let p = GgmlQuantizedMatmulIdParams { n_tokens: tokens as u32, ..p };
                    let rows = tokens * 8;
                    let words = rows * cols;
                    let mut input = Guarded::new(&device, tokens * k, DType::F32);
                    let mut ids = Guarded::new(&device, rows, DType::U32);
                    let mut raw = Guarded::new(&device, words, DType::F32);
                    let mut candidate = Guarded::new(&device, words, DType::F32);
                    for activation_mode in 0..4 {
                        let one_hot = activation_mode == 0;
                        input.reset(tokens * k, 0);
                        for token in 0..tokens {
                            if one_hot { input.payload_mut(tokens * k)[token * k + (token * 37 + 19) % k] = 1.0f32.to_bits(); }
                            else {
                                for value in &mut input.payload_mut(tokens * k)[token * k..(token + 1) * k] {
                                    *value = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
                                }
                            }
                        }
                        if activation_mode >= 2 {
                            input.payload_mut(tokens * k)[(tokens - 1) * k + 7] =
                                if activation_mode == 2 { f32::NAN.to_bits() } else { f32::INFINITY.to_bits() };
                        }
                        let input_digest = input.digest(tokens * k);
                        for exceptional in [false, true] { for masked in [false, true] {
                            if matches!(kind, MvKind::Geglu) && masked { continue; }
                            ids.reset(rows, 0);
                            for (pair, id) in ids.payload_mut(rows).iter_mut().enumerate() {
                                // Long repeated-expert chunks exercise all NC=1..4 grouped tails.
                                *id = if exceptional && pair % 7 == 0 { if pair % 2 == 0 { 64 } else { u32::MAX } }
                                    else if pair % 11 < 7 { 3 } else { (pair * 13 % 64) as u32 };
                            }
                            raw.reset(words, 0x7fc1_1111); candidate.reset(words, 0x7fc2_2222);
                            let iv = input.view(tokens * k); let idv = ids.view(rows);
                            let rv = raw.view(words); let pv = candidate.view(words);
                            let mut encoder = device.command_encoder().unwrap();
                            mv_dispatch(false, kind, masked, &mut encoder, &mut registry, &device,
                                &iv, &source, &planes, &idv, &rv, &p, &routing);
                            mv_dispatch(true, kind, masked, &mut encoder, &mut registry, &device,
                                &iv, &source, &planes, &idv, &pv, &p, &routing);
                            encoder.commit_and_wait().unwrap();
                            assert_eq!(raw.payload(words), candidate.payload(words), "{ty:?} {kind:?} N={n} K={k} tokens={tokens} masked={masked}");
                            for pair in 0..rows {
                                let values = &candidate.payload(words)[pair * cols..(pair + 1) * cols];
                                let expert = ids.payload(rows)[pair] as usize;
                                if expert >= 64 {
                                    if masked { assert!(values.iter().all(|&x| x == 0)); }
                                    else { assert!(values.iter().all(|&x| f32::from_bits(x).is_nan())); }
                                } else if one_hot && !matches!(kind, MvKind::Geglu) {
                                    let column = ((pair / 8) * 37 + 19) % k;
                                    for (row, &actual) in values.iter().enumerate() {
                                        let expected = scalar_weight(&source, &p, expert, row, column);
                                        let expected = if expected == 0.0 { 0.0 } else { expected };
                                        assert_eq!(actual, expected.to_bits(), "independent one-hot {ty:?} {kind:?} pair={pair} row={row}");
                                    }
                                } else if activation_mode >= 2 && pair / 8 == tokens - 1 {
                                    assert!(values.iter().any(|&x| !f32::from_bits(x).is_finite()));
                                } else {
                                    assert!(values.iter().all(|&x| f32::from_bits(x).is_finite()));
                                }
                            }
                            raw.check(words); candidate.check(words); ids.check(rows); input.check(tokens * k);
                            assert_eq!(input.digest(tokens * k), input_digest, "activation mutation");
                        }}
                    }
                }
                source.unchanged();
            }
        }
    }
}

#[test]
#[ignore = "M5 hardware; auto MV/MM, paired concurrency, slotted reuse and MM poison"]
fn prepared_legacy_auto_pair_slotted_and_schedule_reuse() {
    let device = MlxDevice::new().unwrap();
    assert!(device.name().starts_with("Apple M5"));
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for tensor in [false, true] { for compact in [false, true] {
            let routing = policy(tensor, compact, true);
            let mut registry = KernelRegistry::new();
            registry.freeze_ggml_routing_policy(routing).unwrap();
            for top_k in [1, 6, 8] {
                let base = params(ty, 512, 2048, 64, 256, top_k);
                let mut rng = Rng(0xa17e_1d66_0994_2318 ^ top_k as u64);
                let first = Source::new(&device, &base, &mut rng, false);
                let second = Source::new(&device, &base, &mut rng, false);
                let first_planes = prepare(&device, &mut registry, &first, &base);
                let second_planes = prepare(&device, &mut registry, &second, &base);
                for tokens in [1, 8, 32, 33, 65, 128, 256] {
                    let p = GgmlQuantizedMatmulIdParams { n_tokens: tokens as u32, ..base };
                    let rows = tokens * top_k;
                    let words = rows * 512;
                    let mut input = Guarded::new(&device, tokens * 2048, DType::F32);
                    let mut slotted = Guarded::new(&device, rows * 2048, DType::F32);
                    for buffer in [&mut input, &mut slotted] {
                        for v in buffer.payload_mut(buffer.capacity) {
                            *v = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
                        }
                    }
                    let mut ids = Guarded::new(&device, rows, DType::U32);
                    let mut ra = Guarded::new(&device, words, DType::F32);
                    let mut rb = Guarded::new(&device, words, DType::F32);
                    let mut pa = Guarded::new(&device, words, DType::F32);
                    let mut pb = Guarded::new(&device, words, DType::F32);
                    let mut rs = Guarded::new(&device, words, DType::F32);
                    let mut ps = Guarded::new(&device, words, DType::F32);
                    let mut raw_scratch = IdMmScratch::alloc_with_top_k(&device, 64, tokens as u32, top_k as u32).unwrap();
                    let mut prepared_scratch = IdMmScratch::alloc_with_top_k(&device, 64, tokens as u32, top_k as u32).unwrap();
                    for invalid in [None, Some(64u32), Some(u32::MAX)] {
                        ids.reset(rows, 0);
                        for (i, id) in ids.payload_mut(rows).iter_mut().enumerate() {
                            *id = ((i / top_k * 7 + i % top_k * 13) % 64) as u32;
                        }
                        if let Some(id) = invalid { ids.payload_mut(rows)[rows / 2] = id; }
                        for b in [&mut ra, &mut rb, &mut pa, &mut pb, &mut rs, &mut ps] {
                            b.reset(words, 0x7fc5_6666);
                        }
                        let iv = input.view(tokens * 2048);
                        let sv = slotted.view(rows * 2048);
                        let idv = ids.view(rows);
                        let rav = ra.view(words); let rbv = rb.view(words);
                        let pav = pa.view(words); let pbv = pb.view(words);
                        let rsv = rs.view(words); let psv = ps.view(words);
                        let mut encoder = device.command_encoder().unwrap();
                        if tokens > routing.expert_mm_threshold as usize {
                            quantized_matmul_id_ggml_pooled_pair_with_policy(&mut encoder, &mut registry,
                                &device, &iv, &first.view, &second.view, &idv, &rav, &rbv,
                                &mut raw_scratch, &p, &routing).unwrap();
                            quantized_matmul_id_ggml_prepared_pooled_pair_with_policy(&mut encoder, &mut registry,
                                &device, &iv, &first_planes, &second_planes, &idv, &pav, &pbv,
                                &mut prepared_scratch, &p, &routing).unwrap();
                            // Separate outputs, no projection barrier: preserve real paired
                            // concurrency while consuming the same immutable routing schedule.
                            quantized_matmul_id_ggml_pooled_slotted_reusing_schedule_with_policy(
                                &mut encoder, &mut registry, &device, &sv, &first.view, &idv,
                                &rsv, &mut raw_scratch, &p, &routing).unwrap();
                            quantized_matmul_id_ggml_prepared_pooled_slotted_reusing_schedule_with_policy(
                                &mut encoder, &mut registry, &device, &sv, &first_planes, &idv,
                                &psv, &mut prepared_scratch, &p, &routing).unwrap();
                        } else {
                            quantized_matmul_id_ggml_pooled_with_policy(&mut encoder, &mut registry,
                                &device, &iv, &first.view, &idv, &rav, &mut raw_scratch, &p, &routing).unwrap();
                            quantized_matmul_id_ggml_prepared_with_policy(&mut encoder, &mut registry,
                                &device, &iv, &first_planes, &idv, &pav, &mut prepared_scratch, &p, &routing).unwrap();
                        }
                        encoder.commit_and_wait().unwrap();
                        assert_eq!(ra.payload(words), pa.payload(words),
                            "first: ty={ty:?} tensor={tensor} compact={compact} top_k={top_k} tokens={tokens} invalid={invalid:?}");
                        if tokens > routing.expert_mm_threshold as usize {
                            assert_eq!(rb.payload(words), pb.payload(words),
                                "second: ty={ty:?} tensor={tensor} compact={compact} top_k={top_k} tokens={tokens} invalid={invalid:?}");
                            assert_eq!(rs.payload(words), ps.payload(words),
                                "slotted: ty={ty:?} tensor={tensor} compact={compact} top_k={top_k} tokens={tokens} invalid={invalid:?}");
                            if invalid.is_some() {
                                for b in [&pa, &pb, &ps] {
                                    assert!(b.payload(words).iter().all(|&x| f32::from_bits(x).is_nan()));
                                }
                            }
                        } else if invalid.is_some() {
                            assert!(pa.payload(words)[rows / 2 * 512..(rows / 2 + 1) * 512]
                                .iter().all(|&x| f32::from_bits(x).is_nan()));
                        } else {
                            assert!(pa.payload(words).iter().all(|&x| f32::from_bits(x).is_finite()));
                        }
                        for b in [&ra, &rb, &pa, &pb, &rs, &ps] { b.check(words); }
                        ids.check(rows); input.check(tokens * 2048); slotted.check(rows * 2048);
                    }
                }
                first.unchanged(); second.unchanged();
            }
        }}
    }
}

#[test]
#[ignore = "Apple GPU; fused Q6 gate/up/SiLU full bitwise output and schedule reuse"]
fn prepared_legacy_fused_q6_silu_and_poison() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    registry.freeze_ggml_routing_policy(policy(false, false, true)).unwrap();
    for (n, k) in [(512, 2048), (2048, 512)] { for top_k in [1, 8] {
        let base = params(GgmlType::Q6_K, n, k, 64, 256, top_k);
        let mut rng = Rng(0xf065_6a7e_a790_1e5e ^ n as u64);
        let gate = Source::new(&device, &base, &mut rng, false);
        let up = Source::new(&device, &base, &mut rng, false);
        let gate_planes = prepare(&device, &mut registry, &gate, &base);
        let up_planes = prepare(&device, &mut registry, &up, &base);
        for tokens in [1, 17, 32, 33, 65, 128, 256] {
            let p = GgmlQuantizedMatmulIdParams { n_tokens: tokens as u32, ..base };
            let dispatch = GgmlIdMmDispatchParams { n_tokens: p.n_tokens, top_k: p.top_k,
                n: p.n, k: p.k, n_experts: p.n_experts, expert_stride: p.expert_stride, ggml_type: p.ggml_type };
            let rows = tokens * top_k; let words = rows * n;
            let mut input = Guarded::new(&device, tokens * k, DType::F32);
            for v in input.payload_mut(tokens * k) {
                *v = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
            }
            let mut ids = Guarded::new(&device, rows, DType::U32);
            let mut raw = Guarded::new(&device, words, DType::F32);
            let mut candidate = Guarded::new(&device, words, DType::F32);
            let mut reused = Guarded::new(&device, words, DType::F32);
            let raw_scratch = IdMmScratch::alloc_with_top_k(&device, 64, tokens as u32, top_k as u32).unwrap();
            let mut htpe = raw_scratch.htpe().clone();
            let mut hids = raw_scratch.hids().clone();
            let mut scratch = IdMmScratch::alloc_with_top_k(&device, 64, tokens as u32, top_k as u32).unwrap();
            if tokens == 33 && n == 512 && top_k == 8 {
                let mut guarded = Guarded::new(&device, words + 1, DType::F32);
                guarded.reset(words + 1, 0x4555_6666);
                let misaligned = guarded.root.slice_view((GUARD * 4 + 4) as u64, words);
                let mut encoder = device.command_encoder().unwrap();
                let result = quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(
                    &mut encoder, &mut registry, &device, &input.view(tokens * k),
                    &gate_planes, &up_planes, &ids.view(rows), &misaligned,
                    &mut scratch, &p, false,
                );
                assert!(matches!(result, Err(mlx_native::MlxError::InvalidArgument(_))));
                encoder.commit_and_wait().unwrap();
                assert!(guarded.payload(words + 1).iter().all(|&word| word == 0x4555_6666));
                guarded.check(words + 1);
            }
            for invalid in [None, Some(64u32), Some(u32::MAX)] {
                ids.reset(rows, 0);
                for (i, id) in ids.payload_mut(rows).iter_mut().enumerate() {
                    *id = ((i / top_k * 7 + i % top_k * 13) % 64) as u32;
                }
                if let Some(id) = invalid { ids.payload_mut(rows)[rows / 2] = id; }
                raw.reset(words, 0x7fc1_1111); candidate.reset(words, 0x7fc2_2222);
                reused.reset(words, 0x7fc3_3333);
                let iv = input.view(tokens * k); let idv = ids.view(rows);
                let rv = raw.view(words); let pv = candidate.view(words); let sv = reused.view(words);
                let mut encoder = device.command_encoder().unwrap();
                dispatch_id_mm_fused_gate_up_silu_for_test(&mut encoder, &mut registry, &device,
                    &iv, &gate.view, &up.view, &idv, &mut htpe, &mut hids, &rv, &dispatch).unwrap();
                quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(&mut encoder,
                    &mut registry, &device, &iv, &gate_planes, &up_planes, &idv, &pv,
                    &mut scratch, &p, false).unwrap();
                quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(&mut encoder,
                    &mut registry, &device, &iv, &gate_planes, &up_planes, &idv, &sv,
                    &mut scratch, &p, true).unwrap();
                encoder.commit_and_wait().unwrap();
                assert_eq!(raw.payload(words), candidate.payload(words));
                assert_eq!(raw.payload(words), reused.payload(words));
                if invalid.is_some() {
                    assert!(candidate.payload(words).iter().all(|&x| f32::from_bits(x).is_nan()));
                } else {
                    assert!(candidate.payload(words).iter().all(|&x| f32::from_bits(x).is_finite()));
                    assert!(candidate.payload(words).iter().any(|&x| f32::from_bits(x) != 0.0));
                }
                raw.check(words); candidate.check(words); reused.check(words);
                ids.check(rows); input.check(tokens * k);
            }
        }
        gate.unchanged(); up.unchanged();
    }}
}

#[test]
#[ignore = "Apple GPU; failed paired preflight cannot encode partial numerical work"]
fn prepared_legacy_pair_preflight_leaves_first_output_untouched() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let routing = policy(false, false, true);
    let p = params(GgmlType::Q4_K, 512, 2048, 64, 65, 8);
    let mut rng = Rng(0x8d6e_e71d_0110_493e);
    let source = Source::new(&device, &p, &mut rng, false);
    let planes = prepare(&device, &mut registry, &source, &p);
    let mut input = Guarded::new(&device, 65 * 2048, DType::F32);
    input.reset(65 * 2048, 1.0f32.to_bits());
    let mut ids = Guarded::new(&device, 65 * 8, DType::U32);
    ids.reset(65 * 8, 0);
    let words = 65 * 8 * 512;
    let mut first = Guarded::new(&device, words, DType::F32);
    first.reset(words, 0x4555_6666);
    let second = Guarded::new(&device, words - 1, DType::F32);
    let mut scratch = IdMmScratch::alloc(&device, 64, 65).unwrap();
    let mut encoder = device.command_encoder().unwrap();
    assert!(quantized_matmul_id_ggml_prepared_pooled_pair_with_policy(&mut encoder,
        &mut registry, &device, &input.view(65 * 2048), &planes, &planes, &ids.view(65 * 8),
        &first.view(words), &second.view(words - 1), &mut scratch, &p, &routing).is_err());
    encoder.commit_and_wait().unwrap();
    assert!(first.payload(words).iter().all(|&x| x == 0x4555_6666));
    first.check(words); second.check(words - 1);
    source.unchanged();
}

#[test]
#[ignore = "Apple GPU; fused token indexing across 32768 with independent one-row dispatches"]
fn prepared_legacy_fused_q6_large_token_index() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    registry.freeze_ggml_routing_policy(policy(false, false, true)).unwrap();
    let p = params(GgmlType::Q6_K, 64, 256, 8, 32769, 1);
    let mut rng = Rng(0x3276_8f06_60aa_5e19);
    let gate = Source::new(&device, &p, &mut rng, false);
    let up = Source::new(&device, &p, &mut rng, false);
    let gate_planes = prepare(&device, &mut registry, &gate, &p);
    let up_planes = prepare(&device, &mut registry, &up, &p);
    let tokens = p.n_tokens as usize;
    let words = tokens * 64;
    let mut input = Guarded::new(&device, tokens * 256, DType::F32);
    for value in input.payload_mut(tokens * 256) {
        *value = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
    }
    let input_digest = input.digest(tokens * 256);
    let mut ids = Guarded::new(&device, tokens, DType::U32);
    ids.reset(tokens, 0);
    for (token, id) in ids.payload_mut(tokens).iter_mut().enumerate() { *id = (token % 8) as u32; }
    let mut raw = Guarded::new(&device, words, DType::F32);
    let mut candidate = Guarded::new(&device, words, DType::F32);
    raw.reset(words, 0x7fc1_1111); candidate.reset(words, 0x7fc2_2222);
    let raw_scratch = IdMmScratch::alloc_with_top_k(&device, 8, p.n_tokens, 1).unwrap();
    let mut htpe = raw_scratch.htpe().clone(); let mut hids = raw_scratch.hids().clone();
    let mut scratch = IdMmScratch::alloc_with_top_k(&device, 8, p.n_tokens, 1).unwrap();
    let dispatch = GgmlIdMmDispatchParams { n_tokens: p.n_tokens, top_k: 1,
        n: p.n, k: p.k, n_experts: p.n_experts, expert_stride: p.expert_stride, ggml_type: p.ggml_type };
    let mut encoder = device.command_encoder().unwrap();
    dispatch_id_mm_fused_gate_up_silu_for_test(&mut encoder, &mut registry, &device,
        &input.view(tokens * 256), &gate.view, &up.view, &ids.view(tokens),
        &mut htpe, &mut hids, &raw.view(words), &dispatch).unwrap();
    quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(&mut encoder,
        &mut registry, &device, &input.view(tokens * 256), &gate_planes, &up_planes,
        &ids.view(tokens), &candidate.view(words), &mut scratch, &p, false).unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(raw.payload(words), candidate.payload(words));
    assert!(candidate.payload(words).iter().all(|&x| f32::from_bits(x).is_finite()));
    let one_scratch = IdMmScratch::alloc_with_top_k(&device, 8, 1, 1).unwrap();
    let mut one_htpe = one_scratch.htpe().clone(); let mut one_hids = one_scratch.hids().clone();
    let mut one = Guarded::new(&device, 64, DType::F32);
    for token in [0, 32767, 32768] {
        one.reset(64, 0x7fc3_3333);
        let iv = input.root.slice_view(((GUARD + token * 256) * 4) as u64, 256);
        let idv = ids.root.slice_view(((GUARD + token) * 4) as u64, 1);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_id_mm_fused_gate_up_silu_for_test(&mut encoder, &mut registry, &device,
            &iv, &gate.view, &up.view, &idv, &mut one_htpe, &mut one_hids,
            &one.view(64), &GgmlIdMmDispatchParams { n_tokens: 1, ..dispatch }).unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(&candidate.payload(words)[token * 64..(token + 1) * 64], one.payload(64),
            "fused row independence at token {token}");
        one.check(64);
    }
    raw.check(words); candidate.check(words); ids.check(tokens); input.check(tokens * 256);
    assert_eq!(input.digest(tokens * 256), input_digest);
    gate.unchanged(); up.unchanged();
}
