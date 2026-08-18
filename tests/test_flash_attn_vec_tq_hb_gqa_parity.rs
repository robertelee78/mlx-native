//! Exact-output gate for KV-head-cooperative TQ-HB decode attention.
//!
//! The optimized kernel must preserve every per-query-head arithmetic order
//! while loading each packed K/V vector once for two query heads that share a
//! KV head. The legacy per-query-head kernel is the production oracle.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::ops::flash_attn_vec_tq_hb::{self, FlashAttnVecTqHbParams, GqaTile};
use mlx_native::{DType, KernelRegistry, MlxDevice};

const NH: u32 = 24;
const NKV: u32 = 4;
const HD: u32 = 256;

struct Xor(u64);

impl Xor {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as u32
    }

    fn signed(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }
}

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, shape)
        .expect("allocate f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("map f32 buffer")
        .copy_from_slice(values);
    buffer
}

fn u8_buffer(device: &MlxDevice, values: &[u8], shape: Vec<usize>) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len(), DType::U8, shape)
        .expect("allocate u8 buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("map u8 buffer")
        .copy_from_slice(values);
    buffer
}

fn run_case(kv_seq_len: u32, nsg: u32, codebook_bits: u32, tile: GqaTile) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_vec_tq_hb::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry);

    let cap = kv_seq_len as usize;
    let mut rng = Xor::new(0xC001_D00D ^ kv_seq_len as u64 ^ ((nsg as u64) << 32));
    let q: Vec<f32> = (0..(NH * HD)).map(|_| rng.signed()).collect();
    let kv_elements = NKV as usize * cap * HD as usize;
    let mask = match codebook_bits {
        5 => 0x1f,
        6 => 0x3f,
        8 => 0xff,
        _ => unreachable!(),
    };
    let k: Vec<u8> = (0..kv_elements)
        .map(|_| (rng.next_u32() as u8) & mask)
        .collect();
    let v: Vec<u8> = (0..kv_elements)
        .map(|_| (rng.next_u32() as u8) & mask)
        .collect();
    let norms: Vec<f32> = (0..NKV as usize * cap)
        .map(|_| 0.5 + rng.signed().abs())
        .collect();

    let q = f32_buffer(&device, &q, vec![NH as usize, HD as usize]);
    let k = u8_buffer(&device, &k, vec![NKV as usize, cap, HD as usize]);
    let v = u8_buffer(&device, &v, vec![NKV as usize, cap, HD as usize]);
    let k_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
    let v_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
    let output_bytes = (NH * HD * 4) as usize;
    let legacy = device
        .alloc_buffer(output_bytes, DType::F32, vec![NH as usize, HD as usize])
        .expect("legacy output");
    let cooperative = device
        .alloc_buffer(output_bytes, DType::F32, vec![NH as usize, HD as usize])
        .expect("cooperative output");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, HD);
    let legacy_tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("legacy tmp");
    let cooperative_tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("cooperative tmp");
    let params = FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: HD,
        kv_seq_len,
        kv_capacity: kv_seq_len,
        scale: 1.0 / (HD as f32).sqrt(),
        mask_type: 0,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits,
        fuse_fwht_pre: 0,
        nsg,
    };

    let mut encoder = device.command_encoder().expect("legacy encoder");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &k,
        &k_norms,
        &v,
        &v_norms,
        &legacy,
        &legacy_tmp,
        &params,
    )
    .expect("legacy dispatch");
    encoder.commit_and_wait().expect("legacy completion");

    let mut encoder = device.command_encoder().expect("cooperative encoder");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_gqa(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &k,
        &k_norms,
        &v,
        &v_norms,
        &cooperative,
        &cooperative_tmp,
        &params,
        tile,
    )
    .expect("cooperative dispatch");
    encoder.commit_and_wait().expect("cooperative completion");

    let legacy = legacy.as_slice::<f32>().expect("read legacy");
    let cooperative = cooperative.as_slice::<f32>().expect("read cooperative");
    for (index, (&expected, &actual)) in legacy.iter().zip(cooperative).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "kL={kv_seq_len} NSG={nsg} cbits={codebook_bits} tile={tile:?}: output[{index}] differs: legacy={expected:?}, cooperative={actual:?}"
        );
    }
}

#[test]
fn qwen38_q2_matches_legacy_at_boundaries_and_nsg_modes() {
    for &(kv_seq_len, nsg) in &[(1, 1), (31, 1), (32, 1), (33, 1), (128, 4)] {
        for codebook_bits in [5, 6, 8] {
            run_case(kv_seq_len, nsg, codebook_bits, GqaTile::Q2);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Legacy,
    Q2,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "M5 Max performance gate; allocates a 105K TQ cache"]
fn bench_qwen38_gqa_tiles() {
    for kv_seq_len in [8_192u32, 32_768, 65_536, 104_966] {
        let device = MlxDevice::new().expect("Metal device");
        let mut registry = KernelRegistry::new();
        flash_attn_vec_tq_hb::register(&mut registry);
        mlx_native::ops::flash_attn_vec::register(&mut registry);

        let cap = kv_seq_len as usize;
        let mut rng = Xor::new(0x51A7_2026 ^ kv_seq_len as u64);
        let q: Vec<f32> = (0..(NH * HD)).map(|_| rng.signed()).collect();
        let kv_elements = NKV as usize * cap * HD as usize;
        let k: Vec<u8> = (0..kv_elements).map(|_| rng.next_u32() as u8).collect();
        let v: Vec<u8> = (0..kv_elements).map(|_| rng.next_u32() as u8).collect();
        let norms: Vec<f32> = (0..NKV as usize * cap)
            .map(|_| 0.5 + rng.signed().abs())
            .collect();
        let q = f32_buffer(&device, &q, vec![NH as usize, HD as usize]);
        let k = u8_buffer(&device, &k, vec![NKV as usize, cap, HD as usize]);
        let v = u8_buffer(&device, &v, vec![NKV as usize, cap, HD as usize]);
        let k_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
        let v_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
        let output_bytes = (NH * HD * 4) as usize;
        let output = device
            .alloc_buffer(output_bytes, DType::F32, vec![NH as usize, HD as usize])
            .expect("output");
        let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, HD);
        let tmp = device
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .expect("tmp");
        let params = FlashAttnVecTqHbParams {
            num_heads: NH,
            num_kv_heads: NKV,
            head_dim: HD,
            kv_seq_len,
            kv_capacity: kv_seq_len,
            scale: 1.0 / (HD as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
            fuse_fwht_pre: 0,
            nsg: 4,
        };

        let mut run = |variant: Variant| -> (f64, f64) {
            let mut encoder = device.command_encoder().expect("benchmark encoder");
            let wall_start = std::time::Instant::now();
            match variant {
                Variant::Legacy => flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &q,
                    &k,
                    &k_norms,
                    &v,
                    &v_norms,
                    &output,
                    &tmp,
                    &params,
                ),
                Variant::Q2 => flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_gqa(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &q,
                    &k,
                    &k_norms,
                    &v,
                    &v_norms,
                    &output,
                    &tmp,
                    &params,
                    GqaTile::Q2,
                ),
            }
            .expect("benchmark dispatch");
            let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time().expect("GPU time");
            (
                (gpu_end - gpu_start) * 1_000.0,
                wall_start.elapsed().as_secs_f64() * 1_000.0,
            )
        };

        for variant in [Variant::Legacy, Variant::Q2] {
            for _ in 0..8 {
                let _ = run(variant);
            }
        }

        let mut gpu = [Vec::new(), Vec::new()];
        let mut wall = [Vec::new(), Vec::new()];
        for block in 0..7 {
            let variants = [Variant::Legacy, Variant::Q2];
            for offset in 0..2 {
                let index = (block + offset) % 2;
                let (gpu_ms, wall_ms) = run(variants[index]);
                gpu[index].push(gpu_ms);
                wall[index].push(wall_ms);
            }
        }
        let legacy_gpu = median(&mut gpu[0]);
        let q2_gpu = median(&mut gpu[1]);
        let legacy_wall = median(&mut wall[0]);
        let q2_wall = median(&mut wall[1]);
        eprintln!(
            "GQA_BENCH kL={kv_seq_len} legacy_gpu_ms={legacy_gpu:.4} q2_gpu_ms={q2_gpu:.4} q2_gpu_speedup={:.3} legacy_wall_ms={legacy_wall:.4} q2_wall_ms={q2_wall:.4}",
            legacy_gpu / q2_gpu,
        );
    }
}

#[test]
#[ignore = "M5 Max sustained-load gate; allocates a 105K TQ cache"]
fn bench_qwen38_q2_sustained_1000_steps() {
    const KV_SEQ_LEN: u32 = 104_966;
    const LAYERS_PER_STEP: usize = 16;
    const STEPS: usize = 1_000;

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_vec_tq_hb::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry);

    let cap = KV_SEQ_LEN as usize;
    let mut rng = Xor::new(0x7A3A_2026);
    let q: Vec<f32> = (0..(NH * HD)).map(|_| rng.signed()).collect();
    let kv_elements = NKV as usize * cap * HD as usize;
    let k: Vec<u8> = (0..kv_elements).map(|_| rng.next_u32() as u8).collect();
    let v: Vec<u8> = (0..kv_elements).map(|_| rng.next_u32() as u8).collect();
    let norms: Vec<f32> = (0..NKV as usize * cap)
        .map(|_| 0.5 + rng.signed().abs())
        .collect();
    let q = f32_buffer(&device, &q, vec![NH as usize, HD as usize]);
    let k = u8_buffer(&device, &k, vec![NKV as usize, cap, HD as usize]);
    let v = u8_buffer(&device, &v, vec![NKV as usize, cap, HD as usize]);
    let k_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
    let v_norms = f32_buffer(&device, &norms, vec![NKV as usize, cap]);
    let output_bytes = (NH * HD * 4) as usize;
    let output = device
        .alloc_buffer(output_bytes, DType::F32, vec![NH as usize, HD as usize])
        .expect("output");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, HD);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("tmp");
    let params = FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: HD,
        kv_seq_len: KV_SEQ_LEN,
        kv_capacity: KV_SEQ_LEN,
        scale: 1.0 / (HD as f32).sqrt(),
        mask_type: 0,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: 8,
        fuse_fwht_pre: 0,
        nsg: 4,
    };

    let mut step_ms = Vec::with_capacity(STEPS);
    for step in 0..(STEPS + 8) {
        let mut encoder = device.command_encoder().expect("sustained encoder");
        for _ in 0..LAYERS_PER_STEP {
            flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_gqa(
                &mut encoder,
                &mut registry,
                &device,
                &q,
                &k,
                &k_norms,
                &v,
                &v_norms,
                &output,
                &tmp,
                &params,
                GqaTile::Q2,
            )
            .expect("Q2 dispatch");
        }
        let started = std::time::Instant::now();
        encoder.commit_and_wait().expect("sustained completion");
        if step >= 8 {
            step_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }

    let quarter = STEPS / 4;
    let mut first = step_ms[..quarter].to_vec();
    let mut last = step_ms[(STEPS - quarter)..].to_vec();
    let first_p50 = median(&mut first);
    let last_p50 = median(&mut last);
    let ratio = last_p50 / first_p50;
    eprintln!(
        "GQA_SUSTAINED steps={STEPS} layers_per_step={LAYERS_PER_STEP} first_quarter_p50_ms={first_p50:.4} last_quarter_p50_ms={last_p50:.4} thermal_ratio={ratio:.3}"
    );
    assert!(
        ratio <= 1.15,
        "Q2 sustained-load regression: last/first quarter p50 {ratio:.3} exceeds 1.15"
    );
}
