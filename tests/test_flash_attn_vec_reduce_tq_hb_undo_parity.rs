//! ADR-028 §iter-485 H3 — Parity test for fused reduce + FWHT-sign-undo.
//!
//! Verifies that running:
//!   `flash_attn_vec_tq_hb` + `flash_attn_vec_reduce_tq_hb_undo`
//! produces the SAME output as:
//!   `flash_attn_vec_tq_hb` + `flash_attn_vec_reduce` + `fwht_sign_undo`
//!
//! within 1e-4 abs/rel tolerance per element on a synthetic Q/K/V fixture.
//!
//! Fixture is sized to force NWG > 1 (multi-WG reduce path, which is the
//! production decode path for kv_seq_len > 32). The synthetic Q is already
//! FWHT-rotated by the kernel (fuse_fwht_pre=1) so the test exercises the
//! same call chain as production gemma4 decode.

use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::flash_attn_vec_reduce_tq_hb_undo;
use mlx_native::ops::flash_attn_vec_tq_hb::{
    self, FlashAttnVecTqHbParams,
};
use mlx_native::ops::fwht_standalone;
use mlx_native::ops::hadamard_quantize_kv;

// Simple xorshift PRNG so the test is deterministic without pulling in `rand`.
struct Xor { s: u64 }
impl Xor {
    fn new(seed: u64) -> Self { Self { s: seed.wrapping_add(1) } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        // Uniform in [-1, 1).
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    }
}

fn nrmse(a: &[f32], b: &[f32]) -> f32 {
    let mut sq = 0.0f64;
    let mut norm = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y) as f64;
        sq += d * d;
        norm += (*x as f64) * (*x as f64);
    }
    if norm == 0.0 { 0.0 } else { ((sq / norm).sqrt()) as f32 }
}

fn write_f32_buf(device: &MlxDevice, name: &str, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let n = data.len();
    let mut buf = device
        .alloc_buffer(n * 4, DType::F32, shape)
        .unwrap_or_else(|e| panic!("alloc {name}: {e}"));
    buf.as_mut_slice::<f32>()
        .unwrap_or_else(|e| panic!("write {name}: {e}"))
        .copy_from_slice(data);
    buf
}

/// Encode K (and V independently) into TQ-HB-packed format via the GPU encoder.
/// Returns (packed_bytes, norms).
fn encode_hb_via_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rows_f32: &[f32],   // flat [nkv * kvl * hd] f32
    nkv: usize,
    kvl: usize,
    hd: usize,
    cbits: u32,
) -> (Vec<u8>, Vec<f32>) {
    let norms_per_pos = (hd / 256).max(1);
    let mut packed = vec![0u8; nkv * kvl * hd];
    let mut norms = vec![0f32; nkv * kvl * norms_per_pos];

    for kv_h in 0..nkv {
        for p in 0..kvl {
            let row_off = (kv_h * kvl + p) * hd;
            let row = &rows_f32[row_off..row_off + hd];

            let mut src = device
                .alloc_buffer(hd * 4, DType::F32, vec![1, hd])
                .expect("alloc src");
            src.as_mut_slice::<f32>().unwrap().copy_from_slice(row);

            let dst_packed = device
                .alloc_buffer(kvl * hd, DType::U8, vec![1, kvl, hd])
                .expect("alloc dst_packed");
            let dst_norms = device
                .alloc_buffer(kvl * 4 * norms_per_pos, DType::F32, vec![1, kvl, norms_per_pos])
                .expect("alloc dst_norms");

            let mut enc = device.command_encoder().expect("enc");
            hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
                &mut enc, registry, device.metal_device(),
                &src, &dst_packed, &dst_norms,
                1, hd as u32, kvl as u32, p as u32,
                false, 1.0, cbits,
            ).expect("encode hb");
            enc.commit_and_wait().expect("commit encode");

            let p_src = dst_packed.as_slice::<u8>().unwrap();
            packed[row_off..row_off + hd].copy_from_slice(&p_src[p * hd..(p + 1) * hd]);
            let n_src = dst_norms.as_slice::<f32>().unwrap();
            for i in 0..norms_per_pos {
                norms[(kv_h * kvl + p) * norms_per_pos + i] = n_src[p * norms_per_pos + i];
            }
        }
    }

    (packed, norms)
}

fn run_path(
    seed: u64,
    fused: bool,
) -> Vec<f32> {
    // Production gemma4 shape: 16 query heads, 4 KV heads, D=256, cbits=8.
    let nh: u32 = 16;
    let nkv: u32 = 4;
    let hd: u32 = 256;
    let kvl: u32 = 128;   // > 32 → multi-WG reduce path
    let cbits: u32 = 8;

    let device = MlxDevice::new().expect("MlxDevice");
    // KernelRegistry::new() auto-registers all bundled shader sources,
    // including flash_attn_vec_tq_hb, flash_attn_vec_reduce*, fwht_standalone,
    // hadamard_quantize_kv, and the new flash_attn_vec_reduce_tq_hb_undo.
    let mut registry = KernelRegistry::new();

    let mut rng = Xor::new(seed);

    // Q in raw domain (kernel applies FWHT-pre internally via fuse_fwht_pre=1).
    let q_data: Vec<f32> = (0..(nh as usize * hd as usize)).map(|_| rng.next_f32()).collect();

    // K, V random f32 per (kv_head, pos).
    let kv_total = (nkv * kvl * hd) as usize;
    let k_raw: Vec<f32> = (0..kv_total).map(|_| rng.next_f32()).collect();
    let v_raw: Vec<f32> = (0..kv_total).map(|_| rng.next_f32()).collect();

    let (k_packed, k_norms) = encode_hb_via_gpu(
        &device, &mut registry, &k_raw, nkv as usize, kvl as usize, hd as usize, cbits,
    );
    let (v_packed, v_norms) = encode_hb_via_gpu(
        &device, &mut registry, &v_raw, nkv as usize, kvl as usize, hd as usize, cbits,
    );

    let q_buf = write_f32_buf(&device, "Q", &q_data, vec![nh as usize, 1, hd as usize]);

    let mut k_packed_buf = device
        .alloc_buffer(k_packed.len(), DType::U8, vec![nkv as usize, kvl as usize, hd as usize])
        .expect("alloc k_packed");
    k_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&k_packed);

    let k_norms_buf = write_f32_buf(
        &device, "k_norms", &k_norms,
        vec![nkv as usize * kvl as usize],
    );

    let mut v_packed_buf = device
        .alloc_buffer(v_packed.len(), DType::U8, vec![nkv as usize, kvl as usize, hd as usize])
        .expect("alloc v_packed");
    v_packed_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&v_packed);

    let v_norms_buf = write_f32_buf(
        &device, "v_norms", &v_norms,
        vec![nkv as usize * kvl as usize],
    );

    let output_buf = device
        .alloc_buffer((nh * hd * 4) as usize, DType::F32, vec![nh as usize, 1, hd as usize])
        .expect("alloc out");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(nh, hd);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    let scale = 1.0_f32 / (hd as f32).sqrt();

    // Determine NWG the way the production dispatcher does. compute_nwg is
    // private — replicate logic here. At kvl=128 < 512, default nwg = 16.
    let _ = flash_attn_vec_tq_hb::FlashAttnVecTqHbParams { num_heads: 0, num_kv_heads: 0, head_dim: 0, kv_seq_len: 0, kv_capacity: 0, scale: 0.0, mask_type: 0, sliding_window: 0, softcap: 0.0, ring_start: 0, scale_factor_d512: 0.0, codebook_bits: 0, fuse_fwht_pre: 0, nsg: 1 };
    // For the test, force nwg to 16 (decode default) by setting kv_seq_len
    // below 512. Avoid env overrides by ensuring HF2Q_TQ_NWG isn't set.
    let prev_nwg_env = std::env::var("HF2Q_TQ_NWG").ok();
    std::env::remove_var("HF2Q_TQ_NWG");

    let params = FlashAttnVecTqHbParams {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        kv_seq_len: kvl,
        kv_capacity: kvl,
        scale,
        mask_type: 0,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: cbits,
        fuse_fwht_pre: 1,   // kernel applies FWHT-pre internally (matches prod)
        nsg: 1,
    };

    let mut encoder = device.command_encoder().expect("enc");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
        &mut encoder, &mut registry, &device,
        &q_buf, &k_packed_buf, &k_norms_buf, &v_packed_buf, &v_norms_buf,
        &output_buf, &tmp_buf, &params,
    ).expect("dispatch tq_hb");
    encoder.commit_and_wait().expect("commit sdpa");

    // The flash_attn_vec_tq_hb dispatcher internally runs reduce when nwg > 1
    // and writes the final output to `output_buf`. That output is in the
    // ROTATED domain; we now choose either:
    //   - fused path: re-run the reduce-with-undo on tmp_buf → output_buf
    //     (overwriting the prior reduce output)
    //   - unfused path: apply fwht_sign_undo on output_buf in-place
    //
    // Actually, since the SDPA dispatcher already ran the (non-fused) reduce
    // into output_buf, the parity comparison is:
    //   UNFUSED:  output_buf (already reduced)  →  fwht_sign_undo → final
    //   FUSED:    re-run reduce_tq_hb_undo on tmp_buf into output_buf,
    //             which produces the same final value directly.
    //
    // To exercise the fused path correctly, we re-dispatch on `tmp_buf`
    // (which still holds the per-WG partials because the unfused reduce only
    // reads from it). This matches the production wiring: forward_mlx.rs
    // gates the unfused reduce-then-undo vs the fused-reduce inside the
    // tq_hb dispatcher. For the test, however, the simplest A/B is:
    //   - Run SDPA into tmp_buf only (no reduce), then run our chosen reduce
    //     variant. This requires nwg > 1, which we have.
    //
    // The existing dispatcher always runs the unfused reduce. Rather than
    // duplicating the SDPA call without reduce, we use the fact that:
    //   - For UNFUSED parity: call fwht_sign_undo on output_buf in-place.
    //   - For FUSED parity:   re-run the fused reduce on tmp_buf into
    //     output_buf — `tmp_buf` was not modified by the unfused reduce.

    let mut enc = device.command_encoder().expect("enc2");
    if fused {
        flash_attn_vec_reduce_tq_hb_undo::dispatch_flash_attn_vec_reduce_tq_hb_undo(
            &mut enc, &mut registry, &device,
            &tmp_buf, &output_buf,
            nh, hd, 16, // nwg=16 matches compute_nwg(kvl=128)
        ).expect("dispatch fused reduce-undo");
    } else {
        fwht_standalone::dispatch_fwht_sign_undo_f32(
            &mut enc, &mut registry, device.metal_device(),
            &output_buf, nh, hd,
        ).expect("dispatch undo");
    }
    enc.commit_and_wait().expect("commit reduce/undo");

    if let Some(v) = prev_nwg_env { std::env::set_var("HF2Q_TQ_NWG", v); }

    output_buf.as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn reduce_tq_hb_undo_fused_vs_unfused_parity() {
    let unfused = run_path(0xc0ffee_u64, false);
    let fused   = run_path(0xc0ffee_u64, true);

    assert_eq!(unfused.len(), fused.len());

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&a, &b)) in unfused.iter().zip(fused.iter()).enumerate() {
        let abs = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(1e-8);
        let rel = abs / denom;
        if abs > max_abs { max_abs = abs; worst_idx = i; }
        if rel > max_rel { max_rel = rel; }
    }

    println!(
        "H3 fused vs unfused: max_abs={max_abs:.6e}  max_rel={max_rel:.6e}  worst_idx={worst_idx} \
         a={:.6e} b={:.6e}  nrmse={:.6e}",
        unfused[worst_idx], fused[worst_idx], nrmse(&unfused, &fused),
    );

    // 1e-4 abs/rel tolerance per spec.
    assert!(max_abs < 1e-4, "H3 parity: max_abs {max_abs:.6e} exceeds 1e-4");
    assert!(max_rel < 1e-4, "H3 parity: max_rel {max_rel:.6e} exceeds 1e-4");
}
