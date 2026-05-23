//! ADR-033 §Pi Task #20 debug — instrumented Metal kernel debug for
//! `dequantize_iq4_xs`.
//!
//! Calls `hf2q_dequant_iq4_xs_dump` (standalone debug kernel that
//! exists ONLY for this test) to dump the per-il output of
//! `dequantize_iq4_xs` for one block. Compares to the canonical CPU
//! reference (`test_only_dequantize_iq4_xs`). Localizes whether the
//! mm_id top_k=8 bug is in the dequant function or in the mm_id
//! template's interaction with it.
//!
//! - If THIS test PASSES → dequantize_iq4_xs is correct → bug is in
//!   the mm_id kernel template's tile/staging at top_k=8.
//! - If THIS test FAILS → dequantize_iq4_xs has a bug → fix the
//!   dequant.

use mlx_native::gguf::{test_only_dequantize_iq4_xs, test_only_kvalues_iq4_nl};
use mlx_native::{DType, KernelArg, KernelRegistry, MlxBuffer, MlxDevice};

const QK_K: usize = 256;
const BLOCK_IQ4_XS_BYTES: usize = 136;
const SUB: usize = 32;

fn xs64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 13;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545F4914F6CDD1D)
}

fn random_pm1(state: &mut u64) -> f32 {
    let bits = xs64(state);
    ((bits >> 11) as f32) / (1u64 << 53) as f32 * 2.0 - 1.0
}

/// Same IQ4_XS reference quantizer as the mv_id/mm_id parity tests.
fn ref_quantize_iq4_xs(row: &[f32]) -> Vec<u8> {
    let kv = test_only_kvalues_iq4_nl();
    assert_eq!(row.len(), QK_K);
    let mut out = Vec::with_capacity(BLOCK_IQ4_XS_BYTES);
    let mut sub_scales = [0.0f32; 8];
    let mut max_scale: f32 = 0.0;
    for ib in 0..8 {
        let sub = &row[ib * SUB..(ib + 1) * SUB];
        let amax = sub.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let d_sub = if amax == 0.0 { 0.0 } else { -amax / kv[0] as f32 };
        sub_scales[ib] = d_sub;
        if d_sub.abs() > max_scale.abs() {
            max_scale = d_sub;
        }
    }
    let d = -max_scale / 32.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut scales_h: u16 = 0;
    let mut scales_l = [0u8; 4];
    let mut qs = [0u8; QK_K / 2];

    let nearest_codebook = |t: f32| -> u8 {
        let mut best_idx: u8 = 0;
        let mut best_err = f32::MAX;
        for (i, &k) in kv.iter().enumerate() {
            let e = (t - k as f32).abs();
            if e < best_err {
                best_err = e;
                best_idx = i as u8;
            }
        }
        best_idx
    };

    for ib in 0..8 {
        let l_raw = (id * sub_scales[ib]).round() as i32;
        let l_signed = l_raw.clamp(-32, 31);
        let dl = d * (l_signed as f32);
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        let sub_chunk = &row[ib * SUB..(ib + 1) * SUB];
        let mut l_buf = [0u8; SUB];
        for j in 0..SUB {
            l_buf[j] = nearest_codebook(idl * sub_chunk[j]);
        }
        let qs_sub = &mut qs[16 * ib..16 * (ib + 1)];
        for j in 0..16 {
            qs_sub[j] = l_buf[j] | (l_buf[16 + j] << 4);
        }
        let l_unsigned = (l_signed + 32) as u8;
        let l_l = l_unsigned & 0xf;
        let l_h = l_unsigned >> 4;
        if ib % 2 == 0 {
            scales_l[ib / 2] = l_l;
        } else {
            scales_l[ib / 2] |= l_l << 4;
        }
        scales_h |= (l_h as u16) << (2 * ib);
    }
    out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out.extend_from_slice(&scales_h.to_le_bytes());
    out.extend_from_slice(&scales_l);
    out.extend_from_slice(&qs);
    assert_eq!(out.len(), BLOCK_IQ4_XS_BYTES);
    out
}

#[test]
fn adr033_pi_task20_dequant_iq4_xs_dump_matches_cpu() {
    // Build one IQ4_XS block from synthetic data with varied magnitudes.
    let mut state = 0xDEAD_BEEF_DECAFu64;
    let mut row = vec![0.0_f32; QK_K];
    for v in row.iter_mut() {
        *v = random_pm1(&mut state) * 0.5;
    }
    let block_bytes = ref_quantize_iq4_xs(&row);
    assert_eq!(block_bytes.len(), BLOCK_IQ4_XS_BYTES);

    // GPU dequant via debug kernel.
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let pipeline = registry
        .get_pipeline("hf2q_dequant_iq4_xs_dump", device.metal_device())
        .expect("compile hf2q_dequant_iq4_xs_dump");

    let mut src_buf: MlxBuffer = device
        .alloc_buffer(BLOCK_IQ4_XS_BYTES, DType::U8, vec![BLOCK_IQ4_XS_BYTES])
        .unwrap();
    src_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&block_bytes);

    let mut dst_buf: MlxBuffer = device
        .alloc_buffer(QK_K * 4, DType::F32, vec![QK_K])
        .unwrap();

    let mut encoder = device.command_encoder().unwrap();
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(&src_buf)),
            (1, KernelArg::Buffer(&dst_buf)),
        ],
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(1, 1, 1),
    );
    encoder.commit_and_wait().unwrap();

    let gpu_out: &[f32] = dst_buf.as_slice().unwrap();
    assert_eq!(gpu_out.len(), QK_K);

    // CPU reference — full block of 256 elements.
    let mut cpu_out = vec![0.0_f32; QK_K];
    test_only_dequantize_iq4_xs(&block_bytes, &mut cpu_out).unwrap();

    // Compare. Both should produce IDENTICAL outputs (kernel and CPU
    // ref are both pure functions of the block bytes with no rounding
    // between them at the f32 level). Tolerance ~1e-6 for f32
    // half→float round-trip.
    let mut max_abs = 0.0_f32;
    for il in 0..16 {
        let ib32 = il / 2;
        let half = il % 2;
        // GPU dumps in linear order [il, j] for il=0..16, j=0..16.
        // Our CPU reference produces full 256-element dequant, where
        // sub-block ib32 occupies elements [ib32*32 .. (ib32+1)*32).
        // Within each sub-block: low half (il%2==0) = elements [0..16),
        // high half (il%2==1) = elements [16..32).
        for j in 0..16 {
            let gpu_idx = il * 16 + j;
            let cpu_idx = ib32 * 32 + half * 16 + j;
            let gpu_v = gpu_out[gpu_idx];
            let cpu_v = cpu_out[cpu_idx];
            let err = (gpu_v - cpu_v).abs();
            if err > max_abs {
                max_abs = err;
            }
            // f16-precision rounding noise ~1e-5 — the dequant kernel
            // writes to half4x4 (f16 intermediate) which round-trips
            // back to f32 with up to 2^-12 relative error per element.
            assert!(
                err < 1e-3,
                "il={il} ib32={ib32} half={half} j={j} gpu_idx={gpu_idx} cpu_idx={cpu_idx} \
                 GPU {gpu_v} vs CPU {cpu_v} (err {err})"
            );
        }
    }
    eprintln!(
        "[adr-033 §Pi Task #20 dequant dump] dequantize_iq4_xs matches CPU \
         reference for all 16 il calls — max_abs_err={max_abs:.2e}"
    );
}
