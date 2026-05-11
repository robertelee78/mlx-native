//! ADR-028 iter-368 — parity test for `kernel_mul_mv_q8_0_f32_nr2`.
//!
//! New kernel ports llama.cpp's `kernel_mul_mv_q8_0_f32_impl` with N_R0_Q8_0=2
//! and N_SG_Q8_0=4 (peer pattern) — 128 threads/TG, 2 rows/TG with cross-SG
//! reduction.  The original `kernel_mul_mv_q8_0_f32` uses 64 threads/TG, 8
//! rows/TG (each SG handles 4 different rows independently).
//!
//! Both kernels compute the same Q8_0 dequant × F32 dot product math.
//! Reduction order differs slightly (cross-SG sum vs single-SG sum) so we
//! allow rel_error < 1e-4.
//!
//! Falsifier: this test must PASS for the iter-368 perf claim to be
//! load-bearing.  If parity fails, the kernel is buggy.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice};

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
    buf
}

fn run_q8_0_mv(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    use_nr2: bool,
    weights_packed: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    if use_nr2 {
        std::env::set_var("HF2Q_Q8_0_MV_NR2", "1");
    } else {
        std::env::set_var("HF2Q_Q8_0_MV_NR2", "0");
    }

    let mut w_buf = device
        .alloc_buffer(weights_packed.len(), DType::U8, vec![weights_packed.len()])
        .expect("alloc w");
    w_buf.as_mut_slice::<u8>().expect("w").copy_from_slice(weights_packed);

    let mut i_buf = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("alloc i");
    i_buf.as_mut_slice::<f32>().expect("i").copy_from_slice(input);

    let mut o_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc o");

    let params = GgmlQuantizedMatmulParams {
        m: 1,
        k: k as u32,
        n: n as u32,
        ggml_type: GgmlType::Q8_0,
    };

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::quantized_matmul_ggml(
        &mut enc, registry, device,
        &i_buf, &w_buf, &mut o_buf, &params,
    ).expect("dispatch");
    enc.commit_and_wait().expect("commit");

    o_buf.as_slice::<f32>().expect("o").to_vec()
}

#[test]
fn parity_q8_0_mv_nr2_n128_k2048() {
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let n = 128;   // even multiple of 8 (existing) and 2 (NR2)
    let k = 2048;  // multiple of QK8_0=32

    let weights_f32 = pseudo_random_f32(0xCAFE, n * k);
    let mut packed = Vec::new();
    for row in 0..n {
        packed.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xBEEF, k);

    let baseline = run_q8_0_mv(&device, &mut registry, false, &packed, &input, n, k);
    let nr2      = run_q8_0_mv(&device, &mut registry, true,  &packed, &input, n, k);

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in baseline.iter().zip(nr2.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        assert!(rel < 1e-4,
            "i={i} baseline={a:.6e} nr2={b:.6e} abs={abs:.4e} rel={rel:.4e}");
    }
    println!("n=128 k=2048 max_abs={max_abs:.4e} max_rel={max_rel:.4e}");
}

#[test]
fn parity_q8_0_mv_nr2_n2816_k2816() {
    // gemma4 production hidden_size shape (2816 × 2816 — square attention proj).
    let device = MlxDevice::new().expect("dev");
    let mut registry = KernelRegistry::new();
    let n = 2816;
    let k = 2816;

    let weights_f32 = pseudo_random_f32(0xDEAD, n * k);
    let mut packed = Vec::new();
    for row in 0..n {
        packed.extend_from_slice(&pack_q8_0(&weights_f32[row * k..(row + 1) * k]));
    }
    let input = pseudo_random_f32(0xFACE, k);

    let baseline = run_q8_0_mv(&device, &mut registry, false, &packed, &input, n, k);
    let nr2      = run_q8_0_mv(&device, &mut registry, true,  &packed, &input, n, k);

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in baseline.iter().zip(nr2.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-6 { abs / a.abs() } else { abs };
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        assert!(rel < 1e-4,
            "i={i} baseline={a:.6e} nr2={b:.6e} abs={abs:.4e} rel={rel:.4e}");
    }
    println!("gemma4 n=2816 k=2816 max_abs={max_abs:.4e} max_rel={max_rel:.4e}");
}
