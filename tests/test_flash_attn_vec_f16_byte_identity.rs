//! ADR-028 iter-168: numerical bisect of F16 KV vs F32 KV in flash_attn_vec.
//!
//! ADR-009 (2026-04-16) measured F16 KV path producing 19× worse cache_k
//! drift and 45× worse sdpa_out drift vs llama.cpp baseline. Hypothesis:
//! the kernel has a layout/alignment bug, NOT a precision tradeoff — peer's
//! F16 KV is byte-identical to F32 KV (their `2327/2327`), ours drifts.
//!
//! This test exercises the f16kv kernel directly:
//!   1. Generate random F32 K/V
//!   2. Cast to F16 element-wise (lossy storage)
//!   3. Run F32 kernel against F32 buffers (baseline)
//!   4. Run F16 kernel against F16 buffers (path under test)
//!   5. Compute rel_rms between F32 and F16 GPU outputs
//!
//! Expected if KERNEL CORRECT (just F16 precision tradeoff):
//!   rel_rms ~ F16 ULP × accumulation depth = O(1e-3) per output element.
//!
//! Expected if KERNEL HAS BUG (per ADR-009 finding):
//!   rel_rms much larger than F16 epsilon — points to bug origin.

use half::f16;
use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Deterministic pseudo-random f32 in [-1, 1].
fn pseudo_random(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((x >> 33) as u32) & 0x7FFFFF;
    (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
}

fn fill_random(buf: &mut [f32], seed: u64) {
    for (i, val) in buf.iter_mut().enumerate() {
        *val = pseudo_random(seed + i as u64);
    }
}

/// Run flash_attn_vec with given KV dtype, return GPU output as Vec<f32>.
fn run_kernel(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    q_data: &[f32],
    k_data_f32: &[f32],
    v_data_f32: &[f32],
    kv_dtype: DType,
) -> Vec<f32> {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = num_heads as usize * head_dim as usize;
    let kv_elems = num_kv_heads as usize * kv_capacity as usize * head_dim as usize;

    let q_bytes = q_elems * 4;
    let kv_bytes_f32 = kv_elems * 4;
    let kv_bytes = kv_elems * kv_dtype.size_of();
    let out_bytes = q_elems * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elems])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, kv_dtype, vec![kv_elems])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, kv_dtype, vec![kv_elems])
        .expect("alloc V");
    let output_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc output");

    q_buf
        .as_mut_slice::<f32>()
        .expect("q slice")
        .copy_from_slice(q_data);

    match kv_dtype {
        DType::F32 => {
            assert_eq!(kv_bytes, kv_bytes_f32);
            k_buf
                .as_mut_slice::<f32>()
                .expect("k slice f32")
                .copy_from_slice(k_data_f32);
            v_buf
                .as_mut_slice::<f32>()
                .expect("v slice f32")
                .copy_from_slice(v_data_f32);
        }
        DType::F16 => {
            // Cast F32 source → F16 element-wise, write as u16 bits.
            let k_f16: Vec<u16> = k_data_f32
                .iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect();
            let v_f16: Vec<u16> = v_data_f32
                .iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect();
            k_buf
                .as_mut_slice::<u16>()
                .expect("k slice u16")
                .copy_from_slice(&k_f16);
            v_buf
                .as_mut_slice::<u16>()
                .expect("v slice u16")
                .copy_from_slice(&v_f16);
        }
        _ => panic!("unsupported KV dtype"),
    }

    let tmp_bytes = flash_attn_vec::tmp_buffer_bytes(num_heads, head_dim);
    let tmp_elems = tmp_bytes / 4;
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_elems])
        .expect("alloc tmp");

    let params = FlashAttnVecParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        softcap: 0.0,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec::flash_attn_vec(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &output_buf,
        &tmp_buf,
        &params,
    )
    .expect("flash_attn_vec dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    output_buf
        .as_slice::<f32>()
        .expect("output slice")
        .to_vec()
}

/// Compute relative RMS error: sqrt(sum((a-b)^2) / sum(b^2)).
fn rel_rms(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let d = (*a as f64) - (*e as f64);
        diff_sq += d * d;
        ref_sq += (*e as f64) * (*e as f64);
    }
    if ref_sq == 0.0 {
        return f32::INFINITY;
    }
    (diff_sq / ref_sq).sqrt() as f32
}

/// Round F32 → F16 → F32 element-wise (simulate F16 storage).
fn round_to_f16(buf: &[f32]) -> Vec<f32> {
    buf.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

/// dk256: gemma4 production head dim, sliding-window layer geometry.
#[test]
fn test_flash_attn_vec_dk256_f32_vs_f16_rel_rms() {
    let num_heads = 16u32;
    let num_kv_heads = 8u32;
    let head_dim = 256u32;
    let kv_seq_len = 240u32; // ADR-009's measured sliding regime
    let kv_capacity = 1024u32; // gemma4 sliding window
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_type = 2u32; // sliding window
    let sliding_window = 1024u32;
    let seed = 42u64;

    let q_elems = num_heads as usize * head_dim as usize;
    let kv_elems = num_kv_heads as usize * kv_capacity as usize * head_dim as usize;

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10000);
    fill_random(&mut v_data, seed + 20000);

    // F32 path: full precision baseline.
    let f32_output = run_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        &q_data,
        &k_data,
        &v_data,
        DType::F32,
    );

    // F32 path with F16-rounded inputs: this is the BEST-CASE F16 baseline.
    // If the F16 kernel matches THIS, the kernel is correct (precision-only
    // difference comes from the F16 storage, not the kernel math).
    let k_f16_rounded = round_to_f16(&k_data);
    let v_f16_rounded = round_to_f16(&v_data);
    let f32_with_f16_inputs_output = run_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        &q_data,
        &k_f16_rounded,
        &v_f16_rounded,
        DType::F32,
    );

    // F16 path: actual kernel under test.
    let f16_output = run_kernel(
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        &q_data,
        &k_data,
        &v_data,
        DType::F16,
    );

    let rms_f32_vs_f16inputs = rel_rms(&f32_with_f16_inputs_output, &f32_output);
    let rms_f16_vs_f32 = rel_rms(&f16_output, &f32_output);
    let rms_f16_vs_f32inputs = rel_rms(&f16_output, &f32_with_f16_inputs_output);

    eprintln!("F32 baseline ←→ F32-with-F16-inputs rel_rms: {rms_f32_vs_f16inputs:.6e}");
    eprintln!("F32 baseline ←→ F16-kernel rel_rms:          {rms_f16_vs_f32:.6e}");
    eprintln!("F32-with-F16-inputs ←→ F16-kernel rel_rms:   {rms_f16_vs_f32inputs:.6e}");

    // Diagnostic only (no assert): print the ratio.
    if rms_f32_vs_f16inputs > 0.0 {
        let amplification = rms_f16_vs_f32 / rms_f32_vs_f16inputs;
        eprintln!("F16-kernel amplification over F16-inputs: {amplification:.2}×");
    }

    // Pass condition: F16 kernel output should match F32-with-F16-inputs
    // baseline within F16 ULP × accumulation. A ratio < 5× means kernel
    // is correct; > 19× would reproduce ADR-009's reported amplification.
    assert!(
        rms_f16_vs_f32inputs < 1e-2,
        "F16 kernel diverges from F32-with-F16-inputs by {rms_f16_vs_f32inputs:.6e} \
         (threshold 1e-2). Amplification = {:.2}×",
        if rms_f32_vs_f16inputs > 0.0 {
            rms_f16_vs_f32inputs / rms_f32_vs_f16inputs
        } else {
            f32::INFINITY
        }
    );
}
