//! ADR-028 iter-218: parity test for `fused_post_ff_norm2_endlayer_f32`.
//!
//! Verifies the new fused kernel produces byte-identical (or within f32
//! noise) output vs the sequential 2-kernel pipeline:
//!   (a) fused_norm_add_f32(attn_out, moe_accum, w2)        → mlp_down
//!   (b) fused_norm_add_scalar_f32(residual, mlp_down, w3, scalar) → hidden
//!
//! Bisect-confirmed +2.7% lever (iter-208).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, KernelRegistry, MlxDevice};
use mlx_native::ops::{fused_norm_add, rms_norm};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    fused_norm_add::register(&mut registry);
    rms_norm::register(&mut registry);
    (device, registry)
}

fn alloc_f32(device: &MlxDevice, data: &[f32]) -> mlx_native::MlxBuffer {
    let byte_len = data.len() * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![data.len()])
        .expect("alloc f32");
    buf.as_mut_slice::<f32>().expect("slice").copy_from_slice(data);
    buf
}

#[test]
fn fused_post_ff_norm2_endlayer_byte_identity_vs_sequential() {
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;
    let rows: u32 = 1;
    let dim: u32 = 2816; // gemma4 hidden_size
    let n = (rows as usize) * (dim as usize);

    // Deterministic inputs (same as production decode shape).
    let attn_out: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013 - 1.7).sin()).collect();
    let moe_accum: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017 + 0.4).cos()).collect();
    let residual: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011 - 2.1).sin()).collect();
    let w2: Vec<f32> = (0..dim as usize).map(|i| 1.0 + ((i as f32) * 0.05).sin() * 0.1).collect();
    let w3: Vec<f32> = (0..dim as usize).map(|i| 1.0 + ((i as f32) * 0.07).cos() * 0.1).collect();
    let layer_scalar: Vec<f32> = vec![0.95];

    // SEQUENTIAL reference (2 dispatches).
    let attn_out_buf = alloc_f32(&device, &attn_out);
    let moe_accum_buf = alloc_f32(&device, &moe_accum);
    let residual_buf = alloc_f32(&device, &residual);
    let w2_buf = alloc_f32(&device, &w2);
    let w3_buf = alloc_f32(&device, &w3);
    let scalar_buf = alloc_f32(&device, &layer_scalar);
    let mlp_down_seq_buf = alloc_f32(&device, &vec![0.0; n]);
    let hidden_seq_buf = alloc_f32(&device, &vec![0.0; n]);

    let metal_dev = device.metal_device();
    {
        let mut enc = device.command_encoder().expect("encoder");
        // (a) mlp_down = attn_out + norm(moe_accum, w2)
        fused_norm_add::dispatch_fused_norm_add_f32(
            &mut enc, &mut registry, metal_dev,
            &attn_out_buf, &moe_accum_buf, &w2_buf, &mlp_down_seq_buf,
            dim, rows, eps,
        ).expect("seq fused_norm_add");
        enc.memory_barrier();
        // (b) hidden = (residual + norm(mlp_down, w3)) * scalar
        fused_norm_add::dispatch_fused_norm_add_scalar_f32(
            &mut enc, &mut registry, metal_dev,
            &residual_buf, &mlp_down_seq_buf, &w3_buf, &hidden_seq_buf,
            &scalar_buf,
            rows, dim, eps,
            false, // scalar_is_vector
        ).expect("seq fused_norm_add_scalar");
        enc.commit_and_wait().expect("seq commit");
    }

    let mlp_down_seq: Vec<f32> = mlp_down_seq_buf
        .as_slice::<f32>().expect("seq mlp_down slice").to_vec();
    let hidden_seq: Vec<f32> = hidden_seq_buf
        .as_slice::<f32>().expect("seq hidden slice").to_vec();

    // FUSED kernel (1 dispatch).
    let mlp_down_fused_buf = alloc_f32(&device, &vec![0.0; n]);
    let hidden_fused_buf = alloc_f32(&device, &vec![0.0; n]);

    {
        let mut enc = device.command_encoder().expect("encoder fused");
        rms_norm::dispatch_fused_post_ff_norm2_endlayer_f32(
            &mut enc, &mut registry, metal_dev,
            &attn_out_buf, &moe_accum_buf, &residual_buf,
            &w2_buf, &w3_buf, &scalar_buf,
            &mlp_down_fused_buf, &hidden_fused_buf,
            eps, rows, dim,
            false, // scalar_is_vector
        ).expect("fused dispatch");
        enc.commit_and_wait().expect("fused commit");
    }

    let mlp_down_fused: Vec<f32> = mlp_down_fused_buf
        .as_slice::<f32>().expect("fused mlp_down").to_vec();
    let hidden_fused: Vec<f32> = hidden_fused_buf
        .as_slice::<f32>().expect("fused hidden").to_vec();

    // Compare. Both kernels do identical math (add → norm → multiply →
    // norm → multiply → add → scalar mul) so should be byte-identical.
    // Allow tiny f32 noise from FMA reordering.
    let mut max_abs_mlp = 0.0f32;
    let mut max_abs_hidden = 0.0f32;
    for i in 0..n {
        max_abs_mlp = max_abs_mlp.max((mlp_down_fused[i] - mlp_down_seq[i]).abs());
        max_abs_hidden = max_abs_hidden.max((hidden_fused[i] - hidden_seq[i]).abs());
    }
    let max_ref_mlp = mlp_down_seq.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
    let max_ref_hidden = hidden_seq.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
    let rel_mlp = max_abs_mlp / max_ref_mlp;
    let rel_hidden = max_abs_hidden / max_ref_hidden;

    eprintln!("[parity] mlp_down  max_abs={max_abs_mlp:.3e}  rel={rel_mlp:.3e}");
    eprintln!("[parity] hidden    max_abs={max_abs_hidden:.3e}  rel={rel_hidden:.3e}");
    assert!(rel_mlp < 1e-5, "mlp_down rel error {rel_mlp:.3e} > 1e-5");
    assert!(rel_hidden < 1e-5, "hidden rel error {rel_hidden:.3e} > 1e-5");
}
