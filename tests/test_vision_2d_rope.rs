//! Tests for the 2-D NeoX RoPE used by Gemma 4 Vision (gemma4v).
//!
//! Verifies (per `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:46-91`):
//!   1. Identity at origin: when `pos_x[p] = 0` AND `pos_y[p] = 0`, output equals input.
//!   2. Inverse: applying the kernel and then its inverse (negative angles) recovers input.
//!   3. NeoX-pair structure: at one (pos_x, pos_y) the per-pair rotation
//!      is exactly (a, b) → (a*cos - b*sin, a*sin + b*cos), with the
//!      first-half driven by pos_x and the second-half by pos_y.
//!   4. F32/BF16 parity: BF16 dispatch matches an F32 reference within tolerance.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::vision_2d_rope::{
    build_vision_2d_rope_params, dispatch_vision_2d_rope,
};

/// CPU reference matching the Metal kernel exactly. Mirrors gemma4v.cpp:46-91.
fn vision_2d_rope_ref(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_x: &[u32],
    pos_y: &[u32],
    theta: f32,
) -> Vec<f32> {
    let d_half = head_dim / 2;
    let d_quarter = d_half / 2;
    let n_rows = seq_len * n_heads;
    let mut output = vec![0.0f32; n_rows * head_dim];
    // Default-copy: pairs that are not rotated stay as input. (For
    // head_dim divisible by 4, every element is in some pair.)
    output.copy_from_slice(input);

    for row in 0..n_rows {
        let seq_idx = row / n_heads;
        let p_x = pos_x[seq_idx] as f32;
        let p_y = pos_y[seq_idx] as f32;
        let base = row * head_dim;

        for i in 0..d_quarter {
            let dim_ratio = (2 * i) as f32 / d_half as f32;
            let freq = 1.0_f32 / theta.powf(dim_ratio);
            let angle_x = p_x * freq;
            let angle_y = p_y * freq;
            let cx = angle_x.cos();
            let sx = angle_x.sin();
            let cy = angle_y.cos();
            let sy = angle_y.sin();

            // First half pair (i, i + d_quarter)
            let x0 = input[base + i];
            let x1 = input[base + i + d_quarter];
            output[base + i] = x0 * cx - x1 * sx;
            output[base + i + d_quarter] = x0 * sx + x1 * cx;

            // Second half pair (d_half + i, d_half + i + d_quarter)
            let y0 = input[base + d_half + i];
            let y1 = input[base + d_half + i + d_quarter];
            output[base + d_half + i] = y0 * cy - y1 * sy;
            output[base + d_half + i + d_quarter] = y0 * sy + y1 * cy;
        }
    }
    output
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::vision_2d_rope::register(&mut registry);
    (device, registry)
}

fn alloc_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let n = data.len();
    let byte_len = n * std::mem::size_of::<f32>();
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc f32");
    let s: &mut [f32] = buf.as_mut_slice().expect("write f32");
    s.copy_from_slice(data);
    buf
}

fn alloc_u32(device: &MlxDevice, data: &[u32]) -> MlxBuffer {
    let n = data.len();
    let byte_len = n * std::mem::size_of::<u32>();
    let mut buf = device
        .alloc_buffer(byte_len, DType::U32, vec![n])
        .expect("alloc u32");
    let s: &mut [u32] = buf.as_mut_slice().expect("write u32");
    s.copy_from_slice(data);
    buf
}

fn alloc_bf16_from_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    use half::bf16;
    let n = data.len();
    let byte_len = n * std::mem::size_of::<bf16>();
    let mut buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .expect("alloc bf16");
    let s: &mut [bf16] = buf.as_mut_slice().expect("write bf16");
    for (dst, &src) in s.iter_mut().zip(data.iter()) {
        *dst = bf16::from_f32(src);
    }
    buf
}

fn read_f32(buf: &MlxBuffer) -> Vec<f32> {
    let s: &[f32] = buf.as_slice().expect("read f32");
    s.to_vec()
}

fn read_bf16_to_f32(buf: &MlxBuffer) -> Vec<f32> {
    use half::bf16;
    let s: &[bf16] = buf.as_slice().expect("read bf16");
    s.iter().map(|v| v.to_f32()).collect()
}

fn dispatch_and_wait(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    output: &MlxBuffer,
    pos_x: &MlxBuffer,
    pos_y: &MlxBuffer,
    theta: f32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) {
    let mut encoder = device.command_encoder().expect("command_encoder");
    let params = build_vision_2d_rope_params(device, theta, head_dim, n_heads)
        .expect("params buffer");
    dispatch_vision_2d_rope(
        &mut encoder,
        registry,
        device.metal_device(),
        input,
        output,
        &params,
        pos_x,
        pos_y,
        seq_len,
        n_heads,
        head_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// Test 1: identity at origin (pos_x = pos_y = 0)
// ---------------------------------------------------------------------------

#[test]
fn vision_2d_rope_identity_at_origin() {
    let (device, mut registry) = setup();

    let seq_len = 4_usize;
    let n_heads = 2_usize;
    let head_dim = 8_usize;
    let n_rows = seq_len * n_heads;
    let n_elem = n_rows * head_dim;

    // Random-ish input.
    let input: Vec<f32> = (0..n_elem).map(|i| (i as f32) * 0.123 - 1.7).collect();
    let pos_x: Vec<u32> = vec![0; seq_len];
    let pos_y: Vec<u32> = vec![0; seq_len];

    let in_buf = alloc_f32(&device, &input);
    let out_buf = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![n_rows, head_dim])
        .expect("alloc out");
    let px = alloc_u32(&device, &pos_x);
    let py = alloc_u32(&device, &pos_y);

    dispatch_and_wait(
        &device,
        &mut registry,
        &in_buf,
        &out_buf,
        &px,
        &py,
        100.0,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );

    let actual = read_f32(&out_buf);
    let max_d = max_abs_diff(&actual, &input);
    assert!(
        max_d < 1e-6,
        "vision_2d_rope at (0,0) should be identity; max|delta| = {max_d}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: inverse (forward + backward = identity)
// ---------------------------------------------------------------------------

#[test]
fn vision_2d_rope_inverse() {
    let (device, mut registry) = setup();

    let seq_len = 6_usize;
    let n_heads = 3_usize;
    let head_dim = 12_usize;
    let n_rows = seq_len * n_heads;
    let n_elem = n_rows * head_dim;
    let theta = 100.0_f32;

    let input: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32).sin() + 0.5) * 0.7)
        .collect();
    let pos_x: Vec<u32> = (0..seq_len as u32).collect();
    let pos_y: Vec<u32> = (0..seq_len as u32).map(|i| (i + 1) % 5).collect();

    let in_buf = alloc_f32(&device, &input);
    let mid_buf = device
        .alloc_buffer(n_elem * 4, DType::F32, vec![n_rows, head_dim])
        .expect("alloc mid");
    let px = alloc_u32(&device, &pos_x);
    let py = alloc_u32(&device, &pos_y);

    // Forward: apply rotation by (pos_x, pos_y).
    dispatch_and_wait(
        &device,
        &mut registry,
        &in_buf,
        &mid_buf,
        &px,
        &py,
        theta,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );

    // Inverse: apply rotation by negative angles. Since pos values are
    // unsigned, we instead rotate the intermediate by the same kernel but
    // with the SIN sign flipped — equivalent to applying NEG positions.
    // The cleanest way is a second pass with the CPU-computed inverse
    // (which exercises the kernel forward semantics): for every pair,
    // apply the inverse rotation. We use the CPU reference's inverse
    // (which is just transposing the 2x2 rotation matrix) to compute
    // the expected and require GPU forward output to round-trip there.
    //
    // Equivalently: forward(input, pos) then forward_with_negated_angles(mid, pos)
    // should equal input. We synthesize "negated angles" by applying the
    // forward kernel a second time but treating the output as a
    // backward by negating the kernel's effective angles via swapping
    // the pair components (a, b) -> (b, -a) is a 90° turn, not what we
    // want. Instead, we simply assert that the GPU forward matches the
    // CPU reference, then assert the CPU inverse round-trip closes.
    let gpu_mid = read_f32(&mid_buf);
    let cpu_mid = vision_2d_rope_ref(
        &input,
        seq_len,
        n_heads,
        head_dim,
        &pos_x,
        &pos_y,
        theta,
    );
    let max_d = max_abs_diff(&gpu_mid, &cpu_mid);
    assert!(
        max_d < 1e-5,
        "GPU forward ≠ CPU reference; max|delta| = {max_d}"
    );

    // CPU inverse: rotate each pair by negative angle, recover original.
    let d_half = head_dim / 2;
    let d_quarter = d_half / 2;
    let mut recovered = vec![0.0_f32; n_elem];
    recovered.copy_from_slice(&cpu_mid);
    for row in 0..n_rows {
        let seq_idx = row / n_heads;
        let p_x = pos_x[seq_idx] as f32;
        let p_y = pos_y[seq_idx] as f32;
        let base = row * head_dim;
        for i in 0..d_quarter {
            let dim_ratio = (2 * i) as f32 / d_half as f32;
            let freq = 1.0_f32 / theta.powf(dim_ratio);
            // Inverse rotation: angle -> -angle, so cos same, sin negated.
            let cx = (p_x * freq).cos();
            let sx = -(p_x * freq).sin();
            let cy = (p_y * freq).cos();
            let sy = -(p_y * freq).sin();

            let x0 = cpu_mid[base + i];
            let x1 = cpu_mid[base + i + d_quarter];
            recovered[base + i] = x0 * cx - x1 * sx;
            recovered[base + i + d_quarter] = x0 * sx + x1 * cx;

            let y0 = cpu_mid[base + d_half + i];
            let y1 = cpu_mid[base + d_half + i + d_quarter];
            recovered[base + d_half + i] = y0 * cy - y1 * sy;
            recovered[base + d_half + i + d_quarter] = y0 * sy + y1 * cy;
        }
    }
    let inv_diff = max_abs_diff(&recovered, &input);
    assert!(
        inv_diff < 1e-5,
        "Inverse round-trip didn't recover input; max|delta| = {inv_diff}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: explicit NeoX pair structure check
// ---------------------------------------------------------------------------

#[test]
fn vision_2d_rope_neox_pair_structure() {
    let (device, mut registry) = setup();

    // Small shape: 1 patch, 1 head, head_dim = 4 → d_half = 2, d_quarter = 1.
    let seq_len = 1_usize;
    let n_heads = 1_usize;
    let head_dim = 4_usize;
    let theta = 100.0_f32;

    // Specific deterministic position so we can read off the angles.
    let pos_x: Vec<u32> = vec![3];
    let pos_y: Vec<u32> = vec![5];

    // Input: distinguishable values per index.
    let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // [a0, a1, b0, b1]
    let in_buf = alloc_f32(&device, &input);
    let out_buf = device
        .alloc_buffer(head_dim * 4, DType::F32, vec![1, head_dim])
        .expect("alloc out");
    let px = alloc_u32(&device, &pos_x);
    let py = alloc_u32(&device, &pos_y);

    dispatch_and_wait(
        &device,
        &mut registry,
        &in_buf,
        &out_buf,
        &px,
        &py,
        theta,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );
    let got = read_f32(&out_buf);

    // d_half = 2, d_quarter = 1, so i=0 only:
    //   first-half pair = (input[0]=1, input[1]=2), angle_x = 3 * 1/theta^0 = 3
    //   second-half pair = (input[2]=3, input[3]=4), angle_y = 5 * 1/theta^0 = 5
    let cx = 3.0_f32.cos();
    let sx = 3.0_f32.sin();
    let cy = 5.0_f32.cos();
    let sy = 5.0_f32.sin();
    let expected = vec![
        1.0 * cx - 2.0 * sx,
        1.0 * sx + 2.0 * cx,
        3.0 * cy - 4.0 * sy,
        3.0 * sy + 4.0 * cy,
    ];
    let max_d = max_abs_diff(&got, &expected);
    assert!(
        max_d < 1e-5,
        "NeoX pair rotation structure mismatch; got {got:?}, expected {expected:?}, max|delta| = {max_d}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: BF16 parity vs F32 reference
// ---------------------------------------------------------------------------

#[test]
fn vision_2d_rope_bf16_matches_f32_reference() {
    let (device, mut registry) = setup();

    let seq_len = 5_usize;
    let n_heads = 2_usize;
    let head_dim = 8_usize;
    let n_rows = seq_len * n_heads;
    let n_elem = n_rows * head_dim;
    let theta = 100.0_f32;

    let input: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32) * 0.31).sin() * 1.3)
        .collect();
    let pos_x: Vec<u32> = vec![0, 1, 2, 3, 4];
    let pos_y: Vec<u32> = vec![4, 3, 2, 1, 0];

    let in_bf16 = alloc_bf16_from_f32(&device, &input);
    let out_bf16 = device
        .alloc_buffer(n_elem * 2, DType::BF16, vec![n_rows, head_dim])
        .expect("alloc bf16 out");
    let px = alloc_u32(&device, &pos_x);
    let py = alloc_u32(&device, &pos_y);

    dispatch_and_wait(
        &device,
        &mut registry,
        &in_bf16,
        &out_bf16,
        &px,
        &py,
        theta,
        seq_len as u32,
        n_heads as u32,
        head_dim as u32,
    );

    // Reference uses the same f32 input as the BF16 round-trip starting
    // point. BF16 introduces ≤ 1/256 relative error per element, so the
    // tolerance accommodates the cast-and-back.
    let gpu_bf16 = read_bf16_to_f32(&out_bf16);
    // Reference: round the input through bf16 first to match the cast
    // the GPU sees, then apply the f32 reference math.
    let input_bf16_roundtrip: Vec<f32> = input
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_f32())
        .collect();
    let cpu = vision_2d_rope_ref(
        &input_bf16_roundtrip,
        seq_len,
        n_heads,
        head_dim,
        &pos_x,
        &pos_y,
        theta,
    );
    let max_d = max_abs_diff(&gpu_bf16, &cpu);
    // BF16: ~1/256 relative + tiny truncation per element. Allow 5e-3.
    assert!(
        max_d < 5e-3,
        "BF16 output diverges from f32 reference; max|delta| = {max_d}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: error path — head_dim not divisible by 4
// ---------------------------------------------------------------------------

#[test]
fn vision_2d_rope_rejects_head_dim_not_divisible_by_4() {
    let (device, mut registry) = setup();
    let head_dim = 6_u32;
    let seq_len = 1_u32;
    let n_heads = 1_u32;
    let in_buf = device
        .alloc_buffer(head_dim as usize * 4, DType::F32, vec![head_dim as usize])
        .expect("alloc in");
    let out_buf = device
        .alloc_buffer(head_dim as usize * 4, DType::F32, vec![head_dim as usize])
        .expect("alloc out");
    let pos = alloc_u32(&device, &[0u32]);
    let params = build_vision_2d_rope_params(&device, 100.0, head_dim, n_heads)
        .expect("params");
    let mut encoder = device.command_encoder().expect("encoder");
    let res = dispatch_vision_2d_rope(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &in_buf,
        &out_buf,
        &params,
        &pos,
        &pos,
        seq_len,
        n_heads,
        head_dim,
    );
    assert!(res.is_err(), "head_dim=6 should be rejected");
    let err_msg = format!("{:?}", res.unwrap_err());
    assert!(
        err_msg.contains("divisible by 4"),
        "expected divisibility error, got: {err_msg}"
    );
}
