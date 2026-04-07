//! Tests for the MoE dispatch (expert FFN with routing) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};
use mlx_native::ops::moe_dispatch::{moe_dispatch, ExpertWeights, MoeDispatchParams};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Reference GELU approximation matching the shader.
fn gelu_ref(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    0.5 * x * (1.0 + inner.tanh())
}

/// Reference expert FFN: gate_proj + up_proj -> GELU -> down_proj
fn expert_ffn_ref(
    input: &[f32],
    gate_proj: &[f32],  // [intermediate_dim, input_dim] row-major
    up_proj: &[f32],    // [intermediate_dim, input_dim] row-major
    down_proj: &[f32],  // [input_dim, intermediate_dim] row-major
    input_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    // gate_out = gate_proj @ input
    let mut gate_out = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        for k in 0..input_dim {
            gate_out[i] += gate_proj[i * input_dim + k] * input[k];
        }
    }

    // up_out = up_proj @ input
    let mut up_out = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        for k in 0..input_dim {
            up_out[i] += up_proj[i * input_dim + k] * input[k];
        }
    }

    // hidden = GELU(gate_out) * up_out
    let mut hidden = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        hidden[i] = gelu_ref(gate_out[i]) * up_out[i];
    }

    // output = down_proj @ hidden
    let mut output = vec![0.0f32; input_dim];
    for i in 0..input_dim {
        for k in 0..intermediate_dim {
            output[i] += down_proj[i * intermediate_dim + k] * hidden[k];
        }
    }

    output
}

#[test]
fn test_moe_dispatch_single_expert() {
    let (device, mut registry) = setup();

    let input_dim = 8;
    let intermediate_dim = 4;

    // Simple input
    let input_data: Vec<f32> = (0..input_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Expert weights — small known values
    let gate_data: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) - 16.0) * 0.01)
        .collect();
    let up_data: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) - 8.0) * 0.02)
        .collect();
    let down_data: Vec<f32> = (0..input_dim * intermediate_dim)
        .map(|i| ((i as f32) - 12.0) * 0.015)
        .collect();

    let routing_weight = 1.0f32;

    // Reference computation
    let ref_output = expert_ffn_ref(
        &input_data, &gate_data, &up_data, &down_data,
        input_dim, intermediate_dim,
    );
    // Apply routing weight
    let ref_output: Vec<f32> = ref_output.iter().map(|&v| v * routing_weight).collect();

    // Create GPU buffers
    let mut input_buf = device
        .alloc_buffer(input_dim * 4, DType::F32, vec![input_dim])
        .expect("input");
    input_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&input_data);

    let mut gate_buf = device
        .alloc_buffer(gate_data.len() * 4, DType::F32, vec![intermediate_dim, input_dim])
        .expect("gate");
    gate_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&gate_data);

    let mut up_buf = device
        .alloc_buffer(up_data.len() * 4, DType::F32, vec![intermediate_dim, input_dim])
        .expect("up");
    up_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&up_data);

    let mut down_buf = device
        .alloc_buffer(down_data.len() * 4, DType::F32, vec![input_dim, intermediate_dim])
        .expect("down");
    down_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&down_data);

    let output_buf = device
        .alloc_buffer(input_dim * 4, DType::F32, vec![input_dim])
        .expect("output");

    // Scratch buffers
    let scratch_gate = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("sg");
    let scratch_up = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("su");
    let scratch_hidden = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("sh");
    let scratch_expert = device.alloc_buffer(input_dim * 4, DType::F32, vec![input_dim]).expect("se");

    let experts = vec![ExpertWeights {
        gate_proj: &gate_buf,
        up_proj: &up_buf,
        down_proj: &down_buf,
    }];

    let mut encoder = device.command_encoder().expect("encoder");
    moe_dispatch(
        &mut encoder, &mut registry, device.metal_device(),
        &input_buf, &experts, &[routing_weight], &output_buf,
        &scratch_gate, &scratch_up, &scratch_hidden, &scratch_expert,
        &MoeDispatchParams {
            input_dim,
            intermediate_dim,
            n_selected: 1,
        },
    ).expect("moe_dispatch");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = output_buf.as_slice().expect("read");
    for i in 0..input_dim {
        let diff = (output[i] - ref_output[i]).abs();
        assert!(
            diff < 1e-3,
            "single expert mismatch at {}: expected {}, got {}, diff {}",
            i, ref_output[i], output[i], diff
        );
    }
}

#[test]
fn test_moe_dispatch_two_experts_weighted() {
    let (device, mut registry) = setup();

    let input_dim = 8;
    let intermediate_dim = 4;

    let input_data: Vec<f32> = (0..input_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Expert 0 weights
    let gate0: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) - 10.0) * 0.01)
        .collect();
    let up0: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) + 5.0) * 0.01)
        .collect();
    let down0: Vec<f32> = (0..input_dim * intermediate_dim)
        .map(|i| ((i as f32) - 20.0) * 0.005)
        .collect();

    // Expert 1 weights
    let gate1: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) * 0.5 - 8.0) * 0.01)
        .collect();
    let up1: Vec<f32> = (0..intermediate_dim * input_dim)
        .map(|i| ((i as f32) * 0.3 + 2.0) * 0.02)
        .collect();
    let down1: Vec<f32> = (0..input_dim * intermediate_dim)
        .map(|i| ((i as f32) - 15.0) * 0.01)
        .collect();

    let w0 = 0.7f32;
    let w1 = 0.3f32;

    // Reference: weighted sum of expert outputs
    let out0 = expert_ffn_ref(&input_data, &gate0, &up0, &down0, input_dim, intermediate_dim);
    let out1 = expert_ffn_ref(&input_data, &gate1, &up1, &down1, input_dim, intermediate_dim);
    let ref_output: Vec<f32> = (0..input_dim)
        .map(|i| w0 * out0[i] + w1 * out1[i])
        .collect();

    // Create GPU buffers
    let mut input_buf = device.alloc_buffer(input_dim * 4, DType::F32, vec![input_dim]).expect("in");
    input_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&input_data);

    let create_weight_buf = |data: &[f32], shape: Vec<usize>| -> mlx_native::MlxBuffer {
        let mut buf = device.alloc_buffer(data.len() * 4, DType::F32, shape).expect("wb");
        buf.as_mut_slice::<f32>().expect("w").copy_from_slice(data);
        buf
    };

    let mut gate0_buf = create_weight_buf(&gate0, vec![intermediate_dim, input_dim]);
    let mut up0_buf = create_weight_buf(&up0, vec![intermediate_dim, input_dim]);
    let mut down0_buf = create_weight_buf(&down0, vec![input_dim, intermediate_dim]);
    let mut gate1_buf = create_weight_buf(&gate1, vec![intermediate_dim, input_dim]);
    let mut up1_buf = create_weight_buf(&up1, vec![intermediate_dim, input_dim]);
    let mut down1_buf = create_weight_buf(&down1, vec![input_dim, intermediate_dim]);

    // Suppress unused_mut warnings — buffers need to be mutable for as_mut_slice
    let _ = (&mut gate0_buf, &mut up0_buf, &mut down0_buf);
    let _ = (&mut gate1_buf, &mut up1_buf, &mut down1_buf);

    let output_buf = device.alloc_buffer(input_dim * 4, DType::F32, vec![input_dim]).expect("out");
    let scratch_gate = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("sg");
    let scratch_up = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("su");
    let scratch_hidden = device.alloc_buffer(intermediate_dim * 4, DType::F32, vec![intermediate_dim]).expect("sh");
    let scratch_expert = device.alloc_buffer(input_dim * 4, DType::F32, vec![input_dim]).expect("se");

    let experts = vec![
        ExpertWeights { gate_proj: &gate0_buf, up_proj: &up0_buf, down_proj: &down0_buf },
        ExpertWeights { gate_proj: &gate1_buf, up_proj: &up1_buf, down_proj: &down1_buf },
    ];

    let mut encoder = device.command_encoder().expect("enc");
    moe_dispatch(
        &mut encoder, &mut registry, device.metal_device(),
        &input_buf, &experts, &[w0, w1], &output_buf,
        &scratch_gate, &scratch_up, &scratch_hidden, &scratch_expert,
        &MoeDispatchParams {
            input_dim,
            intermediate_dim,
            n_selected: 2,
        },
    ).expect("moe_dispatch");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = output_buf.as_slice().expect("read");
    for i in 0..input_dim {
        let diff = (output[i] - ref_output[i]).abs();
        let tol = ref_output[i].abs() * 1e-3 + 1e-5;
        assert!(
            diff < tol,
            "two-expert mismatch at {}: expected {}, got {}, diff {}",
            i, ref_output[i], output[i], diff
        );
    }
}

// ---- Validation tests ----

#[test]
fn test_moe_dispatch_zero_input_dim() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::F32, vec![16]).expect("buf");
    let mut encoder = device.command_encoder().expect("enc");

    let result = moe_dispatch(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &[], &[], &buf,
        &buf, &buf, &buf, &buf,
        &MoeDispatchParams {
            input_dim: 0,
            intermediate_dim: 4,
            n_selected: 1,
        },
    );
    assert!(result.is_err(), "zero input_dim should error");
}

#[test]
fn test_moe_dispatch_mismatched_experts() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::F32, vec![16]).expect("buf");
    let mut encoder = device.command_encoder().expect("enc");

    let result = moe_dispatch(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &[], &[0.5, 0.5], &buf,
        &buf, &buf, &buf, &buf,
        &MoeDispatchParams {
            input_dim: 8,
            intermediate_dim: 4,
            n_selected: 2,
        },
    );
    assert!(result.is_err(), "mismatched expert count should error");
}
