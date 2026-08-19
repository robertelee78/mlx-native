//! Real Metal proof that expert strides address the selected expert matrix,
//! not the padding between matrices. Validation-only capture tests live in
//! `ggml_explicit_routing_policy`; this test commits and compares outputs.

#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::f16;
use mlx_native::{
    quantized_matmul_id_ggml_pooled_pair_with_policy, quantized_matmul_id_ggml_pooled_with_policy,
    quantized_matmul_id_ggml_with_policy, quantized_matmul_id_swiglu_q4_0, DType,
    GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType, IdMmScratch,
    KernelRegistry, MlxBuffer, MlxDevice,
};

const N: usize = 32;
const K: usize = 32;
const EXPERTS: usize = 2;

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut bytes = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        bytes.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for value in block {
            bytes.push((value * inverse).round().clamp(-128.0, 127.0) as i8 as u8);
        }
    }
    bytes
}

fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % 32, 0);
    let mut bytes = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        bytes.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for index in 0..16 {
            let low = (block[index] * inverse + 8.0).round().clamp(0.0, 15.0) as u8;
            let high = (block[index + 16] * inverse + 8.0).round().clamp(0.0, 15.0) as u8;
            bytes.push(low | (high << 4));
        }
    }
    bytes
}

fn expert_matrices(pack: fn(&[f32]) -> Vec<u8>) -> Vec<Vec<u8>> {
    (0..EXPERTS)
        .map(|expert| {
            let values: Vec<_> = (0..N * K)
                .map(|index| {
                    let centered = (index % 37) as f32 - 18.0;
                    centered * (expert as f32 + 1.0) / 29.0
                })
                .collect();
            pack(&values)
        })
        .collect()
}

fn layout(matrices: &[Vec<u8>], stride: usize) -> Vec<u8> {
    let matrix_bytes = matrices[0].len();
    let mut bytes = vec![0xa5; stride * (matrices.len() - 1) + matrix_bytes];
    for (expert, matrix) in matrices.iter().enumerate() {
        bytes[expert * stride..expert * stride + matrix_bytes].copy_from_slice(matrix);
    }
    bytes
}

fn u8_buffer(device: &MlxDevice, values: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len(), DType::U8, vec![values.len()])
        .expect("u8 buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("u8 slice")
        .copy_from_slice(values);
    buffer
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .copy_from_slice(values);
    buffer
}

fn ids_buffer(device: &MlxDevice, count: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(count * 4, DType::U32, vec![count])
        .expect("ids buffer");
    buffer.as_mut_slice::<u32>().expect("ids slice").fill(1);
    buffer
}

fn exact(left: &[f32], right: &[f32], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length");
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "{label}[{index}]");
    }
}

fn policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    }
}

fn run_auto_mv(weight_bytes: &[u8], stride: usize) -> Vec<f32> {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let weights = u8_buffer(&device, weight_bytes);
    let input_values: Vec<_> = (0..K).map(|index| index as f32 / 17.0 - 0.8).collect();
    let input = f32_buffer(&device, &input_values);
    let ids = ids_buffer(&device, 1);
    let output = f32_buffer(&device, &[0.0; N]);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: 1,
        n: N as u32,
        k: K as u32,
        n_experts: EXPERTS as u32,
        expert_stride: stride as u64,
        ggml_type: GgmlType::Q8_0,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weights,
        &ids,
        &output,
        &params,
        &policy(),
    )
    .expect("auto MV");
    encoder.commit_and_wait().expect("auto MV wait");
    output.as_slice::<f32>().expect("auto MV output").to_vec()
}

fn run_pooled(weight_bytes: &[u8], stride: usize, pair: bool) -> (Vec<f32>, Vec<f32>) {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let rows = 33usize;
    let weights = u8_buffer(&device, weight_bytes);
    let input_values: Vec<_> = (0..rows * K)
        .map(|index| (index % 41) as f32 / 23.0 - 0.9)
        .collect();
    let input = f32_buffer(&device, &input_values);
    let ids = ids_buffer(&device, rows);
    let first = f32_buffer(&device, &vec![0.0; rows * N]);
    let second = f32_buffer(&device, &vec![0.0; rows * N]);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: rows as u32,
        top_k: 1,
        n: N as u32,
        k: K as u32,
        n_experts: EXPERTS as u32,
        expert_stride: stride as u64,
        ggml_type: GgmlType::Q8_0,
    };
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS as u32, rows as u32).expect("scratch");
    let mut encoder = device.command_encoder().expect("encoder");
    if pair {
        quantized_matmul_id_ggml_pooled_pair_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weights,
            &weights,
            &ids,
            &first,
            &second,
            &mut scratch,
            &params,
            &policy(),
        )
        .expect("pooled pair");
    } else {
        quantized_matmul_id_ggml_pooled_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weights,
            &ids,
            &first,
            &mut scratch,
            &params,
            &policy(),
        )
        .expect("pooled");
    }
    encoder.commit_and_wait().expect("pooled wait");
    (
        first.as_slice::<f32>().expect("first output").to_vec(),
        second.as_slice::<f32>().expect("second output").to_vec(),
    )
}

fn run_q4_swiglu(weight_bytes: &[u8], stride: usize) -> Vec<f32> {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let weights = u8_buffer(&device, weight_bytes);
    let gate_values: Vec<_> = (0..K).map(|index| index as f32 / 31.0 - 0.4).collect();
    let up_values: Vec<_> = (0..K).map(|index| 0.7 - index as f32 / 47.0).collect();
    let gate = f32_buffer(&device, &gate_values);
    let up = f32_buffer(&device, &up_values);
    let ids = ids_buffer(&device, 1);
    let output = f32_buffer(&device, &[0.0; N]);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: 1,
        n: N as u32,
        k: K as u32,
        n_experts: EXPERTS as u32,
        expert_stride: stride as u64,
        ggml_type: GgmlType::Q4_0,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_swiglu_q4_0(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        &weights,
        &ids,
        &output,
        &params,
    )
    .expect("Q4 SwiGLU");
    encoder.commit_and_wait().expect("Q4 SwiGLU wait");
    output.as_slice::<f32>().expect("Q4 output").to_vec()
}

#[test]
fn selected_second_expert_matches_tight_layout_across_padded_entrypoints() {
    let q8 = expert_matrices(pack_q8_0);
    let q8_tight_stride = q8[0].len();
    let q8_padded_stride = q8_tight_stride + 64;
    let tight = layout(&q8, q8_tight_stride);
    let padded = layout(&q8, q8_padded_stride);

    exact(
        &run_auto_mv(&tight, q8_tight_stride),
        &run_auto_mv(&padded, q8_padded_stride),
        "auto MV",
    );
    let tight_pooled = run_pooled(&tight, q8_tight_stride, false);
    let padded_pooled = run_pooled(&padded, q8_padded_stride, false);
    exact(&tight_pooled.0, &padded_pooled.0, "pooled MM");
    let tight_pair = run_pooled(&tight, q8_tight_stride, true);
    let padded_pair = run_pooled(&padded, q8_padded_stride, true);
    exact(&tight_pair.0, &padded_pair.0, "pooled pair first");
    exact(&tight_pair.1, &padded_pair.1, "pooled pair second");

    let q4 = expert_matrices(pack_q4_0);
    let q4_tight_stride = q4[0].len();
    let q4_padded_stride = q4_tight_stride + 64;
    exact(
        &run_q4_swiglu(&layout(&q4, q4_tight_stride), q4_tight_stride),
        &run_q4_swiglu(&layout(&q4, q4_padded_stride), q4_padded_stride),
        "Q4 SwiGLU",
    );
}
