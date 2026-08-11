//! Public contract tests for paired MoE prefill projections.

#![cfg(target_vendor = "apple")]

use mlx_native::{
    quantized_matmul_id_ggml_pooled_pair, DType, GgmlQuantizedMatmulIdParams, GgmlType,
    IdMmScratch, KernelRegistry, MlxDevice, MlxError,
};

const N_TOKENS: u32 = 33;
const TOP_K: u32 = 1;
const N: u32 = 1;
const K: u32 = 256;
const N_EXPERTS: u32 = 1;
const Q5_K_BLOCK_BYTES: usize = 176;

fn params(n_tokens: u32) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens,
        top_k: TOP_K,
        n: N,
        k: K,
        n_experts: N_EXPERTS,
        expert_stride: Q5_K_BLOCK_BYTES as u64,
        ggml_type: GgmlType::Q5_K,
    }
}

#[test]
fn pooled_pair_rejects_overlapping_outputs_and_decode_sized_work() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let input = device
        .alloc_buffer(
            (N_TOKENS * K) as usize * 4,
            DType::F32,
            vec![N_TOKENS as usize, K as usize],
        )
        .expect("input");
    let weight = device
        .alloc_buffer(Q5_K_BLOCK_BYTES, DType::U8, vec![Q5_K_BLOCK_BYTES])
        .expect("weight");
    let ids = device
        .alloc_buffer(
            (N_TOKENS * TOP_K) as usize * 4,
            DType::U32,
            vec![N_TOKENS as usize, TOP_K as usize],
        )
        .expect("ids");
    let first_output = device
        .alloc_buffer(
            (N_TOKENS * TOP_K * N) as usize * 4,
            DType::F32,
            vec![(N_TOKENS * TOP_K * N) as usize],
        )
        .expect("first output");
    let second_output = device
        .alloc_buffer(
            (N_TOKENS * TOP_K * N) as usize * 4,
            DType::F32,
            vec![(N_TOKENS * TOP_K * N) as usize],
        )
        .expect("second output");
    let mut scratch = IdMmScratch::alloc(&device, N_EXPERTS, N_TOKENS).expect("routing scratch");

    let mut encoder = device.command_encoder().expect("overlap encoder");
    match quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &weight,
        &ids,
        &first_output,
        &first_output,
        &mut scratch,
        &params(N_TOKENS),
    ) {
        Err(MlxError::InvalidArgument(message)) => {
            assert!(message.contains("output ranges must not overlap"));
        }
        other => panic!("overlapping pair outputs must fail before encoding: {other:?}"),
    }

    let mut encoder = device.command_encoder().expect("read-alias encoder");
    match quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &weight,
        &ids,
        &first_output,
        &weight,
        &mut scratch,
        &params(N_TOKENS),
    ) {
        Err(MlxError::InvalidArgument(message)) => {
            assert!(message.contains("output range must not overlap"));
        }
        other => panic!("pair output/read alias must fail before encoding: {other:?}"),
    }

    let mut aliased_scratch =
        IdMmScratch::alloc(&device, N_EXPERTS, N_TOKENS).expect("aliased routing scratch");
    aliased_scratch.htpe = input.clone();
    let mut encoder = device
        .command_encoder()
        .expect("scratch/read-alias encoder");
    match quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &weight,
        &ids,
        &first_output,
        &second_output,
        &mut aliased_scratch,
        &params(N_TOKENS),
    ) {
        Err(MlxError::InvalidArgument(message)) => {
            assert!(message.contains("scratch range must not overlap input"));
        }
        other => panic!("pair scratch/read alias must fail before encoding: {other:?}"),
    }

    let mut overlapping_scratch =
        IdMmScratch::alloc(&device, N_EXPERTS, N_TOKENS).expect("overlapping routing scratch");
    overlapping_scratch.htpe = overlapping_scratch.hids.clone();
    let mut encoder = device.command_encoder().expect("scratch-overlap encoder");
    match quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &weight,
        &ids,
        &first_output,
        &second_output,
        &mut overlapping_scratch,
        &params(N_TOKENS),
    ) {
        Err(MlxError::InvalidArgument(message)) => {
            assert!(message.contains("scratch ranges must not overlap"));
        }
        other => panic!("overlapping pair scratch must fail before encoding: {other:?}"),
    }

    let mut encoder = device.command_encoder().expect("decode encoder");
    match quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &weight,
        &ids,
        &first_output,
        &second_output,
        &mut scratch,
        &params(32),
    ) {
        Err(MlxError::InvalidArgument(message)) => {
            assert!(message.contains("requires the mm_id route"));
        }
        other => panic!("decode-sized pair work must remain on the existing route: {other:?}"),
    }
}
