//! DeepSeek-V4 0731 index score, causal mask, offset, and top-512 parity.

#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::deepseek_indexer::{
    dispatch_deepseek_indexer, dispatch_deepseek_indexer_into, DeepSeekIndexerParams, DEEPSEEK_INDEXER_HEADS,
    DEEPSEEK_INDEXER_HEAD_DIM, DEEPSEEK_INDEXER_RATIO, DEEPSEEK_INDEXER_TOP_K,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const H: usize = DEEPSEEK_INDEXER_HEADS;
const D: usize = DEEPSEEK_INDEXER_HEAD_DIM;
const K: usize = DEEPSEEK_INDEXER_TOP_K;

fn bf16_values(len: usize, salt: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let value = ((i * 37 + salt * 19) % 101) as f32 - 50.0;
            bf16::from_f32(value * scale)
        })
        .collect()
}

#[test]
fn strided_output_writes_only_the_requested_tail() {
    let device = MlxDevice::new().unwrap();
    let queries = 5;
    let kv_len = 2;
    let prefix = queries;
    let stride = prefix + K;
    let q = bf16_buffer(
        &device,
        &bf16_values(queries * H * D, 1, 0.003),
        vec![1, queries, H, D],
    );
    let kv_values = bf16_values(kv_len * D, 2, 0.004);
    let weights_values = vec![1.0; queries * H];
    let expected = reference(
        q.as_slice::<bf16>().unwrap(),
        &kv_values,
        &weights_values,
        1,
        queries,
        kv_len,
        0,
        128,
    );
    let kv = bf16_buffer(&device, &kv_values, vec![1, kv_len, D]);
    let weights = f32_buffer(&device, &weights_values, vec![1, queries, H]);
    let scratch = f32_buffer(
        &device,
        &vec![0.0; queries * kv_len],
        vec![1, queries, kv_len],
    );
    let mut output = device
        .alloc_buffer(queries * stride * 4, DType::I32, vec![1, queries, stride])
        .unwrap();
    output.as_mut_slice::<i32>().unwrap().fill(77);
    let params = DeepSeekIndexerParams {
        batch: 1,
        query_len: queries as u32,
        kv_len: kv_len as u32,
        start_pos: 0,
        ratio: 4,
        heads: H as u32,
        head_dim: D as u32,
        top_k: K as u32,
        offset: 128,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_indexer_into(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        stride,
        prefix,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let actual = output.as_slice::<i32>().unwrap();
    for query in 0..queries {
        let row = &actual[query * stride..(query + 1) * stride];
        assert!(row[..prefix].iter().all(|&value| value == 77));
        assert_eq!(&row[prefix..], &expected[query * K..(query + 1) * K]);
    }
}

fn bf16_buffer(device: &MlxDevice, data: &[bf16], shape: Vec<usize>) -> MlxBuffer {
    let buffer = device
        .alloc_buffer(data.len() * 2, DType::BF16, shape)
        .unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buffer.contents_ptr() as *mut bf16,
            data.len(),
        );
    }
    buffer
}

fn f32_buffer(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len() * 4, DType::F32, shape)
        .unwrap();
    buffer.as_mut_slice().unwrap().copy_from_slice(data);
    buffer
}

fn reference(
    q: &[bf16],
    kv: &[bf16],
    weights: &[f32],
    batch: usize,
    queries: usize,
    kv_len: usize,
    start_pos: usize,
    offset: i32,
) -> Vec<i32> {
    let mut output = vec![-1; batch * queries * K];
    for b in 0..batch {
        for query in 0..queries {
            let valid = ((start_pos + query + 1) / DEEPSEEK_INDEXER_RATIO).min(kv_len);
            let mut scores = Vec::with_capacity(valid);
            for candidate in 0..valid {
                let mut score = 0.0f32;
                let mut invalid = false;
                for head in 0..H {
                    let q_base = ((b * queries + query) * H + head) * D;
                    let kv_base = (b * kv_len + candidate) * D;
                    let mut dot = 0.0f32;
                    for feature in 0..D {
                        let qv = q[q_base + feature].to_f32();
                        let kvv = kv[kv_base + feature].to_f32();
                        invalid |= !qv.is_finite() || !kvv.is_finite();
                        dot += qv * kvv;
                    }
                    let weight = weights[(b * queries + query) * H + head];
                    let contribution = dot.max(0.0) * weight;
                    invalid |= !weight.is_finite() || !contribution.is_finite();
                    score += contribution;
                }
                if !invalid && score.is_finite() {
                    scores.push((candidate, score));
                }
            }
            scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let base = (b * queries + query) * K;
            for (slot, &(index, _)) in scores.iter().take(K).enumerate() {
                output[base + slot] = index as i32 + offset;
            }
        }
    }
    output
}

fn run_case(batch: usize, queries: usize, kv_len: usize, start_pos: usize, offset: i32) {
    let q = bf16_values(batch * queries * H * D, 1, 0.003);
    let kv = bf16_values(batch * kv_len * D, 2, 0.004);
    let weights = (0..batch * queries * H)
        .map(|i| ((i * 17 % 29) as f32 - 11.0) * 0.007)
        .collect::<Vec<_>>();
    let expected = reference(&q, &kv, &weights, batch, queries, kv_len, start_pos, offset);
    let device = MlxDevice::new().unwrap();
    let q = bf16_buffer(&device, &q, vec![batch, queries, H, D]);
    let kv = bf16_buffer(&device, &kv, vec![batch, kv_len, D]);
    let weights = f32_buffer(&device, &weights, vec![batch, queries, H]);
    let scratch = f32_buffer(
        &device,
        &vec![7.0; batch * queries * kv_len],
        vec![batch, queries, kv_len],
    );
    let output = device
        .alloc_buffer(batch * queries * K * 4, DType::I32, vec![batch, queries, K])
        .unwrap();
    let params = DeepSeekIndexerParams {
        batch: batch as u32,
        query_len: queries as u32,
        kv_len: kv_len as u32,
        start_pos: start_pos as u32,
        ratio: DEEPSEEK_INDEXER_RATIO as u32,
        heads: H as u32,
        head_dim: D as u32,
        top_k: K as u32,
        offset,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_indexer(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(output.as_slice::<i32>().unwrap(), expected);
}

#[test]
fn prefill_and_decode_match_cpu_with_causal_sentinels_and_offset() {
    run_case(1, 12, 3, 0, 128);
    run_case(2, 1, 12, 39, 257);
}

#[test]
fn representative_0731_decode_selects_top512_of_640() {
    run_case(1, 1, 640, 2559, 128);
}

#[test]
fn ties_are_deterministic_and_early_prefill_is_all_sentinel() {
    let device = MlxDevice::new().unwrap();
    let queries = 5;
    let kv_len = 2;
    let q = bf16_buffer(
        &device,
        &vec![bf16::ZERO; queries * H * D],
        vec![1, queries, H, D],
    );
    let kv = bf16_buffer(
        &device,
        &bf16_values(kv_len * D, 2, 0.01),
        vec![1, kv_len, D],
    );
    let weights = f32_buffer(&device, &vec![1.0; queries * H], vec![1, queries, H]);
    let scratch = f32_buffer(
        &device,
        &vec![0.0; queries * kv_len],
        vec![1, queries, kv_len],
    );
    let output = device
        .alloc_buffer(queries * K * 4, DType::I32, vec![1, queries, K])
        .unwrap();
    let params = DeepSeekIndexerParams {
        batch: 1,
        query_len: queries as u32,
        kv_len: kv_len as u32,
        start_pos: 0,
        ratio: 4,
        heads: H as u32,
        head_dim: D as u32,
        top_k: K as u32,
        offset: 9,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_indexer(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let result = output.as_slice::<i32>().unwrap();
    assert!(result[..3 * K].iter().all(|&x| x == -1));
    assert_eq!(result[3 * K], 9);
    assert_eq!(result[4 * K], 9);
    assert!(result[4 * K + 1..5 * K].iter().all(|&x| x == -1));
}

#[test]
fn malformed_and_nonfinite_inputs_fail_closed() {
    let device = MlxDevice::new().unwrap();
    let mut q_values = bf16_values(H * D, 1, 0.002);
    q_values[0] = bf16::NAN;
    let q = bf16_buffer(&device, &q_values, vec![1, 1, H, D]);
    let kv = bf16_buffer(&device, &bf16_values(2 * D, 2, 0.002), vec![1, 2, D]);
    let weights = f32_buffer(&device, &vec![1.0; H], vec![1, 1, H]);
    let scratch = f32_buffer(&device, &[0.0, 0.0], vec![1, 1, 2]);
    let output = device
        .alloc_buffer(K * 4, DType::I32, vec![1, 1, K])
        .unwrap();
    let mut params = DeepSeekIndexerParams {
        batch: 1,
        query_len: 1,
        kv_len: 2,
        start_pos: 7,
        ratio: 4,
        heads: H as u32,
        head_dim: D as u32,
        top_k: K as u32,
        offset: 0,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_indexer(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert!(output.as_slice::<i32>().unwrap().iter().all(|&x| x == -1));
    params.offset = -2;
    let mut encoder = device.command_encoder().unwrap();
    assert!(dispatch_deepseek_indexer(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        &params
    )
    .is_err());
    params.offset = 0;
    params.top_k = 8;
    assert!(dispatch_deepseek_indexer(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &weights,
        &scratch,
        &output,
        &params
    )
    .is_err());
}
