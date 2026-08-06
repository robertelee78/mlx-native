//! Official DeepSeek-V4 0731 sparse-attention parity and validation gates.

#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, dispatch_deepseek_sparse_attention_flash_decode, register,
    DeepSeekSparseAttentionParams, DEEPSEEK_INDEX_TOP_K, DEEPSEEK_SPARSE_HEADS,
    DEEPSEEK_SPARSE_HEAD_DIM,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const H: usize = DEEPSEEK_SPARSE_HEADS;
const D: usize = DEEPSEEK_SPARSE_HEAD_DIM;

fn data(len: usize, salt: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let raw = ((i.wrapping_mul(37) + salt * 19) % 101) as f32 - 50.0;
            bf16::from_f32(raw * scale)
        })
        .collect()
}

fn bf16_buffer(device: &MlxDevice, values: &[bf16], shape: Vec<usize>) -> MlxBuffer {
    let buffer = device
        .alloc_buffer(values.len() * 2, DType::BF16, shape)
        .expect("allocate bf16");
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents_ptr() as *mut bf16,
            values.len(),
        );
    }
    buffer
}

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, shape)
        .expect("allocate f32");
    buffer.as_mut_slice().unwrap().copy_from_slice(values);
    buffer
}

fn i32_buffer(device: &MlxDevice, values: &[i32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::I32, shape)
        .expect("allocate i32");
    buffer.as_mut_slice().unwrap().copy_from_slice(values);
    buffer
}

fn reference(
    q: &[bf16],
    kv: &[bf16],
    sinks: &[f32],
    indices: &[i32],
    batch: usize,
    queries: usize,
    kv_len: usize,
    top_k: usize,
    scale: f32,
) -> Vec<bf16> {
    let mut output = vec![bf16::ZERO; batch * queries * H * D];
    for b in 0..batch {
        for query in 0..queries {
            let selected =
                &indices[(b * queries + query) * top_k..(b * queries + query + 1) * top_k];
            for head in 0..H {
                let out_base = ((b * queries + query) * H + head) * D;
                let q_row = &q[out_base..out_base + D];
                let mut invalid = !sinks[head].is_finite()
                    || q_row.iter().any(|x| !x.to_f32().is_finite())
                    || selected
                        .iter()
                        .any(|&i| i < -1 || i as usize >= kv_len && i != -1);
                let mut logits = Vec::with_capacity(top_k);
                for &index in selected {
                    if index == -1 {
                        continue;
                    }
                    let kv_base = (b * kv_len + index as usize) * D;
                    let kv_row = &kv[kv_base..kv_base + D];
                    invalid |= kv_row.iter().any(|x| !x.to_f32().is_finite());
                    let dot = q_row
                        .iter()
                        .zip(kv_row)
                        .map(|(a, v)| a.to_f32() * v.to_f32())
                        .sum::<f32>();
                    let logit = dot * scale;
                    invalid |= !logit.is_finite();
                    logits.push((logit, kv_row));
                }
                if invalid || logits.is_empty() {
                    continue;
                }
                let maximum = logits.iter().map(|x| x.0).fold(sinks[head], f32::max);
                let denominator = (sinks[head] - maximum).exp()
                    + logits.iter().map(|x| (x.0 - maximum).exp()).sum::<f32>();
                for d in 0..D {
                    let numerator = logits
                        .iter()
                        .map(|(logit, row)| (logit - maximum).exp() * row[d].to_f32())
                        .sum::<f32>();
                    output[out_base + d] = bf16::from_f32(numerator / denominator);
                }
            }
        }
    }
    output
}

fn run_case(batch: usize, queries: usize, kv_len: usize, indices: Vec<i32>) {
    let top_k = indices.len() / (batch * queries);
    let q = data(batch * queries * H * D, 1, 0.0015);
    let kv = data(batch * kv_len * D, 3, 0.002);
    let sinks = (0..H).map(|h| h as f32 * 0.003 - 0.1).collect::<Vec<_>>();
    let scale = 1.0 / (D as f32).sqrt();
    let expected = reference(
        &q, &kv, &sinks, &indices, batch, queries, kv_len, top_k, scale,
    );
    let device = MlxDevice::new().expect("Metal device");
    let q = bf16_buffer(&device, &q, vec![batch, queries, H, D]);
    let kv = bf16_buffer(&device, &kv, vec![batch, kv_len, D]);
    let sinks = f32_buffer(&device, &sinks, vec![H]);
    let indices = i32_buffer(&device, &indices, vec![batch, queries, top_k]);
    let output = device
        .alloc_buffer(
            batch * queries * H * D * 2,
            DType::BF16,
            vec![batch, queries, H, D],
        )
        .unwrap();
    let params = DeepSeekSparseAttentionParams {
        batch: batch as u32,
        query_len: queries as u32,
        kv_len: kv_len as u32,
        top_k: top_k as u32,
        heads: H as u32,
        head_dim: D as u32,
        scale,
    };
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &indices,
        &output,
        &params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("completion");
    let actual =
        unsafe { std::slice::from_raw_parts(output.contents_ptr() as *const bf16, expected.len()) };
    for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (got.to_f32() - want.to_f32()).abs();
        assert!(
            delta <= 0.0025,
            "output[{i}] delta={delta}: {got:?} != {want:?}"
        );
    }

    if queries == 1 && top_k >= 384 {
        let gathered = device
            .alloc_buffer(top_k * D * 2, DType::BF16, vec![1, 1, top_k, D])
            .unwrap();
        let mask = device
            .alloc_buffer(top_k * 2, DType::BF16, vec![1, top_k])
            .unwrap();
        let mut invalid_global = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        invalid_global.as_mut_slice::<u32>().unwrap().fill(0);
        let mut invalid_heads = device.alloc_buffer(H * 4, DType::U32, vec![H]).unwrap();
        invalid_heads.as_mut_slice::<u32>().unwrap().fill(0);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_deepseek_sparse_attention_flash_decode(
            &mut encoder,
            &mut registry,
            &device,
            &q,
            &kv,
            &sinks,
            &indices,
            &gathered,
            &mask,
            &invalid_global,
            &invalid_heads,
            &output,
            &params,
        )
        .expect("flash dispatch");
        encoder.commit_and_wait().expect("flash completion");
        let flash = unsafe {
            std::slice::from_raw_parts(output.contents_ptr() as *const bf16, expected.len())
        };
        for (i, (got, want)) in flash.iter().zip(expected.iter()).enumerate() {
            let delta = (got.to_f32() - want.to_f32()).abs();
            assert!(
                delta <= 0.0025,
                "flash output[{i}] delta={delta}: {got:?} != {want:?}"
            );
        }
    }
}

#[test]
fn decode_and_prefill_match_cpu_with_sentinels_duplicates_and_arbitrary_order() {
    run_case(1, 1, 5, vec![4, 1, -1, 1, 0, 3, -1]);
    let mut prefill = Vec::new();
    for query in 0..3 {
        prefill.extend([8 - query, 2, -1, query, 6, 2, 10, -1, 4]);
    }
    run_case(1, 3, 11, prefill);
}

#[test]
fn representative_0731_decode_uses_64_by_512_and_topk_512() {
    let indices = (0..DEEPSEEK_INDEX_TOP_K)
        .map(|i| ((i * 193 + 17) % 640) as i32)
        .collect();
    run_case(1, 1, 640, indices);
}

#[test]
fn all_sentinel_and_bad_dynamic_values_fail_closed() {
    run_case(1, 1, 3, vec![-1; 8]);

    let device = MlxDevice::new().unwrap();
    let mut q_values = data(H * D, 4, 0.002);
    q_values[0] = bf16::INFINITY;
    let q = bf16_buffer(&device, &q_values, vec![1, 1, H, D]);
    let kv = bf16_buffer(&device, &data(3 * D, 5, 0.002), vec![1, 3, D]);
    let sinks = f32_buffer(&device, &vec![0.0; H], vec![H]);
    let indices = i32_buffer(&device, &[0], vec![1, 1, 1]);
    let output = bf16_buffer(&device, &vec![bf16::ONE; H * D], vec![1, 1, H, D]);
    let params = DeepSeekSparseAttentionParams {
        batch: 1,
        query_len: 1,
        kv_len: 3,
        top_k: 1,
        heads: H as u32,
        head_dim: D as u32,
        scale: 1.0 / (D as f32).sqrt(),
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &indices,
        &output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let values = unsafe { std::slice::from_raw_parts(output.contents_ptr() as *const bf16, H * D) };
    assert!(values[..D].iter().all(|x| x.to_f32() == 0.0));
    assert!(values[D..].iter().any(|x| x.to_f32() != 0.0));

    let bad_indices = i32_buffer(&device, &[3], vec![1, 1, 1]);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &bad_indices,
        &output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert!(values.iter().all(|x| x.to_f32() == 0.0));
}

#[test]
fn malformed_shapes_dtypes_and_params_are_rejected_before_encoding() {
    let device = MlxDevice::new().unwrap();
    let q = bf16_buffer(&device, &data(H * D, 1, 0.001), vec![1, 1, H, D]);
    let kv = bf16_buffer(&device, &data(D, 2, 0.001), vec![1, 1, D]);
    let sinks = f32_buffer(&device, &vec![0.0; H], vec![H]);
    let bad_indices = f32_buffer(&device, &[0.0], vec![1, 1, 1]);
    let output = bf16_buffer(&device, &vec![bf16::ZERO; H * D], vec![1, 1, H, D]);
    let mut params = DeepSeekSparseAttentionParams {
        batch: 1,
        query_len: 1,
        kv_len: 1,
        top_k: 1,
        heads: H as u32,
        head_dim: D as u32,
        scale: 1.0 / (D as f32).sqrt(),
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    assert!(dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &bad_indices,
        &output,
        &params,
    )
    .is_err());
    params.head_dim = 128;
    let indices = i32_buffer(&device, &[0], vec![1, 1, 1]);
    assert!(dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &indices,
        &output,
        &params,
    )
    .is_err());
    params.head_dim = D as u32;
    params.scale = f32::NAN;
    assert!(dispatch_deepseek_sparse_attention(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &kv,
        &sinks,
        &indices,
        &output,
        &params,
    )
    .is_err());
}
