//! DeepSeek-V4 0731 compressor parity across prefill and incremental state.

#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams, DEEPSEEK_COMPRESS_RATIO_LONG,
    DEEPSEEK_COMPRESS_RATIO_OVERLAP,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + salt * 19) % 101) as f32 - 50.0) * scale)
        .collect()
}

fn f32_buffer(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len() * 4, DType::F32, shape)
        .unwrap();
    buffer.as_mut_slice().unwrap().copy_from_slice(data);
    buffer
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

fn read_bf16(buffer: &MlxBuffer, len: usize) -> Vec<bf16> {
    unsafe { std::slice::from_raw_parts(buffer.contents_ptr() as *const bf16, len).to_vec() }
}

fn compressed_reference(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    norm: &[f32],
    ratio: usize,
    dim: usize,
    block: usize,
    epsilon: f32,
) -> Vec<bf16> {
    let overlap = ratio == DEEPSEEK_COMPRESS_RATIO_OVERLAP;
    let projected = if overlap { 2 * dim } else { dim };
    let mut pooled = vec![0.0f32; dim];
    for feature in 0..dim {
        let mut entries = Vec::with_capacity(if overlap { 2 * ratio } else { ratio });
        for item in 0..if overlap { 2 * ratio } else { ratio } {
            if overlap && item < ratio && block == 0 {
                continue;
            }
            let source_block = if overlap && item < ratio {
                block - 1
            } else {
                block
            };
            let token = item % ratio;
            let source_feature = if overlap && item >= ratio {
                dim + feature
            } else {
                feature
            };
            let index = (source_block * ratio + token) * projected + source_feature;
            entries.push((
                score[index] + ape[token * projected + source_feature],
                kv[index],
            ));
        }
        let maximum = entries
            .iter()
            .map(|x| x.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = entries.iter().map(|x| (x.0 - maximum).exp()).sum();
        pooled[feature] = entries
            .iter()
            .map(|x| (x.0 - maximum).exp() * x.1)
            .sum::<f32>()
            / denominator;
    }
    let rounded = pooled
        .into_iter()
        .map(|x| bf16::from_f32(x).to_f32())
        .collect::<Vec<_>>();
    let variance = rounded.iter().map(|x| x * x).sum::<f32>() / dim as f32;
    let scale = (variance + epsilon).sqrt().recip();
    rounded
        .iter()
        .zip(norm)
        .map(|(x, weight)| bf16::from_f32(x * scale * weight))
        .collect()
}

fn expected_prefill_state(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    seq: usize,
    ratio: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let overlap = ratio == DEEPSEEK_COMPRESS_RATIO_OVERLAP;
    let coff = if overlap { 2 } else { 1 };
    let projected = coff * dim;
    let mut kv_state = vec![0.0; coff * ratio * projected];
    let mut score_state = vec![f32::NEG_INFINITY; coff * ratio * projected];
    let cutoff = seq - seq % ratio;
    let copy = |slot: usize, token: usize, kv_state: &mut [f32], score_state: &mut [f32]| {
        for feature in 0..projected {
            let src = token * projected + feature;
            let dst = slot * projected + feature;
            kv_state[dst] = kv[src];
            score_state[dst] = score[src] + ape[(token % ratio) * projected + feature];
        }
    };
    if overlap && cutoff >= ratio {
        for token in 0..ratio {
            copy(
                token,
                cutoff - ratio + token,
                &mut kv_state,
                &mut score_state,
            );
        }
    }
    let offset = if overlap { ratio } else { 0 };
    for token in cutoff..seq {
        copy(
            offset + token - cutoff,
            token,
            &mut kv_state,
            &mut score_state,
        );
    }
    (kv_state, score_state)
}

fn assert_bf16_close(got: &[bf16], want: &[bf16], label: &str) {
    assert_eq!(got.len(), want.len());
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        let delta = (got.to_f32() - want.to_f32()).abs();
        assert!(delta <= 0.005, "{label}[{index}] delta={delta}");
    }
}

fn assert_state(got: &[f32], want: &[f32], label: &str) {
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            got.to_bits() == want.to_bits() || (got - want).abs() <= 1e-6,
            "{label}[{index}] {got} != {want}"
        );
    }
}

fn run_ratio_case(ratio: usize, dim: usize, prefill: usize) {
    let device = MlxDevice::new().unwrap();
    let overlap = ratio == DEEPSEEK_COMPRESS_RATIO_OVERLAP;
    let coff = if overlap { 2 } else { 1 };
    let projected = coff * dim;
    let epsilon = 1e-6;
    let total = if overlap { 12 } else { 256 };
    let all_kv = values(total * projected, 1, 0.003);
    let all_score = values(total * projected, 2, 0.002);
    let ape = values(ratio * projected, 3, 0.001);
    let norm = values(dim, 4, 0.002)
        .into_iter()
        .map(|x| x + 1.0)
        .collect::<Vec<_>>();
    let state_len = coff * ratio * projected;
    let kv_state = f32_buffer(
        &device,
        &vec![7.0; state_len],
        vec![1, coff * ratio, projected],
    );
    let score_state = f32_buffer(
        &device,
        &vec![7.0; state_len],
        vec![1, coff * ratio, projected],
    );
    let cache_len = total / ratio + 1;
    let cache = bf16_buffer(
        &device,
        &vec![bf16::ZERO; cache_len * dim],
        vec![1, cache_len, dim],
    );
    let ape_buffer = f32_buffer(&device, &ape, vec![ratio, projected]);
    let norm_buffer = f32_buffer(&device, &norm, vec![dim]);
    let mut registry = KernelRegistry::new();

    let dispatch = |start_pos: usize,
                    seq: usize,
                    kv_state: &MlxBuffer,
                    score_state: &MlxBuffer,
                    registry: &mut KernelRegistry| {
        let kv = f32_buffer(
            &device,
            &all_kv[start_pos * projected..(start_pos + seq) * projected],
            vec![1, seq, projected],
        );
        let score = f32_buffer(
            &device,
            &all_score[start_pos * projected..(start_pos + seq) * projected],
            vec![1, seq, projected],
        );
        let params = DeepSeekCompressorParams {
            batch: 1,
            seq_len: seq as u32,
            start_pos: start_pos as u32,
            ratio: ratio as u32,
            head_dim: dim as u32,
            cache_len: cache_len as u32,
            epsilon,
        };
        let output = bf16_buffer(
            &device,
            &vec![bf16::ONE; params.output_slots() * dim],
            vec![1, params.output_slots(), dim],
        );
        let mut encoder = device.command_encoder().unwrap();
        dispatch_deepseek_compressor(
            &mut encoder,
            registry,
            &device,
            &kv,
            &score,
            &ape_buffer,
            &norm_buffer,
            kv_state,
            score_state,
            &output,
            &cache,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        read_bf16(&output, params.output_slots() * dim)
    };

    let prefill_output = dispatch(0, prefill, &kv_state, &score_state, &mut registry);
    let blocks = prefill / ratio;
    for block in 0..blocks {
        let expected =
            compressed_reference(&all_kv, &all_score, &ape, &norm, ratio, dim, block, epsilon);
        assert_bf16_close(
            &prefill_output[block * dim..(block + 1) * dim],
            &expected,
            "prefill",
        );
        let cache_values = read_bf16(&cache, cache_len * dim);
        assert_bf16_close(
            &cache_values[block * dim..(block + 1) * dim],
            &expected,
            "prefill cache",
        );
    }
    let (mut expected_kv_state, mut expected_score_state) =
        expected_prefill_state(&all_kv, &all_score, &ape, prefill, ratio, dim);
    assert_state(
        kv_state.as_slice().unwrap(),
        &expected_kv_state,
        "prefill kv state",
    );
    assert_state(
        score_state.as_slice().unwrap(),
        &expected_score_state,
        "prefill score state",
    );

    for position in prefill..total {
        let output = dispatch(position, 1, &kv_state, &score_state, &mut registry);
        let slot = (if overlap { ratio } else { 0 }) + position % ratio;
        for feature in 0..projected {
            let src = position * projected + feature;
            let dst = slot * projected + feature;
            expected_kv_state[dst] = all_kv[src];
            expected_score_state[dst] =
                all_score[src] + ape[(position % ratio) * projected + feature];
        }
        if (position + 1) % ratio == 0 {
            let block = position / ratio;
            let expected =
                compressed_reference(&all_kv, &all_score, &ape, &norm, ratio, dim, block, epsilon);
            assert_bf16_close(&output[..dim], &expected, "decode");
            if overlap {
                expected_kv_state.copy_within(ratio * projected..2 * ratio * projected, 0);
                expected_score_state.copy_within(ratio * projected..2 * ratio * projected, 0);
            }
            let cache_values = read_bf16(&cache, cache_len * dim);
            assert_bf16_close(
                &cache_values[block * dim..(block + 1) * dim],
                &expected,
                "cache",
            );
        } else {
            assert!(output.iter().all(|x| x.to_f32() == 0.0));
        }
        assert_state(
            kv_state.as_slice().unwrap(),
            &expected_kv_state,
            "decode kv state",
        );
        assert_state(
            score_state.as_slice().unwrap(),
            &expected_score_state,
            "decode score state",
        );
    }
}

#[test]
fn ratio4_overlap_prefill_and_incremental_match_for_both_production_dims() {
    run_ratio_case(DEEPSEEK_COMPRESS_RATIO_OVERLAP, 128, 10);
    run_ratio_case(DEEPSEEK_COMPRESS_RATIO_OVERLAP, 512, 10);
}

#[test]
fn ratio128_nonoverlap_prefill_and_boundary_update_match() {
    run_ratio_case(DEEPSEEK_COMPRESS_RATIO_LONG, 512, 255);
}

#[test]
fn malformed_and_nonfinite_inputs_fail_closed() {
    let device = MlxDevice::new().unwrap();
    let dim = 128;
    let projected = 2 * dim;
    let mut kv_data = values(4 * projected, 1, 0.002);
    kv_data[dim] = f32::NAN;
    let kv = f32_buffer(&device, &kv_data, vec![1, 4, projected]);
    let score = f32_buffer(
        &device,
        &values(4 * projected, 2, 0.002),
        vec![1, 4, projected],
    );
    let ape = f32_buffer(
        &device,
        &values(4 * projected, 3, 0.001),
        vec![4, projected],
    );
    let norm = f32_buffer(&device, &vec![1.0; dim], vec![dim]);
    let state_shape = vec![1, 8, projected];
    let kv_state = f32_buffer(&device, &vec![0.0; 8 * projected], state_shape.clone());
    let score_state = f32_buffer(
        &device,
        &vec![f32::NEG_INFINITY; 8 * projected],
        state_shape,
    );
    let output = bf16_buffer(&device, &vec![bf16::ONE; dim], vec![1, 1, dim]);
    let cache = bf16_buffer(&device, &vec![bf16::ONE; 2 * dim], vec![1, 2, dim]);
    let mut params = DeepSeekCompressorParams {
        batch: 1,
        seq_len: 4,
        start_pos: 0,
        ratio: 4,
        head_dim: dim as u32,
        cache_len: 2,
        epsilon: 1e-6,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_compressor(
        &mut encoder,
        &mut registry,
        &device,
        &kv,
        &score,
        &ape,
        &norm,
        &kv_state,
        &score_state,
        &output,
        &cache,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert!(read_bf16(&output, dim).iter().all(|x| x.to_f32() == 0.0));
    params.head_dim = 512;
    let mut encoder = device.command_encoder().unwrap();
    assert!(dispatch_deepseek_compressor(
        &mut encoder,
        &mut registry,
        &device,
        &kv,
        &score,
        &ape,
        &norm,
        &kv_state,
        &score_state,
        &output,
        &cache,
        &params
    )
    .is_err());
}
