//! DeepSeek-V4 Hyper-Connection CPU-vs-Metal parity and validation gates.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_head_weights, dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
    register, DEEPSEEK_HC_EPS, DEEPSEEK_HC_MULT, DEEPSEEK_HC_SINKHORN_ITERS,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const HC: usize = 4;
const MIX: usize = 24;

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + salt * 19) % 101) as f32 - 50.0) * scale)
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn split_reference(mixes: &[f32], scale: &[f32], base: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let tokens = mixes.len() / MIX;
    let mut pre = vec![0.0; tokens * HC];
    let mut post = vec![0.0; tokens * HC];
    let mut comb = vec![0.0; tokens * HC * HC];
    for token in 0..tokens {
        let m = &mixes[token * MIX..(token + 1) * MIX];
        for branch in 0..HC {
            pre[token * HC + branch] =
                sigmoid(m[branch] * scale[0] + base[branch]) + DEEPSEEK_HC_EPS;
            post[token * HC + branch] =
                2.0 * sigmoid(m[HC + branch] * scale[1] + base[HC + branch]);
        }
        let matrix = &mut comb[token * HC * HC..(token + 1) * HC * HC];
        for source in 0..HC {
            let offset = source * HC;
            let row_max = (0..HC)
                .map(|destination| {
                    m[2 * HC + offset + destination] * scale[2]
                        + base[2 * HC + offset + destination]
                })
                .fold(f32::NEG_INFINITY, f32::max);
            let mut row_sum = 0.0;
            for destination in 0..HC {
                let index = offset + destination;
                matrix[index] =
                    (m[2 * HC + index] * scale[2] + base[2 * HC + index] - row_max).exp();
                row_sum += matrix[index];
            }
            for destination in 0..HC {
                matrix[offset + destination] =
                    matrix[offset + destination] / row_sum + DEEPSEEK_HC_EPS;
            }
        }
        normalize_columns(matrix);
        for _ in 1..DEEPSEEK_HC_SINKHORN_ITERS {
            for source in 0..HC {
                let offset = source * HC;
                let sum: f32 = matrix[offset..offset + HC].iter().sum();
                for destination in 0..HC {
                    matrix[offset + destination] /= sum + DEEPSEEK_HC_EPS;
                }
            }
            normalize_columns(matrix);
        }
    }
    (pre, post, comb)
}

fn normalize_columns(matrix: &mut [f32]) {
    for destination in 0..HC {
        let sum: f32 = (0..HC)
            .map(|source| matrix[source * HC + destination])
            .sum();
        for source in 0..HC {
            matrix[source * HC + destination] /= sum + DEEPSEEK_HC_EPS;
        }
    }
}

fn pre_reference(x: &[f32], weights: &[f32], tokens: usize, embd: usize) -> Vec<f32> {
    let mut out = vec![0.0; tokens * embd];
    for token in 0..tokens {
        for e in 0..embd {
            for source in 0..HC {
                out[token * embd + e] +=
                    x[(token * HC + source) * embd + e] * weights[token * HC + source];
            }
        }
    }
    out
}

fn head_weights_reference(mixes: &[f32], scale: f32, base: &[f32]) -> Vec<f32> {
    mixes
        .iter()
        .enumerate()
        .map(|(index, mix)| sigmoid(mix * scale + base[index % HC]) + DEEPSEEK_HC_EPS)
        .collect()
}

fn post_reference(
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    tokens: usize,
    embd: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; tokens * HC * embd];
    for token in 0..tokens {
        for destination in 0..HC {
            for e in 0..embd {
                let mut value = x[token * embd + e] * post[token * HC + destination];
                for source in 0..HC {
                    value += residual[(token * HC + source) * embd + e]
                        * comb[(token * HC + source) * HC + destination];
                }
                out[(token * HC + destination) * embd + e] = value;
            }
        }
    }
    out
}

fn buffer(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len() * 4, DType::F32, shape)
        .expect("allocate f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("map f32 buffer")
        .copy_from_slice(data);
    buffer
}

fn output(device: &MlxDevice, len: usize, shape: Vec<usize>) -> MlxBuffer {
    device
        .alloc_buffer(len * 4, DType::F32, shape)
        .expect("allocate output")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && (got - want).abs() <= tolerance,
            "{label}[{index}] {got} != {want} (tol {tolerance})"
        );
    }
}

#[test]
fn production_constants_are_pinned() {
    assert_eq!(DEEPSEEK_HC_MULT, 4);
    assert_eq!(DEEPSEEK_HC_SINKHORN_ITERS, 20);
    assert_eq!(DEEPSEEK_HC_EPS, 1e-6);
}

#[test]
fn split_sinkhorn_matches_cpu_for_decode_and_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    for tokens in [1usize, 17, 257] {
        let mixes = values(tokens * MIX, 1, 0.04);
        let scale = vec![-0.35, 0.2, 0.45];
        let base = values(MIX, 3, 0.005);
        let expected = split_reference(&mixes, &scale, &base);
        let mixes = buffer(&device, &mixes, vec![tokens, MIX]);
        let scale = buffer(&device, &scale, vec![3]);
        let base = buffer(&device, &base, vec![MIX]);
        let pre = output(&device, tokens * HC, vec![tokens, HC]);
        let post = output(&device, tokens * HC, vec![tokens, HC]);
        let comb = output(&device, tokens * HC * HC, vec![tokens, HC, HC]);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_hc_split_sinkhorn(
            &mut encoder,
            &mut registry,
            &device,
            &mixes,
            &scale,
            &base,
            &pre,
            &post,
            &comb,
            tokens as u32,
        )
        .expect("split dispatch");
        encoder.commit_and_wait().expect("split completion");
        assert_close(pre.as_slice().unwrap(), &expected.0, 2e-5, "pre");
        assert_close(post.as_slice().unwrap(), &expected.1, 2e-5, "post");
        assert_close(comb.as_slice().unwrap(), &expected.2, 2e-5, "comb");
        for token in 0..tokens {
            let matrix = &comb.as_slice::<f32>().unwrap()[token * 16..(token + 1) * 16];
            for destination in 0..HC {
                let sum: f32 = (0..HC)
                    .map(|source| matrix[source * HC + destination])
                    .sum();
                assert!((sum - 1.0).abs() < 2e-5, "column {destination} sum {sum}");
            }
        }
    }
}

#[test]
fn head_weights_match_cpu_for_decode_and_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    for tokens in [1usize, 17, 257] {
        let mixes_data = values(tokens * HC, 13, 0.04);
        let scale_data = [-0.35];
        let base_data = values(HC, 14, 0.005);
        let expected = head_weights_reference(&mixes_data, scale_data[0], &base_data);
        let mixes = buffer(&device, &mixes_data, vec![tokens, HC]);
        let scale = buffer(&device, &scale_data, vec![1]);
        let base = buffer(&device, &base_data, vec![HC]);
        let weights = output(&device, tokens * HC, vec![tokens, HC]);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_hc_head_weights(
            &mut encoder,
            &mut registry,
            &device,
            &mixes,
            &scale,
            &base,
            &weights,
            tokens as u32,
        )
        .expect("head weights dispatch");
        encoder.commit_and_wait().expect("head weights completion");
        assert_close(weights.as_slice().unwrap(), &expected, 2e-6, "head weights");
    }
}

#[test]
fn pre_and_post_match_cpu_for_llama_representative_shapes() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    for (embd, tokens) in [(1usize, 1usize), (31, 17), (128, 257), (4096, 21)] {
        let x_pre = values(tokens * HC * embd, 5, 0.008);
        let weights = values(tokens * HC, 6, 0.01);
        let expected_pre = pre_reference(&x_pre, &weights, tokens, embd);
        let x_pre = buffer(&device, &x_pre, vec![tokens, HC, embd]);
        let weights = buffer(&device, &weights, vec![tokens, HC]);
        let pre_out = output(&device, tokens * embd, vec![tokens, embd]);

        let x = values(tokens * embd, 7, 0.009);
        let residual = values(tokens * HC * embd, 8, 0.007);
        let post = values(tokens * HC, 9, 0.01)
            .into_iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>();
        let mixes = values(tokens * MIX, 10, 0.02);
        let scale = [-0.3, 0.15, 0.4];
        let base = values(MIX, 11, 0.004);
        let (_, _, comb) = split_reference(&mixes, &scale, &base);
        let expected_post = post_reference(&x, &residual, &post, &comb, tokens, embd);
        let x = buffer(&device, &x, vec![tokens, embd]);
        let residual = buffer(&device, &residual, vec![tokens, HC, embd]);
        let post = buffer(&device, &post, vec![tokens, HC]);
        let comb = buffer(&device, &comb, vec![tokens, HC, HC]);
        let post_out = output(&device, tokens * HC * embd, vec![tokens, HC, embd]);

        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_hc_pre(
            &mut encoder,
            &mut registry,
            &device,
            &x_pre,
            &weights,
            &pre_out,
            tokens as u32,
            embd as u32,
        )
        .expect("pre dispatch");
        dispatch_hc_post(
            &mut encoder,
            &mut registry,
            &device,
            &x,
            &residual,
            &post,
            &comb,
            &post_out,
            tokens as u32,
            embd as u32,
        )
        .expect("post dispatch");
        encoder.commit_and_wait().expect("pre/post completion");
        assert_close(pre_out.as_slice().unwrap(), &expected_pre, 3e-5, "hc_pre");
        assert_close(
            post_out.as_slice().unwrap(),
            &expected_post,
            4e-5,
            "hc_post",
        );
    }
}

#[test]
fn invalid_shapes_and_dtypes_are_rejected_before_encoding() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    let mixes = output(&device, MIX, vec![1, MIX]);
    let scale = output(&device, 3, vec![3]);
    let base = output(&device, MIX, vec![MIX]);
    let pre_wrong = output(&device, HC, vec![HC]);
    let post = output(&device, HC, vec![1, HC]);
    let comb = output(&device, HC * HC, vec![1, HC, HC]);
    let mut encoder = device.command_encoder().expect("encoder");
    assert!(dispatch_hc_split_sinkhorn(
        &mut encoder,
        &mut registry,
        &device,
        &mixes,
        &scale,
        &base,
        &pre_wrong,
        &post,
        &comb,
        1,
    )
    .is_err());

    let x = output(&device, HC, vec![1, HC, 1]);
    let weights_u32 = device
        .alloc_buffer(HC * 4, DType::U32, vec![1, HC])
        .expect("u32 weights");
    let out = output(&device, 1, vec![1, 1]);
    assert!(dispatch_hc_pre(
        &mut encoder,
        &mut registry,
        &device,
        &x,
        &weights_u32,
        &out,
        1,
        1,
    )
    .is_err());
    let head_scale_wrong = output(&device, 2, vec![2]);
    assert!(dispatch_hc_head_weights(
        &mut encoder,
        &mut registry,
        &device,
        &post,
        &head_scale_wrong,
        &base,
        &post,
        1,
    )
    .is_err());
    assert!(dispatch_hc_pre(&mut encoder, &mut registry, &device, &x, &post, &out, 0, 1,).is_err());

    let x_post = output(&device, 1, vec![1, 1]);
    let comb_wrong = output(&device, HC * HC, vec![HC, HC]);
    let post_out = output(&device, HC, vec![1, HC, 1]);
    assert!(dispatch_hc_post(
        &mut encoder,
        &mut registry,
        &device,
        &x_post,
        &x,
        &post,
        &comb_wrong,
        &post_out,
        1,
        1,
    )
    .is_err());
}

#[test]
fn non_finite_inputs_fail_closed_to_finite_zero_outputs() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    let mut mixes_data = values(2 * MIX, 12, 0.02);
    mixes_data[3] = f32::NAN;
    let mixes = buffer(&device, &mixes_data, vec![2, MIX]);
    let scale = buffer(&device, &[0.2, 0.3, 0.4], vec![3]);
    let base = buffer(&device, &vec![0.0; MIX], vec![MIX]);
    let pre = output(&device, 2 * HC, vec![2, HC]);
    let post = output(&device, 2 * HC, vec![2, HC]);
    let comb = output(&device, 2 * HC * HC, vec![2, HC, HC]);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_hc_split_sinkhorn(
        &mut encoder,
        &mut registry,
        &device,
        &mixes,
        &scale,
        &base,
        &pre,
        &post,
        &comb,
        2,
    )
    .expect("guarded split");
    encoder.commit_and_wait().expect("guarded completion");
    assert!(pre.as_slice::<f32>().unwrap()[..HC]
        .iter()
        .all(|v| *v == 0.0));
    assert!(post.as_slice::<f32>().unwrap()[..HC]
        .iter()
        .all(|v| *v == 0.0));
    assert!(comb.as_slice::<f32>().unwrap()[..HC * HC]
        .iter()
        .all(|v| *v == 0.0));
    assert!(pre.as_slice::<f32>().unwrap()[HC..]
        .iter()
        .all(|v| v.is_finite()));

    let head_mixes = buffer(&device, &[0.0, f32::NAN, 1.0, -1.0], vec![1, HC]);
    let head_scale = buffer(&device, &[0.5], vec![1]);
    let head_base = buffer(&device, &[0.0; HC], vec![HC]);
    let head_weights = output(&device, HC, vec![1, HC]);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_hc_head_weights(
        &mut encoder,
        &mut registry,
        &device,
        &head_mixes,
        &head_scale,
        &head_base,
        &head_weights,
        1,
    )
    .expect("guarded head weights");
    encoder.commit_and_wait().expect("guarded head completion");
    let head_values = head_weights.as_slice::<f32>().unwrap();
    assert!(head_values.iter().all(|value| *value == 0.0));

    let mut x_data = vec![1.0; HC * 8];
    x_data[8 + 2] = f32::INFINITY;
    let x = buffer(&device, &x_data, vec![1, HC, 8]);
    let weights = buffer(&device, &[0.25; HC], vec![1, HC]);
    let out = output(&device, 8, vec![1, 8]);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_hc_pre(
        &mut encoder,
        &mut registry,
        &device,
        &x,
        &weights,
        &out,
        1,
        8,
    )
    .expect("guarded pre");
    encoder.commit_and_wait().expect("guarded pre completion");
    assert_eq!(out.as_slice::<f32>().unwrap()[2], 0.0);
    assert!(out.as_slice::<f32>().unwrap().iter().all(|v| v.is_finite()));

    let x_post = buffer(&device, &[1.0; 8], vec![1, 8]);
    let mut residual_data = vec![0.5; HC * 8];
    residual_data[8 + 3] = f32::NAN;
    let residual = buffer(&device, &residual_data, vec![1, HC, 8]);
    let post_coeff = buffer(&device, &[1.0; HC], vec![1, HC]);
    let comb_coeff = buffer(&device, &[0.25; HC * HC], vec![1, HC, HC]);
    let post_out = output(&device, HC * 8, vec![1, HC, 8]);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_hc_post(
        &mut encoder,
        &mut registry,
        &device,
        &x_post,
        &residual,
        &post_coeff,
        &comb_coeff,
        &post_out,
        1,
        8,
    )
    .expect("guarded post");
    encoder.commit_and_wait().expect("guarded post completion");
    let post_values = post_out.as_slice::<f32>().unwrap();
    assert!((0..HC).all(|destination| post_values[destination * 8 + 3] == 0.0));
    assert!(post_values.iter().all(|value| value.is_finite()));
}
