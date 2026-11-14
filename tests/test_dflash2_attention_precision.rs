#![cfg(target_os = "macos")]

use half::bf16;
use mlx_native::ops::flash_attn_prefill::{
    dispatch_flash_attn_prefill_bf16_d128, register, FlashAttnPrefillLayout,
    FlashAttnPrefillParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn buffer(device: &MlxDevice, values: &[bf16], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 2, DType::BF16, shape)
        .expect("BF16 buffer");
    buffer
        .as_mut_slice::<bf16>()
        .expect("writable BF16 buffer")
        .copy_from_slice(values);
    buffer
}

#[test]
fn bf16_d128_preserves_query_cancellation_before_scaling() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    let (heads, rows, keys, dim) = (4, 8, 2, 128);
    let params = FlashAttnPrefillParams {
        n_heads: heads as u32,
        n_kv_heads: 1,
        head_dim: dim as u32,
        seq_len_q: rows as u32,
        seq_len_k: keys as u32,
        batch: 1,
        scale: 1.0 / (dim as f32).sqrt(),
        do_causal: false,
    };

    for head_major in [true, false] {
        let index = |head, row, d| {
            if head_major {
                (head * rows + row) * dim + d
            } else {
                (row * heads + head) * dim + d
            }
        };
        let mut query = vec![bf16::ZERO; heads * rows * dim];
        for head in 0..heads {
            for row in 0..rows {
                for d in 0..dim {
                    // At row/head zero each pair's unscaled dot cancels
                    // exactly. Scaling and rounding Q first destroys that
                    // cancellation despite nonzero, exactly BF16 inputs.
                    let value = if d % 2 == 0 {
                        1.0
                    } else {
                        1.0 + (1 + (row + head) % rows) as f32 / 128.0
                    };
                    query[index(head, row, d)] = bf16::from_f32(value);
                }
            }
        }
        let mut key = vec![bf16::ZERO; keys * dim];
        let mut value = vec![bf16::ZERO; keys * dim];
        for k in 0..keys {
            let sign = if k == 0 { 1.0 } else { -1.0 };
            for d in 0..dim {
                key[k * dim + d] = bf16::from_f32(
                    sign * if d % 2 == 0 { 1.0 + 1.0 / 128.0 } else { -1.0 },
                );
                value[k * dim + d] = bf16::from_f32(sign * (1.0 + (d % 7) as f32 / 8.0));
            }
        }
        let query_shape = if head_major {
            vec![1, heads, rows, dim]
        } else {
            vec![1, rows, heads, dim]
        };
        let kv_shape = if head_major {
            vec![1, 1, keys, dim]
        } else {
            vec![1, keys, 1, dim]
        };
        let q = buffer(&device, &query, query_shape.clone());
        let k = buffer(&device, &key, kv_shape.clone());
        let v = buffer(&device, &value, kv_shape);
        let out = buffer(&device, &vec![bf16::NAN; query.len()], query_shape);
        let layout = if head_major {
            FlashAttnPrefillLayout::HeadMajor
        } else {
            FlashAttnPrefillLayout::SeqMajor
        };
        let mut encoder = device.command_encoder().expect("command encoder");
        dispatch_flash_attn_prefill_bf16_d128(
            &mut encoder, &device, &mut registry, &q, &k, &v, None, &out, &params, layout,
        )
        .expect("D128 attention dispatch");
        encoder.commit_and_wait().expect("checked attention completion");
        let actual = out.as_slice::<bf16>().expect("attention output");
        for head in 0..heads {
            for row in 0..rows {
                // Independent F64 dot, scaling, softmax and value reduction
                // from the exact BF16 inputs, not the shader's load order.
                let mut scores = [0.0_f64; 2];
                for k in 0..keys {
                    for d in 0..dim {
                        scores[k] += query[index(head, row, d)].to_f64()
                            * key[k * dim + d].to_f64();
                    }
                    scores[k] *= params.scale as f64;
                }
                let maximum = scores[0].max(scores[1]);
                let probabilities = scores.map(|score| (score - maximum).exp());
                let denominator = probabilities.iter().sum::<f64>();
                for d in 0..dim {
                    let expected = (0..keys)
                        .map(|k| probabilities[k] * value[k * dim + d].to_f64())
                        .sum::<f64>()
                        / denominator;
                    let observed = actual[index(head, row, d)].to_f64();
                    let bound = 2.0e-6_f64.max(expected.abs() / 128.0);
                    assert!(
                        observed.is_finite() && (observed - expected).abs() <= bound,
                        "head_major={head_major} head={head} row={row} d={d}: \
                         actual={observed} expected={expected} bound={bound}"
                    );
                }
            }
        }
    }
}
