#![cfg(target_vendor = "apple")]

use half::{bf16, f16};
use mlx_native::ops::deepseek_sparse_prefill_mask::{
    self, dispatch_deepseek_sparse_prefill_mask, dispatch_deepseek_sparse_prefill_mask_f16,
    DeepSeekSparsePrefillMaskParams,
};
use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams;
use mlx_native::ops::flash_attn_prefill_blk::{
    self, dispatch_flash_attn_prefill_blk, dispatch_flash_attn_prefill_blk_f16, BlkParams,
};
use mlx_native::ops::flash_attn_prefill_d512::{
    self, dispatch_flash_attn_prefill_bf16_d512_with_blk_and_sinks,
    dispatch_flash_attn_prefill_bf16_d512_with_sinks,
    dispatch_flash_attn_prefill_f16_d512_with_blk_and_sinks,
    dispatch_flash_attn_prefill_f16_d512_with_sinks,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn run_d512_skip_map_parity(dtype: DType) {
    let device = MlxDevice::new().unwrap();
    let rows = 16usize;
    let heads = 64usize;
    let kv_len = 512usize;
    let head_dim = 512usize;
    let top_k = 130usize;

    let mut q = device
        .alloc_buffer(
            heads * rows * head_dim * 2,
            dtype,
            vec![1, heads, rows, head_dim],
        )
        .unwrap();
    let mut kv = device
        .alloc_buffer(kv_len * head_dim * 2, dtype, vec![1, 1, kv_len, head_dim])
        .unwrap();
    match dtype {
        DType::F16 => {
            for (index, value) in q.as_mut_slice::<f16>().unwrap().iter_mut().enumerate() {
                *value = f16::from_f32(((index % 97) as f32 - 48.0) * 0.002);
            }
            for (index, value) in kv.as_mut_slice::<f16>().unwrap().iter_mut().enumerate() {
                *value = f16::from_f32(((index % 89) as f32 - 44.0) * 0.003);
            }
        }
        DType::BF16 => {
            for (index, value) in q.as_mut_slice::<bf16>().unwrap().iter_mut().enumerate() {
                *value = bf16::from_f32(((index % 97) as f32 - 48.0) * 0.002);
            }
            for (index, value) in kv.as_mut_slice::<bf16>().unwrap().iter_mut().enumerate() {
                *value = bf16::from_f32(((index % 89) as f32 - 44.0) * 0.003);
            }
        }
        other => panic!("unsupported test dtype {other:?}"),
    }
    let mut sinks = device
        .alloc_buffer(heads * 4, DType::F32, vec![heads])
        .unwrap();
    sinks.as_mut_slice::<f32>().unwrap().fill(0.125);

    let mut indices = device
        .alloc_buffer(rows * top_k * 4, DType::I32, vec![1, rows, top_k])
        .unwrap();
    for (row, row_indices) in indices
        .as_mut_slice::<i32>()
        .unwrap()
        .chunks_exact_mut(top_k)
        .enumerate()
    {
        for (slot, index) in row_indices.iter_mut().enumerate() {
            *index = ((slot * 17 + row * 5) % 192) as i32;
        }
    }
    let mask = device
        .alloc_buffer(rows * kv_len * 2, dtype, vec![rows, kv_len])
        .unwrap();
    let blk_rows = rows.div_ceil(8);
    let blk_cols = kv_len.div_ceil(64);
    let blk = device
        .alloc_buffer(blk_rows * blk_cols, DType::U8, vec![blk_rows, blk_cols])
        .unwrap();

    let mut registry = KernelRegistry::new();
    deepseek_sparse_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);
    flash_attn_prefill_d512::register(&mut registry);
    let mut prep = device.command_encoder().unwrap();
    let mask_params = DeepSeekSparsePrefillMaskParams {
        batch: 1,
        query_len: rows as u32,
        kv_len: kv_len as u32,
        top_k: top_k as u32,
        heads: heads as u32,
    };
    match dtype {
        DType::F16 => dispatch_deepseek_sparse_prefill_mask_f16(
            &mut prep,
            &mut registry,
            &device,
            &indices,
            &mask,
            &mask_params,
        ),
        DType::BF16 => dispatch_deepseek_sparse_prefill_mask(
            &mut prep,
            &mut registry,
            &device,
            &indices,
            &mask,
            &mask_params,
        ),
        other => panic!("unsupported test dtype {other:?}"),
    }
    .unwrap();
    prep.memory_barrier();
    let blk_params = BlkParams {
        seq_len_q: rows as u32,
        seq_len_k: kv_len as u32,
        bq: 8,
        bk: 64,
    };
    match dtype {
        DType::F16 => dispatch_flash_attn_prefill_blk_f16(
            &mut prep,
            &device,
            &mut registry,
            &mask,
            &blk,
            &blk_params,
        ),
        DType::BF16 => dispatch_flash_attn_prefill_blk(
            &mut prep,
            &device,
            &mut registry,
            &mask,
            &blk,
            &blk_params,
        ),
        other => panic!("unsupported test dtype {other:?}"),
    }
    .unwrap();
    prep.commit_and_wait().unwrap();
    let classifications = blk.as_slice::<u8>().unwrap();
    assert!(classifications.contains(&0), "expected fully masked tiles");
    assert!(classifications.contains(&1), "expected mixed sparse tiles");

    let out_without = device
        .alloc_buffer(
            heads * rows * head_dim * 2,
            dtype,
            vec![1, heads, rows, head_dim],
        )
        .unwrap();
    let out_with = device
        .alloc_buffer(
            heads * rows * head_dim * 2,
            dtype,
            vec![1, heads, rows, head_dim],
        )
        .unwrap();
    let params = FlashAttnPrefillParams {
        n_heads: heads as u32,
        n_kv_heads: 1,
        head_dim: head_dim as u32,
        seq_len_q: rows as u32,
        seq_len_k: kv_len as u32,
        batch: 1,
        scale: 1.0 / (head_dim as f32).sqrt(),
        do_causal: false,
    };

    let mut plain = device.command_encoder().unwrap();
    match dtype {
        DType::F16 => dispatch_flash_attn_prefill_f16_d512_with_sinks(
            &mut plain,
            &device,
            &mut registry,
            &q,
            &kv,
            &kv,
            Some(&mask),
            &sinks,
            &out_without,
            &params,
        ),
        DType::BF16 => dispatch_flash_attn_prefill_bf16_d512_with_sinks(
            &mut plain,
            &device,
            &mut registry,
            &q,
            &kv,
            &kv,
            Some(&mask),
            &sinks,
            &out_without,
            &params,
        ),
        other => panic!("unsupported test dtype {other:?}"),
    }
    .unwrap();
    plain.commit_and_wait().unwrap();

    let mut skipped = device.command_encoder().unwrap();
    match dtype {
        DType::F16 => dispatch_flash_attn_prefill_f16_d512_with_blk_and_sinks(
            &mut skipped,
            &device,
            &mut registry,
            &q,
            &kv,
            &kv,
            &mask,
            &blk,
            &sinks,
            &out_with,
            &params,
        ),
        DType::BF16 => dispatch_flash_attn_prefill_bf16_d512_with_blk_and_sinks(
            &mut skipped,
            &device,
            &mut registry,
            &q,
            &kv,
            &kv,
            &mask,
            &blk,
            &sinks,
            &out_with,
            &params,
        ),
        other => panic!("unsupported test dtype {other:?}"),
    }
    .unwrap();
    skipped.commit_and_wait().unwrap();

    match dtype {
        DType::F16 => {
            let plain_values = out_without.as_slice::<f16>().unwrap();
            let skipped_values = out_with.as_slice::<f16>().unwrap();
            assert_eq!(plain_values.len(), skipped_values.len());
            for (index, (plain, skipped)) in plain_values.iter().zip(skipped_values).enumerate() {
                assert_eq!(
                    plain.to_bits(),
                    skipped.to_bits(),
                    "F16 D512 skip output drift at element {index}"
                );
            }
        }
        DType::BF16 => {
            let plain_values = out_without.as_slice::<bf16>().unwrap();
            let skipped_values = out_with.as_slice::<bf16>().unwrap();
            assert_eq!(plain_values.len(), skipped_values.len());
            for (index, (plain, skipped)) in plain_values.iter().zip(skipped_values).enumerate() {
                assert_eq!(
                    plain.to_bits(),
                    skipped.to_bits(),
                    "BF16 D512 skip output drift at element {index}"
                );
            }
        }
        other => panic!("unsupported test dtype {other:?}"),
    }
}

#[test]
fn f16_d512_skip_map_is_byte_identical_with_sinks() {
    run_d512_skip_map_parity(DType::F16);
}

#[test]
fn bf16_d512_skip_map_is_byte_identical_with_sinks() {
    run_d512_skip_map_parity(DType::BF16);
}
