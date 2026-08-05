#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::deepseek_sparse_prefill_mask::{
    dispatch_deepseek_sparse_prefill_mask, DeepSeekSparsePrefillMaskParams,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn selected_positions_form_additive_mask() {
    let device = MlxDevice::new().unwrap();
    let queries = 2;
    let kv = 4;
    let top_k = 3;
    let heads = 2;
    let mut indices = device
        .alloc_buffer(queries * top_k * 4, DType::I32, vec![1, queries, top_k])
        .unwrap();
    indices
        .as_mut_slice::<i32>()
        .unwrap()
        .copy_from_slice(&[0, -1, -1, 0, 2, -1]);
    let mask = device
        .alloc_buffer(
            heads * queries * kv * 2,
            DType::BF16,
            vec![1, heads, queries, kv],
        )
        .unwrap();
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_sparse_prefill_mask(
        &mut encoder,
        &mut registry,
        &device,
        &indices,
        &mask,
        &DeepSeekSparsePrefillMaskParams {
            batch: 1,
            query_len: queries as u32,
            kv_len: kv as u32,
            top_k: top_k as u32,
            heads: heads as u32,
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let values = mask.as_slice::<bf16>().unwrap();
    for head in 0..heads {
        for query in 0..queries {
            let row = &values
                [(head * queries + query) * kv..(head * queries + query + 1) * kv];
            assert_eq!(row[0].to_f32(), 0.0);
            assert_eq!(row[2].to_f32(), if query == 1 { 0.0 } else { f32::NEG_INFINITY });
        }
    }
}
