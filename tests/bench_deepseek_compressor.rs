//! Opt-in DeepSeek-V4 compressor latency receipt.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use std::time::Instant;

#[test]
#[ignore = "performance gate"]
fn benchmark_ratio4_and_ratio128_prefill() {
    let device = MlxDevice::new().unwrap();
    for (ratio, seq) in [(4usize, 128usize), (128, 256)] {
        let dim = 512usize;
        let coff = if ratio == 4 { 2 } else { 1 };
        let projected = coff * dim;
        let kv = device
            .alloc_buffer(seq * projected * 4, DType::F32, vec![1, seq, projected])
            .unwrap();
        let score = device
            .alloc_buffer(seq * projected * 4, DType::F32, vec![1, seq, projected])
            .unwrap();
        let ape = device
            .alloc_buffer(ratio * projected * 4, DType::F32, vec![ratio, projected])
            .unwrap();
        let mut norm = device.alloc_buffer(dim * 4, DType::F32, vec![dim]).unwrap();
        norm.as_mut_slice::<f32>().unwrap().fill(1.0);
        let state_shape = vec![1, coff * ratio, projected];
        let kv_state = device
            .alloc_buffer(
                coff * ratio * projected * 4,
                DType::F32,
                state_shape.clone(),
            )
            .unwrap();
        let score_state = device
            .alloc_buffer(coff * ratio * projected * 4, DType::F32, state_shape)
            .unwrap();
        let output = device
            .alloc_buffer(
                seq / ratio * dim * 2,
                DType::BF16,
                vec![1, seq / ratio, dim],
            )
            .unwrap();
        let cache = device
            .alloc_buffer(
                seq / ratio * dim * 2,
                DType::BF16,
                vec![1, seq / ratio, dim],
            )
            .unwrap();
        let params = DeepSeekCompressorParams {
            batch: 1,
            seq_len: seq as u32,
            start_pos: 0,
            ratio: ratio as u32,
            head_dim: dim as u32,
            cache_len: (seq / ratio) as u32,
            epsilon: 1e-6,
            write_cache: 1,
        };
        let mut registry = KernelRegistry::new();
        let start = Instant::now();
        for _ in 0..20 {
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
        }
        println!(
            "ratio={ratio} seq={seq} avg_ms={:.3}",
            start.elapsed().as_secs_f64() * 50.0
        );
    }
}
