//! Decode/prefill benchmark for the production DeepSeek-V4 HC pipeline.
//!
//! Run with:
//! `cargo test --release --test bench_deepseek_hyper_connection -- --ignored --nocapture`

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn, register,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const HC: usize = 4;
const MIX: usize = 24;
const EMBED: usize = 4096;
const WARMUP: usize = 5;
const SAMPLES: usize = 25;

fn filled(device: &MlxDevice, len: usize, shape: Vec<usize>, value: f32) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, shape)
        .expect("allocate benchmark buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("map benchmark buffer")
        .fill(value);
    buffer
}

fn bench_shape(device: &MlxDevice, registry: &mut KernelRegistry, tokens: usize, label: &str) {
    let mixes = filled(device, tokens * MIX, vec![tokens, MIX], 0.01);
    let scale = filled(device, 3, vec![3], 0.2);
    let base = filled(device, MIX, vec![MIX], 0.0);
    let pre = filled(device, tokens * HC, vec![tokens, HC], 0.0);
    let post = filled(device, tokens * HC, vec![tokens, HC], 0.0);
    let comb = filled(device, tokens * HC * HC, vec![tokens, HC, HC], 0.0);
    let residual = filled(device, tokens * HC * EMBED, vec![tokens, HC, EMBED], 0.01);
    let reduced = filled(device, tokens * EMBED, vec![tokens, EMBED], 0.0);
    let expanded = filled(device, tokens * HC * EMBED, vec![tokens, HC, EMBED], 0.0);

    let mut run = || {
        let mut encoder = device.command_encoder().expect("benchmark encoder");
        dispatch_hc_split_sinkhorn(
            &mut encoder,
            registry,
            device,
            &mixes,
            &scale,
            &base,
            &pre,
            &post,
            &comb,
            tokens as u32,
        )
        .expect("split dispatch");
        encoder.memory_barrier();
        dispatch_hc_pre(
            &mut encoder,
            registry,
            device,
            &residual,
            &pre,
            &reduced,
            tokens as u32,
            EMBED as u32,
        )
        .expect("pre dispatch");
        encoder.memory_barrier();
        dispatch_hc_post(
            &mut encoder,
            registry,
            device,
            &reduced,
            &residual,
            &post,
            &comb,
            &expanded,
            tokens as u32,
            EMBED as u32,
        )
        .expect("post dispatch");
        encoder.commit_and_wait().expect("benchmark completion");
    };

    for _ in 0..WARMUP {
        run();
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = std::time::Instant::now();
        run();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[SAMPLES / 2];
    let p10 = samples[SAMPLES / 10];
    let p90 = samples[SAMPLES * 9 / 10];
    let tokens_per_second = tokens as f64 / (median * 1e-6);
    println!(
        "DeepSeek-V4 HC {label}: tokens={tokens} embd={EMBED} median={median:.2} us \
         p10={p10:.2} us p90={p90:.2} us throughput={tokens_per_second:.1} tok/s"
    );
}

#[test]
#[ignore = "performance benchmark; requires an idle Apple GPU"]
fn benchmark_decode_and_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    register(&mut registry);
    bench_shape(&device, &mut registry, 1, "decode");
    bench_shape(&device, &mut registry, 257, "prefill");
}
