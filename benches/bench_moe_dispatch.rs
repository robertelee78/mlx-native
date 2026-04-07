//! Criterion benchmark for the MoE expert dispatch kernel.
//!
//! Uses Gemma 4 MoE dimensions:
//! - 8 selected experts
//! - input_dim = 2816
//! - intermediate_dim = 704

use criterion::{criterion_group, criterion_main, Criterion};
use mlx_native::ops::moe_dispatch::{self, ExpertWeights, MoeDispatchParams};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// Allocate an f32 Metal buffer filled with a constant value.
fn alloc_f32_buffer(device: &MlxDevice, shape: Vec<usize>, fill: f32) -> MlxBuffer {
    let n: usize = shape.iter().copied().product();
    let byte_len = n * 4;
    let mut buf = device.alloc_buffer(byte_len, DType::F32, shape).unwrap();
    {
        let slice: &mut [f32] = buf.as_mut_slice().unwrap();
        for v in slice.iter_mut() {
            *v = fill;
        }
    }
    buf
}

fn bench_moe_dispatch_gemma4(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping benchmarks");
            return;
        }
    };
    let mut registry = KernelRegistry::new();

    // Gemma 4 MoE dimensions.
    let input_dim = 2816usize;
    let intermediate_dim = 704usize;
    let n_selected = 8usize;

    // Allocate input and output buffers.
    let input = alloc_f32_buffer(&device, vec![input_dim], 0.01);
    let output = alloc_f32_buffer(&device, vec![input_dim], 0.0);

    // Scratch buffers.
    let scratch_gate = alloc_f32_buffer(&device, vec![intermediate_dim], 0.0);
    let scratch_up = alloc_f32_buffer(&device, vec![intermediate_dim], 0.0);
    let scratch_hidden = alloc_f32_buffer(&device, vec![intermediate_dim], 0.0);
    let scratch_expert = alloc_f32_buffer(&device, vec![input_dim], 0.0);

    // Expert weights: 8 experts, each with gate/up/down projections.
    let mut gate_projs = Vec::with_capacity(n_selected);
    let mut up_projs = Vec::with_capacity(n_selected);
    let mut down_projs = Vec::with_capacity(n_selected);

    for _ in 0..n_selected {
        gate_projs.push(alloc_f32_buffer(
            &device,
            vec![input_dim, intermediate_dim],
            0.001,
        ));
        up_projs.push(alloc_f32_buffer(
            &device,
            vec![input_dim, intermediate_dim],
            0.001,
        ));
        down_projs.push(alloc_f32_buffer(
            &device,
            vec![intermediate_dim, input_dim],
            0.001,
        ));
    }

    let expert_weights: Vec<ExpertWeights<'_>> = (0..n_selected)
        .map(|i| ExpertWeights {
            gate_proj: &gate_projs[i],
            up_proj: &up_projs[i],
            down_proj: &down_projs[i],
        })
        .collect();

    // Uniform routing weights (1/n_selected each).
    let routing_weights: Vec<f32> = vec![1.0 / n_selected as f32; n_selected];

    let params = MoeDispatchParams {
        input_dim,
        intermediate_dim,
        n_selected,
    };

    // Warm up: compile all pipelines.
    {
        let mut enc = device.command_encoder().unwrap();
        moe_dispatch::moe_dispatch(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &input,
            &expert_weights,
            &routing_weights,
            &output,
            &scratch_gate,
            &scratch_up,
            &scratch_hidden,
            &scratch_expert,
            &params,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }

    c.bench_function(
        &format!(
            "moe_dispatch_{n_selected}experts_[{input_dim}]->[{intermediate_dim}]->[{input_dim}]"
        ),
        |b| {
            b.iter(|| {
                let mut enc = device.command_encoder().unwrap();
                moe_dispatch::moe_dispatch(
                    &mut enc,
                    &mut registry,
                    device.metal_device(),
                    &input,
                    &expert_weights,
                    &routing_weights,
                    &output,
                    &scratch_gate,
                    &scratch_up,
                    &scratch_hidden,
                    &scratch_expert,
                    &params,
                )
                .unwrap();
                enc.commit_and_wait().unwrap();
            });
        },
    );
}

criterion_group!(benches, bench_moe_dispatch_gemma4);
criterion_main!(benches);
