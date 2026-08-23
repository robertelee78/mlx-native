//! Short-row native BF16 projection spike.
//!
//! Compares the current tensor tile, the existing large tensor tile, the
//! row-wise GEMV kernel, and a weight-reusing short-row GEMV candidate.

use half::bf16;
use mlx_native::ops::dense_gemv_bf16::{dense_gemv_bf16_f32, dense_gemv_bf16_f32_tiled4};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const WARMUP: usize = 4;
const ROUNDS: usize = 21;
const REPEATS_PER_CB: usize = 1;

#[derive(Clone, Copy, Debug)]
enum Route {
    Tensor32,
    Tensor128,
    RowGemv,
    TiledGemv4,
}

impl Route {
    const ALL: [Self; 4] = [
        Self::Tensor32,
        Self::Tensor128,
        Self::RowGemv,
        Self::TiledGemv4,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Tensor32 => "tensor32",
            Self::Tensor128 => "tensor128",
            Self::RowGemv => "row_gemv",
            Self::TiledGemv4 => "tiled_gemv4",
        }
    }
}

struct Shape {
    label: String,
    m: u32,
    n: u32,
    k: u32,
}

const BASE_SHAPES: &[(&str, u32, u32)] = &[
    ("ffn_gate_up", 17_408, 5_120),
    ("ffn_down", 5_120, 17_408),
    ("attn_q", 6_144, 5_120),
    ("attn_kv", 1_024, 5_120),
];

fn shapes() -> Vec<Shape> {
    BASE_SHAPES
        .iter()
        .flat_map(|&(label, n, k)| {
            (1..=16).map(move |m| Shape {
                label: format!("{label}_m{m}"),
                m,
                n,
                k,
            })
        })
        .collect()
}

fn alloc_filled_bf16(device: &MlxDevice, elements: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("allocate BF16 weight");
    for (index, value) in buffer
        .as_mut_slice::<u16>()
        .expect("map BF16 weight")
        .iter_mut()
        .enumerate()
    {
        let sample = ((index.wrapping_mul(17) % 257) as f32 - 128.0) / 4096.0;
        *value = bf16::from_f32(sample).to_bits();
    }
    buffer
}

fn alloc_filled_f32(device: &MlxDevice, elements: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("allocate F32 input");
    for (index, value) in buffer
        .as_mut_slice::<f32>()
        .expect("map F32 input")
        .iter_mut()
        .enumerate()
    {
        let coarse = (index.wrapping_mul(29) % 251) as f32 - 125.0;
        let fine = (index.wrapping_mul(43) % 19) as f32 - 9.0;
        *value = coarse / 1003.0 + fine / 17_003.0;
    }
    buffer
}

fn alloc_output(device: &MlxDevice, elements: usize) -> MlxBuffer {
    device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("allocate F32 output")
}

fn encode(
    route: Route,
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) {
    match route {
        Route::Tensor32 => {
            std::env::remove_var("HF2Q_LARGE_TILE_MM");
            dense_matmul_bf16_f32_tensor(encoder, registry, device, weight, input, output, params)
                .expect("tensor32 dispatch");
        }
        Route::Tensor128 => {
            std::env::set_var("HF2Q_LARGE_TILE_MM", "1");
            dense_matmul_bf16_f32_tensor(encoder, registry, device, weight, input, output, params)
                .expect("tensor128 dispatch");
        }
        Route::RowGemv => {
            dense_gemv_bf16_f32(encoder, registry, device, weight, input, output, params)
                .expect("row GEMV dispatch")
        }
        Route::TiledGemv4 => {
            dense_gemv_bf16_f32_tiled4(encoder, registry, device, weight, input, output, params)
                .expect("batched GEMV4 dispatch")
        }
    }
}

fn run_once(
    route: Route,
    repeats: usize,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> (f64, f64) {
    let mut encoder = device.command_encoder().expect("command encoder");
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        encode(
            route,
            &mut encoder,
            registry,
            device,
            weight,
            input,
            output,
            params,
        );
    }
    let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time().expect("GPU completion");
    (
        started.elapsed().as_secs_f64() * 1e6 / repeats as f64,
        (gpu_end - gpu_start) * 1e6 / repeats as f64,
    )
}

#[derive(Clone, Copy)]
struct Distribution {
    p25: f64,
    median: f64,
    p75: f64,
}

fn distribution(mut values: Vec<f64>) -> Distribution {
    assert_eq!(values.len(), ROUNDS, "incomplete benchmark receipt");
    values.sort_by(f64::total_cmp);
    Distribution {
        p25: values[(values.len() - 1) / 4],
        median: values[values.len() / 2],
        p75: values[(values.len() - 1) * 3 / 4],
    }
}

fn output_for(
    route: Route,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Vec<f32> {
    let output = alloc_output(device, (params.m * params.n) as usize);
    run_once(route, 1, registry, device, weight, input, &output, params);
    output.as_slice::<f32>().expect("read output").to_vec()
}

fn main() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let shape_filter = std::env::var("MLX_BENCH_SHAPE_FILTER").ok();
    let summary_only = std::env::var("MLX_BENCH_SUMMARY_ONLY").as_deref() == Ok("1");
    let winner_only = std::env::var("MLX_BENCH_WINNER_ONLY").as_deref() == Ok("1");

    println!(
        "commit={} rounds={} repeats={}",
        option_env!("MLX_NATIVE_BENCH_COMMIT").unwrap_or("unbound"),
        ROUNDS,
        REPEATS_PER_CB
    );
    for shape in shapes() {
        if shape_filter
            .as_deref()
            .is_some_and(|filter| !shape.label.contains(filter))
        {
            continue;
        }
        let params = DenseMmBf16F32Params {
            m: shape.m,
            n: shape.n,
            k: shape.k,
            src0_batch: 1,
            src1_batch: 1,
        };
        let weight = alloc_filled_bf16(&device, (shape.n * shape.k) as usize);
        let input = alloc_filled_f32(&device, (shape.m * shape.k) as usize);
        let outputs: Vec<(Route, MlxBuffer)> = Route::ALL
            .into_iter()
            .map(|route| (route, alloc_output(&device, (shape.m * shape.n) as usize)))
            .collect();

        let reference = output_for(
            Route::Tensor32,
            &mut registry,
            &device,
            &weight,
            &input,
            &params,
        );
        let row_reference = output_for(
            Route::RowGemv,
            &mut registry,
            &device,
            &weight,
            &input,
            &params,
        );
        let reference_l1 = reference
            .iter()
            .map(|value| value.abs() as f64)
            .sum::<f64>();
        let reference_max = reference.iter().copied().fold(0.0f32, |acc, value| {
            assert!(value.is_finite(), "non-finite reference output");
            acc.max(value.abs())
        });
        assert!(reference_l1 > 1e-3, "reference output is vacuously zero");
        if !summary_only {
            println!(
                "shape={} reference_l1={reference_l1:.9} reference_max_abs={reference_max:.9}",
                shape.label
            );
        }
        for route in Route::ALL {
            let candidate = output_for(route, &mut registry, &device, &weight, &input, &params);
            let max_abs = reference
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let bit_differences = reference
                .iter()
                .zip(candidate.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs_vs_row = row_reference
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let bit_differences_vs_row = row_reference
                .iter()
                .zip(candidate.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if !summary_only {
                println!(
                    "shape={} route={} max_abs_vs_tensor32={max_abs:.9} bit_differences={bit_differences} max_abs_vs_row={max_abs_vs_row:.9} bit_differences_vs_row={bit_differences_vs_row}",
                    shape.label,
                    route.label()
                );
            }
            if matches!(route, Route::TiledGemv4) {
                assert_eq!(
                    bit_differences_vs_row, 0,
                    "{} batched GEMV must match row GEMV bit-for-bit",
                    shape.label
                );
            }
        }

        for (route, output) in &outputs {
            for _ in 0..WARMUP {
                run_once(
                    *route,
                    REPEATS_PER_CB,
                    &mut registry,
                    &device,
                    &weight,
                    &input,
                    output,
                    &params,
                );
            }
        }

        let mut wall: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(ROUNDS));
        let mut gpu: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(ROUNDS));
        let orders = [[0usize, 1, 2, 3], [3usize, 2, 1, 0]];
        for round in 0..ROUNDS {
            for &index in &orders[round % orders.len()] {
                let (route, output) = &outputs[index];
                let (wall_us, gpu_us) = run_once(
                    *route,
                    REPEATS_PER_CB,
                    &mut registry,
                    &device,
                    &weight,
                    &input,
                    output,
                    &params,
                );
                wall[index].push(wall_us);
                gpu[index].push(gpu_us);
            }
        }

        let wall: [Distribution; 4] =
            std::array::from_fn(|index| distribution(std::mem::take(&mut wall[index])));
        let gpu: [Distribution; 4] =
            std::array::from_fn(|index| distribution(std::mem::take(&mut gpu[index])));
        if winner_only {
            let winner = (0..Route::ALL.len())
                .min_by(|&left, &right| wall[left].median.total_cmp(&wall[right].median))
                .expect("at least one route");
            println!(
                "shape={} winner={} tensor32_wall_us={:.3} tensor128_wall_us={:.3} row_wall_us={:.3} tiled4_wall_us={:.3} winner_gpu_us={:.3}",
                shape.label,
                Route::ALL[winner].label(),
                wall[0].median,
                wall[1].median,
                wall[2].median,
                wall[3].median,
                gpu[winner].median,
            );
        } else {
            for (index, route) in Route::ALL.into_iter().enumerate() {
                println!(
                    "shape={} route={} wall_p25_us={:.3} median_wall_us={:.3} wall_p75_us={:.3} gpu_p25_us={:.3} median_gpu_us={:.3} gpu_p75_us={:.3}",
                    shape.label,
                    route.label(),
                    wall[index].p25,
                    wall[index].median,
                    wall[index].p75,
                    gpu[index].p25,
                    gpu[index].median,
                    gpu[index].p75,
                );
            }
        }
    }
}
