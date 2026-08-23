//! ABBA benchmark for the Q4_0 direct-F32 64x32 tensor candidate.

use half::f16;
use mlx_native::{
    dispatch_mm_for_test, dispatch_mm_q4_0_tensor_64x32_for_test, DType, GgmlQuantizedMatmulParams,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};
use std::collections::BTreeSet;
use std::time::Instant;

const WARMUPS_PER_ROUTE: usize = 5;
const ABBA_ROUNDS: usize = 21;
const SWEEP_M: &[u32] = &[9, 16, 24, 32, 48, 64, 96, 128, 129];
const QUALIFICATION_M: &[u32] = &[
    9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
    4095, 4096, 4097,
];
const EDGE_M: &[u32] = &[9, 129, 2048, 4096];
const PROJECTIONS: &[(&str, u32, u32)] = &[
    ("square_768", 768, 768),
    ("expand_768_3072", 3072, 768),
    ("contract_3072_768", 768, 3072),
];
const EDGE_N: &[u32] = &[
    1, 63, 64, 65, 127, 128, 129, 767, 768, 769, 3071, 3072, 3073,
];
const EDGE_K: &[u32] = &[32, 64, 96, 128, 160, 256, 768, 800, 3072, 17408];
const PRODUCTION_WIDTHS: &[(&str, u32, u32)] = &[
    ("square_1536", 1536, 1536),
    ("expand_1536_8960", 8960, 1536),
    ("contract_8960_1536", 1536, 8960),
    ("square_5120", 5120, 5120),
    ("expand_5120_17408", 17408, 5120),
    ("contract_17408_5120", 5120, 17408),
];

#[derive(Clone, Copy, Debug)]
enum Route {
    CurrentV2,
    Tensor64x32,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::CurrentV2 => "current_v2_64x128",
            Self::Tensor64x32 => "candidate_64x32",
        }
    }
}

struct Shape {
    label: String,
    m: u32,
    n: u32,
    k: u32,
}

fn shapes() -> Vec<Shape> {
    if std::env::var("MLX_BENCH_QUALIFICATION").as_deref() == Ok("1") {
        return qualification_shapes();
    }
    let m_values: &[u32] = if std::env::var("MLX_BENCH_M_SWEEP").as_deref() == Ok("1") {
        SWEEP_M
    } else {
        &[32]
    };
    m_values
        .iter()
        .flat_map(|&m| {
            PROJECTIONS.iter().map(move |&(label, n, k)| Shape {
                label: format!("{label}_m{m}"),
                m,
                n,
                k,
            })
        })
        .collect()
}

fn selected_shapes(shape_filter: Option<&str>) -> Result<Vec<Shape>, String> {
    let selected: Vec<_> = shapes()
        .into_iter()
        .filter(|shape| shape_filter.is_none_or(|filter| shape.label.contains(filter)))
        .collect();
    if selected.is_empty() {
        return Err(format!(
            "MLX_BENCH_SHAPE_FILTER={} matched no shapes",
            shape_filter.unwrap_or("<none>")
        ));
    }
    Ok(selected)
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn selected_shape_count(shape_filter: Option<&str>) -> Result<usize, String> {
    selected_shapes(shape_filter).map(|shapes| shapes.len())
}

fn qualification_shapes() -> Vec<Shape> {
    let section = std::env::var("MLX_BENCH_QUALIFICATION_SECTION").unwrap_or_else(|_| "all".into());
    let include = |name: &str| section == "all" || section == name;
    let mut seen = BTreeSet::new();
    let mut shapes = Vec::new();
    let mut push = |label: String, m: u32, n: u32, k: u32| {
        if seen.insert((m, n, k)) {
            shapes.push(Shape { label, m, n, k });
        }
    };
    if include("spine") {
        for &m in QUALIFICATION_M {
            for &(label, n, k) in PROJECTIONS {
                push(format!("qualification_spine_{label}_m{m}"), m, n, k);
            }
        }
    }
    if include("edge") {
        for &m in EDGE_M {
            for &n in EDGE_N {
                push(format!("qualification_edge_n{n}_k768_m{m}"), m, n, 768);
            }
            for &k in EDGE_K {
                push(format!("qualification_edge_n768_k{k}_m{m}"), m, 768, k);
            }
        }
    }
    if include("production") {
        for &m in &[32, 129, 2048, 4096] {
            for &(label, n, k) in PRODUCTION_WIDTHS {
                push(format!("qualification_production_{label}_m{m}"), m, n, k);
            }
        }
    }
    shapes
}

fn sample(index: usize, salt: usize) -> f32 {
    let coarse = (index.wrapping_mul(29).wrapping_add(salt * 17) % 257) as f32 - 128.0;
    let fine = (index.wrapping_mul(43).wrapping_add(salt * 11) % 31) as f32 - 15.0;
    coarse / 251.0 + fine / 16_381.0
}

fn q4_weight_bytes(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let mut bytes = Vec::with_capacity(n * (k / 32) * 18);
    let mut block = [0.0f32; 32];
    for row in 0..n {
        for block_index in 0..k / 32 {
            for (index, value) in block.iter_mut().enumerate() {
                *value = sample(row * k + block_index * 32 + index, 7);
            }
            let amax = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
            let scale = amax / 7.0;
            let inverse_scale = if scale == 0.0 { 0.0 } else { scale.recip() };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for index in 0..16 {
                let low = (block[index] * inverse_scale + 8.0)
                    .round()
                    .clamp(0.0, 15.0) as u8;
                let high = (block[index + 16] * inverse_scale + 8.0)
                    .round()
                    .clamp(0.0, 15.0) as u8;
                bytes.push(low | (high << 4));
            }
        }
    }
    bytes
}

fn alloc_weight(device: &MlxDevice, shape: &Shape) -> MlxBuffer {
    let bytes = q4_weight_bytes(shape.n as usize, shape.k as usize);
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("allocate Q4_0 weights");
    buffer
        .as_mut_slice::<u8>()
        .expect("map Q4_0 weights")
        .copy_from_slice(&bytes);
    buffer
}

fn alloc_input(device: &MlxDevice, shape: &Shape) -> MlxBuffer {
    let elements = (shape.m * shape.k) as usize;
    let mut buffer = device
        .alloc_buffer(
            elements * size_of::<f32>(),
            DType::F32,
            vec![shape.m as usize, shape.k as usize],
        )
        .expect("allocate F32 input");
    for (index, value) in buffer
        .as_mut_slice::<f32>()
        .expect("map F32 input")
        .iter_mut()
        .enumerate()
    {
        *value = sample(index, 23);
    }
    buffer
}

fn alloc_output(device: &MlxDevice, shape: &Shape) -> MlxBuffer {
    let elements = (shape.m * shape.n) as usize;
    device
        .alloc_buffer(
            elements * size_of::<f32>(),
            DType::F32,
            vec![shape.m as usize, shape.n as usize],
        )
        .expect("allocate F32 output")
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    route: Route,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
) -> (f64, f64) {
    let mut encoder = device.command_encoder().expect("create command encoder");
    let started = Instant::now();
    match route {
        Route::CurrentV2 => {
            dispatch_mm_for_test(
                &mut encoder,
                registry,
                device,
                input,
                weight,
                output,
                params,
            )
            .expect("encode current V2");
        }
        Route::Tensor64x32 => {
            dispatch_mm_q4_0_tensor_64x32_for_test(
                &mut encoder,
                registry,
                device,
                input,
                weight,
                output,
                params,
            )
            .expect("encode 64x32 candidate");
        }
    }
    let (gpu_start, gpu_end) = encoder
        .commit_wait_with_gpu_time()
        .expect("complete Q4_0 MM");
    (
        started.elapsed().as_secs_f64() * 1e6,
        (gpu_end - gpu_start) * 1e6,
    )
}

#[derive(Clone, Copy)]
struct Distribution {
    p25: f64,
    median: f64,
    p75: f64,
    mean: f64,
    standard_deviation: f64,
}

fn distribution(mut values: Vec<f64>) -> Distribution {
    assert_eq!(values.len(), ABBA_ROUNDS * 2);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    values.sort_by(f64::total_cmp);
    Distribution {
        p25: values[(values.len() - 1) / 4],
        median: values[values.len() / 2],
        p75: values[(values.len() - 1) * 3 / 4],
        mean,
        standard_deviation: variance.sqrt(),
    }
}

fn assert_bitwise_parity(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    let differences = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    assert_eq!(differences, 0, "current V2 and 64x32 outputs differ");
    assert!(left.iter().all(|value| value.is_finite()));
    assert!(left.iter().any(|value| value.to_bits() != 0));
}

fn main() {
    let shape_filter = std::env::var("MLX_BENCH_SHAPE_FILTER").ok();
    let selected_shapes =
        selected_shapes(shape_filter.as_deref()).unwrap_or_else(|error| panic!("{error}"));
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    println!(
        "device={} base={} qtype=Q4_0 layout=weight[N,K]_input[M,K]_output[M,N] activation=F32 native_weight_bytes=true warmups_per_route={} abba_rounds={} samples_per_route={} dispatches_per_sample=1 parity=bitwise sweep={} qualification={} qualification_section={}",
        device.metal_device().name(),
        option_env!("MLX_NATIVE_BENCH_COMMIT").unwrap_or("unbound"),
        WARMUPS_PER_ROUTE,
        ABBA_ROUNDS,
        ABBA_ROUNDS * 2,
        std::env::var("MLX_BENCH_M_SWEEP").as_deref() == Ok("1"),
        std::env::var("MLX_BENCH_QUALIFICATION").as_deref() == Ok("1"),
        std::env::var("MLX_BENCH_QUALIFICATION_SECTION").as_deref().unwrap_or("all"),
    );

    for shape in selected_shapes {
        let weight = alloc_weight(&device, &shape);
        let input = alloc_input(&device, &shape);
        let control_output = alloc_output(&device, &shape);
        let candidate_output = alloc_output(&device, &shape);
        let params = GgmlQuantizedMatmulParams {
            m: shape.m,
            n: shape.n,
            k: shape.k,
            ggml_type: GgmlType::Q4_0,
        };

        for _ in 0..WARMUPS_PER_ROUTE {
            run_once(
                Route::CurrentV2,
                &mut registry,
                &device,
                &input,
                &weight,
                &control_output,
                &params,
            );
            run_once(
                Route::Tensor64x32,
                &mut registry,
                &device,
                &input,
                &weight,
                &candidate_output,
                &params,
            );
        }
        assert_bitwise_parity(
            control_output.as_slice::<f32>().expect("read control"),
            candidate_output.as_slice::<f32>().expect("read candidate"),
        );

        let mut control_wall = Vec::with_capacity(ABBA_ROUNDS * 2);
        let mut control_gpu = Vec::with_capacity(ABBA_ROUNDS * 2);
        let mut candidate_wall = Vec::with_capacity(ABBA_ROUNDS * 2);
        let mut candidate_gpu = Vec::with_capacity(ABBA_ROUNDS * 2);
        for round in 0..ABBA_ROUNDS {
            let order = if round % 2 == 0 {
                [
                    Route::CurrentV2,
                    Route::Tensor64x32,
                    Route::Tensor64x32,
                    Route::CurrentV2,
                ]
            } else {
                [
                    Route::Tensor64x32,
                    Route::CurrentV2,
                    Route::CurrentV2,
                    Route::Tensor64x32,
                ]
            };
            for route in order {
                let output = match route {
                    Route::CurrentV2 => &control_output,
                    Route::Tensor64x32 => &candidate_output,
                };
                let (wall, gpu) = run_once(
                    route,
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    output,
                    &params,
                );
                match route {
                    Route::CurrentV2 => {
                        control_wall.push(wall);
                        control_gpu.push(gpu);
                    }
                    Route::Tensor64x32 => {
                        candidate_wall.push(wall);
                        candidate_gpu.push(gpu);
                    }
                }
            }
        }

        let control_wall = distribution(control_wall);
        let control_gpu = distribution(control_gpu);
        let candidate_wall = distribution(candidate_wall);
        let candidate_gpu = distribution(candidate_gpu);
        let control_threadgroups =
            u64::from(shape.m).div_ceil(128) * u64::from(shape.n).div_ceil(64);
        let candidate_threadgroups =
            u64::from(shape.m).div_ceil(32) * u64::from(shape.n).div_ceil(64);
        for (route, wall, gpu, threadgroups) in [
            (
                Route::CurrentV2,
                control_wall,
                control_gpu,
                control_threadgroups,
            ),
            (
                Route::Tensor64x32,
                candidate_wall,
                candidate_gpu,
                candidate_threadgroups,
            ),
        ] {
            println!(
                "shape={} M={} N={} K={} route={} threadgroups={} wall_p25_us={:.3} wall_median_us={:.3} wall_p75_us={:.3} wall_mean_us={:.3} wall_stddev_us={:.3} gpu_p25_us={:.3} gpu_median_us={:.3} gpu_p75_us={:.3} gpu_mean_us={:.3} gpu_stddev_us={:.3}",
                shape.label,
                shape.m,
                shape.n,
                shape.k,
                route.label(),
                threadgroups,
                wall.p25,
                wall.median,
                wall.p75,
                wall.mean,
                wall.standard_deviation,
                gpu.p25,
                gpu.median,
                gpu.p75,
                gpu.mean,
                gpu.standard_deviation,
            );
        }
        println!(
            "shape={} candidate_speedup_wall_pct={:.3} candidate_speedup_gpu_pct={:.3} bit_differences=0",
            shape.label,
            (control_wall.median / candidate_wall.median - 1.0) * 100.0,
            (control_gpu.median / candidate_gpu.median - 1.0) * 100.0,
        );
    }
}
