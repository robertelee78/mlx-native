//! Native BF16 expert-ID route microbenchmark.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::{
    calibrate_dense_matmul_id_routes, dense_matmul_id, DType, DenseMatmulIdCalibrationCase,
    DenseMatmulIdCalibrationLimits, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity,
    DenseMatmulIdParams, DenseMatmulIdRoute, DenseMatmulIdScratch, KernelRegistry, MlxDevice,
};
use std::time::Instant;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

#[test]
#[ignore = "production-width hardware benchmark"]
fn production_width_bf16_grouped_prefill_vs_direct() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let m = std::env::var("MLX_DENSE_MATMUL_ID_M")
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .expect("MLX_DENSE_MATMUL_ID_M must be U32")
        })
        .unwrap_or(33);
    assert!(
        matches!(m, 9 | 33),
        "benchmark covers the M=9/M=33 switch widths"
    );
    let n = 2048u32;
    let k = 4096u32;
    let top_k = 6u32;
    // Production N/K/top-k, reduced resident expert count so the benchmark is
    // safe to run alongside ordinary development without a multi-GiB fixture.
    let n_experts = 8u32;
    let matrix_bytes = u64::from(n) * u64::from(k) * 2;
    let expert_stride_bytes = matrix_bytes + 64;
    let total_weight_bytes =
        ((u64::from(n_experts) - 1) * expert_stride_bytes + matrix_bytes) as usize;
    let mut weights = device
        .alloc_buffer(
            total_weight_bytes,
            DType::BF16,
            vec![total_weight_bytes / 2],
        )
        .expect("weights");
    let stride_words = expert_stride_bytes as usize / 2;
    let matrix_words = matrix_bytes as usize / 2;
    let weight_words = weights.as_mut_slice::<u16>().expect("weights slice");
    for expert in 0..n_experts as usize {
        let pattern: Vec<u16> = (0..8192usize)
            .map(|index| {
                let raw = (((index * 29 + expert * 71) % 1021) as f32 - 510.0) / 4096.0;
                bf16::from_f32(raw).to_bits()
            })
            .collect();
        let matrix = &mut weight_words[expert * stride_words..expert * stride_words + matrix_words];
        for chunk in matrix.chunks_mut(pattern.len()) {
            chunk.copy_from_slice(&pattern[..chunk.len()]);
        }
    }
    for expert in 0..n_experts as usize - 1 {
        let padding_start = expert * expert_stride_bytes as usize + matrix_bytes as usize;
        weights.as_mut_slice::<u8>().unwrap()[padding_start..padding_start + 64].fill(0xD7);
    }

    let input_elements = m as usize * k as usize;
    let mut input = device
        .alloc_buffer(input_elements * 4, DType::F32, vec![m as usize, k as usize])
        .expect("input");
    let adversarial = [
        1.000_976_6f32,
        -1.000_976_6,
        f32::from_bits(0x3f80_0001),
        f32::from_bits(0xbf80_0001),
        0.333_333_34,
        -0.142_857_15,
        0.000_012_345_679,
        -0.000_009_765_625,
    ];
    for (index, value) in input.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        let base = adversarial[(index * 5 + index / k as usize) % adversarial.len()];
        *value = base * (1.0 + ((index * 17) % 11) as f32 / 4096.0);
    }
    let mut ids = device
        .alloc_buffer(
            m as usize * top_k as usize * 4,
            DType::U32,
            vec![m as usize, top_k as usize],
        )
        .expect("ids");
    let expert_order = [1u32, 3, 7, 0, 2, 5, 4, 6];
    for token in 0..m as usize {
        for slot in 0..top_k as usize {
            ids.as_mut_slice::<u32>().unwrap()[token * top_k as usize + slot] =
                expert_order[(token + slot) % expert_order.len()];
        }
    }
    let direct_output = device
        .alloc_buffer(
            m as usize * top_k as usize * n as usize * 4,
            DType::F32,
            vec![m as usize, top_k as usize, n as usize],
        )
        .unwrap();
    let grouped_output = device
        .alloc_buffer(
            m as usize * top_k as usize * n as usize * 4,
            DType::F32,
            vec![m as usize, top_k as usize, n as usize],
        )
        .unwrap();
    let scratch = DenseMatmulIdScratch::new(&device, n_experts, m).unwrap();

    let base = DenseMatmulIdParams {
        m,
        n,
        k,
        top_k,
        n_experts,
        expert_stride_bytes,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };
    let direct = DenseMatmulIdParams {
        route: DenseMatmulIdRoute::Direct,
        ..base
    };

    let mut run_once = |params: &DenseMatmulIdParams, output| {
        let start = Instant::now();
        let mut encoder = device.command_encoder().unwrap();
        let receipt = dense_matmul_id(
            &mut encoder,
            &mut registry,
            &device,
            &weights,
            &input,
            &ids,
            output,
            Some(&scratch),
            params,
        )
        .unwrap();
        let (gpu_start, gpu_end) = encoder.commit_wait_with_gpu_time().unwrap();
        (
            receipt.route,
            receipt.dispatch_count,
            start.elapsed().as_secs_f64() * 1e3,
            (gpu_end - gpu_start).max(0.0) * 1e3,
        )
    };
    for _ in 0..2 {
        let _ = run_once(&direct, &direct_output);
        let _ = run_once(&base, &grouped_output);
    }
    let mut direct_wall = Vec::with_capacity(9);
    let mut direct_gpu = Vec::with_capacity(9);
    let mut grouped_wall = Vec::with_capacity(9);
    let mut grouped_gpu = Vec::with_capacity(9);
    for sample in 0..9 {
        let (direct_sample, grouped_sample) = if sample % 2 == 0 {
            (
                run_once(&direct, &direct_output),
                run_once(&base, &grouped_output),
            )
        } else {
            let grouped_sample = run_once(&base, &grouped_output);
            let direct_sample = run_once(&direct, &direct_output);
            (direct_sample, grouped_sample)
        };
        assert_eq!(direct_sample.0, DenseMatmulIdRoute::Direct);
        assert_eq!(direct_sample.1, 1);
        assert_eq!(grouped_sample.0, DenseMatmulIdRoute::GroupedPrefill);
        assert_eq!(grouped_sample.1, 2);
        direct_wall.push(direct_sample.2);
        direct_gpu.push(direct_sample.3);
        grouped_wall.push(grouped_sample.2);
        grouped_gpu.push(grouped_sample.3);
    }
    let summarize = |wall_ms: &mut Vec<f64>, gpu_ms: &mut Vec<f64>| {
        let wall_min = wall_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let wall_max = wall_ms.iter().copied().fold(0.0f64, f64::max);
        let gpu_min = gpu_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let gpu_max = gpu_ms.iter().copied().fold(0.0f64, f64::max);
        let wall_median = median(wall_ms);
        let gpu_median = median(gpu_ms);
        (
            wall_median,
            wall_min,
            wall_max,
            gpu_median,
            gpu_min,
            gpu_max,
        )
    };

    let direct_timing = summarize(&mut direct_wall, &mut direct_gpu);
    let grouped_timing = summarize(&mut grouped_wall, &mut grouped_gpu);

    let direct_values = direct_output.as_slice::<f32>().unwrap();
    let grouped_values = grouped_output.as_slice::<f32>().unwrap();
    let max_delta = direct_values
        .iter()
        .zip(grouped_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let bitwise_equal = direct_values
        .iter()
        .zip(grouped_values)
        .all(|(direct, grouped)| direct.to_bits() == grouped.to_bits());
    assert!(bitwise_equal, "grouped/direct max delta {max_delta}");

    println!(
        "dense_matmul_id BF16 production-width M={m} N={n} K={k} top_k={top_k} experts={n_experts}: \
         direct wall median/min/max={:.3}/{:.3}/{:.3}ms gpu={:.3}/{:.3}/{:.3}ms; \
         grouped wall median/min/max={:.3}/{:.3}/{:.3}ms gpu={:.3}/{:.3}/{:.3}ms; \
         speedup={:.2}x max_delta={max_delta:.3e} bitwise_equal={bitwise_equal}",
        direct_timing.0,
        direct_timing.1,
        direct_timing.2,
        direct_timing.3,
        direct_timing.4,
        direct_timing.5,
        grouped_timing.0,
        grouped_timing.1,
        grouped_timing.2,
        grouped_timing.3,
        grouped_timing.4,
        grouped_timing.5,
        direct_timing.0 / grouped_timing.0,
    );

    // The production activation API runs the same exact-shape gate once,
    // freezes the decision, and accounts for its load-time ceiling.
    let mut auto_registry = KernelRegistry::new();
    let calibration_started = Instant::now();
    let (plan, calibration) = calibrate_dense_matmul_id_routes(
        &mut auto_registry,
        &device,
        1,
        DenseMatmulIdCalibrationLimits {
            max_elapsed_ms: 1_000,
            max_cases: 1,
            // Balanced and maximally-skewed profiles each run Direct proof +
            // Grouped proof + 5 AB pairs, followed by one cleanup boundary.
            max_submissions: 25,
        },
        &[DenseMatmulIdCalibrationCase {
            weight: &weights,
            params: base,
        }],
    )
    .expect("production-width activation calibration");
    assert_eq!(calibration.declared_cases, 1);
    assert!(calibration.calibration_submissions <= 25);
    assert_eq!(plan.decision_count(), 1);
    println!(
        "dense_matmul_id activation calibration M={m}: selected={:?} status={:?} \
         wall_elapsed={:.3}ms receipt_elapsed={:.3}ms submissions={} dispatches={} \
         cache_hits={} deadline_overrun={:.3}ms plan={}",
        calibration.decisions[0].selected_route,
        calibration.decisions[0].status,
        calibration_started.elapsed().as_secs_f64() * 1e3,
        calibration.elapsed_ms,
        calibration.calibration_submissions,
        calibration.calibration_dispatches,
        calibration.process_cache_hits,
        calibration.deadline_overrun_ms,
        plan.plan_id(),
    );
}
