//! ADR-065 E6 probe 5: logical weight loads and observed bandwidth for 16
//! projection rows as four groups of 4 versus two groups of 8.
//!
//! For each Qwen3.8-27B projection shape, K-quant codec and route
//! (`mv_ext` and the default `served` route, as in probe 1) the probe
//! encodes the 16 activation columns either as four 4-column dispatches or
//! as two 8-column dispatches in one command buffer and records:
//!
//! * `weight_passes` — logical passes over the weight matrix, the sum over
//!   resolved kernel dispatches of the column-group threadgroup count
//!   (`grid[1]`); each column group streams the whole weight once;
//! * `logical_weight_bytes = weight_passes * weight_bytes`;
//! * median GPU time (`GPUEndTime - GPUStartTime`) over `E6_BW_REPS`
//!   (default 30) command buffers after 5 warm-ups, and the derived
//!   logical and unique weight bandwidth in GB/s.
//!
//! Asserted: both groupings produce bitwise-identical outputs and the
//! recorded pass counts are positive. Timing is a measurement; the test is
//! `#[ignore]`d so it runs only on a quiet host (`--ignored`).

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");

use mlx_native::{
    quantized_matmul_ggml_with_policy_and_trace, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlType, GgmlWorkloadClass,
};

const SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate_up", 17_408, 5_120),
    ("ffn_down", 5_120, 17_408),
    ("deltanet_qkv", 10_240, 5_120),
    ("attn_output", 5_120, 6_144),
];
const ROWS: usize = 16;

fn policy_for(route: &str) -> GgmlRoutingPolicy {
    match route {
        "mv_ext" => GgmlRoutingPolicy {
            dense_q5k_canonical_q4x4: false,
            dense_decode_mvn: false,
            dense_decode_mv_ext: true,
            ..GgmlRoutingPolicy::default()
        },
        "served" => GgmlRoutingPolicy::default(),
        other => panic!("unknown route {other}"),
    }
}

fn fill_kquant(weight: &mut MlxBuffer, ggml_type: GgmlType, rng: &mut E6Rng) {
    let block_bytes = ggml_type.block_bytes() as usize;
    for (block, bytes) in weight
        .as_mut_slice::<u8>()
        .expect("weight bytes")
        .chunks_exact_mut(block_bytes)
        .enumerate()
    {
        for byte in bytes.iter_mut() {
            *byte = rng.next_u8();
        }
        let d = half::f16::from_f32(0.000_5 + (block % 29) as f32 * 0.000_031_25);
        let dmin = half::f16::from_f32(0.000_25 + (block % 13) as f32 * 0.000_015_625);
        match ggml_type {
            GgmlType::Q4_K | GgmlType::Q5_K => {
                bytes[..2].copy_from_slice(&d.to_bits().to_le_bytes());
                bytes[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            GgmlType::Q6_K => bytes[208..210].copy_from_slice(&d.to_bits().to_le_bytes()),
            other => panic!("not a probed K-quant: {other:?}"),
        }
    }
}

/// Encode the 16 columns as `16 / group` dispatches of width `group`.
/// Returns the logical weight passes and the resolved route label.
#[allow(clippy::too_many_arguments)]
fn encode_groups(
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    route: &str,
    group: usize,
) -> (u64, String) {
    let mut passes = 0u64;
    let mut resolved = String::new();
    for index in 0..ROWS / group {
        let column = index * group;
        let input_view = input.slice_view((column * k * 4) as u64, group * k);
        let output_view = output.slice_view((column * n * 4) as u64, group * n);
        let trace = quantized_matmul_ggml_with_policy_and_trace(
            encoder,
            registry,
            device,
            &input_view,
            weight,
            &output_view,
            &GgmlQuantizedMatmulParams {
                m: group as u32,
                n: n as u32,
                k: k as u32,
                ggml_type,
            },
            &policy_for(route),
            GgmlWorkloadClass::ContinuousWidth,
        )
        .expect("dense K-quant projection");
        resolved = format!("{:?}", trace.resolved_route);
        passes += trace
            .dispatches
            .iter()
            .map(|dispatch| dispatch.encoded.grid[1])
            .sum::<u64>();
    }
    (passes, resolved)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    values[values.len() / 2]
}

fn probe(ggml_type: GgmlType, route: &str) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let reps: usize = std::env::var("E6_BW_REPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30);
    for &(shape, n, k) in SHAPES {
        let mut rng = E6Rng::new(0xe6_0bb0 ^ ((n as u64) << 20) ^ k as u64);
        let weight_bytes =
            n * (k / ggml_type.block_values() as usize) * ggml_type.block_bytes() as usize;
        let mut weight = device
            .alloc_buffer(weight_bytes, DType::U8, vec![weight_bytes])
            .expect("weight");
        fill_kquant(&mut weight, ggml_type, &mut rng);
        let activations: Vec<f32> = (0..ROWS * k).map(|_| rng.signed(1.0)).collect();
        let input = e6_f32_buffer(&device, &activations);
        let mut outputs = Vec::new();
        for group in [4usize, 8] {
            let output = device
                .alloc_buffer(ROWS * n * 4, DType::F32, vec![ROWS, n])
                .expect("output");
            let mut times = Vec::with_capacity(reps);
            let mut passes = 0;
            let mut resolved = String::new();
            for rep in 0..reps + 5 {
                let mut encoder = device.command_encoder().expect("encoder");
                (passes, resolved) = encode_groups(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &weight,
                    &input,
                    &output,
                    n,
                    k,
                    ggml_type,
                    route,
                    group,
                );
                let (start, end) = encoder.commit_wait_with_gpu_time().expect("execution");
                if rep >= 5 {
                    times.push(end - start);
                }
            }
            assert!(
                passes > 0,
                "{route} {ggml_type:?} {shape}: no weight passes recorded"
            );
            let seconds = median(times);
            let logical = passes as f64 * weight_bytes as f64;
            e6_record(
                "weight_load_bandwidth_16_rows",
                &format!(
                    "{route}/{ggml_type:?}/{shape}/groups{}x{group}",
                    ROWS / group
                ),
                &[
                    ("resolved_route", resolved),
                    ("n", n.to_string()),
                    ("k", k.to_string()),
                    ("weight_bytes", weight_bytes.to_string()),
                    ("weight_passes", passes.to_string()),
                    ("logical_weight_bytes", (logical as u64).to_string()),
                    ("median_gpu_us", format!("{:.2}", seconds * 1e6)),
                    ("logical_gbps", format!("{:.1}", logical / seconds / 1e9)),
                    (
                        "unique_gbps",
                        format!("{:.1}", weight_bytes as f64 / seconds / 1e9),
                    ),
                    ("reps", reps.to_string()),
                ],
            );
            outputs.push(e6_read_f32(&output));
        }
        let compare = E6Compare::new(&outputs[0], &outputs[1]);
        assert!(
            compare.bitwise(),
            "{route} {ggml_type:?} {shape}: 2x8 output differs from 4x4 ({} mismatches)",
            compare.mismatched
        );
    }
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_mv_ext_q4_k() {
    probe(GgmlType::Q4_K, "mv_ext");
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_mv_ext_q5_k() {
    probe(GgmlType::Q5_K, "mv_ext");
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_mv_ext_q6_k() {
    probe(GgmlType::Q6_K, "mv_ext");
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_served_q4_k() {
    probe(GgmlType::Q4_K, "served");
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_served_q5_k() {
    probe(GgmlType::Q5_K, "served");
}

#[test]
#[ignore = "GPU timing; run with --ignored on a quiet host"]
fn e6_bandwidth_served_q6_k() {
    probe(GgmlType::Q6_K, "served");
}
