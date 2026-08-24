//! Activation-cost receipt for strict per-buffer Q4_0 route proof.

use half::f16;
use mlx_native::{
    calibrate_dense_q4_routes, validate_dense_q4_cartesian_acceptance, DType, DenseQ4BaseShape,
    DenseQ4CalibrationCase, DenseQ4CalibrationLimits, DenseQ4CartesianAcceptanceRequirements,
    DenseQ4InputLayout, KernelRegistry, MlxBuffer, MlxDevice,
};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

const REACHABLE_M: &[u32] = &[9, 16, 24, 32, 48, 64, 96, 128, 129, 2048, 4096];
const STRUCTURAL_PROJECTIONS: &[(&str, u32, u32)] = &[
    ("square_768", 768, 768),
    ("expand_768_3072", 3072, 768),
    ("contract_3072_768", 768, 3072),
];
const ENCODER_PROJECTIONS: &[(&str, u32, u32)] = &[
    ("square_1536", 1536, 1536),
    ("expand_1536_8960", 8960, 1536),
    ("contract_8960_1536", 1536, 8960),
];
const QWEN_DENSE_PROJECTIONS: &[(&str, u32, u32)] = &[
    ("square_5120", 5120, 5120),
    ("expand_5120_17408", 17408, 5120),
    ("contract_17408_5120", 5120, 17408),
];
const DEFAULT_BUDGET_MS: u64 = 15_000;

struct Profile {
    name: &'static str,
    layers: usize,
    projections: &'static [(&'static str, u32, u32)],
}

fn profile() -> Profile {
    match std::env::var("MLX_Q4_CALIBRATION_PROFILE").as_deref() {
        Ok("encoder_1536_8960") => Profile {
            name: "encoder_1536_8960",
            layers: 28,
            projections: ENCODER_PROJECTIONS,
        },
        Ok("qwen_dense_5120_17408") => Profile {
            name: "qwen_dense_5120_17408",
            layers: 64,
            projections: QWEN_DENSE_PROJECTIONS,
        },
        Ok("structural_768_3072") | Err(_) => Profile {
            name: "structural_768_3072",
            layers: 65,
            projections: STRUCTURAL_PROJECTIONS,
        },
        Ok(value) => panic!(
            "unsupported MLX_Q4_CALIBRATION_PROFILE={value}; expected structural_768_3072, \
             encoder_1536_8960, or qwen_dense_5120_17408"
        ),
    }
}

fn env_number<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn native_q4_weight(device: &MlxDevice, n: u32, k: u32, ordinal: usize) -> MlxBuffer {
    let bytes = n as usize * (k as usize / 32) * 18;
    let mut weight = device
        .alloc_buffer(bytes, DType::U8, vec![bytes])
        .expect("allocate native Q4_0 weight");
    let values = weight.as_mut_slice::<u8>().expect("map native Q4_0 weight");
    let salt = ordinal.wrapping_mul(17).wrapping_add(3);
    for (block_index, block) in values.chunks_exact_mut(18).enumerate() {
        let scale = f16::from_f32(0.0078125 + (salt % 127) as f32 / 65_536.0);
        block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
        for (byte_index, packed) in block[2..].iter_mut().enumerate() {
            let low = (block_index + byte_index + salt) % 15 + 1;
            let high = (block_index * 3 + byte_index * 5 + salt * 7) % 15 + 1;
            *packed = low as u8 | ((high as u8) << 4);
        }
    }
    weight
}

fn main() {
    let profile = profile();
    let layers = env_number("MLX_Q4_CALIBRATION_LAYERS", profile.layers);
    let budget_ms = env_number("MLX_Q4_CALIBRATION_BUDGET_MS", DEFAULT_BUDGET_MS);
    let reachable_m: Vec<u32> = std::env::var("MLX_Q4_CALIBRATION_M")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.parse::<u32>().expect("parse MLX_Q4_CALIBRATION_M"))
                .collect()
        })
        .unwrap_or_else(|| REACHABLE_M.to_vec());
    let projection_filter = std::env::var("MLX_Q4_CALIBRATION_PROJECTION").ok();
    let projections: Vec<_> = profile
        .projections
        .iter()
        .copied()
        .filter(|(label, _, _)| {
            projection_filter
                .as_deref()
                .is_none_or(|wanted| wanted == *label)
        })
        .collect();
    assert!(layers > 0, "layer multiplicity must be nonzero");
    assert!(!reachable_m.is_empty(), "reachable M set must be nonempty");
    assert!(
        !projections.is_empty(),
        "projection filter matched no projection"
    );

    let device = MlxDevice::new().expect("Metal device");
    let allocation_started = Instant::now();
    let groups: Vec<_> = profile
        .projections
        .iter()
        .filter(|(label, _, _)| {
            projection_filter
                .as_deref()
                .is_none_or(|wanted| wanted == *label)
        })
        .enumerate()
        .map(|(projection_index, &(label, n, k))| {
            let weights: Vec<_> = (0..layers)
                .map(|layer| {
                    native_q4_weight(
                        &device,
                        n,
                        k,
                        projection_index.wrapping_mul(layers).wrapping_add(layer),
                    )
                })
                .collect();
            (label, n, k, weights)
        })
        .collect();
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1000.0;
    let cases: Vec<_> = groups
        .iter()
        .flat_map(|(_, n, k, weights)| {
            weights.iter().map(|weight| DenseQ4CalibrationCase {
                weight,
                shape: DenseQ4BaseShape {
                    n: *n,
                    k: *k,
                    batch: 1,
                    input_layout: DenseQ4InputLayout::Contiguous,
                },
                reachable_m: &reachable_m,
            })
        })
        .collect();
    let limits = DenseQ4CalibrationLimits {
        max_elapsed_ms: budget_ms,
        max_shapes: (projections.len() * reachable_m.len()) as u32,
    };

    let mut cold_registry = KernelRegistry::new();
    let cold_started = Instant::now();
    let (_cold_plan, cold) =
        calibrate_dense_q4_routes(&mut cold_registry, &device, 1, limits, &cases)
            .expect("cold Q4 activation calibration");
    let cold_wall_ms = cold_started.elapsed().as_secs_f64() * 1000.0;

    let mut warm_registry = KernelRegistry::new();
    let warm_started = Instant::now();
    let (_warm_plan, warm) =
        calibrate_dense_q4_routes(&mut warm_registry, &device, 2, limits, &cases)
            .expect("timing-cache Q4 reactivation calibration");
    let warm_wall_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        cold_wall_ms.is_finite() && cold_wall_ms > 0.0 && cold_wall_ms <= DEFAULT_BUDGET_MS as f64,
        "cold Q4 activation wall time {cold_wall_ms:.3} ms exceeded the publication ceiling"
    );
    assert!(
        warm_wall_ms.is_finite() && warm_wall_ms > 0.0 && warm_wall_ms <= DEFAULT_BUDGET_MS as f64,
        "reactivation Q4 wall time {warm_wall_ms:.3} ms exceeded the publication ceiling"
    );
    validate_dense_q4_cartesian_acceptance(
        &cold,
        &warm,
        &DenseQ4CartesianAcceptanceRequirements {
            // Acceptance stays pinned to the complete named profile. The
            // environment overrides remain useful for diagnostics, but
            // reduced layer/projection/row runs cannot publish, and a budget
            // override cannot expand the profile's qualification ceiling.
            expected_base_shapes: profile.projections.len() as u32,
            expected_weight_buffers_per_base: profile.layers as u32,
            reachable_m: REACHABLE_M.to_vec(),
            required_compatibility_m: vec![2048, 4096],
            minimum_candidate_decisions: 1,
            maximum_elapsed_ms: DEFAULT_BUDGET_MS,
        },
    )
    .expect("Q4 activation receipts satisfy the publishable Cartesian contract");

    let receipt = json!({
        "schema_version": 1,
        "profile": profile.name,
        "hardware": device.name(),
        "dtype": "q4_0_native_bytes_x_f32_to_f32",
        "reachable_m": reachable_m,
        "projections": projections.iter().map(|(label, n, k)| {
            json!({ "label": label, "n": n, "k": k, "batch": 1 })
        }).collect::<Vec<_>>(),
        "layers_per_projection": layers,
        "total_distinct_weight_buffers": layers * projections.len(),
        "total_native_weight_bytes": groups.iter().flat_map(|(_, _, _, weights)| weights)
            .map(MlxBuffer::data_byte_len).sum::<usize>(),
        "budget_ms": budget_ms,
        "allocation_ms": allocation_ms,
        "cold_wall_ms": cold_wall_ms,
        "reactivation_wall_ms": warm_wall_ms,
        "cold": cold,
        "reactivation": warm,
    });
    let encoded = serde_json::to_string_pretty(&receipt).expect("serialize activation receipt");
    println!("{encoded}");

    let output_path = std::env::var_os("MLX_Q4_CALIBRATION_RECEIPT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "target/bench-receipts/dense_q4_activation_calibration-{}.json",
                profile.name
            ))
        });
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("create activation receipt directory");
    }
    std::fs::write(&output_path, encoded).expect("persist activation receipt");
    eprintln!("persisted_receipt={}", output_path.display());
}
