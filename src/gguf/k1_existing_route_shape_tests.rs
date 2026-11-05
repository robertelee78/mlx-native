//! K1 design spike only: synthetic packed bytes, existing routes, no default changes.
//! No model, quantizer, external runtime or full logit/quality qualification.
use super::{dequantize_q4_k, dequantize_q8_0};
use crate::ggml_capability::{plan_dense_auto_route, DenseAutoPlan};
use crate::{
    quantized_matmul_ggml_with_policy, DType, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};
use std::time::Instant;
const LIMIT: usize = 128 * 1024 * 1024;
const GUARD: usize = 16;
const SENTINEL: f32 = -1234567.0;
const WARM: usize = 25;
const SAMPLES: usize = 8;
const REPS: usize = 4;
// Prospective fixed thresholds, never derived from a favorable observed sample.
const ABS: f64 = 0.002;
const REL: f64 = 0.0002;

fn fill_blocks(bytes: &mut [u8], codec: GgmlType) {
    // Valid bounded block encodings, not quantization of source/model weights.
    let width = codec.block_bytes() as usize;
    for (i, block) in bytes.chunks_exact_mut(width).enumerate() {
        for (j, v) in block.iter_mut().enumerate() {
            *v = (i.wrapping_mul(17).wrapping_add(j * 29).wrapping_add(31) % 251) as u8;
        }
        block[..2].copy_from_slice(&half::f16::from_f32(0.015625).to_bits().to_le_bytes());
        if codec == GgmlType::Q4_K {
            block[2..4].copy_from_slice(&half::f16::from_f32(0.0078125).to_bits().to_le_bytes());
        }
    }
}
fn userdata(n: usize, k: usize, codec: GgmlType) -> usize {
    let w = n
        .checked_mul(k / codec.block_values() as usize)
        .unwrap()
        .checked_mul(codec.block_bytes() as usize)
        .unwrap();
    // Host packed Vec + GPU weights; max M8 host/GPU inputs; two guarded
    // GPU outputs AND owned host comparison copies; one decoded CPU row.
    2 * w + 2 * 8 * k * 4 + 4 * (8 * n + 2 * GUARD) * 4 + k * 4 + 65536
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= ABS + REL * a.abs().max(b.abs())
}
fn median(v: &mut [f64]) -> f64 {
    assert_eq!(v.len(), SAMPLES);
    v.sort_by(f64::total_cmp);
    (v[SAMPLES / 2 - 1] + v[SAMPLES / 2]) * 0.5
}
fn output(device: &MlxDevice, m: usize, n: usize) -> (MlxBuffer, MlxBuffer) {
    let mut parent = device
        .alloc_buffer((m * n + 2 * GUARD) * 4, DType::F32, vec![m * n + 2 * GUARD])
        .expect("guarded output");
    parent.as_mut_slice::<f32>().unwrap().fill(SENTINEL);
    let view = parent
        .data_view(GUARD * 4, m * n * 4, DType::F32, vec![m, n])
        .unwrap();
    (parent, view)
}
fn check_guards(parent: &MlxBuffer, count: usize) {
    let a = parent.as_slice::<f32>().unwrap();
    assert!(a[..GUARD]
        .iter()
        .chain(&a[GUARD + count..])
        .all(|v| v.to_bits() == SENTINEL.to_bits()));
}
fn encode(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    out: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    policy: &GgmlRoutingPolicy,
    reps: usize,
) -> (f64, f64) {
    let start = Instant::now();
    let mut enc = device.command_encoder().expect("encoder");
    for repetition in 0..reps {
        // The same output is reused. Explicitly order
        // calls in the concurrent encoder; charge this barrier to BOTH arms.
        if repetition > 0 {
            enc.memory_barrier();
        }
        quantized_matmul_ggml_with_policy(
            &mut enc, registry, device, input, weight, out, params, policy,
        )
        .expect("existing explicit route");
    }
    // This API waits for completion; host wall includes encode/submit/wait.
    // GPU timestamps cover the whole repeated CB, not isolated instruction cost.
    let (a, b) = enc
        .commit_wait_with_gpu_time()
        .expect("checked completed GPU timing");
    assert!(a.is_finite() && b.is_finite() && b > a);
    (
        (b - a) * 1e6 / reps as f64,
        start.elapsed().as_secs_f64() * 1e6 / reps as f64,
    )
}
fn case(codec: GgmlType, n: usize, k: usize) {
    let declared = userdata(n, k, codec);
    assert!(declared <= LIMIT, "userdata {declared} exceeds {LIMIT}");
    // Device/driver/compiled-pipeline fixed overhead is Unknown, outside userdata.
    let device = MlxDevice::new().expect("required Metal device; unavailable is failure");
    let mut registry = KernelRegistry::new();
    let row_bytes = k / codec.block_values() as usize * codec.block_bytes() as usize;
    let mut packed = vec![0u8; n * row_bytes];
    fill_blocks(&mut packed, codec);
    let mut weight = device
        .alloc_buffer(packed.len(), DType::U8, vec![packed.len()])
        .unwrap();
    weight
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&packed);
    let baseline = GgmlRoutingPolicy::default();
    let candidate = GgmlRoutingPolicy {
        dense_decode_mvn: false,
        dense_decode_mv_ext: true,
        ..baseline
    };
    for m in [1usize, 2, 4, 8] {
        let base_route = plan_dense_auto_route(codec, m as u32, k as u32, &baseline);
        let alt_route = plan_dense_auto_route(codec, m as u32, k as u32, &candidate);
        if m == 1 {
            assert_eq!(base_route, alt_route);
        } else if codec == GgmlType::Q4_K && m == 2 {
            // K-quant ext is admitted only at M4..8; M2 is a fallback control.
            assert_eq!(alt_route, DenseAutoPlan::Mv);
        } else {
            assert_eq!(alt_route, DenseAutoPlan::WidthMvExt);
        }
        if codec == GgmlType::Q4_K && m > 1 {
            assert_eq!(base_route, DenseAutoPlan::Q4kWidthMn);
        }
        if codec == GgmlType::Q8_0 {
            assert_eq!(base_route, DenseAutoPlan::Mv);
            assert!(baseline.dense_q8_0_mv_nr2);
        }
        let classification = if m == 1 {
            "identical-route-control"
        } else if codec == GgmlType::Q4_K && m == 2 {
            "exact-mvn-vs-scalar-mv-control"
        } else {
            "existing-width-mv-ext-candidate"
        };
        eprintln!("K1_ROUTE class={classification} codec={codec:?} m={m} n={n} k={k} baseline={base_route:?} candidate={alt_route:?} q8_nr2={} userdata={declared}", baseline.dense_q8_0_mv_nr2);
        let host_input: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 257) as f32 - 128.0) / 128.0)
            .collect();
        let mut input = device
            .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
            .unwrap();
        input
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&host_input);
        let (base_parent, base_out) = output(&device, m, n);
        let (alt_parent, alt_out) = output(&device, m, n);
        let params = GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: codec,
        };
        for _ in 0..WARM {
            encode(
                &device,
                &mut registry,
                &input,
                &weight,
                &base_out,
                &params,
                &baseline,
                1,
            );
            encode(
                &device,
                &mut registry,
                &input,
                &weight,
                &alt_out,
                &params,
                &candidate,
                1,
            );
        }
        let mut bg = Vec::new();
        let mut ag = Vec::new();
        let mut bw = Vec::new();
        let mut aw = Vec::new();
        // Eight fixed pairs, alternate AB/BA. Retain every raw observation.
        for sample in 0..SAMPLES {
            for arm in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                let (out, policy) = if arm == 0 {
                    (&base_out, &baseline)
                } else {
                    (&alt_out, &candidate)
                };
                let (gpu, wall) = encode(
                    &device,
                    &mut registry,
                    &input,
                    &weight,
                    out,
                    &params,
                    policy,
                    REPS,
                );
                eprintln!("K1_SAMPLE codec={codec:?} m={m} sample={sample} arm={arm} gpu_us={gpu:.9} host_us={wall:.9} reps={REPS}");
                if arm == 0 {
                    bg.push(gpu);
                    bw.push(wall);
                } else {
                    ag.push(gpu);
                    aw.push(wall);
                }
            }
        }
        check_guards(&base_parent, m * n);
        check_guards(&alt_parent, m * n);
        let b = base_out.as_slice::<f32>().unwrap().to_vec();
        let a = alt_out.as_slice::<f32>().unwrap().to_vec();
        // |input|<=1. Q4_K: d=1/64, scale<=63, q<=15,
        // dmin=1/128,min<=63 => |weight|<16, |dot|<81920.
        // Q8_0: d=1/64, |q|<=128 => |dot|<=16384.
        // SENTINEL is outside either bound; every cell must be overwritten.
        assert!(
            b.iter()
                .chain(&a)
                .all(|v| v.to_bits() != SENTINEL.to_bits()),
            "unwritten logical output cell"
        );
        assert!(b.iter().any(|v| *v != 0.0));
        if m == 1 {
            assert!(
                b.iter().zip(&a).all(|(x, y)| x.to_bits() == y.to_bits()),
                "M1 identical-route control"
            );
        }
        let mut max_abs = 0.0f64;
        for (&x, &y) in b.iter().zip(&a) {
            assert!(close(x as f64, y as f64), "all-output mismatch {x} {y}");
            max_abs = max_abs.max((x as f64 - y as f64).abs());
        }
        // Independent existing GGUF dequantizer + f64 CPU dot for five rows
        // across ALL input columns, including first/last/boundary output rows.
        let mut decoded = vec![0f32; k];
        for row in [0, 1, n / 2, n - 2, n - 1] {
            let bytes = &packed[row * row_bytes..(row + 1) * row_bytes];
            match codec {
                GgmlType::Q4_K => dequantize_q4_k(bytes, &mut decoded).unwrap(),
                GgmlType::Q8_0 => dequantize_q8_0(bytes, &mut decoded).unwrap(),
                _ => unreachable!(),
            }
            for col in 0..m {
                let expected: f64 = decoded
                    .iter()
                    .zip(&host_input[col * k..(col + 1) * k])
                    .map(|(&x, &y)| x as f64 * y as f64)
                    .sum();
                assert!(
                    close(expected, b[col * n + row] as f64),
                    "CPU baseline row{row} col{col}"
                );
                assert!(
                    close(expected, a[col * n + row] as f64),
                    "CPU candidate row{row} col{col}"
                );
            }
        }
        eprintln!("K1_RESULT codec={codec:?} m={m} baseline_gpu_us={:.9} candidate_gpu_us={:.9} baseline_host_us={:.9} candidate_host_us={:.9} max_abs={max_abs:.9} abs={ABS} rel={REL} cpu_rows=5", median(&mut bg), median(&mut ag), median(&mut bw), median(&mut aw));
    }
    // All userdata and device/registry owners drop before the next shape.
}
#[test]
fn existing_route_shape_budget_and_route_controls() {
    assert!(userdata(17408, 5120, GgmlType::Q4_K) <= LIMIT);
    assert!(userdata(4096, 8192, GgmlType::Q8_0) <= LIMIT);
    let p = GgmlRoutingPolicy {
        dense_decode_mv_ext: true,
        ..GgmlRoutingPolicy::default()
    };
    assert_eq!(
        plan_dense_auto_route(GgmlType::Q4_K, 4, 5120, &p),
        DenseAutoPlan::Q4kWidthMn
    );
    let alt = GgmlRoutingPolicy {
        dense_decode_mvn: false,
        ..p
    };
    assert_eq!(
        plan_dense_auto_route(GgmlType::Q4_K, 2, 5120, &alt),
        DenseAutoPlan::Mv
    );
    assert_eq!(
        plan_dense_auto_route(GgmlType::Q4_K, 4, 5120, &alt),
        DenseAutoPlan::WidthMvExt
    );
    assert_eq!(
        plan_dense_auto_route(GgmlType::Q8_0, 2, 8192, &alt),
        DenseAutoPlan::WidthMvExt
    );
    assert!(!close(f64::NAN, 1.0));
    assert!(!close(1.0, 2.0));
}
#[test]
#[ignore = "K1 design-only actual-shape GPU comparison; isolated reviewed hardware packet required"]
fn k1_existing_routes_actual_q38_q4k_and_ds_q8_shapes() {
    eprintln!("K1_BEGIN userdata_limit={LIMIT} warm={WARM} samples={SAMPLES} reps={REPS} precision=unchanged synthetic=true");
    case(GgmlType::Q4_K, 17408, 5120);
    case(GgmlType::Q8_0, 4096, 8192);
    eprintln!("K1_EXISTING_ROUTES_COMPLETE shapes=2 widths=1,2,4,8 full_q1=false end_to_end=false");
}
