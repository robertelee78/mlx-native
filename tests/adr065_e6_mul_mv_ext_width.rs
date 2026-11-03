//! ADR-065 E6 probe 1: K-quant multi-column matvec at 8 rows versus 4 rows.
//!
//! A verify window groups the rows of several slots into one projection.
//! The probe asks whether an 8-row dispatch produces, for every column, the
//! same bits as the 4-row dispatch that holds the same column. Two routes are
//! probed at every Qwen3.8-27B dense projection shape and K-quant codec:
//!
//! * `mv_ext` — the generic `mul_mv_ext` route (policy: canonical Q5_K and mN
//!   disabled, `dense_decode_mv_ext` enabled). This is the ADR-065 E6
//!   acceptance bullet.
//! * `served` — the default routing policy, i.e. what hf2q serves today
//!   (Q4_K/Q6_K exact mN tiles, Q5_K canonical q4x4).
//!
//! Every case asserts column bitwise equality and emits an `E6_RECORD` line.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");

use mlx_native::{
    quantized_matmul_ggml_with_policy_and_trace, GgmlQuantizedMatmulParams,
    GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass,
};

/// Qwen3.8-27B dense projections as `(name, N output rows, K input width)`.
/// Hidden 5120, FFN 17408, 24 query heads x 256 (gated Q doubles it),
/// 4 KV heads x 256, DeltaNet 16 key heads + 48 value heads x 128.
const QWEN38_SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate_up", 17_408, 5_120),
    ("ffn_down", 5_120, 17_408),
    ("attn_q_gate", 12_288, 5_120),
    ("attn_kv", 1_024, 5_120),
    ("attn_output", 5_120, 6_144),
    ("deltanet_qkv", 10_240, 5_120),
    ("deltanet_z", 6_144, 5_120),
    ("deltanet_output", 5_120, 6_144),
];

const UNWRITTEN: u32 = 0x7fc1_e6e6;

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
            GgmlType::Q6_K => {
                // ql[128] qh[64] scales[16] d(f16)
                bytes[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            other => panic!("not a probed K-quant: {other:?}"),
        }
    }
}

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

fn expected_route(route: &str, ggml_type: GgmlType) -> GgmlResolvedKernelRoute {
    match (route, ggml_type) {
        ("mv_ext", _) => GgmlResolvedKernelRoute::DenseWidthMvExt,
        ("served", GgmlType::Q4_K) => GgmlResolvedKernelRoute::DenseQ4kWidthMn,
        ("served", GgmlType::Q5_K) => GgmlResolvedKernelRoute::DenseQ5kCanonicalQ4x4,
        ("served", GgmlType::Q6_K) => GgmlResolvedKernelRoute::DenseQ6kWidthMn,
        other => panic!("no expected route for {other:?}"),
    }
}

/// Run one dense projection of `m` columns and return its `[m, n]` output
/// plus the observed pipeline origins.
#[allow(clippy::too_many_arguments)]
fn project(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    m: usize,
    n: usize,
    k: usize,
    ggml_type: GgmlType,
    route: &str,
) -> (Vec<f32>, Vec<String>) {
    let mut output = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
        .expect("output");
    output
        .as_mut_slice::<f32>()
        .expect("output slice")
        .fill(f32::from_bits(UNWRITTEN));
    let mut encoder = device.command_encoder().expect("encoder");
    let trace = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        registry,
        device,
        input,
        weight,
        &output,
        &GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        },
        &policy_for(route),
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("dense K-quant projection");
    assert_eq!(trace.resolved_route, expected_route(route, ggml_type));
    encoder.commit_and_wait().expect("projection execution");
    let origins = trace
        .dispatches
        .iter()
        .map(|dispatch| {
            let label = &dispatch.pipeline.pipeline_label;
            e6_pipeline_origin(registry, label, label.contains('|'))
        })
        .collect();
    let values = e6_read_f32(&output);
    for (index, value) in values.iter().enumerate() {
        assert_ne!(
            value.to_bits(),
            UNWRITTEN,
            "{route} m={m} element {index} unwritten"
        );
    }
    (values, origins)
}

fn probe_codec(ggml_type: GgmlType, route: &str) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for &(shape, n, k) in QWEN38_SHAPES {
        let mut rng = E6Rng::new(0xe6_0001 ^ ((n as u64) << 20) ^ k as u64);
        let blocks = n * (k / ggml_type.block_values() as usize);
        let bytes = blocks * ggml_type.block_bytes() as usize;
        let mut weight = device
            .alloc_buffer(bytes, DType::U8, vec![bytes])
            .expect("weight");
        fill_kquant(&mut weight, ggml_type, &mut rng);
        let activations: Vec<f32> = (0..8 * k).map(|_| rng.signed(1.0)).collect();
        let input8 = e6_f32_buffer(&device, &activations);

        let (wide, origins) = project(
            &device,
            &mut registry,
            &weight,
            &input8,
            8,
            n,
            k,
            ggml_type,
            route,
        );
        let mut narrow = Vec::with_capacity(8 * n);
        for group in 0..2 {
            let view = input8.slice_view((group * 4 * k * 4) as u64, 4 * k);
            let (part, _) = project(
                &device,
                &mut registry,
                &weight,
                &view,
                4,
                n,
                k,
                ggml_type,
                route,
            );
            narrow.extend_from_slice(&part);
        }
        let compare = E6Compare::new(&narrow, &wide);
        let mut fields = e6_compare_fields(&compare);
        fields.push(("type", format!("{ggml_type:?}")));
        fields.push(("n", n.to_string()));
        fields.push(("k", k.to_string()));
        fields.push(("origin", origins.join(",")));
        e6_record(
            "mul_mv_ext_8_vs_4",
            &format!("{route}/{ggml_type:?}/{shape}"),
            &fields,
        );
        assert!(
            compare.bitwise(),
            "{route} {ggml_type:?} {shape} (N={n} K={k}): 8-row columns differ from 4-row columns \
             ({} of {} elements, first {:?})",
            compare.mismatched,
            compare.elements,
            compare.first_mismatch
        );
    }
}

#[test]
fn e6_mul_mv_ext_q4_k_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q4_K, "mv_ext");
}

#[test]
fn e6_mul_mv_ext_q5_k_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q5_K, "mv_ext");
}

#[test]
fn e6_mul_mv_ext_q6_k_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q6_K, "mv_ext");
}

#[test]
fn e6_served_q4_k_route_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q4_K, "served");
}

#[test]
fn e6_served_q5_k_route_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q5_K, "served");
}

#[test]
fn e6_served_q6_k_route_8_rows_match_4_rows_at_qwen38_shapes() {
    probe_codec(GgmlType::Q6_K, "served");
}
