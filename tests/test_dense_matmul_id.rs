//! Native scalar expert-ID matmul parity and contract tests.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::{bf16, f16};
use mlx_native::{
    dense_matmul_id, dense_matmul_id_capability, dispatch_count, reset_counters,
    trace_dense_matmul_id, CapturedNode, DType, DenseMatmulIdCapability, DenseMatmulIdInputLayout,
    DenseMatmulIdMultiplicity, DenseMatmulIdParams, DenseMatmulIdRoute, DenseMatmulIdScratch,
    DispatchKind, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice,
    DENSE_MATMUL_ID_SCHEMA_VERSION,
};

fn scalar_value(dtype: DType, value: f32) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(value).to_f32(),
        DType::F16 => f16::from_f32(value).to_f32(),
        DType::F32 => value,
        other => panic!("unsupported test dtype {other}"),
    }
}

fn finite_scalar_edge_classes(dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => [
            0x0000_0000u32,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x0080_0000,
            0x8080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
        ]
        .into_iter()
        .map(f32::from_bits)
        .collect(),
        DType::F16 => [
            0x0000u16, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x7bff, 0xfbff,
        ]
        .into_iter()
        .map(|bits| f16::from_bits(bits).to_f32())
        .collect(),
        DType::BF16 => [
            0x0000u16, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x7f7f, 0xff7f,
        ]
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect(),
        other => panic!("unsupported test dtype {other}"),
    }
}

fn write_scalar(bytes: &mut [u8], offset: usize, dtype: DType, value: f32) {
    match dtype {
        DType::BF16 => bytes[offset..offset + 2]
            .copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes()),
        DType::F16 => {
            bytes[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
        }
        DType::F32 => bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes()),
        other => panic!("unsupported test dtype {other}"),
    }
}

fn make_weights(
    device: &MlxDevice,
    dtype: DType,
    experts: usize,
    n: usize,
    k: usize,
) -> (MlxBuffer, usize, Vec<f32>) {
    let matrix_bytes = n * k * dtype.size_of();
    let padding = match dtype {
        DType::F32 => 12,
        DType::F16 | DType::BF16 => 14,
        _ => unreachable!(),
    };
    let stride = matrix_bytes + padding;
    let mut buffer = device
        .alloc_buffer(stride * experts, dtype, vec![experts, n, k])
        .expect("weights");
    let bytes = buffer.as_mut_slice::<u8>().expect("weight bytes");
    bytes.fill(0xD7);
    let mut values = vec![0.0f32; experts * n * k];
    for expert in 0..experts {
        for col in 0..n {
            for inner in 0..k {
                let raw = (((expert * 131 + col * 29 + inner * 17) % 257) as f32 - 128.0) / 193.0;
                let stored = scalar_value(dtype, raw);
                values[(expert * n + col) * k + inner] = stored;
                write_scalar(
                    bytes,
                    expert * stride + (col * k + inner) * dtype.size_of(),
                    dtype,
                    raw,
                );
            }
        }
    }
    (buffer, stride, values)
}

fn ids_for(m: usize, top_k: usize, duplicates: bool) -> Vec<u32> {
    let unique = [1u32, 3, 7, 0, 2, 5, 4, 6];
    let repeated = [1u32, 1, 3, 7, 7, 3];
    let mut ids = Vec::with_capacity(m * top_k);
    for token in 0..m {
        for slot in 0..top_k {
            ids.push(if duplicates {
                repeated[slot % repeated.len()]
            } else if top_k == 1 {
                unique[token % 3]
            } else {
                unique[(token + slot) % unique.len()]
            });
        }
    }
    ids
}

fn input_for(m: usize, top_k: usize, k: usize, layout: DenseMatmulIdInputLayout) -> Vec<f32> {
    let rows = match layout {
        DenseMatmulIdInputLayout::SharedPerToken => m,
        DenseMatmulIdInputLayout::Slotted => m * top_k,
    };
    (0..rows * k)
        .map(|index| (((index * 43 + 11) % 211) as f32 - 105.0) / 173.0)
        .collect()
}

fn cpu_reference(
    _dtype: DType,
    weights: &[f32],
    input: &[f32],
    ids: &[u32],
    params: &DenseMatmulIdParams,
) -> Vec<f32> {
    let m = params.m as usize;
    let top_k = params.top_k as usize;
    let n = params.n as usize;
    let k = params.k as usize;
    let mut output = vec![0.0f32; m * top_k * n];
    for token in 0..m {
        for slot in 0..top_k {
            let flat = token * top_k + slot;
            let expert = ids[flat] as usize;
            let input_row = match params.input_layout {
                DenseMatmulIdInputLayout::SharedPerToken => token,
                DenseMatmulIdInputLayout::Slotted => flat,
            };
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    let activation = input[input_row * k + inner];
                    sum += weights[(expert * n + col) * k + inner] * activation;
                }
                output[(flat * n) + col] = sum;
            }
        }
    }
    output
}

fn run_case(
    dtype: DType,
    m: usize,
    top_k: usize,
    layout: DenseMatmulIdInputLayout,
    multiplicity: DenseMatmulIdMultiplicity,
    duplicates: bool,
) -> DenseMatmulIdRoute {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let experts = 8usize;
    let n = 7usize;
    let k = 35usize;
    let (weights, expert_stride_bytes, weight_values) = make_weights(&device, dtype, experts, n, k);
    let input_values = input_for(m, top_k, k, layout);
    let ids_values = ids_for(m, top_k, duplicates);
    let mut input = device
        .alloc_buffer(input_values.len() * 4, DType::F32, vec![input_values.len()])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input_values);
    let mut ids = device
        .alloc_buffer(ids_values.len() * 4, DType::U32, vec![m, top_k])
        .expect("ids");
    ids.as_mut_slice::<u32>()
        .expect("ids slice")
        .copy_from_slice(&ids_values);
    let mut output = device
        .alloc_buffer(m * top_k * n * 4, DType::F32, vec![m, top_k, n])
        .expect("output");
    output
        .as_mut_slice::<u32>()
        .expect("output poison")
        .fill(0x7FC0_7BAD);

    let params = DenseMatmulIdParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        top_k: top_k as u32,
        n_experts: experts as u32,
        expert_stride_bytes: expert_stride_bytes as u64,
        input_layout: layout,
        id_multiplicity: multiplicity,
        route: if m >= 9 && multiplicity == DenseMatmulIdMultiplicity::DistinctPerToken {
            DenseMatmulIdRoute::GroupedPrefill
        } else {
            DenseMatmulIdRoute::Direct
        },
    };
    let capability = dense_matmul_id_capability(dtype, &params).expect("capability");
    let scratch = DenseMatmulIdScratch::new(&device, experts as u32, m as u32).expect("scratch");
    let mut encoder = device.command_encoder().expect("encoder");
    let receipt = dense_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &output,
        Some(&scratch),
        &params,
    )
    .expect("dense_matmul_id");
    encoder.commit_and_wait().expect("commit");
    assert_eq!(receipt.route, capability.route);
    assert_eq!(
        receipt.dispatch_count,
        if receipt.route == DenseMatmulIdRoute::Direct {
            1
        } else {
            2
        }
    );

    let expected = cpu_reference(dtype, &weight_values, &input_values, &ids_values, &params);
    let actual = output.as_slice::<f32>().expect("output slice");
    let tolerance = match dtype {
        DType::F32 => 2e-4,
        DType::F16 => 3e-2,
        DType::BF16 => 6e-2,
        _ => unreachable!(),
    };
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let delta = (got - want).abs();
        assert!(
            delta <= tolerance + tolerance * want.abs(),
            "dtype={dtype} M={m} top_k={top_k} layout={layout:?} route={:?} mismatch at {index}: got={got} want={want} delta={delta}",
            receipt.route,
        );
    }
    receipt.route
}

fn run_scalar_route_coherence_case(
    dtype: DType,
    m: usize,
    layout: DenseMatmulIdInputLayout,
    include_invalid_ids: bool,
) {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let experts = 8usize;
    let top_k = 6usize;
    let n = 7usize;
    let k = 35usize;
    let (mut weights, expert_stride_bytes, _) = make_weights(&device, dtype, experts, n, k);
    let scalar_edges = finite_scalar_edge_classes(dtype);
    let weight_bytes = weights.as_mut_slice::<u8>().expect("weight bytes");
    for expert in 0..experts {
        for col in 0..n {
            for inner in 0..k {
                let edge = scalar_edges[(inner + col * 3 + expert * 5) % scalar_edges.len()];
                write_scalar(
                    weight_bytes,
                    expert * expert_stride_bytes + (col * k + inner) * dtype.size_of(),
                    dtype,
                    edge,
                );
            }
        }
    }
    let rows = match layout {
        DenseMatmulIdInputLayout::SharedPerToken => m,
        DenseMatmulIdInputLayout::Slotted => m * top_k,
    };
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
    let input_values: Vec<f32> = (0..rows * k)
        .map(|index| {
            let base = adversarial[(index * 5 + index / k) % adversarial.len()];
            let scaled = base * (1.0 + ((index * 17) % 11) as f32 / 4096.0);
            // Maximum finite F32/BF16 weights remain finite through F32 FMA,
            // so the canary isolates route coherence from overflow behavior.
            scaled / 1_048_576.0
        })
        .collect();
    let mut ids_values = ids_for(m, top_k, false);
    if include_invalid_ids {
        ids_values[1] = experts as u32;
        ids_values[(m / 2) * top_k + (top_k - 1)] = u32::MAX;
    }
    let mut input = device
        .alloc_buffer(input_values.len() * 4, DType::F32, vec![rows, k])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input_values);
    let mut ids = device
        .alloc_buffer(ids_values.len() * 4, DType::U32, vec![m, top_k])
        .expect("ids");
    ids.as_mut_slice::<u32>()
        .expect("ids slice")
        .copy_from_slice(&ids_values);
    let mut direct_output = device
        .alloc_buffer(m * top_k * n * 4, DType::F32, vec![m, top_k, n])
        .expect("direct output");
    let mut grouped_output = device
        .alloc_buffer(m * top_k * n * 4, DType::F32, vec![m, top_k, n])
        .expect("grouped output");
    direct_output
        .as_mut_slice::<u32>()
        .expect("direct poison")
        .fill(0x7FC0_7BAD);
    grouped_output
        .as_mut_slice::<u32>()
        .expect("grouped poison")
        .fill(0x7FC0_7BAD);
    let scratch = DenseMatmulIdScratch::new(&device, experts as u32, m as u32).expect("scratch");
    let grouped_params = DenseMatmulIdParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        top_k: top_k as u32,
        n_experts: experts as u32,
        expert_stride_bytes: expert_stride_bytes as u64,
        input_layout: layout,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };
    let direct_params = DenseMatmulIdParams {
        route: DenseMatmulIdRoute::Direct,
        ..grouped_params
    };

    let mut direct_encoder = device.command_encoder().expect("direct encoder");
    let direct_receipt = dense_matmul_id(
        &mut direct_encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &direct_output,
        Some(&scratch),
        &direct_params,
    )
    .expect("direct dispatch");
    direct_encoder.commit_and_wait().expect("direct completion");
    let mut grouped_encoder = device.command_encoder().expect("grouped encoder");
    let grouped_receipt = dense_matmul_id(
        &mut grouped_encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &grouped_output,
        Some(&scratch),
        &grouped_params,
    )
    .expect("grouped dispatch");
    grouped_encoder
        .commit_and_wait()
        .expect("grouped completion");
    assert_eq!(direct_receipt.route, DenseMatmulIdRoute::Direct);
    assert_eq!(grouped_receipt.route, DenseMatmulIdRoute::GroupedPrefill);

    let direct = direct_output.as_slice::<f32>().expect("direct values");
    let grouped = grouped_output.as_slice::<f32>().expect("grouped values");
    let max_delta = direct
        .iter()
        .zip(grouped)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        direct.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        grouped
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "dtype={dtype} M={m} layout={layout:?} invalid_ids={include_invalid_ids} direct/grouped reduction trajectory changed; max_delta={max_delta:e}"
    );
    println!(
        "native scalar forced-route coherence dtype={dtype} M={m} layout={layout:?} invalid_ids={include_invalid_ids}: max_delta={max_delta:e}, bitwise_equal=true"
    );
}

#[test]
fn bf16_route_matrix_covers_decode_prefill_layouts_and_odd_tail() {
    for &m in &[1usize, 2, 8, 9, 33] {
        for &top_k in &[1usize, 6] {
            for &layout in &[
                DenseMatmulIdInputLayout::SharedPerToken,
                DenseMatmulIdInputLayout::Slotted,
            ] {
                let route = run_case(
                    DType::BF16,
                    m,
                    top_k,
                    layout,
                    DenseMatmulIdMultiplicity::DistinctPerToken,
                    false,
                );
                assert_eq!(
                    route,
                    if m <= 8 {
                        DenseMatmulIdRoute::Direct
                    } else {
                        DenseMatmulIdRoute::GroupedPrefill
                    }
                );
            }
        }
    }
}

#[test]
fn f16_and_f32_remain_native_scalar_routes() {
    assert_eq!(
        run_case(
            DType::F16,
            9,
            6,
            DenseMatmulIdInputLayout::SharedPerToken,
            DenseMatmulIdMultiplicity::DistinctPerToken,
            false,
        ),
        DenseMatmulIdRoute::GroupedPrefill,
    );
    assert_eq!(
        run_case(
            DType::F32,
            2,
            6,
            DenseMatmulIdInputLayout::Slotted,
            DenseMatmulIdMultiplicity::DistinctPerToken,
            false,
        ),
        DenseMatmulIdRoute::Direct,
    );
}

#[test]
fn bf16_f32_activation_is_not_rounded_to_weight_dtype() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut weights = device
        .alloc_buffer(2, DType::BF16, vec![1, 1, 1])
        .expect("weights");
    weights.as_mut_slice::<u16>().unwrap()[0] = bf16::from_f32(1.0).to_bits();
    let activation = 1.000_976_6f32;
    assert_ne!(bf16::from_f32(activation).to_f32(), activation);
    let mut input = device
        .alloc_buffer(4, DType::F32, vec![1, 1])
        .expect("input");
    input.as_mut_slice::<f32>().unwrap()[0] = activation;
    let mut ids = device.alloc_buffer(4, DType::U32, vec![1, 1]).expect("ids");
    ids.as_mut_slice::<u32>().unwrap()[0] = 0;
    let output = device
        .alloc_buffer(4, DType::F32, vec![1, 1, 1])
        .expect("output");
    let params = DenseMatmulIdParams {
        m: 1,
        n: 1,
        k: 1,
        top_k: 1,
        n_experts: 1,
        expert_stride_bytes: 2,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::Direct,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    dense_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &output,
        None,
        &params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("completion");
    assert_eq!(
        output.as_slice::<f32>().unwrap()[0].to_bits(),
        activation.to_bits()
    );
}

#[test]
fn scalar_grouped_and_direct_routes_share_adversarial_f32_trajectory_at_tails() {
    for &dtype in &[DType::F32, DType::F16, DType::BF16] {
        for &m in &[9usize, 33] {
            for &layout in &[
                DenseMatmulIdInputLayout::SharedPerToken,
                DenseMatmulIdInputLayout::Slotted,
            ] {
                run_scalar_route_coherence_case(dtype, m, layout, false);
            }
        }
    }
}

#[test]
fn invalid_expert_ids_fully_overwrite_poison_in_both_routes_and_layouts() {
    for &layout in &[
        DenseMatmulIdInputLayout::SharedPerToken,
        DenseMatmulIdInputLayout::Slotted,
    ] {
        run_scalar_route_coherence_case(DType::BF16, 9, layout, true);
    }
}

#[test]
fn nonzero_slice_views_preserve_bitwise_routes_and_exact_dispatch_trace() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let m = 9usize;
    let top_k = 6usize;
    let n = 7usize;
    let k = 35usize;
    let experts = 8usize;
    let prefix_bytes = 16usize;

    let (base_weights, stride, _) = make_weights(&device, DType::BF16, experts, n, k);
    let source_weight_bytes = base_weights.as_slice::<u8>().unwrap().to_vec();
    let mut weight_parent = device
        .alloc_buffer(
            prefix_bytes + source_weight_bytes.len() + prefix_bytes,
            DType::BF16,
            vec![(prefix_bytes * 2 + source_weight_bytes.len()) / 2],
        )
        .unwrap();
    weight_parent.as_mut_slice::<u8>().unwrap().fill(0xA5);
    weight_parent.as_mut_slice::<u8>().unwrap()
        [prefix_bytes..prefix_bytes + source_weight_bytes.len()]
        .copy_from_slice(&source_weight_bytes);
    let weights = weight_parent.slice_view(
        prefix_bytes as u64,
        source_weight_bytes.len() / DType::BF16.size_of(),
    );

    let input_values = input_for(m, top_k, k, DenseMatmulIdInputLayout::SharedPerToken);
    let mut input_parent = device
        .alloc_buffer(
            prefix_bytes + input_values.len() * 4 + prefix_bytes,
            DType::F32,
            vec![input_values.len() + 8],
        )
        .unwrap();
    input_parent
        .as_mut_slice::<u32>()
        .unwrap()
        .fill(0x7fc0_00d1);
    input_parent.as_mut_slice::<f32>().unwrap()[4..4 + input_values.len()]
        .copy_from_slice(&input_values);
    let input = input_parent.slice_view(prefix_bytes as u64, input_values.len());

    let id_values = ids_for(m, top_k, false);
    let mut ids_parent = device
        .alloc_buffer(
            prefix_bytes + id_values.len() * 4 + prefix_bytes,
            DType::U32,
            vec![id_values.len() + 8],
        )
        .unwrap();
    ids_parent.as_mut_slice::<u32>().unwrap().fill(u32::MAX);
    ids_parent.as_mut_slice::<u32>().unwrap()[4..4 + id_values.len()].copy_from_slice(&id_values);
    let ids = ids_parent.slice_view(prefix_bytes as u64, id_values.len());

    let output_elements = m * top_k * n;
    let mut direct_parent = device
        .alloc_buffer(
            prefix_bytes + output_elements * 4 + prefix_bytes,
            DType::F32,
            vec![output_elements + 8],
        )
        .unwrap();
    let mut grouped_parent = device
        .alloc_buffer(
            prefix_bytes + output_elements * 4 + prefix_bytes,
            DType::F32,
            vec![output_elements + 8],
        )
        .unwrap();
    direct_parent
        .as_mut_slice::<u32>()
        .unwrap()
        .fill(0x7fc0_00d2);
    grouped_parent
        .as_mut_slice::<u32>()
        .unwrap()
        .fill(0x7fc0_00d2);
    let direct_output = direct_parent.slice_view(prefix_bytes as u64, output_elements);
    let grouped_output = grouped_parent.slice_view(prefix_bytes as u64, output_elements);

    let grouped_params = DenseMatmulIdParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        top_k: top_k as u32,
        n_experts: experts as u32,
        expert_stride_bytes: stride as u64,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };
    let direct_params = DenseMatmulIdParams {
        route: DenseMatmulIdRoute::Direct,
        ..grouped_params
    };
    let scratch = DenseMatmulIdScratch::new(&device, experts as u32, m as u32).unwrap();

    let mut direct_encoder = device.command_encoder().unwrap();
    let direct_trace = trace_dense_matmul_id(
        &mut direct_encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &direct_output,
        Some(&scratch),
        &direct_params,
    )
    .unwrap();
    direct_encoder.commit_and_wait().unwrap();
    assert_eq!(direct_trace.encoded.len(), 1);
    assert_eq!(
        direct_trace.encoded[0].dispatch_kind,
        DispatchKind::ThreadGroups
    );
    assert_eq!(direct_trace.encoded[0].grid, [1, 54, 1]);
    assert_eq!(direct_trace.encoded[0].threads_per_threadgroup, [64, 1, 1]);

    let mut grouped_encoder = device.command_encoder().unwrap();
    let grouped_trace = trace_dense_matmul_id(
        &mut grouped_encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &grouped_output,
        Some(&scratch),
        &grouped_params,
    )
    .unwrap();
    grouped_encoder.commit_and_wait().unwrap();
    assert_eq!(grouped_trace.encoded.len(), 2);
    assert_eq!(grouped_trace.encoded[0].grid, [1, 1, 1]);
    assert_eq!(
        grouped_trace.encoded[0].threads_per_threadgroup,
        [experts as u64, 1, 1]
    );
    assert_eq!(grouped_trace.encoded[1].grid, [2, 1, experts as u64]);
    assert_eq!(
        grouped_trace.encoded[1].threads_per_threadgroup,
        [256, 1, 1]
    );
    assert_eq!(grouped_trace.encoded[1].threadgroup_memory, vec![(0, 4096)]);

    assert_eq!(
        direct_output
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        grouped_output
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    for parent in [&direct_parent, &grouped_parent] {
        let bits = parent.as_slice::<u32>().unwrap();
        assert!(bits[..4].iter().all(|value| *value == 0x7fc0_00d2));
        assert!(bits[4 + output_elements..]
            .iter()
            .all(|value| *value == 0x7fc0_00d2));
    }
}

#[test]
fn duplicate_ids_are_explicitly_supported_without_a_hidden_distinctness_assumption() {
    assert_eq!(
        run_case(
            DType::BF16,
            33,
            6,
            DenseMatmulIdInputLayout::Slotted,
            DenseMatmulIdMultiplicity::MayRepeat,
            true,
        ),
        DenseMatmulIdRoute::Direct,
    );
}

#[test]
fn capability_rejects_impossible_scalar_contracts_before_encoding() {
    let base = DenseMatmulIdParams {
        m: 9,
        n: 7,
        k: 35,
        top_k: 6,
        n_experts: 8,
        expert_stride_bytes: 504,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::Direct,
    };
    assert!(dense_matmul_id_capability(DType::U32, &base).is_err());
    assert!(
        dense_matmul_id_capability(DType::BF16, &DenseMatmulIdParams { m: 0, ..base }).is_err()
    );
    assert!(dense_matmul_id_capability(
        DType::BF16,
        &DenseMatmulIdParams {
            expert_stride_bytes: 489,
            ..base
        }
    )
    .is_err());
    assert!(dense_matmul_id_capability(
        DType::BF16,
        &DenseMatmulIdParams {
            n_experts: u32::MAX,
            expert_stride_bytes: u64::MAX,
            ..base
        }
    )
    .is_err());

    let capability = dense_matmul_id_capability(DType::BF16, &base).unwrap();
    assert_eq!(capability.schema_version, DENSE_MATMUL_ID_SCHEMA_VERSION);
    let json = serde_json::to_string(&capability).unwrap();
    let round_trip: DenseMatmulIdCapability = serde_json::from_str(&json).unwrap();
    assert_eq!(round_trip, capability);
    let mut value = serde_json::to_value(&capability).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<DenseMatmulIdCapability>(value).is_err());
}

#[test]
fn buffer_and_scratch_extent_failures_encode_nothing() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let params = DenseMatmulIdParams {
        m: 9,
        n: 7,
        k: 35,
        top_k: 6,
        n_experts: 8,
        expert_stride_bytes: 504,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };
    let capability = dense_matmul_id_capability(DType::BF16, &params).unwrap();
    let short_weights = device
        .alloc_buffer(
            capability.required_weight_bytes - 1,
            DType::BF16,
            vec![capability.required_weight_bytes / 2],
        )
        .unwrap();
    let input = device
        .alloc_buffer(capability.required_input_bytes, DType::F32, vec![9, 35])
        .unwrap();
    let ids = device
        .alloc_buffer(capability.required_ids_bytes, DType::U32, vec![9, 6])
        .unwrap();
    let output = device
        .alloc_buffer(capability.required_output_bytes, DType::F32, vec![9, 6, 7])
        .unwrap();
    let scratch = DenseMatmulIdScratch::new(&device, 8, 9).unwrap();
    let mut encoder = device.command_encoder().unwrap();

    reset_counters();
    let error = dense_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &short_weights,
        &input,
        &ids,
        &output,
        Some(&scratch),
        &params,
    )
    .unwrap_err();
    assert!(error.to_string().contains("weights requires"));
    assert_eq!(
        dispatch_count(),
        0,
        "extent rejection must precede encoding"
    );

    let (weights, _, _) = make_weights(&device, DType::BF16, 8, 7, 35);
    let undersized = DenseMatmulIdScratch::new(&device, 7, 8).unwrap();
    let error = dense_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &weights,
        &input,
        &ids,
        &output,
        Some(&undersized),
        &params,
    )
    .unwrap_err();
    assert!(error.to_string().contains("scratch capacity"));
    assert_eq!(
        dispatch_count(),
        0,
        "scratch rejection must precede encoding"
    );
}

#[test]
fn output_alias_is_rejected_before_encoding() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let params = DenseMatmulIdParams {
        m: 1,
        n: 1,
        k: 4,
        top_k: 1,
        n_experts: 1,
        expert_stride_bytes: 8,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::Direct,
    };
    let weights = device
        .alloc_buffer(8, DType::BF16, vec![1, 1, 4])
        .expect("weights");
    let input_and_output = device
        .alloc_buffer(16, DType::F32, vec![1, 4])
        .expect("aliased input/output");
    let ids = device.alloc_buffer(4, DType::U32, vec![1, 1]).expect("ids");
    let mut encoder = device.command_encoder().expect("encoder");
    reset_counters();
    let error = dense_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &weights,
        &input_and_output,
        &ids,
        &input_and_output,
        None,
        &params,
    )
    .unwrap_err();
    assert!(error.to_string().contains("output must not overlap input"));
    assert_eq!(dispatch_count(), 0, "alias rejection must precede encoding");
}

#[test]
fn graph_session_declares_both_grouped_dispatch_hazards() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let m = 9usize;
    let top_k = 6usize;
    let n = 7usize;
    let k = 35usize;
    let experts = 8usize;
    let (weights, stride, _) = make_weights(&device, DType::BF16, experts, n, k);
    let input_values = input_for(m, top_k, k, DenseMatmulIdInputLayout::SharedPerToken);
    let ids_values = ids_for(m, top_k, false);
    let mut input = device
        .alloc_buffer(input_values.len() * 4, DType::F32, vec![m, k])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_values);
    let mut ids = device
        .alloc_buffer(ids_values.len() * 4, DType::U32, vec![m, top_k])
        .expect("ids");
    ids.as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids_values);
    let output = device
        .alloc_buffer(m * top_k * n * 4, DType::F32, vec![m, top_k, n])
        .expect("output");
    let scratch = DenseMatmulIdScratch::new(&device, experts as u32, m as u32).unwrap();
    let params = DenseMatmulIdParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        top_k: top_k as u32,
        n_experts: experts as u32,
        expert_stride_bytes: stride as u64,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };

    let executor = GraphExecutor::new(device.clone());
    let mut session = executor.begin_recorded().expect("recorded session");
    let receipt = session
        .dense_matmul_id(
            &mut registry,
            &device,
            &weights,
            &input,
            &ids,
            &output,
            Some(&scratch),
            &params,
        )
        .expect("graph dense_matmul_id");
    assert_eq!(receipt.dispatch_count, 2);
    session
        .finish_with_reorder()
        .expect("grouped dispatches must carry complete hazard ranges");
}

#[test]
fn grouped_scratch_reuse_encodes_inter_call_barrier_before_second_map() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let m = 9usize;
    let top_k = 6usize;
    let n = 7usize;
    let k = 35usize;
    let experts = 8usize;
    let (weights, stride, _) = make_weights(&device, DType::BF16, experts, n, k);
    let input_values = input_for(m, top_k, k, DenseMatmulIdInputLayout::SharedPerToken);
    let ids_values = ids_for(m, top_k, false);
    let mut input = device
        .alloc_buffer(input_values.len() * 4, DType::F32, vec![m, k])
        .unwrap();
    input
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_values);
    let mut ids = device
        .alloc_buffer(ids_values.len() * 4, DType::U32, vec![m, top_k])
        .unwrap();
    ids.as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids_values);
    let output = device
        .alloc_buffer(m * top_k * n * 4, DType::F32, vec![m, top_k, n])
        .unwrap();
    let scratch = DenseMatmulIdScratch::new(&device, experts as u32, m as u32).unwrap();
    let params = DenseMatmulIdParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        top_k: top_k as u32,
        n_experts: experts as u32,
        expert_stride_bytes: stride as u64,
        input_layout: DenseMatmulIdInputLayout::SharedPerToken,
        id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
        route: DenseMatmulIdRoute::GroupedPrefill,
    };

    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    for _ in 0..2 {
        dense_matmul_id(
            &mut encoder,
            &mut registry,
            &device,
            &weights,
            &input,
            &ids,
            &output,
            Some(&scratch),
            &params,
        )
        .unwrap();
    }
    let captured = encoder.take_capture().unwrap();
    assert_eq!(captured.len(), 8);
    for index in [0usize, 2, 4, 6] {
        assert!(
            matches!(captured[index], CapturedNode::Barrier),
            "expected barrier at captured node {index}"
        );
    }
    for index in [1usize, 3, 5, 7] {
        assert!(
            matches!(captured[index], CapturedNode::Dispatch { .. }),
            "expected dispatch at captured node {index}"
        );
    }
    // The critical reuse edge is grouped1 -> barrier -> map2. Each invocation
    // also retains its independent map -> barrier -> grouped RAW edge.
    assert!(matches!(captured[3], CapturedNode::Dispatch { .. }));
    assert!(matches!(captured[4], CapturedNode::Barrier));
    assert!(matches!(captured[5], CapturedNode::Dispatch { .. }));
}
