//! Correctness and boundary tests for the four-row-tiled BF16 GEMV path.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::dense_gemv_bf16::{dense_gemv_bf16_f32, dense_gemv_bf16_f32_tiled4};
use mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn bf16_values(count: usize, seed: usize) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let coarse = (index.wrapping_mul(37).wrapping_add(seed) % 251) as f32 - 125.0;
            bf16::from_f32(coarse / 509.0).to_bits()
        })
        .collect()
}

fn f32_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let coarse = (index.wrapping_mul(29).wrapping_add(seed) % 241) as f32 - 120.0;
            let fine = (index.wrapping_mul(43).wrapping_add(seed) % 17) as f32 - 8.0;
            coarse / 1003.0 + fine / 17_003.0
        })
        .collect()
}

fn buffer_from_bf16(device: &MlxDevice, values: &[u16]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 2, DType::BF16, vec![values.len()])
        .expect("allocate BF16 buffer");
    buffer
        .as_mut_slice::<u16>()
        .expect("map BF16 buffer")
        .copy_from_slice(values);
    buffer
}

fn buffer_from_f32(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("allocate F32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("map F32 buffer")
        .copy_from_slice(values);
    buffer
}

fn output_buffer(device: &MlxDevice, elements: usize) -> MlxBuffer {
    device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("allocate output buffer")
}

fn execute(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseMmBf16F32Params,
    tiled4: bool,
) {
    let mut encoder = device.command_encoder().expect("command encoder");
    if tiled4 {
        dense_gemv_bf16_f32_tiled4(
            &mut encoder,
            registry,
            device,
            weight,
            input,
            output,
            params,
        )
        .expect("tiled4 dispatch");
    } else {
        dense_gemv_bf16_f32(
            &mut encoder,
            registry,
            device,
            weight,
            input,
            output,
            params,
        )
        .expect("row dispatch");
    }
    encoder.commit_and_wait().expect("GPU completion");
}

fn cpu_reference(weight: &[u16], input: &[f32], params: &DenseMmBf16F32Params) -> Vec<f32> {
    let mut output = vec![0.0f32; (params.src1_batch * params.m * params.n) as usize];
    let broadcast = params.src1_batch / params.src0_batch;
    for batch in 0..params.src1_batch as usize {
        let weight_batch = batch / broadcast as usize;
        for row in 0..params.m as usize {
            for out in 0..params.n as usize {
                let mut sum = 0.0f32;
                for column in 0..params.k as usize {
                    let weight_index =
                        (weight_batch * params.n as usize + out) * params.k as usize + column;
                    let input_index =
                        (batch * params.m as usize + row) * params.k as usize + column;
                    sum += f32::from(bf16::from_bits(weight[weight_index])) * input[input_index];
                }
                output[(batch * params.m as usize + row) * params.n as usize + out] = sum;
            }
        }
    }
    output
}

#[test]
fn rows1_to_16_match_row_gemv_bitwise_at_production_k() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let n = 257u32;
    let k = 5_120u32;
    let weights = bf16_values((n * k) as usize, 11);
    let weight = buffer_from_bf16(&device, &weights);

    for m in 1..=16u32 {
        let params = DenseMmBf16F32Params {
            m,
            n,
            k,
            src0_batch: 1,
            src1_batch: 1,
        };
        let inputs = f32_values((m * k) as usize, m as usize * 101);
        let input = buffer_from_f32(&device, &inputs);
        let row_output = output_buffer(&device, (m * n) as usize);
        let tiled4_output = output_buffer(&device, (m * n) as usize);
        execute(
            &device,
            &mut registry,
            &weight,
            &input,
            &row_output,
            &params,
            false,
        );
        execute(
            &device,
            &mut registry,
            &weight,
            &input,
            &tiled4_output,
            &params,
            true,
        );
        let identity = registry
            .pipeline_identity("hf2q_dense_gemv_bf16_f32_r1_4")
            .expect("tiled4 pipeline identity");
        assert_eq!(identity.kernel_name, "hf2q_dense_gemv_bf16_f32_r1_4");
        let row = row_output.as_slice::<f32>().expect("row output");
        let tiled4 = tiled4_output.as_slice::<f32>().expect("tiled4 output");
        assert!(
            row.iter().any(|value| value.abs() > 1e-6),
            "reference output must be non-vacuous"
        );
        assert!(
            row.iter()
                .zip(tiled4.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "M={m} tiled4 output diverged from the ordinary row kernel"
        );
    }
}

#[test]
fn odd_n_m5_and_gqa_broadcast_match_cpu_and_preserve_guard() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let params = DenseMmBf16F32Params {
        m: 5,
        n: 5,
        k: 516,
        src0_batch: 2,
        src1_batch: 8,
    };
    let weights = bf16_values((params.src0_batch * params.n * params.k) as usize, 7);
    let inputs = f32_values((params.src1_batch * params.m * params.k) as usize, 19);
    let expected = cpu_reference(&weights, &inputs, &params);
    let weight = buffer_from_bf16(&device, &weights);
    let input = buffer_from_f32(&device, &inputs);
    let output_elements = (params.src1_batch * params.m * params.n) as usize;
    let guard = 8usize;
    let mut output = output_buffer(&device, output_elements + guard);
    output
        .as_mut_slice::<f32>()
        .expect("map guarded output")
        .fill(f32::from_bits(0x7f00_00a5));
    execute(
        &device,
        &mut registry,
        &weight,
        &input,
        &output,
        &params,
        true,
    );
    let actual = output.as_slice::<f32>().expect("read output");
    let max_abs = actual[..output_elements]
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-5, "CPU max_abs={max_abs}");
    assert!(actual[..output_elements]
        .iter()
        .any(|value| value.abs() > 1e-6));
    assert!(actual[output_elements..]
        .iter()
        .all(|value| value.to_bits() == 0x7f00_00a5));

    let row_output = output_buffer(&device, output_elements);
    execute(
        &device,
        &mut registry,
        &weight,
        &input,
        &row_output,
        &params,
        false,
    );
    let row_actual = row_output.as_slice::<f32>().expect("read row output");
    let row_max_abs = row_actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(row_max_abs < 2e-5, "row CPU max_abs={row_max_abs}");
}

#[test]
fn invalid_contracts_fail_before_encoding() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let valid = DenseMmBf16F32Params {
        m: 4,
        n: 8,
        k: 512,
        src0_batch: 1,
        src1_batch: 1,
    };
    let weight = buffer_from_bf16(&device, &bf16_values((valid.n * valid.k) as usize, 3));
    let input = buffer_from_f32(&device, &f32_values((valid.m * valid.k) as usize, 5));
    let output = output_buffer(&device, (valid.m * valid.n) as usize);

    let mut encoder = device.command_encoder().expect("command encoder");
    let zero_rows = DenseMmBf16F32Params { m: 0, ..valid };
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &input,
        &output,
        &zero_rows,
    )
    .is_err());
    let unaligned_k = DenseMmBf16F32Params { k: 511, ..valid };
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &input,
        &output,
        &unaligned_k,
    )
    .is_err());

    let wrong_input = device
        .alloc_buffer(
            input.data_byte_len(),
            DType::BF16,
            vec![input.data_byte_len() / 2],
        )
        .expect("wrong dtype input");
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &wrong_input,
        &output,
        &valid,
    )
    .is_err());

    let short_weight = weight.slice_view(0, (valid.n * valid.k - 1) as usize);
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &short_weight,
        &input,
        &output,
        &valid,
    )
    .is_err());

    let misaligned_weight = weight.slice_view(2, (valid.n * valid.k - 1) as usize);
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &misaligned_weight,
        &input,
        &output,
        &valid,
    )
    .is_err());

    let invalid_ratio = DenseMmBf16F32Params {
        src1_batch: i16::MAX as u32 + 1,
        ..valid
    };
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &input,
        &output,
        &invalid_ratio,
    )
    .is_err());

    let oversized_batch = DenseMmBf16F32Params {
        src0_batch: i32::MAX as u32 + 1,
        src1_batch: i32::MAX as u32 + 1,
        ..valid
    };
    assert!(dense_gemv_bf16_f32_tiled4(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &input,
        &output,
        &oversized_batch,
    )
    .is_err());
}
