//! DeepSeek-V4 official activation-simulation parity gates.

#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_hadamard_mxfp4_bf16, dispatch_deepseek_mxfp8_fake_quant_bf16,
    DeepSeekMxfp8Params, DEEPSEEK_INDEX_WIDTH, DEEPSEEK_MAIN_QUANTIZED_WIDTH, DEEPSEEK_MAIN_WIDTH,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn bf16_buffer(device: &MlxDevice, values: &[bf16], shape: Vec<usize>) -> MlxBuffer {
    let buffer = device
        .alloc_buffer(values.len() * 2, DType::BF16, shape)
        .expect("allocate BF16 buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents_ptr() as *mut bf16,
            values.len(),
        );
    }
    buffer
}

fn read_bf16(buffer: &MlxBuffer, len: usize) -> Vec<bf16> {
    unsafe { std::slice::from_raw_parts(buffer.contents_ptr() as *const bf16, len).to_vec() }
}

fn pow2_ceil_positive(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((exponent + 127) as u32) << 23)
}

fn quantize_e4m3fn(value: f32) -> f32 {
    let magnitude = value.abs().min(448.0);
    let step = if magnitude < 0.015625 {
        0.001953125
    } else {
        let exponent = ((magnitude.to_bits() >> 23) & 0xff) as i32 - 127;
        f32::from_bits(((exponent - 3 + 127) as u32) << 23)
    };
    value.signum()
        * (magnitude / step)
            .round_ties_even()
            .mul_add(step, 0.0)
            .min(448.0)
}

fn mxfp8_reference(values: &mut [bf16], row_width: usize, quantized_width: usize) {
    for row in values.chunks_exact_mut(row_width) {
        for block in row[..quantized_width].chunks_exact_mut(64) {
            let maximum = block
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(1.0e-4f32, f32::max);
            let scale = pow2_ceil_positive(maximum / 448.0);
            for value in block {
                *value = bf16::from_f32(quantize_e4m3fn(value.to_f32() / scale) * scale);
            }
        }
    }
}

fn quantize_e2m1(value: f32) -> f32 {
    let x = value.abs().min(6.0);
    let rounded = if x <= 0.25 {
        0.0
    } else if x < 0.75 {
        0.5
    } else if x <= 1.25 {
        1.0
    } else if x < 1.75 {
        1.5
    } else if x <= 2.5 {
        2.0
    } else if x < 3.5 {
        3.0
    } else if x <= 5.0 {
        4.0
    } else {
        6.0
    };
    value.signum() * rounded
}

fn hadamard_mxfp4_reference(values: &[bf16]) -> Vec<bf16> {
    let mut output = values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    for row in output.chunks_exact_mut(DEEPSEEK_INDEX_WIDTH) {
        for stride in (0..7).map(|shift| 1usize << shift) {
            for base in (0..DEEPSEEK_INDEX_WIDTH).step_by(2 * stride) {
                for column in 0..stride {
                    let left = row[base + column];
                    let right = row[base + column + stride];
                    row[base + column] = left + right;
                    row[base + column + stride] = left - right;
                }
            }
        }
        for value in row.iter_mut() {
            *value *= 0.08838834764831845;
        }
        for block in row.chunks_exact_mut(32) {
            let maximum = block
                .iter()
                .map(|value| value.abs())
                .fold(7.052966104933725e-38f32, f32::max);
            let scale = pow2_ceil_positive(maximum / 6.0);
            for value in block {
                *value = quantize_e2m1(*value / scale) * scale;
            }
        }
    }
    output.into_iter().map(bf16::from_f32).collect()
}

#[test]
fn main_kv_mxfp8_matches_exact_cpu_reference_and_preserves_rope_tail() {
    let rows = 2;
    let input = (0..rows * DEEPSEEK_MAIN_WIDTH)
        .map(|index| {
            let raw = ((index * 73 + 19) % 1009) as f32 - 504.0;
            bf16::from_f32(raw * (1.0 + (index / 64) as f32 * 0.03125))
        })
        .collect::<Vec<_>>();
    let mut expected = input.clone();
    mxfp8_reference(
        &mut expected,
        DEEPSEEK_MAIN_WIDTH,
        DEEPSEEK_MAIN_QUANTIZED_WIDTH,
    );

    let device = MlxDevice::new().expect("Metal device");
    let data = bf16_buffer(&device, &input, vec![rows, DEEPSEEK_MAIN_WIDTH]);
    let mut registry = KernelRegistry::new();
    let params = DeepSeekMxfp8Params {
        rows: rows as u32,
        row_width: DEEPSEEK_MAIN_WIDTH as u32,
        quantized_width: DEEPSEEK_MAIN_QUANTIZED_WIDTH as u32,
        block_size: 64,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_deepseek_mxfp8_fake_quant_bf16(&mut encoder, &mut registry, &device, &data, &params)
        .expect("dispatch MXFP8");
    encoder.commit_and_wait().expect("MXFP8 completion");
    assert_eq!(read_bf16(&data, input.len()), expected);
    for row in 0..rows {
        let tail = row * DEEPSEEK_MAIN_WIDTH + DEEPSEEK_MAIN_QUANTIZED_WIDTH;
        assert_eq!(
            &read_bf16(&data, input.len())[tail..tail + 64],
            &input[tail..tail + 64]
        );
    }
}

#[test]
fn indexer_hadamard_mxfp4_matches_exact_cpu_reference() {
    let rows = 3;
    let input = (0..rows * DEEPSEEK_INDEX_WIDTH)
        .map(|index| bf16::from_f32(((index * 29 + 7) % 127) as f32 / 16.0 - 4.0))
        .collect::<Vec<_>>();
    let expected = hadamard_mxfp4_reference(&input);
    let device = MlxDevice::new().expect("Metal device");
    let data = bf16_buffer(&device, &input, vec![1, rows, DEEPSEEK_INDEX_WIDTH]);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_deepseek_hadamard_mxfp4_bf16(&mut encoder, &mut registry, &device, &data, rows as u32)
        .expect("dispatch Hadamard MXFP4");
    encoder
        .commit_and_wait()
        .expect("Hadamard MXFP4 completion");
    assert_eq!(read_bf16(&data, input.len()), expected);
}

#[test]
fn activation_simulation_rejects_shape_and_parameter_drift() {
    let device = MlxDevice::new().expect("Metal device");
    let data = bf16_buffer(&device, &[bf16::ZERO; 64], vec![64]);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("encoder");
    let invalid = DeepSeekMxfp8Params {
        rows: 1,
        row_width: 64,
        quantized_width: 63,
        block_size: 64,
    };
    assert!(dispatch_deepseek_mxfp8_fake_quant_bf16(
        &mut encoder,
        &mut registry,
        &device,
        &data,
        &invalid,
    )
    .is_err());
    assert!(
        dispatch_deepseek_hadamard_mxfp4_bf16(&mut encoder, &mut registry, &device, &data, 1,)
            .is_err()
    );
}
