use half::{bf16, f16};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use mlx_native::ops::{
    ActivationPrecision, dflash2_attention_pack, dflash2_dynamic_conv,
    elementwise, embedding_dense, embedding_kquant, embedding_q2_k,
    embedding_q4_0, embedding_q4_k, embedding_q5_0, embedding_q8_0, rms_norm,
};

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device.alloc_buffer(values.len() * 4, DType::F32, shape).unwrap();
    buffer.as_mut_slice::<f32>().unwrap().copy_from_slice(values);
    buffer
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn assert_boundary(actual: &[f32], full: &[f32]) {
    assert_eq!(actual.len(), full.len());
    for (index, (&actual, &full)) in actual.iter().zip(full).enumerate() {
        if full.is_nan() {
            assert!(actual.is_nan(), "element {index}");
            assert_eq!(actual.to_bits() & 0xffff, 0, "element {index}");
        } else {
            assert_eq!(actual.to_bits(), rounded(full).to_bits(), "element {index}: {full}");
        }
    }
}

#[test]
fn weighted_norm_rounds_only_the_completed_f32_result() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    // Scalar fallback, vectorized small rows, per-head rows, hidden/final rows.
    for dim in [1, 2, 3, 4, 128, 256, 2048] {
        for exact_ties in [true, false] {
            let input_values: Vec<f32> = (0..2 * dim).map(|i| {
                if exact_ties { if i % 2 == 0 { 1.0 } else { -1.0 } }
                else { (i as f32 * 0.017).sin() + 0.0031 }
            }).collect();
            let weights: Vec<f32> = (0..dim).map(|i| {
                if exact_ties { f32::from_bits(0x3f80_8000 + ((i % 2) as u32 * 0x1_0000)) }
                else { 0.83 + i as f32 * 0.00013 }
            }).collect();
            let input = f32_buffer(&device, &input_values, vec![2, dim]);
            let weight = f32_buffer(&device, &weights, vec![dim]);
            let params = f32_buffer(&device, &[if exact_ties { 0.0 } else { 1e-6 }, dim as f32], vec![2]);
            let full = device.alloc_buffer(2 * dim * 4, DType::F32, vec![2, dim]).unwrap();
            let boundary = device.alloc_buffer(2 * dim * 4, DType::F32, vec![2, dim]).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            rms_norm::dispatch_rms_norm(
                &mut encoder, &mut registry, device.metal_device(), &input, &weight,
                &full, &params, 2, dim as u32,
            ).unwrap();
            rms_norm::dispatch_rms_norm_with_precision(
                &mut encoder, &mut registry, device.metal_device(), &input, &weight,
                &boundary, &params, 2, dim as u32, ActivationPrecision::Bf16,
            ).unwrap();
            encoder.commit_and_wait().unwrap();
            assert_boundary(boundary.as_slice::<f32>().unwrap(), full.as_slice::<f32>().unwrap());
            if exact_ties {
                for (i, &value) in full.as_slice::<f32>().unwrap().iter().enumerate() {
                    assert_eq!(value.to_bits(), (input_values[i] * weights[i % dim]).to_bits());
                }
            }
        }
    }
}

#[test]
fn dynamic_convolution_rounds_after_both_taps_and_residual() {
    use dflash2_dynamic_conv::{DFlash2ConvolutionStage, DFlash2DynamicConvParams};
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let params = DFlash2DynamicConvParams { lanes: 2, rows_per_lane: 2, hidden: 4, groups: 4, stage: 0 };
    let input_values = [
        -1.0, 1.0078125, -1.0078125, 1.0078125,
        1.0078125, -1.0078125, 1.0078125, -1.0078125,
        -1.0, -1.0078125, 1.0078125, 1.0078125,
        1.0078125, 1.0078125, -1.0078125, -1.0078125,
    ];
    let residual_values = [0.0, 0.0, 0.0, -1.015625, 0.0, 1.5, -1.5, 1.015625,
        0.0, 0.0, 0.0, -1.015625, 0.0, -1.5, 1.5, 1.015625];
    let base_values = [1.0, 1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.0];
    let mut dynamic_values = vec![0.0; 4 * 16];
    for row in 0..4 {
        for stage in 0..2 {
            dynamic_values[row * 16 + stage * 8] = 0.0078125;
            dynamic_values[row * 16 + stage * 8 + 3] = 0.0078125;
            dynamic_values[row * 16 + stage * 8 + 4] = 0.015625;
        }
    }
    let input = f32_buffer(&device, &input_values, vec![4, 4]);
    let dynamic = f32_buffer(&device, &dynamic_values, vec![4, 16]);
    let residual = f32_buffer(&device, &residual_values, vec![4, 4]);
    let mut base = device.alloc_buffer(32, DType::BF16, vec![2, 2, 4]).unwrap();
    for (dst, value) in base.as_mut_slice::<u16>().unwrap().iter_mut().zip(base_values) {
        *dst = bf16::from_f32(value).to_bits();
    }
    for stage in [DFlash2ConvolutionStage::Prepare, DFlash2ConvolutionStage::Residual] {
        let full = device.alloc_buffer(64, DType::F32, vec![4, 4]).unwrap();
        let boundary = device.alloc_buffer(64, DType::F32, vec![4, 4]).unwrap();
        let mut encoder = device.command_encoder().unwrap();
        dflash2_dynamic_conv::dispatch_dflash2_dynamic_conv_f32(
            &mut encoder, &mut registry, &device, &input, &dynamic, &base, Some(&residual),
            &full, params, stage,
        ).unwrap();
        dflash2_dynamic_conv::dispatch_dflash2_dynamic_conv_f32_with_precision(
            &mut encoder, &mut registry, &device, &input, &dynamic, &base, Some(&residual),
            &boundary, params, stage, ActivationPrecision::Bf16,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        let actual = boundary.as_slice::<f32>().unwrap();
        assert_boundary(actual, full.as_slice::<f32>().unwrap());
        for row in 0..4 {
            for channel in 0..4 {
                let i = row * 4 + channel;
                let coefficient = if channel == 0 || channel == 3 { 1.0078125 } else { 1.5 };
                let first = input_values[i] * coefficient;
                let second = if row % 2 == 1 && channel == 0 { input_values[i - 4] * 1.015625 } else { 0.0 };
                let residual = if stage == DFlash2ConvolutionStage::Residual { residual_values[i] } else { 0.0 };
                assert_eq!(actual[i].to_bits(), rounded(first + second + residual).to_bits(), "row {row} channel {channel}");
            }
        }
        // First-tap rounding would erase the two-tap cancellation witness.
        assert_eq!(actual[4], 1.0 / 16384.0);
        if stage == DFlash2ConvolutionStage::Residual {
            // Pre-residual rounding would produce zero instead.
            assert_eq!(actual[3], 1.0 / 16384.0);
        }
    }
}

#[test]
fn scalar_boundary_uses_rne_half_ties_and_preserves_special_values() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let values = [f32::from_bits(0x3f80_8000), f32::from_bits(0x3f81_8000),
        -f32::from_bits(0x3f80_8000), -f32::from_bits(0x3f81_8000),
        0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let input = f32_buffer(&device, &values, vec![values.len()]);
    let mut encoder = device.command_encoder().unwrap();
    elementwise::scalar_mul_f32_with_precision(
        &mut encoder, &mut registry, device.metal_device(), &input, &input,
        values.len(), 1.0, ActivationPrecision::Bf16,
    ).unwrap();
    encoder.commit_and_wait().unwrap();
    assert_boundary(input.as_slice::<f32>().unwrap(), &values);
}

#[test]
fn dense_entry_last_writer_handles_scale_and_non_anchor_masks() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let values = [1.00390625, 1.01171875, -1.00390625, -1.01171875,
        0.3331, -0.7741, 2.019, -2.019];
    let ids_values = [0u32, 1, 0, 1, 0, 1];
    let mut ids = device.alloc_buffer(24, DType::U32, vec![6]).unwrap();
    ids.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids_values);
    let mut mask = device.alloc_buffer(8, DType::BF16, vec![4]).unwrap();
    let mask_values = [0.3125, -0.625, 1.75, -2.5];
    for (dst, value) in mask.as_mut_slice::<u16>().unwrap().iter_mut().zip(mask_values) {
        *dst = bf16::from_f32(value).to_bits();
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let element_bytes = if dtype == DType::F32 { 4 } else { 2 };
        let mut weight = device.alloc_buffer(values.len() * element_bytes, dtype, vec![2, 4]).unwrap();
        match dtype {
            DType::F32 => weight.as_mut_slice::<f32>().unwrap().copy_from_slice(&values),
            DType::F16 => {
                for (dst, &value) in weight.as_mut_slice::<u16>().unwrap().iter_mut().zip(&values) {
                    *dst = f16::from_f32(value).to_bits();
                }
            }
            DType::BF16 => {
                for (dst, &value) in weight.as_mut_slice::<u16>().unwrap().iter_mut().zip(&values) {
                    *dst = bf16::from_f32(value).to_bits();
                }
            }
            _ => unreachable!(),
        }
        for scale in [1.0, 1.5] {
            for masked in [false, true] {
                let output = device.alloc_buffer(96, DType::F32, vec![6, 4]).unwrap();
                let mut encoder = device.command_encoder().unwrap();
                embedding_dense::embedding_gather_dense_with_precision(
                    &mut encoder, &mut registry, &device, &weight, &ids, &output,
                    &embedding_dense::EmbeddingDenseParams { vocab_size: 2, embed_dim: 4, n_tokens: 6 },
                    if scale == 1.0 { ActivationPrecision::Bf16 } else { ActivationPrecision::F32 },
                ).unwrap();
                if scale != 1.0 {
                    encoder.memory_barrier();
                    elementwise::scalar_mul_f32_with_precision(
                        &mut encoder, &mut registry, device.metal_device(), &output, &output,
                        24, scale, ActivationPrecision::Bf16,
                    ).unwrap();
                }
                if masked {
                    encoder.memory_barrier();
                    dflash2_attention_pack::dispatch_dflash2_mask_rows_f32(
                        &mut encoder, &mut registry, &device, &mask, &output,
                        dflash2_attention_pack::DFlash2MaskRowsParams { lanes: 2, rows: 3, hidden: 4 },
                    ).unwrap();
                }
                encoder.commit_and_wait().unwrap();
                for (row, &token) in ids_values.iter().enumerate() {
                    for channel in 0..4 {
                        let value = values[token as usize * 4 + channel];
                        let stored = match dtype {
                            DType::F32 => value,
                            DType::F16 => f16::from_f32(value).to_f32(),
                            DType::BF16 => rounded(value),
                            _ => unreachable!(),
                        };
                        let expected = if masked && row % 3 != 0 { mask_values[channel] } else { rounded(stored * scale) };
                        assert_eq!(
                            output.as_slice::<f32>().unwrap()[row * 4 + channel].to_bits(),
                            expected.to_bits(),
                            "dtype={dtype:?} scale={scale} masked={masked} row={row} channel={channel}",
                        );
                    }
                }
                if dtype == DType::F32 && scale != 1.0 {
                    assert_ne!(rounded(values[0] * scale), rounded(rounded(values[0]) * scale));
                }
            }
        }
    }
}

#[test]
fn every_quantized_gather_rounds_its_existing_dequantization_result() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut ids = device.alloc_buffer(16, DType::U32, vec![4]).unwrap();
    ids.as_mut_slice::<u32>().unwrap().copy_from_slice(&[2, 0, 1, 2]);
    macro_rules! check_format {
        ($module:ident, $params:ident, $old:ident, $new:ident, $kernel:literal, $block:expr, $qk:expr, $scale:expr, $minimum:expr) => {{
            let n_blocks = 3 * 512 / $qk;
            let mut bytes: Vec<u8> = (0..n_blocks * $block).map(|i| (i.wrapping_mul(73).wrapping_add(19) % 256) as u8).collect();
            for block in bytes.chunks_exact_mut($block) {
                block[$scale..$scale + 2].copy_from_slice(&f16::from_f32(1.00390625).to_le_bytes());
                if let Some(offset) = $minimum {
                    block[offset..offset + 2].copy_from_slice(&f16::from_f32(0.1015625).to_le_bytes());
                }
            }
            let mut weight = device.alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()]).unwrap();
            weight.as_mut_slice::<u8>().unwrap().copy_from_slice(&bytes);
            let full = device.alloc_buffer(4 * 512 * 4, DType::F32, vec![4, 512]).unwrap();
            let boundary = device.alloc_buffer(4 * 512 * 4, DType::F32, vec![4, 512]).unwrap();
            let params = $module::$params { vocab_size: 3, embed_dim: 512, n_tokens: 4 };
            let mut encoder = device.command_encoder().unwrap();
            $module::$old(&mut encoder, &mut registry, &device, &weight, &ids, &full, &params).unwrap();
            $module::$new(&mut encoder, &mut registry, &device, &weight, &ids, &boundary, &params, ActivationPrecision::Bf16).unwrap();
            encoder.commit_and_wait().unwrap();
            let full = full.as_slice::<f32>().unwrap();
            let boundary = boundary.as_slice::<f32>().unwrap();
            assert_boundary(boundary, full);
            assert!(full.iter().zip(boundary).any(|(a, b)| a != b), "{} needs a rounding witness", $kernel);
            assert!(full.iter().any(|v| *v < 0.0), "{} needs negative values", $kernel);
        }};
    }
    check_format!(embedding_q4_0, EmbeddingQ4_0Params, embedding_gather_q4_0, embedding_gather_q4_0_with_precision, "embedding_gather_q4_0_f32", 18usize, 32usize, 0usize, None::<usize>);
    check_format!(embedding_q5_0, EmbeddingQ5_0Params, embedding_gather_q5_0, embedding_gather_q5_0_with_precision, "embedding_gather_q5_0_f32", 22usize, 32usize, 0usize, None::<usize>);
    check_format!(embedding_q2_k, EmbeddingQ2KParams, embedding_gather_q2_k, embedding_gather_q2_k_with_precision, "embedding_gather_q2_k_f32", 84usize, 256usize, 80usize, Some(82usize));
    check_format!(embedding_q4_k, EmbeddingQ4KParams, embedding_gather_q4_k, embedding_gather_q4_k_with_precision, "embedding_gather_q4_k_f32", 144usize, 256usize, 0usize, Some(2usize));
    check_format!(embedding_kquant, EmbeddingQ5KParams, embedding_gather_q5_k, embedding_gather_q5_k_with_precision, "embedding_gather_q5_k_f32", 176usize, 256usize, 0usize, Some(2usize));
    check_format!(embedding_kquant, EmbeddingQ6KParams, embedding_gather_q6_k, embedding_gather_q6_k_with_precision, "embedding_gather_q6_k_f32", 210usize, 256usize, 208usize, None::<usize>);
    check_format!(embedding_q8_0, EmbeddingQ8_0Params, embedding_gather_q8_0, embedding_gather_q8_0_with_precision, "embedding_gather_q8_0_f32", 34usize, 32usize, 0usize, None::<usize>);
}
