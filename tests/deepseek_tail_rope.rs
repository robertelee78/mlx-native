use half::bf16;
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, dispatch_deepseek_tail_rope_f32_to_bf16,
    DeepSeekTailRopeParams,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn frequencies(dim: usize, original: usize, base: f32, factor: f32) -> Vec<f32> {
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let mut result: Vec<f32> = (0..dim / 2)
        .map(|pair| 1.0 / base.powf((2 * pair) as f32 / dim as f32))
        .collect();
    if original == 0 {
        return result;
    }
    let correction = |rotations: f32| {
        dim as f32 * (original as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let low = correction(beta_fast).floor().max(0.0) as usize;
    let high = correction(beta_slow).ceil().min((dim - 1) as f32) as usize;
    let upper = if low == high {
        high as f32 + 0.001
    } else {
        high as f32
    };
    for (pair, frequency) in result.iter_mut().enumerate() {
        let ramp = ((pair as f32 - low as f32) / (upper - low as f32)).clamp(0.0, 1.0);
        *frequency = *frequency / factor * ramp + *frequency * (1.0 - ramp);
    }
    result
}

fn cpu_rotate(values: &mut [f32], position: u32, freq: &[f32], inverse: bool) {
    let tail = values.len() - freq.len() * 2;
    for (pair, frequency) in freq.iter().enumerate() {
        let angle = position as f32 * frequency * if inverse { -1.0 } else { 1.0 };
        let (sine, cosine) = angle.sin_cos();
        let real = values[tail + pair * 2];
        let imag = values[tail + pair * 2 + 1];
        values[tail + pair * 2] = real * cosine - imag * sine;
        values[tail + pair * 2 + 1] = real * sine + imag * cosine;
    }
}

#[test]
fn tail_rope_f32_to_bf16_and_inverse_match_cpu() {
    let (batch, seq, heads, head_dim, rope_dim) = (1usize, 3usize, 2usize, 128usize, 64usize);
    let positions = [0u32, 17, 1_000_000];
    let freq = frequencies(rope_dim, 65_536, 160_000.0, 16.0);
    let values: Vec<f32> = (0..batch * seq * heads * head_dim)
        .map(|index| ((index * 37 + 11) % 251) as f32 / 64.0 - 1.5)
        .collect();
    let mut expected = values.clone();
    for sequence in 0..seq {
        for head in 0..heads {
            let start = (sequence * heads + head) * head_dim;
            cpu_rotate(
                &mut expected[start..start + head_dim],
                positions[sequence],
                &freq,
                false,
            );
        }
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::deepseek_tail_rope::register(&mut registry);
    let mut input = device
        .alloc_buffer(
            values.len() * 4,
            DType::F32,
            vec![batch, seq, heads, head_dim],
        )
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&values);
    let mut positions_buf = device
        .alloc_buffer(seq * 4, DType::U32, vec![seq])
        .expect("positions");
    positions_buf
        .as_mut_slice::<u32>()
        .expect("position slice")
        .copy_from_slice(&positions);
    let mut freq_buf = device
        .alloc_buffer(freq.len() * 4, DType::F32, vec![freq.len()])
        .expect("frequencies");
    freq_buf
        .as_mut_slice::<f32>()
        .expect("frequency slice")
        .copy_from_slice(&freq);
    let rotated = device
        .alloc_buffer(
            values.len() * 2,
            DType::BF16,
            vec![batch, seq, heads, head_dim],
        )
        .expect("rotated");
    let params = DeepSeekTailRopeParams {
        batch: batch as u32,
        seq_len: seq as u32,
        heads: heads as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        inverse: 0,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_deepseek_tail_rope_f32_to_bf16(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &positions_buf,
        &freq_buf,
        &rotated,
        &params,
    )
    .expect("forward rope");
    encoder.commit_and_wait().expect("forward completion");
    let actual = rotated.as_slice::<u16>().expect("rotated slice");
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let got = bf16::from_bits(got).to_f32();
        let want = bf16::from_f32(want).to_f32();
        assert!(
            (got - want).abs() <= 0.03125,
            "index {index}: {got} != {want}"
        );
    }

    let recovered = device
        .alloc_buffer(
            values.len() * 2,
            DType::BF16,
            vec![batch, seq, heads, head_dim],
        )
        .expect("recovered");
    let mut inverse = params;
    inverse.inverse = 1;
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_deepseek_tail_rope_bf16(
        &mut encoder,
        &mut registry,
        &device,
        &rotated,
        &positions_buf,
        &freq_buf,
        &recovered,
        &inverse,
    )
    .expect("inverse rope");
    encoder.commit_and_wait().expect("inverse completion");
    for (index, (&got, &want)) in recovered
        .as_slice::<u16>()
        .expect("recovered slice")
        .iter()
        .zip(&values)
        .enumerate()
    {
        let got = bf16::from_bits(got).to_f32();
        assert!(
            (got - want).abs() <= 0.0625,
            "index {index}: {got} != {want}"
        );
    }
}

#[test]
fn tail_rope_rejects_nonfinite_frequencies_before_encoding() {
    let device = MlxDevice::new().expect("Metal device");
    let input = device
        .alloc_buffer(128 * 4, DType::F32, vec![1, 1, 1, 128])
        .expect("input");
    let output = device
        .alloc_buffer(128 * 2, DType::BF16, vec![1, 1, 1, 128])
        .expect("output");
    let positions = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("positions");
    let mut freq = device
        .alloc_buffer(32 * 4, DType::F32, vec![32])
        .expect("freq");
    freq.as_mut_slice::<f32>().expect("freq slice")[0] = f32::NAN;
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("encoder");
    assert!(dispatch_deepseek_tail_rope_f32_to_bf16(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &positions,
        &freq,
        &output,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: 1,
            heads: 1,
            head_dim: 128,
            rope_dim: 64,
            inverse: 0,
        },
    )
    .is_err());
}
