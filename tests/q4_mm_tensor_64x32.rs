#![cfg(mlx_native_has_metal_tensor_artifact)]

use half::f16;
use mlx_native::{
    dispatch_mm_for_test, dispatch_mm_q4_0_tensor_64x32_for_test, DType, GgmlQuantizedMatmulParams,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

#[derive(Clone, Copy)]
enum InputPattern {
    Canonical,
    Adversarial,
}

impl InputPattern {
    fn label(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Adversarial => "adversarial",
        }
    }

    fn value(self, index: usize, salt: usize) -> f32 {
        match self {
            Self::Canonical => {
                let mixed = index
                    .wrapping_mul(6364136223846793005usize)
                    .wrapping_add(salt.wrapping_mul(1442695040888963407usize));
                ((mixed >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 0.25
            }
            Self::Adversarial => {
                const VALUES: [f32; 16] = [
                    0.0,
                    -0.0,
                    f32::from_bits(1),
                    -f32::from_bits(1),
                    1.0 / 3.0,
                    -1.0 / 7.0,
                    1.000_976_6,
                    -0.999_511_7,
                    0.031_25,
                    -0.062_5,
                    3.141_592_7,
                    -2.718_281_7,
                    1.0e-6,
                    -1.0e-6,
                    127.0 / 257.0,
                    -63.0 / 131.0,
                ];
                let base = VALUES[(index.wrapping_mul(13).wrapping_add(salt)) % VALUES.len()];
                let fine =
                    ((index.wrapping_mul(29).wrapping_add(salt * 7) % 37) as f32 - 18.0) / 16_381.0;
                base + fine
            }
        }
    }
}

fn pack_q4_0(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let mut bytes = Vec::with_capacity(n * (k / 32) * 18);
    let mut block = [0.0f32; 32];
    for row in 0..n {
        for block_index in 0..k / 32 {
            for (index, value) in block.iter_mut().enumerate() {
                *value = InputPattern::Canonical.value(row * k + block_index * 32 + index, 7);
            }
            let amax = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
            let scale = amax / 7.0;
            let inverse_scale = if scale == 0.0 { 0.0 } else { scale.recip() };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for index in 0..16 {
                let low = (block[index] * inverse_scale + 8.0)
                    .round()
                    .clamp(0.0, 15.0) as u8;
                let high = (block[index + 16] * inverse_scale + 8.0)
                    .round()
                    .clamp(0.0, 15.0) as u8;
                bytes.push(low | (high << 4));
            }
        }
    }
    bytes
}

fn alloc_weight(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("allocate Q4_0 weights");
    buffer
        .as_mut_slice::<u8>()
        .expect("map Q4_0 weights")
        .copy_from_slice(bytes);
    buffer
}

fn alloc_input(device: &MlxDevice, m: usize, k: usize, pattern: InputPattern) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(m * k * size_of::<f32>(), DType::F32, vec![m, k])
        .expect("allocate F32 input");
    for (index, value) in buffer
        .as_mut_slice::<f32>()
        .expect("map F32 input")
        .iter_mut()
        .enumerate()
    {
        *value = pattern.value(index, 23);
    }
    buffer
}

fn poison_output(device: &MlxDevice, m: usize, n: usize, salt: u32) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(m * n * size_of::<f32>(), DType::F32, vec![m, n])
        .expect("allocate F32 output");
    for (index, value) in buffer
        .as_mut_slice::<f32>()
        .expect("map F32 output")
        .iter_mut()
        .enumerate()
    {
        *value = f32::from_bits(0x7fc0_0000 | ((index as u32 + salt) & 0x003f_ffff));
    }
    buffer
}

fn assert_bitwise_eq(label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: output length");
    if let Some((index, (left, right))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (left, right))| left.to_bits() != right.to_bits())
    {
        panic!(
            "{label}: first mismatch at {index}: {left:?} ({:#010x}) != {right:?} ({:#010x})",
            left.to_bits(),
            right.to_bits()
        );
    }
}

fn check_shape(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    m: usize,
    n: usize,
    k: usize,
    pattern: InputPattern,
) {
    let original_weight_bytes = pack_q4_0(n, k);
    let weight = alloc_weight(device, &original_weight_bytes);
    let input = alloc_input(device, m, k, pattern);
    let control = poison_output(device, m, n, 0x101);
    let candidate = poison_output(device, m, n, 0x301);
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q4_0,
    };

    let mut encoder = device.command_encoder().expect("create control encoder");
    dispatch_mm_for_test(
        &mut encoder,
        registry,
        device,
        &input,
        &weight,
        &control,
        &params,
    )
    .expect("dispatch current V2 control");
    encoder
        .commit_and_wait()
        .expect("complete current V2 control");

    let mut encoder = device.command_encoder().expect("create candidate encoder");
    dispatch_mm_q4_0_tensor_64x32_for_test(
        &mut encoder,
        registry,
        device,
        &input,
        &weight,
        &candidate,
        &params,
    )
    .expect("dispatch 64x32 candidate");
    encoder.commit_and_wait().expect("complete 64x32 candidate");

    let control = control.as_slice::<f32>().expect("read control output");
    let candidate = candidate.as_slice::<f32>().expect("read candidate output");
    assert!(control.iter().all(|value| value.is_finite()));
    assert!(control.iter().any(|value| value.to_bits() != 0));
    assert_bitwise_eq(
        &format!("{} M={m} N={n} K={k}", pattern.label()),
        control,
        candidate,
    );
    assert_eq!(
        weight.as_slice::<u8>().expect("read Q4_0 weights"),
        original_weight_bytes,
        "both routes must leave native Q4_0 bytes unchanged"
    );
}

#[test]
fn q4_64x32_is_bitwise_current_v2_at_exact_m32_projection_shapes() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for pattern in [InputPattern::Canonical, InputPattern::Adversarial] {
        for (n, k) in [(768, 768), (3072, 768), (768, 3072)] {
            check_shape(&device, &mut registry, 32, n, k, pattern);
        }
    }
}

#[test]
fn q4_64x32_is_bitwise_current_v2_at_partial_tiles() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for (m, n, k) in [(9, 769, 800), (33, 129, 96), (129, 65, 160)] {
        check_shape(&device, &mut registry, m, n, k, InputPattern::Adversarial);
    }
}

#[test]
fn q4_64x32_is_bitwise_at_qualification_boundaries() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for &m in &[
        9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
        2049, 4095, 4096, 4097,
    ] {
        check_shape(&device, &mut registry, m, 65, 96, InputPattern::Adversarial);
    }
    for &n in &[
        1, 63, 64, 65, 127, 128, 129, 767, 768, 769, 3071, 3072, 3073,
    ] {
        check_shape(&device, &mut registry, 33, n, 96, InputPattern::Adversarial);
    }
    for &k in &[32, 64, 96, 128, 160, 256, 768, 800, 3072, 17408] {
        check_shape(&device, &mut registry, 33, 65, k, InputPattern::Adversarial);
    }
}
