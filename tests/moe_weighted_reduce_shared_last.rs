#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::moe_weighted_reduce::{
    dispatch_moe_weighted_reduce, dispatch_moe_weighted_reduce_shared_last,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn upload(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(
            values.len() * DType::F32.size_of(),
            DType::F32,
            vec![values.len()],
        )
        .expect("allocate upload");
    buffer
        .as_mut_slice::<f32>()
        .expect("upload slice")
        .copy_from_slice(values);
    buffer
}

#[test]
fn shared_last_matches_separate_shared_output_bitwise() {
    const TOKENS: usize = 3;
    const TOP_K: usize = 2;
    const HIDDEN: usize = 7;
    const STRIDE: usize = TOP_K + 1;

    let routed_weights = [0.75f32, 0.25, 0.4, 0.6, 0.125, 0.875];
    let mut routed_outputs = vec![0.0f32; TOKENS * TOP_K * HIDDEN];
    let mut shared_outputs = vec![0.0f32; TOKENS * HIDDEN];
    let mut combined_weights = vec![0.0f32; TOKENS * STRIDE];
    let mut combined_outputs = vec![0.0f32; TOKENS * STRIDE * HIDDEN];
    for token in 0..TOKENS {
        for route in 0..TOP_K {
            combined_weights[token * STRIDE + route] = routed_weights[token * TOP_K + route];
            for hidden in 0..HIDDEN {
                let value = (token * 100 + route * 10 + hidden) as f32 * 0.03125 - 2.0;
                routed_outputs[(token * TOP_K + route) * HIDDEN + hidden] = value;
                combined_outputs[(token * STRIDE + route) * HIDDEN + hidden] = value;
            }
        }
        combined_weights[token * STRIDE + TOP_K] = 91.0;
        for hidden in 0..HIDDEN {
            let value = (token * 17 + hidden) as f32 * -0.0625 + 1.25;
            shared_outputs[token * HIDDEN + hidden] = value;
            combined_outputs[(token * STRIDE + TOP_K) * HIDDEN + hidden] = value;
        }
    }

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let routed_weights = upload(&device, &routed_weights);
    let routed_outputs = upload(&device, &routed_outputs);
    let shared_outputs = upload(&device, &shared_outputs);
    let combined_weights = upload(&device, &combined_weights);
    let combined_outputs = upload(&device, &combined_outputs);
    let shared_logits = upload(&device, &[-2.0, 0.0, 1.5]);
    let residual_values: Vec<f32> = (0..TOKENS * HIDDEN)
        .map(|index| index as f32 * 0.0078125 - 0.25)
        .collect();
    let residual = upload(&device, &residual_values);
    let mut separate = upload(&device, &vec![0.0; TOKENS * HIDDEN]);
    let mut combined = upload(&device, &vec![0.0; TOKENS * HIDDEN]);

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_moe_weighted_reduce(
        &mut encoder,
        &mut registry,
        &device,
        &routed_weights,
        &routed_outputs,
        &shared_logits,
        &shared_outputs,
        &residual,
        &mut separate,
        TOKENS as u32,
        TOP_K as u32,
        HIDDEN as u32,
        true,
    )
    .expect("separate reduction");
    dispatch_moe_weighted_reduce_shared_last(
        &mut encoder,
        &mut registry,
        &device,
        &combined_weights,
        &combined_outputs,
        &shared_logits,
        &residual,
        &mut combined,
        TOKENS as u32,
        TOP_K as u32,
        HIDDEN as u32,
        true,
    )
    .expect("combined reduction");
    encoder.commit_and_wait().expect("commit");

    let separate_bits: Vec<u32> = separate
        .as_slice::<f32>()
        .expect("separate output")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let combined_bits: Vec<u32> = combined
        .as_slice::<f32>()
        .expect("combined output")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(combined_bits, separate_bits);
}
