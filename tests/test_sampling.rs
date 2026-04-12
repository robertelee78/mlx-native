//! Tests for the softmax_sample GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::softmax_sample;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    softmax_sample::register(&mut registry);
    (device, registry)
}

/// Run a single softmax_sample dispatch and return (token_id, logprob).
fn run_sample(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    logits: &[f32],
    temperature: f32,
    random_val: f32,
) -> (u32, f32) {
    let n = logits.len() as u32;

    let mut logits_buf = device
        .alloc_buffer(n as usize * 4, DType::F32, vec![n as usize])
        .expect("alloc logits");
    logits_buf
        .as_mut_slice::<f32>()
        .expect("write logits")
        .copy_from_slice(logits);

    let scratch_buf = device
        .alloc_buffer(n as usize * 4, DType::F32, vec![n as usize])
        .expect("alloc scratch");

    let out_token = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc out_token");
    let out_logprob = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc out_logprob");

    // params: [n_elements as f32, temperature, random_val]
    let mut params_buf = device
        .alloc_buffer(12, DType::F32, vec![3])
        .expect("alloc params");
    {
        let s = params_buf.as_mut_slice::<f32>().expect("write params");
        s[0] = n as f32;
        s[1] = temperature;
        s[2] = random_val;
    }

    let mut encoder = device.command_encoder().expect("encoder");
    softmax_sample::dispatch_softmax_sample_f32(
        &mut encoder,
        registry,
        device.metal_device(),
        &logits_buf,
        &scratch_buf,
        &out_token,
        &out_logprob,
        &params_buf,
        n,
        temperature,
        random_val,
    )
    .expect("dispatch_softmax_sample_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let token_id = out_token.as_slice::<u32>().expect("read token")[0];
    let logprob = out_logprob.as_slice::<f32>().expect("read logprob")[0];

    (token_id, logprob)
}

#[test]
fn test_softmax_sample_deterministic() {
    // With a very skewed distribution, the sample should always pick the max
    let (device, mut registry) = setup();

    let mut logits = vec![0.0f32; 100];
    logits[42] = 100.0; // Overwhelmingly likely

    // Note: random_val=0.0 picks the first CDF entry (token 0) which is
    // correct kernel behavior. We test with values > 0 where the overwhelming
    // logit at idx 42 dominates the CDF.
    for r in [0.01, 0.1, 0.5, 0.9, 0.99] {
        let (token_id, _logprob) = run_sample(&device, &mut registry, &logits, 1.0, r);
        assert_eq!(
            token_id, 42,
            "With overwhelming logit at idx 42, T=1.0 random_val={}: got token {}",
            r, token_id
        );
    }
}

#[test]
fn test_softmax_sample_uniform_distribution() {
    // With uniform logits and T=1.0, run many samples and verify all indices
    // appear with roughly equal frequency.
    let (device, mut registry) = setup();

    let n = 10;
    let logits = vec![0.0f32; n]; // Uniform
    let n_samples = 1000;

    let mut counts = vec![0u32; n];

    for i in 0..n_samples {
        // Use evenly spaced random values to cover the CDF
        let r = (i as f32 + 0.5) / n_samples as f32;
        let (token_id, _) = run_sample(&device, &mut registry, &logits, 1.0, r);
        assert!(
            (token_id as usize) < n,
            "Token id {} out of range [0, {})",
            token_id,
            n
        );
        counts[token_id as usize] += 1;
    }

    // With uniform logits, each token should appear ~100 times out of 1000.
    // Allow generous tolerance: each should appear at least 50 and at most 200.
    let expected_per_token = n_samples as f32 / n as f32;
    for (idx, &count) in counts.iter().enumerate() {
        assert!(
            count >= 50 && count <= 200,
            "Uniform distribution: token {} appeared {} times (expected ~{:.0}), counts={:?}",
            idx,
            count,
            expected_per_token,
            counts
        );
    }
}

#[test]
fn test_softmax_sample_temperature_effect() {
    // Higher temperature should flatten the distribution;
    // lower temperature should sharpen it.
    let (device, mut registry) = setup();

    let mut logits = vec![0.0f32; 5];
    logits[0] = 2.0; // Slightly preferred

    // At low temperature (0.1), the max should dominate
    let mut low_temp_hits = 0;
    for i in 0..100 {
        let r = (i as f32 + 0.5) / 100.0;
        let (token_id, _) = run_sample(&device, &mut registry, &logits, 0.1, r);
        if token_id == 0 {
            low_temp_hits += 1;
        }
    }

    // At high temperature (10.0), distribution should be more uniform
    let mut high_temp_hits = 0;
    for i in 0..100 {
        let r = (i as f32 + 0.5) / 100.0;
        let (token_id, _) = run_sample(&device, &mut registry, &logits, 10.0, r);
        if token_id == 0 {
            high_temp_hits += 1;
        }
    }

    assert!(
        low_temp_hits > high_temp_hits,
        "Low temperature should concentrate more: low_temp_hits={}, high_temp_hits={}",
        low_temp_hits,
        high_temp_hits
    );
}

#[test]
fn test_softmax_sample_validation_errors() {
    let (device, mut registry) = setup();

    let logits_buf = device
        .alloc_buffer(40, DType::F32, vec![10])
        .expect("buf");
    let scratch_buf = device
        .alloc_buffer(40, DType::F32, vec![10])
        .expect("buf");
    let out_token = device.alloc_buffer(4, DType::U32, vec![1]).expect("buf");
    let out_logprob = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");
    let params_buf = device.alloc_buffer(12, DType::F32, vec![3]).expect("buf");

    let mut encoder = device.command_encoder().expect("encoder");

    // Zero elements
    let r = softmax_sample::dispatch_softmax_sample_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &logits_buf,
        &scratch_buf,
        &out_token,
        &out_logprob,
        &params_buf,
        0,
        1.0,
        0.5,
    );
    assert!(r.is_err(), "Should error on n_elements=0");

    // Negative temperature
    let r = softmax_sample::dispatch_softmax_sample_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &logits_buf,
        &scratch_buf,
        &out_token,
        &out_logprob,
        &params_buf,
        10,
        -1.0,
        0.5,
    );
    assert!(r.is_err(), "Should error on negative temperature");

    // random_val out of range
    let r = softmax_sample::dispatch_softmax_sample_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &logits_buf,
        &scratch_buf,
        &out_token,
        &out_logprob,
        &params_buf,
        10,
        1.0,
        1.0, // should be < 1.0
    );
    assert!(r.is_err(), "Should error on random_val=1.0");
}
