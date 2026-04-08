//! Temperature-scaled softmax + categorical sample, entirely on GPU.
//!
//! For stochastic (temperature > 0) decoding this replaces the 1MB
//! GPU→CPU logits readback with an 8-byte readback: the sampled token index
//! and its log-probability.
//!
//! The kernel runs three parallel passes (max, exp-sum, normalize) using
//! threadgroup reductions, then a sequential CDF scan by thread 0 to draw the
//! sample.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the softmax_sample kernel (embedded at compile time).
pub static SOFTMAX_SAMPLE_SHADER_SOURCE: &str =
    include_str!("../shaders/softmax_sample.metal");

/// Register the softmax_sample shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("softmax_sample_f32", SOFTMAX_SAMPLE_SHADER_SOURCE);
}

/// Dispatch a temperature-scaled softmax + categorical sample on the GPU.
///
/// Computes `softmax(logits / temperature)` entirely on the GPU, then samples
/// one token index using the provided uniform random value.  Only 8 bytes
/// (token_id u32 + logprob f32) are transferred back to the CPU.
///
/// # Arguments
///
/// * `encoder`      - Command encoder to record the dispatch into.
/// * `registry`     - Kernel registry (must have `softmax_sample_f32` registered).
/// * `device`       - Metal device for pipeline compilation.
/// * `logits`       - Input logits buffer `[n_elements]` (f32).
/// * `scratch`      - Scratch buffer `[n_elements]` (f32) used for intermediate
///                    probability values.  May be a transient allocation; must
///                    not alias `logits`.
/// * `out_token`    - Output buffer `[1]` (u32) — sampled token index.
/// * `out_logprob`  - Output buffer `[1]` (f32) — log-probability of the
///                    sampled token.
/// * `params_buf`   - Params buffer `[3]` (f32) containing:
///                    `[n_elements as f32, temperature, random_val]`
/// * `n_elements`   - Vocabulary size (number of logits).
/// * `temperature`  - Sampling temperature (must be > 0.0).
/// * `random_val`   - Uniform random value in `[0, 1)` for categorical sample.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `n_elements` is 0.
/// - `temperature` is not positive.
/// - `random_val` is not in `[0, 1)`.
/// - Buffer sizes are inconsistent.
pub fn dispatch_softmax_sample_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    logits: &MlxBuffer,
    scratch: &MlxBuffer,
    out_token: &MlxBuffer,
    out_logprob: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_elements: u32,
    temperature: f32,
    random_val: f32,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "softmax_sample_f32: n_elements must be > 0".into(),
        ));
    }
    if temperature <= 0.0 {
        return Err(MlxError::InvalidArgument(format!(
            "softmax_sample_f32: temperature must be > 0, got {}",
            temperature
        )));
    }
    if !(0.0..1.0).contains(&random_val) {
        return Err(MlxError::InvalidArgument(format!(
            "softmax_sample_f32: random_val must be in [0, 1), got {}",
            random_val
        )));
    }
    if logits.element_count() != n_elements as usize {
        return Err(MlxError::InvalidArgument(format!(
            "softmax_sample_f32: logits element count {} != n_elements {}",
            logits.element_count(),
            n_elements
        )));
    }
    if scratch.element_count() != n_elements as usize {
        return Err(MlxError::InvalidArgument(format!(
            "softmax_sample_f32: scratch element count {} != n_elements {}",
            scratch.element_count(),
            n_elements
        )));
    }
    if out_token.element_count() < 1 {
        return Err(MlxError::InvalidArgument(
            "softmax_sample_f32: out_token must have at least 1 element".into(),
        ));
    }
    if out_logprob.element_count() < 1 {
        return Err(MlxError::InvalidArgument(
            "softmax_sample_f32: out_logprob must have at least 1 element".into(),
        ));
    }

    let pipeline = registry.get_pipeline("softmax_sample_f32", device)?;

    // Threadgroup size: next power-of-two of n_elements, capped at 1024.
    let tg_size = std::cmp::min(1024, n_elements.next_power_of_two()) as u64;

    // Shared memory: tg_size floats for the reduction (max pass and sum pass).
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, logits),
            (1, scratch),
            (2, out_token),
            (3, out_logprob),
            (4, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(1, 1, 1),       // single threadgroup
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
