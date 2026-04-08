//! Greedy argmax GPU dispatch — finds the index of the maximum value in a
//! float array entirely on the GPU.
//!
//! For greedy (temperature=0) decoding with vocab_size=262144, this replaces
//! a 1MB GPU→CPU logits readback with an 8-byte readback: the (index, value)
//! pair.  The kernel uses a single threadgroup with shared-memory tree
//! reduction.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the argmax kernel (embedded at compile time).
pub static ARGMAX_SHADER_SOURCE: &str = include_str!("../shaders/argmax.metal");

/// Register argmax shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("argmax_f32", ARGMAX_SHADER_SOURCE);
}

/// Dispatch an argmax operation on the GPU.
///
/// Finds the index of the maximum element in `input` and writes the result to
/// `out_index` and `out_value`.  The entire reduction runs in a single Metal
/// threadgroup, returning 8 bytes instead of the full vocab-size logits array.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have `argmax_f32` registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[n_elements]` (f32).
/// * `out_index`  - Output buffer `[1]` (u32) — index of the maximum element.
/// * `out_value`  - Output buffer `[1]` (f32) — value of the maximum element.
/// * `params_buf` - Params buffer `[1]` (u32) — contains `n_elements`.
/// * `n_elements` - Number of elements in `input`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `n_elements` is 0.
/// - `input` element count does not match `n_elements`.
/// - `out_index` or `out_value` element count is not 1.
pub fn dispatch_argmax_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    out_index: &MlxBuffer,
    out_value: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_elements: u32,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "argmax_f32: n_elements must be > 0".into(),
        ));
    }
    if input.element_count() != n_elements as usize {
        return Err(MlxError::InvalidArgument(format!(
            "argmax_f32: input element count {} != n_elements {}",
            input.element_count(),
            n_elements
        )));
    }
    if out_index.element_count() < 1 {
        return Err(MlxError::InvalidArgument(
            "argmax_f32: out_index must have at least 1 element".into(),
        ));
    }
    if out_value.element_count() < 1 {
        return Err(MlxError::InvalidArgument(
            "argmax_f32: out_value must have at least 1 element".into(),
        ));
    }

    let pipeline = registry.get_pipeline("argmax_f32", device)?;

    // Threadgroup size: next power-of-two of n_elements, capped at 1024.
    // Must be a power of 2 for the tree reduction to be correct.
    let tg_size = std::cmp::min(1024, n_elements.next_power_of_two()) as u64;

    // Shared memory:
    //   index 0 — tg_size floats for value reduction
    //   index 1 — tg_size uints  for index tracking
    let float_shared = tg_size * 4; // sizeof(float) = 4
    let uint_shared  = tg_size * 4; // sizeof(uint)  = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, out_index),
            (2, out_value),
            (3, params_buf),
        ],
        &[(0, float_shared), (1, uint_shared)],
        MTLSize::new(1, 1, 1),       // single threadgroup
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
