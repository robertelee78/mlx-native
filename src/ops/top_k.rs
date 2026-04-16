//! GPU top-K dispatch — returns the K largest elements of a float array.
//!
//! Used by the Q8 lm_head rerank path to avoid a full 1 MB logits readback.
//! After Q8 matmul writes the full vocabulary of logits, this kernel selects
//! the top-K on GPU; only K * 8 bytes of (index, value) pairs come back to
//! CPU for exact F32 reranking.
//!
//! Output order is NOT guaranteed — callers that need sorted order should
//! sort themselves. The rerank path sorts implicitly by picking argmax over
//! the reranked logits.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static TOP_K_SHADER_SOURCE: &str = include_str!("../shaders/top_k.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("top_k_f32", TOP_K_SHADER_SOURCE);
}

/// Dispatch a top-K selection on the GPU.
///
/// # Arguments
///
/// * `encoder`     - Command encoder to record the dispatch into.
/// * `registry`    - Kernel registry (must have `top_k_f32` registered).
/// * `device`      - Metal device for pipeline compilation.
/// * `input`       - Input buffer `[n_elements]` (f32).
/// * `out_indices` - Output buffer `[k]` (u32) — indices of top-K elements.
/// * `out_values`  - Output buffer `[k]` (f32) — values of top-K elements.
/// * `params_buf`  - Params buffer `[2]` (u32) — `[n_elements, k]`.
/// * `n_elements`  - Number of elements in `input`.
/// * `k`           - Number of top elements to return. Must be <= 128.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements == 0`, `k == 0`,
/// `k > 128`, or buffer sizes don't match.
pub fn dispatch_top_k_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    out_indices: &MlxBuffer,
    out_values: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_elements: u32,
    k: u32,
) -> Result<()> {
    if n_elements == 0 || k == 0 {
        return Err(MlxError::InvalidArgument(
            "top_k_f32: n_elements and k must be > 0".into(),
        ));
    }
    if k > 128 {
        return Err(MlxError::InvalidArgument(format!(
            "top_k_f32: k ({}) must be <= 128 (MAX_K in shader)",
            k
        )));
    }
    if input.element_count() < n_elements as usize {
        return Err(MlxError::InvalidArgument(format!(
            "top_k_f32: input element count {} < n_elements {}",
            input.element_count(),
            n_elements
        )));
    }
    if out_indices.element_count() < k as usize {
        return Err(MlxError::InvalidArgument(format!(
            "top_k_f32: out_indices ({}) < k ({})",
            out_indices.element_count(),
            k
        )));
    }
    if out_values.element_count() < k as usize {
        return Err(MlxError::InvalidArgument(format!(
            "top_k_f32: out_values ({}) < k ({})",
            out_values.element_count(),
            k
        )));
    }

    let pipeline = registry.get_pipeline("top_k_f32", device)?;

    // tg_size choice: threadgroup shared memory on Apple Silicon is ~32 KB.
    // Shared = tg_size * K * (4 + 4) bytes.
    //   K=64  → tg_size <= 64  (32 KB)
    //   K=32  → tg_size <= 128
    //   K=128 → tg_size <= 32
    // Use tg_size=32 for K up to 128 to be safe across Apple generations.
    let tg_size: u64 = match k {
        1..=32 => 128,
        33..=64 => 64,
        _ => 32,
    };

    // Shared memory: tg_size * K each for values (float) and indices (uint).
    let float_shared = tg_size * (k as u64) * 4;
    let uint_shared  = tg_size * (k as u64) * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, out_indices),
            (2, out_values),
            (3, params_buf),
        ],
        &[(0, float_shared), (1, uint_shared)],
        MTLSize::new(1, 1, 1),          // single threadgroup
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
