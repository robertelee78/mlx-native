//! Fast Walsh-Hadamard Transform (FWHT) GPU kernel dispatch.
//!
//! Applies an in-place, normalized FWHT to a flat buffer shaped
//! `[num_heads, head_dim]`.  One Metal threadgroup is dispatched per head;
//! each threadgroup has `head_dim` threads that cooperate through shared
//! memory using the standard butterfly pattern.
//!
//! The transform is normalized so that H·H = I (applying it twice returns
//! the original vector), which is required for the random-feature / scrambled
//! Hadamard use-case in Gemma-4 attention.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};

/// MSL source for the Hadamard transform kernel (embedded at compile time).
pub static HADAMARD_SHADER_SOURCE: &str = include_str!("../shaders/hadamard.metal");

/// Register the Hadamard transform shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("hadamard_transform", HADAMARD_SHADER_SOURCE);
}

/// Dispatch an in-place normalized Fast Walsh-Hadamard Transform on the GPU.
///
/// Transforms `data` in-place.  After this call the GPU contains
/// `H(data)` normalized by `1/sqrt(head_dim)`.
///
/// # Arguments
///
/// * `encoder`   — Command encoder to record the dispatch into.
/// * `registry`  — Kernel registry (must have `hadamard_transform` registered).
/// * `device`    — Metal device for pipeline compilation.
/// * `data`      — F32 buffer of shape `[num_heads, head_dim]`, modified in-place.
/// * `head_dim`  — Number of elements per head.  Must be a power of two and ≤ 8192.
/// * `num_heads` — Number of heads (threadgroups dispatched).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `head_dim` is not a power of two,
/// if the buffer is too small, or if `head_dim > 8192` (exceeds Metal's 32 KB
/// threadgroup memory limit for f32).
pub fn dispatch_hadamard_transform(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    data: &MlxBuffer,
    head_dim: u32,
    num_heads: u32,
) -> Result<()> {
    if num_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    // head_dim must be a power of two (butterfly pattern requirement).
    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_transform: head_dim must be a power of two, got {}",
            head_dim
        )));
    }

    // 32 KB threadgroup memory limit: head_dim * 4 bytes ≤ 32768 bytes → head_dim ≤ 8192
    if head_dim > 8192 {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_transform: head_dim {} exceeds Metal 32 KB threadgroup memory limit \
             (max 8192 for f32)",
            head_dim
        )));
    }

    let required_elements = (num_heads as u64) * (head_dim as u64);
    if (data.element_count() as u64) < required_elements {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_transform: data has {} elements but need {} \
             (num_heads={} * head_dim={})",
            data.element_count(),
            required_elements,
            num_heads,
            head_dim,
        )));
    }

    let pipeline = registry.get_pipeline("hadamard_transform", device)?;

    let head_dim_bytes = head_dim.to_ne_bytes();
    let num_heads_bytes = num_heads.to_ne_bytes();

    // Shared memory: head_dim floats (4 bytes each) per threadgroup.
    let shared_mem_bytes = (head_dim as u64) * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(data)),
            (1, KernelArg::Bytes(&head_dim_bytes)),
            (2, KernelArg::Bytes(&num_heads_bytes)),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );

    Ok(())
}
