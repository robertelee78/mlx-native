//! Helper utilities for encoding compute dispatches with inline constant
//! parameters (bytes) alongside buffer bindings.
//!
//! The base [`CommandEncoder::encode`] method only supports buffer bindings.
//! These helpers extend encoding to support Metal `set_bytes` for small
//! constant parameter structs, which avoids allocating a full Metal buffer
//! for a few bytes of configuration data.
//!
//! `KernelArg` and `as_bytes` are defined in `crate::encoder` and re-exported
//! here for backward compatibility.

use metal::{ComputePipelineStateRef, MTLSize};

use crate::encoder::CommandEncoder;

// Re-export from encoder module where KernelArg now lives.
pub use crate::encoder::{KernelArg, as_bytes};

/// Encode a compute pass with mixed buffer and bytes bindings.
///
/// This is an extension of [`CommandEncoder::encode`] that additionally
/// supports `set_bytes` for small constant parameter structs.
///
/// # Arguments
///
/// * `encoder`          — The command encoder to record into.
/// * `pipeline`         — The compiled compute pipeline.
/// * `bindings`         — Slice of `(index, KernelArg)` pairs.
/// * `grid_size`        — Total threads to launch.
/// * `threadgroup_size` — Threads per threadgroup.
pub fn encode_with_args(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipelineStateRef,
    bindings: &[(u64, KernelArg<'_>)],
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) {
    // Use the encoder's persistent compute encoder via encode_with_args_dispatch.
    // This delegates to CommandEncoder's own dispatch methods that reuse the
    // same compute encoder across calls.
    encoder.encode_with_args(pipeline, bindings, grid_size, threadgroup_size);
}

/// Encode a compute pass with threadgroups and mixed buffer/bytes bindings.
pub fn encode_threadgroups_with_args(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipelineStateRef,
    bindings: &[(u64, KernelArg<'_>)],
    threadgroups: MTLSize,
    threadgroup_size: MTLSize,
) {
    encoder.encode_threadgroups_with_args(pipeline, bindings, threadgroups, threadgroup_size);
}

/// Encode a compute pass with threadgroups, mixed buffer/bytes bindings, and
/// threadgroup shared memory allocations.
///
/// Combines the capabilities of [`encode_threadgroups_with_args`] (inline bytes
/// via `set_bytes`) and the encoder's `encode_threadgroups_with_shared` (shared
/// memory allocation).  Required by fused kernels that need both constant-struct
/// parameters and a threadgroup scratch buffer for reduction.
///
/// # Arguments
///
/// * `encoder`          — The command encoder to record into.
/// * `pipeline`         — The compiled compute pipeline.
/// * `bindings`         — Slice of `(index, KernelArg)` pairs.
/// * `threadgroup_mem`  — Slice of `(index, byte_length)` pairs for threadgroup memory.
/// * `threadgroups`     — Number of threadgroups to dispatch.
/// * `threadgroup_size` — Threads per threadgroup.
pub fn encode_threadgroups_with_args_and_shared(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipelineStateRef,
    bindings: &[(u64, KernelArg<'_>)],
    threadgroup_mem: &[(u64, u64)],
    threadgroups: MTLSize,
    threadgroup_size: MTLSize,
) {
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        bindings,
        threadgroup_mem,
        threadgroups,
        threadgroup_size,
    );
}

// as_bytes is re-exported from crate::encoder above.
