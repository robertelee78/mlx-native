//! Helper utilities for encoding compute dispatches with inline constant
//! parameters (bytes) alongside buffer bindings.
//!
//! The base [`CommandEncoder::encode`] method only supports buffer bindings.
//! These helpers extend encoding to support Metal `set_bytes` for small
//! constant parameter structs, which avoids allocating a full Metal buffer
//! for a few bytes of configuration data.

use metal::{ComputePipelineStateRef, MTLSize};

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;

/// A buffer or inline-bytes binding for a compute kernel argument slot.
pub enum KernelArg<'a> {
    /// Bind an existing Metal buffer at the given index.
    Buffer(&'a MlxBuffer),
    /// Bind inline bytes (small constant data) at the given index.
    /// The data must be `Pod` and is copied into the command encoder.
    Bytes(&'a [u8]),
}

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
    let cmd_buf = encoder.metal_command_buffer();
    let compute_encoder = cmd_buf.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(pipeline);

    for &(index, ref arg) in bindings {
        match arg {
            KernelArg::Buffer(buf) => {
                compute_encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
            }
            KernelArg::Bytes(bytes) => {
                compute_encoder.set_bytes(index, bytes.len() as u64, bytes.as_ptr() as *const _);
            }
        }
    }

    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    compute_encoder.end_encoding();
}

/// Encode a compute pass with threadgroups and mixed buffer/bytes bindings.
pub fn encode_threadgroups_with_args(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipelineStateRef,
    bindings: &[(u64, KernelArg<'_>)],
    threadgroups: MTLSize,
    threadgroup_size: MTLSize,
) {
    let cmd_buf = encoder.metal_command_buffer();
    let compute_encoder = cmd_buf.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(pipeline);

    for &(index, ref arg) in bindings {
        match arg {
            KernelArg::Buffer(buf) => {
                compute_encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
            }
            KernelArg::Bytes(bytes) => {
                compute_encoder.set_bytes(index, bytes.len() as u64, bytes.as_ptr() as *const _);
            }
        }
    }

    compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    compute_encoder.end_encoding();
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
    let cmd_buf = encoder.metal_command_buffer();
    let compute_encoder = cmd_buf.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(pipeline);

    for &(index, ref arg) in bindings {
        match arg {
            KernelArg::Buffer(buf) => {
                compute_encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
            }
            KernelArg::Bytes(bytes) => {
                compute_encoder.set_bytes(index, bytes.len() as u64, bytes.as_ptr() as *const _);
            }
        }
    }

    for &(index, byte_length) in threadgroup_mem {
        compute_encoder.set_threadgroup_memory_length(index, byte_length);
    }

    compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    compute_encoder.end_encoding();
}

/// Convert a `Pod` value to a byte slice suitable for `KernelArg::Bytes`.
///
/// # Safety
///
/// The caller must ensure `T` has the same layout as the corresponding
/// MSL struct in the shader (matching field order, sizes, and alignment).
pub fn as_bytes<T: bytemuck::Pod>(val: &T) -> &[u8] {
    bytemuck::bytes_of(val)
}
