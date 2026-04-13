//! [`CommandEncoder`] — batched GPU command submission.
//!
//! Wraps a Metal command buffer.  Encode one or more compute kernel dispatches,
//! then call [`commit_and_wait`](CommandEncoder::commit_and_wait) to submit the
//! entire batch and block until the GPU finishes.
//!
//! # Persistent compute encoder
//!
//! A single Metal `ComputeCommandEncoder` is kept alive across multiple
//! dispatches within the same command buffer.  This avoids the overhead of
//! creating and ending a new compute encoder per dispatch — the same pattern
//! candle uses (`compute_per_buffer`).  On a forward pass with ~800 dispatches
//! this saves ~800 encoder create/end cycles.

use std::sync::atomic::{AtomicU64, Ordering};

use metal::{
    CommandBuffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineStateRef,
    MTLCommandBufferStatus, MTLDispatchType, MTLSize,
};
#[allow(unused_imports)]
use objc::{msg_send, sel, sel_impl};

use crate::buffer::MlxBuffer;
use crate::error::{MlxError, Result};

/// A buffer or inline-bytes binding for a compute kernel argument slot.
pub enum KernelArg<'a> {
    /// Bind an existing Metal buffer at the given index.
    Buffer(&'a MlxBuffer),
    /// Bind an existing Metal buffer at the given index with a byte offset.
    BufferWithOffset(&'a MlxBuffer, u64),
    /// Bind inline bytes (small constant data) at the given index.
    /// The data must be `Pod` and is copied into the command encoder.
    Bytes(&'a [u8]),
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

/// Apply a slice of `KernelArg` bindings to a compute encoder.
#[inline]
fn apply_bindings(encoder: &ComputeCommandEncoderRef, bindings: &[(u64, KernelArg<'_>)]) {
    for &(index, ref arg) in bindings {
        match arg {
            KernelArg::Buffer(buf) => {
                encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
            }
            KernelArg::BufferWithOffset(buf, offset) => {
                encoder.set_buffer(index, Some(buf.metal_buffer()), *offset);
            }
            KernelArg::Bytes(bytes) => {
                encoder.set_bytes(index, bytes.len() as u64, bytes.as_ptr() as *const _);
            }
        }
    }
}

/// Number of times `commit_and_wait()` has been called (CPU sync points).
static SYNC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of times an encode method has been called (GPU dispatches).
static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset both `SYNC_COUNT` and `DISPATCH_COUNT` to zero.
pub fn reset_counters() {
    SYNC_COUNT.store(0, Ordering::Relaxed);
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

/// Read the current value of `SYNC_COUNT`.
///
/// Each call to `commit_and_wait()` increments this counter.
pub fn sync_count() -> u64 {
    SYNC_COUNT.load(Ordering::Relaxed)
}

/// Read the current value of `DISPATCH_COUNT`.
///
/// Each call to `encode()`, `encode_threadgroups()`, or
/// `encode_threadgroups_with_shared()` increments this counter.
pub fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// A batched compute command encoder.
///
/// Keeps a single Metal `ComputeCommandEncoder` alive across multiple
/// dispatches.  The encoder is created on the first dispatch and ended
/// only when the command buffer is committed.  This mirrors candle's
/// `compute_per_buffer` pattern and avoids per-dispatch encoder overhead.
///
/// # Typical usage
///
/// ```ignore
/// let mut enc = device.command_encoder()?;
/// // Multiple dispatches share the same compute encoder:
/// enc.encode_threadgroups(pipeline1, &buffers1, tg1, tg_size1);
/// enc.encode_threadgroups(pipeline2, &buffers2, tg2, tg_size2);
/// enc.commit_and_wait()?;
/// ```
pub struct CommandEncoder {
    cmd_buf: CommandBuffer,
    /// Raw pointer to the persistent compute encoder.
    /// Non-null when a compute pass is active.
    /// The encoder borrows from `cmd_buf` but we cannot express this
    /// lifetime in safe Rust, so we use a raw pointer.
    /// SAFETY: the pointer is valid as long as `cmd_buf` is alive and
    /// `end_encoding()` has not been called on it.
    active_encoder: *const ComputeCommandEncoderRef,
}

impl CommandEncoder {
    /// Create a new command encoder from the given command queue.
    ///
    /// This immediately creates a Metal command buffer.
    pub(crate) fn new(queue: &CommandQueue) -> Result<Self> {
        let cmd_buf = queue.new_command_buffer().to_owned();
        Ok(Self {
            cmd_buf,
            active_encoder: std::ptr::null(),
        })
    }

    /// Get or create the persistent compute encoder.
    ///
    /// On the first call, creates a new compute encoder from the command
    /// buffer.  On subsequent calls, returns the existing one.
    ///
    /// SAFETY: The returned reference borrows from `self.cmd_buf` which is
    /// alive for the lifetime of this `CommandEncoder`.  The raw pointer is
    /// valid until `end_active_encoder()` is called.
    #[inline]
    fn get_or_create_encoder(&mut self) -> &ComputeCommandEncoderRef {
        if self.active_encoder.is_null() {
            // Use MTLDispatchTypeConcurrent to allow independent dispatches
            // to overlap on the GPU.  Memory barriers are inserted between
            // dependent dispatches via `memory_barrier()`.
            let encoder = self
                .cmd_buf
                .compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            self.active_encoder = encoder as *const ComputeCommandEncoderRef;
        }
        // SAFETY: active_encoder is non-null and points to a valid encoder
        // owned by cmd_buf.
        unsafe { &*self.active_encoder }
    }

    /// End the active compute encoder if one exists.
    #[inline]
    fn end_active_encoder(&mut self) {
        if !self.active_encoder.is_null() {
            // SAFETY: the pointer was obtained from cmd_buf.new_compute_command_encoder()
            // and has not been ended yet.
            unsafe { &*self.active_encoder }.end_encoding();
            self.active_encoder = std::ptr::null();
        }
    }

    /// Insert a memory barrier with scope `MTLBarrierScopeBuffers`.
    ///
    /// When the encoder uses `MTLDispatchTypeConcurrent`, all dispatches can
    /// execute concurrently unless separated by a barrier.  Call this between
    /// dispatches where the later dispatch reads a buffer written by an
    /// earlier one.
    ///
    /// This is the same pattern llama.cpp uses:
    /// `[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]`
    pub fn memory_barrier(&mut self) {
        if self.active_encoder.is_null() {
            return;
        }
        // SAFETY: active_encoder is non-null and valid.
        let encoder = unsafe { &*self.active_encoder };
        // MTLBarrierScopeBuffers = 1 << 0 = 1
        const MTL_BARRIER_SCOPE_BUFFERS: u64 = 1;
        unsafe {
            let _: () = objc::msg_send![encoder, memoryBarrierWithScope: MTL_BARRIER_SCOPE_BUFFERS];
        }
    }

    /// Set the compute pipeline state for subsequent dispatches.
    ///
    /// This begins a new compute pass if one is not already active.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipelineStateRef) {
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
    }

    /// Bind a buffer to a compute kernel argument slot.
    ///
    /// The `index` corresponds to the `[[buffer(N)]]` attribute in the MSL shader.
    pub fn set_buffer(&self, index: u64, buffer: &MlxBuffer) {
        let _ = (index, buffer);
    }

    /// Dispatch threads on the GPU.
    pub fn dispatch_threads(&self, grid_size: MTLSize, threadgroup_size: MTLSize) {
        let _ = (grid_size, threadgroup_size);
    }

    /// Encode a complete compute pass: set pipeline, bind buffers, dispatch.
    ///
    /// Reuses the persistent compute encoder — no per-dispatch encoder
    /// creation overhead.
    ///
    /// # Arguments
    ///
    /// * `pipeline`         — The compiled compute pipeline to execute.
    /// * `buffers`          — Slice of `(index, &MlxBuffer)` pairs for buffer bindings.
    /// * `grid_size`        — Total number of threads to launch.
    /// * `threadgroup_size` — Threads per threadgroup.
    pub fn encode(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
        }
        encoder.dispatch_threads(grid_size, threadgroup_size);
    }

    /// Encode a compute pass using threadgroups instead of raw thread counts.
    ///
    /// Reuses the persistent compute encoder — no per-dispatch encoder
    /// creation overhead.
    pub fn encode_threadgroups(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a compute pass using threadgroups with shared threadgroup memory.
    ///
    /// Like [`encode_threadgroups`](Self::encode_threadgroups), but additionally
    /// allocates threadgroup memory at the specified indices.  This is required
    /// for kernels that use `threadgroup` memory (e.g. reductions in rms_norm
    /// and softmax).
    ///
    /// # Arguments
    ///
    /// * `pipeline`         — The compiled compute pipeline to execute.
    /// * `buffers`          — Slice of `(index, &MlxBuffer)` pairs for buffer bindings.
    /// * `threadgroup_mem`  — Slice of `(index, byte_length)` pairs for threadgroup memory.
    /// * `threadgroups`     — Number of threadgroups to dispatch.
    /// * `threadgroup_size` — Threads per threadgroup.
    pub fn encode_threadgroups_with_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        buffers: &[(u64, &MlxBuffer)],
        threadgroup_mem: &[(u64, u64)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for &(index, buf) in buffers {
            encoder.set_buffer(index, Some(buf.metal_buffer()), 0);
        }
        for &(index, byte_length) in threadgroup_mem {
            encoder.set_threadgroup_memory_length(index, byte_length);
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings (dispatch_threads).
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        encoder.dispatch_threads(grid_size, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings (dispatch_thread_groups).
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_threadgroups_with_args(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Encode a dispatch with mixed buffer/bytes bindings and shared memory.
    ///
    /// Reuses the persistent compute encoder.
    pub fn encode_threadgroups_with_args_and_shared(
        &mut self,
        pipeline: &ComputePipelineStateRef,
        bindings: &[(u64, KernelArg<'_>)],
        threadgroup_mem: &[(u64, u64)],
        threadgroups: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let encoder = self.get_or_create_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        apply_bindings(encoder, bindings);
        for &(index, byte_length) in threadgroup_mem {
            encoder.set_threadgroup_memory_length(index, byte_length);
        }
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Commit the command buffer and block until the GPU finishes execution.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn commit_and_wait(&mut self) -> Result<()> {
        SYNC_COUNT.fetch_add(1, Ordering::Relaxed);

        // End the persistent compute encoder before committing.
        self.end_active_encoder();

        self.cmd_buf.commit();
        self.cmd_buf.wait_until_completed();

        match self.cmd_buf.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            MTLCommandBufferStatus::Error => {
                Err(MlxError::CommandBufferError(
                    "GPU command buffer completed with error status".into(),
                ))
            }
            status => Err(MlxError::CommandBufferError(format!(
                "Unexpected command buffer status after wait: {:?}",
                status
            ))),
        }
    }

    /// Commit the command buffer WITHOUT blocking.
    ///
    /// The GPU begins executing the encoded commands immediately.  Call
    /// [`wait_until_completed`](Self::wait_until_completed) later to block
    /// the CPU and check for errors.  This allows the CPU to continue doing
    /// other work (e.g. preparing the next batch) while the GPU runs.
    pub fn commit(&mut self) {
        self.end_active_encoder();
        self.cmd_buf.commit();
    }

    /// Block until a previously committed command buffer completes.
    ///
    /// Must be called after [`commit`](Self::commit).  Do not call after
    /// [`commit_and_wait`](Self::commit_and_wait) — that method already waits.
    ///
    /// # Errors
    ///
    /// Returns `MlxError::CommandBufferError` if the GPU reports an error.
    pub fn wait_until_completed(&self) -> Result<()> {
        self.cmd_buf.wait_until_completed();
        match self.cmd_buf.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            MTLCommandBufferStatus::Error => Err(MlxError::CommandBufferError(
                "GPU command buffer completed with error status".into(),
            )),
            status => Err(MlxError::CommandBufferError(format!(
                "Unexpected command buffer status after wait: {:?}",
                status
            ))),
        }
    }

    /// Borrow the underlying Metal command buffer.
    #[inline]
    pub fn metal_command_buffer(&self) -> &CommandBuffer {
        &self.cmd_buf
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        // End the persistent compute encoder before the command buffer
        // is dropped, otherwise Metal will assert:
        // "Command encoder released without endEncoding"
        self.end_active_encoder();
    }
}
