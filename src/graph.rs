//! [`GraphExecutor`] — batched Metal dispatch for single-encoder forward passes.
//!
//! llama.cpp's speed advantage over candle is NOT the kernels (Phase 0 proved
//! candle's are as fast or faster per-call).  It is the dispatch pattern:
//! 1 encoder per command buffer instead of ~120.  This module implements that
//! pattern.
//!
//! # Usage
//!
//! ```ignore
//! let mut executor = GraphExecutor::new(device.clone());
//! let mut session = executor.begin()?;
//!
//! // All ops encode into the same command buffer — no per-op encoder creation.
//! session.rms_norm(&mut registry, device.metal_device(), input, weight, output, params, rows, dim)?;
//! session.quantized_matmul(&mut registry, &device, input, weight, scales, biases, &qparams)?;
//! session.elementwise_add(&mut registry, device.metal_device(), a, b, out, n, DType::F32)?;
//!
//! // Single GPU sync point for the entire forward pass.
//! session.finish()?;
//! ```
//!
//! # Design
//!
//! The `GraphSession` holds a single `CommandEncoder`.  Each op method delegates
//! to the existing op dispatch functions in [`crate::ops`], passing the session's
//! shared encoder.  No new Metal code is needed — the ops already work with a
//! shared encoder.  The executor just prevents creating a new encoder per op.
//!
//! There is no formal graph IR (DAG, topological sort, etc.).  The target
//! architecture doc explicitly chose sequential op calls, matching ggml-metal's
//! pattern.  See `/opt/hf2q/docs/arch-target-inference-path.md` section 5.

use crate::device::MlxDevice;
use crate::encoder::CommandEncoder;
use crate::error::Result;
use crate::kernel_registry::KernelRegistry;
use crate::ops;

// Re-export types used in the public API so callers don't need separate imports.
pub use crate::buffer::MlxBuffer;
pub use crate::dtypes::DType;

/// Batched Metal dispatch — encodes multiple ops into a single `CommandEncoder`.
///
/// Create one per model (or per forward-pass loop).  Call [`begin`](Self::begin)
/// at the start of each forward pass to get a [`GraphSession`] that holds the
/// shared encoder.
pub struct GraphExecutor {
    device: MlxDevice,
}

impl GraphExecutor {
    /// Create a new graph executor backed by the given device.
    pub fn new(device: MlxDevice) -> Self {
        Self { device }
    }

    /// Begin a new forward pass.
    ///
    /// Returns a [`GraphSession`] that holds a fresh `CommandEncoder`.  All ops
    /// encoded through the session share this single encoder.  Call
    /// [`GraphSession::finish`] to commit and wait.
    pub fn begin(&self) -> Result<GraphSession<'_>> {
        let encoder = self.device.command_encoder()?;
        Ok(GraphSession {
            encoder,
            device: &self.device,
        })
    }

    /// Borrow the underlying device.
    pub fn device(&self) -> &MlxDevice {
        &self.device
    }
}

/// A single forward pass execution context.
///
/// All ops are encoded into one `CommandEncoder`.  Call [`finish`](Self::finish)
/// to commit the command buffer and wait for GPU completion — this is the ONLY
/// sync point per forward pass.
///
/// If an op returns an error, the session can be dropped without committing.
/// The underlying command buffer is abandoned (never committed to the GPU).
pub struct GraphSession<'a> {
    encoder: CommandEncoder,
    device: &'a MlxDevice,
}

impl<'a> GraphSession<'a> {
    /// Encode an RMS normalization into this session's encoder.
    ///
    /// Delegates to [`ops::rms_norm::dispatch_rms_norm`].
    pub fn rms_norm(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        dim: u32,
    ) -> Result<()> {
        ops::rms_norm::dispatch_rms_norm(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            output,
            params_buf,
            rows,
            dim,
        )
    }

    /// Encode a quantized matrix multiplication into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul::quantized_matmul`].
    /// Returns the freshly allocated output buffer.
    pub fn quantized_matmul(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        scales: &MlxBuffer,
        biases: &MlxBuffer,
        params: &ops::quantized_matmul::QuantizedMatmulParams,
    ) -> Result<MlxBuffer> {
        ops::quantized_matmul::quantized_matmul(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            scales,
            biases,
            params,
        )
    }

    /// Encode a SIMD-optimized quantized matmul into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul::quantized_matmul_simd`].
    /// Returns the freshly allocated output buffer.
    pub fn quantized_matmul_simd(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        scales: &MlxBuffer,
        biases: &MlxBuffer,
        params: &ops::quantized_matmul::QuantizedMatmulParams,
    ) -> Result<MlxBuffer> {
        ops::quantized_matmul::quantized_matmul_simd(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            scales,
            biases,
            params,
        )
    }

    /// Encode a GGML block-format quantized mat-vec into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul_ggml::quantized_matmul_ggml`].
    pub fn quantized_matmul_ggml(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        output: &mut MlxBuffer,
        params: &ops::quantized_matmul_ggml::GgmlQuantizedMatmulParams,
    ) -> Result<()> {
        ops::quantized_matmul_ggml::quantized_matmul_ggml(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            output,
            params,
        )
    }

    /// Encode an expert-routed GGML block-format quantized mat-vec into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml`].
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul_id_ggml(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        ids: &MlxBuffer,
        output: &mut MlxBuffer,
        params: &ops::quantized_matmul_id_ggml::GgmlQuantizedMatmulIdParams,
    ) -> Result<()> {
        ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            ids,
            output,
            params,
        )
    }

    /// Encode scaled dot-product attention into this session's encoder.
    ///
    /// Delegates to [`ops::sdpa::sdpa`].
    pub fn sdpa(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        q: &MlxBuffer,
        k: &MlxBuffer,
        v: &MlxBuffer,
        output: &MlxBuffer,
        params: &ops::sdpa::SdpaParams,
        batch_size: u32,
    ) -> Result<()> {
        ops::sdpa::sdpa(
            &mut self.encoder,
            registry,
            device,
            q,
            k,
            v,
            output,
            params,
            batch_size,
        )
    }

    /// Encode an elementwise add into this session's encoder.
    ///
    /// Delegates to [`ops::elementwise::elementwise_add`].
    pub fn elementwise_add(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        a: &MlxBuffer,
        b: &MlxBuffer,
        output: &MlxBuffer,
        n_elements: usize,
        dtype: DType,
    ) -> Result<()> {
        ops::elementwise::elementwise_add(
            &mut self.encoder,
            registry,
            device,
            a,
            b,
            output,
            n_elements,
            dtype,
        )
    }

    /// Encode an elementwise multiply into this session's encoder.
    ///
    /// Delegates to [`ops::elementwise::elementwise_mul`].
    pub fn elementwise_mul(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        a: &MlxBuffer,
        b: &MlxBuffer,
        output: &MlxBuffer,
        n_elements: usize,
        dtype: DType,
    ) -> Result<()> {
        ops::elementwise::elementwise_mul(
            &mut self.encoder,
            registry,
            device,
            a,
            b,
            output,
            n_elements,
            dtype,
        )
    }

    /// Encode a RoPE transform into this session's encoder.
    ///
    /// Delegates to [`ops::rope::dispatch_rope`].
    pub fn rope(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        positions_buf: &MlxBuffer,
        seq_len: u32,
        head_dim: u32,
    ) -> Result<()> {
        ops::rope::dispatch_rope(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            positions_buf,
            seq_len,
            head_dim,
        )
    }

    /// Encode a GELU activation into this session's encoder.
    ///
    /// Delegates to [`ops::gelu::dispatch_gelu`].
    pub fn gelu(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
    ) -> Result<()> {
        ops::gelu::dispatch_gelu(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
        )
    }

    /// Encode a softmax into this session's encoder.
    ///
    /// Delegates to [`ops::softmax::dispatch_softmax`].
    pub fn softmax(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        ops::softmax::dispatch_softmax(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            rows,
            cols,
        )
    }

    /// Encode a softcap into this session's encoder.
    ///
    /// Delegates to [`ops::softcap::dispatch_softcap`].
    pub fn softcap(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        cap: f32,
    ) -> Result<()> {
        ops::softcap::dispatch_softcap(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            cap,
        )
    }

    /// Encode an RMS norm without learned scale (f32) into this session's encoder.
    ///
    /// Delegates to [`ops::rms_norm::dispatch_rms_norm_no_scale_f32`].
    pub fn rms_norm_no_scale_f32(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        dim: u32,
    ) -> Result<()> {
        ops::rms_norm::dispatch_rms_norm_no_scale_f32(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            rows,
            dim,
        )
    }

    /// Encode a NeoX RoPE (f32) with optional freq_factors into this session's encoder.
    ///
    /// Delegates to [`ops::rope::dispatch_rope_neox_f32`].
    #[allow(clippy::too_many_arguments)]
    pub fn rope_neox_f32(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        positions_buf: &MlxBuffer,
        freq_factors: Option<&MlxBuffer>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
    ) -> Result<()> {
        ops::rope::dispatch_rope_neox_f32(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            positions_buf,
            freq_factors,
            seq_len,
            n_heads,
            head_dim,
            rope_dim,
        )
    }

    /// Borrow the underlying command encoder for direct op dispatch.
    ///
    /// Use this when you need to call an op function that is not wrapped by
    /// a `GraphSession` method.  The returned encoder is the same shared
    /// encoder — all dispatches still go into the same command buffer.
    pub fn encoder_mut(&mut self) -> &mut CommandEncoder {
        &mut self.encoder
    }

    /// Borrow the device reference.
    pub fn device(&self) -> &MlxDevice {
        self.device
    }

    /// Commit the command buffer and wait for GPU completion.
    ///
    /// This is the ONLY sync point per forward pass.  After this call, all
    /// output buffers are readable by the CPU.
    ///
    /// Consumes the session — no further ops can be encoded.
    pub fn finish(mut self) -> Result<()> {
        self.encoder.commit_and_wait()
    }

    /// Commit the command buffer WITHOUT waiting.
    ///
    /// The GPU begins executing immediately.  Use this for fire-and-forget
    /// dispatch when you do not need results until later.
    ///
    /// Consumes the session.
    pub fn commit(mut self) -> CommandEncoder {
        self.encoder.commit();
        self.encoder
    }
}
