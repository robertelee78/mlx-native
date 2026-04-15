//! [`KernelRegistry`] — lazy compilation and caching of Metal compute pipelines.
//!
//! MSL shader source is embedded at compile time via `include_str!`.  On first
//! access, the source is compiled into a Metal library, the named function is
//! extracted, and a `ComputePipelineState` is created and cached.  Subsequent
//! calls return the cached pipeline.

use std::collections::HashMap;

use metal::ComputePipelineState;

use crate::error::{MlxError, Result};

/// Registry that lazily compiles and caches Metal compute pipelines from
/// embedded MSL source.
///
/// # Usage
///
/// ```ignore
/// let mut registry = KernelRegistry::new();
/// let pipeline = registry.get_pipeline("elementwise_add", device.metal_device())?;
/// encoder.encode(&pipeline, &buffers, grid, tg);
/// ```
///
/// # Thread Safety
///
/// `KernelRegistry` is **not** `Sync` by default (it uses `&mut self` for
/// `get_pipeline` to allow mutable cache insertion).  If you need concurrent
/// access, wrap it in a `Mutex` or use one registry per thread.
pub struct KernelRegistry {
    /// Cached pipelines keyed by kernel function name.
    cache: HashMap<String, ComputePipelineState>,
    /// MSL source text keyed by kernel function name.
    ///
    /// Populated at construction time with all embedded shader sources.
    sources: HashMap<String, &'static str>,
}

impl KernelRegistry {
    /// Create a new registry with all embedded shader sources pre-registered.
    ///
    /// No compilation happens here — shaders are compiled lazily on first use.
    pub fn new() -> Self {
        let mut sources = HashMap::new();

        // Register embedded shader sources.
        sources.insert(
            "placeholder".into(),
            include_str!("shaders/placeholder.metal"),
        );
        sources.insert(
            "quantized_matmul".into(),
            include_str!("shaders/quantized_matmul.metal"),
        );
        sources.insert(
            "quantized_matmul_simd".into(),
            include_str!("shaders/quantized_matmul.metal"),
        );
        sources.insert(
            "quantized_matmul_simd_bf16".into(),
            include_str!("shaders/quantized_matmul.metal"),
        );
        sources.insert(
            "quantized_matmul_simd_bf16_expert".into(),
            include_str!("shaders/quantized_matmul.metal"),
        );

        // GGML block-format quantized mat-vec kernels (ADR-006 Phase 3)
        let ggml_src: &'static str =
            include_str!("shaders/quantized_matmul_ggml.metal");
        sources.insert("kernel_mul_mv_q4_0_f32".into(), ggml_src);
        sources.insert("kernel_mul_mv_q8_0_f32".into(), ggml_src);
        sources.insert("kernel_mul_mv_q6_K_f32".into(), ggml_src);

        // Expert-routed (MoE) quantized matmul kernel (Story 2.1)
        sources.insert(
            "quantized_matmul_id".into(),
            include_str!("shaders/quantized_matmul_id.metal"),
        );

        // Expert-routed (MoE) GGML block-format quantized matmul kernels
        let ggml_id_src: &'static str =
            include_str!("shaders/quantized_matmul_id_ggml.metal");
        sources.insert("kernel_mul_mv_id_q4_0_f32".into(), ggml_id_src);
        sources.insert("kernel_mul_mv_id_q8_0_f32".into(), ggml_id_src);
        sources.insert("kernel_mul_mv_id_q6_K_f32".into(), ggml_id_src);

        // Embedding kernels (Story 1.5)
        let embedding_src: &'static str = include_str!("shaders/embedding.metal");
        sources.insert("embedding_gather_4bit".into(), embedding_src);
        sources.insert("embedding_gather_6bit".into(), embedding_src);

        // MoE gate kernel (Story 1.5)
        let moe_gate_src: &'static str = include_str!("shaders/moe_gate.metal");
        sources.insert("moe_gate".into(), moe_gate_src);

        // MoE dispatch kernels (Story 1.5)
        let moe_dispatch_src: &'static str = include_str!("shaders/moe_dispatch.metal");
        sources.insert("fused_gelu_mul".into(), moe_dispatch_src);
        sources.insert("moe_swiglu_fused".into(), moe_dispatch_src);
        sources.insert("moe_swiglu_batch".into(), moe_dispatch_src);
        sources.insert("moe_accumulate".into(), moe_dispatch_src);
        sources.insert("moe_weighted_sum".into(), moe_dispatch_src);
        sources.insert("zero_buffer".into(), moe_dispatch_src);
        sources.insert("naive_matvec_f32".into(), moe_dispatch_src);
        sources.insert("moe_gather_topk_weights".into(), moe_dispatch_src);

        // Batched KV cache copy kernels
        let kv_cache_src: &'static str = include_str!("shaders/kv_cache_copy.metal");
        sources.insert("kv_cache_copy_batch_f32".into(), kv_cache_src);
        sources.insert("kv_cache_copy_batch_f32_to_f16".into(), kv_cache_src);

        // Elementwise and transpose kernels (Story 1.5)
        let elementwise_src: &'static str = include_str!("shaders/elementwise.metal");
        sources.insert("elementwise_add_f32".into(), elementwise_src);
        sources.insert("elementwise_add_f16".into(), elementwise_src);
        sources.insert("elementwise_mul_f32".into(), elementwise_src);
        sources.insert("elementwise_mul_f16".into(), elementwise_src);
        sources.insert("elementwise_add_bf16".into(), elementwise_src);
        sources.insert("elementwise_mul_bf16".into(), elementwise_src);
        sources.insert("cast_f16_to_f32".into(), elementwise_src);
        sources.insert("cast_f32_to_f16".into(), elementwise_src);
        sources.insert("cast_bf16_to_f32".into(), elementwise_src);
        sources.insert("cast_f32_to_bf16".into(), elementwise_src);
        sources.insert("scalar_mul_bf16".into(), elementwise_src);
        sources.insert("scalar_mul_f32".into(), elementwise_src);
        sources.insert("embedding_gather_scale_f32".into(), elementwise_src);
        sources.insert("permute_021_bf16".into(), elementwise_src);
        sources.insert("transpose_2d_f32".into(), elementwise_src);
        sources.insert("transpose_2d_f16".into(), elementwise_src);

        // Attention kernels (Story 1.3)
        let sdpa_src: &'static str = include_str!("shaders/sdpa.metal");
        sources.insert("sdpa".into(), sdpa_src);
        sources.insert("sdpa_bf16".into(), sdpa_src);
        let sdpa_sliding_src: &'static str = include_str!("shaders/sdpa_sliding.metal");
        sources.insert("sdpa_sliding".into(), sdpa_sliding_src);
        sources.insert("sdpa_sliding_bf16".into(), sdpa_sliding_src);

        // Flash attention vector kernels — SIMD-vectorized decode-path SDPA
        // (ported from llama.cpp flash_attn_ext_vec)
        let flash_attn_vec_src: &'static str =
            include_str!("shaders/flash_attn_vec.metal");
        sources.insert("flash_attn_vec_dk256".into(), flash_attn_vec_src);
        sources.insert("flash_attn_vec_dk512".into(), flash_attn_vec_src);
        sources.insert("flash_attn_vec_reduce_dk256".into(), flash_attn_vec_src);
        sources.insert("flash_attn_vec_reduce_dk512".into(), flash_attn_vec_src);
        // F16 KV variants (Phase 4a)
        sources.insert("flash_attn_vec_f16kv_dk256".into(), flash_attn_vec_src);
        sources.insert("flash_attn_vec_f16kv_dk512".into(), flash_attn_vec_src);

        // RoPE, normalization, activation kernels (Story 1.4)
        let rope_src: &'static str = include_str!("shaders/rope.metal");
        sources.insert("rope_f32".into(), rope_src);
        sources.insert("rope_f16".into(), rope_src);
        sources.insert("rope_bf16".into(), rope_src);
        sources.insert("rope_neox_bf16".into(), rope_src);
        sources.insert("rope_neox_f32".into(), rope_src);
        let rms_norm_src: &'static str = include_str!("shaders/rms_norm.metal");
        sources.insert("rms_norm_f32".into(), rms_norm_src);
        sources.insert("rms_norm_f16".into(), rms_norm_src);
        sources.insert("rms_norm_bf16".into(), rms_norm_src);
        sources.insert("rms_norm_no_scale_bf16".into(), rms_norm_src);
        sources.insert("rms_norm_no_scale_f32".into(), rms_norm_src);
        // Fused RMS norm + elementwise multiply kernels (Phase 4e.2)
        sources.insert("rms_norm_mul_f32".into(), rms_norm_src);
        sources.insert("rms_norm_mul_f16".into(), rms_norm_src);
        sources.insert("rms_norm_mul_bf16".into(), rms_norm_src);
        let gelu_src: &'static str = include_str!("shaders/gelu.metal");
        sources.insert("gelu_f32".into(), gelu_src);
        sources.insert("gelu_f16".into(), gelu_src);
        sources.insert("gelu_bf16".into(), gelu_src);
        let softmax_src: &'static str = include_str!("shaders/softmax.metal");
        sources.insert("softmax_f32".into(), softmax_src);
        sources.insert("softmax_f16".into(), softmax_src);
        sources.insert("softmax_bf16".into(), softmax_src);
        let softcap_src: &'static str = include_str!("shaders/softcap.metal");
        sources.insert("softcap_f32".into(), softcap_src);
        sources.insert("softcap_f16".into(), softcap_src);
        sources.insert("softcap_bf16".into(), softcap_src);

        // Fused norm-add kernels — Gemma4 post-attention / post-FFN ordering:
        //   normed = rms_norm(input, weight, eps);  output = residual + normed
        let fused_norm_add_src: &'static str =
            include_str!("shaders/fused_norm_add_bf16.metal");
        sources.insert("fused_norm_add_bf16".into(), fused_norm_add_src);
        sources.insert("fused_norm_add_no_weight_bf16".into(), fused_norm_add_src);

        // Fused head-norm + RoPE f32 kernel — replaces separate rms_norm + rope_neox_f32
        let fused_hnr_f32_src: &'static str =
            include_str!("shaders/fused_head_norm_rope_f32.metal");
        sources.insert("fused_head_norm_rope_f32".into(), fused_hnr_f32_src);

        // Fused norm-add f32 kernels — post-attention / post-FFN / end-of-layer
        let fused_norm_add_f32_src: &'static str =
            include_str!("shaders/fused_norm_add_f32.metal");
        sources.insert("fused_norm_add_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_residual_norm_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_residual_norm_scalar_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_moe_routing_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_norm_add_scalar_f32".into(), fused_norm_add_f32_src);

        // Argsort kernel (Story 2.3) — MoE top-K routing
        let argsort_src: &'static str = include_str!("shaders/argsort.metal");
        sources.insert("argsort_desc_f32".into(), argsort_src);

        // Gather / index_select kernel (Story 2.4)
        let gather_src: &'static str = include_str!("shaders/gather.metal");
        sources.insert("gather_f32".into(), gather_src);

        // F32 KV cache copy kernel (Session merge S1+S2)
        let kv_cache_copy_src: &'static str =
            include_str!("shaders/kv_cache_copy.metal");
        sources.insert("kv_cache_copy".into(), kv_cache_copy_src);
        sources.insert("kv_cache_copy_f32".into(), kv_cache_copy_src);

        // Strided copy kernel (Story 2.5)
        let copy_src: &'static str = include_str!("shaders/copy.metal");
        sources.insert("strided_copy_f32".into(), copy_src);

        // Dense F16 GEMM kernel (Story 2.6) — lm_head projection
        let dense_gemm_src: &'static str = include_str!("shaders/dense_gemm.metal");
        sources.insert("dense_gemm_f16".into(), dense_gemm_src);
        sources.insert("dense_matvec_f16".into(), dense_gemm_src);
        sources.insert("dense_matvec_f16w_f32io".into(), dense_gemm_src);

        // Standalone FWHT for TurboQuant pre/post-rotation (SIMD shuffle, zero barriers)
        let fwht_src: &'static str = include_str!("shaders/fwht_standalone.metal");
        sources.insert("fwht_standalone_f32_d256".into(), fwht_src);
        sources.insert("fwht_standalone_f32_d512".into(), fwht_src);

        // Fast Hadamard quantize (SIMD shuffle, zero barriers)
        let hq_fast_src: &'static str = include_str!("shaders/hadamard_quantize_kv_fast.metal");
        sources.insert("hadamard_quantize_kv_fast_d256".into(), hq_fast_src);
        sources.insert("hadamard_quantize_kv_fast_d512".into(), hq_fast_src);

        // GPU sampling kernels — eliminate logits readback (Phase 6)
        let argmax_src: &'static str = include_str!("shaders/argmax.metal");
        sources.insert("argmax_f32".into(), argmax_src);
        let softmax_sample_src: &'static str =
            include_str!("shaders/softmax_sample.metal");
        sources.insert("softmax_sample_f32".into(), softmax_sample_src);

        Self {
            cache: HashMap::new(),
            sources,
        }
    }

    /// Register a shader source at runtime (useful for testing and dynamic
    /// kernel generation).
    pub fn register_source(&mut self, name: impl Into<String>, source: &'static str) {
        let name = name.into();
        // Invalidate any cached pipeline for this name since the source changed.
        self.cache.remove(&name);
        self.sources.insert(name, source);
    }

    /// Get a compiled compute pipeline for the named kernel function.
    ///
    /// On first call for a given name, this compiles the MSL source into a
    /// Metal library, extracts the named function, and creates a
    /// `ComputePipelineState`.  Subsequent calls return the cached pipeline.
    ///
    /// # Errors
    ///
    /// * `MlxError::KernelNotFound` — no source registered for this name.
    /// * `MlxError::ShaderCompilationError` — MSL compilation or pipeline
    ///   creation failed.
    pub fn get_pipeline(
        &mut self,
        name: &str,
        device: &metal::DeviceRef,
    ) -> Result<&ComputePipelineState> {
        if !self.cache.contains_key(name) {
            // Slow path: compile the shader.
            let source = self.sources.get(name).ok_or_else(|| {
                MlxError::KernelNotFound(name.to_string())
            })?;

            let compile_opts = metal::CompileOptions::new();
            let library = device
                .new_library_with_source(source, &compile_opts)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            let function = library
                .get_function(name, None)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            self.cache.insert(name.to_string(), pipeline);
        }

        // At this point the pipeline is guaranteed to be in the cache.
        // We use `ok_or_else` instead of `expect` to satisfy the no-panic policy.
        self.cache.get(name).ok_or_else(|| {
            MlxError::KernelNotFound(name.to_string())
        })
    }

    /// Check if a pipeline for the given name is already compiled and cached.
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }

    /// Number of compiled pipelines currently in the cache.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Number of registered shader sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
