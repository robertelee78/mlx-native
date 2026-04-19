//! [`KernelRegistry`] — lazy compilation and caching of Metal compute pipelines.
//!
//! MSL shader source is embedded at compile time via `include_str!`.  On first
//! access, the source is compiled into a Metal library, the named function is
//! extracted, and a `ComputePipelineState` is created and cached.  Subsequent
//! calls return the cached pipeline.

use std::collections::HashMap;

use metal::{ComputePipelineState, FunctionConstantValues, MTLDataType};

use crate::error::{MlxError, Result};

// MTLDataType numeric values (from metal-rs argument.rs, confirmed in Apple Metal spec):
//   Int  = 29
//   Bool = 53
// These are used when calling set_constant_value_at_index so the Metal runtime
// knows how wide each constant value is.

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

        // GGML block-format quantized matrix-matrix kernels
        // (ADR-011 Phase 3 Wave P3a: port of llama.cpp's kernel_mul_mm_<q>_f32).
        // Used at prefill m > 8 to reuse each weight tile across a 32-row
        // block via threadgroup-staged simdgroup MMA, instead of re-reading
        // every block per prompt-token as the mv kernel does.
        let ggml_mm_src: &'static str =
            include_str!("shaders/quantized_matmul_mm.metal");
        sources.insert("kernel_mul_mm_q4_0_f32".into(), ggml_mm_src);
        sources.insert("kernel_mul_mm_q8_0_f32".into(), ggml_mm_src);
        sources.insert("kernel_mul_mm_q6_K_f32".into(), ggml_mm_src);

        // GGML block-format quantized matrix-matrix kernels — tensor API
        // variant (ADR-011 Phase 3 Wave P3b-tensor: port of llama.cpp's
        // kernel_mul_mm_impl `#ifdef GGML_METAL_HAS_TENSOR` branch).
        // Uses Apple's MetalPerformancePrimitives `tensor_ops::matmul2d`
        // primitive which on M3+ dispatches to hardware tensor cores for
        // 2-3x the effective FLOP throughput vs the simdgroup MMA path.
        // Only compiled on devices where the tensor API is available; the
        // kernel_registry's runtime-probe (see MlxDevice::has_tensor) gates
        // compilation so non-tensor devices transparently fall back to the
        // non-tensor `kernel_mul_mm_<q>_f32` kernels.
        let ggml_mm_tensor_src: &'static str =
            include_str!("shaders/quantized_matmul_mm_tensor.metal");
        sources.insert("kernel_mul_mm_q4_0_tensor_f32".into(), ggml_mm_tensor_src);
        sources.insert("kernel_mul_mm_q8_0_tensor_f32".into(), ggml_mm_tensor_src);
        sources.insert("kernel_mul_mm_q6_K_tensor_f32".into(), ggml_mm_tensor_src);

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

        // Expert-routed (MoE) GGML block-format QUANTIZED MATRIX-MATRIX kernels
        // (ADR-011 Phase 3 Wave P3a: port of llama.cpp's
        // `kernel_mul_mm_id_map0_ne20_N` + `kernel_mul_mm_id_<q>_f32`).
        // Two-stage dispatch: map0 regroups the token-to-expert table into
        // per-expert routed-token lists, then mm_id stages a 64x32 expert
        // weight tile into threadgroup shmem and reuses it across a 32-row
        // block of that expert's routed tokens.
        let ggml_id_mm_src: &'static str =
            include_str!("shaders/quantized_matmul_id_mm.metal");
        sources.insert("kernel_mul_mm_id_map0_ne20_1".into(), ggml_id_mm_src);
        sources.insert("kernel_mul_mm_id_map0_ne20_8".into(), ggml_id_mm_src);
        sources.insert("kernel_mul_mm_id_q4_0_f32".into(), ggml_id_mm_src);
        sources.insert("kernel_mul_mm_id_q8_0_f32".into(), ggml_id_mm_src);
        sources.insert("kernel_mul_mm_id_q6_K_f32".into(), ggml_id_mm_src);

        // MoE-routed quantized matrix-matrix kernels — tensor API variant
        // (ADR-011 Phase 3 Wave P3b-tensor).  Uses the MPP tensor_ops
        // matmul2d primitive for hardware-tensor-core MMA on M3+.  Only
        // the mm_id kernel is ported — map0 is a short pre-pass (not
        // matmul) and continues to use the simdgroup version.
        let ggml_id_mm_tensor_src: &'static str =
            include_str!("shaders/quantized_matmul_id_mm_tensor.metal");
        sources.insert("kernel_mul_mm_id_q4_0_tensor_f32".into(), ggml_id_mm_tensor_src);
        sources.insert("kernel_mul_mm_id_q8_0_tensor_f32".into(), ggml_id_mm_tensor_src);
        sources.insert("kernel_mul_mm_id_q6_K_tensor_f32".into(), ggml_id_mm_tensor_src);

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
        sources.insert("moe_swiglu_seq".into(), moe_dispatch_src);
        sources.insert("moe_accumulate".into(), moe_dispatch_src);
        sources.insert("moe_weighted_sum".into(), moe_dispatch_src);
        sources.insert("moe_weighted_sum_seq".into(), moe_dispatch_src);
        sources.insert("zero_buffer".into(), moe_dispatch_src);
        sources.insert("naive_matvec_f32".into(), moe_dispatch_src);
        sources.insert("moe_gather_topk_weights".into(), moe_dispatch_src);
        // bf16 variants (Phase 2 bf16 activation path)
        sources.insert("fused_gelu_mul_bf16".into(), moe_dispatch_src);
        sources.insert("moe_swiglu_seq_bf16".into(), moe_dispatch_src);
        sources.insert("moe_weighted_sum_seq_bf16_input".into(), moe_dispatch_src);

        // Batched KV cache copy kernels
        let kv_cache_src: &'static str = include_str!("shaders/kv_cache_copy.metal");
        sources.insert("kv_cache_copy_batch_f32".into(), kv_cache_src);
        sources.insert("kv_cache_copy_batch_f32_to_f16".into(), kv_cache_src);
        sources.insert("kv_cache_copy_seq_f32".into(), kv_cache_src);
        sources.insert("kv_cache_copy_seq_f32_to_f16".into(), kv_cache_src);
        // bf16-source KV cache copy (Phase 2 bf16 activation path)
        sources.insert("kv_cache_copy_seq_bf16".into(), kv_cache_src);

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
        sources.insert("embedding_gather_scale_batch_f32".into(), elementwise_src);
        sources.insert("permute_021_bf16".into(), elementwise_src);
        sources.insert("permute_021_f32".into(), elementwise_src);
        sources.insert("transpose_2d_f32".into(), elementwise_src);
        sources.insert("transpose_2d_f16".into(), elementwise_src);

        // Attention kernels (Story 1.3)
        let sdpa_src: &'static str = include_str!("shaders/sdpa.metal");
        sources.insert("sdpa".into(), sdpa_src);
        sources.insert("sdpa_bf16".into(), sdpa_src);
        let sdpa_sliding_src: &'static str = include_str!("shaders/sdpa_sliding.metal");
        sources.insert("sdpa_sliding".into(), sdpa_sliding_src);
        sources.insert("sdpa_sliding_bf16".into(), sdpa_sliding_src);

        // Flash-attention tiled prefill kernel (ADR-011 Phase 1).
        // Ten entry points; all backed by the same shader source.
        // Pipelines are compiled with function constants via
        // `get_pipeline_with_bool_constants` — not `get_pipeline`.
        let flash_attn_prefill_src: &'static str =
            include_str!("shaders/flash_attn_prefill.metal");
        // D=256 variants (BQ=32, BK=16, WM=4, WN=1 — 128 threads/threadgroup)
        sources.insert(
            "steel_attention_float32_bq32_bk16_bd256_wm4_wn1_maskfloat32".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_float32_bq32_bk16_bd256_wm4_wn1_maskbool_".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbool_".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_float16_bq32_bk16_bd256_wm4_wn1_maskfloat16".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_float16_bq32_bk16_bd256_wm4_wn1_maskbool_".into(),
            flash_attn_prefill_src,
        );
        // D=512 variants (BQ=8, BK=8, WM=1, WN=1 — 32 threads/threadgroup)
        // NOTE: f32 at D=512 is NOT instantiated — threadgroup memory exceeds
        // the 32 KB Metal limit (candle sdpa.rs:86-94).
        sources.insert(
            "steel_attention_bfloat16_bq8_bk8_bd512_wm1_wn1_maskbfloat16".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_bfloat16_bq8_bk8_bd512_wm1_wn1_maskbool_".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_float16_bq8_bk8_bd512_wm1_wn1_maskfloat16".into(),
            flash_attn_prefill_src,
        );
        sources.insert(
            "steel_attention_float16_bq8_bk8_bd512_wm1_wn1_maskbool_".into(),
            flash_attn_prefill_src,
        );

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
        sources.insert("rms_norm_no_scale_f32_dual".into(), rms_norm_src);
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

        // Fused head-norm + RoPE bf16 kernels (single-token + batch prefill)
        // Both entry points live in the same .metal file.
        let fused_hnr_bf16_src: &'static str =
            include_str!("shaders/fused_head_norm_rope_bf16.metal");
        sources.insert("fused_head_norm_rope_bf16".into(), fused_hnr_bf16_src);
        sources.insert("fused_head_norm_rope_batch_bf16".into(), fused_hnr_bf16_src);

        // Fused norm-add f32 kernels — post-attention / post-FFN / end-of-layer
        let fused_norm_add_f32_src: &'static str =
            include_str!("shaders/fused_norm_add_f32.metal");
        sources.insert("fused_norm_add_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_residual_norm_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_residual_norm_scalar_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_moe_routing_f32".into(), fused_norm_add_f32_src);
        sources.insert("fused_moe_routing_batch_f32".into(), fused_norm_add_f32_src);
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
        sources.insert("offset_copy_f32".into(), copy_src);

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
        // Top-K kernel for Q8 rerank: avoids full-logits readback.
        let top_k_src: &'static str = include_str!("shaders/top_k.metal");
        sources.insert("top_k_f32".into(), top_k_src);

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

    /// Get a compiled compute pipeline for the named kernel, specialized with
    /// Metal function constants (both bool and i32 in one call).
    ///
    /// `bool_constants` contains `(index, value)` pairs mapping to
    /// `[[function_constant(index)]]` bool declarations in the MSL shader.
    /// `int_constants` contains `(index, value)` pairs mapping to
    /// `[[function_constant(index)]]` int (int32_t) declarations in the MSL
    /// shader.
    ///
    /// Pipelines are cached by a composite key:
    /// `"<name>|<index>:b<0|1>|...|<index>:i<value>|..."`.  The 'b' prefix
    /// marks bool entries and the 'i' prefix marks i32 entries, making the
    /// format unambiguous regardless of constant ordering.  Distinct
    /// `(name, constants)` combinations each compile to a separate pipeline;
    /// the slow compilation path runs at most once per unique combination.
    ///
    /// # Errors
    ///
    /// * `MlxError::KernelNotFound` — no source registered for this name.
    /// * `MlxError::ShaderCompilationError` — MSL compilation, function
    ///   specialisation, or pipeline creation failed.
    pub fn get_pipeline_with_constants(
        &mut self,
        name: &str,
        device: &metal::DeviceRef,
        bool_constants: &[(usize, bool)],
        int_constants: &[(usize, i32)],
    ) -> Result<&ComputePipelineState> {
        // Build a composite cache key so distinct constant combinations each
        // compile to their own pipeline.  Bool entries use the 'b' type marker
        // and i32 entries use 'i'; this prevents a collision between, e.g.,
        // bool index 5 value 1 and int index 5 value 1.
        let mut cache_key = name.to_string();
        for &(index, value) in bool_constants {
            cache_key.push('|');
            cache_key.push_str(&index.to_string());
            cache_key.push_str(if value { ":b1" } else { ":b0" });
        }
        for &(index, value) in int_constants {
            cache_key.push('|');
            cache_key.push_str(&index.to_string());
            cache_key.push(':');
            cache_key.push('i');
            cache_key.push_str(&value.to_string());
        }

        if !self.cache.contains_key(&cache_key) {
            // Slow path: compile the shader with function constant specialisation.
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

            // Build the FunctionConstantValues object with all bool and i32
            // constants.  Metal's set_constant_value_at_index reads the value
            // through a raw pointer; the pointed-to bytes must match the size
            // declared in the MSL shader (1 byte for bool, 4 bytes for int).
            let fcv = FunctionConstantValues::new();

            for &(index, value) in bool_constants {
                // MTLDataType::Bool = 53 (metal-rs argument.rs).
                // The Metal runtime reads it as an Objective-C BOOL (uint8_t).
                let v: u8 = if value { 1 } else { 0 };
                fcv.set_constant_value_at_index(
                    (&v as *const u8).cast::<std::ffi::c_void>(),
                    MTLDataType::Bool,
                    index as u64,
                );
            }

            for &(index, value) in int_constants {
                // MTLDataType::Int = 29 (metal-rs argument.rs).
                // The Metal runtime reads 4 bytes as a signed 32-bit integer,
                // matching the Metal shader type `constant int`.
                fcv.set_constant_value_at_index(
                    (&value as *const i32).cast::<std::ffi::c_void>(),
                    MTLDataType::Int,
                    index as u64,
                );
            }

            let function = library
                .get_function(name, Some(fcv))
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

            self.cache.insert(cache_key.clone(), pipeline);
        }

        self.cache.get(&cache_key).ok_or_else(|| {
            MlxError::KernelNotFound(name.to_string())
        })
    }

    /// Get a compiled compute pipeline for the named kernel, specialized with
    /// Metal bool function constants.
    ///
    /// The `bool_constants` slice contains `(index, value)` pairs.  Each pair
    /// maps to a `[[function_constant(index)]]` declaration in the MSL shader.
    ///
    /// This is a thin wrapper around [`get_pipeline_with_constants`] that
    /// passes an empty `int_constants` slice.  Existing callers continue to
    /// work without modification; the cache-key format for pure-bool pipelines
    /// is compatible (bool entries carry the 'b' type marker, which is the
    /// only format ever written by this wrapper).
    ///
    /// # Errors
    ///
    /// * `MlxError::KernelNotFound` — no source registered for this name.
    /// * `MlxError::ShaderCompilationError` — MSL compilation, function
    ///   specialisation, or pipeline creation failed.
    pub fn get_pipeline_with_bool_constants(
        &mut self,
        name: &str,
        device: &metal::DeviceRef,
        bool_constants: &[(usize, bool)],
    ) -> Result<&ComputePipelineState> {
        self.get_pipeline_with_constants(name, device, bool_constants, &[])
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal Metal shader that uses a single int function constant.
    ///
    /// The kernel writes the constant value N into the first element of the
    /// output buffer, allowing the test to verify that the Metal compiler
    /// actually sees distinct specialisations for N=4 and N=8.
    ///
    /// The shader is intentionally trivial — we only need it to *compile* with
    /// an int function constant; correctness of the kernel logic is not under
    /// test here.
    const INT_FC_TEST_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant int test_N [[function_constant(100)]];

kernel void int_fc_test_kernel(
    device int* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        out[0] = test_N;
    }
}
"#;

    /// Verify that `get_pipeline_with_constants` produces distinct cached
    /// pipelines for different i32 function-constant values, and that
    /// `get_pipeline_with_bool_constants` (the backward-compat wrapper) still
    /// works correctly with the new 'b'-prefixed cache-key format.
    ///
    /// This test requires a real Metal device and is therefore marked
    /// `#[ignore]` on non-Apple platforms, but runs unconditionally on macOS.
    #[test]
    fn test_int_fc_distinct_pipelines_and_bool_compat() {
        let device = metal::Device::system_default()
            .expect("no Metal device — run on Apple Silicon or x86 Mac with Metal support");

        let mut registry = KernelRegistry::new();

        // Register the inline test shader under a name that cannot collide with
        // any production kernel.
        registry.register_source("int_fc_test_kernel", INT_FC_TEST_SHADER);

        // Compile with N=4.
        let p4_ptr = registry
            .get_pipeline_with_constants(
                "int_fc_test_kernel",
                &device,
                &[],                  // no bool constants
                &[(100, 4_i32)],      // int constant index 100 = 4
            )
            .expect("pipeline N=4 should compile") as *const _;

        // Cache must now have exactly 1 entry for this kernel.
        // (Other production kernels may already be in cache from new(); here
        // we check that the N=4 key was inserted.)
        let count_after_n4 = registry.cached_count();

        // Compile with N=8 — must produce a SEPARATE pipeline.
        let p8_ptr = registry
            .get_pipeline_with_constants(
                "int_fc_test_kernel",
                &device,
                &[],
                &[(100, 8_i32)],
            )
            .expect("pipeline N=8 should compile") as *const _;

        // Cache must have grown by exactly 1.
        assert_eq!(
            registry.cached_count(),
            count_after_n4 + 1,
            "N=8 must produce a new cache entry"
        );

        // The two pipelines must be distinct objects in the cache.
        assert_ne!(
            p4_ptr, p8_ptr,
            "N=4 and N=8 specialisations must be separate ComputePipelineState objects"
        );

        // A second call with N=4 must return the SAME pipeline (cache hit, no
        // new compilation).
        let p4_again_ptr = registry
            .get_pipeline_with_constants(
                "int_fc_test_kernel",
                &device,
                &[],
                &[(100, 4_i32)],
            )
            .expect("pipeline N=4 cache hit should succeed") as *const _;

        assert_eq!(
            registry.cached_count(),
            count_after_n4 + 1,
            "repeated N=4 call must be a cache hit, not a new entry"
        );
        assert_eq!(
            p4_ptr, p4_again_ptr,
            "repeated N=4 call must return the same pipeline pointer"
        );

        // Verify backward compatibility: get_pipeline_with_bool_constants must
        // still route through get_pipeline_with_constants and produce a cached
        // pipeline without panicking.
        //
        // We register a separate bool-constant shader that does NOT use a bool
        // function constant (so the Metal compiler ignores missing FCs for
        // this trivial case) — but the call path and cache-key format are what
        // matter here.  We reuse the int_fc_test_kernel source; the bool FC is
        // simply unused by the shader (Metal allows unused FCs when the shader
        // declares them with `function_constant` but the value is never read).
        //
        // To avoid a Metal compiler error for an undeclared function constant,
        // we register a separate bare-kernel shader for the bool wrapper test.
        const BARE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bare_kernel(device int* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) { out[0] = 42; }
}
"#;
        registry.register_source("bare_kernel", BARE_SHADER);

        let count_before_bool = registry.cached_count();
        let _bool_pipeline = registry
            .get_pipeline_with_bool_constants("bare_kernel", &device, &[])
            .expect("bool-constants wrapper with empty slice must succeed");

        assert_eq!(
            registry.cached_count(),
            count_before_bool + 1,
            "bool-constants wrapper must insert one new cache entry"
        );
    }
}
