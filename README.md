# mlx-native

[![Crates.io](https://img.shields.io/crates/v/mlx-native.svg)](https://crates.io/crates/mlx-native)
[![docs.rs](https://docs.rs/mlx-native/badge.svg)](https://docs.rs/mlx-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pure-Rust Metal GPU compute library for transformer inference on Apple Silicon. Built as the GPU backend for the [hf2q](https://github.com/robertelee78/hf2q) inference engine.

## When to use this

mlx-native is the right tool when **all** of these hold:

- You're running transformer (or Mamba / Gated DeltaNet) **inference on Apple Silicon**
- Your weights are **GGUF, MLX-quant, or safetensors** (no PyTorch checkpoints, no ONNX)
- You want **low Metal decode latency** and are willing to drive a kernel-dispatch API
- You're fine assembling the forward pass yourself — there is no `Tensor` type, no `Module` system, no model zoo

Reach for **[candle](https://github.com/huggingface/candle)** instead if you need autograd / training, multi-backend support (CUDA / CPU / WASM), Python bindings, ONNX import, a built-in model zoo, or a high-level tensor algebra surface. The two are complementary: candle is "PyTorch-shaped Rust ML framework," mlx-native is the Metal compute backend of a llama.cpp-shaped inference engine.

### What we do that candle's Metal backend doesn't

- **One `MTLCommandBuffer` + one `ComputeCommandEncoder` per forward pass** (`GraphExecutor` / `GraphSession`) — candle's Metal backend defaults to flushing into a fresh command buffer every 50 dispatches (`CANDLE_METAL_COMPUTE_PER_BUFFER`, verified in `candle-metal-kernels/src/metal/commands.rs`)
- **TurboQuant KV cache** — Lloyd-Max codebooks (2 / 3 / 4-bit nibble-packed) and byte-packed higher-bit (5 / 6 / 8-bit) variants, with fused Hadamard incoherence transform
- **MoE routing on GPU** — `moe_gate` + `moe_softmax_topk` + V3 parallel SG-tournament top-K (`fused_moe_routing_f32`, default-on) + expert-routed quantized matmul (no CPU round-trip for top-k expert selection)
- **Custom Metal kernels for state-space models** — `gated_delta_net` (+ chunk variants), `ssm_conv`, `ssm_norm_gate`, `tri_solve`, `cumsum`
- **Shape-specialized prefill** — D=256 / D=512 tiled flash-attention kernels (bf16 + F16 / BF16 `_resume` dispatchers) tuned for production model shapes (Qwen3, Gemma 4)
- **Fused norm-family kernels** — `fused_norm_add`, `fused_residual_norm`, `fused_post_attn_triple_norm`, `fused_post_ff_norm2_endlayer` (+ `_v2` default-on), `fused_moe_wsum_norm_add`, `fused_moe_wsum_post_ff_norm2_endlayer_v2`, `fused_head_norm_rope` (+ batch family)
- **Precompiled `.metallib` at packaging time** — 112 shaders baked in; eliminates cold-start MSL compile (`MLX_PRECOMPILED_METALLIB` default-on; set `=0` to disable)
- **GPU-resident sampling** — `softmax_sample` eliminates the logits-to-CPU readback on the hot path
- **Sliding-window KV cache copy with ring wrap** — single GPU kernel instead of CPU-side index math
- **Explicit barrier control** — `session.barrier()` and `session.barrier_between(reads, writes)` for precise GPU sync between dependent ops

### Trade-offs to know going in

- **Apple Silicon only.** No CPU, no CUDA, no WASM. If you need to ship cross-platform, this is the wrong layer.
- **No autograd.** A growing set of backward + optimizer kernels exists — SiLU / RMSNorm / softmax / log / row-sum / embedding-scatter / exp / divide / sqrt / outer-product / conv1d-depthwise-causal / MoE-weighted-sum / MoE-SwiGLU backward, differentiable affine qdq, Adam step, and `flash_attn_train` (forward + backward through attention with dQ/dK/dV) — but you wire the training loop yourself; there is no `Var` / `VarMap` / autodiff / `Module` system.
- **GGML matmul coverage is the inference subset, not the full set.** Q2_K, Q3_K, Q4_0, Q8_0, and Q6_K have dense mat-vec / mat-mat plus expert-routed variants; the production shapes also use tensor-core paths where implemented. Q4_K and Q5_K have dense mat-vec / mat-mat plus expert-routed (`mm_id`) variants. Q5_1 and IQ4_NL have dense and expert-routed variants. Q4_1, Q5_0, Q8_1, and Q8_K are not supported in the Metal matmul path. MLX-format affine quantization supports 4 / 6 / 8-bit (no 3-bit).
- **No high-level model code.** This is a kernel library; the consumer (e.g. hf2q) builds the actual transformer forward pass.

## Status

**Active development, pre-1.0.** API may change between minor versions (`0.x.0 → 0.(x+1).0` signals breaking changes). Public functions and structs evolve as new model families are added. Patch versions (`0.x.y → 0.x.(y+1)`) are non-breaking.

Supported model families used in production:
- **Qwen3 / Qwen3.5 / Qwen3.6** (dense + MoE, GGUF)
- **Qwen3-VL** (vision-tower kernels: im2col / bilinear-resize / 2×2 block-merge / feature-concat + `RopeMultiMode::Vision`)
- **Gemma 4** (dense, with SWA + softcap, GQA)
- **DeepSeek V4 Flash 0731** (compressed sparse attention + MoE kernels)
- **BERT-style** embeddings (bge-small-en-v1.5)
- Generic transformer kernels for custom architectures

## What is this?

A thin, safe wrapper around Apple's Metal framework focused on compute shader dispatch for neural network inference. It handles buffer management, MSL shader compilation, and GPU command encoding so callers can focus on graph construction and execution.

**Apple Silicon only** — leverages unified memory (`StorageModeShared`) for zero-copy CPU↔GPU buffer access.

## Design principles

- **No panics** — all public APIs return `Result<T, MlxError>`
- **Zero-copy** — `StorageModeShared` buffers on Apple Silicon unified memory
- **Thread-safe** — `MlxDevice` and `MlxBuffer` are `Send + Sync`
- **Lazy compilation** — MSL shaders compiled on first use, then cached
- **Buffer pooling** — power-of-two arena allocator for reuse
- **Single-encoder graphs** — `GraphExecutor` batches all dispatches in a forward pass into one `MTLCommandBuffer` + one `ComputeCommandEncoder`. Measured ~13× lower per-dispatch wall vs per-op-encoder patterns at typical Q/K/V projection shape (Q5_K, n=4096, k=2816) in `bench_encoder_pattern_compare`; matches candle's default 50-per-buffer flush at production shapes

## Quick start

A Q4_0 GGUF mat-vec dispatch:

```rust
use mlx_native::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
    MlxDevice, KernelRegistry, DType,
};

let device = MlxDevice::new()?;
let mut registry = KernelRegistry::new();

let input  = device.alloc_buffer(k * 4, DType::F32, vec![k])?;              // f32 input
let weight = /* mmap GGUF Q4_0 blocks into an MlxBuffer */;
let output = device.alloc_buffer(n * 4, DType::F32, vec![n])?;

let mut enc = device.command_encoder()?;
quantized_matmul_ggml(
    &mut enc, &mut registry, &device,
    &input, &weight, &output,
    &GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q4_0,
    },
)?;
enc.commit_and_wait()?;
```

For multi-op forward passes, use `GraphExecutor` to batch all dispatches into a single command buffer with one GPU sync:

```rust
let executor = GraphExecutor::new(MlxDevice::new()?); // takes ownership
let mut session = executor.begin()?;

session.rms_norm(/* ... */)?;
session.barrier();                  // explicit barrier between dependent ops
session.quantized_matmul_ggml(/* ... */)?;
session.barrier();
session.flash_attn_vec(/* ... */)?;

session.finish()?;                  // one commit_and_wait for the whole pass
```

## Key types

| Type | Purpose |
|------|---------|
| `MlxDevice` | Metal device + command queue (entry point) |
| `MlxBuffer` | Typed Metal buffer with shape/dtype metadata + byte_offset slicing |
| `MlxBufferPool` | Arena allocator with power-of-two bucketing |
| `CommandEncoder` | Compute command submission (single dispatch path) |
| `KernelRegistry` | Lazy MSL compilation + pipeline cache |
| `GraphExecutor` / `GraphSession` | Single-encoder batched forward passes |
| `ComputeGraph` | Recorded graph IR (capture, fuse, replay) |
| `EncoderSession` | Per-stage encoder lifecycle with MTLSharedEvent stage-fences (ADR-019, `HF2Q_ENCODER_SESSION=1`) |
| `DispatchRecord` | Pre-baked dispatch metadata to skip per-call pipeline lookup + env-var reads (ADR-029) |
| `DType` | Element data type enum (F32, F16, BF16, U8/16/32, I32) |
| `MlxError` | Unified error type |
| `GgufFile` / `TensorInfo` | GGUF model file mmap + metadata |
| `SafetensorsFile` | Safetensors mmap + tensor loading |

Hot-path command-buffer and compute-encoder factories are scoped for pool-less
long-lived Rust workers. Command-buffer destruction is likewise scoped, and
compute-encoder labels are cleared immediately after `endEncoding` captures
their trace row. The isolated regression proves bounded command-buffer,
CFString, and autorelease-pool populations across repeated labeled sync and
async waves.
See [the command-buffer lifetime note](https://github.com/robertelee78/mlx-native/blob/main/docs/command-buffer-autorelease-lifetime-2026-08-08.md).

## GPU operations

### Attention
- `flash_attn_vec` — SIMD-vectorized decode-path SDPA (NWG-parallel, llama.cpp port)
- `flash_attn_vec_tq` / `flash_attn_vec_tq_hb` — TurboQuant-quantized KV variants (Lloyd-Max + Hadamard)
- `flash_attn_vec_hybrid` — F16-K + TQ-HB-V SDPA (memory savings without full KV quant cost)
- `flash_attn_vec_peer_port_f16` (+ `_nwg32` NWG=32 variant with reduce dispatcher) — verbatim peer kernel port for F16 decode
- `flash_attn_prefill` (D=256, D=512) — Tiled prefill with bf16 kernels, SWA mask, sentinel handling — plus F16/BF16 `_resume` dispatchers for restart from arbitrary `qL` offset
- `flash_attn_train` — forward + backward (`dQ` / `dK` / `dV` via FA-2 Algorithm 4) bf16 kernels at D=64 / D=256, the missing piece for transformer training on this backend
- `sdpa` / `sdpa_sliding` — Reference SDPA with optional sliding window; `do_causal` flag toggles causal vs bidirectional (DFlash drafter block-diffusion)
- `sdpa_decode` — Tiled decode-path SDPA with N_SG=4 simdgroups

### Matrix multiplication
- **GGUF formats**: Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q5_1, Q6_K, Q8_0, IQ4_NL, I16 — mat-vec + mul_mm kernels (peer-parity with llama.cpp inference subset; tensor-core paths where implemented)
- **GGUF expert-routed (`mm_id`)**: Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q5_1, Q6_K, Q8_0, IQ4_NL (top_k>1 MoE mat-vec + tensor-mm where implemented)
- **MLX format**: 4/6/8-bit affine quantization (`quantized_matmul`)
- **MLX fused dequant+matmul**: `qmm_affine_t_f32` + `qmm_affine_t_f32_tiled` (2.29× over non-tiled), simdgroup-MMA `qmm_affine_t_f32_simd` / `qmm_affine_simd4` variants, and packed-U32 `qmm_affine_t_packed_simd4_b4`
- **MoE expert-routed**: `quantized_matmul_id` / `_id_ggml` / `_id_into` (top_k=1 tensor-mm fast path; `_into` accepts caller-provided output buffer)
- **Dense BF16**: `dense_mm_bf16_tensor`, `dense_gemv_bf16_f32` (M=1 decode)
- **Dense F16**: `dense_gemm_f16`, `dense_matvec_f16`

### Normalization
- `rms_norm` — RMS normalization (f32 + triple-output variants)
- `l2_norm` — L2 normalization
- `fused_residual_norm` — RMS norm + residual add (`_f32`, `_bf16`, `_scalar_f32` variants)
- `fused_norm_add` — MoE weighted_sum + RMS norm + add (`_f32`, `_bf16`, `_no_weight_bf16`, `_scalar_f32` variants)
- `fused_post_attn_triple_norm_f32` (+ `_v2`) — post-attention norm trio
- `fused_post_ff_norm2_endlayer_f32` (+ `_v2` default-on) — post-FFN norm + residual at layer end
- `fused_moe_wsum_norm_add` / `_dnorm_add` — MoE weighted-sum + norm + add
- `fused_moe_wsum_post_ff_norm2_endlayer_f32_v2` — MoE-FFN end-of-layer fusion
- `fused_head_norm_rope` — Per-head RMS norm + RoPE (`_f32`, `_bf16`, plus `_batch_*` family for prefill including `_batch_f32_with_bf16` and `_batch_f32_with_bf16_f32_perm`)

### Activation & gating
- `gelu` — GeLU activation (F32, BF16)
- `silu_mul` — SwiGLU (SiLU + elementwise multiply)
- `fused_gelu_mul_bf16` — fused GeLU + elementwise multiply (bf16)
- `sigmoid_mul` — Sigmoid-gated multiply
- `softmax`, `softcap`, `scale_mask_softmax` — Softmax variants (float4-vectorized)
- `softmax_sample` — Sampling from logits

### Position encoding
- `rope` — Standard RoPE
- `rope_multi` — Multi-axis RoPE with IMROPE (Qwen3.5) and Vision (Qwen3-VL ViT 2D positions) modes

### MoE
- `moe_gate` — Gate logits → weights
- `moe_softmax_topk` — GPU softmax + top-k expert selection
- `fused_moe_routing_f32` / `_batch_f32` — V3 parallel SG-tournament top-K (default-on, +11.5% vs V2)
- `moe_dispatch` — Per-expert matvec sequence with proper barriers
- `moe_weighted_reduce` — Weighted sum across selected experts

### State-space (Mamba/Gated DeltaNet)
- `ssm_conv` — Depthwise causal 1D convolution + SiLU
- `ssm_norm_gate` — Norm + gate fusion (eliminates CPU bridge)
- `gated_delta_net` — Fused GDN kernel (decode)
- `gated_delta_net_chunk` / `_chunk_o` / `_kkt` / `_recompute_wu` — chunk-mode forward
- `chunk_gated_delta_rule` / `_tri_solve_invert` — chunk-rule decomposition with triangular inversion
- `compute_g_beta` — GDN g/beta computation
- `tri_solve` — Lower-triangular unit-diagonal forward substitution
- `cumsum` — Cumulative sum

### Memory & layout
- `kv_cache_copy` — Linear + sliding-window KV cache copy (with ring-wrap)
- `kv_cache_copy_seq_bf16` / `_seq_bf16_to_bf16_head_major` — BF16 sequence-batched cache copies (incl. head-major layout for prefill)
- `embedding` — Embedding lookup
- `gather` — Indexed gather (F16, nibble-packed)
- `transpose`, `permute_021` — Layout conversions
- `copy`, `offset_copy` — Strided copy
- `argmax`, `argsort`, `top_k` — Reductions

### Dispatch pre-bake (ADR-029)
Pre-baked `DispatchRecord` objects skip per-dispatch pipeline lookups, env-var reads, and parameter packing — meaningful on short-prompt decode hot paths.
- `build_q6k_nr2_m1_record` — dense Q6_K mv NR2 m=1
- `build_q6k_id_nr2_m1_record` — MoE Q6_K_ID NR2 m=1
- `build_q8_0_id_decode_record` — MoE Q8_0_ID regular decode
- `build_rms_norm_decode_record` — per-(dtype, rows, dim) RMSNorm decode

### Vision / ViT (Qwen3-VL prelude)
- `im2col_2d_3ch_f32` + `add_bias_row_2d_f32` — patch-embed helpers
- `bilinear_resize_2d_f32` — antialiased 2-D resize
- `block_merge_2x2_f32` — 2×2 spatial merge / permutation
- `feature_concat_f32` — strided channel-axis concat

### Hadamard / TurboQuant
- `hadamard` — Standalone FWHT (D=128/256/512)
- `hadamard_quantize_kv` — Fused Hadamard + KV quantization
- `tq_dequantize_kv` — TurboQuant KV dequantization

### Quantize / dequantize (qdq)
- `qdq_q4_0_f32`, `qdq_q8_0_f32` — GPU-side dequant for legacy GGUF blocks
- `qdq_affine_init_f32` / `qdq_affine_forward_f32` — MLX-format affine qdq with differentiable variants
- `qdq_affine_backward_scales_f32`, `qdq_affine_backward_biases_f32` — backward through quantization parameters

### Backward & training kernels
- `flash_attn_train_fwd_bf16_{d64,d256}` + `flash_attn_train_bwd_bf16_{d64,d256}` — attention forward (with logsumexp output) and backward (dQ / dK / dV via FA-2 Algorithm 4)
- `silu_backward_f32`, `softmax_backward`, `log_backward_f32`, `row_sum_backward_f32`, `exp_backward_f32`, `divide_backward_f32`, `sqrt_backward_f32`
- `rms_norm_compute_rms_inv` + `rms_norm_backward_dx` + `rms_norm_backward_dw`
- `outer_product` forward + backward
- `conv1d_depthwise_causal` forward + backward
- `take_along_axis` (gather + scatter-backward)
- `moe_weighted_sum_seq` backward; `moe_swiglu_seq` fused backward
- `embedding_lookup_f32` + `embedding_scatter_add_f32` (forward + scatter-add backward)
- `adam_update_f32` — fused Adam optimizer step (m / v moments + bias-correction)
- `slice_2d_cols_f32` + `copy_2d_cols_into_f32` — strided 2-D slice / scatter for column-major training layouts

## Weight loading

Load safetensors and GGUF models directly into Metal buffers via mmap:

```rust
use mlx_native::{MlxDevice, SafetensorsFile, GgufFile};

let device = MlxDevice::new()?;

// Safetensors — returns (dtype, shape, buffer)
use std::path::Path;
let st = SafetensorsFile::open(Path::new("model.safetensors"))?;
let (dtype, shape, buf) =
    st.load_tensor("model.layers.0.self_attn.q_proj.weight", &device)?;

// GGUF — raw block format passed through to GPU (no intermediate dequant)
let gguf = GgufFile::open(Path::new("model-Q4_K_M.gguf"))?;
for name in gguf.tensor_names() {
    let buf = gguf.load_tensor(name, &device)?;
    /* ... */
}
```

## Third-party licenses

This crate includes Metal kernels and dispatch code derived from:
- [candle](https://github.com/huggingface/candle) (Apache-2.0) — see `LICENSE-APACHE-candle`
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) — see `LICENSE-MIT-llamacpp`

Per-file attribution headers identify which kernels are derived from which upstream.

## License

MIT — see [LICENSE](LICENSE).
