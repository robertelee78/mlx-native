# mlx-native

[![Crates.io](https://img.shields.io/crates/v/mlx-native.svg)](https://crates.io/crates/mlx-native)
[![docs.rs](https://docs.rs/mlx-native/badge.svg)](https://docs.rs/mlx-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pure-Rust Metal GPU compute library for transformer inference on Apple Silicon. Built as the GPU backend for the [hf2q](https://github.com/robertelee78/hf2q) inference engine.

## Status

**Active development, pre-1.0.** API may change between minor versions (`0.x.0 → 0.(x+1).0` signals breaking changes). Public functions and structs evolve as new model families are added. Patch versions (`0.x.y → 0.x.(y+1)`) are non-breaking.

Supported model families used in production:
- **Qwen3 / Qwen3.5 / Qwen3.6** (dense + MoE, GGUF)
- **Gemma 3 / Gemma 4** (dense, with SWA + softcap, GQA)
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
- **Single-encoder graphs** — `GraphExecutor` batches dispatches for ~120× lower per-token overhead than per-op encoders (matches the llama.cpp pattern)

## Quick start

A Q4_0 GGUF mat-vec dispatch:

```rust
use mlx_native::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
    MlxDevice, KernelRegistry, DType,
};

let device = MlxDevice::new()?;
let mut registry = KernelRegistry::new();

let input      = device.alloc_buffer(k * 4, DType::F32, vec![k])?;          // f32 input
let weight     = /* mmap GGUF Q4_0 blocks into an MlxBuffer */;
let mut output = device.alloc_buffer(n * 4, DType::F32, vec![n])?;

let mut enc = device.command_encoder()?;
quantized_matmul_ggml(
    &mut enc, &mut registry, &device,
    &input, &weight, &mut output,
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
| `DType` | Element data type enum (F32, F16, BF16, U8/16/32, I32) |
| `MlxError` | Unified error type |
| `GgufFile` / `TensorInfo` | GGUF model file mmap + metadata |
| `SafetensorsFile` | Safetensors mmap + tensor loading |

## GPU operations

### Attention
- `flash_attn_vec` — SIMD-vectorized decode-path SDPA (NWG-parallel, llama.cpp port)
- `flash_attn_vec_tq` / `flash_attn_vec_tq_hb` — TurboQuant-quantized KV variants (Lloyd-Max + Hadamard)
- `flash_attn_prefill` (D=256, D=512) — Tiled prefill with bf16 kernels, SWA mask, sentinel handling
- `sdpa` / `sdpa_sliding` — Reference SDPA with optional sliding window
- `sdpa_decode` — Tiled decode-path SDPA with N_SG=4 simdgroups

### Matrix multiplication
- **GGUF formats**: Q4_0, Q5_K, Q6_K, Q8_0, I16 — mat-vec + mul_mm tensor-core kernels
- **MLX format**: 4/6/8-bit affine quantization (`quantized_matmul`)
- **MoE expert-routed**: `quantized_matmul_id` / `_id_ggml` (top_k=1 tensor-mm fast path)
- **Dense BF16**: `dense_mm_bf16_tensor`, `dense_gemv_bf16_f32` (M=1 decode)
- **Dense F16**: `dense_gemm_f16`, `dense_matvec_f16`

### Normalization
- `rms_norm` — RMS normalization (f32 + triple-output variants)
- `l2_norm` — L2 normalization
- `fused_residual_norm` — RMS norm + residual add
- `fused_norm_add` — MoE weighted_sum + RMS norm + add
- `fused_head_norm_rope` — Per-head RMS norm + RoPE (with bf16 co-write variants)

### Activation & gating
- `gelu` — GeLU activation (F32, BF16)
- `silu_mul` — SwiGLU (SiLU + elementwise multiply)
- `sigmoid_mul` — Sigmoid-gated multiply
- `softmax`, `softcap`, `scale_mask_softmax` — Softmax variants
- `softmax_sample` — Sampling from logits

### Position encoding
- `rope` — Standard RoPE
- `rope_multi` — Multi-axis RoPE with IMROPE mode (Qwen3.5)

### MoE
- `moe_gate` — Gate logits → weights
- `moe_softmax_topk` — GPU softmax + top-k expert selection
- `moe_dispatch` — Per-expert matvec sequence with proper barriers
- `moe_weighted_reduce` — Weighted sum across selected experts

### State-space (Mamba/Gated DeltaNet)
- `ssm_conv` — Depthwise causal 1D convolution + SiLU
- `ssm_norm_gate` — Norm + gate fusion (eliminates CPU bridge)
- `gated_delta_net` — Fused GDN kernel
- `compute_g_beta` — GDN g/beta computation
- `tri_solve` — Lower-triangular unit-diagonal forward substitution
- `cumsum` — Cumulative sum

### Memory & layout
- `kv_cache_copy` — Linear + sliding-window KV cache copy (with ring-wrap)
- `embedding` — Embedding lookup
- `gather` — Indexed gather (F16, nibble-packed)
- `transpose`, `permute_021` — Layout conversions
- `copy`, `offset_copy` — Strided copy
- `argmax`, `argsort`, `top_k` — Reductions

### Hadamard / TurboQuant
- `hadamard` — Standalone FWHT (D=128/256/512)
- `hadamard_quantize_kv` — Fused Hadamard + KV quantization
- `tq_dequantize_kv` — TurboQuant KV dequantization

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
