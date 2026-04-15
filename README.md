# mlx-native

[![Crates.io](https://img.shields.io/crates/v/mlx-native.svg)](https://crates.io/crates/mlx-native)
[![docs.rs](https://docs.rs/mlx-native/badge.svg)](https://docs.rs/mlx-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pure-Rust Metal GPU compute library for MLX-compatible inference on Apple Silicon.

## What is this?

mlx-native provides a thin, safe wrapper around Apple's Metal framework focused on compute shader dispatch for neural network inference. It handles buffer management, shader compilation, and GPU command encoding so you can focus on model logic.

**Apple Silicon only** — requires a Mac with an M-series chip (or Intel Mac with discrete AMD GPU, though optimized for Apple Silicon unified memory).

## Features

- **No panics** — all public APIs return `Result<T, MlxError>`
- **Zero-copy** — `StorageModeShared` buffers on unified memory
- **Thread-safe** — `MlxDevice` and `MlxBuffer` are `Send + Sync`
- **Lazy compilation** — MSL shaders compiled on first use, then cached
- **Buffer pooling** — arena allocator with power-of-two bucketing for reuse
- **Compute graph** — record and replay GPU dispatch sequences

## Quick start

```rust
use mlx_native::{MlxDevice, DType};

let device = MlxDevice::new()?;
let buf = device.alloc_buffer(1024, DType::F32, vec![256])?;
let encoder = device.command_encoder()?;
```

## Key types

| Type | Purpose |
|------|---------|
| `MlxDevice` | Metal device + command queue (entry point) |
| `CommandEncoder` | Batched compute command submission |
| `MlxBuffer` | Typed Metal buffer with shape/dtype metadata |
| `MlxBufferPool` | Arena allocator with power-of-two bucketing |
| `KernelRegistry` | Lazy MSL compilation + pipeline cache |
| `ComputeGraph` | Recorded GPU dispatch sequence for replay |
| `DType` | Element data type enum |
| `MlxError` | Unified error type |

## GPU operations

mlx-native includes optimized Metal compute shaders for:

- Quantized matrix multiplication (MLX and GGML formats)
- Flash attention (scalar and TurboQuant variants, with sliding window)
- Fused RMSNorm + residual add
- Fused head norm + RoPE
- RoPE positional encoding
- Softmax and softcap
- GeLU activation
- Embedding lookup
- Mixture-of-experts gating and dispatch
- Hadamard transform (standalone and quantize-KV)
- Gather, transpose, copy, argmax, argsort

## Weight loading

Load safetensors and GGUF model files directly into Metal buffers:

```rust
use mlx_native::{MlxDevice, SafetensorsFile};

let device = MlxDevice::new()?;
let st = SafetensorsFile::open("model.safetensors")?;
let buffer = st.load_tensor(&device, "model.layers.0.self_attn.q_proj.weight")?;
```

## Third-party licenses

This crate includes code derived from:
- [candle](https://github.com/huggingface/candle) (Apache-2.0) — see `LICENSE-APACHE-candle`
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) — see `LICENSE-MIT-llamacpp`

## License

MIT — see [LICENSE](LICENSE).
