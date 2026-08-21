# Peer Benchmarks — llama.cpp Reference Comparisons

This is the **single sanctioned home** for llama.cpp comparisons in this
repository. Code, comments, and other docs must not reference llama.cpp by
name — they say "the reference implementation" and, where a comparison
carries data, that data lives here. See `AGENTS.md` for the policy.

**Attribution**: portions of the kernel library are derived from
[llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) — the legal
notice ships as `LICENSE-MIT-llamacpp` at the repo root. Some quantized
kernels arrived via candle's vendored copies (`LICENSE-APACHE-candle`).

## Pinned reference commits

Comparisons and ports in this repo's history were made against specific
upstream commits. Where a doc or historical commit message mentions "the
pinned reference commit", it is one of:

| commit | context |
|--------|---------|
| `c8e03ce8122b7af76f836d53efde6df1ce5ec437` | dense-mm tensor-fallback hypothesis check (2026-08-05) |
| `15586e2d7165570fb3aa7c26e0d442e289ef69de` | residency-set keep-alive lifecycle match (2026-08-06) |
| `f9e832c10e9444cb168ddcb579cc62c154f3068b` | Q3_K kernel port |
| `da4495332` | ADR-029 iter-162 H93 peer-grounded Q-kernel port |
| `6ea215d17` | flash-attn / mat-vec peer ports |

## Structural diff — MoE matmul (from the 2026-04-27 W-5b.13 audit)

Side-by-side state of the mlx-native MoE matmul vs llama.cpp at audit time
(Qwen3.6 27B uses dense FFN and does not exercise the `_id` path):

| component | mlx-native | llama.cpp | status |
|-----------|------------|-----------|--------|
| Dense mm kernel | `kernel_mul_mm_q4_0_tensor_f32` (`quantized_matmul_mm_tensor.metal`) | `kernel_mul_mm_q4_0_f32` (`ggml-metal.metal:10104`) | **byte-identical** (mpp::tensor_ops::matmul2d, same tile geom, same dequant) |
| MoE `_id` mm kernel | `kernel_mul_mm_id_q4_0_f32` + `_tensor_f32` (`quantized_matmul_id_mm.metal`) | `kernel_mul_mm_id_q4_0_f32` (same file, lines 9708-9716) | **structurally aligned** (map0 + mm_id; mlx-native has ne20_1 + ne20_8 templates only, llama.cpp has 1/2/4/5/6/8/10/16/22) |
| Routing threshold (mv vs mm) | `MM_ROUTING_THRESHOLD = 8` | `ne11_mm_min = 8` (`ggml-metal-ops.cpp:2046`) | identical |
| Tensor-API probe | one-shot `OnceLock` (`probe_tensor_mm`) | `GGML_METAL_HAS_TENSOR` compile-time | both produce the same hot-path kernel name on M5 Max |
| Activation fusion (silu_mul) | NOT fused into mm kernel — separate `dispatch_silu_mul` | NOT fused into mm kernel — same shape | **parity** (both pay 1 silu_mul dispatch per layer) |
| Concurrent dispatch | `enc.memory_barrier()` between mm calls | `ggml_metal_op_concurrency_reset` | **parity** |

mlx-native's `_id` mm has a tighter top_k template instantiation set
(`ne20_1`, `ne20_8`) than llama.cpp's 9 variants (1/2/4/5/6/8/10/16/22).
Not a bottleneck for current production models (top_k=1 and top_k=8 are
covered); would matter if a top_k=2/4/5/6/10/16/22 model arrives.

## Wall-clock and throughput comparisons

- **DeepSeek prefill gate (2026-08-06)** — matched three-trial M5 Max gate,
  4,987-token prompt, 100.05 GiB hf2q-produced GGUF: hf2q medians
  **674.737 prompt tok/s / 33.908 decode tok/s** vs llama.cpp medians
  **672.914 / 31.643**. Every greedy transcript exact; zero prompt-cache
  hits; 60 s between trials.
- **Qwen3.6 27B DWQ46 pp4096 (2026-04-27 audit)** — wall-clock ratio vs
  llama.cpp was **4.34×**, attributed ~63% to hf2q wrapper/dispatch
  overhead and ~37% kernel-side; projected **3.4–3.7×** after mirroring the
  MoE-Q wrapper optimizations into the dense path.
- **Dense-mm tensor fallback (2026-08-05)** — tensor route 223 µs vs tiled
  fallback 294 µs (1.318×) on M5 Max; the fallback geometry matches
  llama.cpp's non-tensor baseline at commit `c8e03ce81`.
- **Dispatch cost** — llama.cpp's Metal backend implies ~0.14 µs/dispatch
  (~150 µs across ~1,070 dispatches); used as the calibration reference in
  `examples/dispatch_cost_calibration.rs`.

## Measurements retained namelessly in code (2026-08-20 sweep)

The reference sweep kept comparative measurements in code comments but
stripped the peer's name (comments now say "the peer" / "the reference").
The notable ones, with the peer identified, for the record:

- Flash-attn D=512 unroll sweep vs the peer's `MIN(DK8/2, 4*NSG)` formula:
  unroll(4)=34.30, (8)=34.11, (16)=35.85, (32, peer full unroll)=36.64
  ms/call at FA_GL@4K (`flash_attn_prefill_d512.metal`).
- `flash_attn_vec_hybrid`: estimated ~1.05× of llama.cpp per-dispatch at
  F16-K; 1.81× per-dispatch K-side gap measured.
- `flash_attn_vec_tq` FOR_UNROLL backport targeted a ~14 pp decode gap vs
  llama.cpp (gemma-26B-dwq cn=1: 0.86× → expected 0.91–0.94×).
- NWG=1 vec peer port falsified at tg5000: −25% vs llama.cpp
  (`flash_attn_vec_peer_port_f16_reduce.metal`).
- Peer decode dispatch overhead ~11.4 µs/dispatch (`bench_dispatch_overhead`);
  11.22 µs/dispatch FA=1 APEX-Q5_K_M (`test_quantized_matmul_id_mm`).
