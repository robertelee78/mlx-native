# Peer Benchmarks — llama.cpp Reference Comparisons

This is the **single sanctioned home** for llama.cpp comparisons in this
repository. Code, comments, and other docs must not reference llama.cpp by
name — they say "the reference implementation" and, where a comparison
carries data, that data lives here. See `AGENTS.md` for the policy.

**Attribution**: portions of the kernel library are derived from
[llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) — the legal
notice ships as `LICENSE-MIT-llamacpp` at the repo root. Some quantized
kernels arrived via candle's vendored copies (`LICENSE-APACHE-candle`). The
prepared GGUF projection (`gguf_repacked_mm.metal`), the staged AF4G64 draft
projection (`dflash2_af4g64_staged.metal`), the ring draft attention
(`dflash2_ring_attention.metal`) and the MoE expert live tiles
(`moe_expert_tiles.metal`) are derived from
[Splash](https://github.com/incoai/splash) commit `031a25a6` (Apache-2.0,
`LICENSE-APACHE-splash`); each shader's header lists its sources and changes.

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

- **Prepared AF4G64 DFlash projection (2026-11-11)** — downstream hf2q
  Qwen3.6 four-agent stage profile on M5 Max, identical checkpoint and
  scheduler geometry, every run `CalibrationPass` with proven cleanup. The
  accepted kernel stores 64-weight groups in 36 bytes, tiles 64 output rows,
  prepares an eight-row BF16 activation table once per projection, and applies
  affine scale/bias in FP32 after exact `128 + code` register-matrix products.
  These are topology-distorting stage diagnostics, not an end-to-end claim
  against Splash.

  | projection route | width-4 draft GPU median |
  |---|---:|
  | BF16 production baseline | 9.318 ms |
  | generic Q4_K experiment | 8.118 ms |
  | prepared AF4G64 register tile | 7.878 ms |

- **Row-exact few-row mat-vec (2026-11-06, `ops::row_exact_matvec`)** —
  `bench_row_exact_matvec`, M5 Max, median of 3 runs (each the median of 15
  command buffers of 16 serialized dispatches, GPU time), DeepSeek-V4
  decode shapes `[n x k]` at m = 1 / 2 / 3 / 4 input rows, warm weights,
  against the reference implementation's own kernels on the same shapes
  (its perf mode, warm; m = 1 its `mul_mv`, m = 2..4 its `mul_mv_ext`).
  Row-exact means each row is bitwise the m = 1 result, which the
  reference implementation's m-dependent kernel choice is not.

  | weight | shape | row-exact µs | reference µs |
  |---|---|---|---|
  | F16 | 24 x 16384 | 3.67 / 4.49 / 5.33 / 6.51 | 4.72 / 13.76 / 9.18 / 9.78 |
  | F16 | 4 x 16384 | 3.21 / 4.15 / 5.06 / 6.14 | 4.48 / 13.37 / 8.53 / 8.53 |
  | F16 | 256 x 4096 | 3.54 / 3.90 / 4.83 / 5.64 | 3.98 / 6.06 / 5.14 / 5.15 |
  | F16 | 1024 x 4096 | 6.28 / 7.16 / 8.24 / 9.83 | 5.97 / 7.94 / 9.37 / 9.77 |
  | F16 | 512 x 4096 | 4.54 / 5.23 / 5.84 / 6.73 | 4.68 / 7.10 / 6.44 / 6.70 |
  | F16 | 8192 x 1024 | 18.33 / 18.64 / 19.35 / 20.61 | 18.43 / 19.23 / 20.83 / 20.91 |
  | F16 | 64 x 4096 | 2.75 / 2.91 / 3.50 / 4.02 | 3.41 / 5.08 / 4.64 / 4.80 |
  | Q8_0 | 32768 x 1024 | 36.79 / 37.64 / 39.82 / 46.06 | 45.06 / 43.84 / 44.51 / 46.65 |
  | Q8_0 | 4096 x 8192 | 40.23 / 41.11 / 42.30 / 47.30 | 42.29 / 62.99 / 63.35 / 62.78 |
  | Q8_0 | 2048 x 4096 | 9.06 / 10.80 / 12.49 / 14.55 | 8.97 / 14.94 / 15.03 / 15.08 |
  | Q8_0 | 4096 x 2048 | 8.77 / 10.06 / 11.63 / 13.49 | 9.86 / 15.06 / 15.17 / 15.17 |
  | Q8_0 | 1024 x 4096 | 5.75 / 6.78 / 7.88 / 9.13 | 5.79 / 9.27 / 9.37 / 9.51 |
  | Q8_0 | 512 x 4096 | 4.09 / 4.80 / 6.05 / 6.95 | 4.47 / 6.12 / 6.33 / 6.17 |
  | Q8_0 | 129280 x 4096 (cold) | 990 / 1011 / 994 / 1019 | 970 / 1029 / 1048 / 1043 |
  | Q8_0 | 8 x (1024 x 4096) grouped | 39.39 / 40.29 / 41.91 / 47.34 | 38.83 / 47.84 / 56.92 / 60.22 |

- **MoE compact schedule builder (2026-11-07)** — model-free, M5 Max, 128
  experts × top-8, uniform distinct routing, Gemma prefill op microbench
  (median GPU µs at 458 → 1832 tokens, one hold): the reference
  implementation's `kernel_mul_mm_id_map0` (one threadgroup of `n_experts`
  threads, every thread scans every routed id in a count and a scatter pass)
  **78.6 → 279.3**; mlx-native's branch-free `map0c` with the same geometry
  **210 → 795**; the token-parallel `map0c` (one thread per token, lane-mask
  ranks) **9.9 → 19.9**, 14× under the reference at 1832 tokens, with an
  unchanged schedule. The schedule builder is no longer structurally aligned
  with the reference (structural diff table above).

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
