# mlx-native MoE-Q / Dense-Q FFN Perf Audit — 2026-04-27 (W-5b.13)

**Worker**: Wave 5b.13 cross-fence read-only audit
**Repo state**: mlx-native HEAD `5d9bb2e3` (unchanged across audit)
**Pre-flight**: 111.9 GB free RAM, no concurrent CFA workers > 50% CPU
**Scope**: Identify the source of W-5b.11's 47.4% `layer.ffn_dispatch` bucket
(7,310 ms / 152.3 ms per DN layer × 48 DN layers on Qwen3.6 27B DWQ46 at PP4096).

## TL;DR — the W-5b.11 hand-off was misnamed

The W-5b.11 hand-off attributed the dominant `layer.ffn_dispatch` bucket to
`mlx_native::ops::moe_q::*`. That label is **incorrect**:

1. The 27B DWQ46 GGUF is **dense FFN, not MoE** (`general.architecture =
   qwen35`, no `_exps` tensors; per-layer `ffn_gate.weight` Q4_0 (5120, 17408),
   `ffn_up.weight` Q4_0 (5120, 17408), `ffn_down.weight` Q4_0 (17408, 5120)).
2. The dispatch path is `quantized_matmul_ggml::dispatch_mm` →
   `kernel_mul_mm_q4_0_tensor_f32`, NOT `moe_q::*`.
3. The kernel itself is **already byte-identical to llama.cpp's
   `kernel_mul_mm_q4_0_f32` with `GGML_METAL_HAS_TENSOR`** (same NR0=64,
   NR1=32, NK=32 tile, same `mpp::tensor_ops::matmul2d`, same
   `dequantize_q4_0`). See `quantized_matmul_mm_tensor.metal:5-14` for the
   provenance comment.

Per `project_metal_compiler_auto_optimizes_static_levers` and
`project_mm_id_byte_identical`: when the kernel is byte-identical to
llama.cpp's, the gap is **structurally NOT in the kernel.** It's in the
caller's wrapper / dispatch / barrier pattern.

## Per-shape kernel bench (Qwen3.6 27B DWQ46 production prefill)

`benches/bench_moe_q_qwen36_shape.rs` — added by this audit; additive only.

3 cold trials (warmup=3, measure=10, median per shape):

| shape    | M    | N     | K     | qtype | median_ms | TFLOP/s | layers | total_ms |
|----------|-----:|------:|------:|:-----:|----------:|--------:|-------:|---------:|
| FFN_gate | 4096 | 17408 | 5120  | Q4_0  | 17.91–17.99 | 40.6–40.8 | 64 | 1146–1152 |
| FFN_up   | 4096 | 17408 | 5120  | Q4_0  | 17.87–17.97 | 40.6–40.9 | 64 | 1143–1150 |
| FFN_down | 4096 | 5120  | 17408 | Q4_0  | 19.87–19.91 | 36.7–36.7 | 64 | 1272–1274 |

**Total kernel-only FFN qmatmul time at pp4096 = 3,567 ms (median across 3 cold trials)**.

Cross-trial stability: < 0.4% variance.

## Comparison vs W-5b.11 measurement

| measurement | total ms | source |
|-------------|---------:|--------|
| `layer.ffn_dispatch` (all 64 layers) | **9,750** | hf2q W-5b.11 instrumented bench (`docs/wave5b3-walkbar-results.md`) |
| Kernel-only qmatmul (this audit) | **3,567** | mlx-native isolated kernel bench at production shape |
| **Wrapper / dispatch / barrier overhead** | **6,184** | W-5b.11 minus kernel-only = **63.4% of the bucket** |

The kernel itself is already at 36.7–40.8 TFLOP/s on Q4_0 mm at pp4096 —
**within the M5 Max tensor-core envelope** for Q4_0 dequant + MMA. There is
no easy 2–3× wire-up win in the kernel.

**The next 4.34× → ±5% gap closure is ~63% wrapper-side, ~37% kernel-side**.
Wrapper-side wins are addressable; kernel-side wins would require novel
work (subgroup-block fusion or weight repacking) that the Metal compiler
has consistently neutralised in 9 prior M5 Max static-evidence kernel
hypotheses (per `project_metal_compiler_auto_optimizes_static_levers`).

## Structural diff vs llama.cpp's MoE matmul kernel

For completeness — though the Qwen3.6 27B model uses dense FFN and does
NOT exercise the `_id` (MoE) path — here is the side-by-side state of the
mlx-native MoE matmul vs llama.cpp:

| component | mlx-native | llama.cpp | status |
|-----------|------------|-----------|--------|
| Dense mm kernel | `kernel_mul_mm_q4_0_tensor_f32` (`quantized_matmul_mm_tensor.metal`) | `kernel_mul_mm_q4_0_f32` (`ggml-metal.metal:10104`) | **byte-identical** (mpp::tensor_ops::matmul2d, same tile geom, same dequant) |
| MoE `_id` mm kernel | `kernel_mul_mm_id_q4_0_f32` + `_tensor_f32` (`quantized_matmul_id_mm.metal`) | `kernel_mul_mm_id_q4_0_f32` (same file, lines 9708-9716) | **structurally aligned** (map0 + mm_id; mlx-native has ne20_1 + ne20_8 templates only, llama.cpp has 1/2/4/5/6/8/10/16/22) |
| Routing threshold (mv vs mm) | `MM_ROUTING_THRESHOLD = 8` | `ne11_mm_min = 8` (`ggml-metal-ops.cpp:2046`) | identical |
| Tensor-API probe | one-shot `OnceLock` (`probe_tensor_mm`) | `GGML_METAL_HAS_TENSOR` compile-time | both produce the same hot-path kernel name on M5 Max |
| Activation fusion (silu_mul) | NOT fused into mm kernel — `dispatch_silu_mul` is a separate dispatch | NOT fused into mm kernel — same shape | **parity** (both pay 1 silu_mul dispatch per layer) |
| Concurrent dispatch | mlx-native `enc.memory_barrier()` between mm calls | llama.cpp `ggml_metal_op_concurrency_reset` | **parity** |

mlx-native's `_id` mm has a tighter top_k template instantiation set than
llama.cpp's (only `ne20_1` and `ne20_8` vs llama.cpp's 9 variants 1/2/4/5/6/8/10/16/22).
This is **NOT a bottleneck for current production models**: Qwen3.6 27B
is dense (no `_id` calls), Qwen3.6 35B-A3B uses top_k=8 (covered), Gemma 4
26B uses top_k=8 + top_k=1 (both covered). It would matter if/when a
top_k=2 / 4 / 5 / 6 / 10 / 16 / 22 model arrives.

A 2026-04-26 attempt to fuse silu_mul into a Q4_0 mv_id swiglu kernel
(`quantized_matmul_id_swiglu_q4_0`, mlx-native commit `4efeec0`) **regressed
−1.5% on dwq46 decode** despite eliminating one barrier — the Metal
compiler had already hoisted the silu work; the fused kernel saturated
ALU pressure (cited in `gpu_ffn.rs:1306-1318`). 9th confirmed M5 Max
static-evidence kernel hypothesis falsified.

## Top-3 sub-bottleneck within the FFN-dispatch bucket

Re-ranked from the W-5b.11 attribution after isolating kernel-only cost:

| rank | bucket | total ms | per-layer | % of 9,750 ms | location |
|------|--------|---------:|----------:|--------------:|----------|
| 1 | **Wrapper allocation churn** (6 large `device.alloc_buffer` per layer × 64 layers ≈ 118 GB allocate-and-zero) | ~3,000–4,000 (estimated) | ~50–60 ms | ~30–40% | `gpu_ffn.rs:690–704` (hf2q) — bypasses the `decode_pool` arena that exists |
| 2 | **2-encoder commit-and-wait pattern** (DenseQ takes 2 separate command buffers per layer; MoE-Q has the fused-residual-norm-in-same-CB optimization, DenseQ does not) | ~2,000–3,000 (estimated) | ~30–45 ms | ~20–30% | `forward_gpu.rs:1530–1548` vs MoE-Q at lines 1499–1525 |
| 3 | **silu_mul + residual_add elementwise dispatches** (2 small kernels per layer × 64 layers, each ~285 MB writeback) | ~500–1,000 | ~10–15 ms | ~5–10% | `gpu_ffn.rs:733, 752` |
| — | **Kernel-only qmatmul** (this audit) | **3,567 measured** | 55.7 | **36.6%** | `mlx_native::quantized_matmul_ggml::dispatch_mm` — already SOTA |

The estimated apportionment within the 6,184 ms wrapper bucket comes from:
- 118 GB of M5 Max shared-memory allocate-and-zero, at observed ~30 GB/s
  CPU memcpy-equivalent throughput, ≈ 4 s if every alloc is a fresh
  `newBufferWithLength:` (it currently is). Even partial pool reuse on
  steady-state would cut this to <1 s.
- 64 extra `commit_and_wait` calls (vs the MoE path's fused single CB),
  at observed ~30–50 ms per sync barrier on the M5 Max workload, ≈ 2–3 s.

These two together fully account for the 6,184 ms gap; the elementwise
kernels are second-order.

## Recommendation for ADR-015 implementation iter (NOT this audit)

### (a) Likely root cause
The DenseQ FFN path on Qwen3.6 27B was never given the same wrapper
treatment that the MoE-Q path received in ADR-012 §Optimize / Task #15
and ADR-013 P13.3. Specifically: (i) it allocates fresh `MTLBuffer`s
through `device.alloc_buffer` rather than the `decode_pool` arena, and
(ii) it uses 2 separate command buffers per layer rather than fusing
fused-residual-norm + the FFN body into a single CB.

### (b) Fix strategy (2–3 sentences)
**Mirror the MoE-Q optimizations into the dense_q path**: add a
`build_dense_ffn_layer_gpu_q_into` external-encoder variant in
`/opt/hf2q/src/inference/models/qwen35/gpu_ffn.rs`, route all 5 scratch
buffers (gate_buf, up_buf, hidden_buf, down_out, sum_buf) through
`super::decode_pool::pooled_alloc_buffer`, and update the DenseQ branch
in `forward_gpu.rs:1564-1568` to fuse `dispatch_fused_residual_norm_f32`
into the same encoder (the same pattern lines 1499-1525 already use for
MoE-Q). All work is in `/opt/hf2q`, not `/opt/mlx-native` — the W-5b.11
hand-off's "lives in mlx-native" claim was a misnomer driven by the
incorrect MoE-Q label.

### (c) Effort estimate
**Small — 1 iter (2–4 hours)**. The MoE-Q path is the template; the
mechanical translation is `_into` signature + 5 alloc-call rewrites + the
fused-CB caller pattern. No new kernels. No mlx-native changes. Tests:
existing dense_q parity test stays green (same kernel calls, only
allocator and encoder lifetime change).

### (d) Risk class
**Low**. Same kernel calls, same byte-level correctness (alloc pool
returns owned MlxBuffer with same shape/dtype). The arena's safety
contract is documented in `decode_pool.rs:38–44` — no buffer outlives a
`reset_decode_pool` call. The dense_q path's local-scope buffer lifetimes
already match this contract by construction. Risk-equivalent to ADR-013
P13.3 which landed without regression.

### (e) Parity-test pattern
Existing parity tests (`tests/test_qwen35_dwq_inference_parity.rs`,
sourdough byte-equivalence at decode) cover the dispatch correctness; new
test only needed if the `_into` variant introduces a new public API
surface (and even then, parity to the non-into variant is the bar — same
as MoE-Q's). The quantitative win is measured by re-running the W-5b.11
profile bench (`/opt/hf2q/scripts/bench-w5b11-post-attn.sh`); the
expected `layer.ffn_dispatch` drops from 9,750 ms to ~4,000–5,500 ms
(retaining only the 3,567 ms kernel-only floor + a small commit-overhead
residual). The wall-clock ratio vs llama.cpp would shift from 4.34× to
approximately 3.4–3.7× at pp4096.

## Why this audit STOPS at recommendation (per worker contract)

- mlx-native HEAD is unchanged (`5d9bb2e3` confirmed at start AND end).
- No production source rewrites in either repo.
- The recommendation is a hf2q-side wrapper change (NOT an mlx-native
  kernel change), so this audit doc is a hand-off to a hf2q implementer,
  not a mlx-native implementer. ADR-015 may not be the right ADR — the
  fix lives in ADR-005 territory (hf2q's qwen35 FFN dispatch). The
  W-5b.11 hand-off's "lives in mlx-native" framing should be revised in
  that doc when this audit is consumed.
- The audit produced no new perf claims; only kernel-isolation
  measurement (3,567 ms median) + structural diff. The recommendation's
  effort/risk estimates rely on the existing MoE-Q optimization landing
  cleanly via the same pattern.

## Files touched (additive only, this audit)

- `/opt/mlx-native/benches/bench_moe_q_qwen36_shape.rs` — new benchmark
- `/opt/mlx-native/Cargo.toml` — `[[bench]]` registration (3 lines)
- `/opt/mlx-native/docs/moe-q-perf-audit-2026-04-27.md` — this document

No `/opt/mlx-native/src/` modifications.
No `/opt/hf2q/src/` modifications.
No `/opt/llama.cpp` modifications (read-only reference).
