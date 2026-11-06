# Column-invariant few-row MMA mat-vec for Gemma 4 decode

Status: validated 2026-11-06 at `a888477` and later (`feat/gemma-decode-op-parity`
rebased onto mlx-native 0.17.0, `4a53ce8`), Apple M5 Max, both metallib
paths. Adopted: `quantized_matmul_ggml_column_invariant_mma`, a port of the
reference implementation's few-row simdgroup-MMA mat-vec (`mul_mv_mma`).
The column-invariant `mul_mv_ext` variant measured slower than MMA at every
width and was deleted.

## Contract (Class B, column-invariant)

```rust
mlx_native::quantized_matmul_ggml_column_invariant_mma(
    encoder, registry, device, input, weight, output, &GgmlQuantizedMatmulParams,
)
```

- Admits widths `m ∈ 1..=8`; types Q4_K, Q6_K, Q5_0 (generic kind, `K % 64
  == 0`) and Q8_0 (block kind, `K % 32 == 0`). src1 stride and offset must
  be 16-byte aligned. Everything else is rejected with `InvalidArgument`.
- Not bitwise equal to the stock mv routes (different accumulation order).
  Each output is bounded against an f64 reference over the exactly
  dequantized weights by `2·K·ε_f32·Σ|w·x|`.
- Column-invariant (Class A inside the entry): column `j` of a width-`w`
  result is bitwise identical to the width-1 result for that input row, at
  every `w ∈ 1..=8`, under a permuted column order, and next to NaN / +Inf
  columns. Batched decode therefore equals serial decode, which the
  operator requires.
- Deterministic: repeats are bitwise identical. Views at non-zero
  16-byte-aligned offsets give the offset-0 bits.
- Invariance holds within one process. The precompiled and runtime
  metallib paths may step `nsg` down differently at the pipeline thread
  limit (see the tiling rule), so their bits may differ from each other;
  Class B allows that, and both paths meet the same bound.
- Additive: no routing-policy field or existing entry changes; default
  routing is unchanged.

## Tiling rule

The reference's rule, `rt = 1` (widths ≤ 8):

- `nsg` = 32 for ≤ 64 rows, 16 for ≤ 6144 rows, else 8, halved while it
  exceeds the number of K steps (K/32 for the block kind, K/64 otherwise);
- `nt` = 4, halved while it exceeds `min(16384 / (nsg·256), N / 1024)`;
- the pipeline thread limit may halve `nsg` further; threadgroup memory is
  `nsg·nt·256` bytes; grid `(⌈N / 8·nt⌉, 1, 1)`, threadgroup `(32, nsg, 1)`.

Every input depends on (type, N, K) only, so every width runs the same
pipeline; this is what makes the result column-invariant.

A sweep over `nt ∈ {1, 2, 4}` × `nsg ∈ 4..32` was within run-to-run noise of
the rule (`probe-mma-sweep1`, `probe-mma-confirm`), so the rule stays. A
K-loop unroll spilled registers and ran 2–6× slower (`probe-mma-unroll`); it
was reverted.

## Results

Model-free, cold weights, Gemma 4 decode dense projections plus the lm head,
µs per decode step:

| Arm | m = 1 | m = 2 | m = 3 | m = 4 | m = 8 |
|---|---|---|---|---|---|
| stock (`quantized_matmul_ggml`) | 2816 | 3751 | 4544 | 5822 | 11609 |
| MMA (`mma_ci`) | 3891 | 3978 | 3995 | 3995 | 4072 |

MMA reads each weight once per step at any width, so its cost is nearly flat
in `m`; it overtakes stock from `m = 2` and is 2.85× faster at `m = 8`.
The deleted column-invariant `mul_mv_ext` measured 4401 / 4426 / 4597 /
4669 µs at widths 1–4 in the pre-rebase run (`3c5fbb9`), slower than MMA
at every width.

**Width-1 trade-off.** At `m = 1` MMA costs about 38% more than stock.
This is accepted: the invariance requirement rules out switching to stock
at width 1, because the width-1 column would then differ bitwise from the
same column inside a wider batch.

Receipts (under `/Users/robert/.local/state/hf2q/v0130/q4-gemma-decode/`):
`op-parity-a888477-20261106T203120Z` (the table above),
`op-parity-3c5fbb9-20261106T193731Z` (pre-rebase, incl. `mul_mv_ext`),
`probe-mma-sweep1`, `probe-mma-confirm`, `probe-mma-unroll`, and
`DECODE-OP-PARITY.md`.

## Gate suite (model-free, GPU; both metallib paths)

Each with `MLX_PRECOMPILED_METALLIB=1` and `=0`:

```bash
cargo test --release --locked --test test_mma_column_invariant -- --test-threads=1 --nocapture
```

Covers the lm head Q6_K 262144×2816, Q4_K 4096×2816, 8192×2816, 2048×2816,
2112×2816, 2816×4096, 2816×8192, Q6_K 2048×2816, Q5_0/Q8_0 2816×2112
(K = 64·33), a small N that is not a multiple of the row tile for every
admitted type, an output tail canary, and the rejections. Each bound shape
prints `MMA_BOUND type=… n=… k=… C=2 worst_ratio=…` (asserted ≤ 1).
Added after review: NaN / +Inf neighbour columns at widths 2..8 (Q4_K
2048×2816, Q8_0 2816×2112); an f32-activation guard
(`MMA_F16_GUARD … ratio=…`: the GPU's summed error must be ≤ 1/4 of an
f64 reference over f16-rounded activations, which the K-scaled bound is
too loose to detect); and input / weight / output views at non-zero
16-byte-aligned offsets with sentinels, bitwise equal to offset 0. At
`a888477` the 6 MMA gates passed 6/6 on both paths; the added gates are
pending that rerun.

## Bench

`cargo bench --locked --bench bench_gemma_decode_op_parity` (families
`dense`, `lmhead`, `experts`; arms `stock` and `mma_ci`) writes
`cells.jsonl` and `metadata.json` to `GEMMA_OP_PARITY_OUT`.
