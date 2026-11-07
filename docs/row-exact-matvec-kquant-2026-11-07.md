# Row-exact few-row mat-vec for Q4_K, Q6_K and Q5_0 (2026-11-07)

This extends `ops::row_exact_matvec` (F16/Q8_0) with Q4_K, Q6_K and Q5_0 weights, for Gemma 4 and Qwen decode widths m = 1..8. The serving rule is batched ≡ serial: each output row is bitwise the same at every m. `quantized_matmul_ggml_column_invariant_mma` meets that rule too, but it is about 50% slower than the stock mat-vec at m = 1. These kernels meet the same rule at stock speed for m = 1.

## Contract (numerics class B)
- **Row exactness.** Row j of the output depends only on (W, x[j], K, T). T is the threadgroup width, chosen from the weight shape and never from m.
- **Fixed per-thread order.** Each thread walks a fixed set of quant units in ascending order:
  - Q4_K and Q6_K: 32 values per unit.
  - Q5_0: 16 values per unit.
- **Dequantization.** Each unit is dequantized once per weight row with an m-independent formula:
  - Q4_K: `fma(d·sc, q, −dmin·mn)`, with the nibble scale folded in exactly as 2^-s.
  - Q6_K: `fma(d·sc, q, −32·d·sc)`.
  - Q5_0: `fma(d, q, −16·d)`.
- **Accumulation.** Every input row then gets an explicit `fma` chain in element order. Partial sums meet through `simd_sum` and the fixed threadgroup reduction.
- **Not bitwise to stock.** Results differ from the stock mat-vec routes. They are bounded against an f64 reference by the test's γ_K·Σ|w·x| bound and are deterministic.

## Default schedule
The schedule depends on (type, rows, K) only:
- nsg is 1 from 1024 rows up for Q4_K, Q6_K, Q5_0 and F16, and from 4096 rows up for Q8_0.
- nr0 is 2 for Q4_K, Q6_K and Q8_0, and 4 for Q5_0.
- At Q8_0 2816 rows the nsg rule now rounds down to a power of two. Every DeepSeek-V4 F16/Q8_0 shape keeps its previous schedule.

## Validation (Apple M5 Max)
**Tests.** `tests/row_exact_matvec.rs` passes 8/8 and `row_exact_matvec_resolve_from_source` passes 1/1, both with the default metallib and with `MLX_PRECOMPILED_METALLIB=0`. The new types are checked for:
- the F64 bound;
- row exactness for B = 1..8 under nr0 1/2/4 at the default nsg and at nsg 1 and 4;
- determinism.

They run on these shapes:
- Q4_K 4096×2816, 2816×4096, 2048×2816 and 8192×2048;
- Q6_K 2048×2816, the lm head 262144×2816, and 8192×2048;
- Q5_0 2816×2112;
- small odd-N cells.

**Benchmark.** `bench_gemma_decode_op_parity` ran in a root-granted quiet window, on cold weights, with 3 interleaved repetitions per cell (median of the repetitions). Receipt: `row-exact-5b66a4f-quiet`. Columns are Gemma 4 decode dense projections excluding the lm head, µs per step at widths 1–4, and seconds per run on the V2 four-agent step mix (w1 ×1, w2 ×66, w3 ×1, w4 ×403):

| route | w1 | w2 | w3 | w4 | V2 s/run |
|---|---|---|---|---|---|
| stock mat-vec | 1735 | 2652 | 3143 | 4113 | 1.837 |
| column-invariant MMA | 2661 | 2664 | 2646 | 2665 | 1.255 |
| row_exact (default) | 1739 | 1796 | 2049 | 2888 | 1.286 |

Per operation at m = 1, Q4_K, Q5_0 and Q8_0 are at stock, ±2%.

**lm head.** The Q6_K 262144×2816 lm head streams 605 MB per call. It is thermally sensitive, so it was measured with a 6 s cooldown per cell. Comparing within each repetition, row_exact at m = 1 is at or below stock: 1057–1078 µs against 1114–1236 µs. At m = 4 it saves 300–460 µs against stock.

## Use
- Entry: `dispatch_row_exact_matvec(encoder, registry, device, RowExactWeight::{Q4_K, Q6_K, Q5_0, Q8_0, F16}, w, x, y, &RowExactMatvecShape::contiguous(m, n, k))`.
- Shape requirements: K must be a multiple of 256 for Q4_K and Q6_K, and of 32 for Q5_0. A Q4_K weight must start on a 16-byte boundary.
- hf2q selects this route per weight with `HF2Q_GEMMA_DECODE_DENSE=row_exact`.
