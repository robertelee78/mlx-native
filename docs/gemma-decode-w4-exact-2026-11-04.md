# Exact width-4 decode kernels for Gemma 4 (Q4-GEMMA-DECODE)

Status: M0 adoption frozen 2026-11-05 (V1 at `718e811`, Apple M5 Max,
macOS 26B5112a, Apple metal 32023.922, precompiled metallib). Adopted: F2
only, frozen at R0 = 4; hf2q uses it at decode widths n_tokens ≤ 4. The GQA
flash `flash_attn_vec_hybrid_batched_gqa` stays as hf2q's warm-prefill
attention primitive. Every other lever was dropped and its code deleted.
Governing design: hf2q `docs/research/adr063-q4-gemma-decode-design.md`
(revision 2): §3 contract, §7.5 thresholds, §11 clean cutover.

## Contract (Class A against stock)

Each kept kernel writes, for every output element, the same u32 bits as the
stock path it replaces, on the same metallib path:

| Lever | Entry | Reference |
|---|---|---|
| F2 | `quantized_matmul_id_ggml_mv_grouped_geglu` (explicit; kernel `kernel_mul_mv_id_q4_K_f32_grouped_geglu_rt4`) | `kernel_mul_mv_id_q4_K_f32` gate_up into `[pairs, n]`, then `moe_swiglu_batch_encode(.., n/2, pairs)` |
| GQA flash | `flash_attn_vec_hybrid_batched_gqa` (explicit group and per-slot buckets; kernels `flash_attn_vec_hybrid_batched_gqa{1,2,4,8}_dk{256,512}`) | stock batched / per-slot vec-hybrid flash and reduce at each slot's own (nwg, nsg) |

F2 takes the stacked `[n_experts, n, K]` Q4_K gate_up weight (rows `0..n/2`
gate, `n/2..n` up) and writes `[n_tokens * top_k, n/2]` F32. It rejects
other types and odd `n`, and validates buffers exactly as the stock `mv_id`
entry does (with the output row width `n/2`). It does not cap the width:
callers choose the decode widths that take it (parity is proven at top-8 up
to 16 tokens and top-1 up to 128 pairs). Out-of-range ids produce the stock
NaN poison mapped through the epilogue. Both kernels sit in the same `.metal`
files as their references, so both share Metal's default math mode on both
metallib paths.

The PR is additive: no `GgmlRoutingPolicy` field, capability route or
existing entry point changes, and default routing is bit-identical.

## M0 results and adoption (design §7.5)

Ratios are lever / stock median at the cell (lower is faster). Source:
`m0-adoption.json` of the V1 run.

| Lever | Rule | Measured | Verdict |
|---|---|---|---|
| D1 body (row-tiled Q4_K/Q6_K) | ≤ 0.80 at m = 4 over the shape mix; never > 1.02 | every geometry slower at m = 4: Q4_K 1.19–1.32, Q6_K V 1.61–1.71 | dropped |
| D1 lm head (Q6_K 262144×2816) | ≤ 0.80 at m = 4 | 1.57–1.62 at m = 4 (1.02–1.06 at m = 1) | dropped |
| D2 (Q5_0/Q8_0 mN) | ≤ 0.67 each at m = 4 | Q5_0 1.148, Q8_0 1.019 | dropped |
| E1 (grouped `mv_id` gate_up + down) | ≤ 0.85 at the V2 pattern; never > 1.02 at U = 32 | V2 0.98 (R0 = 2), 0.93 (R0 = 4); U = 32 worst 1.51–1.68 | dropped |
| F1 (dual gate+up) | ≥ 3% better (≤ 0.97) than the adopted route at m = 4 | 0.988 against two D1 launches; D1 itself not adopted | dropped |
| F2 (grouped gate_up + GELU epilogue) | ≥ 3% better than the adopted non-fused route (stock gate_up + `moe_swiglu_batch`) | R0 = 4: 0.699; R0 = 2: 0.722; R0 = 1: 0.782 | **adopted, R0 = 4** |
| A1 (GQA decode flash) | ≤ 0.85 weighted over V2 KV lengths at n_q = 4; never > 1.02 | d256 G2 0.948; d512 G2 0.871, G4 1.72, G8 4.16; worst uniform cells 1.03–4.81 | dropped for decode |
| A2 (mixed-bucket decode flash) | V2 mixed-bucket share ≥ 10% | 0% (18 mixed cells measured) | dropped |

F2 needs no shape table: its one geometry is the fastest measured and is
compiled in. D1/D2/F1/E1 kernels, routes, routing-policy fields and their
tests are deleted (§11 clean cutover). The GQA kernels stay because hf2q's
warm prefill (Q4-GEMMA-ATTN-EXACT-G2, G ≤ 2) uses them; only hf2q's decode
use is dropped.

## Findings that still apply

- **Q4_K row sums decode scales in reference order.** In the S0 spike, a
  row loop that hoisted each row's scale decode into per-row arrays drifted
  1 ULP (`0x48b392fe` vs `0x48b392fd`). F2's per-row loop decodes the scales
  as scalars inside the block loop, in the stock statement order.
- **Rows run one after another (E1 finding, kept by F2).** Sharing a
  member's `yl`/`yh`/`sumy` across rows in one K loop changes Metal's
  default-math contraction and drifted 1 ULP on about 40% of outputs. Each
  SIMD group therefore runs its four gate/up rows sequentially, each row's K
  loop serving the whole member chunk (up to four pairs routed to the same
  expert). Members stay interleaved, so each expert's weights are read once
  per step.
- **F2 poison.** A compile-time `NAN` fed into the GELU epilogue was folded
  away under default math; the poison is rebuilt at run time from `p.r2` so
  the epilogue runs on the same bits as `moe_swiglu_batch` does.
- **GQA flash d512 packed V.** The compiled stock kernel evaluates the V
  scale as `(w * (1/sf)) * norm`; the clone spells it that way (a literal
  `norm / sf * w` drifted 1 ULP for non-power-of-two scale factors).
- **GQA flash d256 V weight underflow.** Stock computes
  `v_norm * inv_sqrt_dv * w` with the 2^-4 scale applied last; a plain
  product was compiled to scale `w` first, flushed subnormals and wrote 0
  where stock wrote ~1.4e-38. `ldexp` pins the scale-last order.

Each finding is compiler-dependent. Rerun the suite after every pin bump and
host OS / Metal toolchain update.

## Parity suite (model-free, GPU; both metallib paths)

Each with `MLX_PRECOMPILED_METALLIB=1` and `=0`:

```bash
cargo test --release --test gemma_gelu_epilogue_bitwise -- --test-threads=1                     # F2: T-N6, T-N3 replay, T-N4
cargo test --release --test gemma_decode_attn_gqa_and_mixed_bucket_bitwise -- --test-threads=1  # GQA flash: T-N8
cargo test --release --test gemma_warm_attn_exact_s0 -- --test-threads=1                        # GQA flash: warm S0-a / X1
```

`gemma_gelu_epilogue_bitwise` covers the Gemma gate_up shape (N = 1408,
K = 2816, 128 experts) at top-8 for 1..=16 tokens over every routing pattern
(distinct, one expert, random, pooled, skewed, invalid ids first / middle /
last / inside a group), NaN/∞ token rows, row tails (n_ff = 7, 515), a short
K, top-1 up to 128 pairs, non-zero-offset views and sentinel bytes outside
each view, NaN isolation between pairs, and the encoded kernel and grid. It
also replays real routing (design §2.4): the checked-in sample
`tests/fixtures/gemma_routing_witness_sample.json` (extracted by
`tests/fixtures/gemma_routing_witness_extract.py` from the Q1 witness
`q1-gemma-routing1511-t0-on-actual-v1`) always runs, and the full witness
(rows 128–143 of every layer, rows 29–127 of every third layer) runs when
present; otherwise the test prints `SKIPPED`. With `GEMMA_DECODE_V2_OVERLAP`
naming a file that holds the hf2q `[ROUTING_OVERLAP]` line, it replays one
pattern per observed (width, U).

## Bench

`cargo bench --bench bench_gemma_decode_w4` (families `roofline`, `f2`,
`attn`) writes `cells.jsonl` and `identity.json`. `f2` times stock gate_up +
`moe_swiglu_batch` against F2 at n_tokens 4, 8 and 16 over synthetic and V2
patterns. `attn` times the GQA flash against the stock batched and per-slot
routes. The warm-prefill arm of the GQA flash is `vec_gqa` in
`benches/bench_flash_attn_warm_precise.rs`.

A multi-row GQA generalisation (R > 1) was measured bitwise but slower than
R = 1 at every G and is not included.
