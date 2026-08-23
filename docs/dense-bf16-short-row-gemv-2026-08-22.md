# Native BF16 short-row GEMV — 2026-08-22

Status: accepted primitive at source `bebbddf`; registry publication and the
downstream clean-source gate remain release conditions.

## Decision

`dense_gemv_bf16_f32_tiled4` consumes the caller's BF16 weight buffer
directly and reuses each weight load across four adjacent F32 input rows. It
does not expand, dequantize, or re-encode the matrix. Its F32 reduction order
matches `dense_gemv_bf16_f32`, so every completed row is bit-identical to the
ordinary row path.

The operation is a family-neutral primitive, not a hidden route policy.
Latency depends on M and the N/K aspect ratio. Callers must select among the
ordinary row, four-row-tiled, and tensor paths from measured evidence. The
library therefore does not encode a model name or a blanket `M <= 16`
threshold.

Both row kernels fail closed on zero dimensions, incompatible dtypes,
unaligned logical views, K values that cannot support aligned vector loads,
undersized logical spans, integer overflow, and invalid batch broadcasting.
Odd output widths duplicate the final valid weight pointer for the unused
second row rather than reading beyond the logical view.

## Hypothesis and reformulation

The initial hypothesis was that one four-row kernel should replace tensor MM
for every M from 2 through 4. A 21-round matrix falsified that rule: M=2 was
consistently faster on the existing row path, and the small KV projection also
preferred row execution. The kernel itself generalizes safely to arbitrary M
by dispatching `ceil(M / 4)` row tiles, so the deciding benchmark was expanded
to M=1 through 16 rather than restricting the public API to the first observed
case.

The expanded matrix also falsified a universal short-row cutoff. For example,
the four-row tile wins selected M=5 through 8 aspect ratios, while tensor MM
wins other matrices at the same M. The accepted result is therefore the exact
primitive plus its benchmark; downstream selection remains explicit and must
be proven on the model shapes it serves.

## Correctness evidence

The release-mode focused test proves:

- bit identity with ordinary row GEMV for every M from 1 through 16 at
  production K=5,120;
- an independent CPU oracle for M=5, odd N=5, K=516, and 2-to-8 batch
  broadcast;
- output guard preservation across the odd-N and M-tail case;
- actual compilation of the new pipeline;
- fail-closed dtype, size, alignment, K, dimension, and broadcast validation.

The same test passed with the embedded metallib, with
`MLX_PRECOMPILED_METALLIB=0`, and with `MLX_UNRETAINED_REFS=1`. The locked
all-target/all-feature check also passed.

## Performance evidence

`bench_dense_bf16_short_rows` runs one dispatch per command buffer, four
warmups, and 21 reversed-order samples per route. It reports median and
interquartile wall and GPU time and aborts if the tiled result differs by one
bit from ordinary row GEMV. The four production matrices are:

| Role | N | K |
|---|---:|---:|
| FFN gate/up | 17,408 | 5,120 |
| FFN down | 5,120 | 17,408 |
| attention Q | 6,144 | 5,120 |
| attention K/V | 1,024 | 5,120 |

On an M5 Max, the dedicated M=4 receipt measured these median wall times in
microseconds:

| Role | tensor-32 | ordinary row | tiled-4 | Best |
|---|---:|---:|---:|---|
| FFN gate/up | 645.291 | 632.916 | 523.417 | tiled-4 |
| FFN down | 871.333 | 645.500 | 532.250 | tiled-4 |
| attention Q | 378.458 | 314.208 | 283.459 | tiled-4 |
| attention K/V | 270.500 | 170.458 | 182.500 | ordinary row |

The complete M=1 through 16 matrix is reproducible with:

```bash
MLX_BENCH_SUMMARY_ONLY=1 MLX_BENCH_WINNER_ONLY=1 \
  cargo bench --locked --bench bench_dense_bf16_short_rows
```

The downstream source spike used the exact 54,657,734,208-byte Qwen3.8 27B
BF16 artifact with SHA-256
`f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee`,
the six checked-in code/repetition requests, greedy sampling, and one server
slot. Against source `681842461816616eb9f273b7c0d1f0d9c62fda1`, routing the
hot dense FFN verifier projections through this primitive preserved all six
response bodies byte-for-byte and moved aggregate code throughput from 19.82
to 23.12 tokens/s (+16.6%) and repetition from 19.59 to 23.11 tokens/s
(+17.9%). This is downstream spike evidence, not registry-release authority
or a claim against the external reference implementation.

## Release conditions

1. Publish an immutable crate containing `bebbddf` and verify the registry
   archive checksum.
2. Pin that registry version in hf2q without a local Cargo patch.
3. Centralize the measured BF16 selector across applicable model families;
   do not duplicate a Qwen-only threshold at leaf call sites.
4. Repeat exact trajectory, physical-width, model-swap, and matched
   performance gates from the clean published dependency.
