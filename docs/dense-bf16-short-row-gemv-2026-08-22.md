# Native BF16 short-row GEMV — 2026-08-22

Status: accepted primitive and frozen calibration policy for mlx-native
0.12.2; registry publication and downstream clean-source gates remain release
conditions.

## Decision

`dense_gemv_bf16_f32_tiled4` consumes the caller's BF16 weight buffer
directly and reuses each weight load across four adjacent F32 input rows. It
does not expand, dequantize, or re-encode the matrix. Its F32 reduction order
matches `dense_gemv_bf16_f32`, so every completed row is bit-identical to the
ordinary row path.

Latency depends on M, the N/K aspect ratio, the physical device, and the exact
compiled pipelines. `dense_matmul_bf16_f32_auto` therefore reads a
registry-local immutable route plan created by
`calibrate_dense_bf16_routes`; the library does not encode a model name or a
blanket performance winner. M=1 through 16 is the explicit short-row coherence
boundary: these widths use the row-reduction equivalence class, then calibration
chooses between ordinary row and tiled-four execution. Calibration accepts
borrowed weights and declared reachable M values, records wall and GPU
distributions, and freezes the plan before request-visible work. It never
authorizes a route from a different reduction class. A poisoned logical-output
and guard-region execution still proves full writes, finite values, bit
identity, and exact dispatch identity before either route may enter the timing
comparison.

The normal route decision is keyed only by the copyable physical shape. It does
not allocate, read environment variables, synchronize, or touch the process
cache; encoding then uses the registry's ordinary pipeline lookup. Missing
shapes take a deterministic exact fallback from the frozen plan. Loading a
second model with a previously calibrated device/build/shape key reuses
metadata through keyed single-flight cells with zero timing submissions;
neither the cache nor a plan retains model buffers.

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

The expanded matrix also falsified a universal short-row winner. For example,
the four-row tile wins selected M=5 through 8 aspect ratios, while tensor MM
wins other matrices at the same M. Tensor MM is not an admissible short-row
winner because it changes reduction semantics across physical batch widths. A
Qwen-only threshold spike then proved whole-model value but not a family-neutral
policy. The accepted reformulation is an exact route set plus centralized
pre-serve calibration and an immutable engine-epoch plan. Model graphs supply
weights and reachable physical widths; they do not choose kernels.

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

`test_dense_bf16_auto` additionally proves a real Metal activation over every
M from 1 through 16, full encoded pipeline/grid/threadgroup/shared-memory
identity, a one-shot plan, non-vacuous output, conflicting-plan rejection, and
a second activation with sixteen process-cache hits and zero calibration
submissions. It passes in the same three runtime modes. Calibration is bounded
to declared M=1..16 shapes and a caller budget. The current engine plan freezes
a typed compatibility fallback when its budget expires. The process cache
retains only completed intrinsic decisions: budget fallbacks and transient
failures are evicted so a later model activation can retry with its own
available budget without changing the already-live plan.

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
| FFN gate/up | 651.417 | 666.458 | 532.917 | tiled-4 |
| FFN down | 891.041 | 664.417 | 538.958 | tiled-4 |
| attention Q | 356.417 | 319.917 | 294.125 | tiled-4 |
| attention K/V | 269.916 | 182.042 | 178.750 | overlap |

The K/V wall distributions overlap; GPU medians were 25.625 microseconds for
ordinary row and 32.250 for tiled-4. A selector must retain the compatibility
route when calibration cannot establish a material, stable winner.

The complete M=1 through 16 matrix is reproducible with:

```bash
MLX_NATIVE_BENCH_COMMIT=a112660 \
  MLX_BENCH_SUMMARY_ONLY=1 MLX_BENCH_WINNER_ONLY=1 \
  cargo bench --locked --bench bench_dense_bf16_short_rows
```

The downstream source spike used the exact 54,657,734,208-byte Qwen3.8 27B
BF16 artifact with SHA-256
`f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee`,
the six checked-in code/repetition requests, greedy sampling, and one server
slot. The request contract and exact runnable comparison harness are
`scripts/qwen38_matched_reference_contract.sh` and
`scripts/qwen38_matched_reference_abba.sh` in hf2q. Against source
`681842461816616eb9f273b7c0d1f0d9c62fda1`, routing the hot dense FFN verifier
projections through this primitive preserved all six response bodies
byte-for-byte and moved aggregate code throughput from 19.82 to 23.12
tokens/s (+16.6%) and repetition from 19.59 to 23.11 tokens/s (+17.9%). This
is downstream spike evidence, not registry-release authority or a claim
against the external reference implementation.

## Release conditions

1. Publish an immutable crate containing `bebbddf` and verify the registry
   archive checksum.
2. Pin that registry version in hf2q without a local Cargo patch.
3. Construct and freeze one route plan during mandatory activation for every
   applicable model family; install the same plan into its worker registries.
4. Repeat exact trajectory, physical-width, model-swap, and matched
   performance gates from the clean published dependency.
