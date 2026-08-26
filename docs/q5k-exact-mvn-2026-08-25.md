# Q5_K exact multi-column matvec

Status: the exact multi-column route shipped in 0.14.0. The canonical Q4x4
decode promotion is a 0.15.0 release candidate; registry publication and
downstream clean-source gates remain open.

## Contract

For every column and every supported continuous width `m=2..8`, the Q5_K
multi-column route must produce the same F32 bits as an independent invocation
of the production scalar Q5_K matvec. Sharing packed weight loads is allowed;
changing a column's floating-point accumulation or SIMD reduction tree is not.

Physical kernels are limited to widths 2 through 5. Wider logical batches use
the established independent-column tiling: 6=`3+3`, 7=`4+3`, 8=`4+4`.

## Spike receipt

At mlx-native base `1d9073a5`, the smallest hardware spike compared one
`r1=4` Q5_K kernel with four scalar calls using deterministic packed Q5_K
blocks, `M=4`, `N=512`, and `K=5120`. All 2,048 output F32 values were equal by
`to_bits()` on Apple Silicon. The focused test completed in 0.34 seconds after
compilation.

This accepts the literal-scalar-tree design and rejects `mul_mv_ext` as the
coherence route: that kernel intentionally uses a different dequantization and
reduction tree.

## Acceptance

- Exact `to_bits()` parity for every logical width 2 through 8.
- Canonical capability, planner, dispatcher, and resolved-trace receipts name
  the Q5_K exact-width route.
- Explicit opt-out still reaches ordinary scalar-tree matvec dispatch.
- Kernel timing is measured separately; exactness is not inferred from token
  agreement or a floating-point tolerance.

## Completion receipts

The canonical Q5_K path passed exact `to_bits()` comparison against independent
scalar invocations at every logical width 2 through 8, including the multi-tile
buffer-offset cases. Canonical capability and resolved-dispatch trace tests
identify the new route; the existing Q4_K and Q6_K exact-width parity suites
remain green. The Q5_K byte gate passed under both the packaged precompiled
metallib and forced runtime-source compilation.

On the same idle Apple-Silicon host, the checked-in ignored microbenchmark used
`M=4`, `N=5120`, `K=5120`, valid deterministic Q5_K blocks, explicit routing
policies, and GPU start/end timestamps. Two consecutive release runs reported:

- scalar-tree MV: 92.227 and 92.805 microseconds;
- exact Q5_K mN: 79.375 and 79.461 microseconds, 1.162x and 1.168x faster;
- `mul_mv_ext`: 47.282 and 46.940 microseconds.

Each number is the median of five samples containing 100 dispatches. The exact
route recovers a measured part of the weight-reload cost without changing any
output bit. The remaining gap to `mul_mv_ext` is real: its vector-dot reduction
is faster but does not meet this route's scalar-identity contract.

## Historical 0.14 fused gate/up row-identity proof

The dense fused gate/up/SiLU primitive is a separate operator and therefore has
its own width-invariance gate. The focused Metal test enumerates the complete
capability-admitted codec set—Q8_0, Q4_K, Q5_K, Q6_K, and IQ4_NL—and compares
every output F32 bit from one multi-row dispatch with independent `m=1` fused
dispatches over distinct input rows. All codecs passed at every width 2 through
8 with `N=64` and `K=512`.

The gate passed in release mode through both the packaged precompiled metallib
and forced runtime-source compilation. The five existing codec-specific fused
versus unfused parity suites also remained green (15 tests total). No runtime
change was required.

## Corrective bounds and logical-view audit

Independent review after the first implementation found two release blockers
that ordinary output parity could not expose:

- the final padded SIMD group formed and dereferenced a packed Q5_K row before
  the old output-store guard rejected it when `N` was odd; and
- exact-mN tiles supplied relative column offsets to the deliberately absolute
  `BufferWithOffset` API, losing a nonzero `MlxBuffer::byte_offset()`.

The source audit found both patterns in the inherited Q4_K and Q6_K exact
routes as well. Q4_K, Q5_K, and the baseline Q6_K scalar kernels now return
SIMD-group-uniformly before forming a padded row pointer. Q6_K NR2 and exact-mN
also guard the second row of a partially valid two-row group before forming its
row pointer. A static source-order gate pins those guards before `offset0`,
`x`, `x_base`, and per-row `xr` formation because a later store guard is not a
memory-safety proof.

All exact-mN tile bindings now checked-add the logical view base to the tile's
relative input/output offset before pipeline resolution or command encoding.
The dataflow audit found no additional change necessary: tracked ranges already
use each view's logical base and extent, while captured `BufferWithOffset`
bindings record the now-correct absolute offset.

The hardware gate uses odd `N=513`. Q5_K passes `to_bits()` parity against
independent scalar authorities at every logical width 2 through 8 and asserts
the resolved production route plus physical dispatch count. A second canonical
test covers Q4_K, Q5_K, and Q6_K at widths 4 and 7 using distinct input rows
inside nonzero-base parent views; every logical output row is bit-identical and
all input/output parent prefix and suffix canaries remain unchanged. Both tests
pass through the packaged metallib and forced runtime-source compilation.

Raw Q4_K/Q5_K/Q6_K mN dispatch helpers are no longer public. The validated
canonical dispatcher surface is the only external route: its default,
explicit-policy, and traced wrappers all share the same validation before the
private codec tiles are selected. The redundant direct Q6_K helper and
unreachable physical width-6/7/8 Q6_K kernels were removed.
`cargo +stable semver-checks check-release` against published
0.13.0 reports the intended missing-helper APIs and route-enum discriminant
changes, so publication follows the README's pre-1.0 breaking-change contract
as 0.14.0 rather than a patch release.

## Canonical Q4x4 decode promotion

The follow-up route keeps the public validated matmul dispatcher as the only
caller-facing surface. Internally, Q5_K decode widths 1 through 8 use one
fixed-width Q4x4 Metal kernel and tile wider logical batches without exposing a
raw dispatch helper. The kernel preserves the scalar Q5_K dequantization,
accumulation, and reduction tree for each output value; it never dequantizes or
requantizes the weight tensor.

Model-free hardware gates passed bit equality for packaged and runtime-compiled
Metal, odd output dimensions, nonzero logical buffer offsets, every continuous
logical width 1 through 8, and the staged physical widths. Mutation tests prove
the candidate flag reaches the actual resolved route rather than merely a
receipt field. The production default is enabled for the codec, independent of
model labels; `HF2Q_Q5K_CANONICAL_Q4X4=0` remains a diagnostic opt-out.

On the idle M5 Max host, the real Qwen3.8 Q5_K artifact exercised 508 target
decisions without a bit mismatch. A same-process paired verifier measurement
reported 6,105.525 ms for the prior exact route and 4,527.208 ms for Q4x4, a
25.851% reduction, with median rounds of 94.818 ms and 70.389 ms. A fixed
139-sample corpus retained 1.0 argmax agreement and changed perplexity from
1.514567 to 1.513641. A Qwen3.6 Q5_K hardware gate separately proved the same
canonical route at scalar and four-row widths while MoE expert projections
remained on the independent `mm_id` path.

The older fused Q5_K gate/up operator was removed during promotion: its own
row-invariance test passed, but its results were not numerically coherent with
the canonical exact projection route. hf2q had already retired that operator.
Keeping Q5_K on the canonical pair of validated projections preserves the
model's native GGUF bytes and one arithmetic authority.

The added routing-policy field and serialized capability/trace discriminants
are a public pre-1.0 contract change, so the candidate is versioned 0.15.0 and
is checked against the published 0.14.0 baseline. Final performance claims
remain downstream-gated on a published registry artifact, exact dependency
pin, multi-artifact and multi-slot serving, model swaps, and a matched peer
comparison.
