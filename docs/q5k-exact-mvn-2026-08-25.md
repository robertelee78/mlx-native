# Q5_K exact multi-column matvec

Status: implemented and locally proven on Apple Silicon; not published.

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

## Fused gate/up row-identity release proof

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
