# Changelog

## Unreleased

### Performance

- Q8_0 dense matvec at `M = 2..=8` input rows (dense auto and independent-
  weight batched entry points) now runs `kernel_mul_mv_q8_0_f32_nr2_r{M}`,
  which reads each weight block once per call for all rows instead of once
  per row. Outputs are bit-identical to per-row `kernel_mul_mv_q8_0_f32_nr2`;
  `M = 1` is unchanged. Opt out with `HF2Q_Q8_0_MV_R1=0` or
  `GgmlRoutingPolicy::dense_q8_0_mv_r1 = false`. See
  `docs/q8-0-mv-nr2-multi-row-2026-10-29.md`.

### Breaking (pre-1.0 receipt contract)

- `GgmlRoutingPolicy` gains `dense_q8_0_mv_r1`; `GgmlKernelRoute` gains
  `DenseQ8_0MvNr2Rows` and `DenseBatchedQ8_0MvNr2Rows`;
  `GgmlResolvedKernelRoute` gains `DenseQ8_0MvNr2Rows`.
  `GGML_CAPABILITY_SCHEMA_VERSION` is now 3 and
  `GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION` is now 4.
