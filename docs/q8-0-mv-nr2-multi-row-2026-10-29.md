# Q8_0 NR2 exact multi-row matvec

Status: implemented on branch `perf/q8-0-mv-multirow-exact` (base `9c67e1f`,
0.15.15). Hardware parity and bandwidth receipts are pending; this note must
be updated with them before release.

## Problem

Joint decode runs B=2..8 input rows per step through the dense GGUF matvec.
Q8_0 used `kernel_mul_mv_q8_0_f32_nr2` with grid `(ceil(N/2), M, B)`: one
threadgroup per input row, so every Q8_0 weight block was read from DRAM once
per input row. With dense Q8_0 weights dominating the per-token read volume,
an M-row step paid roughly M times the weight traffic of one row.

## Contract

For every input row, output row, batch entry, and width `M=2..8`, the
multi-row route must produce the same F32 bits as an independent single-row
`kernel_mul_mv_q8_0_f32_nr2` call on that input row. Sharing weight loads is
allowed; changing a row's accumulation order or reduction tree is not.
`M=1` dispatch is unchanged.

## Kernel

`kernel_mul_mv_q8_0_f32_nr2_r{R1}` for R1 in 2..=8 (template over R1) uses
NR2's buffers and parameters with grid `(ceil(N/2), ceil(M/R1), batch)`,
threads `(32, 4, 1)`, and `256 * R1` bytes of threadgroup memory. Production
routing selects `R1 = M`, so one dispatch with `grid.y = 1` covers all rows.

Each threadgroup copies its two weight rows' NQ=8 int8 quants and half scale
per block into registers once and applies them to every live input row.
Bit-identity with NR2 per input row follows from:

- the same NR0=2 / NSG=4 / NW=32 / NQ=8 lane partition, so each
  `(output row, input row)` accumulator visits the same blocks `ib` in the
  same order on the same `(sgitg, tiisg)` lane;
- the same per-block expression forms: `sumq` starts at zero and accumulates
  `q * yl[iq]` for `iq = 0..7` in order (int8 weight operand, F32 register
  operand), then `sumf += sumq * d` with the block's half scale; the register
  copy holds the identical int8/half values NR2 reads through its pointer;
- the same reduction: per-simdgroup `simd_sum`, then one 32-lane threadgroup
  slot per `(output row, input row)` pair in which lane `sgitg` holds that
  simdgroup's partial and all other lanes are zero, then a final `simd_sum`.

Bounds defense in Metal: the second row of a partial two-row group is never
dereferenced, and input rows past `M` in the final y-tile are neither read nor
written. The Rust dispatcher validates buffers and dimensions before
selection.

## Routing

`GgmlRoutingPolicy::dense_q8_0_mv_r1` (default `true`; environment
`HF2Q_Q8_0_MV_R1=0`/`false`/`off` opts out) selects the route only when the
matvec type is Q8_0, NR2 is selected (`dense_q8_0_mv_nr2`), and `M` is in
2..=8. One predicate is shared by the dispatcher and `ggml_capability`, which
reports `DenseQ8_0MvNr2Rows` (dense auto) or `DenseBatchedQ8_0MvNr2Rows`
(independent-weight batched matvec), specialized for `ContinuousWidth`. An
explicit `dense_decode_mv_ext` opt-in keeps precedence. Opting out restores the
per-row NR2 grid. The resolved trace validates the exact kernel, geometry, and
threadgroup memory as `GgmlResolvedKernelRoute::DenseQ8_0MvNr2Rows`.

The new policy field and route discriminants change serialized receipts:
`GGML_CAPABILITY_SCHEMA_VERSION` is 3 and
`GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION` is 4.

## Gates

- `tests/q8_0_mv_nr2_rows_byte_parity.rs`: `to_bits()` parity against
  independent single-row NR2 calls for `N ∈ {1, 2, 63, 4096}`,
  `K ∈ {32, 256, 4096, 4128}`, `M ∈ {2, 3, 4, 5, 8}` with mixed-magnitude
  signed inputs and full-range quants/signed scales; partial final tiles via
  an explicit R1 with an untouched guard row; the batched entry point; and the
  opt-out route, including graph-capture shared memory and dataflow.
- `tests/q8_0_mv_nr2_rows_speed.rs` (ignored): median GPU time and
  unique-byte GB/s for `M=1` NR2, `M=4` per-row NR2, and `M=4` multi-row at
  `N×K` = 4096×4096, 4096×16384, and 16384×4096.
