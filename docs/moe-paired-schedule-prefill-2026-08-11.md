# Paired MoE schedule reuse for large prefill

Date: 2026-08-11

Status: implementation candidate; focused Metal parity and microbenchmark pass,
exact-commit packaging and downstream hf2q integration remain pending.

## Decision

For large expert-routed prefill, expose a family-neutral paired projection
primitive instead of introducing another quantization-specific fused arithmetic
kernel:

```text
quantized_matmul_id_ggml_pooled_pair(
    input, gate_weight, up_weight, ids, gate_output, up_output, scratch, params
)
```

The first projection builds the existing `mm_id` per-expert routed-row
schedule. The second projection reuses that schedule in the same command
encoder. Both projections retain the existing quantized `mm_id` kernels and
may overlap after the map barrier. The API owns both projections, buffers,
parameters, and schedule lifetime, so callers cannot accidentally reuse a
schedule with different ids or dimensions.

This is intentionally a matrix-matrix prefill API. Matrix-vector/tiny-decode
fusion remains a separate measured experiment.

## Hypothesis ladder

1. A literal Q6_K gate+up+activation megakernel was the first spike. Although
   its parity proof passed, measured execution was roughly 11–24 times slower
   than the existing dispatch sequence because the dual projection increased
   register/threadgroup pressure. It is not selected for production.
2. Reusing only the routing schedule preserves the tuned arithmetic kernels,
   removes one `map0` dispatch and global barrier, and permits the two ordinary
   projections to overlap. This is the accepted direction for the candidate.
3. End-to-end family integration and tiny-decode specialization are deliberately
   deferred until the primitive is sealed and measured independently.

## Focused benchmark

Command:

```text
MLX_RUN_MOE_SCHEDULE_BENCH=1 \
  cargo test --release --locked \
  --test moe_prepared_schedule_prefill_microbench -- --nocapture
```

Environment: Apple M5 Max; `pmset -g therm` reported no thermal or performance
warning before and after. Each row uses three warmup pairs and 21 alternating
A/B samples. Values below are medians; the checked-in harness also prints p10
and p90. The benchmark covers gate and up projections only, excluding
activation and down projection so the delta remains attributable to schedule
reuse and projection overlap.

| Family shape | Quant | Tokens | Independent | Paired | Speedup |
|---|---:|---:|---:|---:|---:|
| Qwen3.6, K=2048, N=512, 256 experts, top-k 8 | Q5_K | 64 | 1.176 ms | 1.099 ms | 1.0699x |
| same | Q5_K | 256 | 1.613 ms | 1.542 ms | 1.0457x |
| same | Q5_K | 1024 | 1.831 ms | 1.751 ms | 1.0456x |
| same | Q5_K | 2048 | 3.204 ms | 3.067 ms | 1.0445x |
| DeepSeek-V4, K=4096, N=2048, 256 experts, top-k 6 | Q2_K | 64 | 4.346 ms | 4.209 ms | 1.0326x |
| same | Q2_K | 256 | 7.643 ms | 7.558 ms | 1.0112x |
| same | Q2_K | 1024 | 8.527 ms | 8.446 ms | 1.0096x |
| same | Q2_K | 2048 | 16.289 ms | 16.132 ms | 1.0098x |

The focused result is positive at every tested prefill size, with a larger
relative benefit for the smaller Qwen projection. It is not an end-to-end
tokens-per-second claim.

## Correctness and safety contract

- Q5_K top-k 8/top-k 1 and Q2_K top-k 6 paired outputs are bit-identical to
  the corresponding ordinary `mm_id` projection and retain their established
  reference tolerances.
- Q6_K low-level prepared-schedule output is bit-identical to the existing
  ordinary projection across the current parity shapes.
- The public pair requires the `mm_id` route and rejects decode-sized work.
- Both projections share one input, ids table, quantization type, expert count,
  and matrix shape. Distinct weights and outputs are passed atomically.
- Output write ranges must not overlap. Weight/output extents are validated
  before the first projection is encoded.
- Scratch capacity and all existing `mm_id` validation remain load-bearing.

## Remaining acceptance work

1. Seal the source and package, then rerun parity, the public contract test,
   the focused benchmark, and locked all-target/all-feature checks from the
   exact commit.
2. Publish and pin an exact mlx-native revision before using the primitive in
   hf2q; do not rely on a local Cargo patch.
3. Integrate one family at a time in hf2q and report exact model, prompt,
   sampling settings, prefill throughput, semantic output parity, and multiple
   thermally controlled runs.
4. Measure tiny decode independently. Do not infer decode benefit from this
   matrix-matrix result.
