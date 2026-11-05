# Exact width-4 decode kernels for Gemma 4 (Q4-GEMMA-DECODE)

Status: kernels, dispatchers, parity suite and M0 bench landed on
`feat/gemma-decode-w4`; M0 timing (V1) and the geometry freeze are pending
the root's GPU lane. Governing design: hf2q
`docs/research/adr063-q4-gemma-decode-design.md` (revision 2), §3 contract.

## Contract (Class A against stock)

Every new kernel writes, for every output element, the same u32 bits as the
stock kernel today's decode uses for that tensor, on the same metallib path:

| Lever | New entry / route | Reference | Reference flags honoured |
|---|---|---|---|
| D1 | `kernel_mul_mv_{q4_K,q6_K}_f32_rt{R0}_r1_{R1}_nsg{NSG}`; route `DenseGemmaRowTiled` (`dense_row_tiled_q4k`, `dense_row_tiled_q6k`, `dense_row_tiled_m1`) | `kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_q6_K_f32_nr2` per column | Q6_K needs `dense_q6k_mv_nr2`; all need `!dense_decode_mv_ext` |
| D2 | `kernel_mul_mv_q5_0_f32_mN_r1_{R1}`, `kernel_mul_mv_q8_0_f32_nr2_mN_r1_{R1}`; route `DenseGemmaMn` (`dense_mn_q5_0`, `dense_mn_q8_0`) | `kernel_mul_mv_q5_0_f32`, `kernel_mul_mv_q8_0_f32_nr2` per column | Q8_0 needs `dense_q8_0_mv_nr2` |
| F1 | `quantized_matmul_ggml_dual_gate_up_q4k` (explicit only) | two stock Q4_K launches | — |
| E1 | `kernel_mul_mv_id_{q4_K_f32_grouped_rt{1,2,4},q5_0_f32_grouped,q8_0_f32_grouped}`; route `ExpertMvGrouped` (`expert_grouped_*`) | stock per-pair `mv_id` | Q8_0 needs `!expert_q8_0_mv_nr2` |
| F2 | `quantized_matmul_id_ggml_mv_grouped_geglu_with_geometry` (explicit only) | stock `mv_id` gate_up then `moe_swiglu_batch` | — |
| A1/A2 | `flash_attn_vec_hybrid_batched_gqa{1,2,4,8}_dk{256,512}` via `flash_attn_vec_hybrid_batched_gqa` (explicit, per-slot buckets) | stock batched / per-slot vec-hybrid flash and reduce at each slot's own (nwg, nsg) | — |

New `GgmlRoutingPolicy` fields are `#[serde(default)]`, default `false`, never
resolved from the environment, and carried by `combine_routing_policies`. A
dense lever is honoured only for shapes in the frozen table
(`row_tiled_geometry`, `dense_mn_shape_admitted`); other shapes keep their
stock route. The new kernels sit in the same `.metal` files as their
references, so both share Metal's default math mode on both metallib paths.

Baked decode records follow the same routing. `build_q8_0_id_decode_record`
returns `None` (callers fall through to `dispatch_id_mv`, which takes E1)
whenever the policy plans E1 for that call, so a record never runs the stock
kernel for a step the capability API reports as `ExpertMvGrouped`. The Q6_K
`_id` record needs no such check because E1 has no Q6_K member.

## Findings that shaped the kernels

- **S0 (Q4_K R0 = 2): exact.** The first D1 draft, which hoisted each row's
  scale decode into per-row arrays, drifted 1 ULP (`0x48b392fe` vs
  `0x48b392fd`, n = 4096, column 0, row 0) even at R0 = 1. Decoding each
  row's scales as scalars inside the row loop, in the reference statement
  order, made every R0 ≥ 2 cell exact. The degenerate R0 = 1, R1 = 1 clone
  still drifted, so R0 = 1 is not instantiated: one row per SIMD group is the
  stock geometry and stays on the stock kernels.
- **E1 Q4_K rows run one after another.** Sharing a member's `yl`/`yh`/`sumy`
  across R0 rows drifted 1 ULP on about 40% of outputs; each SIMD group
  therefore runs its R0 rows sequentially, each row's K loop serving the whole
  member chunk. Members stay interleaved, so each expert's weights are still
  read once per step.
- **F2 poison.** A compile-time `NAN` fed into the GELU epilogue was folded
  away under default math; the poison is built at run time with Metal's
  `NAN` bits so the epilogue runs on it as `moe_swiglu_batch` does.
- **A1 d512 packed V.** The compiled stock kernel evaluates the V scale as
  `(w * (1/sf)) * norm`; the clone spells it that way (a literal
  `norm / sf * w` drifted 1 ULP for non-power-of-two scale factors).

Each finding is compiler-dependent. Rerun the suite after every pin bump and
host OS / Metal toolchain update.

## Validation record (V0, model-free)

Toolchain: macOS 27.2 (26B5112a), Apple metal version 32023.922
(metalfe-32023.922.4). Commands, each with `MLX_PRECOMPILED_METALLIB=1`
and `=0`:

```bash
cargo test --release --test gemma_decode_w4_exact -- --test-threads=1                   # S0, T-N1, T-N2, T-N4, T-N7, F1
cargo test --release --test gemma_expert_grouped_mv_id_bitwise -- --test-threads=1       # T-N3
cargo test --release --test gemma_gelu_epilogue_bitwise -- --test-threads=1              # T-N6
cargo test --release --test gemma_decode_attn_gqa_and_mixed_bucket_bitwise -- --test-threads=1  # T-N8
```

T-N3 also replays real routing (design §2.4): the checked-in sample
`tests/fixtures/gemma_routing_witness_sample.json` (extracted by
`tests/fixtures/gemma_routing_witness_extract.py` from the Q1 witness
`q1-gemma-routing1511-t0-on-actual-v1`) always runs, and the full witness
(rows 128–143 of every layer, rows 29–127 of every third layer) runs when
present; otherwise the test prints `SKIPPED`. With `GEMMA_DECODE_V2_OVERLAP`
naming a file that holds the hf2q `[ROUTING_OVERLAP]` line, it replays one
pattern per observed (width, U).

All passed on both paths at `f2d273b`. With `GEMMA_DECODE_RECEIPT_DIR` set,
`gemma_decode_w4_exact` records an output digest per cell (T-N5); all 1,304
recorded dense and F1 cells had identical digests on both paths.

## M0 (pending)

`cargo bench --bench bench_gemma_decode_w4` (families `roofline`, `d1`,
`lmhead`, `d2`, `e1`, `f1`, `attn`) writes `cells.jsonl` and `identity.json`.
The adopt/drop thresholds are the design's §7.5. The geometry table in
`src/ops/gemma_decode_dense.rs` holds the pre-M0 default (R0 = 2, NSG = 2)
and is replaced by the M0 freeze; levers that miss their threshold are
removed in the promotion commit.
