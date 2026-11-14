# Threadgroup broadcast WAR race in decode routing and fused norms (0.15.16)

## Defect

Several kernels finish a cross-simdgroup reduction by writing the total to a
threadgroup slot, barrier, and have every thread read it. The next phase then
stores into the same slot with no barrier in between. A simdgroup that reaches
the read after that store uses a clobbered value.

| Kernel | Broadcast read | Overwriting store |
|---|---|---|
| `fused_moe_routing_f32` (+ `_selection_witness`) | `max_val = shared[0]` | `shared[tid] = local_sum` |
| `fused_moe_routing_f32_v2` / `_v3` (+ witnesses) | `max_val = shared[num_experts + 0]` | `shared[num_experts + i] = e` |
| `fused_post_ff_norm2_endlayer_f32` / `_v2` | `rms_inv_moe = rsqrt(shared[0] ..)` | `shared[tid]` / `shared[sgitg]` partial sum |
| `fused_moe_wsum_post_ff_norm2_endlayer_f32_v2` | `rms_inv_moe = rsqrt(sg_scratch[0] ..)` | `sg_scratch[sgitg]` |
| `fused_post_attn_triple_norm_f32` / `_v2` | `rms_inv_attn = rsqrt(shared[0] ..)` | `shared[tid]` / `shared[sgitg]` |

The batched routing kernels already carried the barrier (99fea55); the
single-token kernels and the two-reduction norms did not.

Effect on routing: at `tg = 64` the late simdgroup owns experts 32..63 and
96..127; a clobbered max rescales all of them against the other simdgroup's
experts and top-K picks the wrong experts. Effect on the norms: the late
simdgroup normalizes its slice with thread 0's partial sum.

Downstream symptom (hf2q, Gemma 4 decode): rare run-to-run differences in
selected experts and logits from identical inputs. Two independent runs
selected an all-simdgroup-1 top-8 at one (position, layer), with the weights
of the experts both runs kept differing by one common factor.

## Fix

One `threadgroup_barrier(mem_flags::mem_threadgroup)` after each broadcast
read. Non-raced executions produce the same bits as before.

## Evidence

Apple M5 Max, macOS, debug test profile.

- `tests/test_fused_moe_routing_decode_max_race.rs` (6 routing kernels,
  8192 concurrent instances x 64 command buffers each): before the fix 2 of 2
  runs failed (104 and 91 divergent instances of 3.1M; divergent top-K
  `[127..120]` instead of `[1..8]`); after, 3 of 3 runs had 0.
- `tests/test_fused_two_reduction_norm_race.rs` (5 kernels, 2048 identical
  rows x 16 dispatches each): before, 2 of 2 runs failed (50 and 63 divergent
  rows of 163,840); after, 3 of 3 runs had 0.
- GPU time per dispatch, dependent chain of 30k decode-shape dispatches,
  median of 6 (release): `fused_moe_routing_f32_v3` 5.66/5.77 us fixed vs
  5.76/5.76 us before; `fused_post_ff_norm2_endlayer_f32_v2` (dim 2816)
  3.526/3.546 us fixed vs 3.550/3.528 us before. No measurable change.
- hf2q Gemma 4 T3 replay (production kernels, 100 decode re-runs of rows
  128..144 per lane per process, concurrent GPU load): on 0.15.15, 4 of 6
  processes diverged; with this fix 15 of 15 were bit-exact (96,000 decode
  steps). This was a local path patch for diagnosis only; hf2q adoption
  requires the published crate and a locked pin.

## Chunk recurrence follow-on (0.22.1)

The K128 and K256 chunk inter-state kernels had the same dependency error.
Every SIMDgroup reads the old shared `bh` state for the output snapshot and
the `w @ bh` product, but a faster SIMDgroup could start scaling that state
before another finished reading it. Both kernels now place a threadgroup
barrier before the cooperative scale, retaining the post-scale and
end-of-chunk barriers. No reduction, gate or rounding order changes.

The first debug-layer check also exposed an independent K256 resource error:
the old V32 tile requested more shared memory than the device advertises.

| K256 inter-state layout | Threads | Shared bytes |
| --- | ---: | ---: |
| Previous V32 | 128 | 36,864 |
| Corrected V16 | 64 | 18,432 |

The observed device limit was 32,768 bytes. The corrected two-SIMDgroup
layout covers four BT row tiles and two V column tiles per SIMDgroup;
each output cell retains the same increasing K/BT reduction sequence.
K128 and the downstream K256 output kernel's V32/TG256 layout are unchanged.
Global state/output layouts are unchanged.

### Scoped validation

Apple M5 Max, 128 GiB, macOS 27.2, Rust/Cargo 1.99.1:

- 28 source-bound positions passed with owned process cleanup: embedded and
  runtime-source paths, each plain, default shader validation and Metal API
  debug validation; two exact regressions and K256 inter-state/full-pipeline
  smoke checks, plus the original reference fixture and structural checks.
- The nonzero-state regression checks independent first-chunk values and
  eight reused-buffer output/state trials for K128 and K256.
- The nonuniform K256 oracle checks both batches/chunks, every head, V16
  boundary and K tile against an independent exact recurrence.
- Pipeline identities from both exact regressions prove the selected origin;
  enabling precompiled lookup alone is not accepted as embedded coverage.
  The local matrix's embedded artifact SHA256 was
  `5c67c50ed273a562e963b9dcb86a34da1d2d49b298f568f2a25cb96c71487ecb`.
- The preceding 36,864-byte debug failure is retained, not counted as a pass.
  These results establish neither global numerical equivalence nor an hf2q
  model-serving or performance qualification.

Reproduce the focused assertions with the full existing fixture:

```sh
cargo test --release --locked --test test_gated_delta_net_chunk -- --test-threads=1 --nocapture
cargo test --release --locked --lib chunk_inter_state_k256_dispatch_smoke -- --test-threads=1 --nocapture
cargo test --release --locked --lib chunk_k256_pipeline_smoke -- --test-threads=1 --nocapture
```

Run each exact GPU assertion in a fresh process for
`MLX_PRECOMPILED_METALLIB=1`/`0` (same setting for
`MLX_PRECOMPILED_METALLIB_FCV`), plain, `MTL_SHADER_VALIDATION=1` with
`MTL_SHADER_VALIDATION_COMPILER_INLINING=default`, and `MTL_DEBUG_LAYER=1`.
No selective validation exclusion or full-inlining workaround is used.
