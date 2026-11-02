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
