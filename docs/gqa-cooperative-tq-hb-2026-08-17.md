# GQA-cooperative TQ-HB decode attention

Date: 2026-08-17  
Status: Q2 candidate; downstream hf2q end-to-end gate pending

## Problem and hypothesis

The scalar TQ-HB decode kernel dispatches one workgroup per query head. For
grouped-query attention, every query head in a group independently loads and
dequantizes the same packed K/V vector. Qwen3.8-27B has 24 query heads and four
KV heads, so the scalar layout repeats that work six times per KV head.

The hypothesis was that one workgroup can compute two or three query heads
from the same KV group while preserving each head's dot-product, online
softmax, value accumulation, split-K, and final-reduction order. The packed
K/V load and codebook lookup are then shared across the query-head tile.

## Implemented primitive

`flash_attn_vec_tq_hb_gqa` is an explicit, family-neutral D=256 primitive with
two variants: `GqaTile::Q2` and `GqaTile::Q3`. The first version accepts only
full, unmasked attention with caller-rotated Q, `ring_start=0`, and
`softcap=0`. Callers outside that contract must retain the scalar kernel.

At NSG=4, Q2 uses 11,264 bytes of threadgroup memory and Q3 uses 16,896 bytes.
Q3 saves more logical K/V work but reduces occupancy more sharply.

## Correctness evidence

The focused Metal test compares every output `f32::to_bits()` against the
scalar production kernel for Qwen3.8 geometry Hq=24, Hkv=4, D=256 across:

- TQ codebook bits 5, 6, and 8;
- Q2 and Q3;
- NSG 1 and 4;
- sequence lengths 1, 31, 32, 33, and 128.

Result: all cases were bit-identical on the M5 Max host.

## Isolated performance spike

Command:

```text
cargo test --release --locked --test test_flash_attn_vec_tq_hb_gqa_parity \
  bench_qwen38_gqa_tiles -- --ignored --nocapture
```

The test warms each pipeline eight times and then reports the median of seven
paired, rotated blocks using both Metal GPU timestamps and host wall time.

| KV length | scalar GPU ms | Q2 GPU ms | Q2 speedup | Q3 GPU ms | Q3 speedup |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.3772 | 0.2560 | 1.473x | 0.2597 | 1.452x |
| 32,768 | 0.5780 | 0.4141 | 1.396x | 0.4497 | 1.285x |
| 65,536 | 1.1155 | 0.8155 | 1.368x | 0.8926 | 1.250x |
| 104,966 | 1.8447 | 1.2522 | 1.473x | 1.4269 | 1.293x |

Q2 is the accepted candidate. Q3 is rejected because its occupancy cost
outweighs the additional reuse. Q2 narrowly missed the initial 1.50x isolated
stretch target at 105K, so isolated timing alone is not release authority.

## Sustained-load result

The checked-in sustained test encodes 16 Q2 attention dispatches per simulated
decode step for 1,000 steps at KV length 104,966. The first-quarter p50 was
28.4728 ms/step and the last-quarter p50 was 29.5260 ms/step, a 1.037 ratio.
This passes the 1.15 thermal-stability ceiling.

## Remaining acceptance gates

Before enabling this primitive in hf2q:

1. Publish and byte-verify a new `mlx-native` crate; a local Cargo patch is not
   a landed result.
2. Run Qwen3.8-27B at short and approximately 105K context with identical
   prompt, sampling, cache, and artifact settings.
3. Require exact greedy output/tool semantics, at least 15% end-to-end decode
   improvement at 105K, and no more than 2% short-context regression. If the
   short gate fails, hf2q must route to the scalar kernel below a measured
   crossover.

The two-dimensional query-position x query-head verifier needed for
speculative decoding is a separate follow-up. It must earn its own parity and
performance evidence.
