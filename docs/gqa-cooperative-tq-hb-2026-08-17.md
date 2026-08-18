# GQA-cooperative TQ-HB decode attention

Date: 2026-08-17; evidence refreshed 2026-08-18
Status: Q2 primitive published in `mlx-native` 0.10.9; downstream hf2q
end-to-end gates are pending

## Problem and hypothesis

The scalar TQ-HB decode kernel dispatches one workgroup per query head. For
grouped-query attention, every query head in a group independently loads and
dequantizes the same packed K/V vector. Qwen3.8-27B has 24 query heads and four
KV heads, so the scalar layout repeats that work six times per KV head.

The accepted hypothesis is that one workgroup can compute two query heads from
the same KV group while preserving each head's dot-product, online softmax,
value accumulation, split-K, and final-reduction order. The packed K/V load
and codebook lookup are then shared across the query-head tile.

## Implemented primitive

`flash_attn_vec_tq_hb_gqa` is an explicit, family-neutral D=256 primitive with
the measured `GqaTile::Q2` variant. The first version accepts only
full, unmasked attention with caller-rotated Q, `ring_start=0`, and
`softcap=0`. Callers outside that contract must retain the scalar kernel.

At NSG=4, Q2 uses 11,264 bytes of threadgroup memory. A pre-landing spike also
measured a wider tile, but its occupancy cost outweighed its additional reuse;
that rejected specialization is not present in the candidate API, registry,
or shader instantiations.

## Correctness evidence

The focused Metal test compares every output `f32::to_bits()` against the
scalar production kernel for Qwen3.8 geometry Hq=24, Hkv=4, D=256 across:

- TQ codebook bits 5, 6, and 8 at sequence lengths 1, 31, 32, 33, and
  128, spanning NSG 1 and 4;
- TQ8 at kL=2,049/NSG=2 and kL=8,192/NSG=4, where each case processes
  multiple 32-token KV chunks in at least one simdgroup;
- TQ8 at kL=128 with a larger 256-token physical capacity, proving the
  capacity-strided cache layout remains bit-identical;
- valid prewarm compilation for TQ5, TQ6, and TQ8;
- fail-fast prewarm rejection for codebook widths 0, 4, 7, and 9.

Result: all parity cases were bit-identical on the M5 Max host, all supported
prewarms compiled, and every invalid width returned the expected
`InvalidArgument`. The focused suite ran four non-ignored tests; the two
hardware benchmarks remain explicitly ignored in the ordinary test command.

## Isolated performance spike

Command:

```text
cargo test --release --locked --test test_flash_attn_vec_tq_hb_gqa_parity \
  bench_qwen38_gqa_tiles -- --ignored --nocapture
```

The command was run in three separate processes. Each process warmed each
pipeline eight times and then measured seven paired, rotated blocks using both
Metal GPU timestamps and host wall time. The table reports the median of the
three per-process GPU medians; brackets show the full min-max range across the
21 samples. The speedup column is the median of the three paired per-process
speedup ratios.

| KV length | scalar GPU ms, median [range] | Q2 GPU ms, median [range] | paired Q2 speedup |
|---:|---:|---:|---:|
| 8,192 | 0.3095 [0.3005-0.8717] | 0.2099 [0.2019-0.5202] | 1.497x |
| 32,768 | 0.5791 [0.5715-1.4406] | 0.4139 [0.4065-0.9846] | 1.411x |
| 65,536 | 1.5320 [1.1395-2.1087] | 1.0218 [0.8104-1.2973] | 1.437x |
| 104,966 | 1.8532 [1.8096-2.0702] | 1.2919 [1.2717-1.4168] | 1.437x |

The broad ranges at 8K-65K show that host/GPU scheduling and frequency were
not controlled tightly enough to turn these isolated measurements into a
release claim. The 105K measurements were materially tighter, but their
1.437x paired median is still only directional kernel evidence. It is not an
end-to-end hf2q decode result or release authority.

## Sustained-load result

The checked-in sustained test encodes 16 Q2 attention dispatches per simulated
decode step for 1,000 steps at KV length 104,966. In the refreshed run, the
first-quarter p50 was 28.5210 ms/step (27.6359-29.2563 min-max) and the
last-quarter p50 was 28.5399 ms/step (27.7823-29.2028 min-max), a 1.001 ratio.
This passes the 1.15 within-run thermal-stability ceiling. It was one process;
process-to-process sustained-load dispersion was not measured.

## Remaining acceptance gates

Before enabling this primitive in hf2q:

1. Run Qwen3.8-27B at short and approximately 105K context with identical
   prompt, sampling, cache, and artifact settings.
2. Require exact greedy output/tool semantics, at least 15% end-to-end decode
   improvement at 105K, and no more than 2% short-context regression. If the
   short gate fails, hf2q must route to the scalar kernel below a measured
   crossover.

The immutable dependency gate is complete. Release workflow
`32120737230` tested exact source and the packed archive, published version
0.10.9, retested the downloaded registry archive, and verified both release
surfaces. Tag `v0.10.9` resolves to
`2bdf51b8b94a10aefc51a8a756b6e83f17773fa9`; crates.io and the GitHub release
asset are byte-identical with SHA-256
`5eb643eb35bcf582202a4534bdd4baa609ff51d8ca5c4203ce5cf2683e9ed323`.

The two-dimensional query-position x query-head verifier needed for
speculative decoding is a separate follow-up. It must earn its own parity and
performance evidence.
