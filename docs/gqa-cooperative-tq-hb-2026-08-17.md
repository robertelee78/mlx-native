# GQA-cooperative TQ-HB decode attention

Date: 2026-08-17; evidence refreshed 2026-08-18
Status: register-resident-Q follow-up merged on `main` and included in the
0.10.10 release candidate; registry publication and downstream exact-artifact
gates are pending

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

The 0.10.9 kernel stored Q in threadgroup memory and reloaded it inside every
KV-position dot product. At NSG=4 that used 11,264 bytes per workgroup. The
follow-up keeps each lane's 32-byte Q slice in registers, removes the repeated
threadgroup reads, and reduces threadgroup memory to 10,240 bytes. This stays
below one third of the 32 KiB allocation bound and can admit three resident
workgroups where the previous allocation admitted at most two.

A pre-landing spike also measured a wider Q3 tile, but its occupancy cost
outweighed its additional reuse; that rejected specialization is not present
in the API, registry, or shader instantiations.

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

Result after moving Q into registers: all parity cases were bit-identical on
the M5 Max host, all supported prewarms compiled, and every invalid width
returned the expected `InvalidArgument`. The focused suite ran four
non-ignored tests; the two hardware benchmarks remain explicitly ignored in
the ordinary test command.

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
1.437x paired median predicted too little margin: a sealed hf2q 0.1.7 spike
using the published 0.10.9 kernel produced 17.1767 tok/s OFF versus 19.5490
tok/s AUTO, only 13.8112%, and correctly failed the 15% gate.

## Reformulated launch and register-Q spikes

The first reformulated hypothesis was that Q2 needed a different split or
simdgroup schedule. Direct 104,966-token measurements falsified it. Q2 GPU
time was 1.2821 ms at NSG=4, 1.3767 ms at NSG=2, and 2.6017 ms at NSG=1.
Sweeping NWG over 16, 20, 24, 28, and 32 found only about 1.3% absolute Q2
improvement at the best point, without enough end-to-end margin. Neither
schedule change is retained.

The second hypothesis was that the shared-Q bank imposed both repeated
threadgroup traffic and an occupancy cliff. Keeping Q lane-local in registers
preserved the same half conversion and per-head arithmetic. Three separate
104,966-token processes measured:

| process | scalar GPU ms | register-Q Q2 GPU ms | paired speedup |
|---:|---:|---:|---:|
| 1 | 1.8319 | 1.1431 | 1.603x |
| 2 | 1.8371 | 1.1413 | 1.610x |
| 3 | 1.8424 | 1.1408 | 1.615x |

The 1,000-step, 16-dispatch-per-step sustained run remained thermally flat:
first-quarter p50 28.4938 ms, last-quarter p50 28.4770 ms, ratio 0.999.

The decisive downstream spike used hf2q source
`6ef4be6e497bf6df6b2eac2a2a84b054ec236d8c`, model SHA-256
`0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`,
105,097 prompt tokens, 512 greedy tokens, and fixed OFF/AUTO/AUTO/OFF order.
OFF measured 16.6051 and 16.5413 tok/s; AUTO measured 20.8975 and 20.3203
tok/s. The means were 16.5732 and 20.6089 tok/s, a 24.3506% improvement;
within-arm spread was 0.3849% and 2.8008%. All four responses had identical
semantic SHA-256
`e292d422cf493f02ae5bb30e056a0ba124e34fc7c5a4d2afe825fcbf30a21a6b`.

That downstream run used a local path-patched mlx-native worktree. It proves
the implementation hypothesis, not release authority. hf2q must consume the
published immutable crate and repeat its protected exact-artifact gate.

## Sustained-load result

The checked-in sustained test encodes 16 Q2 attention dispatches per simulated
decode step for 1,000 steps at KV length 104,966. The register-Q result above
passes the 1.15 within-run thermal-stability ceiling. It was one process;
process-to-process sustained-load dispersion was not measured.

## Remaining acceptance gates

Before enabling register-Q Q2 by default in hf2q:

1. Publish and byte-verify the new immutable mlx-native crate.
2. Pin that registry version and checksum in hf2q without a local patch.
3. Repeat Qwen3.8-27B at short and approximately 105K context under the
   protected thermal envelope with identical prompt, sampling, cache, and
   artifact.
4. Require exact greedy output/tool semantics, at least 15% end-to-end decode
   improvement, no more than 2% short-context regression, no fatal runtime
   signatures, and the documented spread and telemetry gates. Re-verify the
   8,192-token crossover with the published register-Q kernel; hf2q's selector
   must retain the scalar kernel below the accepted crossover.

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
