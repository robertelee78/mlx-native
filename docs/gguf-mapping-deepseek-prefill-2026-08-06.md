# GGUF mapping and DeepSeek sparse-prefill decision — 2026-08-06

## Scope and ownership

`mlx-native` owns the reusable Metal mechanisms in this change:

- read-only file-backed GGUF resources and typed tensor views;
- nested view-offset propagation and logical CPU reads;
- strict dispatch validation for D=512 heads-as-rows attention;
- kernel parity tests and opt-in production-shape benchmarks.

hf2q owns model-family policy: which DeepSeek tensors are mapped, the sparse
crossover threshold, cache lifetime, OpenAI/tool semantics, launcher behavior,
and end-to-end quality/performance acceptance.

## Hypothesis and mapped-weight result

Copying a 100.05 GiB immutable GGUF into anonymous Metal buffers duplicated
bytes already available through macOS file-backed paging and contributed to
memory-pressure kills. A single whole-file Metal buffer is not portable because
the artifact exceeds the device's per-buffer limit. Creating one no-copy Metal
resource per tensor is also wrong because page-aligned virtual ranges overlap.

The accepted design partitions sorted tensor payloads at Metal's maximum
buffer length, creates an independent read-only mmap for each segment, and
creates typed tensor views into those resources. On the DeepSeek artifact this
produced two Metal resources, 107,422,652,416 file-backed tensor bytes, and
7,518,556 anonymous bytes retained for expanded elementwise state. Load time
fell from about 40 seconds to 0.85-0.98 seconds. Mapped-versus-owned embedding,
six layer-0 matrices, and full-model top-three logits were bit-exact. A 120K
agentic run completed without a kill and reported 2.0 GiB process RSS after the
cached continuation.

The mapping boundary fails closed: every mmap range is checked against the
current file length before access, so a truncated payload returns a GGUF error
instead of faulting beyond EOF. Packed quantized shapes cannot widen a logical
CPU view beyond their physical tensor bytes, and the scratch buffer pool drops
file-backed handles instead of retaining a Metal object after its mmap owner.
Default-on tests cover all three cases.

## Hypothesis and sparse-prefill result

The old adapter invoked the D=512 flash kernel with `qL=1` and 64 logical
heads for every original token. The kernel tile owns eight physical query rows,
so seven rows were idle. The accepted path reinterprets the same contiguous
`[tokens, 64, 512]` storage as `[tokens, 8, 8, 512]`: physical heads become
query rows, and no transpose or temporary allocation is required. The token
mask broadcasts across the eight rows while learned sinks index all 64
physical heads.

The 16-query A/B test uses 64 distinct sinks and a mixed mask and is bit-exact
to the former path. The production benchmark measured:

| Queries | Former tile | Heads-as-rows tile | Speedup |
|---:|---:|---:|---:|
| 64 | 4.991 ms | 0.827 ms | 6.03x |
| 128 | 9.708 ms | 1.460 ms | 6.65x |
| 256 | 19.212 ms | 2.643 ms | 7.27x |
| 512 | 38.087 ms | 5.037 ms | 7.56x |
| 1,024 | 75.910 ms | 9.760 ms | 7.78x |

Rejected spikes are not in production: a one-simdgroup Q=1 variant did not
solve the tile utilization correctly; BF16-to-F16 conversion around attention
did not beat the accepted path; and temporary index-overlap instrumentation was
removed after it answered the diagnostic question. The long-prefill indexer
benchmark and packed-attention benchmark remain as opt-in performance receipts.
