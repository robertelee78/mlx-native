# GGUF mapping and DeepSeek sparse-prefill decision — 2026-08-06

**Status:** Implemented
**Updated:** 2026-08-19

## Scope and ownership

`mlx-native` owns the reusable Metal mechanisms in this change:

- read-only file-backed GGUF resources and typed tensor views;
- Metal residency-set lifecycle and command-buffer dependency ordering;
- explicit uninitialized scratch allocation for graphs with proven full writes;
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

## Residency heartbeat and command-buffer result

Repeated inference exposed a Metal resource-preparation stall below the model
graph. On an identical second DeepSeek request, the first reordered layer group
took 706.790 ms wall time but only 233.018 ms of GPU execution; 473.772 ms was
spent before the GPU interval. Fresh pool allocation consumed only 2.301 ms,
retaining the complete 3.96 GiB scratch high-water left a 426.565 ms residual,
and the measured kernels were not slower. The inactive 100 GiB weight set was
being prepared at the first consuming command buffer.

The accepted fix is generic to `mlx-native`. A live residency set owns a weak
background heartbeat that calls Metal's `requestResidency` every five
milliseconds for 180 seconds after the most recent command-buffer submission.
The same mutex serializes the heartbeat with allocation membership changes;
the thread stops after the final set owner is dropped. The duration is
configurable with `MLX_NATIVE_RESIDENCY_KEEP_ALIVE_SECONDS`, and zero disables
the heartbeat for controlled diagnosis. The default matches the lifecycle in
pinned llama.cpp commit
`15586e2d7165570fb3aa7c26e0d442e289ef69de`; no model family calls it directly.

The second-request pre-GPU residual fell to 2.487 ms. A matched three-trial
M5 Max gate on the same 4,987-token prompt and 100.05 GiB hf2q-produced GGUF
measured hf2q medians of 674.737 prompt tok/s and 33.908 decode tok/s versus
llama.cpp medians of 672.914 and 31.643 tok/s. Every greedy transcript was
exact, all prompt-cache counts were zero, and trials were separated by 60
seconds. This accepts the heartbeat as both a repeat-run stability fix and a
family-neutral runtime improvement.

The graph API now also supports a dependency-checked reorder that retains all
referenced buffers until completion. hf2q uses it to submit four DeepSeek
verifier layers per command buffer with explicit layer barriers, reducing the
three-chunk prompt from 385 to 289 total synchronizations. A 43-layer command
buffer reduced synchronizations to 259 but did not improve throughput and
increased lifetime risk, so four is the accepted model policy rather than an
mlx-native default.

Finally, `MlxBufferPool::alloc_uninitialized` makes the existing reused-buffer
contract explicit for fresh scratch allocations. Default `alloc` and
`alloc_batch` still clear fresh Metal memory. The uninitialized operation is
safe only when a caller has whole-graph proof that every consumed byte is
written; hf2q established that proof with byte-identical zero-fill and hostile
`0xA5` runs across 11,954 intermediate files. Other model families remain on
the clearing allocator until they pass an equivalent producer-coverage gate.

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

## D=512 non-aligned KV-tail determinism — 2026-08-19

The layer-major cohort-prefill spike in hf2q should have been numerically
identical to serial prefill, but the first prototype diverged at layer 3 on the
last row of a 255-row cohort. Replacing only the gathered K/V scratch contents
with zeros restored parity. A smaller Metal regression then isolated the
mechanism: with `kL=641`, two identical logical K/V views produced different
output when bytes beyond the view were changed from zero to NaN.

The D=512 kernel loaded each final 8-row K/V matrix fragment before masking
rows beyond `kL`. The accepted correction preserves the existing aligned fast
path. Only the final partial block is populated lane-by-lane, with valid rows
loaded in the original matrix layout and nonexistent rows set to zero before
the same matrix-multiply and online-softmax sequence. There is no dispatcher,
model-policy, or public-API change.

Correctness evidence on an Apple M5 Max 40-core GPU:

- all 63 non-zero remainders modulo 64 are guard-independent under both NSG=4
  and NSG=8;
- separate `top_k=385` and `top_k=417` sparse-attention cases (tail
  remainders 1 and 33) match the independent CPU reference;
- the complete sparse-attention suite passes 10/10 and the D=512 flash-prefill
  suite passes 11/11;
- the hf2q DeepSeek-V4 real-artifact gate passes all 43 layers bit-for-bit at
  255 cohort rows, with both cache positions ending at 255.

The same downstream gate measured serial prefill at 1,339.917 ms and cohort
prefill at 1,129.545 ms, a 1.1862x speedup. The mlx-native production-aligned
benchmark is neutral within noise: three-run medians were 15.569 ms on
v0.10.11 and 15.560 ms with this correction; the packed-flash component was
9.834 ms and 9.812 ms respectively. Registry publication and the exact hf2q
dependency pin remain separate release gates.
