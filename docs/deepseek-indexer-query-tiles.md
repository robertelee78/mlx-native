# Fixed-256 internal indexer query tiles

This additive primitive implements the scoped contract in issue #54. It does
not change an HF transaction, context, slots, compressor, model policy, kernel
arithmetic or output selection geometry. Existing public dispatchers and Metal
shaders remain unchanged. No HF adoption or version/dependency change is included.

`deepseek_indexer_query_tile_plan` accepts only batch one and production indexer
geometry. It returns checked score/top-k workspace counts for min(256,Q) rows.
The dispatcher validates complete arguments, physical/logical extents and writable
alias exclusions, rejects file-backed or non-CPU-writable scratch/top-k/output,
prepares every query/weight/output view and actual-tail bank
layout, then obtains all four pipelines and validates thread/memory limits before
encoding. CPU metadata grows with checked ceil(Q/256), with checked reservation;
there are no Metal allocations or full payload copies inside the dispatcher.
Preparation errors encode no commands; driver/panic recovery is not transactional.
Alias checks use virtual address ranges for known shared views, not universal
physical-page alias detection across distinct mappings or external raw backings.
Callers must supply known nonaliasing backing provenance; unknown aliasing is
forbidden. Read-only mapped q/KV/weights remain eligible.

Each tile advances start_pos and query/weight/output byte origins. KV and the
attention-index offset stay unchanged. Width256 preserves the existing eight-row
MMA grouping. Key blocks, reductions, tie order, merge tree and finalization are
unchanged. Capture ranges include parent byte offsets and actual partial banks.
Explicit direct barriers include workspace reuse; recorded replay recomputes
hazards from the same sliced ranges. Callers retain command-lifetime ownership.

At Q2048/K65536 the two workspace roots total134217728 bytes versus1073741824:
896MiB less logical payload. This is an arithmetic result, not measured physical
peak, performance, 3/4GiB pool fit, full Q5 admission, or model-quality acceptance.
Retained historical bins and every other allocation domain still count.

## Controls and status

At initial source-only delivery, the author had not run compilation, pure tests,
Metal controls or performance. Root subsequently ran the validation recorded
below and owns its exact source-bound evidence. The pure filter is `query_tiles_` (nine unit tests).
Production preparation orchestration is exercised with tile/pipeline failure
callbacks and an encode sink; these are model-free control-flow controls, not
claims that a real driver/pipeline compiler failure was injected.

The ignored integration filter is
`query_tiles_bounded_mma_matches_untiled_and_streamed_cpu` in target
`deepseek_indexer_query_tiles`. Run alone under an external process watchdog.
Nine initialized, guarded Metal roots total11914948 requested bytes; numeric host
bound73736 gives11988684 combined bytes, below16MiB. This explicit packet bound
excludes opaque driver/compiler/allocator overhead and is not whole-process memory.
No full F32Q, full CPU score matrix or output copy is retained. The CPU oracle
reads actual BF16 sparse inputs, proves inactive components zero, and sorts one
row of scalar scores by descending score then ascending key.

Q257/K1025 tests the partial query tile and key-block1024 edge. Three cases
reuse the same roots only after both arms drain and prior graph/command owners
drop: causal_winner (start3843/tail score32), early_causal (start0/tail32), and
cross_block_tie (start3843/tail15). Query and weight head-zero feature values are
zero on rows divisible by3, otherwise one. Thus row0/255 are zero and row256 is
one: losing either query or weight tile origin numerically loses the row256
key1024 winner; losing the absolute start independently excludes that key.
The tail32 case requires row255/256 to change top winner; start0 proves
sentinels. Tail15 ties key1024 across the1024-key block boundary and explicitly
requires rank64, after all64 older score15 keys (15,31,...,1023). Case/tail-score
markers distinguish every completion, replay and oracle result. Full parent guards and untouched output
columns are checked. Capture assertions verify every sliced range and tail-bank
origin before consuming those nodes through ComputeGraph reorder and
encode_with_barriers (the same steps as finish_with_reorder, split from its
immediate commit to encode both arms before either submits). Both waits are
collected before asserting either status or reading outputs. A final actual
undersized-output rejection must leave the capture empty. There is no skip/pass
fallback. This primitive control does not qualify full2048 model runtime or speed.

Future authorized performance work must compare fresh-process identical model
settings with parity and multiple-run timing; eight tiles may lose throughput.
Native publication plus verified downstream registry pin precedes HF adoption.

## Separately supervised legal endpoint operator timing spike

The ignored target `deepseek_indexer_query_tiles_perf`, exact filter
`query_tiles_legal_endpoint_dense_operator_tradeoff`, was initially a source-only benchmark candidate. The authorized single run and
independent result review are recorded below.
It uses Q2048/K63488/start251904, stride640/column128/index offset128: the legal
253952-token prefix endpoint with8192 output reserve at context262144. This is
one indexer primitive, not a model/prefill transaction or four-slot serving run.

All nine guarded roots remain resident together:1231029376 requested bytes;
explicit numeric host capacities1571840B (<2MiB), requested root ceiling2GiB.
Opaque driver/compiler/allocator metadata is Unknown; these are not physical/RSS
limits or a claim of measured memory saving. Both arms use unchanged public MMA
dispatchers in normal encoding mode, not capture/reorder or profiling. The two
workspace pairs request1040187392B untiled versus130023424B tiled, a910163968B
logical difference. Old full-context endpoint896MiB arithmetic remains separate.

Every actual BF16 input feature is deterministic ±1, all64 heads active; weights
are exactly1/64. Frozen independent Q/KV seeds are32d90dbe781c3a65 and
c1854a2b6790def3. Actual signs, not only the generator, feed a packed popcount
oracle on21 frozen rows spanning causal and query-tile boundaries. Integer dots
and dyadic1/64 score sums are exactly representable by the current MMA arithmetic.
The oracle orders integer numerators descending, then original key ascending.
Full2048-row outputs must also match exactly; every trial checks selected-index
bounds, rewritten output, sampled oracle, untouched prefix and all root guards.
Two CPU-only controls reject an invalid BF16 value and altered expected winner;
real input is never corrupted for a negative GPU test. No full F32 input/output
or score copies are retained; candidate sorting is unstable/in-place.

Exactly A,B,A,B warmups precede five A,B,B,A timing blocks (ten samples/arm).
Each public dispatch through checked commit/wait is timed, including per-call
metadata Vec/view preparation and pipeline cache lookups. Root/fixture/oracle,
output poisoning, encoder creation, thermal queries and printing are excluded.
One arm drains before any next fallible encode; there is no outstanding submit
on a checked-wait error. GPUStartTime/GPUEndTime are absolute seconds, separately
recorded and subtracted only with finite0<start<end. All20 samples, per-block
ratios, arm medians/min/max/range and overall ratios print after measured blocks.
No outlier deletion, dynamic rounds, tile adjustment or predetermined win result.

NSProcessInfo thermalState uses NSInteger/isize, and BOOL getters/availability
use objc::runtime::BOOL (target-specific ABI), with Foundation framework linkage.
Installed SDK NSProcessInfo.h declares thermal states0..3 and low-power getter
availability macOS12. Missing class/selectors/unknown thermal values remain
unqualified. Four observations bracket initialization/warmups, plus immediately
before/after each block; only thermal0 and low-power false qualify. The fixed run
records unfavorable conditions without waiting/retrying; parity is separate,
then timing ends inconclusive if any condition/timestamp is invalid. Endpoint
sampling is not continuous thermal/clock proof. This source does not query or
change power/frequency limits, use root, load a model, purge, or launch helpers.

Root must separately freeze source/binary/environment identity, reject route,
capture/profiling/unretained overrides, verify quiet host/no model listeners and
passive free-memory context>=15GiB, then use the approved600s process-group
watchdog with bounded termination/reap. That requested envelope/preflight is not
qualified admission. The author has not built or executed this benchmark. A
slower tiled result is a valid measured operator tradeoff; no HF adoption, model
speed, full memory fit, or Q5 qualification follows from this spike.

Pipeline provenance is collected after warmups from the SAME registry used by
both arms: actual get_pipeline(...).label() feeds pipeline_identity for the MMA,
block, merge and finalize kernels. All four complete Debug-serialized identity
records print after timing, including origin and runtime-source/embedded-metallib
SHA options. This adds no timed work or Metal payload; identity String/object
metadata remains outside the scoped numeric-host packet. Origin is observed from
the registry, never inferred from a successful build or environment intent.

## Root-observed validation and measured result (2026-10-24)

One source-bound tiny control execution passed all three V3 cases:
causal_winner/tail32, early_causal/tail32, cross_block_tie/tail15. Each recorded
both completed arms, actual reordered capture replay, independent CPU/untiled
parity, guards and the undersized zero-encode negative. The original wrapper
receipt preserved explicit_pass=false because the first completion marker had
the standard Rust harness prefix. A separate qualification-receipt-v2 accepted
exactly that one prefix framing correction against the SAME log/binary/source;
all bounds/oracles/markers remained required. There was no hardware rerun and
no retroactive replacement of the original receipt. Tiny archive identity:

- V3 source freeze SHA256: `8917ab3ef2125495249b43a16579d662fc139d57c914f8151738ba59c1b181e7`.
- Binary SHA256: `c6292cf12149b704cfd5d4cdc3d138da285fef0c52f3128025a9b79ac8121ac7`.
- Log SHA256: `3f216b16037e975d034893e5acd16f5d153c8d00eb88864544d32e3d8b17147a`.

The separately authorized dense timing run, archive
`native54-perf-20261024T223742Z`, completed exactly one ignored test, exit0,
without timeout or retry in its archived supervisor. Hardware context was arm64
Apple M5 Max, macOS27.2, SDK27.0. Four warmups and five ABBA blocks produced ten
timings per arm in one process/session; no outliers were removed.

| Measurement | Untiled | Tiled | Tiled / untiled |
| --- | ---: | ---: | ---: |
| Median dispatch-to-checked-wait latency | 260.421333ms | 246.3129585ms | 0.94582481267 |
| Median GPU interval | 260.1906875ms | 246.0575ms | 0.94568142451 |

Tiled median wall latency was 5.417518733% lower in this single bounded operator
run. Untiled wall min/max were 260.320541/260.762667ms; tiled 246.230208/246.611209ms.
The measurement includes public preparation/metadata/cache lookup through checked
wait and uses normal encoding, not capture/reorder. It is not a full model speed
win, serving benchmark, across-session confidence estimate or general leadership
claim. All 21 independent CPU sample rankings, full A/B output equality, all-row
bounds/rewrites/prefixes/guards and both CPU negatives passed. All 14 passive
observations were available, thermal 0/low-power 0. These endpoints do not prove
continuous thermal/clock stability or OS quietness.

All four actual warmed pipeline identities were PrecompiledMetallib, with
embedded SHA256 `1535691acfc89a29734171c286a4a61ddfa92f971d47825cf6adbd7429782e5b`
and runtime_source_sha256=None. This binds the observed selected embedded
pipelines; it does not qualify the RuntimeSource route. No model was loaded.
Actual K63488 logical two-workspace reduction was 910163968B = 868MiB. The 896MiB
power-of-two bucket difference (also the K65536 logical endpoint formula) is
separate arithmetic. Both arms' roots coexisted; neither quantity was observed
as physical reclamation, pool-cap fit or reduced whole-process footprint.

Timing-run identity, with frozen V4 benchmark source:

- Source freeze SHA256: `e1ec1a01a41aa6a519e90d2a76ec039c23e609870b717f32fc3155758a0e5216`.
- Binary SHA256: `2c9249a82ccd4dd43413fe0544c1d1dc2471c185865d3a6cea8fe57c62b906e2`.
- Log SHA256: `a5d63e7f96110b8c173cfbfaaf3f673562ac59c1e239ac6b33486f64c83cf1bf`.

| Source file | SHA256 |
| --- | --- |
| `src/ops/deepseek_indexer.rs` | `ea66feeef15d6509dd9b15d735981ee5d7405920026b3ce63aa14b706c4496bc` |
| `src/ops/deepseek_indexer/query_tiles.rs` | `1be90924ed87c2036a4115a8977bfd207c6afca4b522da02f8f8fdd0b893e3fc` |
| `src/ops/deepseek_indexer/query_tiles/plan.rs` | `f63b26f9cbc46b7925ccaa8811aaf1fbe830b2efad592d5e068b18113312297a` |
| `src/ops/deepseek_indexer/query_tiles/encode.rs` | `34d3aeb34c0247e234d91efc639257f123572fd6d518422914315b7e2f1129f7` |
| `src/ops/deepseek_indexer/query_tiles/tests.rs` | `bd4ea007b02e95985cae3efd911243dc8e5e6cb5e2ca83bd2f475b606ba0b452` |
| `tests/deepseek_indexer_query_tiles.rs` | `63766e53f3bd0c4a8f8e814abe7ab9a8e7755209197f33d94519328761df2bc5` |
| `tests/deepseek_indexer_query_tiles_perf.rs` | `2193ae2a030fdddd5eca970111c5f38a83dfccdf2f8c726a3eab3c937d031bad` |
| `tests/native54_perf/fixture.rs` | `b7e7ae3ca01df4aa8b9189b70a1d3451ce0857744d3fc0c985c5a91bd2d36e93` |
| `tests/native54_perf/observer.rs` | `508fb721a9b8e2917fcdc7f97acdcb234c013b35f9f0a1885ce4535dba47dc1b` |
| `tests/native54_perf/runner.rs` | `2185c9ce4676677acca9ea6273c115e256e848147dcd2fa8b38d1dc3ebe871a1` |

The pre-benchmark combined validation passed locked all-target/all-feature check
(10.61s), library tests (463 passed/11 ignored), release build (11.64s), and hygiene.
These runs preceded the added timing target. Its V4 release no-run compile then
passed (0.40s), with only the separately known library warning; Objective-C macro
lint suppression belongs at that benchmark crate call site. The result-note edit
changes no Rust bytes; final candidate validation/review remains the integrator's
next step and must not be inferred from those historical checks.

Portable focused commands (execution still requires the reviewed quiet host,
passive preflight, frozen identity/environment and external watchdog):

```sh
cargo test --locked --lib query_tiles_
cargo test --locked --test deepseek_indexer_query_tiles query_tiles_bounded_mma_matches_untiled_and_streamed_cpu -- --ignored --exact --nocapture --test-threads=1
cargo test --release --locked --test deepseek_indexer_query_tiles_perf --no-run
cargo test --release --locked --test deepseek_indexer_query_tiles_perf query_tiles_legal_endpoint_dense_operator_tradeoff -- --ignored --exact --nocapture --test-threads=1
```

Retain raw receipts, original wrapper framing failure and corrected sidecar,
source/binary archives, all 20 timings and all 14 passive observations. This result
qualifies only the declared bounded primitive observation. Native publication,
verified HF pin/adoption, real-model quality/speed, complete B/C/R/G accounting,
3/4GiB cap fit and full ADR063Q5 remain separate and unqualified here.
