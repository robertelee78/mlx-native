# Routing selection witness:0.15.10 release candidate

This unpublished0.15.10 candidate adds explicit optional diagnostic dispatch
APIs. Ordinary routing APIs, default kernels and arithmetic remain unchanged.
The tested primitive source is345af157e132705b72e3e51103c60d17404de4d9 on the
0.15.9 base. This candidate changes package version/lock and evidence prose only;
root still owns required clean checks, CI, registry publication and downstream
exact pin verification. No HF model, Q1 trajectory or speed result is qualified.

## Release changelog (candidate, not published)

- Add `dispatch_qwen_with_witness`, `dispatch_gemma_with_witness` and
  `dispatch_deepseek_score_with_witness` in `ops::routing_selection_witness`.
  `None` delegates to the ordinary API; only an explicitly supplied caller-owned
  witness selects diagnostic kernels. Add typed actual variant/domain and
  `read_after_checked_drain`; the reader adds no wait or completion certificate.
- Export actual kth/next selection-domain scores/IDs and explicit scored,
  all-experts-selected/no-next and Invalid dispositions; never invent a zero
  margin or recompute a raw-logit proxy. No default model/kernel route change.
- Preserve the cached Gemma V1/V2/V3 selector and exact buffer-view/alias/capture
  range checks. No new environment flag, GPU allocation policy or checked wait.
- Retain three pure and three actual variant-process tiny controls at the
  source/binary/evidence scope below. Real-model endpoint/selection parity,
  downstream adoption and speed remain pending, never inferred from this release.

## What is observed

`ops::routing_selection_witness` supplies three optional-witness dispatchers.
`None` calls the ordinary API. `Some(SelectionWitness)` drives the selected IDs
and final weights using a diagnostic copy of that actual producer and returns
its exact pipeline name/variant. These are not auxiliary re-softmax passes.

- Qwen: actual GPU softmax probabilities before selected-top-K renormalization.
- Gemma: the same pre-renormalization domain, before per-expert scaling. The
  ordinary and diagnostic dispatchers share the original cached V1/V2/V3
  selector. No environment value is mislabeled as observed GPU execution.
- DeepSeek-V4 0731 scored layers: actual `sqrt(softplus(logit))+bias`, global256
  expert top6. Final weights remain gathered UNBIASED sqrt-softplus normalized
  times1.5. SIMD reduction groups are not expert-group masking. Hash layers
  retain the ordinary hash API/table/input/selected-ID capture: margin N/A.

Caller-owned witness outputs are F32 `[rows,2]` scores, U32 `[rows,2]` IDs and U32
`[rows]` status,20 requested bytes/row. The pair is selected kth and maximum
unselected candidate under the actual producer's tie rule. The Qwen/Gemma kth
value is retained before top-K renormalization, and the DeepSeek kth value is
stored from the actual tournament winner. The next candidate is read from the
same kernel's remaining selection scratch, not recomputed from logits/weights.
Native selected IDs/order and final weight bits must match the ordinary route
on valid fixtures before any downstream adoption.

Status1 is a valid scored boundary; status2 means K==E and kth+1 is explicitly
unavailable (nextID=u32MAX/nextscore=NaN), not a zero margin; status3 means invalid
input/domain/route and supplies no margin. Rust refuses unsupported dimensions,
dtypes/extents, alignment, aliasing outputs and non-writable outputs. Qwen
witness supports power-of-two E<=256/K<=64; Gemma witness supports scalar/per-slot
E32/64/128/256/K<=64. It does not claim coverage of the distinct native batched
Gemma routing API. Nonfinite producer inputs get explicit invalid status; they
are never smoothed, retried or converted into a valid zero-score route.

The API allocates no GPU buffers, opens no encoder, submits nothing and waits
for nothing. The caller must retain outputs/inputs and use its EXISTING
successful checked drain before CPU reads/capture. `read_after_checked_drain`
adds no synchronization and is not itself a completion witness. All default
kernels remain byte-identical source. Diagnostic stores can perturb compiled
register allocation/timing; primitive parity is not real-model endpoint parity.

## Finite deciding controls and retained actual evidence

Three host-only controls invoke actual decode/range/ABI helpers. One separately
ignored actual Metal control uses tiny owned inputs/views, guard regions and a
single checked drain per case. No model/files, no16-million event loop and no
large allocation. Requested fixture payload per case is below64KiB; this is not
an OS/driver RSS or Metal-allocation limit.

The actual control covers four Qwen fixtures (ties/near/softmax underflow/K==E),
three Gemma fixtures (E128 ties/near with nonuniform scales, E64 all-selected),
DeepSeek biased/unbiased selection, six nonfinite-input cases (including DeepSeek nonleader logit lane1/bias lane33), and actual
invalid-geometry/alias refusal. Near Qwen/Gemma and DeepSeek also use actual
capture/reorder/replay with complete input and five-output range annotations. Every valid old-vs-witness selected ID/order and
weight bit must match. Old K+1 dispatch is used ONLY as an independent fixture
ID oracle where supported; its renormalized weights are ignored and it never
drives the instrumented model graph. Underflow shows why raw-logit differences
cannot substitute for actual rounded selection-domain scores.

Root ran the ignored control in THREE fresh processes to cover the actual
cached Gemma selector, preserving every failed result. Reproduction settings:

```text
HF2Q_FUSED_MOE_ROUTING_V3=0 HF2Q_FUSED_MOE_ROUTING_V2=0  # V1
HF2Q_FUSED_MOE_ROUTING_V3=0 HF2Q_FUSED_MOE_ROUTING_V2=1  # V2
HF2Q_FUSED_MOE_ROUTING_V3=1 HF2Q_FUSED_MOE_ROUTING_V2=1  # V3
```

For each process the exact test is
`ops::routing_selection_witness::tests::actual_selection_witness_preserves_ids_weights_and_domains`
with `--ignored --exact --nocapture --test-threads=1`; the final marker records
which ACTUAL variant returned. Root binds release/debug build scope, binary,
source, finite watchdog/quiet/owned cleanup and raw test output before execution.
The narrow controls below passed; full candidate checks/CI and downstream
models remain pending. Publication, exact HF pin and actual model capture/
endpoint review remain required before downstream Q1 capability qualification.

Retained evidence `native-routing-witness-root-build-v3` binds actual source
345af157e132705b72e3e51103c60d17404de4d9 and debug unoptimized libtest binary
`ef353a4a720155ea9fbc03416e734db5afc0f436313af7da2b88a3ea283afc71`.
Regular focused controls:3 passed/0 failed/1 GPU ignored. The ignored GPU test
was then actually executed once per GemmaV1/V2/V3 process: each1 passed/0 failed/
0 ignored; wall totals0.673640/0.266825/0.265395seconds. These are control durations,
not performance measurements. Each run covers4Qwen+3Gemma+1DSvalid cases and
6invalid cases, exact old/new selectedID/order/F32weightbits, guarded logical
views, checked drain and capture/reorder/replay annotations. Root execution
results and complete stdout/stderr are retained; independent review reproduced
hashes/case counts/source deltas. No independent CPU full-model score oracle or
model parity is claimed. The archive does not include separate PID/PG/recovery
receipts, so those unrecorded host conditions are not independently attested.

Preserve failures `native-routing-witness-root-build-v1` and `...-v2`: first
actual run reached the DS fixture's flat-shape refusal; the next compile failed
because the shape fix accidentally inserted a kind-dependent block in the
Gemma loop. Final345 removed ONLY those unintended five lines and retains the
correct DS `[rows,256]` input/`[rows,6]` output view metadata. The net source
delta from accepted witness kernels is fixture shape-only; no production math
repair or favorable numerical tolerance was introduced. Old failures are not
rescored. Version0.15.10 is not yet a published registry/package identity.
