# Prospective routing selection witness (source spike)

This unpublished source spike adds explicit diagnostic dispatch APIs. It does
not change the default routing APIs, their kernel names or arithmetic. No new
crate version is assigned here: the base is0.15.9/4fee3a5, and root coordinates a
later release only after independent review and meaningful validation. Nothing
here qualifies an HF model, Q1 trajectory or performance result.

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

## Finite deciding controls (AUTHORED, NOT RUN)

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

Root must run the ignored control in THREE fresh processes to cover the actual
cached Gemma selector, preserving every failed result:

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
The new controls, crate/Metal compilation, and downstream graphs remain unrun.
Publication, exact HF pin and actual capture/endpoint review are required before
marking the Q1 selection-margin capability available.
