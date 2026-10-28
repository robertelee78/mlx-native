# Opt-in command submission observation

The `submission_observer` API records native command encoding, real submission
and checked completion. It adds no kernel, wait or scheduler policy. With no
observer installed, existing inference behavior is unchanged.

Install one bounded sink before creating the first native device or encoder.
Explicitly enable each observed encoder with a fresh invocation epoch and set
an opaque operation scope before encoding the operation. `EncoderSession`
exposes its encoder for the same purpose. Scopes identify caller operations;
native events do not establish model requests, slot generations or physical
batch membership. That join belongs to the consumer.

Events contain process-local command/queue/pipeline identities, device registry
identity, POSIX `CLOCK_MONOTONIC` intervals, dispatch kind/grid/threadgroup,
opaque scope and replay provenance. The real checked Metal status is retained.
`Submitted` is not completion, an empty command buffer is not forward progress,
and waiting for a later buffer does not prove an earlier buffer was drained.

## Completeness and replay

The installation status is not a completeness certificate. Bind a baseline and
terminal status to the same measured window. Refuse nonzero unobserved-commit
deltas, unknown required scopes, tainted commands, dropped events or invalid
capture. Require actual `Completed` status for every required submission.
`comparison_qualified` is always false: model quality, physical rows and
comparison eligibility cannot be certified by this crate.

An observed capture carries private per-node metadata; the public `CapturedNode`
enum remains unchanged. The first replay into its owning command may use the
single-use capture binding. Later replays, including in the same epoch, require
an explicit map to the current invocation. Missing, stale or ambiguous mappings
remain Unknown. Reordering keeps attachments aligned. Mixed-scope fusion and
tag/node misalignment cannot become a fabricated known scope. A replay override
does not leak into a subsequent direct operation.

Requesting the raw Metal command buffer taints that command. Internal dual-buffer
children are not automatically opted in; their unobserved submissions remain
Unknown. A consumer must instrument and join those children before qualifying
that route. An unchecked command discarded during rotation remains Unknown;
no synthetic drain is emitted for it.

## Bounds and sink behavior

The install call bounds event count and reserved observation metadata. Capture
and replay maps also have a finite 4,096-entry ceiling. Exhaustion fails closed
and is observable; it must never be treated as an empty trace. A small-operation
test does not prove that a model graph or full coding session fits these limits.
Use independently bound finite control runs and inspect their actual counts.
Do not silently sample events or reset failed status to obtain qualification.

The callback runs outside native resource locks. A sink must remain bounded,
use nonblocking acquisition such as `try_lock`, and avoid native reentry. Sink
errors, reentry and caught unwind panics invalidate observation without changing
the GPU operation or its checked completion result. Installation refuses
panic-abort builds. Do not rely on catching a panic as a memory safety boundary.

## Validation scope

The focused local-state tests exercise replay reuse, direct-operation scope
restoration, tag alignment, raw-buffer taint, quotas, unobserved commits and sink
failures. The ignored two-operation GPU test must run alone in a fresh process
under an external deadline and process-group cleanup guard:

```text
submission_observer::tests::submission_observer_recorded_two_operation_actual
```

That test checks two real F32 additions, their distinct opaque scopes and actual
geometry, one real submission and one checked completion. It asserts zero
unknown/dropped/unobserved metadata and released observation reservations.
Its requested-data envelope is 4 MiB; driver/bootstrap memory is a separate
unknown quantity. It establishes no physical batch-four, model admission,
model-scale trace capacity or performance claim. Model consumers still need
their own configuration-bound correctness, capacity and overhead evidence.
