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

## Observed asynchronous owners (source candidate, not yet published)

A submitted observed command may outlive its Rust encoder. In particular, a
caller may use `commit_labeled`, drop that encoder, and consume its output in a
later command. Encoder destruction and command-buffer rotation now transfer an
unchecked work-bearing observation and actual retained Metal command buffer and
queue into an observer-only pending registry. Ordinary observation-`None`
execution does not allocate or consult this registry. No kernel, queue order,
production wait or default activation changes.

The registry has a process/core maximum of 1024 outstanding commands and also
precharges each retained entry against the configured observer metadata quota.
It holds actual owners, not copied tensor payload. Requested Rust entry/vector
metadata is charged; driver/allocator overhead and prolonged tensor retirement
are not a physical-memory bound. A failed count/metadata/vector admission or
poisoned/contended registry invalidates capture; unchecked owner destruction
records Unknown. There is no unbounded fallback. Retained outstanding owners may
survive until process exit if no later suitable checked wait occurs.

Only an existing `wait_until_completed`/checked completion boundary on an
observed encoder triggers inspection. Its queue identity selects candidates;
it does **not** supply their status. Each actual retained command buffer is
queried independently. Only actual terminal Completed or Error statuses produce
that command's checked completion event; Error also invalidates observation.
Other statuses and different queues stay retained. No sleep, poll-until-ready,
extra GPU wait or inference of health from the last queue buffer is added.
A finite ID watermark and bounded scan prevent concurrent submissions extending
one inspection indefinitely. Status queries under the short registry lock never
invoke an observer sink; terminal ownership/release, callbacks and event
publication happen after unlocking. The count charge survives actual payload
and observation destruction.

`ObservationStatus.pending_commands` reports retained owners and
`pending_command_limit` reports the finite limit. A complete final diagnostic
window requires pending_commands=0 in addition to unchanged zero Unknown/drop
and valid capture counters. Missing final status is Unknown, never zero. These
additive fields require the HF consumer's strict status schema to be updated
when a new public native release is pinned. They do not attest physical memory,
HF logical rows, whole-forward completion, model quality or performance.

The new actual ignored control uses the production async `commit_labeled` path,
drops its producer encoder, then performs an ordinary consumer checked wait.
It requires both actual command IDs to have one terminal checked completion and
checked Drop, all eight CPU-expected F32 values, zero Unknown/dropped events and
zero outstanding owners. Payload is four 16-byte roots (64 B); the separately
predeclared 4 MiB userdata/metadata allowance excludes device/driver/compiler
bootstrap. It must run in a fresh bounded process; authored source is not an
actual device result. The old 0.15.7 whole-model failed trace remains unchanged.

### Explicit rotation fallback (source candidate)

`CommandEncoder::set_observation_rotation_scope(Option<ObservationScope>) ->
Result<()>` declares a separate same-epoch fallback for command-buffer rotation.
It requires an enabled observer; a wrong epoch refuses, clears the saved
fallback and records Unknown. `None` explicitly clears it. Default None remains
Unknown after rotation. The caller must bind an independently declared opaque
auxiliary scope, not a Known RMS/tensor-row scope. Native scope IDs do not encode
HF membership: the HF strict source/manifest consumer must validate that this
fallback is auxiliary.

Rotation creates a fresh command ID and clears capture/replay/dispatch state;
only this explicit fallback becomes the new current/future rotation scope.
It never copies the previous CURRENT Known scope. HF constructor and attachment
adapters must each bind a fresh `auxiliary(epoch)` separately from CURRENT.
No GPU instruction, route, synchronization or scheduling changes.

Three hosted controls check the actual rotation helper: stale/cleared/missing
fallback and removal of current Known/capture/replay state. Two distinct ignored
fresh-process actual EncoderSession controls call `commit_stage` then
`reset_for_next_stage` and a real consumer checked wait. The positive retains
only explicit fallback; the negative deliberately leaves it missing and must
observe one Unknown, never qualified metadata. Each uses64B/four roots/two
existing add operations/eight CPU-expected values under the same4MiB declared
allowance and20s external watchdog proposal. `HF2Q_ENCODER_SESSION=1` is required;
these authored controls have not been run by the author.

## Async ownership and rotation validation

The production asynchronous submit/drop path and EncoderSession rotation were
exercised on Apple Silicon at source `4ef880dab29fb0445f25b60adb3762b4349376fe`
with a release-built test executable. Five pending-registry controls and three
rotation controls passed. Three separately launched actual Metal cases each
ran one ignored test with 64 bytes of payload and eight exact output values:

- Async encoder drop retained one pending owner until an existing consumer wait;
  both actual command buffers then reported checked Completed and owner drop,
  with zero pending owners and zero Unknown observations.
- Explicit rotation fallback preserved the separately supplied scope, checked
  both command buffers and retired all owners with zero Unknown observations.
- Missing rotation fallback preserved an Unknown observation and remained
  unqualified despite correct arithmetic and checked command completion.

These are finite primitive controls, not model, full-forward, utilization or
performance qualification. The original failed error-event control is retained
in the experiment record; the correction records the actual Error status before
sticky invalidation. No new GPU wait or kernel dispatch was introduced.

Reproduce the actual cases separately in fresh processes:

```sh
cargo test --release --locked --lib submission_observer::tests::async_tests::submission_observer_async_drop_then_actual_consumer_checked_wait -- --exact --ignored --nocapture --test-threads=1
HF2Q_ENCODER_SESSION=1 cargo test --release --locked --lib submission_observer::tests::rotation_actual::submission_observer_explicit_rotation_fallback_actual -- --exact --ignored --nocapture --test-threads=1
HF2Q_ENCODER_SESSION=1 cargo test --release --locked --lib submission_observer::tests::rotation_actual::submission_observer_missing_rotation_fallback_unknown_actual -- --exact --ignored --nocapture --test-threads=1
```

Leave profiling overrides unset so the first case exercises genuine asynchronous
submission. Confirm exactly one test ran in each process. Crate 0.15.8 carries
this ownership fix; observation remains absent by default.
