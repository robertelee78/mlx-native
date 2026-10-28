# Optional actual Gemma batch routing selection witness

Prospective additive API, not yet published. No version change or runtime
claim accompanies this source. HF must use a verified published dependency,
exact pin and independently reviewed consuming graph before claiming all144
Gemma routing coverage.

`routing_selection_witness::dispatch_gemma_batch_with_witness` handles the
actual batched-prefill producer. None calls the unchanged ordinary batch API;
Some drives the SAME selected IDs/weights used by consumers and additionally
writes caller-owned F32[rows,2] scores/U32[rows,2] boundary IDs/U32[rows] status.
The optional producer returns actual kernel name and GemmaV1 or GemmaV3
algorithm family. There is no batchV2 kernel, even when scalarV2 is selected.
The old and diagnostic producers share the same cached batchV3 selector;
existing default kernel names, reduction/tie/renorm/scale arithmetic and
threadgroup/shared-memory geometry are preserved. Public enums/layouts remain
unchanged. Scalar replay is not used: the batch reduction is its own path.

Scored status1 exports kth and maximum unselected actual softmax-scratch
scores before selected-weight renormalization and per-expert scale. K==E is
status2 with explicit missing kth+1 sentinel, not zero margin. Nonfinite inputs
or per-expert scale produce row-local status3/Invalid with fully written NaN/
IDMAX evidence and selected outputs. Shared scale invalidity invalidates every
row. Do not consume Invalid IDs in expert execution; the caller must treat the
failed route as invalid diagnostics. Rust rejects unsupported dimensions,
zero rows, u32 index overflow, wrong dtype/extent/alignment/writability and
aliasing before encoding the optional path.

No allocation, commit, wait, extra softmax or CPU reconstruction is performed
by the API. The caller owns outputs and can read only after its existing
successful checked completion; captured graph metadata records actual inputs
and all five output ranges. HF row/layer/slot identity, lifetime/reservation,
full-workload coverage and model endpoints are downstream responsibilities.

Authored deciding controls: one host dimension control uses the actual pure
admission helper; one ignored actual Metal test covers rows1/4/128/512 with
ties/near/scaled weights/allselected, exact old/new IDs/order/weightbits,
actual existing batch K+1 ID oracle, bounded offset views/sentinels, typed
nonfinite at nonzero row/lane and scale33, alias/dimension refusal and actual
capture/reorder/replay. Replay includes an explicit inter-call WAW barrier
inside the test after capture extraction. Both cached batch variants require
separate fresh processes with HF2Q_FUSED_MOE_ROUTING_V3=0 and1; results record
the returned kernel, not an inferred flag. No device-unavailable success path.
Each fixture drops after checked wait, below2MiB combined host/GPU user-data
request by its fixed shape; process/compiler/driver extents are not that bound.

No controls/builds/Metal/model have run at author freeze. Tiny old/new fixture
parity, when actually proven, would still not establish HF model endpoint
parity, numerical tolerance, all144 routing evidence, Q1 completion or speed.
