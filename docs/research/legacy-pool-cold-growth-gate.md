# Optional legacy-pool cold-growth gate

Optional mechanism candidate for Native #53 / hf2q ADR-063 Q5. The initial
quiescent controls have separate historical execution receipts. The ordered
extension also passed its separately supervised tiny control. The independent snapshot
audit is complete; exact final-commit acceptance remains pending; no model admission, selected
numeric fit or performance result is claimed.
No HF policy adoption or default legacy allocator change accompanies this API.

`LegacyPoolGrowthAuthority::install` installs exactly one immutable authority per
linked crate copy. Configurable process/pool/exact-bucket ceilings cover historical
requested bucket bytes, successful roots and constructor attempts. There are no
model-sized defaults. Pool slots, bucket records, generation quarantine records,
rank and batch length have configured finite limits. Registered device/queue and
residency owners are strongly retained; key/slot reuse and authority reset are
unavailable. Other crate copies and allocations are outside coverage.

`bind` constructs an empty `GrowthLimitedMlxBufferPool` on its registered thread.
The private facade exposes no inner pool, `Deref`, import, per-buffer release or
fallback. Legacy public methods/signatures remain available on ordinary
`MlxBufferPool` and retain their existing behavior. Every facade allocation goes
through that pool's actual `alloc_inner`. Device object/registry, queue, residency,
thread, generation, rank, checked bucket and Metal maximum length are validated
before a free pop. Allocation selection and generation quarantine share one lock;
a previously selected operation may finish after quarantine.

Warm exact-bucket reuse costs zero new payload/root/constructor credit. A cold
miss atomically precharges all three scopes before the same
`MlxDevice::new_shared_buffer`. At this source boundary that helper returns Err
only before creation or on raw nil. Nil rolls back unpublished bytes/roots but
keeps the attempt. Nonnil commits historical charge before zeroing, residency or
fallible logical-view construction. Panic retains precharge and poisons the
mutex. Any later error keeps creation history, with no uncapped retry.

Pool free-map entries, per-bucket vectors, issued vector and residency map reserve
configured capacity before registration; `clear` preserves container capacity.
The bounds cover their cardinalities, not host allocator size or opaque driver
metadata. Repeated handouts still create buffer/view/shape metadata, and callers
can retain snapshots, clones or completed commands. These histories are NOT
bounded by successful-root ceilings. HF finite handout/command lifetimes remain a
separate prerequisite. No whole-process or physical-memory bound follows.

`unsafe reset` has the existing arena lifetime requirement: no future CPU/GPU or
recorded accesses to issued handles/views/raw aliases, and submitted uses must be
quiescent. The gate does not discover completion, validate replay or add barriers.
`clear` refuses issued entries, retires eligible free roots and keeps all creation
credit consumed. Pool Drop and model/worker restart also provide no refund.
Quarantine refuses reuse as well as growth, including later registration of that
generation. Quarantine registry exhaustion refuses all pools while leaving
semantic-poison receipts readable. Ordinary mutex poisoning refuses snapshots too.

A bounded batch preflights rank/bucket/identity and length, then delegates actual
legacy batch allocation. If a later cold miss fails, earlier successful history
and arena tracking remain; the call does not publish partial output or refund it.
It must be safely reset before clear. Imports are impossible through this facade;
escaped returned buffers cannot be released back through any public facade API.

## Initial controls (historical execution receipts kept separately)

Pure quota tests use the production `Ledger::construct` decision boundary. They
cover shared concurrent process ceilings, independent byte/root/attempt ceilings
at every scope, nil rollback with attempt retention, fallible consumption after
successful creation, overflow, poison, finite registration and quarantine.

```sh
cargo test --locked --lib legacy_growth::tests -- --test-threads=1
cargo test --locked --lib legacy_growth::allocation_tests::actual_legacy_aababa_growth_gate -- --ignored --exact --nocapture --test-threads=1
```

The ignored control requires a fresh isolated process and available Metal. It
requests at most two actual new Shared roots: 64 and 128 bytes, total 192 bytes.
This is a requested payload envelope, excluding driver/allocator metadata. It
executes no command encoder, GPU dispatch, kernel or model. A,A,B,A,B,A verifies
fresh whole-bucket zero and initialized reused contents/identity; it also exercises
uninitialized reuse, bounded batches and late partial failure, clear/drop without
refund, wrong thread/queue/residency/rank/bucket/overflow refusal, re-registration
and quarantine. Actual resource fields are checked before CPU read/write. All
returned owners drop before the unsafe reset assertion. Require the exact test
count, process success and `LEGACY_GROWTH_AABABA_COMPLETE` marker; a zero-filter
run is not evidence. Device absence fails rather than claiming completion.

HF integration must separately register actual workers and direct allocation
roles, reject inactive bypass, prove existing cleanup/drain/no-future-use paths,
and select retained-bin/decode policy under reviewed numeric caps. This primitive
changes none of those behaviors and cannot qualify full Q5.


## Separate ordered GPU reuse extension

`unsafe reset_for_ordered_gpu_reuse` is distinct from quiescent `reset`. The
initial contract permits exactly ONE unsubmitted, one-shot GraphSession on the
registered queue, consumed once by finish/finish_with_reorder. All outgoing
accesses must already be encoded, outgoing Rust handles dropped, with no later
old-epoch append/access or independently replayable segment. Retained native
capture bindings remain inside that single lineage. Complete physical RAW, WAR
and WAW dependencies and actual barriers order all outgoing reads/writes before
incoming overwrites. Same queue alone is insufficient; reordered graphs discard
literal barriers and must preserve alias dependencies from complete annotations.
No CPU read/refill/write/zeroing of recycled roots is permitted before completion.
The new GPU producer must fully initialize every subsequently read byte. Unknown
encoding/submission/drain ownership requires quarantine, not assumed completion.
CPU submission inputs and already-submitted/multi-command lineages are excluded.
No wait, barrier, scheduler change or HF adoption is added by the method.

```sh
cargo test --locked --lib legacy_growth::ordered_gpu_tests::actual_ordered_gpu_alias_reuse_and_standalone_lineage -- --ignored --exact --nocapture --test-threads=1
```

Run this exact case in its own externally supervised fresh process with capture,
profiling and unretained-reference overrides absent (the test checks them). It
requires the exact one-test result, successful completion and
`LEGACY_GROWTH_ORDERED_GPU_COMPLETE` marker. Metal absence fails. The source packet
requests ten actual roots totaling 640 bytes: three 16-F32 immutable inputs, two
pooled-arm output roots, one pooled scratch root shared by A/B, and four separate
standalone-arm roots. Nine fixture roots are deliberately outside the authority;
its configured quota is one 64-byte root/one attempt. Eight dispatches carry four
inline bytes each, not additional Metal roots. Scoped numeric host storage stays
below 4KiB. Registry/pipeline, capture/DAG, command, allocator and driver metadata
remain Unknown; requested bytes are not a physical allocation prebound.

The GPU computes A=X+Y, C=A*Z, then aliases B onto exactly the same object, byte
offset and physical range after the ordered reset; B=X*Y, D=B+Z. X=i-8,Y=3,Z=2.
All inputs, outputs and independent standalone roots are initialized before
encoding; B is entirely GPU-written with no CPU refill. Every dispatch declares
complete actual range annotations. Real finish_with_reorder must report the
aliased sequence unmoved with three derived boundaries, while the independent
standalone lineage must actually move nodes and use fewer boundaries. This is
scheduler-path evidence, not a claim that the explicit barrier survived reorder.
All fallible encoding precedes each submission and checked finish waits before
output reads. All native resources remain retained through completion.

Both arms must match independent exact F32 CPU C/D values and each other, finite
and canary-free. The same comparison deterministically rejects the hypothetical
premature-overwrite C=(X*Y)*Z without executing a GPU race. The observed pass covers
only the named tiny one-shot alias lineage. It cannot prove all DS node
annotations, mixed CPU inputs, multigroup replay, abort cleanup, memory admission
or performance. Existing quiescent evidence remains a separate contract.


## Validation recorded on 2026-10-24

The quiescent control is historical v1 evidence: exactly one ignored case ran
and passed, exit 0, with `LEGACY_GROWTH_AABABA_COMPLETE` reporting two roots,
192 historical requested bytes, two attempts and no GPU dispatch. Its retained
binary SHA-256 is `a950434d1f53222a369e273f92050185d3ae066ba64310079158fd3e76f78910`;
raw log SHA-256 is `17b5a30ee7d6b73d518e8599830abae721e7d5d82e8d74da0806bdb809c1e581`.
The nine-file author freeze remains separate from the v2 extension.

The v2 ordered control also ran exactly one ignored case and passed, exit 0,
without timeout or skip. Its completed marker reported ten roots/640 requested
bytes, one pooled root, pooled moved=0/barriers=3 and standalone moved=2/barriers=1.
Independent exact CPU values, standalone parity and deterministic wrong-oracle
rejection all passed. Its retained binary SHA-256 is
`ade0e19efa55885685706b1cc762a1cf027977331d6b9336969b9383bbb76aac`;
raw log SHA-256 is `76730b9a61b5359479911a92f2266be06dbbee45abcdc1e6c27c5a3427df27dd`.
The ordered test source SHA-256 is
`b30346e612d0d8daf0cccdcec5bf0031aaead8069bef577bc0bbda9e2cd9ee8d`;
the facade source SHA-256 is
`062e31a6c5e20d27f169a19b1e1df579f53f8bb75f7966447bd660e427ef3327`.
Independent postexecution reviews verified both controls' source/binary/log
bindings and narrow results. These are separate from final candidate acceptance.

| v2 validation | Result |
| --- | --- |
| Eight focused pure quota controls | PASS |
| `cargo check --locked --all-targets --all-features` | PASS, exit 0, 10.56s |
| `cargo test --locked --lib --all-features` | PASS, 454 passed / 11 ignored, 1.32s |
| `cargo build --release --locked` | PASS, exit 0, 11.61s |
| Reference and package hygiene checks | PASS |
| Targeted rustfmt check of seven new Rust modules | PASS |

Ignored library counts are not execution evidence for ignored controls; the two
separate supervised results above supply that evidence. The independent Opus
read-only audit of the immutable v2 source snapshot
completed with exit 0 and empty stderr, finding no concrete quota-gate blocker.
It did not run tests, inspect `src/residency*` or `src/buffer.rs`, verify the
identity implementation's uniqueness or logical-view failure internals, or qualify
the ordered caller contract/HF integration. Independent source review and the actual tiny
controls are separate evidence, not coverage supplied by that snapshot audit.
Requested bucket payload does not bound physical page allocation or returned
metadata. The two ignored controls MUST run in separate fresh processes because
each installs the singleton. Model contexts, four-slot workloads and profitable
modes are unchanged. No HF adoption, numeric campaign budget, returned-metadata or
opaque-G bound, real-model trial, performance win or full Q5 qualification follows.
