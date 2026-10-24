# Optional bounded pool mechanism prototype

This additive prototype leaves `MlxBufferPool`, default allocation, kernels,
versions and downstream integration unchanged. `BoundedMlxBufferPool` creates one
fixed Shared/Automatic/explicitly Tracked heap through the one process-installed
`ProcessPoolAuthority`. Exhaustion, fragmentation, identity/property mismatch and
quota overflow refuse; there is no other heap/global allocation fallback or
external import. Authority coverage excludes legacy/direct/mapped allocations
and separately linked crate copies.

The authority serializes aggregate reservation and constructor attempts. Limits
bound domain count, reserved heap bytes, historical child slots, constructor
attempts, shape rank and batch length. Nil calls consume constructor attempts;
unpublished nil heaps can roll back only their domain/backing/child reservation.
A nonnil heap outside the declared envelope is quarantined and poisons further
creation. That postallocation check cannot prevent an already incurred driver
peak. Actual allocator/driver/OS metadata bounds remain unknown.

Each authority domain retains its actual heap, device, queue, residency owner and
all created child resources until process exit. Child descriptor capacity is
preallocated. No root retains an authority ticket, so there is no strong cycle.
Clear and pool Drop retire reuse eligibility only: they do not remove residency
membership, reclaim child backing, reduce usedSize by design, refund reservations
or restore historical root quota. Existing residency add/commit calls have no
OS success acknowledgement; the prototype does not invent one. A repeated
prefill/clear workload can exhaust physical child capacity or historical quotas
earlier than the legacy pool. DeepSeek prefill high-water capacity remains charged
even if subsequent decode is smaller. This is an explicit mechanism trade-off,
not a production capacity recommendation.

Root status has one membership record. Release consumes the matching non-Clone
lease; failure returns it intact. Drop fully destroys its owned buffer/shape payload before decrementing the
live-lease count, leaving the root Issued. A stalled or panicking destructor
keeps the charge; reset cannot admit a replacement payload while the previous
payload is still being destroyed. Escaped external clones remain outside this
count. Reset refuses while any lease remains live, then preflights all
issued roots before changing status. Free reuse checks actual resource properties
and device object/registry, queue and residency-owner identity before mutation.
Cheap matching device clones work; another queue on the same chip does not.

Release/reset are unsafe: all escaped MlxBuffer/view/raw/RecordedBinding accesses
and GPU work must be quiescent. Zero live lease values cannot establish that.
Raw makeAliasable, purgeable-state or mode mutation is prohibited throughout the
domain. Safe borrowing/mutable access to the buffer does not remove those
recycling obligations. Fresh ordinary allocations zero full bucket capacity;
reused roots preserve bytes. Unsafe uninitialized allocation requires full
producer coverage. Logical views remain distinct from bucket capacity.

Batch prevalidation precedes staging. Late failure publishes no partial vector;
unpublished successful staged roots become Free and remain charged. Shape/result
containers are bounded per call, but caller-retained empty containers and escaped
buffer clones are outside complete metadata coverage. Snapshot state counts are
per-pool; authority aggregate state counts are explicitly unavailable. Constructor
attempts are authority counters. No snapshot is a whole-process footprint receipt.

## Reproduction commands and historical execution — 2026-10-24

Run pure controls without Metal execution:

```sh
cargo test --locked --lib bounded_pool -- --nocapture --test-threads=1
```

The following separately ignored controls require explicit authorization and a
suitable Apple Silicon host:

```sh
cargo test --locked --lib bounded_pool::failure_tests::tiny_constructor_identity_and_retention_controls -- --ignored --nocapture --test-threads=1
cargo test --locked --lib bounded_pool::tests::tiny_tracked_pool_transitions_and_gpu_completion -- --ignored --nocapture --test-threads=1
```

The CPU failure control requests at most four 64 KiB heaps total (quarantined
heaps included) plus one initialized 64-byte standalone buffer for the mutable
lease identity negative control. Individual buffer requests are at most 64 KiB;
injected nil calls allocate nothing. These requested caps exclude driver, device,
residency and host metadata; they are not total physical allocation upper bounds. It covers the real production reservation/failure
boundary, not toy duplicate quota arithmetic. Predicate seams simulate property
drift without illegal mutation. Actual heap-full nil is separately checked.
Its required completion marker is `BOUNDED_POOL_FAILURE_CONTROL completed`.
`BOUNDED_POOL_RESIDENCY` reports whether active-owner coverage was available and
that the explicit None control ran; no environment toggles are used.

The GPU control requests two 64 KiB heaps total and four 1024-byte roots, using
native add then mul in two fully encoded command buffers committed consecutively
on one queue. Both completion results are collected before assertion/readback.
A fresh subprocess exercises public singleton installation; its exact test count
and explicit capability marker are checked. The parent must externally supervise
the process group with a timeout: synchronous GPU waits are not a watchdog.
Run with `MLX_UNRETAINED_REFS`, `MLX_PROFILE_DISPATCH`, `MLX_PROFILE_CB`,
`MLX_ENCODE_TRACE`, `MLX_PROFILE_BARRIERS`, `HF2Q_AUTO_BARRIER`,
`HF2Q_FORCE_SERIAL_DISPATCH` and `HF2Q_PIPELINE_TG_MULT_HINT` absent.

Zero exit alone is insufficient. Require `BOUNDED_POOL_COMMANDS cb1_ok=true
cb2_ok=true` and `BOUNDED_POOL_CONTROL completed` for actual GPU proof;
`unavailable_or_unknown` is a separate nonqualification outcome. This control
uses an independent exact CPU oracle, while the preceding separate tracked heap
spike compared standalone and heap arms. These are distinct evidence boundaries.
Neither proves arbitrary GPU ordering or performance. Explicit Tracked heaps may
incur coarser/heap-wide synchronization costs, requiring representative workload
measurement before adoption. Automatic heapOffset zero is a non-placement
sentinel, never a disjoint-offset proof. No models, speed, physical prebound,
production-pool acceptance or full Q5 admission is claimed here.


The historical pre-correction ignored controls completed on Apple M5 Max, macOS 27.2 (26B5112a), SDK
27.0, 16 KiB host pages. Both exact commands above exited zero without timeout,
with the reviewed source unchanged. The external supervisor covered the GPU
parent and fresh child as one process group. These results were independently
reviewed against the retained executable, source freeze and logs.

The CPU log emitted the completed marker with nil, quarantine, prevalidation,
identity, unchanged-failure, metadata-lease, batch and retained-child controls
true. Active residency-owner and explicit None controls ran; membership remained
retained. The successful mutable-wrapper negative control used the additional
64-byte initialized standalone allocation, returned the wrong-identity lease
intact with unchanged pool state, then restored and released the original.
Residency calls still provide no OS success acknowledgement.

The pooled GPU log observed actual heap bytes 65,536 and Tracked hazards, both
command completion statuses true, and the final exact-CPU-parity/transition
marker. Reuse followed both waits/readback and quiescence; retirement and later
domain creation did not refund reservations. This pooled control compares its
outputs to the CPU oracle only. The earlier separate heap spike compared both
standalone and heap arms; its result is not silently relabeled as a pooled-arm
comparison.

Historically, the seven pure controls passed (two ignored). Locked all-target/all-feature
check passed; the library regression reported 439 passed, zero failed, five
ignored, and the new modules passed rustfmt checking. Those ignored counts are
separate from the two explicitly executed commands and their completion markers.
The locked release build also passed (11.24 seconds as reported by the
integration owner). This build result is separate from the immutable execution
freeze and runtime receipts above; it is not a performance measurement.

### Historical pre-correction execution identities

Historical source base: `33fc6de0547bd551a83991842d34cae07f5e4e5c`. All identities below
are SHA-256. The execution freeze includes the earlier documentation hash;
this results-only documentation update does not change the tested source.

| Evidence | SHA-256 |
| --- | --- |
| Execution freeze receipt | `eadf7d00804c27535eea667fa3d30d015555218020c83aca759093ef49b741a1` |
| Runtime test executable | `188ae2a3e10c8c50c193a8ec2f51d98e436483f5cf40d5c29de42f8ce50958d3` |
| CPU failure-control log | `e09dcff76dd0793c3f3ba5fbaaf94e3838538d5dee1a6d1409ba92a7c28abb1b` |
| Pooled GPU transition-control log | `7a26c1dbbb0e5735571a668f816650daa3442d02d6d997ac667d81189cbf680f` |
| Pure-control log | `fe56a9422b132b8bb7ea39d5f03cf13d56a813e64b4f8899b0f4eccd5c9a34a1` |
| All-target/all-feature check log | `8240b1485db013aeb3bd074796dd1f12440457693ed68ce5ceb76668832d52bc` |
| Library regression log | `ca2ed6770e671af9efa95da039cbb1a54a24feea24c5b8177197c59dc92ed7c4` |
| `src/bounded_pool/authority.rs` | `00622c725c2c7b13686acf16be547e34e438e539a8f1209e1f2f254180c24395` |
| `src/bounded_pool/failure_tests.rs` | `a10e8b5e08623c7b720a1fb70f6772a73b0eb4c682630ea9904ea2552f3ee64c` |
| `src/bounded_pool/mod.rs` | `1eea25ef1286cc4811be134fabb411df357d0c6b75468f04dbfed6b3494810ea` |
| `src/bounded_pool/pool.rs` | `72de478a59c48405c4904f14edc025f8fd2e2ed42c6698ff88ab84615bfcfc5a` |
| `src/bounded_pool/resource.rs` | `ed444f2c52a9668340c8576e6d2d974861260357ac1b82b681ae0a7fd2d882c6` |
| `src/bounded_pool/state.rs` | `70e0c565cb2f88114d19818895f53a08b61181d6d5baf758743866c363be0581` |
| `src/bounded_pool/tests.rs` | `57263e6d13e41e087d008f5bb37b261c0127d0612bcbe3333e6e01c28e1017be` |
| `src/lib.rs` | `6fe99b636d213ec9c0dac9f14974de2cf95b92edfc9b17d1ee6de4ef9c2707ba` |
| `src/device.rs` | `51fa9a232939d7d1b23041c5ed6eaf9e91c5ede22892a80c2fa46aef9dbf357c` |
| `src/residency.rs` | `b886d8772706537d19a710c270f268bbb7b5fe2b77e2da9bc4e61d082660d981` |
| `Pre-execution documentation` | `80e4e2640ef43fcd1672ca1796330b76b1fbe4a9659ce91450cf62b84f42017f` |


These are narrow mechanism results. Whole-memory bounds, arbitrary GPU ordering,
model parity, performance, downstream adoption and full Q5 admission remain
unqualified. The fixed heap and retained child policy has not been shown to fit
production prefill/decode workloads or improve their speed.


After those execution receipts, a public-rustdoc-only polish clarified lifetime
retention, no refunds, fresh-versus-reused initialization and lease Drop. A
localized dead-code lint allowance documents the deliberately retained strong
device/queue/residency owner. No functional change is intended; final candidate
checks bind the resulting source separately. Historical hashes above remain
unchanged.


A later drop-order review suspended acceptance of the historical source above:
the old Drop decremented live count before automatic payload-field destruction,
allowing concurrent stalled payload destructors to outlive their charge. The
corrected lease stores an optional payload and explicitly takes and fully drops
it through a shared private generic helper before decrementing. A pure paused
payload-destructor witness observes charge one throughout the pause and zero
only after destruction returns; it always releases and joins its worker before
assertions. A panic witness requires charge to remain. The corrected-source evidence below supersedes the earlier acceptance; prior
passing receipts remain historical and do not qualify the correction on their own.


## Current corrected-source results

Current base is `df09d313aac9b9bad1f5a9fccb9af725c76c2966`, including merged
native issue #49. This is distinct from the immutable published 0.15.2 source
`7bb30a2fac783e0b340ec2f2086289f9751dda7a`. This optional pool API is unpublished;
no release or downstream adoption is asserted.

The before-fix production-helper witness ran exactly one pure test and failed
as expected with exit 101: during the paused payload destructor the live charge
was 0, versus the required 1. It performed no GPU work. After reordering payload
destruction before decrement, both focused pure tests passed: the paused test
observed charge 1 and the actual runtime `reset_allowed` predicate rejected with
LiveLeases throughout the pause, then charge 0/reset allowed after destruction.
The controlled destructor-panic test kept charge 1. Worker release/join precede
all paused-test assertions. Run this pair separately:

```sh
cargo test --locked --lib bounded_pool::drop_order::tests -- --nocapture --test-threads=1
```

Both exact ignored controls listed above then passed again on the corrected,
merged source on the same Apple M5 Max/macOS 27.2/SDK 27.0 host. CPU completed all
failure/identity/retention assertions, including the replacement-buffer negative;
active residency and explicit None controls emitted completed retention markers.
The pooled GPU control observed a 65,536-byte Tracked heap, both command buffers
successful, exact CPU oracle parity and completed transitions. This remains a
CPU-oracle comparison, not a new standalone-versus-pool comparison. Requested
envelopes remain four 64 KiB heaps plus 64 B standalone payload for CPU, and two
64 KiB heaps for GPU, excluding host/device/driver/residency metadata. No timeout
occurred, tested sources were unchanged and independent postexecution review
verified the retained executable/log/receipt identities.

Corrected-source locked all-target/all-feature check passed. The full library
regression passed 441 tests, zero failed, eight ignored. These counts supersede
the historical 439/five result above; explicit ignored execution and required
markers remain separate. The corrected-source locked release build also passed
(4.87 seconds); the earlier 11.24-second pass belongs to historical source.
No model, performance, whole-memory bound, production fit, adoption or full Q5
qualification follows from these results.

### Corrected execution identities

All identities are SHA-256. This results-only note changes documentation after
the execution freeze; the tested Rust source remains unchanged.

| Evidence | SHA-256 |
| --- | --- |
| Corrected execution freeze v5 | `e41a4a1bb85af481e21df387cf8794c5cf5685f1b9de123cf165ec234eff306e` |
| Corrected runtime test executable | `1fec234aa244f481ce1063b21ab5c89c7cbb30aeef50ea0ee0e9f559fbbb59c3` |
| Corrected CPU control log | `db3b3b0d9f62ae0a70c84ae75ba4847708cf4e9081576d2a7553ffbd789610a9` |
| Corrected pooled GPU control log | `484b2b7e58b1fd17856da00044d8bb539100ca867ac1c1e6d7383fd958816567` |
| Expected before-fix failure log | `e44937eb638ede05b80552acad1c0315a3f1d43399ed6bd995cf002ee06ac4d6` |
| Before-fix helper | `7e053d85c0c45d3763a714d2ef4c7ccd2b87db787db67d7f99987b61e36f57fe` |
| Focused after-fix pure log | `acde97280cbc5c1a00dd1bca54e8b5241d7f710c2a58020fc6c7ea05520d4550` |
| Corrected all-target/all-feature check log | `659e7cbaaf8bd50b8e5fa1f113e73e906e8afb8ba33b562856a1353af706e0c2` |
| Corrected library regression log | `2faa3bae1fea085f8e62e166a044520f7647048d169fef01b5d36f284e603fb5` |
| `src/bounded_pool/drop_order.rs` | `acf16f25f9319876cca3754d9f911f4a2dcf07ac9019420656b7f77b925ec481` |
| `src/bounded_pool/resource.rs` | `ed444f2c52a9668340c8576e6d2d974861260357ac1b82b681ae0a7fd2d882c6` |
| `src/bounded_pool/mod.rs` | `f87333de97306d4eab0eee75072c4dd3ec605c2639a2d46bfa350869a2102eb1` |
| `src/bounded_pool/state.rs` | `70e0c565cb2f88114d19818895f53a08b61181d6d5baf758743866c363be0581` |
| `src/bounded_pool/pool.rs` | `7ca96946f2ea7e3cbd03e90287cc34700c76eafa7c990b7bc69ad5a0e370aeae` |
| `src/bounded_pool/tests.rs` | `57263e6d13e41e087d008f5bb37b261c0127d0612bcbe3333e6e01c28e1017be` |
| `src/bounded_pool/authority.rs` | `e2ef37cabb7e3ebd987d716b4d9f3bf04035a88add2400e174afeb3fe3f594c6` |
| `src/bounded_pool/failure_tests.rs` | `a10e8b5e08623c7b720a1fb70f6772a73b0eb4c682630ea9904ea2552f3ee64c` |
| `src/lib.rs` | `9f51464247089fc64d06134842d321454b9e7b17acd63b3e75c407f6e8752be9` |
| `src/device.rs` | `5269446b8f23a50649aea930be418f7648958097a8c51b912708574dd0159ea1` |
| `src/residency.rs` | `e5969e12e61c8e8c5a76a3d84cda22c98b70dd586997bbdf034233901e5ddf9a` |
