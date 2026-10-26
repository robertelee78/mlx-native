# Optional live and pending requested-root extents

Source candidate; no author build/test/device/model execution. Additive to
NativeRootAuthority's immutable historical quotas. Legacy default and
install_early(history) remain history-only. No HF adoption, version or publication.

`install_early_with_live(history, LiveRootLimits)` installs nonzero live+pending
byte/root caps atomically before any native creation/import. Register actual
devices through the existing handle; different queues/TLS share the process state
mutex. `live_snapshot()` reports pending/live requested bytes/roots, bytehighwater,
denials and actual final-ticket releases. Disabled is limitsNone/coveredfalse,
not zero active usage. Poison returns error/unavailable, never admission credit.
Existing historical quotas never refund even when live charge ends.

Charge is requested MTLBuffer extent, not allocatedSize, unique physical pages,
OS footprint or all process memory. Shared/view/raw/RecordedBinding/command owners
add no extra payload. NoCopy uses actual page-rounded resource length; overlapping
resources may conservatively doublecharge and CPUmmap owners can remain after
Metal release. No unique NoCopy union/refund, metadata/driver/pipeline/command/
heap/foreignroot/CPUcoverage claim. Heap remains refused in coveredmode; unknown
public import poisons. No generic cachedfile credit or GPUcompletion refund.

## Resource lifetime and failure contract

The same process mutex atomically checks history plus live+pending quotas before
raw creation; denial never invokes factory. Metadata ticket and copied block are
created before raw allocation. The block captures only ticket, never Metal/device/
queue. Origin and actual ObjCresource association share it, so the final ticket
cannot drop while either remains. Ticket holds Weak references to the same state
and poisonflag, avoiding authority/resourcecycles. Release takes only state lock,
never calls Metal or panics; resource/blocklastDrop is outside that lock.

Before association each actual resource is restricted to observed capability:
ObjCclass AGXG17XFamilyBuffer, device Apple M5 Max, kernel-reported productversion
27.2/build26B5112a. Two OSqueries are bounded64B each, direct sysctlbyname, no subprocess. This intentionally
narrow prototype refuses unseen classes/platforms conservatively; tinyclass proof
must not be generalized to every large/NoCopydriverresource. Unknown eligibility
means noassociationattempt, rawsuccess preserved, charge retained uncertain,
authoritypoisoned/futureallocation denied. Class support is not an external caller
snapshotcertificate. ObjC fatal/exception cannot be recovered by RustResult.

Private static key is attached once with RETAIN_NONATOMIC to the alreadycopied
block and exactpointerreadback. Duplicate/mismatchedreadback poisons without
changing nonnil result into Err. This matters because Native55 considers every
factoryErr no-root-created. Postrawcountermutexfailure likewise preserves rawOk
and uncertain pendingcharge. Wrapping/publication checks then reject poisoned
coverage; do not replay uncapped. Existing publicationguard remains effective.
Proven nil/Err cancels only pending live charge, retaining historical attempt.
Unwind/uncertain raw outcome conservatively keeps charge. No timeoutrefund.

Actual resource destruction plus finalOrigin release returns live credit once;
commandcompletion or poolclear alone does not. Covered authority retains devices;
driver retirement may delay forever, and cap can then reject subsequent requests.
No promised retirementdeadline or usability/fit result. V4 tinyretainedblit
Completed but finalretentionUnknown; V5 observed finalrelease already BEFORElocal
deviceDrop. Neither supports queueDropcausality or an immediate release guarantee.
CPU Shared/NoCopy raw/recorded capability proof is separate historical evidence.
No failed-spike test is added to production candidate.

## Root-owned validation packet

Pure filter: `cargo test --locked --lib root_authority::live_tests -- --nocapture`.
Eleven actual-core controls: origin/resource final-owner join; nilrollback/history;
denybeforefactory/overflow; pendingrace; postrawuncertaintymustpreserveOk;
rawunwind; postrawmutexpoison; legacy/history-onlycompatibility; actualeligibility/
attachmentdecision unknown/readbackfailure, same-product/different-build rejection,
and deterministic blocked allocation/snapshot waiters poisoned before lock acquire. These are source assertions until root
runs them. Existing root_history_ controls must also remain passing.

Two ignored actualCPU controls, each selected in its OWN fresh process:

```sh
cargo test --locked --lib live_root_shared_actual_raw_recorded_control -- --ignored --nocapture --test-threads=1
cargo test --locked --lib live_root_nocopy_actual_extent_control -- --ignored --nocapture --test-threads=1
```

Shared: one4096Broot, byte/rootcap4096/1, secondrequest1B denied before rawattempt;
raw+RecordedBinding retain charge after wrapperdrop; finalrelease observedexact1.
NoCopy: one privatehostpage file/readonlymmap, logical4096offset64, page<=64KiB,
outerbackingescrow outlives allMetalowners evenonunwind; rawowner retains resource
and backing, finalticketrelease1 while CPUescrowstillowned, proving scope excludes
CPUbacking. No payloadreads/GPU/CB/kernels/models. Defaultdevice constructed but
used wrapper explicitly disablesresidency; no global environment mutation.

Root must freeze exact source/binary/OS/class and supervise each process20s,
TERM2/KILL3/boundedreap, verify temporaryfilecleanup. Require exactly1selectedcase
and matching BEGIN/COMPLETED marker, exit0 and exact numeric assertions. Unsupported
platform/retainedUnknown/panic/signal/timeout/cleanupfailure=no-credit, not favorable
retry. Requestedpayload bounds exclude fixedcounttestmetadata and unknown driver
allocation. No publicmodelstress until separate trusted HF sourceledger, CPU/G/J,
OSobserver/watchdog and actualfamilyqualification are approved.
