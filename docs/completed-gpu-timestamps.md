# Completed command-buffer GPU timestamp metadata candidate

Apple documents GPUStartTime and GPUEndTime as host seconds relative to system
mach time, unavailable/zero until completion:
https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime
https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime
The actual SDK MTLCommandBuffer.h declares both readonly CFTimeInterval properties
(macOS10.15+). This candidate names that domain explicitly, never CLOCK_MONOTONIC,
POSIX time or per-dispatch counter ticks. Mapping to a CPU/F2 window still requires
separate bounded calibration/source proof. An execution envelope is not SM busy
time, occupancy, shader utilization, physical B4, or unique physical memory.

CommandEncoder::completed_gpu_timestamps reads THIS current CB status and, only
when Completed, its raw endpoints. It does not commit, wait, rotate, escape the
CB, mutate counters or certify an observer drain. Read before rotation/drop.
Finite positive strictly ordered endpoints yield Available; Error/nonterminal,
zero, nonfinite or unordered properties yield typed Unknown (invalid read values
retain their f64 bits). Existing truncated duration APIs are unchanged.

An explicit install_submission_observer_with_gpu_timestamps enables a separate
CompletedGpuTimingEvent schema_version1 callback. Existing installer stays off;
existing SubmissionEvent fields/CPU clocks/schema are unchanged. A default trait
method makes old implementations source compatible; if a caller opts in without
implementing that callback, its error invalidates capture rather than silently
losing timing. A per-observation atomic emits one terminal attempt. Direct checked
wait and existing retired-CB registry read the same actual healthy CB before release;
registry ownership/count limits/queues/waits remain unchanged. Empty, Error,
tainted and abandoned observations emit Unknown without inventing endpoint zero.
Unsubmitted Drop also emits Unknown. Unknown timing does not alter existing health,
retirement, unknown-scope or capture-valid semantics; consumers must qualify the
separate stream, match every observed CB-ID and reject missing/Unknown intervals.

Timing callbacks share the existing finite total-event quota; no extra event store,
CB owner, allocator or completion handler is added. Metadata reservation includes
actual enlarged CommandObservation size. Optional reads/callbacks occur outside
pending registry locks; callback panic/error/reentry, quota or clock loss remains
fail-closed via existing invalid/drop counters. Even after invalidation an Unknown
terminal callback is attempted when quota permits. Exact delivery cannot be promised
if a sink fails, a quota is exhausted or the process aborts: completeness is then
invalid, never repaired by accepting missing records. Default-disabled operation
performs no timestamp reads or new sink callbacks.

Authored but unrun: five pure validator/lifecycle/quota/same-retired-resource tests;
one fresh-process ignored <=4MiB actual control (64B tensor data +bounded registry
and sink) performs two F32 adds with the production async producer Drop and checked
consumer wait. It requires exact eight intermediate/final values, two Available
same-CB records, one checked-empty Unknown and one unsubmitted Unknown, unique IDs,
zero pending/drops/scopeUnknown, and reads retained consumer getter repeatedly
without recommit. No model, timing improvement, F2 certification or release claim.
No builds/tests/Metal executions have been run by the author. This candidate is
intended to compose into the same prospective metadata release, not a new release.
