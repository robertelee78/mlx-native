# Private pair diagnostic: real control-phase noise, no advice

Source d485049c553289dd138d149268f8261efb719f3c, release binary
914389585023cd4f815f65927bbb7103311c89429160d94b9c28046a5b60a8eb,
ran once with failure-only pair diagnostics and unchanged acceptance rules.
Its11 native and31 parent controls passed before the independently reviewed run.

The first NoAdvice PRE plateau passed (maximum occupied-union L1 3,440,640B).
The first POST plateau failed at499,089,408B (476MiB), exceeding the unchanged
128MiB limit. Across its maximum pair, Anonymous decreased248,348,672B, Wired
decreased1,196,032B, CachedFiles increased622,592B and Free increased248,922,112B;
other categories were unchanged. This is real net movement from the fully
charged union to Free, not the internal reclassification seen in V1. Its cause
is not attributed by these aggregate observations.

Nineteen dual-alias sweeps/266 pairs and two full Q-byte readbacks passed. No
PAIR_FAILED record appeared, so this run does not explain the earlier combined
pair refusal. One PRE ACK completed; no POST ACK, advice syscall, churn,
conservation comparison or residual completed. The two ADVICE records were
NoAdvice markers with invoked=false. The native predicate and all physical
credit remain unchanged.

The terminal receipt remains failed/incomplete at52.261449seconds. The child
was terminated, reaped and its process group verified empty. Independent audit
reconstructed all18 raw partition frames and retained source/operating evidence.
Root subsequently removed the three generated Q-byte files and directory only
with exact emitted device/inode/length checks after reaping; the separate cleanup
receipt does not refund historical I/O or change the failed experiment.

This small-signal global-conservation probe sequence is stopped. Do not tune
its128MiB gate or rerun a favorable window. The next priority is exact full-
resident source accounting with bounded GGUF/header/map-plan introspection.
Any later integrity-only or larger combined-cohort experiment requires its own
prospective scope/source/guard and cannot inherit a residual from these failures.
No maximum-parent invalidation, production paging, speed or Q5 result is claimed.
