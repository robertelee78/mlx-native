# Private maximum-parent conservation experiment — source contract

This isolated ignored test adds no public API or production policy. It preserves the earlier 64MiB experiment and its zero physical/model credit. Root authorizes only a finite falsification/controlled-host comparison. No execution, compilation, tests or hardware observations have been performed by this source author. Public 0.15.14 registry/tag/source verification is a mandatory runtime prerequisite; main 2109 contains the same read-only production API but is not by itself a verified registry receipt.

Governing design: `/tmp/adr063-max-parent-reclaim-scale-design-q5-design-v2.md`, SHA256 `851e335a7a580d9ff01933024af5346a0733ecc47eb552bb1e8d1c18b964596c`. Independent policy: `/tmp/adr063-max-parent-conservation-decision-policy-f3-v1.md`, SHA256 `5a7cd03bd035a5b19ae1f379afa69fa934c6c28bd44b855df74460c4a19bd4be`. Joint review: `/tmp/adr063-max-parent-joint-hld-independent-review-f3-v2.md`. The source fixes both mechanical ambiguities rather than retaining optional observation points.

## Fixed source census

| Item | Exact count or bound |
| --- | ---: |
| Distinct immutable files / NoCopy parents | 3 |
| Parent/output Q | 922,746,880 B |
| NoCopy parents + ordinary output + final churn | 3,724,541,952 B / 5 roots |
| Unique initialized file payload | 2,768,240,640 B |
| Read-only mappings | 6Q virtual bytes; nonadditive to unique file bytes |
| Host page size / pages per parent | 16,384 B / 56,320 |
| Chunks per alias | 14 = 13 x 4096 + 3072 pages |
| Dual-alias native calls per whole sweep | 28 |
| Precondition sweeps | 6 cuts x 15 = 90 |
| Immediate PRE-to-cut sweeps | 6 |
| Post-cut sweeps | 6 cuts x 3 = 18 |
| Total sweeps / native calls | 114 / 3192 |
| PRE / POST global gates | 6 each / 12 total |
| Parent global frames | 12 x 9 = 108 |
| Whole-Q GPU copies / full byte+SHA oracles | 15 / 13,841,203,200 B |
| Actual advice syscalls | 6 (two aliases per Advice cut) |
| NoAdvice syscall placeholders | 6 explicitly not invoked |
| Commit/sync deltas | 26 / 26 = 15 checked copy CBs + 11 checked membership CBs |
| Source protocol records | 1884, plus existing one CPU-positive marker |

Source record census: BEGIN1 + CHUNK1596 + SWEEP114 + SCHEDULE108 + GATE_READY12 + GATE_ACKED12 + CUT_BEGIN6 + ADVICE12 + POST_ORIGIN6 + READ15 + CHURN1 + COMPLETE1 =1884. Failure may end at any prefix and MUST remain protocol=false; no parent promotion of a partial/exit101 source. Successful protocol still has physical_conservation_qualified=false, driver_residual_qualified=false, physical_credit_bytes=0, model_admission=false. Empirical comparison outcome is separate.

## Exact operation order

One fresh process; fixed NoAdvice then Advice, never randomized or favorably selected. Each arm reads A,B,C,A,B,C. After first B remove A and retain B/C; after first C remove B and retain C/A; after second A remove C and retain A/B. Each read checks its own real command-buffer Result, drops the wrapper before CPU byte access, verifies every byte against the immutable cohort/page/offset formula, verifies initial full file SHA, and has a 10s fail-stop budget. No synthetic golden Q-byte allocation exists.

Every cut: 15 complete dual-alias precondition sweeps at 0,2,...28s, whole block <=30s, final complete PRESENT coverage and clean/noncopied/unwired; PRE gate and all nine parent frames; one fresh immediate sweep with the same predicates; action marker; either two actual MS_SYNC|MS_INVALIDATE calls or two explicit no-syscall records; post sweeps at 0/2.5/5s ending <=7s; POST gate/all nine frames. Advice requires full known PRESENT0/clean/unwired at all three post sweeps. Control records actual counts without requiring absence. No optional sweep or fourth cohort.

ALL six cuts complete before the sole 32MiB churn allocation/write. Then promote the same three retained parents and read A/B/C fully. No parent, FD, mmap, output, or source content is dropped/recreated between arms/cuts. Encoder/session wrappers are deliberately checked and dropped before observation. Root identities and SameFD stat/range relationships remain verified on every sweep and membership change. No own GPU work, source payload reads, new Q allocation, foreign queue, replay, protection/write, unmap/free or alternate advice inside conservation. Hidden driver PFN coherency remains an experimental hypothesis; byte parity does not prove its implementation.

## Query and IPC clocks

Each public native query <=4096 pages and <=50ms own query bracket. Each complete two-alias sweep <=2s including pair hashes/logging/range checks; exact coverage, equal flags, counts, known status, native root ID, actual addresses, FD identity and descriptor are mandatory. Per-chunk native clock brackets are raw process-local Instant observations, NOT parent/POSIX/GPU joins. Full sweep age/skew <=2s; alias/chunk observations are non-atomic.

Dedicated parent-owned stdin PIPE must remain open through completion. Exactly twelve nonce gates; each uses fresh 128-bit `/dev/urandom` nonce and a bounded ASCII JSON ACK <=8192 bytes. Each byte read is poll/deadline guarded; recursive duplicate object keys (including escaped equivalent keys), trailing JSON, EOF/HUP/timeout/nonmatching/extra/typed-invalid keys fail closed. An actual-used serde visitor rejects duplicates before Value construction; bounded ACK bytes and serde default depth remain in force. Gate wait <=5s includes READY/logging, poll/read, parsing and validated-frame acceptance; endpoint checks after newline/strict decode and immediately before acceptance prevent a final50ms poll from bypassing the deadline. Parent must collect nine physical frames at 500ms targets 0..4s, each <=250ms, whole <=4.25s; send ACK <=250ms after last frame and begin first frame <=250ms after receipt of readiness. Parent must retain raw meminfo CSV/stdout and the exact source/binary/parser/header joins; numeric ACKs are not authenticated kernel evidence by themselves.

ACK exact keys: schema=`mlx.max-parent-gate-ack.v1`, nonce, arm, cut, phase, frames=9, valid=true, parent_clock=`parent.monotonic.receipt-bounds`, last_frame_to_send_ns, ready_to_first_frame_ns, global_frames. Each frame has exactly start_ns/end_ns/physical_bytes/partition; partition order Anonymous,CachedFiles,Compressor,Free,Wired,Static,Unaccounted. These times are the parent's single monotonic clock only; source never rebrands them as its Instant epoch. Census retains the previous accepted parent-frame end across all twelve gates; every new plateau first start must be >=that end, so a new nonce cannot authorize replay of old timestamps. A refused replay leaves chronology state unchanged. Parent raw freshness/source joins are still mandatory; monotonic numbers alone are not attestation.

PRE: source receives ACK, completes the fresh full sweep <=2s, and reaches each action invocation <=2.25s after ACK receipt. Parent MUST independently enforce last PRE frame end to receipt of CUT_BEGIN <=2.5s, retain receipt brackets and reject unjoinable delivery delay. No cross-clock subtraction. Each advice syscall must start within the own 2.25s action window; a slow first call may refuse the second rather than enlarge the window. Syscalls are not preemptible; outer watchdog owns abort.

POST: last sweep source end to GATE_READY emission <=50ms, parent receipt of that readiness to first global frame <=250ms. Parent additionally retains the preceding SWEEP marker receipt and bounds its receipt-to-first-frame <=250ms; no silently queued/stale stdout. Thus source/receipt transport uncertainty stays explicit, not simultaneity. POST ACK precedes any later refault/transition; all six cuts precede churn. Failed IPC/clock/identity observation keeps full residual/HOLD.

## Actual-used empirical decision

Child validates all 108 typed frames, independent physical P sums including fully charged U, cadence/nonoverlap, stable P, and every within-plateau FULL L1 <=128MiB. Parent must independently validate the real physical partition/source/known-U status and process/pressure/swap guard; fabricated/unverified numeric arrays cannot qualify.

For all frame pairs, NoAdvice PRE/POST FULL L1 <=128MiB. For all Advice PRE/POST pairs, J=sum absolute deltas of Anonymous,Compressor,Wired,Static,U + abs(deltaFree+deltaCached) <=128MiB. No category cancellation; only declared Free/Cache signal is removed in this cross-phase rule. For each matched cut: A_low=minAdvicePOSTFree-maxAdvicePREFree; C_high=maxControlPOSTFree-minControlPREFree; reduction=clamp(A_low-max(0,C_high)-FULL128MiB,0,Q); residual=ceil_page(Q-reduction). Cached decrease corroborates only, never added. Every cut must have reduction>0 for PositiveUnderFrozenEmpiricalNuisance; otherwise NoPositiveConservation. Unknown/contradictory/noisy input fails rather than retunes. Raw parent frames and all three reductions/residuals remain retained.

This is empirical under a predeclared controlled-host nuisance assumption, not universal OS/driver/PFN attribution. Unrelated release between samples can confound it. Native requested capacity never shrinks. No scaling to 9.891GB cohorts, 774 parents, whole model or 125.613GiB rejection; no physical/admission credit even if all predicates pass.

## Exact source resources and runtime prerequisites

Five-root requested cap 3.46875GiB is installed before device. Files stream through one 1MiB scratch; SHA state is constant; no Q-byte CPU golden copy. Query pairs hold <=32KiB raw flag arrays. Six retained physical plateaus/control comparisons are bounded KiB; ACK <=8KiB, each serialized record <=16KiB; total1884 at that ceiling is below32MiB plus framing, inside64MiB combined runtime output. No source materializes all flags or trace records in a Vec. Directory records exactly three private identity-checked files; cleanup only these in owned TMPDIR subtree.

Selected 18GiB actualFree preload /8GiB runtime protected floor: 8 floor+3.46875 GPU requested+1 CPU/staging+2 driver positive+2 combined age/external growth/reaction+.25 observer=16.71875GiB, leaving1.28125GiB. Process RSS AND footprint <=12GiB, normal pressure/no observed swap growth, global gap <=1s/frame <=250ms. Driver reserve is operational, not a proved upper. 4GiB new owned IO reserve covers3Q fixture files+64MiB logs+64MiB raw samplers+<=256MiB exact APFS binary clone/control assets/metadata; prior history retained with no refund. No model/header data involved.

Root-only outer guard: 600s spawn +TERM3/KILL5,608s absolute including post/cleanup. Source setup/file sync <=30s (checked after blocking operations), each read+oracle <=10s; six precondition blocks<=180s, six post blocks<=42s, twelve gate waits<=60s, six immediate sweeps<=12s;15reads<=150s. These named phases total<=474s plus small records/transitions, within600s; no uninterruptible-call guarantee. New runner must enforce outer deadline, raw retention, phase receipt ages, full history/disk ownership, exact process identity/listeners/quiet guards and cleanup. Existing stdin-DEVNULL runner is NOT compatible. No real run until independent source/runner/operating approval and public14 registry verification.

Eight authored host controls exercise actual helpers: exact full coverage/tail/census/budget; actual schedule early/late/overrun/count refusal; closed nonce/phase/typed age/extra-key gate; presence/dirty/wired/copied absence refusal; all-frame nuisance/no-cancellation/full subtraction/partition corruption; recursive actual pipe decoder duplicates/trailing input; actual cross-gate chronology with fresh continuation/replay/overlap refusal; actual gate endpoint limit/late-newline allowance refusal. They remain UNRUN. Ignored filter: `device::nocopy_residency_tests::max_parent::actual_private_max_parent_controlled_reclaim`.

## Preserved source iteration

V1 exact three files are retained under `/tmp/adr063-max-parent-source-preserved-q5-design-v1`. V2 changes only this document and the private child: strict recursive ACK decoding, all-gate parent-frame chronology, and final endpoint deadline enforcement plus their actual-used controls. Geometry, immutable owners, API, native counters, phase/record counts, selected resource policy and all physical-credit exclusions are unchanged. Registration/Cargo/public production source remain byte-exact V1.
