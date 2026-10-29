# Private8GiB NoCopy sparse-read/parent-pin stage observation

Source-only finite experiment, no public API/version/kernel/dependency change. Root HLD `/tmp/adr063-nocopy-sparse-pin-stages-hld-q5-design-v2.md` SHA9cc5c5c4362a0e98f169009332bc7434986d71bcd97b3b82b0bca18df68b538b supersedes onlyV1 process12→24GiB before any execution; V1 remains retained. Independent stage design reviewer accepted source scope. No test/build/device/GPU/model/observer/commit/tracker execution by author. Root owns source review/build/controls/new disabled runner and operating approval/actual.

One private file8GiB, exact readonly same-FD independent mappings A/B, one NoCopy buffer8GiB from A, one ordinaryOwned64B output. NativeRootAuthority8GiB+64B/2root attempts/2raw/1device installs BEFORE Device. Source writer uses one1MiB block, shared word-generation helper and streamed SHA, sync_all before immutable readonly FD/stat freeze. StreamSHA authenticates the initialized input stream, NOT an independent file rehash. No CPU mmap payload read at any stage. File/mmap/Metal/outowners stay live through all observations; no msync/madvise/purge/unmap/free/protectionwrite/churn while live.

Output64B exists before baseline, as a constant tiny control; its source CPU zero-fill and staged enrollment are disclosed. Four fixed interventions:

1. `preNoCopy`: actual Device+Owned64B and both CPU maps, before8GiB NoCopy. Bothaliases queried as readonly mappings; no encoder exists, own command/commit/sync/dispatch0.
2. `registered-before-ourcommit`: construct8GiB NoCopy, verify actualKnownNative/source/Shared/readonly/extent metadata; no encoder/commit before either fixed snapshot. A uses native root query, B mapping query. Own countersstill0. Published constructor stages additions, but there is no public pending-membership getter: source labels that distinction and does not infer it from allocation count. Heartbeat requestsResidency without flushing pending by source; observations do not prove arbitrarydrivercausality.
3. `after-checked-empty-flush`: exactly one sameDevice EncoderSession commit_and_wait Result checked, wrapper dropped. Actual ordinary residency commit observed via testcounterdelta1. Own command-buffer/commit/sync1/1/1, compute0. No buffer payload GPUread here.
4. `after-checked-four-16B-blits`: one encoder, fourblits from page0/floor(PAGES/3)/floor(2PAGES/3)/lastpage to ordinaryoutput0/16/32/48, one checked commit/drop. Every64B output byte checked against independently addressed word oracle. Own CB/commit/sync2/2/2 compute0. Explicit source GPUread64B is not a claim about hidden hardware traffic.

Eachstage has2fixedfull snapshots at0/3s from its ownInstant origin, startlateness<=250ms; each full dualalias<=2s includinghash/JSON/source checks; finalend<=5s. No favorable prefix/suffix or earlystop. Host16KiB only, otherwiseunsupported:8GiB=524288pages/alias,128chunks×4096. Source alternates one A record then one B record perchunk; no paired flag equality at any point. Eightcomplete dualalias snapshots=2048actualpagequeries/8388608signedrawflags. Eachnativequery<=250ms nativebracket, whole sweep refusal remains2s; no syscallpreemption guarantee. Source/root/range/address/count/status/clock/knownflags/countrecomputation are strict independently. Unknown/stale/partial/error/missing data abort; no zerosubstitution.

One raw singlealias record contains all4096 flags, fullLEflagSHA, nonadditive PRESENT/DIRTY/WIRED/COPIED/PAGEDOUT/otherKnown counts, native/outerstage clockbrackets, actual typed source+FD association. It holds only one16KiB rawflags array and bounded JSON transient. Knownrawflags<=8191 require atmost5serializedbytes/page; reserve4096B fixedmetadata/framing =>24576B record (stricterthan32KiB).2048×24576=50331648B; stage/summaries/failure allowance2MiB leaves64MiB sourcecombinedlogcap; no truncation. Source counts/marker sizes independently guarded. No CPU-payload or rawFlagfullsnapshot vectors retained.

Exact successful protocol2070 `SPARSE_PIN_` records: SETUP1 +BEGIN1 +REGISTERED1 +EMPTY_FLUSH1 +READ1 +STAGE_BEGIN4 +STAGE_END4 +PAGES2048 +SNAPSHOT8 +COMPLETE1. No hiddencalibration query/positive or allocator/control root. Three host controls authoredUNRUN exercise actual coverage/cadence/log guard, actual stream-blockproducer vs sparse64B byte oracle +alignment/anchor distinctions, actual individual source/count/range/clock/flags validity including differingvalidaliasflags and native-root provenance. Ignored filter `device::nocopy_residency_tests::sparse_pin::actual_nocopy_sparse_read_parent_pin_stages` is onefreshprocess only.

Prospectiveoperatingpolicy:32GiBactualFreepre/8GiBruntime,24GiBRSS ANDfootprint, normalpressure/noobservedswapgrowth;90sspawn+TERM3/KILL5=98sabsolute;10GiBfreshownedIO. Physicalpositivecells8protected+8nativeRoot+8unqualifiedfilecache+1CPU+2driver+.25observer+2external/stop=29.25GiB+64B<32. Twoaliases can report16GiBRSS despite sharedfilepages, hence process-only24cap, NOT physicalpartition/refund. Driver and externalcells are operatingpositives, not hardwareupperproof. Source setup8GiBstream/SHA/sync/maps/device<=60s checkedafterblockingwork; fourstagewindows<=20s, remaining10s outerbudget covers ownCB/metadata/cleanup. Outer watchdog owns blocking-call abort, no silentextension. Allsourcepayload/outcapacity reservedbeforeDevice, no additionalnative roots.

Cleanup checks originaldir/filedevino/symlink/type before deletion, after successful ownCBchecks and dropping views/resources; failedprotocolDrop may also remove only that ownedfixture and printscleanup outcome. Root runner must retainrawprefix/reapownedPG/verifyempty/ownedTMPDIR andcleanup independently; sourceDrop is not a guarantee on SIGKILL. No pinning outcome/equality required for protocol success, no physicalcredit/modeladmission or extrapolation. Allprevious64MiB/maxparent/reclaim results stay unchanged.

## Executed finite observation (2026-10-29)

Executed source `7a65687dbd7ed685023cf7d63c1246b191093ebb`, release
lib-test binary SHA256 `326b6f829ce67b1b135ec6944771dfc7bb2576e56dabbf97a4a0c00adf46f166`.
Release compilation passed; three focused host controls passed and the actual
control was initially ignored. The corrected external supervisor passed eighteen
metadata controls and independent one-attempt operating review before execution.

One actual invocation passed: 2,070 protocol records, 2,048 page queries,
8,388,608 raw page flags, eight complete snapshots and the checked 64-byte GPU
output oracle. Runtime was 27.840854 seconds; terminal checks completed at
30.075943 seconds. All 206 process samples passed. The child exited zero and was
reaped with its process group empty; the entire 8 GiB fixture was removed.
The retained evidence bundle is `native-nocopy-sparse-pin-stages-actual-v1`;
terminal receipt SHA256
`514fa9184251185bb064f7336bafdcae8cde12aae0f5b9be3b675b337975a327`.

| Stage | PRESENT pages per alias, both snapshots | WIRED pages per alias, both snapshots |
| --- | ---: | ---: |
| Before NoCopy | 524,288 | 0 |
| Registered before our commit | 524,288 | 0 |
| After checked empty residency flush | 524,288 | 524,288 |
| After four checked 16-byte blits | 524,288 | 524,288 |

These are nonadditive raw-disposition counts for two aliases of the same file;
they do not represent twice the physical pages. The complete tested file became
reported WIRED after the explicit empty flush, before any source GPU payload
read. Sparse payload reads therefore cannot be assumed to preserve sparse
pinning for this finite same-device, same-root residency-set path. This result
does not establish a universal driver cause or a scaling theorem for a real
model, and it gives zero physical memory credit or runtime admission authority.
The sampled child peak RSS was 16,728,064 bytes and peak footprint was
125,436,552 bytes despite the full 8 GiB reported WIRED. These process metrics
are not a physical-memory partition or bound for this native-resource path.

The source invocation is the ignored test filter above with
`HF2Q_ENCODER_SESSION=1`, a private fresh TMPDIR, exactly one test thread and
`--exact --ignored --nocapture`. The external operating guard is part of the
experiment: its preflight, process identities, deadlines, complete raw output,
strict parser and cleanup receipt must be retained; an unguarded libtest run is
not equivalent evidence. No production path, release API or model inference was
changed, and this research branch is not a performance optimization candidate.
