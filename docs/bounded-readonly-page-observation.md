# Bounded readonly page metadata — additive candidate

Source candidate for future0.15.14; not yet published or consumed by HF. The API does not mutate residency, submit/wait GPU work, touch payload, create raw Metal escape, certify command health or grant physical/free/reclaimability/admission credit. Existing default paths and per-tensor GGUF factory are unchanged.

MlxBuffer::observe_nocopy_root_pages(PageQueryRange,PageQueryLimits) borrows a certified KnownNative readonlyNoCopy resource. It queries actual registered ROOT pages even when invoked on a logical subview. Public observe_readonly_mapping_pages borrows a readonlymemmap2::Mmap alias. Native reports mapping extent/address, never invents file identity; caller independently retains verifiedSameFD/stat/fileoffset/mapping associations. No arbitrary integeraddress/task is accepted.

Range is page_offset/page_count1..4096; actual Darwinpagesize must be4096..65536/power2. Every covering-page round/add/multiply and address/span checked. At most4096signedi32 dispositions=16KiB oneboundedVec via try_reserve_exact, oneMachcall, <=4096scan. OutputVec capacity may have allocatorrounding; namedlength/work cap is not physicalRAM guarantee. No rawfilepayload is accessed. Logical final partialpage is observed only as OS covering metadata.

SDK ABI mach_vm_page_range_query count is64bit mach_vm_size_t, dispositions32bit signed. Returnstatus/count/flagschecked before observation eligible. Unsupported platform, noncertified/nonfile/Owned/importedroot, badrange/cap/overflow/alloc/query/partial/unknownflags/stale all typederrors; no zero-default. Callermax_elapsed_ns positive<=250ms applies the actual Mach query bracket, not a preemptive kernel timeout or whole-modelsweepbound. Allocation/descriptor/getpagesize work and boundedresultscan remain separately bounded work; externalwatchdog is required for blocked kernelcall. Clock is process-localRustInstantorigin, no POSIX/GPUjoin. Separatequeries/aliases are not atomic.

PageObservation returns actualsource/range/pagebytes/address/extent/status/count/dispositions and nonadditivePRESENT/DIRTY/WIRED/COPIED/PAGED_OUT/otherknowncounts, with localstart/end/elapsed. SDKrecognizedmask0x1fff is not proof of the currentlyrunningXNU implementation or physicalmeaning; rawflags stay raw. A WIRED0 observation is not Free, reclaimability, continuous exclusion, uniquephysicalbacking or futurelowerpeak. Metadataretainsno sourceowner; consumer must keep actualroot/mapping/FD alive for later joins.

Three pure actual-used helpercontrols cover range/arithmetic/caps, rawstatus/count/unknownflags/time refusal and exactflagpreservation. One ignored actualfreshprocess <=1MiBnameduserdata control exercises realNoCopyroot/subview/sameFDindependentalias, rawstatus/count/timing, Ownedrefusal and root-lifetime separation from returnedmetadata. It performs no dispatch/commit or modelwork, but realMetalallocations require root supervision. All authoredUNRUN; no authorbuild/test/hardware. Release/publish/exactHFpin remains root-owned and pending.

The source-only HF window design remains physicaladmissionHOLD. FullinactivePRESENT retains fullcharge; nativeownedcapacity staysfull~107GB. Tiny5slatencyresult never licenses a wholemodelcredit. This API is a missing measurement capability, not a solution by itself.

## Root validation before publication

Exact API source `b0ebc92870aa627d1f22a55191812e98ae52804e` compiled in the
optimized test profile and passed three pure range/flag/status controls. The
locked all-target/all-feature check passed. Eighteen supervisor/protocol controls
passed before the isolated actual two-page borrowed-alias/NoCopy-root fixture.
It completed with raw words [129,129] in each observation (PRESENT|EXTERNAL),
zero stderr, a reaped empty process group, and independent review. This observed
fixture used no GPU commands. The source asserted subview-root selection, Owned
refusal and CPU metadata-lifetime separation after deferred residency/autorelease
owners drained. None is a physical-release or file-certificate assertion.

Optimized test binary SHA256:
`3bb31b68a8f8355ee85af8f2b663eeaf84f7c058589282a9ee63f8b797c5097d`.
Local evidence names: `native-bounded-page-release-root-build-v2`,
`native-bounded-page-observation-actual-v1`, and
`adr063-bounded-page-independent-actual-review-f3-v1`. The initial test build
refused a mistaken private accessor name; the corrected source above is the
validation authority. It was an unpublished API extension while the package
version remained 0.15.13. Version 0.15.14 requires separate exact-source release
checks and publication; HF consumers must pin that public package and checksum.
