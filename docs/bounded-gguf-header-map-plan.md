# Bounded same-FD GGUF header and exact assumed-geometry map plan

Additive source candidate, no version bump, dependency, mapping, Device, GPU, kernel, paging or admission policy. Existing `open`/`from_file` retain legacy count/string/array limits and buffered-reader behavior through the shared parser. No real model/header/payload, build, test or hardware execution by source author.

`GgufFile::from_file_with_limits(File,GgufHeaderLimits)` consumes the actual caller FD, seeks to zero, and parses using exact/unbuffered File reads. Only after parsing does it create the lazy payload BufReader. It does not read alignment padding or prefetch the first tensor bytes. It does not certify FD read-only mode, stable stat, full content hash, or an association to a separately declared path: those remain caller source-owner obligations. Payload methods remain unchanged and are not constrained by the header budget after parsing.

Limits are explicit: aggregate header read bytes, cumulative declared allocation reservation bytes, metadata entries, tensors, individual string bytes, aggregate array elements across all nested arrays, and array nesting depth (0..64). Existing legacy caps remain additional ceilings. Before allocation, lengths/counts convert with checked arithmetic, strings check announced wire length, arrays check minimum announced wire length and aggregate elements/depth, container capacities are charged, and the second retained tensor-name clone is charged. Nested arrays charge their Vec<MetadataValue> capacity and every child allocation. Overwritten metadata/temporary storage receives no refund. Invalid UTF8/type/rank/duplicate names and bounded budget errors fail without including an arbitrarily large tensor name in the error.

The declared reservation ledger includes string/byte vector requested capacity, shapes, arrays, map entry storage, and an explicit conservative table model: two slots per requested entry, power-of-two/minimum4, entry+control bytes and32-byte overhead. Bounded containers use try_reserve/try_reserve_exact after the charge. This is requested/reservation accounting, NOT a universal allocator or physical-memory hard upper; allocator rounding/metadata and OS file cache require separate positive cells. `bounded_header_usage()` returns successful actual read bytes, cumulative charged reservations, aggregate elements/depth; legacy parsing returns None. A caller needing64MiB sets BOTH aggregate limits64MiB and supplies independent count/depth caps. Read64MiB and declared-reservation64MiB are distinct quantities, not a combined physical claim.

The parser body is extracted to focused `gguf/header.rs` and shared by both wrappers. Existing scalar decoding and GGML shape/byte logic remain the same; bounded parsing adds checked host conversions/alignment and tighter user limits. Legacy behavior is not silently made bounded. Parent gguf module is not mass formatted.

`GgufFile::mapped_tensor_storage_plan_with_limits(page_size,max_buffer_len)` is pure metadata. It calls the SAME existing `plan_mapped_tensor_ranges` via a shared private `mapped_ranges` helper also used by the existing real-Device summary wrapper and actual map_tensor_data. The coalescing algorithm itself is byte-exact. Explicit page/maxbuffer values are assumed geometry, NEVER an observed host capability. The old DeviceRef wrapper still calls host_page_size and actual device.max_buffer_length; old summaries/mapping dispatch are unchanged.

The new immutable `GgufMappedTensorPlan` borrows names from the parsed header (no name or shape clone) and exposes:

- assumed page/maxbuffer, old storage summary, ordered source segments and normalized name-ordered tensor views;
- segment logical file_start/end; page-aligned mapping/registered start; logical mmap end/length; registered root end/length; prefix and tail padding;
- tensor logical source interval/byte length, the same FIRST containing segment used by production load_tensor, logical-view offset, and actual registered-root byte offset;
- union of registered file-page intervals in this ONE source, distinct from summed root resource requests when source pages overlap.

Mapping logical length ends at file_end; registered Metal length includes final page tail. Tail is not extra readable file contents and may extend beyond EOF's logical end; no new file-bound rejection is introduced. Returned source intervals derive from the parsed directory, not a new fstat/physical verification. Parent must join actual retained same-FD extent/identity and a separately observed device capability. File-page union is not unique physical RAM/residency or a refund. Gaps within coalesced source spans remain visible via segments and views. Plan metadata is bounded by the parser's existing100k tensor limit; vectors contain descriptors/pointers, not payload. Its additional declared storage must be charged separately by a caller composing a larger CPU ledger.

Eight meaningful pure controls authored UNRUN: announced huge strings/arrays/catalog and reservation overflow refuse before allocation; aggregate strings/nested array totals/depth are independent; shared legacy/bounded output and name-clone charge parity; actual bounded File cursor proves no payload prefetch; pure plan normalization against old production ranges; prefix/tail/root/view and explicit invalid/overflow controls; bounded header+plan without payload/Device; overlapping split-resource source-page union. Existing real mapping integration test remains the later actual Device-wrapper/resource parity gate; no hardware result is claimed here.

This API supplies missing no-Device diagnostic authority, not full-resident Q5 acceptance. No physical admission, reclaim mechanism, model-load permit, parser pass on an actual selected artifact, or existing failed experiment reinterpretation follows from source completion. Root owns independent review, build/host/actual validation, version/release/public registry verification and subsequent HF exact pin.

## Candidate 0.15.15 validation

Source `3a52dd069b725bf74e00c096290a9ba4a4439747` passed locked
all-target/all-feature checking and the locked release build on Apple Silicon.
Eight focused parser/planner controls and four existing mapped-tensor integration
controls passed. The separate fresh-process mapping control also passed: exact
same-FD header reads stopped at byte135, its declared reservation ledger was862B,
and the actual131268B fixture produced one147456B registered root,192B prefix,
16188B registered tail, three tensor views and one nested view. Eight F32 values
(32 unique CPU payload bytes) matched source bits. Dispatch, commit, sync and
command-buffer counters remained zero; owned process-group and fixture cleanup
were verified. These are mapping/inspection checks, not a model, physical-memory
admission or performance result.

The complete native regression suite and hosted CI are publication gates.
The hf2q consumer must resolve and verify the published0.15.15 registry archive
before adopting these methods; an unpublished local patch is not a delivered
consumer. Existing buffered parsing and the production coalescer remain the
compatibility baseline.
