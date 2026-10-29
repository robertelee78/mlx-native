# Read-only root backing descriptor candidate

`MlxBuffer::backing_snapshot()` complements `root_origin_snapshot()` without
installing authority or allocating, reading payload, submitting commands, changing
residency, or escaping a raw Metal handle. Existing authority/import/heap refusals
are unchanged. The origin remains Legacy/UnknownImport/KnownNative; only the last
supplies a stable authority root ID and requested bytes. No replacement ID is minted.

The snapshot reports actual Metal length, allocatedSize, storageMode, and the
logical view's offset/length. allocatedSize is a native metric, not unique physical
RAM. KnownNative non-file constructor roots have Owned provenance; uncertified
legacy/import roots remain Unknown. NoCopy is identified only by the actual retained
Arc<Mmap> in buffer storage, never a guess from pointer alignment or file names.

NoCopy reports the mmap slice base/length and actual registered Metal CPU base/length.
The latter can include the existing page-rounded final-page tail; logical bytes must
still lie within the mmap slice. All view and pointer endpoint calculations are
checked. No file descriptor, inode, file offset, or content identity is invented.
HF, which owns the verified GGUF FD/mapping, must separately join this range to its
fstat and mapping offset. Alias/view roots share the existing origin and backing.
Independent allocations with equal payload are distinct roots. File-backed resources
cannot enter the pool release/free-list path. Existing captured Metal deallocator
and shared storage retain the mmap independently of snapshot values.

Snapshots retain no owner and addresses are process-local, reusable after release.
Consumers must keep the source buffer/file owner alive while joining ranges. These
are not physical PFNs, live residency, completed-command, final-release, eviction,
or admission certificates. No previous failed retirement observation is rescored.
CPU roles, global OS partition and prospective operating admission remain separate.

Authored controls (not executed): one pure actual-used bounds/overflow test and one
ignored fresh-process <=64KiB CPU-shared/NoCopy metadata control covering clones,
subviews, independent equal-content roots, mmap-owner retention and unknown imports.
The latter creates a device but dispatches no GPU work. Root must supervise and
validate it before publication. No version bump or publication is included; this
additive metadata API may compose into a later coherent release candidate.
