# Opt-in pool memory snapshot

`MlxBufferPool::memory_snapshot` and the bounded growth facade's same method
perform a read-only census of current pool-owned Metal root capacities. They
allocate a temporary host HashMap only when called. Allocation, reset, dispatch,
residency and command paths have no new hooks or hot-path work.

The census unions existing Free, issued and residency-registration owner
containers by live Metal object identity. Views and repeated handles to that
object count once. Labels can overlap and must not be added to retained union.
A residency-only registered root is included when the pool actually retains it.
Distinct objects sharing mapped pages or heap backing remain separate resource
capacities. Root lengths are neither logical view lengths nor physical memory.
The snapshot is one pool's scope; summing several pools requires proof that they
have no common imported roots.

Metadata allocation, inconsistent lengths for one identity and checked sum
failures return an error. A caller must report that observation unavailable,
not zero. This call does not validate unsafe reset contracts or pool health,
wait GPU work, flush residency hints or query physical residency. An empty pool
can still have escaped raw/command/caller owners alive. Physical residency,
owners outside the pool, and lifetime highwater are explicitly unavailable.
`GrowthLimitedMlxBufferPool::snapshot` remains a separate monotonically charged
creation-history ledger; clear cannot refund that ledger.

A consumer may record the maximum of successful observations as an **observed
checkpoint highwater**, with timestamps and generation. It cannot label that
value an all-time peak. A maintained all-time counter would require mutation
hooks; this initial additive API deliberately avoids that allocator change.
No actual `allocatedSize` or device allocator meter is claimed by this API.

HF can collect on its actual TLS owner at existing successful layer/group/token
completion or request disposition and publish a small last-observation value.
An HTTP reader should not create a device, query another thread's TLS or add a
GPU wait. Keep creation history, mapped file-page union, direct/cache roots,
process footprint, global OS categories and physical uncertainty separate.
Never subtract a pool clear from OS occupied bytes or infer final release.

Source-only candidate: five pure census controls are included and have not been
executed by the author. They exercise the production union function for alias
and category deduplication, residency-only roots, explicit unavailability,
inconsistent lengths/overflow/limits and reset label changes. No Metal/model
control, build or regression result is claimed. A later tiny CPU-Metal control
can compare real root/view identities without GPU submission; publication and
HF dependency adoption require their normal independent validation.

The separately ignored `pool_memory_actual_roots_views_reset_and_clear` control
requests exactly two Shared Metal roots (8+16=24 bytes), with active residency
hints required. It checks real pool snapshots, re-registers a same-object view,
resets without any subsequent reuse, then clears while external clone/view
handles still retain the8-byte root. The empty census must still report outside
owners and physical residency unavailable. It encodes/submits no GPU commands,
reads no model, and makes no allocationSize/physical-overhead upper claim. Its
completion marker follows all assertions and explicit alias drops. The control
has not been compiled or executed by the author; root owns later validation.

Root validation on Apple Silicon: `cargo test --locked --lib pool_memory_census`
passed all five pure controls. The ignored actual-root control passed and emitted
`POOL_MEMORY_ACTUAL_COMPLETE roots=2 requested_bytes=24 gpu_submissions=0 outside_owners=Unavailable`.
Locked all-target/all-feature check and release build passed; the library suite
passed 486 tests with 18 ignored. Reference-policy and package-hygiene checks
passed. No hf2q model ran during native validation. These results validate the
current resource census, not physical residency or a downstream model budget.
