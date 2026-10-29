# Owned NoCopy latency: narrow positive sampled result

Source `9745fb3485a8dcd6dfcb52892ef32ae44beba238`, based on public0.15.13
commit `512c023b271666bd1df0e5e4f4b93c47ead7b30f`. Optimized native unit binary:
`823adda2f36e76251a0474e5ffd9660676e5487126af26a5c6c5541812c692b7`.
The actual host classifier passed; the corrected external supervisor/protocol
passed 22 controls and independent one-attempt review before execution.

The experiment retained both 64MiB files, aliases, Metal buffers and device
through all three full 101-sample horizons, with checked independent A/B/A GPU
readbacks. First dual-alias zero-WIRED indices were [0,21,21], zero suffix lengths
[101,80,80], and retained rebound counts [0,0,0]. Delayed first zero was about
1.06 seconds. All three full-WIRED positive controls passed. This is classification
`ExclusionAtRetainedSamplesWithinFrozenHorizonOnly`.

All 303 queries and owner/alias checks completed. Maximum query duration was
9.374ms and maximum start gap59.807ms. The child exited0 after15.527seconds;
complete supervisor validation ended15.908seconds afterspawn, below38seconds.
Its process group was reaped, fixture directory empty, global pressure normal
and swap growth zero. Retained evidence: local campaign `adr063-preflight-20261023`,
`native-owned-nocopy-latency-actual-v1`. Independent review:
`/tmp/adr063-owned-nocopy-latency-independent-actual-review-f3-v1.md` and JSON.

The previous short65ec later-switch trial remains negative within its original
observation window. It has not been rescored. This separately frozen five-second
horizon tests delayed exclusion. Query arrays were summarized and hashed rather
than archived; independent reconstruction of every raw disposition is unavailable.

This does NOT establish continuous settling, a universal1.06s deadline, physical
page ownership/free capacity, model fit, or a performance improvement. Model credit
remains zero. A future family window policy must charge active, prefetched and
retiring resources; validate actual disposition before enlarging its cohort; bound
retiring multiplicity; and stall or stop before a projected capacity breach.
Neither a bool residency API result nor sleeping for1.06s is a valid refund.
No production API, paging policy, release or hf2q dependency change is made here.
