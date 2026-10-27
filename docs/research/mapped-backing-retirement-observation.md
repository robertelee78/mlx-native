# Original NoCopy backing retirement observation

This ignored test separates the original CPU mapping passed to Metal from a
second, independently retained observer mapping of the same owned file. It is
an experimental capability observation, disabled by default. No result has been
executed for this source. It changes no production allocator, API or policy.

Two initialized64MiB owned files have independent A0/A1 and B0/B1 readonly
aliases. Only A0/B0 back actual NoCopy resources. After checked independent
sentinel reads, A is removed from membership; its Metal wrapper and then caller
A0 owner are dropped. A1 remains live for page-disposition queries, with live B
as a control. Later predetermined phases drop device/queue and B0 owners.
Neither completed commands nor wrapper drop are treated as final retirement.

The actual NoCopy deallocator block retains the original mapping. A Weak owner
therefore distinguishes backing still retained by native/resource owners from
backing-owner disposal. Since memmap2 Drop ignores munmap's return, a separate
bounded Basic64 VM-region lookup checks whether the saved old virtual range is
absent. This metadata lookup never dereferences a retired mapping. Split live
baselines, address reuse, unsupported results and delayed retirement remain
Unknown. All returned object ports are cleaned up, including on error paths.

The installed SDK Basic64 ABI uses flavour9, packed4/36bytes/9int output words;
its count parameter is32bit, while addresses/sizes are64bit. The interpreter
uses the containing-or-next region behavior in [Apple XNU's vm_map_region](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map.c).
The inspected source does not attest to every installed kernel. Raw statuses,
counts, regions, source/binary identities and host context remain necessary.

The seven phases contain24 live-alias page-range queries (98,304 dispositions)
and7 VM-region calls. Original Metal mappings and CPU aliases share128MiB unique
file payload; alias address space is256MiB virtual and is not another payload.
The requested/userdata envelope is159,449,216B, including16MiB log and8MiB live
metadata allowances, below256MiB. Native/driver/bootstrap overhead is Unknown.
The root runner must use one fresh process with a20second process-group watchdog.
No models, parallel tests, polling, sleep or favourable retries are allowed.

Compile and pure validation are separate from actual execution:

```sh
cargo test --locked --lib mapping_retirement --no-run
cargo test --locked --lib mapping_retirement -- --test-threads=1
```

The second command should select exactly2 pure tests and1 ignored test. Run the
archived test binary, not a zero-test filter, only under the reviewed external
supervisor, with `HF2Q_ENCODER_SESSION=1` and the exact ignored filter:

```text
device::nocopy_residency_tests::scoped_membership::mapping_retirement::original_backing_mapping_retirement_observation
```

A zero exit alone does not prove exclusion. Require the exact CPU positive,
begin/resource identities, all7 complete raw phase records, final protocol
record, source/binary hashes, bounded output and owned cleanup/process reap.
The classifier independently distinguishes live-device exclusion, still-wired
pages after proved mapping retirement, exclusion only after device retirement,
and Unknown. Every classification grants zero model-memory credit. A finite
positive does not establish causality, a bounded retirement time,107GB behavior,
a production layer-window policy, physical admission, performance or model fit.
