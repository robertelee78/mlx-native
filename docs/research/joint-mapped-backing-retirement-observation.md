# Joint mapped-backing retirement with a live device and output

This ignored, model-free control follows the separate single-parent mapping
retirement observation. It preserves that case and its negative result. It
retires both NoCopy Metal parents and their original CPU mappings before the
same device, queue and ordinary Shared64-byte output are dropped. Independent
readonly aliases observe the owned files; no dead mapping is page-queried.

P0–P4 retain the original lifetime ordering. P5 removes/drains B, drops its
Metal owner and original mapping, then queries only the two surviving aliases.
Both backing Weak values and whole-old-range Basic64 absence must join with
zero observer WIRED flags. Actual device/queue identity and the same output's
identity, non-file-backed Shared storage,64 logical bytes and checked sentinel
contents are checked while those owners remain alive. P6 observes only after
that output/device retirement and cannot promote a P5 negative or Unknown.

The test performs23 live-alias queries and8 metadata-only region calls in seven
phases. Two64MiB files/NoCopy roots share128MiB unique payload; four aliases add
256MiB virtual address space, not another physical payload. Requested user data
is159449216 bytes within256MiB; driver overhead is Unknown. Each case requires
a separately supervised fresh process with20-second wall timeout. No sleeps,
polls, retries, forced deallocation or post-retirement GPU dispatch occur.

Root-controlled commands after independent source review and exact binary
binding:

```sh
cargo test --locked --lib joint_mapping_retirement -- --list
cargo test --locked --lib device::nocopy_residency_tests::scoped_membership::joint_mapping_retirement::state::joint_mapping_retirement_primary_requires_exact_live_device_join -- --exact
cargo test --locked --lib device::nocopy_residency_tests::scoped_membership::joint_mapping_retirement::region::mapping_region_interpreter_never_calls_split_or_bad_output_absent -- --exact
HF2Q_ENCODER_SESSION=1 TEST_BINARY --exact device::nocopy_residency_tests::scoped_membership::joint_mapping_retirement::joint_original_backing_retirement_with_live_device_output --ignored --nocapture
```

The final command needs the external process-group supervisor; an unsupervised
command is not execution approval. Require one actual test, all reserved raw
records, strict classification and owned fixture/process cleanup. Protocol PASS
can report a negative or Unknown. A positive is a finite retained-phase
observation, not causality, continuous peak exclusion, an active-weight window,
model-scale demotion, physical admission, performance or Q5 closure. All result
classes grant zero model-memory credit. No actual result is recorded here.
