# Metal command-buffer and label lifetime

Status: accepted for the `0.10.7` release candidate; downstream exact-artifact
hardware validation remains required.

## Failure

hf2q's long-lived Qwen inference thread stalled in
`-[AGX commandQueue commandBuffer]` after exactly 63 successful bounded
multi-token forwards. The next allocation never returned to Rust. A preserved
heap contained 37,999 live `AGXG17XFamilyCommandBuffer` objects and 37,998
matching implementation objects. Apple's Metal headers document a default
command-queue bound of up to 64 non-completed command buffers.

The worker had no Objective-C autorelease pool. Metal's command-buffer and
compute-encoder factories return autoreleased objects; retaining them for Rust
ownership did not discharge the original +0 autorelease claim.
One hf2q session path additionally rotated to a new empty command buffer after
its final committed stage, then dropped it without submission. These lifetimes
made the queue semaphore cliff deterministic.

## Decision

`mlx-native` owns the family-neutral Objective-C boundary. The following
factory/retain seams execute inside a narrow `objc::rc::autoreleasepool`:

1. initial command-buffer creation in `CommandEncoder::new_with_residency`;
2. replacement command-buffer creation in `reset_command_buffer`;
3. compute-command-encoder creation plus the explicit retain that supports
   persistent borrowing;
4. command-buffer and compute-encoder label setters, whose metal-rs bridge
   creates temporary autoreleased NSStrings.

The Rust-owned or explicit retain occurs inside the pool and survives its
drain. Temporary Objective-C factory claims do not escape. Metal's label
properties copy their values during the setter, so draining only the input
temporary is insufficient: the copied label lives until its Metal owner
releases or replaces it. `PooledCommandBuffer` therefore scopes destruction
after a label is attached or the raw command buffer is exposed to a caller.
Internal unlabeled command buffers have no copied NSString owner and use direct
release after their factory autorelease has already been drained. This avoids
an empty autorelease-pool boundary on the high-rate unlabeled graph path while
preserving the conservative boundary for public escape hatches. In both cases,
`end_active_encoder` clears the compute-encoder label with `setLabel:nil`
immediately after `endEncoding` captures the trace row and before releasing the
explicit retain. This preserves command-buffer and encoder attribution while
preventing completed encoder objects from retaining one CFString each on a
pool-less worker.

Fenced stage commits apply both labels before ending the compute encoder, then
encode their shared-event signal at command-buffer scope. The per-command-buffer
dispatch-counter descriptor no longer receives a redundant Objective-C label:
its durable attribution already comes from Rust-owned dispatch metadata and the
command-buffer label, while a copied descriptor label would create another
high-rate lifetime seam on future counter-capable hardware. The six-mode heap
population proof exercises the serving commit paths, not that currently
unavailable Apple-Silicon dispatch-counter mode.

A failed intermediate spike incorrectly read the nullable label of a fresh
command buffer through metal-rs' non-null `&str` surface; the accepted identity
test compares Metal object pointers instead. Model-family chunk policy and the
decision whether a final stage needs another command buffer remain in hf2q.

## Regression and hardware evidence

`tests/command_buffer_autorelease.rs` runs isolated child processes because a
regression can block inside Objective-C before returning an error. The queue
cliff arm still performs 50,000 command-buffer drops, 50,000 labeled async
commits, and 50,000 `EncoderSession` commit/reset/drop cycles before a sentinel.
The population arm uses `/usr/bin/heap` checkpoints after warmup, 10,000, and
20,000 static-label commits. It covers synchronous, asynchronous-drop,
asynchronous-wait, unlabeled, command-buffer-only, and compute-encoder-only
paths. Every drained checkpoint requires zero live Metal command buffers,
CFString growth no greater than 256 per wave or 512 total, and autorelease-pool
content growth no greater than eight pages. A deliberately unpooled 2,048-label
control must grow by at least 1,024 CFStrings, proving that the detector is live.
Every child and heap attachment has a hard timeout.

The source-bound A/B was exact. Published `0.10.5` retained two CFStrings per
labeled active command buffer. Pooling command-buffer destruction reduced that
to one; the command-buffer-only arm became flat while the compute-encoder-only
arm retained exactly one per commit. Clearing the encoder label after
`endEncoding` made all six production/control arms flat across both 10,000-CB
waves while the unpooled negative control continued to fail.

The initial `0.10.6` implementation conservatively pooled every Rust-owned
command-buffer destruction. A same-host downstream A/B on the 100.05 GiB
DeepSeek-V4 artifact showed that this penalized its unlabeled GraphSession path:
the exact four-agent 6,685-token gate exceeded the unchanged 55-second client
bound, while the label-sensitive `0.10.7` candidate completed all four cold
requests in 50--53 seconds and then reused exactly 6,677 tokens for cached,
automatic-tool, SSE-tool, and tool-result turns. The same candidate's six-mode
heap test reported zero live command buffers and zero CFString or pool-page
growth at all three checkpoints, including `unlabeled-async` and both public
label escape controls.

A fresh Metal System Trace of the final candidate shape then recorded all
three semantic paths with identical `cmdbuffer-label` and `encoder-label`
values: synchronous `phase.iter16_smoke_token`, asynchronous
`phase.iter16_smoke_async`, and fenced
`phase.iter89e2b_stage1.fence`. This proves that clearing the compute-encoder
property after `endEncoding` does not erase ordinary or fenced trace rows. The
sanitized normal and fenced encoder-table SHA-256 values were respectively
`8c15180012913ca9fa9e39b85d075b6f5735c17612e3593501ec2bd701daeef1` and
`04a17021b223be72e020bc3a51faaadea0a267515985c92d0c2ab90710909b22`.
The trace must be repeated from the immutable release commit before
publication; raw `.trace` bundles remain local because they include machine
metadata.

The parent kills and fails a child after a 120-second hosted-runner deadline;
the pre-fix queue cliff occurs near the first 64 stranded command buffers, so
this extra CI margin does not weaken regression detection.
The release suite also exposed that kernel-profiler unit and integration tests
reset process-global tables concurrently; those tests now share test-only
locks so a parallel `cargo test --all-targets --all-features` run cannot erase
a sibling's records and report a false profiler-test failure.

In the original causal hardware spike, the same 87,972-token Qwen SSE request completed
and a subsequent four-agent tool/SSE/cache gate passed in the same process.
Heap enumeration after the long request contained zero live AGX command-buffer
objects, versus 37,999 before. The subsequent agent gate exposed a separate
deterministic CFString rise from 6,658 to 54,243 while labels remained outside
a pool. Published `0.10.5` reproduced essentially the same slope
(6,850 → 54,336 → 101,820), proving that setter-local pools alone did not close
it. The `0.10.7` candidate's model-free heap gate closes both label-owner slopes;
registry publication and downstream exact-pin Qwen/Gemma/DeepSeek hardware
gates remain separate authority.

Raising the queue cap, reducing serving slots, periodic restart, or disabling
labels is not an accepted fix. A generic external supervisor is still required
for unrelated native hangs, because a worker cannot time out an Objective-C
call that never returns.
