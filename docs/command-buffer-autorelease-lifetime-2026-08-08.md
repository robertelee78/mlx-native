# Metal command-buffer autorelease lifetime

## Failure

hf2q's long-lived Qwen inference thread stalled in
`-[AGX commandQueue commandBuffer]` after exactly 63 successful bounded
multi-token forwards. The next allocation never returned to Rust. A preserved
heap contained 37,999 live `AGXG17XFamilyCommandBuffer` objects and 37,998
matching implementation objects. Apple's Metal headers document a default
command-queue bound of up to 64 non-completed command buffers.

The worker had no Objective-C autorelease pool. Metal's command-buffer,
compute-encoder, and label factories return autoreleased objects; retaining
them for Rust ownership did not discharge the original +0 autorelease claim.
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
properties copy their values during the setter, so the temporary NSStrings can
be drained immediately. A failed intermediate spike incorrectly read the
nullable label of a fresh command buffer through metal-rs' non-null `&str`
surface; the accepted identity test compares Metal object pointers instead.
Model-family chunk policy and the decision whether a final stage needs another
command buffer remain in hf2q.

## Regression and hardware evidence

`tests/command_buffer_autorelease.rs` runs in an isolated child process because
a regression can block inside Objective-C before returning an error. On one
raw pool-less Rust thread it performs 50,000 command-buffer drops after opening
their concurrent compute encoders through a compiled no-op pipeline, 50,000
labeled async commits with an active compute encoder and periodic queue drains,
then 50,000 `EncoderSession` commit/reset/drop cycles, and finally commits a
sentinel. The parent kills and fails the child after a 120-second hosted-runner
deadline; the pre-fix queue cliff occurs near the first 64 stranded command
buffers, so this extra CI margin does not weaken regression detection.
The release suite also exposed that kernel-profiler unit and integration tests
reset process-global tables concurrently; those tests now share test-only
locks so a parallel `cargo test --all-targets --all-features` run cannot erase
a sibling's records and report a false profiler-test failure.

In the causal hardware spike, the same 87,972-token Qwen SSE request completed
and a subsequent four-agent tool/SSE/cache gate passed in the same process.
Heap enumeration after the long request contained zero live AGX command-buffer
objects, versus 37,999 before. The subsequent agent gate exposed a separate
deterministic CFString rise from 6,658 to 54,243 while labels remained outside
a pool; the accepted label scope closes that measured leak. The registry
release and downstream exact-pin gates remain separate authority.

Raising the queue cap, reducing serving slots, periodic restart, or disabling
labels is not an accepted fix. A generic external supervisor is still required
for unrelated native hangs, because a worker cannot time out an Objective-C
call that never returns.
