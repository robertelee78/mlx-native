# GGML resolved-dispatch receipts (2026-08-20)

## Status

Implemented for the canonical dense GGML entrypoint and the Q4_K, Q5_K,
Q6_K, Q8_0, and IQ4_NL fused gate/up/SiLU entrypoints. This is an execution
identity primitive, not a latency measurement or an allocator.

## Contract

`*_with_trace` entrypoints receive an explicit `GgmlRoutingPolicy`, recompute
the schema-v1 `GgmlCapability`, execute the same explicit-policy production
entrypoint, and observe the dispatches actually encoded into the supplied
`CommandEncoder`. A successful `GgmlResolvedDispatchTrace` binds:

- the exact typed invocation, dimensions, GGML codec, workload, and policy;
- the recomputed structural capability;
- the encoder/device Metal registry identity;
- a backend-owned concrete route;
- the exact ordered pipeline labels, function constants, launch geometry,
  threadgroup memory, runtime-source hash or precompiled-metallib hash as
  applicable, and crate-controlled compile options.

The backend rejects contradictions between the structural route and observed
pipelines. Device-selected dense MM is resolved to SIMD, tensor-v1, or
tensor-v2 by the same production call. Tensor capability probing is cached in
the device-bound `KernelRegistry`; a registry cannot reuse a pipeline state or
probe result on another physical device. Unexpected compilation errors remain
errors instead of silently selecting the fallback.

## Boundaries

The receipt proves host-side command encoding only. It does not prove command
buffer submission or completion, output correctness, or latency. Performance
admission must separately bind the exact committed/completed workload,
artifact, OS/Metal runtime, hardware profile, warmup/run protocol, and timing
receipt. Runtime-source pipeline identity binds the crate-controlled source and
options, not the opaque driver-produced binary; cross-run evidence therefore
also requires the exact published crate checksum and OS/runtime identity.

Serialized traces are replay data, not an opaque authorization token. A caller
must re-execute/reconcile them on the exact artifact and device before granting
runtime authority. Expert, batched, permuted, embedding, and fused dual-QKVG
resolved traces are outside schema v1. Existing non-traced entrypoints retain
their compatibility behavior.
