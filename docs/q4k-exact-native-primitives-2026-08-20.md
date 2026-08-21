# Exact Q4_K inference primitives

Status: release candidate for `0.10.17`; registry publication and downstream
exact-artifact validation remain separate gates.

## Contract

This change keeps accepted GGUF Q4_K weights in their declared storage and
adds the family-neutral Metal operations needed by native inference:

- direct Q4_K embedding gather into F32 activations;
- raw-bit Q/gate activation deinterleave after one fused projection;
- a column-amortized Q4_K matvec for decode widths 2 through 8.

The width matvec preserves the serial accumulation tree for every output
column. Widths 6, 7, and 8 are tiled as `3+3`, `4+3`, and `4+4`; only the
register-safe kernel widths 2 through 5 are published. Rust validates shapes,
dtypes, logical byte extents, buffer mutability, and token IDs before encoding.
Metal kernels retain their own bounds checks.

The resolved-dispatch receipt identifies the concrete Q4_K width kernels and
their geometry. Capability admission and physical dispatch therefore remain a
single auditable contract rather than an environment-variable inference.

## Evidence

Focused Apple GPU gates on the release-candidate source:

```text
q4_k_embedding:                 4 passed
q_gate_deinterleave:            6 passed
q4k_mv_mn_byte_parity:          M=2..8 byte-identical to serial
ggml_policy_route_execution:    2 passed
ggml_resolved_dispatch_trace:   4 integration + 5 validator tests passed
buffer logical-length contract: 1 passed
```

The embedding parity gate uses the production 5,120-element row width. The
deinterleave gate covers decode widths 1 and 4, an odd tail, a prefill-shaped
batch, tracked dispatch geometry, and fail-before-dispatch validation. The
matvec gate compares every F32 output bit against independent serial calls for
all production widths.

No standalone throughput claim is attached to these micro-gates. End-to-end
speed admission belongs to the downstream model artifact, prompt, cache,
sampling, and server configuration that consume the published crate.
