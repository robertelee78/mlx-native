# Exact Q5_K and Q6_K embedding gather

Status: implementation candidate; registry publication and downstream
exact-artifact validation remain separate gates.

## Contract

This change keeps accepted GGUF Q5_K and Q6_K embedding tensors in their
declared packed storage. A dedicated Metal gather decodes only the rows named
by the input token IDs into F32 activations. It does not create, cache, or
substitute a second weight representation.

The Rust boundary validates the exact logical byte extents, dtypes, dimensions,
token range, writable output, and non-overlapping output storage before a
pipeline is selected or work is encoded. The capability result names the
concrete Q5_K or Q6_K embedding route so admission and execution use the same
contract.

## Evidence

The focused Apple GPU gate uses the production 5,120-element embedding width
and synthetic packed blocks whose expected values are decoded independently on
the CPU. It covers every value in each selected row, repeated and out-of-order
token IDs, and both quantization formats. Every F32 output bit must match the
independent decoder. An out-of-range token must fail before command encoding.

```text
kquant_embedding: 3 passed
capability route: focused Q5_K/Q6_K admission test passed
```

These micro-gates establish exact packed-row execution, not end-to-end model
quality or throughput. Those claims belong to a downstream server run bound to
the published crate, exact model artifact, prompt, sampling settings, and
hardware.
