# Exact Q4_0 embedding gather

Status: implementation candidate; registry publication and downstream
exact-artifact validation remain separate gates.

## Decision

Q4_0 embedding tables execute directly from their GGUF block storage. A
dedicated Metal gather decodes only the selected rows into F32 activations; it
does not expand, cache, or substitute a second resident weight representation.

The motivating production artifact is a supported BERT-family Q4_0 model. The
initial source audit found that Q4_0 matmul execution already existed but the
embedding capability deliberately failed closed because no row-gather kernel
existed. Keeping that rejection would make native-storage model loading
incomplete, so the missing family-neutral primitive is added here rather than
special-casing a model loader.

The Rust boundary validates dimensions, Q4_0 block alignment, exact logical
byte extents, dtypes, writable output, non-overlapping logical ranges, shader
index bounds, and every token ID before selecting a pipeline or encoding work.
The kernel repeats token bounds checks as defense in depth. Unsupported
embedding codecs remain rejected by the capability contract.

## Evidence

The focused Apple GPU gate uses a production BERT hidden width of 1,024 and
compares every F32 output bit with the independent CPU GGUF decoder. Packed
fixtures vary positive and negative half-precision scales and independently
populate the low and high nibbles, including columns 15/16, 31/32, and the
final column. Repeated and out-of-order token IDs are included.

The same gate proves:

- one labeled Metal dispatch for one gather;
- mapped, nonzero-offset GGUF storage and copied storage are bit-identical;
- invalid token IDs, block-misaligned dimensions, wrong dtypes, wrong logical
  extents, arithmetic/index overflows, zero dimensions, and output aliases all
  fail before command encoding;
- the capability route admits Q4_0 while an unsupported Q5_1 embedding request
  continues to fail closed.

```text
q4_0_embedding: 4 passed
capability route: focused Q4_0 admission and Q5_1 rejection passed
```

These gates establish exact packed-row execution. End-to-end storage identity,
model reload, output coherence, and performance remain downstream integration
claims bound to the published crate and exact model artifact.
