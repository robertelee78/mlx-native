# Native dense GGUF scalar storage — 2026-08-22

Status: accepted for mlx-native 0.12.0; downstream exact-artifact
qualification remains a separate gate.

## Decision

GGUF type 30 is represented as `GgmlType::BF16`, sized as one value in two
bytes, and loaded into a `DType::BF16` Metal buffer without changing its bytes.
F32 and F16 follow the same typed-storage rule. A model graph must dispatch a
kernel that consumes the declared dtype or reject the artifact before model
allocation. It must not expand, dequantize, or re-encode a weight merely to fit
another runtime path.

`GgufFile::load_tensor_f32` is intentionally still available for diagnostics
and small graph constants whose consumer explicitly requires F32. Calling it
is a materialization decision; it is not the default weight-loading behavior.

Dense token embeddings use `embedding_gather_dense`. The kernel reads only the
requested rows from F32, F16, or BF16 storage and emits F32 activations. Host
validation pins logical byte extents, dtypes, dimensions, and token bounds
before command encoding.

The machine-readable capability contract also admits direct dense scalar
projections at M=1, continuous widths through eight, and larger prompt or
physical-batch widths. Each receipt names the scalar dtype and decode-versus-
matrix route and accounts for the exact native byte extent.

## Evidence

- GGML type ID 30 maps to BF16 with exact block and byte sizing.
- Raw tensor dtype selection is BF16; the diagnostic BF16-to-F32 conversion
  reproduces the represented values.
- Apple GPU tests gather repeated and reordered rows from F32, F16, and BF16
  tables with exact output and reject an out-of-range token before encoding.
- `cargo test --locked`, the focused dense-embedding and GGUF-BF16 tests, and
  `cargo check --locked --all-targets --all-features` pass on Apple Silicon.

These gates establish parser, storage, and dense-row execution correctness.
They do not claim that a particular model graph has routed every BF16 tensor
role natively or that a whole artifact meets quality or performance gates.
