# Native Q5_0 GGUF runtime — 2026-08-23

Status: source-complete; compile, Metal parity, downstream artifact,
quality, and performance validation pending.

## Contract

Q5_0 is public GGUF storage, not an internal conversion target. Type ID 6 is
admitted as 32 values in a 22-byte block: one F16 scale, a 32-bit high-bit
mask, and sixteen packed low-nibble bytes. The stored block remains an opaque
U8 logical view when copied or file-mapped. Production execution consumes
those bytes directly; no loader may widen or re-encode the matrix to compensate
for a missing kernel.

Native source coverage includes:

- GGUF header parsing, checked row-local byte sizing, raw mapped storage, and
  diagnostic CPU dequantization;
- dense matvec for any admitted M through the ordinary per-row route, the
  opt-in width-amortized route for M=2..8, and simdgroup/tensor matrix-matrix
  routes for arbitrary prompt M including tail tiles;
- independent batched matvec and contiguous or explicitly strided batched
  matrix-matrix execution;
- BF16-input permuted-021 matrix-matrix execution;
- expert-routed matvec, auto/caller-pooled matrix-matrix, shared and slotted
  activation layouts, and paired projections sharing one routing schedule;
- direct embedding gather.

The existing whole-tensor F16-shadow materialization API deliberately rejects
Q5_0. Native Q5_0 coverage must not make a load-time representation
substitution newly possible; diagnostic dequantization remains CPU-only.

Codec-specific fused gate/up and fused expert-down kernels remain limited to
the codecs they actually implement. A Q5_0 caller uses the ordinary native
Q5_0 projections instead of changing the stored representation. Capability
admission exposes that distinction: every native dense, batched, permuted,
expert, and embedding entry point is admitted for Q5_0, while unsupported
fused-Q4 entry points fail closed.

## Proof plan

The checked-in model-free contract pins type ID, block size, raw storage dtype,
high-bit ordering, signed zero point, malformed-block failures, every
applicable capability invocation, and all required Metal symbol families.
Once the shared Apple hardware lane is available, the blocking validation is:

1. focused non-GPU tests plus all-target/all-feature check and release build;
2. Metal-vs-independent-CPU parity at dense M=1/2/8/9/33/129, including forced
   width-amortized, simdgroup, tensor, tail-N, and permuted-021 routes;
3. independent batched MV and contiguous/strided MM parity;
4. expert MV, auto-MM, shared/slotted pooled MM, and pooled-pair parity with
   distinct experts and padded expert strides;
5. embedding parity across low/high nibble and high-bit boundaries, duplicate
   IDs, mapped offsets, and fail-before-encoding malformed IDs/layouts;
6. fail-closed proof that F16-shadow materialization rejects Q5_0, then the
   complete locked test suite;
7. downstream hf2q source-to-stored-to-mapped-to-executed receipts for real
   Q5_0 artifacts, followed by matched quality and performance gates.

No performance claim is made until those gates complete.
