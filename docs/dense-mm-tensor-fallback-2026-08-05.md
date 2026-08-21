# Dense matrix-multiply tensor fallback — 2026-08-05

## Problem

The `mlx-native` 0.10.0 release candidate allowed an SDK without
`<metal_tensor>` to omit five optional tensor shaders from the embedded
metallib.  The quantized matrix-multiply operators already probed their tensor
pipelines and selected simdgroup kernels, but the dense BF16, F16, and F32
operators still requested their tensor pipelines unconditionally.  On the
hosted M1 runner this caused all dense BF16 tests to fail when the registry
retried the unavailable shader as runtime source.

## Hypothesis and reference check

Hypothesis: treating `<metal_tensor>` as an optional build capability is safe
only if every operator that consumes one of those pipelines has a non-tensor
runtime route.

The hypothesis was checked against the peer reference checkout at the
pinned commit listed in `docs/peer-benchmarks.md`.  Its Metal backend
enables its tensor path only after a device/compiler probe and otherwise
compiles dense matrix multiply through a non-tensor simdgroup implementation.
It also disables the tensor API by default before M5/A19 because that path is
not a performance win there.  Therefore retrying a tensor-only source is not
parity behavior; operator dispatch must select a portable implementation.

## Decision

- Probe each dense tensor pipeline once per device-scoped kernel registry
  because supported element types may differ on the same device and SDK.
- Preserve the tuned tensor kernels on capable hardware.
- Dispatch BF16, F16, and F32 to llama-style 64x32 tiled simdgroup-MMA
  kernels when the exact tensor pipeline cannot compile.  Four simdgroups
  reuse shared A/B tiles rather than launching one group per output scalar.
- Round F32 activations to BF16/F16 while staging those two fallback variants,
  matching the tensor kernels' element semantics; the F32 route remains F32.
- Support `MLX_NATIVE_DISABLE_METAL_TENSOR=1` so the fallback can be exercised
  deterministically on tensor-capable development machines.
- Keep shader inventory and runtime selection fail-closed.  Only the exact
  missing `<metal_tensor>` header becomes an unavailable capability;
  unrelated shader and pipeline failures propagate without being cached.

## Evidence before hosted release gate

On the M5 Max, both routes were exercised from separate test processes:

- forced fallback: BF16 11/11, F16 12/12, F32 9/9;
- normal tensor route: BF16 11/11, F16 12/12, F32 9/9;
- tensor-vs-fallback parity at a partial M/N/K tile with 2:4 GQA broadcast:
  BF16, F16, and F32 each pass a `1e-4` maximum-absolute-difference gate;
- BF16 production attention shape (`M=N=128`, `K=256`, 4:16 GQA), 21-sample
  medians after five warmups: tensor 223 us, tiled fallback 294 us on M5 Max
  (1.318x).  The fallback is for environments where the tensor route is
  unavailable; its geometry matches the reference non-tensor baseline
  (`docs/peer-benchmarks.md`);
- shader inventory: 140 files checked, 0 failures on the local SDK.

The hosted M1 result and exact packaged-crate digest are release gates and are
recorded in the release evidence after the source commit is created.
