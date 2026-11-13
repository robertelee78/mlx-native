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

## 2026-11-12: a device outside the Metal 4 GPU family

### Problem

Since 0.10.16 every tensor probe (dense F32/BF16/F16 matmul, quantized
`mul_mm`, expert `mul_mm_id`, tensor flash attention, the tiled verify
attention) treated exactly one failure as "tensor API unavailable": the
runtime compiler's missing `<metal_tensor>` header. Any other failure of the
probe pipeline was returned as the error of the dispatch that asked.

The GitHub-hosted `macos-26` runner (macOS 26.7.2, Xcode 26.6, "Apple
Paravirtual device") has the header and cannot build a tensor pipeline. Each
stage was run on it (scratch branch `diag/ci835-macos26-probe`, run
43347804223):

| stage | `macos-15` (15.8.2, Xcode 16.4) | `macos-26` (26.7.2, Xcode 26.6) | M5 Max |
|---|---|---|---|
| `supportsFamily(Metal4)` | false | false | true |
| source compile of the tensor shader | `fatal error: 'metal_tensor' file not found` | ok | ok |
| function specialisation | n/a | ok | ok |
| pipeline creation | n/a | `error: unsupported deferred-static-alloca-size function body in ...metal18cooperative_tensor...MTL_SIZEAS` | ok |
| default-policy quantized matmul at 9, 32, 33, 64 rows (Q4_0, Q4_K) | simdgroup kernel | error of the probe pipeline `kernel_mul_mm_q4_0_tensor_f32` at every row count | tensor kernel |
| the same with `GgmlTensorMmPreference::ForceSimd` | ok | ok | ok |

The probe runs before any route is chosen, so the failure does not depend on
the shape, the row count or the weight's type: on that device no dense
quantized matmul of more than eight rows could be encoded under the default
policy, and the other tensor probes failed the same way.

### Decision

The tensor API belongs to the Metal 4 GPU family, and the reference
implementation asks the device for that family before it tries a tensor
kernel (`docs/peer-benchmarks.md`). A failed tensor pipeline build now means
"unavailable" in two cases (`dense_mm_capability::is_unavailable_tensor_api`):

- the missing `<metal_tensor>` header, on any device, as before;
- any pipeline build failure on a device that does not support
  `MTLGPUFamilyMetal4`.

On a device that supports Metal 4 every other failure still propagates
without being cached, so a shader or toolchain regression on the hardware the
tensor kernels are tuned for stays fatal. The family is consulted only after a
build failed: a device outside the family whose tensor pipeline does build
keeps the tensor route.

### Gates

- `ops::dense_mm_capability` pins the classifier with the hosted diagnostic.
- `ops::quantized_mm_tensor_unavailable_tests` pins the quantized matmul
  route: with the tensor probe unavailable, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K at
  9, 32, 33, 64 and 130 rows each encode the simdgroup kernel of their type
  and give the bits of the policy that forces it.
- CI job `rust-metal-no-metal4-device` runs both and the dense fallback tests
  on the hosted `macos-26` device, where the unmodified registry must take the
  simdgroup route (`MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR=1`).
