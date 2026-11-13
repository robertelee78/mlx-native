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
- CI job `rust-metal-no-tensor-api` runs both and the dense fallback tests
  on the hosted `macos-26` device, where the unmodified registry must take the
  simdgroup route (`MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR=1`).

## 2026-11-12: kernel families that had no probe

### Problem

The probes above cover the kernels that have a portable twin. A second group
of kernels is compiled from sources that include `<metal_tensor>` and was
requested with a plain `get_pipeline`, with no probe at all. On either host
without the tensor API their first use returned the shader compiler's
diagnostic as the error of the dispatch.

| family | source | entry points | portable twin |
|---|---|---|---|
| prepared GGUF matmul | `gguf_repacked_mm.metal` | `prepare_repacked_gguf_weight`, `dispatch_repacked_gguf_mm*` | none |
| staged AF4G64 projection | `dflash2_af4g64_staged.metal` | `dispatch_dflash2_af4g64_staged`, `_gate_up` | none |
| ring draft attention | `dflash2_ring_attention.metal` | `dispatch_dflash2_ring_attention` | none |
| prepared expert planes | `quantized_matmul_id_grouped_mv.metal` | `prepare_quantized_matmul_id_ggml_grouped_m8`, `quantized_matmul_id_ggml_grouped_m8_*` | none |
| MoE expert live tiles | `moe_expert_tiles.metal` | `moe_tiles_*`, `moe_tiles_fused`, `moe_tiles_column_tile_exact` | none |
| head-major matmul | `quantized_matmul_mm_tensor.metal` (`*_bf16_perm021`) | `quantized_matmul_mm_tensor_perm021*` | none in one dispatch (the caller permutes the activation and takes the ordinary matmul) |
| large-tile F16 matmul | `quantized_matmul_mm_tensor.metal` (`hf2q_mul_mm_tensor_v2_f16`) | `dispatch_mm_v2_f16` | the portable F16 matmul of `dense_mm_f16` |

No other feature of any shader is absent on a supported Mac: of the 169
sources, twelve include `<metal_tensor>` (the seven families above and the
five probed ones), and the rest use only what every Apple-silicon GPU has on
the oldest macOS the crate's `bfloat` kernels already require (`bfloat`,
`simdgroup_float8x8`, 32-bit atomics).

### Decision

`ops::tensor_kernels` names these families (`TensorKernelFamily`) and gives
them one public query, `tensor_kernel_family_available(registry, device,
family)`. It builds one kernel of the family that multiplies through the
tensor API and classifies a failure with the classifier of every other probe
(`dense_mm_capability::is_unavailable_tensor_api`): the missing header, or
any failed build on a device outside the Metal 4 family, is `Ok(false)`;
anything else on a Metal 4 device is the error it is and is not cached. The
answer is cached in the registry, which is bound to one device, in a
fixed-size table, so a family's entry points check it on every dispatch
without building a key.

- A family without a twin returns `MlxError::Unavailable { feature, reason }`
  from every entry point, after its arguments are validated and before it
  allocates or encodes. The query is the contract: a caller asks before it
  chooses the route.
- `dispatch_mm_v2_f16` has a twin and takes it by itself: where the tensor
  kernel is unavailable it encodes `hf2q_dense_mm_f16_f32_fallback`, at
  every K (the portable kernel zero-fills the K tail of its last 32-wide
  tile, so it also serves K below one tile, which the tensor kernel takes
  too). The two routes are Class B to each other, not bitwise: the portable
  kernel stages the activation as half and accumulates in another order.
  On the M5 Max the worst error over `sum(|w * x|)` against the F64 product
  is 2.6e-4 for the portable kernel and 4.4e-4 for the tensor kernel (the
  worst cases are at K = 16 and K = 40, where one term's half rounding is
  not averaged over many terms), and the relative RMS between the two routes
  is at most 5.9e-4; the independent review of this change found at most one
  bit-equal element per run. The gate holds both to
  `(2^-11 + K * 2^-24) * sum(|w * x|)`: the half rounding of one factor and
  the FP32 rounding of K accumulated terms.
- The tensor flash attention and the tiled verify attention already had
  queries; their dispatches now return the same typed error instead of the
  diagnostic when a caller skipped the query.
- `KernelRegistry::simulate_missing_tensor_header` makes a registry compile
  as a host without the header does, for every tensor source, so a consumer
  can test its portable routes on a tensor-capable machine.
- `MLX_NATIVE_DISABLE_METAL_TENSOR=1` answers "unavailable" for every family
  and the typed error then names the variable as the reason
  (`TENSOR_KERNELS_DISABLED`; `tensor_kernel_family_unavailable_reason()`
  says which reason applies in a process). The variable makes a mixed host,
  not a host without the tensor API: it selects the portable dense
  BF16/F16/F32 matmuls and turns these families off, but the quantized and
  expert matmul probes, the tensor flash attention and the tiled verify
  attention do not read it and keep their tensor kernels. Use
  `simulate_missing_tensor_header` to reproduce a host without the API.
- `MlxError` is `#[non_exhaustive]` from this change on. Adding
  `Unavailable` to an exhaustive public enum is a breaking change by itself
  (`cargo semver-checks`: `enum_variant_added`), so the release that carries
  it is 0.22.0, and the attribute makes later variants compatible.

### Gates

- `ops::tensor_kernels_tests`: with the header missing every family answers
  "unavailable" and each of seven entry points returns the typed error; a
  probe source that fails for another reason is an error on a Metal 4
  device; a changed source is asked again; and on the unmodified registry
  every entry point agrees with its family's answer.
- The same module holds the Class B gate of the F16 twin: at seven shapes
  (whole and partial N and K tiles, K = 2048, K = 40 and K = 16) and 130,
  64, 33, 32, 9, 8, 2 and 1 rows, `dispatch_mm_v2_f16` without the tensor
  API encodes the portable kernel, every element is within
  `(2^-11 + K * 2^-24) * sum(|w * x|)` of the F64 product, a repeated dispatch gives the
  same bits, and the rows of a shorter batch are bitwise the leading rows
  of a longer one. Where the device has the tensor API the tensor kernel is
  held to the same F64 bound.
- `kernel_registry::tests::simulated_missing_tensor_header_stops_tensor_kernels_only`.
- CI job `rust-metal-no-tensor-api` runs these and the dense gates on the
  hosted `macos-15` (no header) and `macos-26` (no Metal 4 family) runners
  with `MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR=1`, where every family must be
  unavailable.
