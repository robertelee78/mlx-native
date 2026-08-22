# GGUF execution capability contract — 2026-08-19

## Decision

Artifact producers and mixed-precision policy allocators must reason about the
exact operation that consumes a GGUF tensor, not merely its quantization type
or nominal bits per weight. `ggml_capability` is the versioned, serializable,
device-independent contract for that question.

A request binds:

- the public entry point and all shape/layout fields that affect execution;
- the GGUF block type;
- the decode, continuous-width, prompt, or embedding workload;
- every process routing override that changes the structural branch.

The result binds the selected structural route, exact packed weight extent,
weight-buffer count, scratch ownership, dispatch/barrier counts, whether the
route is specialized for the requested workload, and whether exact-device
kernel probing remains necessary.

The capability implementation and canonical dense and expert dispatchers share
pure route planners. The public `*_with_policy` entry points also consume the
same serialized policy for NR2, width-N, tensor/simdgroup, large-tile, and
expert-threshold decisions. The Q6/Q8 pre-baked dispatch-record builders have
matching `*_with_policy` forms because those records are part of the executed
program too. Legacy entry points resolve environment overrides
once through the public `ggml_routing_policy_from_environment()` resolver and
delegate to those explicit-policy entry points. Evidence-producing callers
resolve that whole dense+expert policy at model load, serialize it, and pass the
same value to every capability query and explicit dispatch. This is deliberate:
a copied predicate or an independently-read environment could allow the receipt
and production execution to drift while both continued to compile.

## Why tensor type alone is insufficient

The same packed type can execute through materially different programs:

- decode matvec, continuous-width `mvN`/`mul_mv_ext`, or prompt matrix-matrix;
- independent batched weights versus one shared weight;
- permuted-output attention projection;
- a fused gate+up+SiLU pair rather than two dense calls;
- expert matvec, auto-allocated `mm_id`, caller-pooled `mm_id`, a paired
  schedule, slotted input, or fused Q4 expert SwiGLU/down;
- direct embedding gather, which supports a narrower type set.

Therefore storage bytes are not an Apple runtime cost model. A producer may
rank an option only after the exact graph invocation is executable and its
device-resolved route has a matched benchmark receipt.

## Fail-closed boundaries

- Tensor-API versus simdgroup MM selection and device limits such as the expert
  schedule threadgroup width remain runtime device/metallib probes. A
  structural `DeviceSelected` result is not a resolved cost claim.
  Cost evidence must execute the corresponding `*_with_policy` entry point and
  record its labeled pipeline-dispatch buckets along with the device, OS,
  metallib, crate revision, routing policy, shape, warmups, runs, and latency
  distribution.
- Unsupported operation/type/layout combinations are rejected rather than
  silently priced as a generic matmul.
- Expert stacks use `(n_experts - 1) * expert_stride + matrix_bytes`, not
  `n_experts * matrix_bytes`. Runtime validation uses the logical view length
  so a padded or offset-backed buffer cannot pass by exposing unrelated bytes
  from its underlying allocation.
- Coupled weights such as fused gate/up and paired expert projections are one
  operation with multiple buffers. A per-tensor allocator must treat that
  operation's compatibility constraints atomically.
- Capability is not a throughput estimate and does not authorize a model
  loader to dequantize or re-quantize the selected representation.
- Dispatch and barrier counts describe the invocation itself. Graph casts,
  copies, encoder-inferred barriers, command-buffer boundaries, and surrounding
  fusion remain part of the measured whole-graph cost.
- F16-shadow entry points are deliberately outside this GGUF contract. An
  allocator must describe those as an explicit stored-GGUF to executed-F16
  transformation and query/measure the F16 operation rather than pretending
  the stored GGUF codec executes directly.
- Unquantized GGUF F32, F16, and BF16 tensors retain their declared two- or
  four-byte scalar storage in the raw loader. `load_tensor_f32` remains an
  explicit diagnostic/materialization request; model loaders must not use it
  to disguise a missing native execution route.
- Direct dense embedding gather supports F32, F16, and BF16 tables and converts
  only selected rows to the graph's F32 activation dtype. This is row execution,
  not a second resident weight representation.

## Evidence and remaining scope

Schema-v1 model-free tests cover dense routing overrides, short-prompt and
IQ4_XS fallbacks, batched and permuted entry points, fused pairs, expert
scratch/stride/layout constraints, direct embeddings, JSON round trips, and
checked padded expert extents. Apple tests prove that explicit width-MVN,
MV_EXT, baseline/NR2, tensor/simdgroup, large-tile, and expert-threshold policy
fields select the corresponding production pipeline labels. Logical-slice
tests cover every fused dense codec and both embedding codecs. Real committed
Metal executions select the second expert across tight and poison-padded
layouts and produce bit-identical outputs for auto-MV, pooled MM, paired MM,
and fused-Q4 SwiGLU. Production dense/expert dispatch uses the same structural
planners, policy object, and checked extent helper.

Still required above this layer:

1. a device-resolved route receipt keyed by Apple chip, OS, metallib, and
   effective routing policy;
2. exact shape-regime latency and memory measurements;
3. an hf2q source-to-stored-to-loaded-to-executed manifest proving that a
   selected precision survives model loading;
4. whole-model quality, behavior, and performance gates before any mixed
   policy is called optimal.
