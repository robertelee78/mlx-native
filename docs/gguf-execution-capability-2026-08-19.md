# GGUF execution capability contract — 2026-08-19

## Decision

Artifact producers and mixed-precision policy allocators must reason about the
exact operation that consumes a GGUF tensor, not merely its quantization type
or nominal bits per weight. `ggml_capability` is the versioned, serializable,
device-independent contract for block-quantized operations and the native
dense/embedding operations it explicitly models. Native scalar expert stacks
are outside that schema and use the separately versioned
`dense_matmul_id_capability`; a scalar `Expert*` rejection from
`ggml_capability` is not authority to transform the matrix.

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
- Direct embedding gather supports native F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q2_K,
  Q4_K, Q5_K, and Q6_K tables and converts only selected rows to the graph's
  F32 activation dtype. This is row execution, not a second resident weight
  representation. Other codecs fail closed at capability admission.
- Dense F32, F16, and BF16 projections expose explicit native scalar routes at
  decode, continuous widths through eight, and prompt/physical-batch widths
  above eight. The capability receipt accounts for the exact two- or four-byte
  tensor extent and never describes the scalar tensor as a quantized block.
- `dense_matmul_id_capability` is the corresponding native scalar expert-stack
  boundary. It accepts F32, F16, or BF16 weights without a shadow
  representation, F32 input/output, U32 expert IDs, shared-per-token or
  per-slot input rows, and an explicit padded expert stride. It returns exact
  logical byte extents, route, dispatch count through the execution receipt,
  and caller-owned scratch requirements. The primitive route is explicit;
  production selection is owned by an activation-time calibration plan rather
  than inferred from a static M threshold. Repeat-allowed routing requires the
  one-dispatch Direct route, while a proven distinct-ID case may select the
  bounded map plus grouped multiply. Grouped invocation owns a pre-map memory
  barrier that orders same-encoder scratch reuse and a separate map-to-multiply
  barrier; scratch reuse across independent encoders still requires the prior
  submission to complete. The grouped multiply retains the direct
  kernel's lane ownership, per-lane accumulation order, and SIMD reduction
  order, so route selection does not change output bits.

  This boundary owns structural admission for native scalar expert stacks; it
  does not share or infer authority from `GGML_CAPABILITY_SCHEMA_VERSION`.
  Capability, activation calibration, and dispatch trace use
  `DENSE_MATMUL_ID_SCHEMA_VERSION`. The exact encoded dispatch geometry and
  compiled pipeline identities are serializable evidence, and a frozen plan
  rejects mutation of every expert shader registered by the primitive.

  `calibrate_dense_matmul_id_routes` consumes the one-shot union of exact cases
  keyed by native dtype, M/N/K, top-k, expert count, padded expert stride, input
  layout, and ID multiplicity. Every distinct same-shape weight identity is
  borrowed for exact extent and identity validation, but only one current
  weight representative supplies empirical Direct/Grouped proof and AB timings
  for that exact shape using adversarial F32 activations and routing.
  Both a balanced distribution and the maximally skewed
  distribution permitted by distinct-per-token routing must independently pass
  the coherence and timing gate before Grouped can win.

  Authority for the remaining declared identities is a source-level
  value-independence theorem, not an unperformed current-buffer proof. Direct
  and Grouped call the same Metal helpers for input addressing, expert-base and
  within-expert weight-scalar addressing, native F32/F16/BF16-to-F32 widening, explicit F32 fused
  multiply-add, and `simd_sum`; the grouped route only reorders independent
  output rows and stages an already-widened scalar. The theorem is bound to the
  complete shape (including stride, layout, multiplicity and tail geometry),
  exact compiled direct/map/grouped pipeline identities, source/build
  fingerprint, native dtype, and a canonical theorem SHA-256 embedded in the
  frozen plan and trace. It never generalizes across dtype, pipeline label,
  shape, stride, layout, multiplicity, K-tail, or N-tail.

  The frozen plan and process cache retain no model pointer or buffer; the
  latter contains reusable route/timing metadata only. A cached Grouped winner
  reruns the representative exact-shape proof under both routing profiles
  (four submissions, six dispatches) before it can enter a new activation
  plan. Receipts distinguish declared identities, theorem-authorized
  identities, empirical exact-shape proof submissions, current timing work,
  and historical cached timing samples. Optional
  Grouped proof or timing failure is activation-local, selects Direct, and does
  not poison reusable timing metadata. Required Direct allocation, execution,
  commit, timing, or readback failure aborts activation without publishing a
  plan. Every proof/timing command-buffer
  attempt is counted immediately before commit; after any nonzero attempt, both
  success and error paths drop proof state and commit exactly one empty cleanup
  boundary before deferred cache eviction or plan validation. Receipts distinguish current
  proof/timing submissions from historical cached timing sample counts. The
  calibrated, process-cache-hit, and Direct/fallback decision counts are a
  disjoint exhaustive partition of declared exact shapes. Plans
  are immutable and scoped to a nonzero activation epoch. An exact declared
  shape uses its frozen decision. An unseen M may execute Direct with
  `UndeclaredDirect` evidence only when dtype, N/K, top-k, expert count, padded
  stride, input layout, and ID multiplicity match an admitted base. An unseen
  base, missing plan, or stale epoch/device fails before encoding. Error,
  budget exhaustion, incoherence,
  direct wins, and unstable results all select Direct. A→B→A model swaps may
  reuse full-shape timing metadata while receiving a new epoch and plan ID.
  Every Grouped activation still executes the exact-shape representative proof;
  tests prove different A/B/A weight bytes, bitwise execution parity, and that
  dropping the caller's last weight reference releases the allocation.

  Parallel execution registries do not recalibrate or accept a transferable
  route token. The main activation plan and receipt carry a one-way authority
  digest over the activation epoch, Metal device identity, exact shape union,
  and sorted logical weight identities (buffer address, byte offset, logical
  byte extent). `freeze_dense_matmul_id_plan_for_cases` recomputes that digest
  from freshly borrowed cases, prepares and validates the exact pipeline set,
  and only then installs the pointer-free plan. The same live model's
  main-to-worker freeze is accepted; an A plan with B buffers, a stale epoch,
  a different device, or a changed pipeline/theorem/build fails before worker
  dispatch.

  Callers that must add graph barriers before encoding use
  `resolve_dense_matmul_id_auto_route`. It is the same resolver consumed by
  `dense_matmul_id_auto`, validates the frozen plan, epoch, device, theorem,
  exact or admitted-base shape, selected-route capability, weight alignment,
  and required logical extent, and performs zero encoder mutations or GPU
  submissions. This keeps an imatrix or `GraphSession` failure ahead of any
  barrier/commit side effect.

  Direct pipeline preparation is an activation prerequisite because no native
  scalar expert call can execute without it. A map or grouped-pipeline
  preparation failure is instead receipted as `ErrorFallback` and freezes
  Direct for the affected dtype; an optional performance route cannot prevent
  an otherwise executable model from activating.

## Native scalar expert-ID spike — 2026-08-23

The first grouped prototype staged both native weights and F32 activations in
the weight scalar type. That was rejected: it silently rounded F32 activations
before multiplication and its matrix reduction did not belong to the direct
route's coherence class. The accepted kernel widens each stored scalar weight
at the multiply, keeps activations F32, shares a 128-element K tile across up
to eight routed rows, and otherwise executes the direct lane/reduction order.
Forced Direct/Grouped tests at M=9 and M=33, both input layouts, adversarial
F32 values, odd K=35, and selected experts including 1, 3, and 7 are bitwise
equal (`max_delta=0`). A single-weight test separately proves an F32 value
that is not BF16-representable survives the BF16-weight path unchanged.
Poisoned-output invalid-ID fixtures also force both routes and both layouts:
the map writes zero rows without accessing an invalid expert, the grouped
multiply skips those rows, and every logical output byte matches Direct. This
matters because output buffers and scratch are caller-owned and may be reused.

The production-width hardware spike used an Apple M5 Max, native BF16 weights,
F32 shared input/output, U32 IDs, M in {9, 33}, N=2,048, K=4,096, top-k=6,
eight resident experts, two warmups, and nine AB-alternated samples. The
grouped route remained bitwise equal and measured 1.57x wall speedup at M=9
(1.961 ms direct versus 1.249 ms grouped) and 1.27x at M=33 (2.978 ms versus
2.352 ms); median GPU times were 1.654/0.783 ms and 2.739/2.125 ms. The exact
reproduction command is the ignored `bench_dense_matmul_id` test with
`MLX_DENSE_MATMUL_ID_M=9` or `=33`. This branch is based on main `1dc4bd33`;
the local hardware receipt is qualification evidence, not release evidence.
The calibrated stability gate selected Direct/`NoStableWinner` at M=9 despite
the raw median and Grouped/`CalibratedWinner` at M=33; production follows that
fail-closed decision rather than the headline ratio.

Three losing shapes were removed rather than hidden: a 32-element K tile paid
too many barriers (0.63x/0.58x at M=9/33); a 512-element tile reduced
occupancy (0.75x/0.68x); and a 16-output tile increased register pressure
(1.15x/1.06x). The remaining 8-row by 8-output, K=128 tile is the measured
winner from this bounded spike.

The original activation-gate spike used one balanced distribution: two proof
submissions, five order-alternated AB pairs, and one empty cleanup boundary.
Its 13-submission/18-dispatch timings are obsolete for the production
authority. The current gate repeats the proof and five AB pairs for balanced
and maximally skewed distinct routing, then issues one cleanup boundary: 25
submissions and 36 kernel dispatches for one representative of an uncached
shape. A cached Grouped decision performs four proof submissions/six
dispatches for one representative of that exact shape, independent of the
number of declared same-shape weight identities.

The rejected full-Cartesian formulation would have proved every distinct
weight at every scheduler-reachable M. Source-derived Qwen coverage alone is
64 layers × (13 gate/up widths × 2 projections + 11 down widths) = 2,368
weight/shape authorities, or 9,472 proof command buffers before timing and
cleanup. This exceeds the 6,145-submission and 15-second activation contract;
an earlier structural 48-layer/3-projection/7-width spike required 4,032 proof
submissions and 734.625 ms even for tiny 5–9-wide F32 matrices. The attempted
Cartesian policy is therefore falsified for universal model-swap use. The
replacement is valid only because the shader now encodes the shared
value-independent scalar operation directly and the plan binds its theorem
digest plus exact pipeline/shape authority; a substring/source-comment canary
is not accepted as proof.

Process-cache reuse is conditioned on the exact prepared pipeline identities,
build fingerprint, theorem digest, and full shape, not merely source text or
M/N/K. Candidate-bearing and all-Direct plans are both fully validated
after the cleanup boundary. Only then is the authoritative deadline sample
taken, followed immediately by an infallible plan install; a positive deadline
overrun cannot freeze Grouped.

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
