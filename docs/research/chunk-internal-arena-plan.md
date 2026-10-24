# Checked thirteen-root ChunkInternalArena payload plan

`ops::chunk_gated_delta_rule::ChunkInternalArenaPlan::for_shape` is an additive,
device-free description of the existing direct-allocation constructor. It keeps
thirteen ordered named roles, dtype, flat one-dimensional logical shape and exact
requested bytes. No Metal buffer, device, model or kernel runs while making a plan.
It is not a physical footprint, opaque metadata bound or admission certificate.

`ChunkInternalArena::new` consumes these SAME thirteen descriptors through the
existing `MlxDevice::alloc_buffer`; allocator, fresh zeroing, root order and
initialization order remain unchanged. Cumsum parameters are allocated before
other parameters but initialized last, as previously. The standalone public
parameter-builder signatures remain unchanged. They and the arena share private
fixed word counts/serialization, avoiding a second parameter-size/content table.
Per-call chunk-O scale update still uses its existing behavior.

The checks keep the constructor's established order: positive dimensions,
H%Hg==0, K128, V<=256, BT64, T%64==0. The plan adds explicit checked arithmetic
before any allocation, rejecting original u32 element-product overflow rather
than debug panic/release wrap or widening into unreviewed GPU indexing ranges.
It adds no new V-multiple requirement or model/context policy. Constructor
acceptance is not a claim every downstream dispatch accepts every shape: existing
subkernels can enforce additional execution constraints. `validate_fits` still
requires exact capacity equality. Device maximum-buffer checks remain in the
existing allocator, outside the pure plan.

## Selected Qwen3.8 endpoint

The current selected HF chunk caller supplies B1,T2048,Hg48,H48,K128,V128,BT64,
NT32. Hg48 follows its expanded/value-head constructor argument, not the model's
16 raw linear key heads. T is a prefill transaction row bound, not context length;
no four-slot multiplier or whole-model peak is inferred.

| Root | Dtype | Flat elements | Requested bytes |
| --- | --- | ---: | ---: |
| Cumsum |F32 |98304 |393216 |
| Strict |F32 |6291456 |25165824 |
| Inverse |F32 |6291456 |25165824 |
| W |BF16 |12582912 |25165824 |
| U |BF16 |12582912 |25165824 |
| Hidden |BF16 |25165824 |50331648 |
| VNew |BF16 |12582912 |25165824 |
| CumsumParams |U32 |5 |20 |
| KktParams |U32 |8 |32 |
| InvertParams |U32 |4 |16 |
| RecomputeWuParams |U32 |8 |32 |
| ChunkParams |U32 |8 |32 |
| ChunkOParams |U32 |11 |44 |
| **13 roots** | | |**176554160** |

Six parameter roots total176bytes. Their actual widths include NT/tile constants
and scale bits; stale arena comments understating several widths are corrected.
KKT's BK constant is64; chunk-O's own BK/BV constants are32/32. Shared word
functions preserve those DISTINCT values rather than inferring one common tile.
The seven large roots total176,553,984bytes. Requested payload is separate from
Metal allocation rounding, driver/pipeline metadata, external outputs, normalized
Q/K, other arenas, caches, anchors and retained command owners.

## Pure validation scope

Focused tests assert all thirteen ordered endpoint descriptors and total, shared
parameter values/padding/scale bits, minimum/multi-batch/grouped-head/MAX_V cases,
unchanged constructor-only V1 acceptance, existing error precedence and explicit
u32 overflow refusal. The author did not execute these tests; integration-owner results are recorded below.

```sh
cargo test --locked --lib ops::chunk_gated_delta_rule::arena_plan::tests -- --nocapture --test-threads=1
```

This checks source arithmetic/shared serialization, not GPU numerical parity,
actual allocation residency, all Qwen graphs or runtime safety/performance. It
does not adopt the optional bounded pool or modify legacy allocation/defaults,
HF inference, kernels, Cargo/version or release policy. Native publication and
verified downstream pin are required before HF can consume the additive API.
The separate DeepSeek indexer plan and any FA arena plan remain separate work;
this constructor plan alone does not close full Q5 memory admission.

An additional ignored constructor control allocates the actual
`ChunkInternalArena::new(B1,T64,Hg1,H1,K128,V128,BT64)` and binds every one of
its13fields to the plan's role/dtype/flatshape/requested bytes. It checks seven
large roots' entire bytes are zero, six parameter buffers against independent
fixed expected word arrays (including distinct tile constants/scale bits), and
all13Metal object identities are distinct. Exact requested buffer payload115,120B
plus finite static/borrowed test descriptors is below16MiB. Opaque device,
residency, Arc and allocator overhead is excluded/Unknown.

This is CPU Metal-buffer allocation/readback only: no encoder, command buffer,
kernel or GPU dispatch. The author has not executed it. Require exactly one
matched ignored case and `CHUNK_ARENA_ALLOCATION completed` with13roots and115120B;
a missing-device failure or zero-filter run is not a pass.

```sh
cargo test --locked --lib ops::chunk_gated_delta_rule::arena_plan::tests::actual_constructor_matches_thirteen_root_plan_and_parameter_contents -- --exact --ignored --nocapture --test-threads=1
```

The existing convenience GPU fixture exercises shared parameter serialization;
it does not call this constructor and must not be reported as `with_arena` GPU
parity. This allocation control establishes no GPU numerical or model result.


## Source-bound validation — 2026-10-24

The integration owner rebased the unchanged plan/control source onto
`adb7e8cb348a205f6f0de2f0aa5a67c9fed608fa` (including the separately validated
selected-shader fix). The ten-file execution freeze includes the earlier note;
this results-only documentation update changes no tested Rust bytes.

Five pure plan controls passed. All-feature library regression passed446 tests,
zero failed, nine ignored (1.38s). Locked all-target/all-feature check passed
(0.88s); both controlled targets compiled with --no-run. Locked release build passed (11.66s), with exit0 confirmed by the integration
owner for both check and build.
These build durations are records, not performance measurements. Ignored counts
are separate from explicitly executed controls.

Two separately supervised exact-one-test controls passed with no timeout, and
independent postexecution review verified their source, binary, logs and fixture
bindings. Both used an external process-group watchdog and no model process:

| Control | Actual result | Log SHA256 |
| --- | --- | --- |
| Actual constructor CPU allocation/readback |13 real distinct roots,115120B, seven whole-zero roots, six independently expected parameter arrays; no encoder/GPU dispatch | `76cd5f0837c68411152a7bcc723f2f2ecef500de7fcc568c731288a4e8172cf0` |
| Existing convenience GPU fixture |Finite actual/reference output and final state; max O error3.815e-6, final-state error8.503e-7, unchanged thresholds | `eba44c45f79589d29c0c19fa4cff1c5446b4e1774b179906a2056d0502331351` |

The first control's required `CHUNK_ARENA_ALLOCATION completed` marker confirms
roots13/requested115120/zeroed7/params6/gpu_dispatch=false. It exercises actual
`new`, not copied planner arithmetic. The second calls the convenience dispatcher
and shared parameter builders/serializers, not `ChunkInternalArena::new` or its
`_with_arena` GPU execution path. These are useful SEPARATE proofs. Their joint
success must not be advertised as arena-forward numerical parity.

Requested explicit Metal payloads were115120B for the constructor and1972408B
for the convenience fixture. Scoped host allowances and these payloads exclude
opaque allocator/device/driver/framework metadata; no full physical-memory bound,
resource final-free inference, model fit, speed or full Q5 admission follows.

Reproduce the convenience regression separately (its existing fixture data is
required; no fixture generation or model downloads are implied):

```sh
cargo test --locked --test test_chunk_gated_delta_rule_fwd test_chunk_gated_delta_rule_fwd_matches_fla_reference -- --exact --nocapture --test-threads=1
```

Require exactly one match, finite actual/reference values and the unchanged
numerical bars. The constructor command/marker above remains separately required.

Exact SHA256 evidence identities:

| Evidence | SHA256 |
| --- | --- |
| Execution receipt | `cdb0503b33ad44251d0d3ef2ab4fd62cf0a10191176bdbc8474eedf54895672d` |
| Ten-file execution freeze | `21cc673d9fea78e7936bf17c1ee775b254be0e305d1a187df5c1691f2b5d9742` |
| Constructor executable | `6b212a038af8b52a78e6eb005661664553ed84b16aea9d962c38820040f1d61c` |
| Convenience executable | `f37979acfa2d95591ef655c8710ce7a9f1573edb63f33eb9b3d5194b55714015` |
| All-feature library regression log | `2ff9469de3761cb34335f56743279fa1c114db3b6ec62164b70793214127c0c7` |
| All-target/all-feature check log | `8cac5421571e91b4feafb941bffea174a3317b5eea58d9fa486b3a5aa3754802` |
| Locked release build log | `7678ab5b4397e4587786ba3cf81330757aa64e52cbd203776b7dc970ef274ff8` |
| Controlled targets --no-run log | `e707f8ebbe9f067e2d1e4c4a682701f0123eb04efa3659bd714d839c7ed9fa42` |
| `src/ops/chunk_gated_delta_rule.rs` | `911e5b2727fd9455cc4b1a04e5f1ae2985fcf68a91d2a7959160f5dc3ec8d1c6` |
| `src/ops/chunk_gated_delta_rule_tri_solve_invert.rs` | `bfcddc25c356376b5d5d15f38d6b0d5d70589689dfc2d1d5879996306a9ff40e` |
| `src/ops/gated_delta_net_chunk.rs` | `8477531a6802101c76467634b1dad5e93ea3507017f3ae0ec6b4ae0c7217ae41` |
| `src/ops/gated_delta_net_chunk_o.rs` | `2e8be6c452a3468422d1aef7e1dd6ec7e766b7341f944fefb17f2979c7b82611` |
| `src/ops/gated_delta_net_kkt.rs` | `8f8b51d38718f744a2a4f85622c5b005858f243379c0152f5d81c5a76b549347` |
| `src/ops/gated_delta_net_recompute_wu.rs` | `8f4c6f8835e93bca6a354b949253cbfc094dff1747b947cb68a8761c71f16add` |
| `src/ops/chunk_gated_delta_rule/arena_plan.rs` | `c71a27b8c970e01dfca66ce26880478c268e338d38362328c3f3402006fb564c` |
| `src/ops/chunk_gated_delta_rule/arena_plan_tests.rs` | `13d7e2cab7b09be8db5f81f756ffb4edf19c2e1ac169bfe2b8841a6a379f5697` |
| `tests/test_chunk_gated_delta_rule_fwd.rs` | `6af5e54a9cf58fcdd209f1fdbc3b9e05439b2a60245b6361ce92dbfb7cbcd6a6` |
