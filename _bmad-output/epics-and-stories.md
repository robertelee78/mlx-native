# mlx-native v0.2.0 — Epics and Stories

**Governing PRD:** `_bmad-output/prd-mlx-native-v0.2.md`
**Governing ADR:** ADR-006 (mlx-native as hf2q's GPU Compute Backend)
**Phase 0 Verdict:** Framework-overhead-dominated → graph scheduler is the primary lever

---

## FR Coverage Map

| FR | Epic | Stories |
|----|------|---------|
| FR-1 Quantized matmul | Epic 1 | 1.1, 1.2 |
| FR-2 Grouped MoE dispatch | Epic 2 | 2.1 |
| FR-3 Attention (SDPA) | — | Already covered; tests exist |
| FR-4 Norm + activation | Epic 2 | 2.2, 2.3 |
| FR-5 RoPE | — | Already covered; tests exist |
| FR-6 Elementwise + utility | Epic 2 | 2.4, 2.5 |
| FR-7 Embedding + sampling | Epic 1 | 1.3 |
| FR-8 KV cache | Epic 1 | 1.3 |
| FR-9 Dense GEMM (lm_head) | Epic 2 | 2.6 |
| FR-10 Graph scheduler | Epic 4 | 4.1, 4.2, 4.3 |
| NFR-1-3 Perf + correctness | Epic 5 | 5.1, 5.2 |
| NFR-4-6 Packaging + CI | Epic 6, 7 | 6.1-6.3, 7.1-7.3 |

---

## Epic 1: Test Gap Closure (Sprint 1)

**Dependencies:** None (this is the foundation)
**FRs covered:** FR-1, FR-7, FR-8
**Rationale:** The most load-bearing kernel in the library (`quantized_matmul`, 1403 LOC) has zero tests. Multiple fused kernels and utility ops are also untested. No further work is safe until we know the existing code is correct.

### Story 1.1: quantized_matmul unit tests

**As a** maintainer of mlx-native,
**I want** unit tests for `quantized_matmul` covering all 6 quant types at hf2q's production shapes,
**So that** I can trust the most critical kernel before borrowing additional code on top of it.

**Acceptance Criteria:**
- Given Q4_0 weights at shape [4096, 4096] (Gemma attention projection), When `quantized_matmul` is called with M=1 (decode) and M=8 (prefill), Then output matches CPU affine-dequant reference at max|Δ| ≤ 1e-4
- Given Q6_K weights at the same shapes, When dispatched, Then same tolerance
- Given Q8_0 weights at the same shapes, When dispatched, Then same tolerance
- Given Q4_1, Q5_0, Q5_1 weights, When dispatched, Then same tolerance
- Given all tests pass, Then each test prints the actual max|Δ| in the assertion message for future reference

**Technical Notes:**
- File: `tests/test_quantized_matmul.rs`
- Reference: CPU affine dequant loop (`(weight_uint * scale + bias) @ input`)
- Shapes from Phase 0 data: M=1 N=4096 K=4096, M=1 N=14336 K=4096, M=8 N=4096 K=4096
- Story points: 3

### Story 1.2: quantized_matmul bitwise validation vs candle

**As a** hf2q integrator,
**I want** proof that mlx-native's `quantized_matmul` produces the same output as candle's `call_quantized_matmul_mv_t`,
**So that** Phase 5 per-op cutover won't introduce silent numerical drift.

**Acceptance Criteria:**
- Given identical Q4_0 packed weight bytes + identical input vector, When dispatched through both candle and mlx-native, Then max|Δ| = 0.000000e0 (bitwise)
- Given Q6_K and Q8_0 same setup, Then bitwise match
- Given the test fixture covers at least 3 shapes per quant type, Then shape-dependent bugs are caught

**Technical Notes:**
- This test lives in hf2q (not mlx-native) since it needs both candle and mlx-native as dependencies
- File: `tests/test_candle_mlx_parity.rs` in hf2q
- Blocked by: Story 1.1 (mlx-native tests must pass first)
- Story points: 5

### Story 1.3: Test untested existing ops

**As a** maintainer,
**I want** unit tests for argmax, softmax_sample, kv_cache_copy, permute_021, and all 3 fused kernels,
**So that** no untested kernel silently breaks during Phase 3/4 work.

**Acceptance Criteria:**
- Given `argmax_f32` on a random [1, 262144] tensor, When dispatched, Then result matches CPU argmax
- Given `softmax_sample_f32` with T=1.0, When run 1000 times, Then output distribution is approximately uniform for uniform logits (chi-squared p > 0.01)
- Given `kv_cache_copy` with sliding window wrap-around, When `seq_pos > window_size`, Then output buffer has correct wrap-around (compared vs CPU copy)
- Given `permute_021_bf16` on [4, 32, 128], When dispatched, Then output matches CPU transpose
- Given `fused_residual_norm_bf16`, When dispatched, Then output matches sequential `elementwise_add` + `rms_norm` at ε ≤ 1e-4
- Given `fused_head_norm_rope_bf16`, When dispatched, Then output matches sequential `rms_norm_no_scale` + `rope_neox` at ε ≤ 1e-4
- Given `fused_norm_add_bf16`, When dispatched, Then output matches sequential `rms_norm` + `elementwise_add` at ε ≤ 1e-4

**Technical Notes:**
- Files: `tests/test_argmax.rs`, `tests/test_sampling.rs`, `tests/test_kv_cache.rs`, `tests/test_fused_ops.rs`
- Story points: 5

---

## Epic 2: Missing Ops (Sprint 1-2)

**Dependencies:** Epic 1 (existing ops must be tested before adding new ones)
**FRs covered:** FR-2, FR-4.2-4.4, FR-6.4-6.6, FR-9

### Story 2.1: Grouped MoE dispatch (kernel_mul_mv_id)

**As a** hf2q integrator,
**I want** `kernel_mul_mv_id` family (Q4_0, Q6_K, Q8_0) in mlx-native,
**So that** MoE layers can dispatch per-token to per-expert quantized weight matrices on GPU.

**Acceptance Criteria:**
- Given 4 tokens routed to 2 experts each (top-2) from 8 available experts, When `quantized_matmul_id` is called with Q4_0 weights, Then output matches candle's `kernel_mul_mv_id_q4_0_f32` at max|Δ| = 0 (bitwise)
- Given Q6_K and Q8_0 expert weights, Then same bitwise match
- Given the kernel source is borrowed from candle, Then attribution header present per ADR-006 §Phase 3

**Technical Notes:**
- Source: `vendor/candle-metal-kernels/src/metal_src/quantized.metal:7544-7618`
- File: `src/ops/quantized_matmul_id.rs` + `src/kernels/quantized_id.metal`
- Attribution: Apache-2.0 from candle with `LICENSE-APACHE-candle` file
- Story points: 8

### Story 2.2: bf16 variants (gelu, softmax, softcap)

**As a** hf2q integrator,
**I want** bf16 dispatch paths for gelu, softmax, and softcap,
**So that** the full bf16 forward pass pipeline doesn't need f32 casts for these ops.

**Acceptance Criteria:**
- Given bf16 input tensor, When `dispatch_gelu` is called with bf16 dtype, Then output matches f32→gelu→bf16 cast chain at ε ≤ 1e-3
- Given bf16 input, When `dispatch_softmax` bf16, Then same tolerance
- Given bf16 input, When `dispatch_softcap` bf16, Then same tolerance

**Technical Notes:**
- Add bf16 kernel variants to existing `.metal` files
- Register new pipelines in `KernelRegistry`
- Story points: 3

### Story 2.3: Argsort descending f32

**As a** hf2q integrator,
**I want** GPU-accelerated argsort (descending) for f32 tensors,
**So that** MoE top-K routing can run on GPU without a CPU round-trip.

**Acceptance Criteria:**
- Given a [batch, num_experts] f32 tensor, When `argsort_desc` is called, Then output indices sort values in descending order, matching CPU `sort_unstable_by` reference
- Given tied values, Then the order among ties is deterministic (stable or consistent)

**Technical Notes:**
- File: `src/ops/argsort.rs` + `src/kernels/argsort.metal`
- Reference: candle's `asort_desc_f32` or ggml's `kernel_argsort_f32_i32_desc`
- Story points: 5

### Story 2.4: Gather / index_select

**As a** hf2q integrator,
**I want** GPU gather (index_select) for f32 tensors,
**So that** MoE scale factors can be gathered by expert index on GPU.

**Acceptance Criteria:**
- Given a [num_experts, hidden] f32 tensor and [batch, top_k] i32 index tensor, When `gather` is called along dim=0, Then output matches CPU index_select
- Given out-of-bounds indices, Then the kernel does not panic (returns zero or clips)

**Technical Notes:**
- File: `src/ops/gather.rs` + `src/kernels/gather.metal`
- Story points: 3

### Story 2.5: Contiguous copy

**As a** hf2q integrator,
**I want** a GPU memcpy kernel for making non-contiguous tensors contiguous,
**So that** strided tensors from transpose/permute can be passed to kernels that require contiguous input.

**Acceptance Criteria:**
- Given a [4, 32, 128] tensor with stride [4096, 128, 1] that has been transposed to [32, 4, 128] with stride [128, 4096, 1], When `contiguous_copy` is called, Then output is a new contiguous buffer with stride [512, 128, 1] and identical values

**Technical Notes:**
- File: `src/ops/copy.rs` + `src/kernels/copy.metal`
- Story points: 3

### Story 2.6: Dense F16 GEMM (lm_head)

**As a** hf2q integrator,
**I want** dense F16 × F16 matrix multiplication for the lm_head projection,
**So that** the final vocabulary projection doesn't need a separate code path.

**Acceptance Criteria:**
- Given F16 weight matrix [vocab_size, hidden_dim] and F16 input [1, hidden_dim], When `dense_gemm_f16` is called, Then output matches CPU reference at ε ≤ 1e-3
- Given argmax of output matches argmax of CPU reference, Then the greedy token selection is preserved

**Technical Notes:**
- Option A: Wrap Metal Performance Shaders `MPSMatrixMultiplication`
- Option B: Port candle's `gemm_nt_f16_f16_32_32_16_2_2` kernel
- File: `src/ops/dense_gemm.rs`
- Story points: 5

---

## Epic 3: Borrow Validation (Sprint 2)

**Dependencies:** Epic 1, Epic 2
**FRs covered:** All FR-1 through FR-9 (cross-cutting validation)

### Story 3.1: Attribution and license compliance

**As a** crate publisher,
**I want** all borrowed code to have proper attribution headers and a LICENSE-APACHE-candle file,
**So that** the crate is license-compliant for crates.io publication.

**Acceptance Criteria:**
- Given any file borrowed from candle, When `grep "derived from candle" src/` is run, Then every borrowed file is found
- Given `LICENSE-APACHE-candle` exists in repo root, Then it contains the verbatim Apache-2.0 license text from candle
- Given CHANGELOG entries for all borrows, Then each entry cites the source file:line

**Technical Notes:**
- Attribution header format per ADR-006 §Phase 3
- Story points: 2

### Story 3.2: Cross-stack bitwise parity suite

**As a** hf2q integrator,
**I want** a comprehensive parity test suite comparing mlx-native vs candle on all borrowed ops,
**So that** Phase 5 per-op cutover has a regression gate.

**Acceptance Criteria:**
- Given the full list of borrowed ops from Phase 3 sub-phases (3a-3h), When each is dispatched with identical input through both mlx-native and candle, Then max|Δ| ≤ stated tolerance per op
- Given the test suite runs in < 60 seconds, Then it's practical for CI

**Technical Notes:**
- File: `tests/test_candle_mlx_parity.rs` in hf2q
- This is the expansion of Story 1.2 to cover all ops, not just quantized matmul
- Story points: 5

---

## Epic 4: Graph Scheduler (Sprints 3-4)

**Dependencies:** Epic 1, Epic 2 (all ops must exist and be tested before the scheduler orchestrates them)
**FRs covered:** FR-10
**Rationale:** Phase 0 proved this is the primary speed lever. The gap is not in individual kernels but in how they're dispatched. This epic is the core of the v0.2.0 value proposition.

### Story 4.1: MlxGraph and MlxNode types

**As a** hf2q integrator,
**I want** a graph IR that represents a sequence of ops with buffer dependencies,
**So that** the forward pass can be described declaratively and executed with batched dispatch.

**Acceptance Criteria:**
- Given a sequence of 5 ops (rms_norm → quantized_matmul → rope → sdpa → elementwise_add), When built into an `MlxGraph`, Then `graph.node_count() == 5` and `graph.validate()` returns Ok
- Given a node with an output buffer used as input to a later node, When `graph.validate()` is called, Then the dependency is tracked and the nodes are in valid topological order
- Given a node with a missing input buffer, When `graph.validate()` is called, Then it returns `Err(MissingBuffer)`

**Technical Notes:**
- File: `src/graph.rs`
- `MlxNode` contains: op type enum, input buffer refs, output buffer ref, kernel params
- `MlxGraph` contains: Vec<MlxNode>, buffer dependency map, compiled pipeline references
- Reference: ggml's `ggml_cgraph` structure (but simpler — no backward pass, no dynamic shapes)
- Story points: 5

### Story 4.2: Graph executor with batched dispatch

**As a** hf2q integrator,
**I want** a graph executor that encodes all ops into shared CommandEncoder(s) and commits once,
**So that** the GPU can pipeline kernel execution instead of stalling between dispatches.

**Acceptance Criteria:**
- Given an `MlxGraph` representing one transformer layer (norm → Q/K/V projections → RoPE → SDPA → O projection → residual add → norm → MoE gate → MoE dispatch → residual add), When `graph.execute(&device)` is called, Then all ops are encoded into ≤ 1 command buffer and committed once
- Given the graph executor output, When compared vs sequential dispatch (one commit per op), Then outputs are bitwise identical
- Given a 5-run microbench on the single-layer graph, When compared vs sequential dispatch, Then the graph executor is measurably faster (≥ 10% speedup)

**Technical Notes:**
- The executor walks nodes in order, calls the appropriate `dispatch_*` function for each, using a single `CommandEncoder`
- Activation buffers are allocated from `MlxBufferPool` at graph execution start and released after `commit_and_wait`
- Weight buffers are pre-allocated at model load time and passed in as graph inputs
- Story points: 8

### Story 4.3: Pipeline prewarming

**As a** hf2q integrator,
**I want** all Metal compute pipelines compiled at graph construction time (not lazily on first dispatch),
**So that** the first forward pass doesn't pay a compilation latency penalty.

**Acceptance Criteria:**
- Given an `MlxGraph` is constructed, When all unique kernel names are extracted, Then `KernelRegistry::prewarm(kernel_names)` compiles all pipelines
- Given prewarming completes in < 2 seconds for the full Gemma 4 26B MoE kernel set, Then the latency target is met
- Given the first `graph.execute()` call, When pipeline cache is checked, Then all pipelines are cache hits (0 JIT compiles during execution)

**Technical Notes:**
- File: `src/kernel_registry.rs` (add `prewarm` method)
- Story points: 3

---

## Epic 5: hf2q Integration (Sprint 4-5)

**Dependencies:** Epic 1, 2, 3, 4
**FRs covered:** SC-5, SC-6, SC-7

### Story 5.1: Per-op cutover (candle → mlx-native)

**As a** hf2q developer,
**I want** to swap each op from candle to mlx-native behind per-op CLI flags,
**So that** the migration is incremental, reversible, and validated at every step.

**Acceptance Criteria:**
- Given `--matmul-backend mlx-native` flag, When the forward pass runs, Then quantized matmul dispatches go through mlx-native
- Given sourdough gate is run after each op swap, Then common-byte-prefix ≥ 3094
- Given canonical bench is run after each op swap, Then tok/s ≥ candle baseline (86 tok/s)
- Given all ops are swapped, Then defaults flip to mlx-native and candle dependency is dropped

**Technical Notes:**
- Per ADR-006 §Phase 5 method
- Cargo feature: `mlx-native-backend = ["dep:mlx-native"]`
- Op cutover order: matmul → SDPA → RoPE → RmsNorm → lm_head → KV cache → elementwise → MoE
- Story points: 13

### Story 5.2: Graph scheduler integration

**As a** hf2q developer,
**I want** the full `Gemma4Model::forward` to use mlx-native's graph executor,
**So that** decode speed reaches ≥ 102 tok/s via GPU pipelining.

**Acceptance Criteria:**
- Given the graph-scheduled forward pass, When canonical bench runs 5 times, Then median tok/s ≥ 102
- Given sourdough gate, Then common-byte-prefix ≥ 3094
- Given byte-identical 16-token greedy gen vs llama.cpp at T=0, Then match preserved

**Technical Notes:**
- This story replaces the per-op sequential dispatch with a single graph execution
- If tok/s < 102, diagnose: is it the graph executor overhead or a remaining kernel issue? Loop back to Epic 4 or Epic 2.
- Story points: 8

---

## Epic 6: Documentation and Packaging (Sprint 5)

**Dependencies:** Epic 1, 2 (ops must exist to document)
**FRs covered:** NFR-4, NFR-5

### Story 6.1: README expansion

**As a** crate consumer,
**I want** a comprehensive README (8KB+) with feature list, quick start, per-op sections, and requirements,
**So that** I can evaluate and use mlx-native without reading source code.

**Acceptance Criteria:**
- Given the README, When read by a Rust developer unfamiliar with mlx-native, Then they can: understand what the crate does, add it to Cargo.toml, dispatch a single quantized matmul, and understand the graph scheduler concept
- Given the README lists all ops with supported dtypes, Then no op is undocumented

**Technical Notes:**
- Template: coreml-native README (feature bullets, Quick Start, per-API sections, requirements, comparison table)
- Story points: 3

### Story 6.2: CHANGELOG and Cargo.toml metadata

**As a** crate publisher,
**I want** a CHANGELOG in Keep-a-Changelog 1.1.0 format and complete Cargo.toml publish metadata,
**So that** `cargo publish --dry-run` succeeds.

**Acceptance Criteria:**
- Given `cargo publish --dry-run` is run, Then it exits 0 with no errors
- Given CHANGELOG.md exists, Then it has [Unreleased] and [0.1.0] sections with compare links
- Given Cargo.toml, Then it has: authors, repository, documentation, keywords (5), categories, readme, homepage, exclude list, docs.rs metadata

**Technical Notes:**
- Story points: 2

### Story 6.3: Examples

**As a** developer,
**I want** runnable examples showing mlx-native's API,
**So that** I can learn by running code, not just reading docs.

**Acceptance Criteria:**
- Given `cargo run --example forward_pass`, Then it loads test weights, builds a single-layer graph, executes it, and prints output shape + first 5 values
- Given `cargo run --example load_and_bench`, Then it runs a 100-iteration bench and prints tok/s

**Technical Notes:**
- Files: `examples/forward_pass.rs`, `examples/load_and_bench.rs`
- Need small test fixture weights (generated, not checked in as binary — use a build script or test helper)
- Story points: 3

---

## Epic 7: CI and Release (Sprint 5)

**Dependencies:** Epic 6
**FRs covered:** NFR-6

### Story 7.1: GitHub Actions CI

**As a** maintainer,
**I want** CI that runs on every push/PR,
**So that** regressions are caught before merge.

**Acceptance Criteria:**
- Given a push to any branch, When CI runs, Then: `cargo build` succeeds, `cargo test` passes, `cargo clippy -- -D warnings` is clean, `cargo doc --no-deps` succeeds (all on macOS-14)
- Given a push, When CI runs the ubuntu job, Then `cargo check` succeeds (cross-platform compile guarantee)

**Technical Notes:**
- File: `.github/workflows/ci.yml`
- Template: coreml-native's CI (two jobs: test-macos + check-linux)
- Story points: 2

### Story 7.2: Integration test fixtures

**As a** tester,
**I want** small fixture weights for pipeline-level integration tests,
**So that** CI can validate the full dispatch path, not just individual kernels.

**Acceptance Criteria:**
- Given `tests/fixtures/` contains small quantized weight files, When `cargo test --test integration` runs, Then a mini forward pass (1 layer, small dims) executes and produces expected output shape
- Given fixture generation is deterministic, Then fixtures can be regenerated from a script if needed

**Technical Notes:**
- Fixtures should be tiny (< 1 MB) — small dims, 1 layer, 1 expert
- Story points: 3

### Story 7.3: Version bump and release prep

**As a** maintainer,
**I want** mlx-native at v0.2.0 with all stories complete,
**So that** hf2q can depend on a versioned, tested, documented crate.

**Acceptance Criteria:**
- Given version is bumped to 0.2.0 in Cargo.toml, Then CHANGELOG [0.2.0] section lists all changes
- Given `cargo publish --dry-run` passes, Then the crate is ready for publication
- Given all CI checks pass on the release branch, Then the release is tagged

**Technical Notes:**
- Story points: 1

---

## Sprint Assignment Summary

| Sprint | Epics | Stories | Total SP | Calendar (est.) |
|--------|-------|---------|----------|-----------------|
| Sprint 1 | Epic 1 + Epic 2 (2.1-2.2) | 1.1, 1.2, 1.3, 2.1, 2.2 | 24 | ~1 week |
| Sprint 2 | Epic 2 (2.3-2.6) + Epic 3 | 2.3, 2.4, 2.5, 2.6, 3.1, 3.2 | 23 | ~1 week |
| Sprint 3 | Epic 4 (4.1-4.2) | 4.1, 4.2 | 13 | ~1 week |
| Sprint 4 | Epic 4 (4.3) + Epic 5 (5.1) | 4.3, 5.1 | 16 | ~1 week |
| Sprint 5 | Epic 5 (5.2) + Epic 6 + Epic 7 | 5.2, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3 | 22 | ~1 week |
| **Total** | **7 epics** | **21 stories** | **98 SP** | **~5 weeks** |
