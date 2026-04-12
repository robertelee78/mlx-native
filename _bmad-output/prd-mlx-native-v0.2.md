# Product Requirements Document: mlx-native v0.2.0

**Version:** 0.2.0
**Date:** 2026-04-12
**Author:** Claude (synthesized from cfa swarm-1775959670785-u058z2)
**Template:** coreml-native PRD (v0.2.0, 2026-03-23)
**Governing ADR:** ADR-006 (mlx-native as hf2q's GPU Compute Backend)
**Phase 0 Verdict:** Framework-overhead-dominated (see `docs/spike-1bNEW30-per-kernel-attribution.md`)

---

## 1. Executive Summary

**Vision:** mlx-native is a pure-Rust Metal GPU compute library for quantized inference on Apple Silicon. It provides the low-level kernel dispatch, buffer management, and (new in v0.2.0) graph-scheduled batched execution that hf2q needs to match llama.cpp's decode speed.

**Differentiator:** Pure Rust + objc2-metal. No C++ dependencies. No framework overhead. Direct Metal command buffer control at the level llama.cpp's ggml-metal operates, but in safe Rust with a typed API.

**Target Users:**
- hf2q (primary consumer — the inference forward pass migrates from candle to mlx-native per ADR-006)
- Future Rust inference projects that need Apple Silicon GPU compute without candle's general-purpose overhead

**Key Constraints:**
- Phase 0 confirmed: the speed gap is framework dispatch overhead (GPU pipelining), NOT kernel implementations
- Kernels are already fast enough; the primary v0.2.0 work is the graph scheduler + batched encoder pattern
- Coherence is non-negotiable: every op must match candle's output at bitwise or epsilon ≤ 1e-5 tolerance
- Engineering mantra applies: no shortcuts, no stubs, measure 3x cut once, Chesterton's fence

---

## 2. Success Criteria

| ID | Criterion | Measurement | Target |
|----|-----------|-------------|--------|
| SC-1 | Op coverage for hf2q Gemma 4 26B MoE forward pass | Every op in the Phase 0 kernel inventory has an mlx-native implementation | 100% coverage (0 missing ops) |
| SC-2 | Test coverage on load-bearing ops | Every op with >10 calls/token in the Phase 0 data has a unit test with stated tolerance | 100% of load-bearing ops tested |
| SC-3 | Bitwise correctness on quantized matmul | `quantized_matmul` output matches MLX affine dequant reference at hf2q's production shapes | max\|Δ\| ≤ 1e-4 on all quant types |
| SC-4 | Graph scheduler exists and batches dispatches | Single `commit_and_wait` per forward pass instead of per-dispatch | ≤ 3 command buffer commits per forward pass |
| SC-5 | Decode speed parity with candle baseline | hf2q + mlx-native canonical bench ≥ candle baseline | ≥ 86 tok/s (Phase 5 gate) |
| SC-6 | Decode speed target (Walk End gate) | hf2q + mlx-native canonical bench ≥ llama.cpp peer | ≥ 102 tok/s (Phase 6 gate) |
| SC-7 | Coherence preservation | Sourdough gate passes at every commit during integration | Common-byte-prefix ≥ 3094 |
| SC-8 | Crate publishable | `cargo publish --dry-run` succeeds with all metadata | Passes with 0 errors |

---

## 3. Product Scope

### Phase 1: Foundation (Sprints 1-2)
- Close test gaps on existing ops (quantized_matmul, fused kernels, KV cache, argmax, softmax_sample)
- Add missing ops (grouped MoE `kernel_mul_mv_id`, argsort, gather/index_select, dense F16 GEMM, bf16 variants of gelu/softmax/softcap, contiguous copy)
- Borrow ops from candle with attribution per ADR-006 Phase 3 discipline
- README expansion (1 line → 8KB+), CHANGELOG creation, Cargo.toml publish metadata

### Phase 2: Graph Scheduler (Sprints 3-4)
- Graph IR: `MlxGraph` + `MlxNode` types for representing a forward pass as a DAG
- Graph executor: walks nodes, encodes all ops into shared `CommandEncoder`(s), single `commit_and_wait`
- Pipeline prewarming: compile all Metal pipelines at model load time (not lazy on first dispatch)
- Buffer arena: per-graph activation allocator using `MlxBufferPool`, reset after each forward pass
- Multiple command buffers in flight (up to 3, matching ggml-metal)

### Phase 3: Release (Sprint 5)
- CI: GitHub Actions (macOS-14 test + clippy + doc; ubuntu compile check)
- Examples: `forward_pass.rs` (single-layer batched dispatch), `load_and_bench.rs`
- Integration tests with fixture weights
- `cargo publish --dry-run` passes
- Version bump to 0.2.0

### Out of Scope (v0.2.0)
- Training ops (backward pass, gradient computation)
- Non-Apple platforms (CUDA, Vulkan)
- Model loading / GGUF parsing (that stays in hf2q)
- HTTP server / inference API (that stays in hf2q)
- CoreML / ANE integration (that stays in coreml-native)

---

## 4. User Journeys

### UJ-1: hf2q migrates a single op from candle to mlx-native

**Actor:** hf2q integration code
**Goal:** Replace one `QMatMul::forward` call with `mlx_native::quantized_matmul()`

1. Load quantized weight buffers into `MlxBuffer` at model init
2. At forward time, call `quantized_matmul(&encoder, &weight_buf, &scale_buf, &bias_buf, &input_buf, &output_buf, params)`
3. Encoder encodes the kernel without committing
4. After all ops in the layer are encoded, call `encoder.commit_and_wait()`
5. Read output from `output_buf`

**Success:** Output matches candle's `QMatMul::forward` at ε ≤ 1e-4. Sourdough gate passes.

### UJ-2: hf2q runs a full forward pass through the graph scheduler

**Actor:** hf2q `Gemma4Model::forward`
**Goal:** Execute entire forward pass with single command buffer submission

1. Build `MlxGraph` from the layer sequence (attention → MoE → residual, 26 layers)
2. Call `graph.execute(&encoder, &buffers)`
3. Graph executor encodes all ops into the encoder in topological order
4. Single `encoder.commit_and_wait()` at the end
5. Read logits from output buffer

**Success:** Decode speed ≥ 102 tok/s. Coherence preserved. ≤ 3 command buffer commits per forward pass.

### UJ-3: Developer adds a new kernel to mlx-native

**Actor:** Contributor
**Goal:** Add a new Metal kernel (e.g., `rotary_embedding_bf16`)

1. Write MSL kernel in `src/kernels/<name>.metal`
2. Register in `KernelRegistry` with pipeline name
3. Write Rust dispatch function in `src/ops/<name>.rs`
4. Write unit test comparing GPU output vs CPU reference at stated tolerance
5. Add entry to CHANGELOG under [Unreleased]

**Success:** `cargo test` passes. `cargo clippy -- -D warnings` clean.

---

## 5. Functional Requirements

### FR-1: Quantized Matrix Operations
- FR-1.1: `quantized_matmul` supports Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0 for both mat-vec (M=1) and mat-mat (M>1)
- FR-1.2: Output matches MLX affine dequant formula at ε ≤ 1e-4
- FR-1.3: Unit tests cover all 6 quant types at hf2q's production shapes (from Phase 0 data)

### FR-2: Grouped MoE Dispatch (NEW — #1 missing kernel)
- FR-2.1: `kernel_mul_mv_id` family for Q4_0, Q6_K, Q8_0 — dispatches per-token to per-expert weight matrices based on routing indices
- FR-2.2: Ported from candle's vendored `quantized.metal:7544-7618` with attribution header
- FR-2.3: Unit test: 4-token × 8-expert × 2-top-k MoE forward; assert bitwise match vs candle's kernel

### FR-3: Attention Operations
- FR-3.1: SDPA full causal (f32/bf16) for global attention layers (head_dim=256, 512)
- FR-3.2: SDPA sliding window (f32/bf16) for sliding attention layers
- FR-3.3: Both support GQA (num_kv_heads < num_heads)

### FR-4: Normalization and Activation
- FR-4.1: RMS norm (f32/f16/bf16) with and without scale
- FR-4.2: GELU pytorch-tanh (f32/f16/bf16 — bf16 variant is NEW)
- FR-4.3: Softmax (f32/f16/bf16 — bf16 variant is NEW)
- FR-4.4: Softcap (f32/f16/bf16 — bf16 variant is NEW)
- FR-4.5: Fused variants: residual+rms_norm, head_norm+rope, norm+add

### FR-5: RoPE
- FR-5.1: Standard and neox variants (f32/f16/bf16)
- FR-5.2: Partial rotary support (for global attention with rotary_dim < head_dim)
- FR-5.3: freq_factors support (for models with non-standard RoPE frequencies)

### FR-6: Elementwise and Utility Operations
- FR-6.1: Add, mul, scalar_mul (f32/f16/bf16)
- FR-6.2: Cast between f32, f16, bf16
- FR-6.3: Transpose 2D, permute 021
- FR-6.4: Argsort descending f32 (NEW — needed for MoE top-K)
- FR-6.5: Gather / index_select (NEW — needed for MoE scale gather)
- FR-6.6: Contiguous copy (NEW — `.contiguous()` analogue)

### FR-7: Embedding and Sampling
- FR-7.1: Quantized embedding gather (4-bit, 6-bit)
- FR-7.2: Argmax f32
- FR-7.3: Softmax + categorical sample f32

### FR-8: KV Cache Management
- FR-8.1: In-place KV cache copy (bf16) for sliding window and global ring buffers
- FR-8.2: Correct wrap-around behavior for sliding window truncation

### FR-9: Dense Matrix Operations (NEW)
- FR-9.1: Dense F16 × F16 GEMM for lm_head projection
- FR-9.2: Expose via Metal Performance Shaders `MPSMatrixMultiplication` or write custom kernel

### FR-10: Graph Scheduler (NEW — primary Phase 4 lever per Phase 0 diagnosis)
- FR-10.1: `MlxGraph` type representing a sequence of ops with buffer dependencies
- FR-10.2: `MlxNode` type representing a single op dispatch with input/output buffer references
- FR-10.3: Graph executor encodes all nodes into shared `CommandEncoder`(s) in dependency order
- FR-10.4: Single `commit_and_wait()` per forward pass (or per command buffer if using multiple in-flight)
- FR-10.5: Pipeline prewarming: all kernels compiled at graph construction time, not on first dispatch
- FR-10.6: Per-graph buffer arena: activation tensors allocated from `MlxBufferPool`, reset after execution

---

## 6. Non-Functional Requirements

### NFR-1: Performance
- Decode throughput ≥ 102 tok/s on M5 Max Gemma 4 26B MoE Q4_K_M (Walk End gate)
- ≤ 3 command buffer commits per forward pass (graph scheduler target)
- Pipeline compile latency < 2s total at model load time (prewarming)
- Zero unnecessary GPU↔CPU synchronization during forward pass

### NFR-2: Correctness
- Every op matches candle's output at bitwise (max|Δ| = 0) or ε ≤ 1e-5 tolerance
- Sourdough gate (common-byte-prefix ≥ 3094) passes at every commit during Phase 5 integration
- Byte-identical 16-token greedy gen vs llama.cpp at T=0 preserved

### NFR-3: Memory Safety
- No memory leaks over 1000 forward passes (autorelease pools drain correctly)
- `MlxBufferPool` tracks all allocations; pool reset at forward-pass exit
- Weight buffers are immutable after load; no aliased mutable access

### NFR-4: API Ergonomics
- No `unsafe` in user-facing API (unsafe confined to internal FFI/objc2 calls)
- Complete rustdoc on every public type and function
- Meaningful compile errors on non-Apple targets (cfg-gated stubs returning `Err(UnsupportedPlatform)`)

### NFR-5: Packaging
- `cargo publish --dry-run` succeeds
- License: MIT OR Apache-2.0
- Minimal dependency tree (objc2, objc2-metal, block2; no framework-level crates)
- `exclude` list hides `_bmad-output/`, `.github/`, `docs/`, `scripts/` from published crate

### NFR-6: Testing
- Unit tests for every op with stated tolerance (ε in test assertion message)
- Integration tests with fixture weights for pipeline-level validation
- CI on GitHub Actions: macOS-14 (build + test + clippy + doc), ubuntu (compile check)
- Criterion benchmarks for quantized_matmul, SDPA, graph executor

---

## 7. Implementation Phases

### Phase 1a: Test Gap Closure + Missing Ops (Sprint 1)

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| quantized_matmul unit tests (all 6 quant types) | FR-1 | P0 |
| kernel_mul_mv_id port from candle (grouped MoE) | FR-2 | P0 |
| bf16 variants: gelu, softmax, softcap | FR-4.2-4.4 | P1 |
| argsort, gather/index_select, contiguous copy | FR-6.4-6.6 | P1 |
| dense F16 GEMM (lm_head) | FR-9 | P1 |
| fused kernel tests (residual_norm, head_norm_rope, norm_add) | FR-4.5 | P1 |
| argmax, softmax_sample, kv_cache_copy, permute_021 tests | FR-7, FR-8 | P2 |

### Phase 1b: Borrow + Validate (Sprint 2)

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| Bitwise validation: all borrowed ops vs candle at hf2q shapes | FR-1 through FR-9 | P0 |
| Attribution headers on all borrowed code | ADR-006 §Phase 3 | P0 |
| LICENSE-APACHE-candle file | ADR-006 §Phase 3 | P0 |

### Phase 2: Graph Scheduler (Sprints 3-4)

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| MlxGraph + MlxNode types | FR-10.1-10.2 | P0 |
| Graph executor (batched encode → single commit) | FR-10.3-10.4 | P0 |
| Pipeline prewarming at graph construction | FR-10.5 | P1 |
| Per-graph buffer arena | FR-10.6 | P1 |
| Multiple command buffers in flight | FR-10.4 | P2 |

### Phase 3: Release (Sprint 5)

| Component | FRs Covered | Priority |
|-----------|-------------|----------|
| README (8KB+), CHANGELOG, Cargo.toml metadata | NFR-5 | P0 |
| GitHub Actions CI | NFR-6 | P0 |
| Examples (forward_pass.rs, load_and_bench.rs) | NFR-4 | P1 |
| Integration test fixtures | NFR-6 | P1 |
| Cross-platform stubs (cfg-gated) | NFR-4 | P2 |

---

## 8. Architecture Overview

### Module Tree (v0.2.0 target)

```
mlx-native/
├── src/
│   ├── lib.rs              # Public API, re-exports
│   ├── device.rs           # MlxDevice (Metal device wrapper)
│   ├── buffer.rs           # MlxBuffer (typed Metal buffer)
│   ├── buffer_pool.rs      # MlxBufferPool (power-of-2 arena)
│   ├── encoder.rs          # CommandEncoder (batched dispatch)
│   ├── kernel_registry.rs  # KernelRegistry (pipeline cache)
│   ├── weight.rs           # Quantized weight loading
│   ├── graph.rs            # NEW: MlxGraph, MlxNode, GraphExecutor
│   ├── ops/
│   │   ├── quantized_matmul.rs    # Q4/Q6/Q8 mat-vec + mat-mat
│   │   ├── quantized_matmul_id.rs # NEW: grouped MoE dispatch
│   │   ├── rms_norm.rs
│   │   ├── rope.rs
│   │   ├── sdpa.rs
│   │   ├── sdpa_sliding.rs
│   │   ├── fused_residual_norm.rs
│   │   ├── fused_head_norm_rope.rs
│   │   ├── fused_norm_add.rs
│   │   ├── moe_dispatch.rs
│   │   ├── moe_gate.rs
│   │   ├── embedding.rs
│   │   ├── elementwise.rs
│   │   ├── gelu.rs
│   │   ├── softmax.rs
│   │   ├── softcap.rs
│   │   ├── argmax.rs
│   │   ├── softmax_sample.rs
│   │   ├── kv_cache_copy.rs
│   │   ├── transpose.rs
│   │   ├── argsort.rs        # NEW
│   │   ├── gather.rs         # NEW
│   │   ├── copy.rs           # NEW (contiguous)
│   │   └── dense_gemm.rs     # NEW (F16 lm_head)
│   └── kernels/              # Metal Shading Language source
│       ├── quantized.metal
│       ├── quantized_id.metal  # NEW
│       ├── rms_norm.metal
│       ├── rope.metal
│       ├── sdpa.metal
│       ├── ...
│       ├── argsort.metal       # NEW
│       ├── gather.metal        # NEW
│       └── copy.metal          # NEW
├── tests/
│   ├── test_quantized_matmul.rs  # NEW (P0)
│   ├── test_quantized_matmul_id.rs # NEW
│   ├── test_sdpa.rs
│   ├── test_rope.rs
│   ├── test_rms_norm.rs
│   ├── test_fused_ops.rs         # NEW
│   ├── test_kv_cache.rs          # NEW
│   ├── test_graph_executor.rs    # NEW
│   └── ...
├── benches/
│   ├── bench_quantized_matmul.rs
│   ├── bench_sdpa.rs
│   ├── bench_moe_dispatch.rs
│   └── bench_graph_executor.rs   # NEW
├── examples/
│   ├── forward_pass.rs           # NEW
│   └── load_and_bench.rs         # NEW
├── _bmad-output/
│   ├── prd-mlx-native-v0.2.md   # This file
│   └── epics-and-stories.md
├── CHANGELOG.md                   # NEW
├── CONTRIBUTING.md                # NEW (optional)
└── Cargo.toml                     # Updated with publish metadata
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Graph scheduler over individual op optimization | Phase 0 proved the gap is framework dispatch overhead, not kernel speed |
| Single CommandEncoder per forward pass | Enables GPU pipelining; matches ggml-metal's dispatch pattern |
| Power-of-2 buffer pool for activations | Reduces allocation overhead; reset-per-forward-pass lifecycle |
| Borrow from candle with attribution, not rewrite | Candle's kernels are already fast (7/9 matched families faster than ggml per-call); borrowing preserves correctness |
| Port ggml-metal framework patterns, not kernel implementations | Phase 0 data: kernels are fine, framework is the gap |
| No Tensor abstraction | mlx-native is a compute library, not an ML framework; hf2q owns the Tensor concept |

---

## 9. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Graph scheduler adds latency for graph construction | Medium | Low | Graph is built once at model init, reused every forward pass; construction cost amortized |
| Borrowed candle kernels don't match mlx-native's CommandEncoder ABI | High | Medium | Phase 3 sub-phase validates each borrow's ABI adaptation; bitwise correctness test catches mismatches |
| MlxBufferPool fragments under MoE's variable expert count | Medium | Medium | Phase 3f tests sliding-window KV growth + MoE allocation patterns; add defrag if needed |
| Pipeline prewarming increases model load time | Low | High | Acceptable trade-off (< 2s target); matches ggml-metal behavior |
| quantized_matmul has latent bugs (1403 LOC, zero tests) | High | Medium | Sprint 1 P0: test all 6 quant types at production shapes before any other work |
| Multiple command buffers in flight cause race conditions | High | Low | Start with 1 command buffer; add concurrency only after single-buffer path is validated |

---

## 10. Dependencies and Prerequisites

- ADR-006 Accepted (done as of 2026-04-12)
- Phase 0 spike data (`docs/spike-1bNEW30-per-kernel-attribution.md`) for kernel inventory and call counts
- candle vendor patches (`vendor/candle-metal-kernels/`, `vendor/candle-nn/`) as borrow source for Phase 3
- llama.cpp ggml-metal source (`/opt/llama.cpp/ggml/src/ggml-metal/`) as reference for Phase 4 graph scheduler
- coreml-native (`/opt/coreml-native/`) as maturation template (README, CI, Cargo.toml metadata patterns)
- Rust stable (edition 2021, MSRV 1.75.0)
- macOS with Apple Silicon for testing (M5 Max is the benchmark target)

---

## 11. Glossary

| Term | Definition |
|------|------------|
| Walk | Phase 1b of hf2q development: port llama.cpp to Rust, matching both coherence AND speed |
| Sourdough gate | hf2q coherence test: compare greedy-decoded output vs llama.cpp; pass at common-byte-prefix ≥ 3094 |
| End gate | Walk completion criterion: ≥ 102 tok/s decode on M5 Max Gemma 4 26B MoE Q4_K_M |
| Graph scheduler | Component that batches multiple Metal kernel dispatches into shared command buffers for GPU pipelining |
| GPU pipelining | Metal GPU's ability to overlap kernel execution when multiple dispatches share a command buffer |
| Affine dequant | MLX quantization formula: `output = (weight_uint * scale + bias) @ input` |
| MoE | Mixture of Experts: FFN layer where each token is routed to a subset of expert weight matrices |
| kernel_mul_mv_id | ggml/candle kernel family for grouped MoE dispatch: per-token routing to per-expert quantized mat-vec |
| CommandEncoder | mlx-native's Metal command buffer wrapper; encodes GPU dispatches |
| MlxBufferPool | Power-of-2 bucketed arena allocator for Metal buffers |
