# flash_attn_train Scoping Report

**Date:** 2026-05-08  
**Audience:** Kernel engineer implementing the backward kernel  
**Scope:** mlx-native `flash_attn_train` forward + backward for differentiable GQA attention on Apple Metal

---

## A. Current Kernel Inventory

**Forward kernels — `/opt/mlx-native/src/ops/`:**

- **`flash_attn_prefill.rs`** (1609 LOC) + **`flash_attn_prefill.metal`** (1713 LOC): The primary prefill kernel. Dispatches 12 entry points across three head-dim families (D=64, D=256, D=512), two dtypes (bf16, f16), and two mask kinds (additive, bool). Tile geometry: BQ=32, BK=16, WM=4, WN=1 for D=64/D=256 (128 threads/threadgroup, 4 simdgroups); BQ=8, BK=8, WM=1, WN=1 for D=512 (32 threads/threadgroup). Buffer layout: Q `[B, H, qL, D]`, K `[B, H_kv, kL, D]`, V `[B, H_kv, kL, D]`, O `[B, H, qL, D]`, params struct (160 bytes), optional mask. Algorithm: online softmax (FA-1 style) using base-2 arithmetic; Q is pre-scaled by `scale * log2(e)`; per-row running max `sum_score` initialized to `-FLT_MAX/2` (llama.cpp finite-sentinel); single output-side guard `sum_score == 0 ? 0 : 1/sum_score`. **No logsumexp output buffer; no backward path.** f32 is not instantiated for any D (threadgroup SMEM overflow — 32 KB limit hit at D=256 f32 BQ=32; see `flash_attn_prefill.rs:42–51`).

- **`flash_attn_prefill_d512.rs`** + **`.metal`** (973 LOC): D=512 variant with reduced tile BQ=8, BK=8.

- **`flash_attn_prefill_mask.rs`** + **`flash_attn_prefill_blk.rs`** + shaders: SWA (sliding-window) mask builder and tile-skip block-classification helpers, both layered on top of the main prefill kernel.

- **`flash_attn_vec.rs`** + **`flash_attn_vec.metal`** (407 LOC): Decode-time kernel (seq_len_q = 1), one-thread-per-head-element layout.

- **`flash_attn_vec_tq.rs`** / **`flash_attn_vec_tq_hb.rs`** + shaders: TQ-dequantized decode variants.

**Backward shaders — none.** No existing Metal shaders expose any attention gradient path. The existing backward shaders in `/opt/mlx-native/src/shaders/` are `rms_norm_backward.metal` (181 LOC), `silu_backward.metal` (56 LOC), and `softmax_backward.metal` (64 LOC). The `softmax_backward.metal` is the closest structural model: one threadgroup per row, uses the forward softmax output `y` + upstream `dy`, computes `dx = y * (dy - dot(dy, y))` in-warp.

**Peer references — no metal attention backward exists.** Search across `/opt/mlx-lm/`, `/opt/candle/`, `/opt/omlx/` found zero `flash_attn_backward`, `attn_backward`, or `sdpa_backward` Metal/CUDA sources. mlx-lm's `qwen3_moe.py` calls MLX's `scaled_dot_product_attention` (a Python-level op, not a custom Metal kernel); MLX's own autograd handles backward through a separate internal VJP — that path is not available in mlx-native's AOT Metal kernel infrastructure.

---

## B. Algorithmic Options for Backward

### Option 1: Recompute-during-backward (FlashAttention-2 textbook)

Forward saves only Q, K, V, and the per-row logsumexp `L[b, h, i] = m_i + log(sum_exp_i)` (one float per query position per head). Backward recomputes the attention probability matrix `P` from Q, K, L tile-by-tile during the backward pass.

**Algorithm (FA-2, Algorithm 4 in Dao 2023):**  
Given `dO`, Q, K, V, O, L:  
1. `D_i = rowsum(O * dO)` — one float per query row  
2. Per KV tile: recompute `S_ij = Q_i K_j^T * scale`, `P_ij = exp(S_ij - L_i)`  
3. `dV_j += P_ij^T dO_i`  
4. `dP_ij = dO_i V_j^T`  
5. `dS_ij = P_ij * (dP_ij - D_i)` (the `D_i` subtraction is the key identity)  
6. `dQ_i += dS_ij K_j * scale`; `dK_j += dS_ij^T Q_i * scale`

**Memory cost:** `B * H * qL * 1 float` for L (logsumexp) + `B * H * qL * 1 float` for D (negligible vs activations). Forward O must also be retained for D computation (it would be held by the tape already). Total added storage: 2 floats per (batch, head, query-position) — for Qwen3.6-35B-A3B at B=1, H=16, qL=2048: 2 * 16 * 2048 * 4 = 256 KB, negligible.

**Compute cost:** forward QK^T recomputed once per backward pass, per KV tile. At seq_len=2048, D=256, H=16: roughly doubles the FLOPs of one forward pass.

**LOC estimate:** 
- New Metal shader `flash_attn_train_bwd.metal`: ~600–900 LOC (two nested tile loops with simdgroup MMA, one pass for dQ, one for dK/dV; or a single two-sweep structure).
- Forward shader modification (`flash_attn_train_fwd.metal`): fork of `flash_attn_prefill.metal` adding a `logsumexp_out [B, H, qL]` output buffer (~50 LOC delta from existing forward).
- Rust dispatch (`flash_attn_train.rs`): ~350 LOC.

**Reference:** Dao et al. 2023 FlashAttention-2 (arXiv:2307.08691) Algorithm 4. Apple simdgroup MMA backward port: no direct prior art in the visible codebase; must be a fresh port. The `rms_norm_backward.metal` two-kernel chain (compute `rms_inv` → `backward_dx` → `backward_dw`) is the structural model for how to split a multi-kernel backward.

**This is the preferred option.** Small memory footprint matches the tape's per-op-minimal storage convention; logsumexp is the only extra output from forward.

---

### Option 2: Materialize full attention matrix during forward

Forward stores the full `B * H * qL * kL` softmax probability matrix `P` in device memory. Backward uses it directly: `dV = P^T dO`, `dP = dO V^T`, `dS = P * (dP - rowsum(P * dP))`, `dQ = dS K * scale`, `dK = dS^T Q * scale`.

**Memory cost:** At Qwen3.6-35B-A3B (H=16, qL=2048, kL=2048, f32): `16 * 2048 * 2048 * 4 = 268 MB` per layer per batch. For a full forward pass over 40 layers: ~10.7 GB in addition to model weights. This exhausts the 96 GB M5 Max unified memory at realistic batch sizes during DWQ calibration sweeps.

**LOC estimate:** Simpler backward shader (~300–400 LOC); forward shader grows by ~100 LOC to emit P. Rust dispatch: ~300 LOC.

**Trade-off:** Simpler backward math, no QK^T recompute, but memory-prohibitive at production sequence lengths. Viable only as a debugging scaffold for small-seq correctness validation (qL ≤ 128), not for production DWQ calibration. Use as a CPU oracle in the test plan (section D).

---

## C. Cross-Attention Proxy Interlock

The current `decoder_layer_on_tape` (defined at `qwen35_moe.rs:368`) and `qwen35_moe_forward_on_tape` (at `qwen35_moe.rs:454`) use a **single matmul proxy** for the entire attention block:

```
r = matmul(y1, w_attn)  // [n_tokens, hidden]
```

where `w_attn` is a `[hidden, hidden]` leaf. This is a deliberate simplification documented at `qwen35_moe.rs:353`: "the attn block is a black box from gradient's perspective."

The swap-in point is `qwen35_moe.rs:395`: `let r = matmul(&y1, weights.w_attn)?;`. Replacing this with a real attention call would require:

1. **A new `GpuTape` OpKind variant** (`FlashAttnTrain`) with fields for `q_idx`, `k_idx`, `v_idx` (three parent indices), carrying the retained `Q`, `K`, `V` buffers and the logsumexp buffer `L` as variant data. The `backward` function at `autograd_gpu_tape.rs:604` would need a new match arm dispatching to three `accumulate` calls (parent_grads[0] → q_idx, [1] → k_idx, [2] → v_idx). No structural change to the tape machinery is needed; all existing two-parent ops prove the pattern.

2. **Upstream plumbing:** The proxy currently takes pre-normed `y1 [n_tokens, hidden]` and produces `r [n_tokens, hidden]`. Real attention needs Q, K, V projections (three matmuls) ahead of it, plus per-head reshape/split into `[B, H, qL, D]` layout. The current tape has no batch-aware reshape that maps `[n_tokens, H*D]` → `[1, H, n_tokens, D]`; the existing `View` op (`autograd_gpu_tape.rs:179`) handles flat reshapes but the attention kernel needs contiguous `[B, H, S, D]` — this is a clean swap only if RoPE is applied *outside* the kernel (see Open Questions).

3. **`DecoderLayerWeights` struct change:** `w_attn: &'a GpuTensor` (one weight) becomes `{w_q, w_k, w_v, w_o, q_norm_w, k_norm_w}: &'a GpuTensor` (six weights), each with explicit shapes. The existing `qwen35_attention_block.rs` `AttentionBlockWeights` struct (`line 99ff`) already has this decomposition for the single-head fixture — it can be extended to multi-head GQA by adding `n_kv_heads` routing.

4. **No refactor of `GpuTape` itself** is needed; only a new `OpKind` arm and a new `flash_attn_train_forward` function that pushes the node. The backward pattern is identical to `OpKind::Matmul` (which itself retains lhs/rhs buffers via node indices `lhs_idx`, `rhs_idx`).

---

## D. Test Plan

**CPU oracle:** `sdpa_reference_f32` (already defined in `tests/test_flash_attn_prefill.rs`) computes the forward pass. Extend it to also return per-row logsumexp `L[i]` and `D[i] = rowsum(O * dO)` — 30 LOC. The backward then uses the FA-2 Algorithm 4 formulas in f32 scalar loops on CPU: `dQ`, `dK`, `dV` all produced in ~100 LOC. This Oracle is self-contained and verifiable against finite-difference.

**Finite-difference falsifier:** Follow the pattern of `rms_norm_backward.rs:530` (`rms_norm_backward_finite_diff_falsifier`). For a small shape (B=1, H=2, qL=32, kL=32, D=64), perturb each element of Q/K/V by ±ε=1e-3 f32, measure `L2(O)` shift, compare to analytic gradient. Tolerance: `atol=5e-3` (matches f32 precision + MMA accumulation). This shape fits in f32 on CPU without the threadgroup-memory constraint.

**GPU parity tests (forward):** After Phase 1, compare `dispatch_flash_attn_train_fwd_f32` output against `sdpa_reference_f32`. Shapes: (B=1, H=1, qL=32, kL=32, D=64), (B=1, H=4, qL=128, kL=128, D=64). Tolerance: `atol=1e-4` (f32 GPU vs f32 CPU). Verify logsumexp buffer `L` matches CPU reference per-row max + log(sum) to `atol=1e-5`.

**GPU parity tests (backward):** After Phase 2, run FA-2 backward on GPU for the same small shapes; compare `dQ`, `dK`, `dV` to CPU oracle. Tolerance: `atol=5e-3`. Also run finite-difference on GPU: this is the load-bearing falsifier.

**Causal mask test:** Run forward + backward with `do_causal=true`, qL=kL=64. Verify dK[i, j] = 0 for j > i (masked positions should have zero contribution to dK). This catches off-by-one errors in the tile-skip causal logic during backward.

**GQA test:** n_heads=8, n_kv_heads=2 (gqa_factor=4). Verify that the KV gradient `dK` and `dV` correctly accumulate contributions from all 4 Q-heads that share each KV head. Shape: B=1, H=8, H_kv=2, qL=64, kL=64, D=64.

**Shape convention:** Pin all test shapes to match the existing `test_flash_attn_prefill.rs` pattern (CPU reference → GPU dispatch → per-element max-diff assert). Add a new `tests/test_flash_attn_train.rs` file.

---

## E. Phasing

### Phase 1: Forward with logsumexp output, F32, D=64

**Goal:** A new `dispatch_flash_attn_train_fwd_f32_d64` that matches `sdpa_reference_f32` at the same shapes, plus emits the logsumexp buffer `L [B, H, qL]` required by Phase 2 backward.

**Work:**
- Fork `flash_attn_prefill.metal` to `flash_attn_train_fwd.metal`. Add a new `device float* L_out [[buffer(8)]]` output buffer. After the K-tile sweep, compute `L[b, h, i] = m_i * log2(e)^{-1} + log(sum_score)` (convert from base-2 accumulator back to natural-log domain for the backward formula). ~60 LOC delta.
- Add a new Rust op file `flash_attn_train.rs` in mlx-native with a `FlashAttnTrainFwdParams` struct and `dispatch_flash_attn_train_fwd_f32_d64`. Use f32 I/O only — avoids the threadgroup-memory overflow (at D=64, f32 BQ=32: 32*64*4 = 8 KB, fits comfortably). ~250 LOC.
- Ship `tests/test_flash_attn_train.rs` with the CPU-oracle forward parity + logsumexp-value tests.

**Measurable output:** `dispatch_flash_attn_train_fwd_f32_d64` passes the CPU oracle parity test at `atol=1e-4`. Logsumexp values correct to `atol=1e-5`.

### Phase 2: Backward, F32, D=64

**Goal:** `dispatch_flash_attn_train_bwd_f32_d64` producing `dQ`, `dK`, `dV` given `dO`, Q, K, V, O, L.

**Work:**
- New Metal shader `flash_attn_train_bwd.metal`. Implement FA-2 Algorithm 4: two-pass structure (one pass produces dV and the dP intermediate; second pass produces dQ and accumulates dK). Tile loops over KV for dV/dK, and over Q for dQ. Simdgroup MMA for all four matrix products (dV += P^T dO, dQ += dS K, dK += dS^T Q, dP = dO V^T). ~700–900 LOC.
- Rust dispatch `dispatch_flash_attn_train_bwd_f32_d64`. ~150 LOC.
- Tests: finite-difference falsifier + CPU oracle dQ/dK/dV parity + causal mask dK-zero test + GQA accumulation test.

**Measurable output:** All tests pass at `atol=5e-3`. The backward is self-consistent with the forward via finite-difference to within the same tolerance.

### Phase 3: GpuTape integration + swap into `qwen35_moe_forward_on_tape`

**Goal:** FlashAttn backward lives on the tape; at least one layer's attention block in `qwen35_moe_forward_on_tape` uses real GQA attention instead of the matmul proxy. Gradient flows to W_q, W_k, W_v, W_o.

**Work:**
- Add `OpKind::FlashAttnTrain` to `autograd_gpu_tape.rs` with fields `{q_idx, k_idx, v_idx, l_buf, q_buf, k_buf, v_buf, o_buf, n_heads, n_kv_heads, head_dim, seq_len, scale, do_causal}`. Add the backward match arm dispatching to `dispatch_flash_attn_train_bwd_f32_d64` and returning `parent_grads[0..2] = [dQ, dK, dV]`. ~200 LOC.
- Add a `flash_attn_train` free function on `GpuTensor` (mirrors `matmul`, `softmax` etc.) that wraps the forward dispatch and pushes the node. ~80 LOC.
- Extend `DecoderLayerWeights` with `{w_q, w_k, w_v, w_o, q_norm_w, k_norm_w}` fields; update `decoder_layer_on_tape` to do the Q/K/V projections and call `flash_attn_train` in place of the matmul proxy.
- Integration test: single-layer `decoder_layer_on_tape` with real attention, run backward, finite-diff verify `dW_q` at 5% rel tol.

**Measurable output:** AC#7-relevant: gradient flows through real attention to W_q, W_k, W_v, W_o. The matmul proxy is gone from the production DWQ training path.

---

## F. Open Questions

1. **RoPE placement in the training kernel.** The production `flash_attn_prefill` path receives Q and K that have already had IMROPE applied (the fused head-norm-rope shader at `fused_head_norm_rope_bf16.metal` runs before the prefill dispatch). Should `flash_attn_train` also receive pre-RoPE Q and K, with a separate differentiable rope op on the tape? Or should the training kernel fuse IMROPE internally? Fusing IMROPE is ~250 extra LOC in the backward shader (rope gradients are analytic: the RoPE transform is orthogonal so `d/dQ(RoPE(Q))` is just `RoPE^T = RoPE^{-1}`). The question is whether the per-head Q/K RMSNorm + RoPE both need to be inside the fused kernel or can be separate tape ops. **Operator decision required before Phase 1.**

2. **D=256 vs D=64 first.** The Qwen3.6-35B-A3B production shape is D=256 (confirmed at `gguf.rs:5342`). D=64 fits f32 in threadgroup memory; D=256 does not (as documented in `flash_attn_prefill.rs:39–51`). Phase 1 at D=64 f32 is safe but will never run on the production model. Does the implementer start at D=64 with a plan to upgrade to D=256 bf16 in Phase 2, or target D=256 bf16 from the start (accepting that the CPU oracle runs at f32 D=64 only, while the GPU test runs at bf16 D=256)? **Operator decision required before Phase 1.**

3. **Causal masking in the training kernel.** DWQ training uses full-sequence teacher forward passes with a causal mask. The existing forward kernel supports `do_causal=true` via a function constant. For the backward, causal masking must also be enforced: positions where `j > i` must produce zero contributions to `dK[j]` and `dV[j]`. Is this correctness requirement in scope for Phase 2, or deferred to Phase 3? **Operator decision required before Phase 2.**

4. **GQA head-grouping in the backward.** Qwen3.6-35B-A3B has `n_heads=16`, `n_kv_heads=2` (`gqa_factor=8`, from `gguf.rs:5330–5331`). The backward KV gradient accumulation is: `dK_j += sum_{g=0..gqa_factor} dS_{g,j}^T Q_{g,i}`. The existing forward kernel handles GQA via the `gqa_factor` param in `AttnParamsGpu`. Should the backward use the same `gqa_factor` field, or require the caller to materialize Q expanded to `[B, H_kv*gqa, S, D]` before dispatch (reducing implementation complexity at the cost of a temporary allocation)? **Operator decision required before Phase 2.**

5. **Tape memory budget for the logsumexp buffer.** The GpuTape currently calls `GpuTape::reset()` between batches to reclaim GPU memory. The `OpKind::FlashAttnTrain` variant needs to own `l_buf` (logsumexp), `q_buf`, `k_buf`, `v_buf` (the pre-attention inputs) — total ~4 × B × H × S × D × 4 bytes per layer. At the 40-layer Qwen3.6-35B-A3B forward pass with B=1, H=16, S=2048, D=256: 4 × 40 × (1 × 16 × 2048 × 256 × 4) ≈ 5.4 GB held simultaneously on tape during backward. This is within M5 Max's 96 GB but represents a meaningful footprint. Does the DWQ calibrator run one layer at a time (tape per layer, reset between layers) or a full-model tape? The current `qwen35_moe_forward_on_tape` does a full-model tape. **Operator decision on layer-granularity vs full-model forward scope.**
