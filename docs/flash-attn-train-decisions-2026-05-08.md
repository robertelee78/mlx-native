# `flash_attn_train` — Architectural Decisions

Companion to `flash-attn-train-kernel-scope-2026-05-08.md`.  Each of the
five open questions in §F of the scope doc has been resolved.  The
implementer can start Phase 1 with no further architectural input
required.

Decisions optimize for: **correctness of gradient signal, codebase
coherence, joint-optimization fidelity to mlx-lm's `dwq.py` algorithm,
and inference quality of the trained s/b values** — *not* for speed of
the training kernel itself (training runs once during DWQ calibration;
the trained safetensors are the output that affects production
inference).

---

## D1. RoPE placement → **Separate differentiable RoPE op on the tape**

**Decision:** RoPE is *not* fused into `flash_attn_train`.  A new
`OpKind::Rope` variant is added to
`/opt/hf2q/src/calibrate/autograd_gpu_tape.rs` carrying its own
forward/backward kernel.  Q and K arrive at `flash_attn_train`
already-rotated.

**Why:**

1. **Mathematical separability.**  RoPE is `R(pos) · x` per position; the
   backward is `RoPE(dx, -pos)` — trivially correct, no recomputation
   needed.  Fusing it into the attention kernel obscures both the
   forward and the backward.
2. **Tape framework idiom.**  Every other primitive on the tape is a
   single named `OpKind` (Matmul, Softmax, Log, RmsNorm, …) with a
   dedicated backward arm at `autograd_gpu_tape.rs:604`.  RoPE
   belongs on that list, not as a sub-feature of an attention op.
3. **Multi-model reuse.**  Qwen35 uses IMROPE with sections=[24,20,20,0];
   future models may use vanilla RoPE, NTK-scaled RoPE, YaRN, etc.  A
   separate op variant is parametrized by section/freq schedule; a
   fused kernel forks per model.
4. **Test surface.**  Standalone RoPE has a clean finite-diff falsifier
   on `[B, H, S, D]` inputs.  Fused, the gradient signal mixes RoPE
   error with attention error and bisection becomes harder.
5. **Production-inference path is irrelevant for this trade-off.**
   `flash_attn_prefill_bf16_d256` fuses RoPE for inference latency
   (`fused_head_norm_rope_bf16.metal` is the existing fusion).
   `flash_attn_train` runs only during the DWQ calibration pass — the
   trained s/b values are the output that affects inference.  Kernel-
   pass count during calibration does not measurably change the
   wall-time of a multi-hour training run.

**Cost:** +1 small Metal shader (`rope_forward.metal`, ~80 LOC), +1
small backward shader (`rope_backward.metal`, ~80 LOC), +1 OpKind +
backward arm in autograd_gpu_tape.rs (~120 LOC).  Total ~280 LOC
versus ~250 LOC for the fused-into-backward shader path — within
noise, but the architecture is materially cleaner.

---

## D2. D=64 vs D=256 first → **D=256 bf16 directly; f32 CPU oracles for all sizes**

**Decision:** Skip D=64 f32 entirely.  Phase 1 ships
`dispatch_flash_attn_train_fwd_bf16_d256` from day one.  All tests
compare bf16 GPU output against an f32 CPU oracle (the same pattern
`flash_attn_prefill`'s tests already use; verified at
`flash_attn_prefill.rs:16-35` — *every* registered kernel is bf16, no
f32 instantiations).

**Why:**

1. **No throwaway code.**  D=64 f32 first means TWO kernel
   implementations to maintain; the f32 path would never run on
   production Qwen3.6 (D=256 per the ADR-021 architecture; f32 D=256
   overflows threadgroup SMEM at ~53.7 KB, documented at
   `flash_attn_prefill.rs:39-51`).  Mantra: no shortcuts, no
   fallback, no stub.  A development-only kernel that production
   never touches is a stub.
2. **Test infrastructure already supports bf16-vs-f32-oracle parity.**
   The existing `flash_attn_prefill` test pattern at
   `tests/test_flash_attn_prefill.rs::test_gpu_bf16_d256_*` uses an
   `sdpa_reference_f32` CPU oracle and asserts `atol=1e-4` against
   bf16 GPU output.  Re-use that pattern; no new oracle infrastructure
   needed.
3. **bf16 numerical stability for backward is solved.**  Apple's
   `simdgroup_matrix<float, 8, 8>` accumulator accepts bf16 inputs
   and accumulates in f32 — the same trick the inference kernel
   uses.  Backward gradients (especially the `D_i = rowsum(O_i *
   dO_i)` term) accumulate in f32 SMEM and only round to bf16 at
   write-out.  `atol=5e-3` for dQ/dK/dV is comfortable.
4. **Memory pressure during training.**  bf16 halves the activation
   footprint vs f32.  At Qwen3.6-35B-A3B production dimensions
   (H=16, qL=2048, D=256), the per-layer Q+K+V+O tensors are 64 MB
   in bf16 vs 128 MB in f32.  Across the 40-layer full-model tape
   (D5 below) that's 2.6 GB vs 5.2 GB — bf16 frees up headroom for
   the Adam state buffers (8 GB combined `m` + `v` for the model's
   trainable s/b across all Linears).

**Cost:** Eliminates the entire D=64-f32-then-D=256-bf16 phasing
(Option B in §E of the scope doc would have been the second pass).
Phase 1 directly targets the production shape.

**Caveat for Phase 1 scope:** dispatchers are emitted for D=64 bf16
AND D=256 bf16 (mirror of the existing prefill family which has
both).  D=64 is needed for tests at small fixture sizes (matmul
backward kernel requires m,n ≥ 32 but attention has its own minimum
working tile size; prefill's D=64 dispatcher is the lower-bound
test target).  This is *not* a stub — D=64 bf16 IS instantiated by
the existing inference family at `flash_attn_prefill.rs:140` and is
production-reachable for smaller models.

---

## D3. Causal + sliding-window mask → **Phase 2 scope (with the backward)**

**Decision:** Phase 2 ships causal AND sliding-window mask backward
together with the dQ/dK/dV implementation.  Not deferred.

**Why:**

1. **Correctness invariant, not a feature.**  DWQ training uses
   causal masking on every batch (next-token-prediction loss).  A
   backward that ignores the mask produces gradients on positions
   that should not have contributed to the loss — the gradient
   direction is wrong, not just slow.  Shipping that as Phase 2 and
   "fixing" it in Phase 3 means Phase 2 is throwaway.
2. **Implementation cost is small.**  The forward already enforces
   `S[i,j] += (j > i) * -INF` before the softmax (mirror of the
   inference path).  Backward uses the SAME mask: `dS[i,j] *= (j ≤
   i)` element-wise, applied right after `dS = P * (dP - D_i)`.
   ~20 LOC delta in the backward shader.
3. **Sliding-window matters for Qwen35.**  Per
   `inference/models/qwen35/forward_gpu.rs`, alternating layers use
   either full attention or sliding-window attention with
   `swa_ratio` per the model config.  DWQ trains across all layers,
   so the backward MUST honor the mask each layer uses.  Same `(j
   ≤ i)` enforcement parameterized by `(window_lo, window_hi)`
   bounds (caller passes the same bounds the forward uses).
4. **Phase 3 is a different concern.**  Phase 3 (tape integration +
   proxy swap) doesn't touch the mask machinery — it threads
   already-correct kernels into `qwen35_moe_forward_on_tape`.
   Pushing mask correctness to Phase 3 mixes orthogonal work.

**Cost:** +20 LOC in `flash_attn_train_bwd.metal`, +1 test
(`backward_zeroes_dk_for_masked_positions`) following the
`tests/test_flash_attn_prefill.rs` mask-test pattern.

---

## D4. GQA accumulation → **Kernel accepts `gqa_factor` param internally**

**Decision:** `flash_attn_train` accepts the same `gqa_factor` field
in its `AttnParamsGpu`-style param block as the existing inference
kernel does (`flash_attn_prefill.rs::AttnParamsGpu.gqa_factor` —
already there).  Backward dK/dV accumulation across the
`gqa_factor` Q-heads sharing each KV head is internal to the
shader.  Caller does NOT pre-expand Q.

**Why:**

1. **Match existing convention.**  The forward inference kernel
   already expects this shape.  A different convention for the
   training kernel would be a coherence wart that operators have
   to remember.  Mantra: do what's right.
2. **Memory cost of pre-expanding.**  Qwen3.6-35B-A3B has H=16,
   H_kv=2 → gqa_factor=8.  Pre-expanding Q from `[B, H_kv, S, D]
   to [B, H, S, D]` blows up Q (and dQ on backward) by 8×.  At
   production dimensions that's 8 × 32 MB = 256 MB additional
   per-layer Q vs the kernel-internal accumulation path.  Across
   the 40-layer tape that compounds.
3. **FA-2 paper algorithm.**  Algorithm 4 is described in
   single-head terms but trivially extends with an outer loop over
   gqa_factor inside the K/V tile.  The simdgroup MMA accumulator
   sums the contributions from all gqa_factor Q-heads sharing a
   KV head into the same dK/dV register tile in one pass.  This
   is how every published FlashAttention-2 GPU implementation
   handles GQA backward; Apple Metal is no exception.
4. **Kernel parameter surface stays small.**  `gqa_factor: u32`
   is one extra field in the existing param struct; it's already
   in the inference kernel's struct; backward kernel signature
   matches forward kernel signature 1:1 except for the added L
   buffer + dY input + 3 grad outputs.

**Cost:** +1 outer loop in the dK/dV accumulation tile.  ~30 LOC
in `flash_attn_train_bwd.metal`.

---

## D5. Tape memory scope → **Full-model tape, joint optimization preserved**

**Decision:** DWQ training builds a single full-model GpuTape per
batch.  All forward layers stack onto one tape; one `backward()`
call computes gradients for ALL trainable s/b across ALL Linears in
the model in a single pass; one Adam step updates all params
together.  *No* per-layer tape reset.

**Why:**

1. **Joint optimization is the whole algorithm.**  `mlx-lm/quant/dwq.py:106`
   computes `losses = kl_div_loss(scale * logits, scale * targets)`
   on the FULL model output.  `mx.value_and_grad(loss_fn)(params,
   …)` in `dwq.py:121` produces gradients on ALL params from ONE
   loss.  This is the cross-layer error-compensation signal that
   Option A relies on — it's why Option B (per-Linear teacher) was
   falsified on 2026-05-08.  Per-layer-reset training collapses
   back to a similar greedy structure: each layer's s/b update sees
   only THAT layer's contribution to the loss.  Different algorithm
   entirely; would have to be re-falsified.
2. **Memory is feasible.**  At Qwen3.6-35B-A3B production
   dimensions, 40 layers × 64 MB per-layer activations (Q+K+V+O+L
   in bf16, per D2) = 2.6 GB on tape simultaneously.  The M5 Max
   has 128 GB unified memory; model weights are ~70 GB at Q5_K_M;
   Adam state for the trainable s/b is ~8 GB (m + v across all
   Linears).  Total ~80 GB — comfortably within budget.
3. **Gradient checkpointing is the canonical solution if memory
   pressure ever appears.**  mlx-lm's `dwq.py:79` already exposes
   `gradient_checkpoint: bool` exactly for this case — discard
   intermediate activations during forward, recompute during
   backward.  Cost: 2× forward FLOPs on the recomputed segment;
   benefit: ~50% peak-memory reduction.  This is a tunable runtime
   knob, NOT an architectural decision that locks us into greedy
   training.  Phase 3 wires this in as `cfg.gradient_checkpoint:
   bool` matching mlx-lm's surface, defaulted off.
4. **Existing FA-2 implementation already does its own internal
   checkpointing.**  Option 1 (recompute-during-backward) in §B of
   the scope doc IS gradient checkpointing for the attention
   matrix specifically.  At qL=2048, kL=2048 the materialized
   attention matrix would be 16 × 2048 × 2048 × 2 bytes (bf16) =
   268 MB per layer × 40 = 10.7 GB — that's what we save by
   choosing Option 1.  Layer-level checkpointing on TOP of attention
   checkpointing is additive; we get the inner savings for free.

**Cost:** Zero additional implementation cost.  This is the
default tape behavior — no changes needed.

---

## Phase 1 unblocked

With these five decisions resolved, Phase 1 has no remaining
architectural input required.  Implementer's worklist:

1. New file `src/ops/rope.rs` + `src/shaders/rope_forward.metal` +
   `src/shaders/rope_backward.metal`.  Standalone op, exposes
   `dispatch_rope_forward_bf16` + `dispatch_rope_backward_bf16`.
   Test: `tests/test_rope.rs` with finite-diff falsifier.
2. New file `src/ops/flash_attn_train.rs` + `src/shaders/flash_attn_train_fwd.metal`.
   Forward only, bf16, D=64 + D=256, with logsumexp `L [B, H, qL]`
   output.  Causal mask + sliding-window mask supported in forward
   (mirror of `flash_attn_prefill.metal`'s mask handling).
   Tests: `tests/test_flash_attn_train.rs::forward_*` comparing bf16
   GPU output + L against `sdpa_reference_f32` CPU oracle at
   `atol=1e-4` for O and `atol=1e-5` for L.

Phase 2 follows once Phase 1 lands and the forward + L parity is
confirmed.

---

*Decisions made by:* /loop iter, 2026-05-08, after re-reading
`~/Documents/mantra.txt` and verifying every claim about existing
code against the source.  Citations to file:line in the scope doc
companion.
