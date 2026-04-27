#!/usr/bin/env python3
"""Independent oracle for chunk_gated_delta_rule_fwd (Wave 5b.1 iter 4).

This is a NAIVE O(T^2) per-token recurrence implementation of the gated
delta rule. It uses ZERO chunk decomposition: it walks the recurrence
across the full T axis per token. The math is:

    Per-head, per-batch:
        S_0 = h0                                   # [V, K] f32 state
        For each token t in 0..T:
            alpha_t = exp(g_t)                     # f32 scalar
            beta_t  = beta[t]                      # f32 scalar
            k_t     = k[t]                         # [K] (post-l2norm)
            v_t     = v[t]                         # [V]
            q_t     = q[t]                         # [K] (post-l2norm)

            # Predict-then-correct delta-rule update (arXiv 2412.06464 §4):
            v_pred  = S_t @ k_t                    # [V]
            err     = v_t - v_pred                 # [V]
            S_{t+1} = alpha_t * S_t + beta_t * outer(err, k_t)
                                                    # [V, K]

            # Output: query the post-update state.
            o_t     = S_{t+1} @ q_t                # [V]

    Final state = S_T.

This is mathematically the SAME quantity the FLA chunk-parallel pipeline
approximates. The FLA reference (the sibling `chunk_gated_delta_rule_fwd_reference.py`)
decomposes the recurrence into chunks of BT and rewrites it via the WY
representation + state-recurrence kernel. The oracle here is the
non-chunked, non-WY, naive recurrence — completely orthogonal math path.

Spec sources for the math:
  - arXiv 2412.06464 §4 (Yang-Hatamizadeh 2024)
  - DeltaNet baseline: Schlag et al. 2021, "Linear Transformers Are Secretly
    Fast Weight Programmers"

Inputs (must match the reference fixture's input bytes exactly — uses the
same SEED + recipe):
  q, k, v, g_log_decay, beta, h0  + use_qk_l2norm_in_kernel=True

Output:
  o, final_state — compared against the reference fixture's outputs.

Tolerance vs the FLA chunk-parallel reference:
  - o            : max_err < 5e-4  (close to single-kernel bar; output is
                                    a single bf16-staged dot per token)
  - final_state  : max_err < 5e-3  (matches per-kernel atol; 6 stages of
                                    bf16 round-off compound on the
                                    state-recurrence path: cumsum -> kkt
                                    bf16 cast -> solve_tril -> recompute_wu
                                    bf16 -> chunk_h bf16 -> back into
                                    f32 b_h)

The chunk-parallel WY reformulation is mathematically LOSSLESS in exact
arithmetic; the gap is purely bf16 staging. The oracle (pure f32) and the
reference (bf16 staged) agreeing to ~5e-3 on a value of magnitude 0.11
(relative ~5%) is the expected bf16 quantum, NOT a math bug.

Asserts on PASS — exits non-zero on FAIL.
"""

# SPDX-License-Identifier: Apache-2.0
# Math reference: arXiv 2412.06464 §4 + Schlag et al. 2021 (DeltaNet)
# Algorithm: naive token-by-token recurrence — completely independent of
# the FLA chunk decomposition + WY representation.

import os
import sys

import torch

# Pull constants + input maker + l2norm from the reference fixture so the
# oracle uses byte-identical inputs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_gated_delta_rule_fwd_reference as ref


def naive_l2norm_per_token_f32(x_row, eps=1e-6):
    """Independent inline f32 L2 norm for a single per-token row vector.

    Spec source: FLA `l2norm_fwd` at
      /opt/vllm/vllm/model_executor/layers/fla/ops/l2norm.py
    Math: y = x / sqrt(sum(x*x, dim=-1) + eps)
    Reference: arXiv 2412.06464 §3 (Yang-Hatamizadeh 2024) — the QK-L2
    pre-step before the gated delta-rule recurrence.

    Wave 5b.1 iter 4.5 (M2): this oracle MUST NOT call back into
    `ref.l2norm_fwd` — that would recreate the iter-1.5 fixture-oracle
    co-conspiracy class (a bug in the reference's batched l2norm would be
    invisible to this oracle). We therefore inline the math in pure f32,
    per-token, no batched intermediate storage.
    """
    f = x_row.float()
    sumsq = (f * f).sum()
    return f / torch.sqrt(sumsq + eps)


def naive_delta_rule_fwd(q, k, v, g_log_decay, beta, h0, scale,
                         use_qk_l2norm_in_kernel):
    """Naive per-token recurrence. NO chunking, NO WY, NO solve_tril.

    Walks the token axis serially per (batch, V-head); compute is f32
    throughout. The chunk-parallel reference rewrites this same recurrence
    using A_inv + WY representation; this oracle does not.

    Wave 5b.1 iter 4.5 (M2): l2norm is now done INLINE per-token in f32
    (see `naive_l2norm_per_token_f32`). The previous version called
    `ref.l2norm_fwd` (a shared FLA-style batched impl) which would have
    masked any l2norm-side bug. The output buffer `o` is also kept in f32
    throughout the recurrence; bf16 cast happens only at the final
    comparison read-back to match the reference's bf16 storage.
    """

    B_, T_, _, K_ = q.shape
    _, _, H_, V_ = v.shape
    Hg_ = q.shape[2]
    GROUP_RATIO_ = H_ // Hg_

    # Keep o in pure f32 throughout the per-token recurrence; only the FINAL
    # read-back is cast to bf16 to match the reference's bf16 storage type.
    # This avoids per-token bf16 round-off in the oracle (oracle is supposed
    # to be independent of bf16 staging concerns of the chunk-parallel path).
    o_f32 = torch.zeros((B_, T_, H_, V_), dtype=torch.float32)
    final_state = torch.zeros((B_, H_, V_, K_), dtype=torch.float32)

    for b in range(B_):
        for i_h in range(H_):
            kh = i_h // GROUP_RATIO_
            S = h0[b, i_h].clone()  # [V, K] f32 (zero in this fixture)

            for t in range(T_):
                # Per-token gates.
                alpha = float(torch.exp(g_log_decay[b, t, i_h]))
                beta_t = float(beta[b, t, i_h])

                k_t = k[b, t, kh, :].float()    # [K]
                v_t = v[b, t, i_h, :].float()   # [V]
                q_t = q[b, t, kh, :].float()    # [K]

                # Inline per-token f32 L2 norm (M2 — independent of
                # ref.l2norm_fwd; spec: FLA l2norm.py / arXiv 2412.06464 §3).
                if use_qk_l2norm_in_kernel:
                    q_t = naive_l2norm_per_token_f32(q_t)
                    k_t = naive_l2norm_per_token_f32(k_t)

                # Predict-then-correct delta-rule update
                # (arXiv 2412.06464 §4 + Schlag 2021):
                v_pred = S @ k_t                 # [V]
                err = v_t - v_pred               # [V]
                S = alpha * S + beta_t * torch.outer(err, k_t)   # [V, K]

                # Output: post-update query.
                o_t = S @ q_t * scale            # [V] (FLA applies scale on output)
                o_f32[b, t, i_h, :] = o_t        # f32 storage; cast at end.

            final_state[b, i_h] = S

    # Cast to bf16 at the end (single quantization point, matches reference
    # storage type for byte-stream comparison; per-token storage was f32).
    o = o_f32.to(torch.bfloat16)
    return o, final_state


def main():
    q, k, v, g_log_decay, beta, h0 = ref.make_inputs()

    # Run the FLA-style chunk-parallel reference.
    o_ref, final_state_ref = ref.chunk_gated_delta_rule_fwd(
        q, k, v, g_log_decay, beta, h0, ref.SCALE,
        use_qk_l2norm_in_kernel=True,
    )

    # Run the naive O(T^2) per-token recurrence (independent math).
    o_naive, final_state_naive = naive_delta_rule_fwd(
        q, k, v, g_log_decay, beta, h0, ref.SCALE,
        use_qk_l2norm_in_kernel=True,
    )

    # Compare bf16 outputs (cast both to f32 for diff).
    o_max_err = (o_ref.float() - o_naive.float()).abs().max().item()
    fs_max_err = (final_state_ref - final_state_naive).abs().max().item()

    o_max_abs = o_ref.float().abs().max().item()
    fs_max_abs = final_state_ref.abs().max().item()

    print("oracle vs reference (chunk-parallel) cross-validation:")
    print(f"  o:           max(|ref|) = {o_max_abs:.6f}    max(|err|) = {o_max_err:.3e}")
    print(f"  final_state: max(|ref|) = {fs_max_abs:.6f}    max(|err|) = {fs_max_err:.3e}")

    # Tolerance bars:
    #   o:           5e-4 — close to single-kernel bar (output is a single
    #                       bf16-staged dot per token).
    #   final_state: 5e-3 — matches per-kernel atol (6 bf16 stages compound
    #                       on the state-recurrence path).
    # The chunk-parallel WY reformulation is mathematically LOSSLESS in exact
    # arithmetic; the gap is purely bf16 staging. The oracle (pure f32)
    # versus the reference (bf16 staged) agreeing to ~5e-3 on a value of
    # magnitude 0.11 (relative ~5%) is the expected bf16 quantum.
    O_TOL = 5e-4
    FS_TOL = 5e-3

    ok = True
    if o_max_err >= O_TOL:
        print(f"FAIL: o max_err {o_max_err:.3e} >= {O_TOL:.0e}")
        ok = False
    if fs_max_err >= FS_TOL:
        print(f"FAIL: final_state max_err {fs_max_err:.3e} >= {FS_TOL:.0e}")
        ok = False

    if ok:
        print(
            f"PASS: oracle agrees with chunk-parallel reference "
            f"(o<{O_TOL:.0e}, final_state<{FS_TOL:.0e})."
        )
        sys.exit(0)
    else:
        print("Oracle disagrees with reference — math path divergence.")
        sys.exit(1)


if __name__ == "__main__":
    main()
