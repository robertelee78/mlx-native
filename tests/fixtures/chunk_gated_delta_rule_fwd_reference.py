#!/usr/bin/env python3
"""End-to-end reference fixture for chunk_gated_delta_rule_fwd
(Wave 5b.1 iter 4).

This is the orchestrator that composes the iter-1..iter-3 kernels into a
single end-to-end chunk-parallel delta-rule forward pass. It mirrors FLA's

    /opt/vllm/vllm/model_executor/layers/fla/ops/chunk.py:23-84

call chain:

    g          = chunk_local_cumsum(g, chunk_size=BT)
    A_strict   = chunk_scaled_dot_kkt_fwd(k, beta, g)        # [B, T, H, BT]
    A_inv      = solve_tril(A_strict)  =  (I + A_strict)^-1   # per chunk-block
    w, u       = recompute_w_u_fwd(k, v, beta, A_inv, g_cumsum=g)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, h0)
    o          = chunk_fwd_o(q, k, v_new, h, g, scale)

Plus the L2-norm pre-step from the autograd-Function forward at chunk.py:106-108:

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

The reference here PRE-RUNS the whole chain in PyTorch. This is byte-faithful
to what FLA's stack produces — it is *not* an independent reimplementation
of the math (that's what the oracle below is). Reference tolerance vs the
GPU orchestrator is loose (1e-2 atol / 5e-2 rtol) because 6 stages of
bf16 round-off compound.

Inputs (deterministic, seed 0xC0FFEE):
  q            : [B, T, Hg, K]    bf16
  k            : [B, T, Hg, K]    bf16
  v            : [B, T, H,  V]    bf16
  g_log_decay  : [B, T, H]        f32  (raw per-token log-decay; NOT cumsumed)
  beta         : [B, T, H]        f32
  h0           : [B, H, V, K]     f32  (zero in this fixture)

Outputs:
  o            : [B, T, H, V]     bf16
  final_state  : [B, H, V, K]     f32

Shape: B=1, T=128, Hg=2, H=4, K=128, V=128, BT=64 -> NT=2,
       use_qk_l2norm_in_kernel=True (Qwen3.6 default).
       scale = 1/sqrt(128) ~= 0.08838834.
"""

# SPDX-License-Identifier: Apache-2.0
# Implementation reference: flash-linear-attention (FLA)
# `chunk_gated_delta_rule_fwd` at
# /opt/vllm/vllm/model_executor/layers/fla/ops/chunk.py:23-84
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT license).

import json
import os

import numpy as np
import torch


FIXTURE_VERSION = "wave5b.1-iter4-orchestrator-v1"
SEED = 0xC0FFEE
B = 1
T = 128
Hg = 2
H = 4
K = 128
V = 128
BT = 64
BK = 64       # kkt + recompute_wu inner tile
BV = 64       # recompute_wu inner tile
BK_O = 32     # chunk_o inner K tile (matches iter-3 kernel default)
BV_O = 32     # chunk_o inner V tile (matches iter-3 kernel default)
EPS_L2 = 1e-6  # FLA l2norm_fwd default

assert T % BT == 0
assert K % BK == 0
assert V % BV == 0
assert K % BK_O == 0
assert V % BV_O == 0
NT = T // BT
GROUP_RATIO = H // Hg
SCALE = K ** -0.5


# ---------------------------------------------------------------------------
# Stage 1: l2norm_fwd (only when use_qk_l2norm_in_kernel=True).
#   x / sqrt(sum(x^2, dim=-1) + eps)
# Operates on the last dim (head-dim K) for both q and k.
# ---------------------------------------------------------------------------
def l2norm_fwd(x, eps=EPS_L2):
    """Reference for FLA l2norm_fwd: per-row L2 norm with eps in the radicand."""
    f = x.float()
    sumsq = (f * f).sum(dim=-1, keepdim=True)
    rstd = torch.rsqrt(sumsq + eps)
    return (f * rstd).to(x.dtype)


# ---------------------------------------------------------------------------
# Stage 2: chunk_local_cumsum on g (per FLA chunk_local_cumsum_scalar:160).
#   For each chunk i_t in 0..NT, replace g[:, i_t*BT : (i_t+1)*BT, :] with
#   its in-chunk cumulative sum along the time axis. Operates in f32.
# ---------------------------------------------------------------------------
def chunk_local_cumsum(g_log_decay, chunk_size=BT):
    out = g_log_decay.clone()
    for i_t in range(NT):
        s, e = i_t * chunk_size, (i_t + 1) * chunk_size
        out[:, s:e, :] = torch.cumsum(g_log_decay[:, s:e, :], dim=1)
    return out


# ---------------------------------------------------------------------------
# Stage 3: chunk_scaled_dot_kkt_fwd — produces [B, T, H, BT] strict-lower-tri A.
# Mirrors iter-2 kkt fixture math.
# ---------------------------------------------------------------------------
def chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum):
    nbk = K // BK
    A = torch.zeros((B, T, H, BT), dtype=torch.float32)
    o_t = torch.arange(BT)
    m_strict = o_t[:, None] > o_t[None, :]   # FLA :94-95 strict lower (no diag).

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                b_beta = beta[b, ts:te, i_h]
                b_g = g_cumsum[b, ts:te, i_h]
                b_A = torch.zeros((BT, BT), dtype=torch.float32)
                for i_k in range(nbk):
                    ks, ke = i_k * BK, (i_k + 1) * BK
                    b_k = k[b, ts:te, kh, ks:ke]
                    # FLA :86 — bf16 cast on the scaled K AFTER the *beta multiply.
                    b_kb_bf16 = (b_k.float() * b_beta[:, None]).to(torch.bfloat16)
                    b_A = b_A + b_kb_bf16.float() @ b_k.float().T
                # FLA :91-92 — multiply by exp(g_i - g_j).
                b_A = b_A * torch.exp(b_g[:, None] - b_g[None, :])
                b_A = torch.where(m_strict, b_A, torch.zeros_like(b_A))
                A[b, ts:te, i_h, :] = b_A
    return A


# ---------------------------------------------------------------------------
# Stage 4: solve_tril — per-chunk (I + A_strict_lower)^-1.
# Spec: /opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:506-530.
# ---------------------------------------------------------------------------
def solve_tril(A_strict):
    Ai = torch.zeros_like(A_strict)
    I_BT = np.eye(BT, dtype=np.float32)
    for b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                L = A_strict[b, ts:te, i_h, :].numpy()
                Ai_chunk = np.linalg.inv(I_BT + L).astype(np.float32)
                Ai[b, ts:te, i_h, :] = torch.from_numpy(Ai_chunk)
    return Ai


# ---------------------------------------------------------------------------
# Stage 5: recompute_w_u_fwd — produces w [B, T, H, K] bf16, u [B, T, H, V] bf16.
# Mirrors iter-2 recompute_wu fixture math; same f32-acc → bf16-store policy.
# ---------------------------------------------------------------------------
def recompute_w_u_fwd(k, v, beta, A_inv, g_cumsum):
    w = torch.zeros((B, T, H, K), dtype=torch.bfloat16)
    u = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    nbv = V // BV
    nbk = K // BK
    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                b_beta = beta[b, ts:te, i_h]
                b_A = A_inv[b, ts:te, i_h, :]
                b_g_exp = torch.exp(g_cumsum[b, ts:te, i_h])
                for i_v in range(nbv):
                    vs, ve = i_v * BV, (i_v + 1) * BV
                    b_v = v[b, ts:te, i_h, vs:ve]
                    b_vb = (b_v.float() * b_beta[:, None]).to(torch.bfloat16)
                    b_u = b_A @ b_vb.float()
                    u[b, ts:te, i_h, vs:ve] = b_u.to(torch.bfloat16)
                for i_k in range(nbk):
                    ks, ke = i_k * BK, (i_k + 1) * BK
                    b_k = k[b, ts:te, kh, ks:ke]
                    b_kb = (b_k.float() * b_beta[:, None] * b_g_exp[:, None]).to(
                        torch.bfloat16
                    )
                    b_w = b_A @ b_kb.float()
                    w[b, ts:te, i_h, ks:ke] = b_w.to(torch.bfloat16)
    return w, u


# ---------------------------------------------------------------------------
# Stage 6: chunk_gated_delta_rule_fwd_h — produces h [B, NT, H, V, K] bf16,
# v_new [B, T, H, V] bf16, final_state [B, H, V, K] f32.
# Mirrors iter-1 chunk fixture math (post iter-1.5 fix).
# ---------------------------------------------------------------------------
def chunk_h_reference(k, w, u, g_cumsum, h0):
    h_out = torch.zeros((B, NT, H, V, K), dtype=torch.bfloat16)
    v_new = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    final_state = torch.zeros((B, H, V, K), dtype=torch.float32)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            b_h = h0[b, i_h].clone()  # [V, K] f32

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = t_start + BT

                h_out[b, i_t, i_h] = b_h.to(torch.bfloat16)

                b_w = w[b, t_start:t_end, i_h, :].float()
                b_u = u[b, t_start:t_end, i_h, :].float()
                b_v_f32 = b_u - b_w @ b_h.T
                v_new[b, t_start:t_end, i_h] = b_v_f32.to(torch.bfloat16)

                last = t_end - 1
                g_last = g_cumsum[b, last, i_h]
                g_blk = g_cumsum[b, t_start:t_end, i_h]
                b_v_f32 = b_v_f32 * torch.exp(g_last - g_blk).unsqueeze(-1)
                b_h = b_h * torch.exp(g_last)

                b_k = k[b, t_start:t_end, kh, :].float()
                b_v_bf16 = b_v_f32.to(torch.bfloat16).float()
                b_h = b_h + b_v_bf16.T @ b_k

            final_state[b, i_h] = b_h

    return h_out, v_new, final_state


# ---------------------------------------------------------------------------
# Stage 7: chunk_fwd_o — produces o [B, T, H, V] bf16.
# Mirrors iter-3 chunk_o fixture math (BK_O = BV_O = 32).
# ---------------------------------------------------------------------------
def chunk_fwd_o_reference(q, k, v_new, h, g_cumsum, scale):
    nbk = K // BK_O
    nbv = V // BV_O
    o = torch.zeros((B, T, H, V), dtype=torch.bfloat16)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                for i_v in range(nbv):
                    vs, ve = i_v * BV_O, (i_v + 1) * BV_O
                    b_o = torch.zeros((BT, BV_O), dtype=torch.float32)
                    b_A = torch.zeros((BT, BT), dtype=torch.float32)

                    for i_k in range(nbk):
                        ks, ke = i_k * BK_O, (i_k + 1) * BK_O
                        b_q = q[b, ts:te, kh, ks:ke]
                        b_k = k[b, ts:te, kh, ks:ke]
                        b_h = h[b, i_t, i_h, vs:ve, ks:ke]
                        b_o = b_o + b_q.float() @ b_h.float().T
                        b_A = b_A + b_q.float() @ b_k.float().T

                    b_g = g_cumsum[b, ts:te, i_h]
                    b_o = b_o * torch.exp(b_g)[:, None]
                    b_A = b_A * torch.exp(b_g[:, None] - b_g[None, :])

                    o_t = torch.arange(BT) + ts
                    m_t = o_t < T
                    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
                    b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))

                    b_v = v_new[b, ts:te, i_h, vs:ve]
                    b_A_bf16 = b_A.to(torch.bfloat16)
                    b_o = b_o * scale + (b_A_bf16.float() @ b_v.float()) * scale

                    o[b, ts:te, i_h, vs:ve] = b_o.to(torch.bfloat16)
    return o


# ---------------------------------------------------------------------------
# End-to-end orchestrator (mirrors chunk.py:23-84 + chunk.py:106-108).
# ---------------------------------------------------------------------------
def chunk_gated_delta_rule_fwd(
    q, k, v, g_log_decay, beta, h0, scale, use_qk_l2norm_in_kernel
):
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    g_cumsum = chunk_local_cumsum(g_log_decay, chunk_size=BT)
    A_strict = chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum)
    A_inv = solve_tril(A_strict)
    w, u = recompute_w_u_fwd(k, v, beta, A_inv, g_cumsum)
    h, v_new, final_state = chunk_h_reference(k, w, u, g_cumsum, h0)
    o = chunk_fwd_o_reference(q, k, v_new, h, g_cumsum, scale)
    return o, final_state


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
def make_inputs():
    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(SEED)

    # Same input recipe as iter-1..3 fixtures for compositional sanity.
    q = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    k = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    v = (torch.randn((B, T, H, V), generator=g_rng, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    # Raw per-token g (NOT cumsumed). Range [-0.05, -0.005] to keep
    # exp(g) bounded — same recipe as iter-1..3 fixtures, but BEFORE cumsum.
    g_log_decay = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.045 - 0.05
    beta = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.9 + 0.1
    h0 = torch.zeros((B, H, V, K), dtype=torch.float32)

    return q, k, v, g_log_decay, beta, h0


def write_bin(path, t):
    arr = t.contiguous()
    if arr.dtype == torch.bfloat16:
        view = arr.view(torch.int16)
        with open(path, "wb") as f:
            f.write(view.numpy().tobytes())
    elif arr.dtype == torch.float32:
        with open(path, "wb") as f:
            f.write(arr.numpy().tobytes())
    else:
        raise ValueError(f"unsupported dtype {arr.dtype}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    q, k, v, g_log_decay, beta, h0 = make_inputs()

    o, final_state = chunk_gated_delta_rule_fwd(
        q, k, v, g_log_decay, beta, h0, SCALE, use_qk_l2norm_in_kernel=True
    )

    # Persist inputs (raw, pre-l2norm/cumsum so the GPU orchestrator can apply
    # them itself).
    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_input_q.bin"), q)
    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_input_k.bin"), k)
    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_input_v.bin"), v)
    write_bin(
        os.path.join(here, "chunk_gated_delta_rule_fwd_input_g_log_decay.bin"),
        g_log_decay,
    )
    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_input_beta.bin"), beta)
    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_input_h0.bin"), h0)

    write_bin(os.path.join(here, "chunk_gated_delta_rule_fwd_o_ref.bin"), o)
    write_bin(
        os.path.join(here, "chunk_gated_delta_rule_fwd_final_state_ref.bin"),
        final_state,
    )

    meta = {
        "version": FIXTURE_VERSION,
        "seed": SEED,
        "B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V,
        "BT": BT, "BK": BK, "BV": BV, "BK_O": BK_O, "BV_O": BV_O, "NT": NT,
        "group_ratio": GROUP_RATIO,
        "scale": SCALE,
        "use_qk_l2norm_in_kernel": True,
        "eps_l2": EPS_L2,
        "fla_reference": (
            "chunk_gated_delta_rule_fwd at "
            "/opt/vllm/vllm/model_executor/layers/fla/ops/chunk.py:23-84 "
            "+ Function.forward l2norm pre-step at chunk.py:106-108"
        ),
        "fla_commit_note": (
            "vendored copy in /opt/vllm; original FLA repo "
            "github.com/sustcsonglin/flash-linear-attention. "
            "vllm /opt/vllm head: "
            + os.popen("cd /opt/vllm && git rev-parse --short HEAD 2>/dev/null").read().strip()
        ),
        "inputs": {
            "q": {"shape": [B, T, Hg, K], "dtype": "bf16",
                  "file": "chunk_gated_delta_rule_fwd_input_q.bin"},
            "k": {"shape": [B, T, Hg, K], "dtype": "bf16",
                  "file": "chunk_gated_delta_rule_fwd_input_k.bin"},
            "v": {"shape": [B, T, H, V], "dtype": "bf16",
                  "file": "chunk_gated_delta_rule_fwd_input_v.bin"},
            "g_log_decay": {"shape": [B, T, H], "dtype": "f32",
                            "file": "chunk_gated_delta_rule_fwd_input_g_log_decay.bin"},
            "beta": {"shape": [B, T, H], "dtype": "f32",
                     "file": "chunk_gated_delta_rule_fwd_input_beta.bin"},
            "h0": {"shape": [B, H, V, K], "dtype": "f32",
                   "file": "chunk_gated_delta_rule_fwd_input_h0.bin"},
        },
        "outputs": {
            "o": {"shape": [B, T, H, V], "dtype": "bf16",
                  "file": "chunk_gated_delta_rule_fwd_o_ref.bin"},
            "final_state": {"shape": [B, H, V, K], "dtype": "f32",
                            "file": "chunk_gated_delta_rule_fwd_final_state_ref.bin"},
        },
        "tolerance": {
            "o_atol": 1e-2, "o_rtol": 5e-2,
            "final_state_atol": 1e-2,
            "_note": (
                "End-to-end bar: 6 stages of bf16 round-off compound. "
                "Single-kernel fixtures use 5e-3 atol; this orchestrator "
                "needs ~2x for cumulative drift."
            ),
        },
    }
    with open(
        os.path.join(here, "chunk_gated_delta_rule_fwd_meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)

    print(f"wrote chunk_gated_delta_rule_fwd fixtures to {here}")
    print(f"  o shape           = {tuple(o.shape)}    ({o.numel() * 2} bytes)")
    print(f"  o.abs().max()     = {o.float().abs().max().item():.6f}")
    print(f"  final_state shape = {tuple(final_state.shape)}    ({final_state.numel() * 4} bytes)")
    print(f"  final_state.abs().max() = {final_state.abs().max().item():.6f}")
    print(f"  scale             = {SCALE:.10f}")


if __name__ == "__main__":
    main()
