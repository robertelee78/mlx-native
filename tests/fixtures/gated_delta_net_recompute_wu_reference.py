#!/usr/bin/env python3
"""Reference fixture generator for recompute_w_u_fwd (Wave 5b.1 iter 2).

Produces the bit-for-bit reference output for `recompute_w_u_fwd` Metal
kernel, which is a straight port of FLA's `recompute_w_u_fwd_kernel` at
/opt/vllm/vllm/model_executor/layers/fla/ops/wy_fast.py:29-117.

The OP UNDER TEST is recompute_w_u alone. However its `A` input is the
output of FLA's solve_tril applied to kkt's output. To make the test
end-to-end correct we generate `A` here via a numerically-clean reference
chain (kkt → numpy linalg inverse of `I + tril(A_pre)` per chunk), so the
Rust test loads a spec-correct `A` directly and compares only the
recompute_w_u output against this fixture.

The math (per chunk i_t, per (batch b, V-head i_h)):

    b_beta = beta[bos+i_t*BT : bos+(i_t+1)*BT, i_h]     # [BT] f32
    b_A    = A[bos+i_t*BT : ..., i_h*BT : i_h*BT+BT]    # [BT, BT] f32 (post-solve)
    b_g    = exp(g[bos+i_t*BT : ..., i_h])              # [BT] f32

    for i_v in range(V // BV):
        b_v  = v[bos+i_t*BT:..., i_h, i_v*BV : i_v*BV+BV]    # [BT, BV] bf16
        # FLA wy_fast.py:92-93 — bf16 cast happens AFTER the scale, BEFORE the dot.
        b_vb = (b_v.float() * b_beta[:, None]).to(bf16)      # [BT, BV] bf16
        b_u  = dot(A_f32, b_vb_bf16, allow_tf32=False)       # [BT, BV] f32
        store p_u : bf16([BT, BV])

    for i_k in range(K // BK):
        b_k  = k[bos+i_t*BT:..., kh, i_k*BK : i_k*BK+BK]     # [BT, BK] bf16
        # FLA wy_fast.py:113-114 — same pattern: scale in f32, bf16 cast, dot.
        b_kb = (b_k.float() * b_beta[:, None] * b_g[:, None]).to(bf16)
        b_w  = dot(A_f32, b_kb_bf16)                          # [BT, BK] f32
        store p_w : bf16([BT, BK])

Note: the dot-arg order is `tl.dot(b_A, b_vb)` — A is the LEFT operand
(unscaled f32) and b_vb is the RIGHT (bf16). Triton promotes the f32 A to
match the lower-precision operand if needed; in PyTorch we promote both
to f32 for the matmul (bytes are bf16 — promotion is lossless).

Inputs:
  k   : [B, T, Hg, K]  bf16
  v   : [B, T, H,  V]  bf16
  beta: [B, T, H]      f32
  g   : [B, T, H]      f32  (already chunk-cumsumed; same as kkt input)
  A   : [B, T, H, BT]  f32  (post-solve_tril; computed in this fixture)

Outputs:
  w   : [B, T, H, K]   bf16
  u   : [B, T, H, V]   bf16

Shape: B=1, T=128, Hg=2, H=4, K=128, V=128, BT=64, BV=64, BK=64.
"""

# SPDX-License-Identifier: Apache-2.0
# Implementation reference: flash-linear-attention (FLA)
# `recompute_w_u_fwd_kernel` at
# /opt/vllm/vllm/model_executor/layers/fla/ops/wy_fast.py:29-117,
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT license).

import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants — same as iter-1 + kkt fixtures.
# ---------------------------------------------------------------------------
FIXTURE_VERSION = "wave5b.1-iter2-recompute-wu-v1"
SEED = 0xC0FFEE
B = 1
T = 128
Hg = 2
H = 4
K = 128
V = 128
BT = 64
BK = 64
BV = 64

assert T % BT == 0
assert K % BK == 0
assert V % BV == 0
NT = T // BT
GROUP_RATIO = H // Hg


# ---------------------------------------------------------------------------
# Reference helpers.
# ---------------------------------------------------------------------------
def chunk_scaled_dot_kkt_pre_solve(k, beta, g):
    """Same as kkt_reference.kkt fixture — but inlined here so this fixture
    is fully self-contained. Returns A_pre (strict-lower-triangular)."""
    nbk = K // BK
    A = torch.zeros((B, T, H, BT), dtype=torch.float32)
    o_t = torch.arange(BT)
    m_A = o_t[:, None] > o_t[None, :]

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                b_beta = beta[b, ts:te, i_h]
                b_g = g[b, ts:te, i_h]
                b_A = torch.zeros((BT, BT), dtype=torch.float32)
                for i_k in range(nbk):
                    ks, ke = i_k * BK, (i_k + 1) * BK
                    b_k = k[b, ts:te, kh, ks:ke]
                    b_kb_bf16 = (b_k.float() * b_beta[:, None]).to(torch.bfloat16)
                    b_A = b_A + b_kb_bf16.float() @ b_k.float().T
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_A = b_A * torch.exp(b_g_diff)
                b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))
                A[b, ts:te, i_h, :] = b_A
    return A


def solve_tril_reference(A_pre):
    """Compute (I + L)^-1 per chunk where L = strict-lower-tri input.

    FLA's solve_tril returns `(I + A)^-1` (per its docstring at
    /opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:514).
    Input A is strictly lower-triangular (kkt output). Output Ai has the
    same shape; per-chunk it equals `(I + L)^-1`.

    Use numpy linalg.inv on the f32 [BT, BT] system — this is the
    spec-faithful reference (no chunked merge_16x16/32x32 splits needed
    for parity).
    """
    Ai = torch.zeros_like(A_pre)
    for b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                L = A_pre[b, ts:te, i_h, :].numpy()        # [BT, BT] f32
                # L should already be strictly lower-tri; the diagonal is
                # zero from the kkt fixture's mask.
                I_BT = np.eye(BT, dtype=np.float32)
                Ai_chunk = np.linalg.inv(I_BT + L).astype(np.float32)
                Ai[b, ts:te, i_h, :] = torch.from_numpy(Ai_chunk)
    return Ai


def recompute_w_u_reference(k, v, beta, g_cumsum, A):
    """Per FLA wy_fast.py:63-116. Returns (w, u) bf16."""
    assert k.shape == (B, T, Hg, K)
    assert v.shape == (B, T, H, V)
    assert beta.shape == (B, T, H)
    assert g_cumsum.shape == (B, T, H)
    assert A.shape == (B, T, H, BT)
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert beta.dtype == torch.float32
    assert g_cumsum.dtype == torch.float32
    assert A.dtype == torch.float32

    w = torch.zeros((B, T, H, K), dtype=torch.bfloat16)
    u = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    nbv = V // BV
    nbk = K // BK

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                b_beta = beta[b, ts:te, i_h]            # [BT] f32
                b_A    = A[b, ts:te, i_h, :]            # [BT, BT] f32
                b_g    = torch.exp(g_cumsum[b, ts:te, i_h])  # [BT] f32 (FLA :72)

                # u-loop (FLA wy_fast.py:74-94).
                for i_v in range(nbv):
                    vs, ve = i_v * BV, (i_v + 1) * BV
                    b_v = v[b, ts:te, i_h, vs:ve]                # bf16 [BT, BV]
                    # FLA :92 — (b_v.float() * b_beta[:, None]).to(bf16)
                    b_vb = (b_v.float() * b_beta[:, None]).to(torch.bfloat16)
                    # FLA :93 — dot(b_A, b_vb)  [bf16-staged A × bf16 vb → f32].
                    # Triton's tl.dot promotes f32 A onto the bf16 path
                    # internally; in PyTorch we promote both to f32 (the bf16
                    # bytes are preserved by the cast back).
                    b_u = b_A @ b_vb.float()                      # [BT, BV] f32
                    u[b, ts:te, i_h, vs:ve] = b_u.to(torch.bfloat16)

                # w-loop (FLA wy_fast.py:96-116).
                for i_k in range(nbk):
                    ks, ke = i_k * BK, (i_k + 1) * BK
                    b_k = k[b, ts:te, kh, ks:ke]                  # bf16 [BT, BK]
                    # FLA :114 — (b_k.float() * b_beta[:, None] * b_g[:, None]).to(bf16)
                    b_kb = (b_k.float() * b_beta[:, None] * b_g[:, None]).to(torch.bfloat16)
                    # FLA :115 — dot(b_A, b_kb).
                    b_w = b_A @ b_kb.float()                       # [BT, BK] f32
                    w[b, ts:te, i_h, ks:ke] = b_w.to(torch.bfloat16)

    return w, u


# ---------------------------------------------------------------------------
# Generate inputs.
# ---------------------------------------------------------------------------
def make_inputs():
    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(SEED)

    # k, beta, g — same recipe as kkt_reference for byte-identity.
    k = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    beta = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.9 + 0.1
    g_raw = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.045 - 0.05
    g = g_raw.clone()
    for i_t in range(NT):
        s, e = i_t * BT, (i_t + 1) * BT
        g[:, s:e, :] = torch.cumsum(g_raw[:, s:e, :], dim=1)

    # v — new field; draw AFTER k/beta/g to match kkt fixture's input bytes.
    v = (torch.randn((B, T, H, V), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    return k, v, beta, g


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

    k, v, beta, g = make_inputs()
    A_pre = chunk_scaled_dot_kkt_pre_solve(k, beta, g)
    A = solve_tril_reference(A_pre)
    w, u = recompute_w_u_reference(k, v, beta, g, A)

    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_input_k.bin"), k)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_input_v.bin"), v)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_input_beta.bin"), beta)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_input_g.bin"), g)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_input_A.bin"), A)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_w_ref.bin"), w)
    write_bin(os.path.join(here, "gated_delta_net_recompute_wu_u_ref.bin"), u)

    meta = {
        "version": FIXTURE_VERSION,
        "seed": SEED,
        "B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V,
        "BT": BT, "BK": BK, "BV": BV, "NT": NT,
        "group_ratio": GROUP_RATIO,
        "fla_reference": (
            "recompute_w_u_fwd_kernel at "
            "/opt/vllm/vllm/model_executor/layers/fla/ops/wy_fast.py:29-117"
        ),
        "A_input_chain": (
            "kkt_pre_solve(k, beta, g) -> numpy linalg.inv(I + L) per chunk; "
            "spec-faithful equivalent to FLA chunk_scaled_dot_kkt_fwd + "
            "solve_tril"
        ),
        "inputs": {
            "k":    {"shape": [B, T, Hg, K], "dtype": "bf16",
                     "file": "gated_delta_net_recompute_wu_input_k.bin"},
            "v":    {"shape": [B, T, H,  V], "dtype": "bf16",
                     "file": "gated_delta_net_recompute_wu_input_v.bin"},
            "beta": {"shape": [B, T, H],     "dtype": "f32",
                     "file": "gated_delta_net_recompute_wu_input_beta.bin"},
            "g":    {"shape": [B, T, H],     "dtype": "f32",
                     "file": "gated_delta_net_recompute_wu_input_g.bin"},
            "A":    {"shape": [B, T, H, BT], "dtype": "f32",
                     "file": "gated_delta_net_recompute_wu_input_A.bin"},
        },
        "outputs": {
            "w": {"shape": [B, T, H, K], "dtype": "bf16",
                  "file": "gated_delta_net_recompute_wu_w_ref.bin"},
            "u": {"shape": [B, T, H, V], "dtype": "bf16",
                  "file": "gated_delta_net_recompute_wu_u_ref.bin"},
        },
        "tolerance": {
            # bf16 outputs; bar matches iter-1 chunk fixture.
            "w_atol": 5e-3, "w_rtol": 2e-2,
            "u_atol": 5e-3, "u_rtol": 2e-2,
        },
    }
    with open(os.path.join(here, "gated_delta_net_recompute_wu_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote recompute_w_u fixtures to {here}")
    print(f"  w shape = {tuple(w.shape)}      ({w.numel() * 2} bytes)")
    print(f"  u shape = {tuple(u.shape)}      ({u.numel() * 2} bytes)")
    print(f"  w.abs().max() = {w.float().abs().max().item():.6f}")
    print(f"  u.abs().max() = {u.float().abs().max().item():.6f}")
    print(f"  A.abs().max() = {A.abs().max().item():.6f}")


if __name__ == "__main__":
    main()
