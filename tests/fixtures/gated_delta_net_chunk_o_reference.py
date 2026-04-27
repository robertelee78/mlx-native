#!/usr/bin/env python3
"""Reference fixture generator for chunk_fwd_o (Wave 5b.1 iter 3).

Produces the bit-for-bit reference output for the per-chunk output kernel
that closes the FLA chunk pipeline. This is a straight port of FLA's
`chunk_fwd_kernel_o` at
/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py:42-138.

The OP UNDER TEST consumes:

  q     : [B, T, Hg, K]   bf16   — per-token query (post-l2 / post-rope)
  k     : [B, T, Hg, K]   bf16   — same K-head layout as q (Hg < H, GQA)
  v_new : [B, T, H,  V]   bf16   — chunk-recurrence-projected V (iter 1 output)
  h     : [B, NT, H, V, K] bf16  — per-chunk-start state (iter 1 output)
  g     : [B, T, H]       f32    — log-decay accumulator (chunk-cumsumed)
  scale : f32                    — = K^-0.5

and emits

  o     : [B, T, H, V]    bf16   — per-token output

The math (per FLA chunk_o.py:90-138, per (b, i_h, i_t, i_v)):

    b_o = zeros[BT, BV] f32
    b_A = zeros[BT, BT] f32

    # K-tile loop (chunk_o.py:93-113):
    for i_k in range(K // BK):
        b_q = q[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]            # bf16 [BT, BK]
        b_k = k[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]            # bf16 [BT, BK]
        b_h = h[b, i_t, i_h, i_v*BV:(i_v+1)*BV, i_k*BK:...]   # bf16 [BV, BK]
        b_o += dot(b_q, trans(b_h))    # [BT, BK] @ [BK, BV] = [BT, BV]
        b_A += dot(b_q, trans(b_k))    # [BT, BK] @ [BK, BT] = [BT, BT]

    # Gate (chunk_o.py:115-120 — NOT skipped here; USE_G is true for GDN):
    b_g = g[b, t_chunk, i_h]                                  # f32 [BT]
    b_o = b_o * exp(b_g)[:, None]
    b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    # Causal mask INCLUSIVE of diagonal (chunk_o.py:122-125):
    o_t = i_t*BT + arange(BT)
    m_A = (o_t[:, None] >= o_t[None, :]) & (o_t < T)[..., None] & (o_t < T)
    b_A = where(m_A, b_A, 0)

    # Closing dot — bf16 cast of b_A AFTER masking, BEFORE dot with v
    # (chunk_o.py:135-137; this is the load-bearing iter-1.5 lesson placement).
    b_v = v_new[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV]           # bf16 [BT, BV]
    b_o = b_o * scale + dot(b_A.to(bf16), b_v) * scale        # f32 acc → bf16 store

    o[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV] = bf16(b_o)

CRITICAL ORDERING (per iter-1.5):
  - The bf16 cast on b_A at line 137 happens AFTER mask + gate, BEFORE the
    final dot with v. Moving or skipping it changes the bf16 round-trip
    and silently shifts numerics.
  - The mask is `>=` (causal+diag) — DIFFERENT from kkt's strict `>`.
  - Two `* scale` multiplications: one on b_o (the h-term), one on the
    dot output (the A·v term). Don't fold them.

Inputs (deterministic, seed 0xC0FFEE):
  q     : [B, T, Hg, K]    bf16
  k     : [B, T, Hg, K]    bf16
  v_new : [B, T, H,  V]    bf16
  h     : [B, NT, H, V, K] bf16
  g     : [B, T, H]        f32  (already chunk-cumsumed)
  scale : f32              (= K^-0.5)

Outputs:
  o     : [B, T, H, V]     bf16

Shape: B=1, T=128, Hg=2, H=4, K=128, V=128, BT=64 → NT=2,
        scale = 1/sqrt(128) ≈ 0.08838834.
"""

# SPDX-License-Identifier: Apache-2.0
# Implementation reference: flash-linear-attention (FLA)
# `chunk_fwd_kernel_o` at
# /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py:42-138,
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT license).

import json
import math
import os

import torch


# ---------------------------------------------------------------------------
# Constants — same B/T/Hg/H/K/V/BT as iter-1+2 fixtures for compositionality.
# ---------------------------------------------------------------------------
FIXTURE_VERSION = "wave5b.1-iter3-chunk-o-v1"
SEED = 0xC0FFEE
B = 1
T = 128
Hg = 2
H = 4
K = 128
V = 128
BT = 64
BK = 32   # iter-3 tile width (kernel-side; reference loops at this size for parity)
BV = 32   # iter-3 tile width (kernel-side)

assert T % BT == 0
assert K % BK == 0
assert V % BV == 0
NT = T // BT
GROUP_RATIO = H // Hg
SCALE = K ** -0.5


# ---------------------------------------------------------------------------
# Reference — direct PyTorch port of chunk_o.py:90-138, structured exactly
# like the Triton kernel (per-tile loops, bf16 cast at line 137).
# ---------------------------------------------------------------------------
def chunk_fwd_o_reference(q, k, v, h, g, scale):
    """Per FLA chunk_o.py:42-138. Returns o bf16."""
    assert q.shape == (B, T, Hg, K)
    assert k.shape == (B, T, Hg, K)
    assert v.shape == (B, T, H, V)
    assert h.shape == (B, NT, H, V, K)
    assert g.shape == (B, T, H)
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert h.dtype == torch.bfloat16
    assert g.dtype == torch.float32

    nbk = K // BK
    nbv = V // BV
    o = torch.zeros((B, T, H, V), dtype=torch.bfloat16)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                for i_v in range(nbv):
                    vs, ve = i_v * BV, (i_v + 1) * BV

                    b_o = torch.zeros((BT, BV), dtype=torch.float32)
                    b_A = torch.zeros((BT, BT), dtype=torch.float32)

                    # K-tile loop (chunk_o.py:93-113).
                    for i_k in range(nbk):
                        ks, ke = i_k * BK, (i_k + 1) * BK
                        b_q = q[b, ts:te, kh, ks:ke]               # bf16 [BT, BK]
                        b_k = k[b, ts:te, kh, ks:ke]               # bf16 [BT, BK]
                        b_h = h[b, i_t, i_h, vs:ve, ks:ke]         # bf16 [BV, BK]
                        # bf16 × bf16 → f32 dot (PyTorch promotes via .float()).
                        # b_o += b_q @ b_h.T    [BT, BK] @ [BK, BV] -> [BT, BV]
                        b_o = b_o + b_q.float() @ b_h.float().T
                        # b_A += b_q @ b_k.T    [BT, BK] @ [BK, BT] -> [BT, BT]
                        b_A = b_A + b_q.float() @ b_k.float().T

                    # Gate (chunk_o.py:115-120; USE_G == True for GDN).
                    b_g = g[b, ts:te, i_h]                          # f32 [BT]
                    b_o = b_o * torch.exp(b_g)[:, None]
                    b_A = b_A * torch.exp(b_g[:, None] - b_g[None, :])

                    # Causal+diag mask (chunk_o.py:122-125 — `>=`, INCLUSIVE
                    # of diagonal — different from kkt's strict `>`).
                    o_t = torch.arange(BT) + ts
                    m_t = o_t < T
                    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
                    b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))

                    # bf16 round-trip on b_A (chunk_o.py:137 — AFTER mask,
                    # BEFORE dot with v). Iter-1.5 lesson: load-bearing.
                    b_v = v[b, ts:te, i_h, vs:ve]                   # bf16 [BT, BV]
                    b_A_bf16 = b_A.to(torch.bfloat16)
                    # Two `* scale`: one on b_o, one on the dot.
                    b_o = b_o * scale + (b_A_bf16.float() @ b_v.float()) * scale

                    o[b, ts:te, i_h, vs:ve] = b_o.to(torch.bfloat16)

    return o


# ---------------------------------------------------------------------------
# Generate inputs deterministically.
# ---------------------------------------------------------------------------
def make_inputs():
    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(SEED)

    # Small-magnitude inputs — same recipe as iter-1+2 for fixture coherence.
    q = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    k = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    # v_new — produced by iter-1's inter-state kernel; here we draw it
    # synthetically (this isolates chunk_fwd_o from the iter-1 path).
    v_new = (torch.randn((B, T, H, V), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    # g: log-decay (chunk-cumsumed). Same recipe as iter-1 fixture.
    g_raw = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.045 - 0.05
    g = g_raw.clone()
    for i_t in range(NT):
        s, e = i_t * BT, (i_t + 1) * BT
        g[:, s:e, :] = torch.cumsum(g_raw[:, s:e, :], dim=1)

    # h: per-chunk-start state. Synthetic, small-magnitude bf16 [B, NT, H, V, K].
    # In production this comes from iter-1's chunk-recurrence kernel.
    h = (torch.randn((B, NT, H, V, K), generator=g_rng, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    return q, k, v_new, h, g


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

    q, k, v_new, h, g = make_inputs()
    o = chunk_fwd_o_reference(q, k, v_new, h, g, SCALE)

    write_bin(os.path.join(here, "gated_delta_net_chunk_o_input_q.bin"), q)
    write_bin(os.path.join(here, "gated_delta_net_chunk_o_input_k.bin"), k)
    write_bin(os.path.join(here, "gated_delta_net_chunk_o_input_v.bin"), v_new)
    write_bin(os.path.join(here, "gated_delta_net_chunk_o_input_h.bin"), h)
    write_bin(os.path.join(here, "gated_delta_net_chunk_o_input_g.bin"), g)
    write_bin(os.path.join(here, "gated_delta_net_chunk_o_o_ref.bin"), o)

    meta = {
        "version": FIXTURE_VERSION,
        "seed": SEED,
        "B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V,
        "BT": BT, "BK": BK, "BV": BV, "NT": NT,
        "group_ratio": GROUP_RATIO,
        "scale": SCALE,
        "fla_reference": (
            "chunk_fwd_kernel_o at "
            "/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py:42-138"
        ),
        "fla_commit_note": (
            "vendored copy in /opt/vllm; original FLA repo "
            "github.com/sustcsonglin/flash-linear-attention. "
            "vllm /opt/vllm head: "
            + os.popen("cd /opt/vllm && git rev-parse --short HEAD 2>/dev/null").read().strip()
        ),
        "inputs": {
            "q":     {"shape": [B, T, Hg, K],    "dtype": "bf16",
                      "file":  "gated_delta_net_chunk_o_input_q.bin"},
            "k":     {"shape": [B, T, Hg, K],    "dtype": "bf16",
                      "file":  "gated_delta_net_chunk_o_input_k.bin"},
            "v_new": {"shape": [B, T, H, V],     "dtype": "bf16",
                      "file":  "gated_delta_net_chunk_o_input_v.bin"},
            "h":     {"shape": [B, NT, H, V, K], "dtype": "bf16",
                      "file":  "gated_delta_net_chunk_o_input_h.bin"},
            "g":     {"shape": [B, T, H],        "dtype": "f32",
                      "file":  "gated_delta_net_chunk_o_input_g.bin"},
        },
        "outputs": {
            "o": {"shape": [B, T, H, V], "dtype": "bf16",
                  "file": "gated_delta_net_chunk_o_o_ref.bin"},
        },
        "tolerance": {
            "o_atol": 5e-3, "o_rtol": 2e-2,
        },
    }
    with open(os.path.join(here, "gated_delta_net_chunk_o_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote chunk_fwd_o fixtures to {here}")
    print(f"  o shape = {tuple(o.shape)}    ({o.numel() * 2} bytes)")
    print(f"  o.abs().max()    = {o.float().abs().max().item():.6f}")
    print(f"  scale            = {SCALE:.10f}")


if __name__ == "__main__":
    main()
