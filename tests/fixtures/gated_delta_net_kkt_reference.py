#!/usr/bin/env python3
"""Reference fixture generator for chunk_scaled_dot_kkt (Wave 5b.1 iter 2).

Produces the bit-for-bit reference output for
`chunk_scaled_dot_kkt` Metal kernel, which is a straight port of FLA's
`chunk_scaled_dot_kkt_fwd_kernel` at
/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-99.

The math (per chunk i_t, per (batch b, V-head i_h)):

    b_beta = beta[bos+i_t*BT : bos+(i_t+1)*BT, i_h]            # [BT] f32
    b_g    = g[bos+i_t*BT : bos+(i_t+1)*BT, i_h]               # [BT] f32 (cumsumed)
    b_A = zeros([BT, BT], f32)
    for i_k in range(K // BK):
        # Note: k uses K-head kh = i_h // (H // Hg) for GQA broadcast.
        b_k  = k[bos+i_t*BT:..., kh, i_k*BK : i_k*BK+BK]       # [BT, BK] bf16
        b_kb = b_k * b_beta[:, None]                            # bf16*f32 → f32
        # FLA line 86: b_A += dot(b_kb.to(b_k.dtype), trans(b_k))
        # The cast b_kb.to(bf16) happens BEFORE the dot, AFTER the scale.
        b_A += dot(b_kb.to(bf16).float(), b_k.float().T)        # [BT, BT] f32
    if g is not None:
        b_g_diff = b_g[:, None] - b_g[None, :]                  # [BT, BT] f32
        b_A *= exp(b_g_diff)
    # Strict lower-tri mask: row > col only.
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = where(m_A, b_A, 0)
    store p_A : [B, T, H, BT]  output_dtype = f32

Output dtype: f32 (per FLA `output_dtype=torch.float32` at line 109).

Inputs (shapes match iter-1 fixture for compatibility):
  k   : [B, T, Hg, K]  bf16
  beta: [B, T, H]      f32
  g   : [B, T, H]      f32  (post-chunk-local cumsum)

Output:
  A   : [B, T, H, BT]  f32

Shape: B=1, T=128, Hg=2, H=4, K=128, BT=64 → NT=2, GROUP_RATIO=2.

Determinism: fixed seed 0xC0FFEE on torch.Generator (CPU) — same seed as
iter-1 fixture so the inputs are byte-identical to iter-1's where applicable
(i.e., k bytes match iter-1's k bytes; beta/g are new fields not in iter-1).

Reference: FLA chunk_scaled_dot_kkt.py:36-99.
"""

# SPDX-License-Identifier: Apache-2.0
# Implementation reference: flash-linear-attention (FLA)
# `chunk_scaled_dot_kkt_fwd_kernel` at
# /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-99,
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT license).

import json
import os

import torch


# ---------------------------------------------------------------------------
# Constants — same as iter-1 fixture, plus BK split.
# ---------------------------------------------------------------------------
FIXTURE_VERSION = "wave5b.1-iter2-kkt-v1"
SEED = 0xC0FFEE
B = 1            # batch
T = 128          # seq len
Hg = 2           # K-head count
H = 4            # V-head count
K = 128          # head_dim_k
BT = 64          # chunk size
BK = 64          # K-tile width (K // BK = 2 inner iterations)

assert T % BT == 0
assert K % BK == 0
NT = T // BT
assert H % Hg == 0
GROUP_RATIO = H // Hg


# ---------------------------------------------------------------------------
# Reference implementation — direct PyTorch port of chunk_scaled_dot_kkt.py:74-99.
# ---------------------------------------------------------------------------
def chunk_scaled_dot_kkt_reference(
    k: torch.Tensor,    # [B, T, Hg, K] bf16
    beta: torch.Tensor, # [B, T, H]     f32
    g: torch.Tensor,    # [B, T, H]     f32 (cumsumed within chunk)
    BT: int,
    BK: int,
):
    """Per-chunk strictly-lower-triangular β·K·K^T·exp(Δg) → A [B, T, H, BT]."""
    assert k.shape == (B, T, Hg, K), f"k {tuple(k.shape)} != ({B},{T},{Hg},{K})"
    assert beta.shape == (B, T, H), f"beta {tuple(beta.shape)} != ({B},{T},{H})"
    assert g.shape == (B, T, H), f"g {tuple(g.shape)} != ({B},{T},{H})"
    assert k.dtype == torch.bfloat16
    assert beta.dtype == torch.float32
    assert g.dtype == torch.float32

    NT_local = T // BT
    nbk = K // BK
    A = torch.zeros((B, T, H, BT), dtype=torch.float32)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO  # GQA-mapped K-head

            for i_t in range(NT_local):
                t_start = i_t * BT
                t_end = t_start + BT

                b_beta = beta[b, t_start:t_end, i_h]            # [BT] f32
                b_g    = g[b, t_start:t_end, i_h]               # [BT] f32

                b_A = torch.zeros((BT, BT), dtype=torch.float32)

                for i_k in range(nbk):
                    k_start = i_k * BK
                    k_end = k_start + BK
                    # b_k: [BT, BK] bf16
                    b_k = k[b, t_start:t_end, kh, k_start:k_end]

                    # FLA line 85: b_kb = b_k * b_beta[:, None]
                    # In Triton this is bf16 * f32 → f32 (Triton promotes).
                    b_kb_f32 = b_k.float() * b_beta[:, None]    # [BT, BK] f32

                    # FLA line 86: b_A += dot(b_kb.to(b_k.dtype), trans(b_k))
                    # The cast b_kb.to(bf16) happens BEFORE the dot, AFTER scale.
                    # Then dot is bf16 × bf16 → f32 accumulator.
                    b_kb_bf16 = b_kb_f32.to(torch.bfloat16)     # FLA :86 cast

                    # bf16 × bf16 with f32 accumulation: promote both operands
                    # to f32 (the bytes are bf16 — promotion is lossless of
                    # the bf16 representation, mirroring `tl.dot(bf16, bf16,
                    # out=f32)`).
                    b_A = b_A + b_kb_bf16.float() @ b_k.float().T  # [BT, BT] f32

                # FLA lines 88-92: gate multiply via exp(g_i - g_j).
                b_g_diff = b_g[:, None] - b_g[None, :]          # [BT, BT] f32
                b_A = b_A * torch.exp(b_g_diff)

                # FLA lines 94-95: strict lower-tri mask (row > col only).
                # No T-boundary masking needed since T is a multiple of BT.
                o_t = torch.arange(BT)
                m_A = o_t[:, None] > o_t[None, :]               # [BT, BT] bool
                b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))

                A[b, t_start:t_end, i_h, :] = b_A

    return A


# ---------------------------------------------------------------------------
# Generate inputs deterministically.
# ---------------------------------------------------------------------------
def make_inputs():
    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(SEED)

    # Reuse iter-1's k generation pattern (same scale 0.1, same dtype).
    # Note: this k is byte-identical to iter-1's input k IF we draw in the
    # same order. We draw k FIRST; iter-1 also draws k first.
    k = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    # beta: positive in [0.1, 1.0] — sigmoid-like distribution to emulate
    # the network's sigmoid(beta_proj) output.
    beta = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.9 + 0.1

    # g: log-decay (cumsumed within each chunk). Same recipe as iter-1.
    g_raw = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.045 - 0.05
    g = g_raw.clone()
    for i_t in range(NT):
        s, e = i_t * BT, (i_t + 1) * BT
        g[:, s:e, :] = torch.cumsum(g_raw[:, s:e, :], dim=1)

    return k, beta, g


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

    k, beta, g = make_inputs()
    A = chunk_scaled_dot_kkt_reference(k, beta, g, BT, BK)

    write_bin(os.path.join(here, "gated_delta_net_kkt_input_k.bin"), k)
    write_bin(os.path.join(here, "gated_delta_net_kkt_input_beta.bin"), beta)
    write_bin(os.path.join(here, "gated_delta_net_kkt_input_g.bin"), g)
    write_bin(os.path.join(here, "gated_delta_net_kkt_A_ref.bin"), A)

    meta = {
        "version": FIXTURE_VERSION,
        "seed": SEED,
        "B": B, "T": T, "Hg": Hg, "H": H, "K": K, "BT": BT, "BK": BK, "NT": NT,
        "group_ratio": GROUP_RATIO,
        "fla_reference": (
            "chunk_scaled_dot_kkt_fwd_kernel at "
            "/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-99"
        ),
        "inputs": {
            "k":    {"shape": [B, T, Hg, K], "dtype": "bf16",
                     "file": "gated_delta_net_kkt_input_k.bin"},
            "beta": {"shape": [B, T, H],     "dtype": "f32",
                     "file": "gated_delta_net_kkt_input_beta.bin"},
            "g":    {"shape": [B, T, H],     "dtype": "f32",
                     "file": "gated_delta_net_kkt_input_g.bin"},
        },
        "outputs": {
            "A": {"shape": [B, T, H, BT], "dtype": "f32",
                  "file": "gated_delta_net_kkt_A_ref.bin"},
        },
        "tolerance": {
            # f32 output, but inputs are bf16 → bf16-cast intermediates inside
            # dots dominate the noise budget. The strict-lower mask zeros half
            # the matrix exactly, so the noise lives only in the lower-tri
            # entries. 5e-3 atol + 2e-2 rtol matches iter-1's bf16 bar.
            "A_atol": 5e-3, "A_rtol": 2e-2,
        },
    }
    with open(os.path.join(here, "gated_delta_net_kkt_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote kkt fixtures to {here}")
    print(f"  A shape       = {tuple(A.shape)}     ({A.numel() * 4} bytes)")
    print(f"  A.abs().max() = {A.abs().max().item():.6f}")
    # Sanity: per-chunk upper-triangle (incl. diagonal) must be exactly zero.
    # A is [B, T, H, BT]; for each chunk i_t, rows are A[b, i_t*BT:(i_t+1)*BT, h, :]
    # which gives a [BT, BT] block. Within that block, strict-lower means
    # row > col, so col >= row entries must be zero.
    upper_max = 0.0
    for b in range(B):
        for h in range(H):
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                block = A[b, ts:te, h, :]                  # [BT, BT]
                o_t = torch.arange(BT)
                up_mask = o_t[:, None] <= o_t[None, :]    # diag + above
                u = block[up_mask].abs().max().item()
                if u > upper_max:
                    upper_max = u
    print(f"  upper-tri (incl. diag) max abs = {upper_max:.3e}  (must be 0.0)")
    assert upper_max == 0.0, "strict lower-tri mask broken in reference"


if __name__ == "__main__":
    main()
