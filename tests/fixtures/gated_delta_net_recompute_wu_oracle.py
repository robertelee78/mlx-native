#!/usr/bin/env python3
"""Independent oracle for recompute_w_u_fwd (Wave 5b.1 iter 2).

Wave 5b.1 iter 1.5 lesson: every new kernel needs a third independent
recipe to prevent fixture-and-kernel co-conspiracy bugs.

The reference fixture (gated_delta_net_recompute_wu_reference.py)
decomposes the math as TWO inner loops over V-tiles and K-tiles, each
of which casts the scaled operand to bf16 right before the dot
(mirroring FLA wy_fast.py:92 and :114). The oracle uses a DIFFERENT
recipe: NO inner loops — instead, cast the FULL [BT, V] / [BT, K] matrix
in one shot, then matmul.

ORACLE RECIPE — single matmul per (b, h, i_t):

    For u (FLA wy_fast.py:74-94):
        b_v_full     = v[b, t_chunk, i_h, :]                    # bf16 [BT, V]
        b_vb_full_f32 = b_v_full.float() * b_beta[:, None]      # [BT, V] f32
        b_vb_full_bf16 = b_vb_full_f32.to(torch.bfloat16)       # FLA :92 cast
        u_chunk = b_A @ b_vb_full_bf16.float()                  # [BT, V] f32
        u[chunk] = u_chunk.to(bf16)

    For w (FLA wy_fast.py:96-116):
        b_k_full     = k[b, t_chunk, kh, :]                     # bf16 [BT, K]
        b_kb_full_f32 = b_k_full.float() * b_beta[:, None] * b_g[:, None]
        b_kb_full_bf16 = b_kb_full_f32.to(torch.bfloat16)       # FLA :114 cast
        w_chunk = b_A @ b_kb_full_bf16.float()                  # [BT, K] f32
        w[chunk] = w_chunk.to(bf16)

This matches the spec when (a) the bf16 cast is on the scaled operand
(post-scale, pre-dot) and (b) the dot accumulator is f32. Tile-decomposition
of the dot is mathematically identical to one-shot ONLY IF those properties
hold; otherwise, the per-tile bf16 round-trip's elementwise nature would
match column-tiles only when the operand-of-second-dim (V or K) is the
column axis of the dot — which it IS here (b_A @ b_vb has BT rows, V cols;
the V cols are tile-partitioned and the bf16 cast is elementwise so it
commutes with the V-tile partition).

This oracle exits with code 1 if the existing reference output deviates
from the bit-exact-spec computation by more than 1e-6.
"""

# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))


def load_meta():
    with open(os.path.join(HERE, "gated_delta_net_recompute_wu_meta.json"), "r") as f:
        return json.load(f)


def read_bf16(path: str, n_elems: int) -> torch.Tensor:
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n_elems * 2, f"{path}: {len(raw)} bytes != {n_elems * 2}"
    flat = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    return flat.view(torch.bfloat16).clone()


def read_f32(path: str, n_elems: int) -> torch.Tensor:
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n_elems * 4, f"{path}: {len(raw)} bytes != {n_elems * 4}"
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return flat.clone()


def recompute_wu_oracle(k, v, beta, g_cumsum, A, B, T, Hg, H, K, V, BT):
    """Single-matmul recipe — no V-tile / K-tile inner loops."""
    NT = T // BT
    GROUP_RATIO = H // Hg

    w = torch.zeros((B, T, H, K), dtype=torch.bfloat16)
    u = torch.zeros((B, T, H, V), dtype=torch.bfloat16)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT

                b_beta = beta[b, ts:te, i_h]                  # [BT] f32
                b_A    = A[b, ts:te, i_h, :]                  # [BT, BT] f32
                b_g    = torch.exp(g_cumsum[b, ts:te, i_h])   # [BT] f32

                # u: full [BT, V] in one shot.
                b_v = v[b, ts:te, i_h, :]                     # bf16 [BT, V]
                b_vb_bf16 = (b_v.float() * b_beta[:, None]).to(torch.bfloat16)
                b_u = b_A @ b_vb_bf16.float()                 # [BT, V] f32
                u[b, ts:te, i_h, :] = b_u.to(torch.bfloat16)

                # w: full [BT, K] in one shot.
                b_k = k[b, ts:te, kh, :]                      # bf16 [BT, K]
                b_kb_bf16 = (b_k.float() * b_beta[:, None] * b_g[:, None]).to(torch.bfloat16)
                b_w = b_A @ b_kb_bf16.float()                 # [BT, K] f32
                w[b, ts:te, i_h, :] = b_w.to(torch.bfloat16)

    return w, u


def main():
    meta = load_meta()
    B = meta["B"]; T = meta["T"]; Hg = meta["Hg"]; H = meta["H"]
    K = meta["K"]; V = meta["V"]; BT = meta["BT"]

    k = read_bf16(os.path.join(HERE, "gated_delta_net_recompute_wu_input_k.bin"),
                  B * T * Hg * K).view(B, T, Hg, K)
    v = read_bf16(os.path.join(HERE, "gated_delta_net_recompute_wu_input_v.bin"),
                  B * T * H * V).view(B, T, H, V)
    beta = read_f32(os.path.join(HERE, "gated_delta_net_recompute_wu_input_beta.bin"),
                    B * T * H).view(B, T, H)
    g = read_f32(os.path.join(HERE, "gated_delta_net_recompute_wu_input_g.bin"),
                 B * T * H).view(B, T, H)
    A = read_f32(os.path.join(HERE, "gated_delta_net_recompute_wu_input_A.bin"),
                 B * T * H * BT).view(B, T, H, BT)

    w_oracle, u_oracle = recompute_wu_oracle(k, v, beta, g, A, B, T, Hg, H, K, V, BT)

    w_existing = read_bf16(
        os.path.join(HERE, "gated_delta_net_recompute_wu_w_ref.bin"),
        B * T * H * K).view(B, T, H, K)
    u_existing = read_bf16(
        os.path.join(HERE, "gated_delta_net_recompute_wu_u_ref.bin"),
        B * T * H * V).view(B, T, H, V)

    # Compare in f32 (bf16 byte-equality would also work since both sides
    # do the same final to-bf16 cast on the same f32 result, but f32 print
    # makes the magnitude readable).
    max_w_err = (w_oracle.float() - w_existing.float()).abs().max().item()
    max_u_err = (u_oracle.float() - u_existing.float()).abs().max().item()

    tol = 1e-6
    print("recompute_w_u oracle vs existing reference fixture:")
    print(f"  max_w_err = {max_w_err:.6e}   (tol {tol:.1e})")
    print(f"  max_u_err = {max_u_err:.6e}   (tol {tol:.1e})")

    failed = (max_w_err > tol) or (max_u_err > tol)
    if failed:
        print()
        print("ORACLE FAIL: recompute_w_u reference fixture deviates from FLA spec.")
        sys.exit(1)
    print()
    print("ORACLE PASS: recompute_w_u reference fixture matches FLA spec ordering.")
    sys.exit(0)


if __name__ == "__main__":
    main()
