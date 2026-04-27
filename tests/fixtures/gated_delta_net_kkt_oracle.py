#!/usr/bin/env python3
"""Independent oracle for chunk_scaled_dot_kkt (Wave 5b.1 iter 2).

Wave 5b.1 iter 1.5 lesson: every new kernel needs a third independent
recipe to prevent fixture-and-kernel co-conspiracy bugs. This file is
that third opinion for the kkt kernel.

The reference fixture (gated_delta_net_kkt_reference.py) decomposes the
math as a per-K-tile loop and casts b_kb to bf16 right before the dot
(mirroring FLA line 86's `tl.dot(b_kb.to(b_k.dtype), trans(b_k))`). The
oracle below uses a DIFFERENT decomposition order that is mathematically
equivalent ONLY IF the spec is implemented correctly.

ORACLE RECIPE — NO loop over K-tiles:

    For each (b, h, i_t):
        kh   = i_h // GROUP_RATIO
        b_k  = k[b, t_chunk, kh, :]                           # [BT, K] bf16
        b_b  = beta[b, t_chunk, i_h]                          # [BT] f32
        b_g  = g[b, t_chunk, i_h]                             # [BT] f32

        # The bf16 round-trip on b_kb happens at FLA line 86, AFTER the
        # scale (line 85), BEFORE the dot. The dot is bf16 × bf16 → f32.
        # If the spec is correct, doing the cast on the FULL [BT, K]
        # matrix in one shot gives exactly the same result as doing it
        # per-K-tile (because (X * scalar).to(bf16) is elementwise).
        b_kb_full_f32  = b_k.float() * b_b[:, None]           # [BT, K] f32
        b_kb_full_bf16 = b_kb_full_f32.to(torch.bfloat16)     # FLA :86 cast
        # bf16 × bf16 → f32 accumulator via promote-to-f32-and-matmul.
        b_A = b_kb_full_bf16.float() @ b_k.float().T          # [BT, BT] f32

        # Apply gate AFTER the dot, then mask.
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * torch.exp(b_g_diff)
        m_A = (torch.arange(BT)[:, None] > torch.arange(BT)[None, :])
        b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))

The reference loops over K // BK tiles and accumulates b_A across iterations;
the oracle does it in one matmul. These are mathematically identical ONLY
if (a) the bf16 cast is on b_kb (post-scale, pre-dot), (b) the dot accumulates
in f32 (one shot or tile-by-tile), and (c) the gate multiply is post-dot
and elementwise (so it commutes with the K-tile partition sum). If any of
those properties is broken in the reference, this oracle disagrees.

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
    with open(os.path.join(HERE, "gated_delta_net_kkt_meta.json"), "r") as f:
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


def kkt_oracle(k, beta, g, B, T, Hg, H, K, BT):
    """Single-matmul recipe — no K-tile loop. Equivalent to spec iff
    (post-scale, pre-dot bf16 cast) and (post-dot gate multiply) hold."""
    NT = T // BT
    GROUP_RATIO = H // Hg
    A = torch.zeros((B, T, H, BT), dtype=torch.float32)

    o_t = torch.arange(BT)
    m_A = o_t[:, None] > o_t[None, :]   # strict lower

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = t_start + BT

                # Load full [BT, K] tile in one shot.
                b_k = k[b, t_start:t_end, kh, :]              # bf16 [BT, K]
                b_b = beta[b, t_start:t_end, i_h]             # f32  [BT]
                b_g = g[b, t_start:t_end, i_h]                # f32  [BT]

                # FLA :85-86 — scale in f32 (Triton promotes), then bf16 cast,
                # then bf16 × bf16 → f32 dot.
                b_kb = (b_k.float() * b_b[:, None]).to(torch.bfloat16)
                b_A = b_kb.float() @ b_k.float().T            # [BT, BT] f32

                # FLA :91-92 — gate multiply (exp(Δg)).
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_A = b_A * torch.exp(b_g_diff)

                # FLA :94-95 — strict lower mask.
                b_A = torch.where(m_A, b_A, torch.zeros_like(b_A))

                A[b, t_start:t_end, i_h, :] = b_A

    return A


def main():
    meta = load_meta()
    B = meta["B"]; T = meta["T"]; Hg = meta["Hg"]; H = meta["H"]
    K = meta["K"]; BT = meta["BT"]

    k = read_bf16(os.path.join(HERE, "gated_delta_net_kkt_input_k.bin"),
                  B * T * Hg * K).view(B, T, Hg, K)
    beta = read_f32(os.path.join(HERE, "gated_delta_net_kkt_input_beta.bin"),
                    B * T * H).view(B, T, H)
    g = read_f32(os.path.join(HERE, "gated_delta_net_kkt_input_g.bin"),
                 B * T * H).view(B, T, H)

    A_oracle = kkt_oracle(k, beta, g, B, T, Hg, H, K, BT)

    A_existing = read_f32(os.path.join(HERE, "gated_delta_net_kkt_A_ref.bin"),
                          B * T * H * BT).view(B, T, H, BT)

    max_err = (A_oracle - A_existing).abs().max().item()

    tol = 1e-6
    print("kkt oracle vs existing reference fixture:")
    print(f"  max_err = {max_err:.6e}   (tol {tol:.1e})")

    # Also sanity-check: per-chunk strict-lower-tri zero region must be 0.0.
    # A is [B, T, H, BT]; for each chunk i_t, rows are A[b, ts:te, h, :]
    # which gives a [BT, BT] block.
    NT = T // BT
    ref_upper = 0.0
    or_upper = 0.0
    o_t = torch.arange(BT)
    up_mask = o_t[:, None] <= o_t[None, :]   # diag + above
    for b in range(B):
        for h in range(H):
            for i_t in range(NT):
                ts, te = i_t * BT, (i_t + 1) * BT
                ref_upper = max(ref_upper, A_existing[b, ts:te, h, :][up_mask].abs().max().item())
                or_upper = max(or_upper, A_oracle[b, ts:te, h, :][up_mask].abs().max().item())
    print(f"  reference upper-tri max = {ref_upper:.3e}  (must be 0.0)")
    print(f"  oracle    upper-tri max = {or_upper:.3e}   (must be 0.0)")

    failed = (max_err > tol) or (ref_upper > 0.0) or (or_upper > 0.0)
    if failed:
        print()
        print("ORACLE FAIL: kkt reference fixture deviates from FLA spec.")
        sys.exit(1)
    print()
    print("ORACLE PASS: kkt reference fixture matches FLA spec ordering.")
    sys.exit(0)


if __name__ == "__main__":
    main()
