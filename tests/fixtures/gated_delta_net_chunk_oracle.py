#!/usr/bin/env python3
"""Independent FLA-line-255 oracle for the chunk-parallel inter-chunk
state-recurrence kernel.

Wave 5b.1 iter 1.5 — Codex iter-1 audit caught that both the Metal kernel
(src/shaders/gated_delta_net_chunk.metal) and the Python reference fixture
(tests/fixtures/gated_delta_net_chunk_reference.py) place the bf16 round-trip
on b_v BEFORE the gate multiply, while FLA's canonical kernel
(/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py) places it
AFTER the gate multiply, BEFORE the outer dot.

Both being wrong in the same way means the existing fixture-vs-GPU test
passes on a co-incident-incorrect answer. This file is a THIRD independent
recipe: a literal transcription of FLA's chunk_delta_h.py:200-261 ordering,
written without reference to the existing fixture's intermediate
expressions. It loads the fixture's input .bin files (the inputs are
spec-correct; only the reference outputs are bf16-misordered) and compares
against the existing reference output .bin files.

Spec — FLA chunk_delta_h.py line ordering (canonical):

    line 213: b_v = b_v * exp(b_g_last - b_g)[:, None]   # gate b_v in f32
    line 215: b_h1 *= b_g_last                            # gate b_h in f32
    line 255: b_v = b_v.to(k.dtype.element_ty)            # bf16 cast
    line 261: b_h1 += tl.trans(tl.dot(b_k, b_v))          # outer dot uses bf16 b_v

There is NO pre-gate bf16 cast on b_v anywhere in chunk_delta_h.py:200-261.
The pre-gate cast in gated_delta_net_chunk_reference.py:171 is an iter-1
implementation accident.

This oracle exits with code 1 if the existing reference outputs deviate
from the bit-exact-spec computation by more than 1e-6.
"""

# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))


def load_meta():
    with open(os.path.join(HERE, "gated_delta_net_chunk_meta.json"), "r") as f:
        return json.load(f)


def read_bf16(path: str, n_elems: int) -> torch.Tensor:
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n_elems * 2, f"{path}: {len(raw)} bytes != {n_elems * 2}"
    # bf16 layout in PyTorch is i16-puned high bits of f32; view back as bf16.
    flat = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    return flat.view(torch.bfloat16).clone()


def read_f32(path: str, n_elems: int) -> torch.Tensor:
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n_elems * 4, f"{path}: {len(raw)} bytes != {n_elems * 4}"
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return flat.clone()


def chunk_h_oracle(k, w, u, g, h0, B, T, Hg, H, K, V, BT):
    """Spec-faithful FLA chunk_delta_h.py:200-261 transcription.

    Inputs match the fixture's input layout exactly. NO pre-gate bf16
    round-trip on b_v — the only bf16 cast on b_v is at FLA line 255,
    which sits AFTER the gate multiply (line 213) and BEFORE the outer
    dot (line 261).
    """
    NT = T // BT
    GROUP_RATIO = H // Hg

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

                # FLA chunk_delta_h.py ~line 95-100 equivalent: snapshot b_h.
                h_out[b, i_t, i_h] = b_h.to(torch.bfloat16)

                # b_v = u_chunk - w_chunk @ b_h^T (f32 accumulation).
                b_w = w[b, t_start:t_end, i_h, :].float()       # [BT, K]
                b_u = u[b, t_start:t_end, i_h, :].float()       # [BT, V]
                b_v = b_u - b_w @ b_h.T                          # [BT, V] f32

                # FLA line 199-203 equivalent: store v_new BEFORE gate.
                v_new[b, t_start:t_end, i_h] = b_v.to(torch.bfloat16)

                # FLA line 205-208: scalar g_last; line 213: gate b_v in f32.
                # NO pre-gate bf16 cast (this is the entire point of the oracle).
                last = t_end - 1
                g_last = g[b, last, i_h]                         # f32 scalar
                g_blk = g[b, t_start:t_end, i_h]                 # f32 [BT]
                b_v = b_v * torch.exp(g_last - g_blk).unsqueeze(-1)   # FLA :213

                # FLA line 215: gate b_h in f32.
                b_h = b_h * torch.exp(g_last)                    # FLA :215

                # FLA line 255: bf16 cast b_v AFTER gate, BEFORE outer dot.
                b_v_bf16 = b_v.to(torch.bfloat16)                # FLA :255

                # FLA line 257-261: outer dot in (bf16 K) × (bf16 V) -> f32 acc.
                b_k = k[b, t_start:t_end, kh, :]                  # bf16 [BT, K]
                # PyTorch bf16 @ bf16 returns bf16; we want f32 accumulator,
                # so we promote the bf16 operands to f32 (the elements are
                # already bf16-quantized — promotion is lossless of the bf16
                # representation, mirroring `tl.dot(bf16, bf16, out=f32)`).
                b_h = b_h + b_v_bf16.float().T @ b_k.float()      # FLA :261

            final_state[b, i_h] = b_h

    return h_out, v_new, final_state


def main():
    meta = load_meta()
    B = meta["B"]; T = meta["T"]; Hg = meta["Hg"]; H = meta["H"]
    K = meta["K"]; V = meta["V"]; BT = meta["BT"]; NT = meta["NT"]

    # Load existing input fixtures (spec-correct).
    k = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_input_k.bin"),
                  B * T * Hg * K).view(B, T, Hg, K)
    w = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_input_w.bin"),
                  B * T * H * K).view(B, T, H, K)
    u = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_input_u.bin"),
                  B * T * H * V).view(B, T, H, V)
    g = read_f32(os.path.join(HERE, "gated_delta_net_chunk_input_g.bin"),
                 B * T * H).view(B, T, H)
    h0 = read_f32(os.path.join(HERE, "gated_delta_net_chunk_input_h0.bin"),
                  B * H * V * K).view(B, H, V, K)

    # Compute spec-faithful reference.
    h_oracle, v_new_oracle, final_oracle = chunk_h_oracle(
        k, w, u, g, h0, B, T, Hg, H, K, V, BT
    )

    # Load existing reference outputs (potentially bug-co-incident).
    h_existing = read_bf16(
        os.path.join(HERE, "gated_delta_net_chunk_state_ref.bin"),
        B * NT * H * V * K,
    ).view(B, NT, H, V, K)
    v_new_existing = read_bf16(
        os.path.join(HERE, "gated_delta_net_chunk_v_new_ref.bin"),
        B * T * H * V,
    ).view(B, T, H, V)
    final_existing = read_f32(
        os.path.join(HERE, "gated_delta_net_chunk_final_ref.bin"),
        B * H * V * K,
    ).view(B, H, V, K)

    # Compare: oracle vs existing reference.
    max_h_err = (h_oracle.float() - h_existing.float()).abs().max().item()
    max_v_err = (v_new_oracle.float() - v_new_existing.float()).abs().max().item()
    max_final_err = (final_oracle - final_existing).abs().max().item()

    tol = 1e-6
    print("FLA-line-255 oracle vs existing reference fixture:")
    print(f"  max_h_err     = {max_h_err:.6e}   (tol {tol:.1e})")
    print(f"  max_v_err     = {max_v_err:.6e}   (tol {tol:.1e})")
    print(f"  max_final_err = {max_final_err:.6e}   (tol {tol:.1e})")

    failed = (max_h_err > tol) or (max_v_err > tol) or (max_final_err > tol)
    if failed:
        print()
        print("ORACLE FAIL: existing reference fixture deviates from "
              "FLA chunk_delta_h.py:200-261 spec ordering.")
        print("Likely cause: pre-gate bf16 round-trip on b_v "
              "(see gated_delta_net_chunk_reference.py:171 — should not exist).")
        sys.exit(1)
    else:
        print()
        print("ORACLE PASS: reference fixture matches FLA spec ordering.")
        sys.exit(0)


if __name__ == "__main__":
    main()
