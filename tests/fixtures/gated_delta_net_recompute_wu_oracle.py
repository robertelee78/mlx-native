#!/usr/bin/env python3
"""Independent oracle for recompute_w_u_fwd (Wave 5b.1 iter 2.5).

Wave 5b.1 iter 1.5 lesson: every new kernel needs a third independent
recipe to prevent fixture-and-kernel co-conspiracy bugs. Wave 5b.1
iter 2 audit (Codex, med-sev): the prior oracle removed BV/BK output-
column tiling but still used the same per-(b, h, chunk) Python loop as
the reference, so a co-conspiracy bug along the chunk-recurrence axis
would have evaded detection. This iter-2.5 rewrite goes further:

ORACLE RECIPE — single batched einsum over (B, NT, H), NO Python loops:

    1. Reshape A         [B, T, H, BT]  -> [B, NT, H, BT, BT]    (chunk axis NT)
    2. Reshape beta      [B, T, H]      -> [B, NT, H, BT]
    3. Reshape v         [B, T, H, V]   -> [B, NT, H, BT, V]
    4. Reshape k         [B, T, Hg, K]  -> [B, NT, Hg, BT, K]    (broadcast over GROUP_RATIO into H)
    5. Reshape g         [B, T, H]      -> [B, NT, H, BT]; b_g = exp(g_chunk)
    6. b_vb = (v_chunk.float() * beta_chunk[..., None]).to(bf16)        # [B, NT, H, BT, V]
       b_kb = (k_chunk.float() * beta_chunk[..., None]
              * b_g[..., None]).to(bf16)                                # [B, NT, H, BT, K]
       u    = einsum("bnhij,bnhjV->bnhiV", A_chunk_f32, b_vb_f32)        # [B, NT, H, BT, V]
       w    = einsum("bnhij,bnhjK->bnhiK", A_chunk_f32, b_kb_f32)        # [B, NT, H, BT, K]
    7. Reshape u, w back to [B, T, H, V] / [B, T, H, K], cast to bf16.

This decomposition is mathematically identical to the reference IFF:
    (a) the bf16 cast is on the scaled operand (post-scale, pre-dot),
    (b) the dot accumulator is f32 (einsum default with f32 inputs),
    (c) no chunk-recurrence — chunks are independent (FLA wy_fast.py:91-94
        and :112-116 make `b_w` / `b_u` purely a function of the i_t-th
        chunk's b_A, b_beta, b_g, b_v / b_k).

If any of those properties is broken in the reference, this einsum
disagrees. Crucially, going from a per-chunk Python `for i_t in range(NT)`
to a single batched einsum exercises a fundamentally different math path:
the reference visits chunks sequentially through the NT loop; the oracle
fires every chunk in one tensor-contraction kernel. A co-conspiracy bug
that walks the chunk axis the same way in both would now diverge.

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
    """Vectorized einsum recipe — NO Python loops over (b, h, i_t).

    Different math path from the reference's per-chunk Python loop. References
    FLA wy_fast.py:91-94 (post-scale bf16 cast on b_vb, then bf16 × f32 dot)
    and :112-116 (same pattern on b_kb with extra b_g multiplier).
    """
    NT = T // BT
    GROUP_RATIO = H // Hg
    assert T == NT * BT
    assert H == Hg * GROUP_RATIO

    # 1. Chunk along the time axis: [B, T, ...] -> [B, NT, BT, ...] -> [B, NT, ..., BT, ...]
    #    For A:    [B, T, H, BT] -> [B, NT, H, BT, BT]
    A_chunk = A.reshape(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4).contiguous()  # [B, NT, H, BT, BT]

    # 2. beta: [B, T, H] -> [B, NT, H, BT]
    beta_chunk = beta.reshape(B, NT, BT, H).permute(0, 1, 3, 2).contiguous()    # [B, NT, H, BT]

    # 3. v: [B, T, H, V] -> [B, NT, H, BT, V]
    v_chunk = v.reshape(B, NT, BT, H, V).permute(0, 1, 3, 2, 4).contiguous()    # [B, NT, H, BT, V] bf16

    # 4. k: [B, T, Hg, K] -> [B, NT, H, BT, K] (broadcast Hg -> H via GROUP_RATIO).
    k_chunk = k.reshape(B, NT, BT, Hg, K).permute(0, 1, 3, 2, 4).contiguous()   # [B, NT, Hg, BT, K]
    # Expand Hg into H by repeat_interleave on the head axis.
    k_chunk = k_chunk.repeat_interleave(GROUP_RATIO, dim=2)                     # [B, NT, H, BT, K]

    # 5. g: [B, T, H] -> [B, NT, H, BT], then exp.
    g_chunk = g_cumsum.reshape(B, NT, BT, H).permute(0, 1, 3, 2).contiguous()   # [B, NT, H, BT]
    b_g = torch.exp(g_chunk)                                                    # [B, NT, H, BT]

    # 6. Post-scale bf16 cast (FLA :92, :114).
    b_vb_f32 = v_chunk.float() * beta_chunk[..., None]                           # [B, NT, H, BT, V] f32
    b_vb_bf16 = b_vb_f32.to(torch.bfloat16)                                      # FLA :92 cast

    b_kb_f32 = k_chunk.float() * beta_chunk[..., None] * b_g[..., None]          # [B, NT, H, BT, K] f32
    b_kb_bf16 = b_kb_f32.to(torch.bfloat16)                                      # FLA :114 cast

    # 7. Batched einsum — one contraction kernel for ALL (B, NT, H) at once.
    #    A: [B, NT, H, BT_i, BT_j]   b_vb: [B, NT, H, BT_j, V]   -> u: [B, NT, H, BT_i, V]
    u_chunk = torch.einsum("bnhij,bnhjV->bnhiV", A_chunk, b_vb_bf16.float())     # [B, NT, H, BT, V] f32
    w_chunk = torch.einsum("bnhij,bnhjK->bnhiK", A_chunk, b_kb_bf16.float())     # [B, NT, H, BT, K] f32

    # 8. Reshape back to [B, T, H, V/K] and cast to bf16.
    u = u_chunk.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(torch.bfloat16).contiguous()
    w = w_chunk.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(torch.bfloat16).contiguous()

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
