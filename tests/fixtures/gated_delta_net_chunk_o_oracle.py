#!/usr/bin/env python3
"""Independent oracle for chunk_fwd_o (Wave 5b.1 iter 3).

Wave 5b.1 iter 1.5 lesson + iter 2.5 strengthening: every new kernel needs a
third independent recipe whose math path is fundamentally different from the
reference, not just a notation reshuffle. The reference fixture
(`gated_delta_net_chunk_o_reference.py`) walks the (B, NT, H, NV, NK) loops
in Python with a per-tile K-tile accumulation, mirroring FLA's Triton kernel.
This oracle takes a different decomposition:

ORACLE RECIPE — fully-vectorised batched einsums, NO Python (b, h, i_t, i_v)
loops, NO K-tile loop, separate computation of the two terms:

    Term-1 (h-term):  o_h_term[b, t, i_h, v]
                      = sum_k q[b, t, kh, k] * h[b, NT_idx(t), i_h, v, k]
                      = einsum("btgK,bnhVK->bthV", q_expanded, h)[gather t→n]

      Implemented as: chunk q→[B, NT, BT, H, K], gather h chunk index,
      einsum over K alone, all chunks at once.

    Term-2 (A·v term): A[b, t_i, i_h, t_j] = q[b, t_i, kh] · k[b, t_j, kh]
      with t_i, t_j confined to the SAME chunk:
        A_chunked = einsum("bnhiK,bnhjK->bnhij", q_chunked_to_H, k_chunked_to_H)
      Then gate (post-dot, per FLA :120):
        A *= exp(g_chunked[..., None] - g_chunked[..., None, :])
      Then mask (`>=`, INCLUSIVE — NOT strict like kkt):
        A = where(t_i >= t_j, A, 0)
      Then bf16 cast on the FULL [B, NT, H, BT, BT] tensor (FLA :137 placement):
        A_bf16 = A.to(bf16)
      Then closing dot:
        o_a_term = einsum("bnhij,bnhjV->bnhiV", A_bf16, v_chunked)

    o = scale * (o_h_term * exp(g)[..., None] + o_a_term)

Equivalent to spec IFF:
  (a) the bf16 cast on b_A is on the FULL post-mask matrix (FLA :137),
  (b) the dot-with-v accumulator is f32 (einsum default with f32 inputs),
  (c) the mask is `>=` (causal+diag),
  (d) two `* scale` placements that combine into one global `* scale`,
  (e) chunk-locality on b_A (chunks are independent — FLA's i_t axis
      doesn't accumulate across chunks for output computation).

This oracle exits with code 1 if the existing reference output deviates from
the bit-exact-spec computation by more than 1e-6.

Critically: the reference walks K in BK-sized tiles AND walks V in BV-sized
tiles AND walks chunks sequentially with Python loops. The oracle does
ALL of (b, NT, h, BT, V, K) in batched contractions. A co-conspiracy bug
that walks K or V in the same tile order in both implementations would
diverge here — there is no tile alignment to share.
"""

# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))


def load_meta():
    with open(os.path.join(HERE, "gated_delta_net_chunk_o_meta.json"), "r") as f:
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


def chunk_fwd_o_oracle(q, k, v, h, g, scale, B, T, Hg, H, K, V, BT):
    """Vectorised einsum recipe — no Python (b, h, i_t, i_v, i_k) loops.

    Reshape strategy:
      - q_chunked, k_chunked: [B, NT, H, BT, K]   (Hg expanded into H by
                                                    repeat_interleave)
      - v_chunked:            [B, NT, H, BT, V]
      - h:                    [B, NT, H, V, K]
      - g_chunked:            [B, NT, H, BT]

    Then a single tensor-contraction kernel computes each term.
    """
    NT = T // BT
    GROUP_RATIO = H // Hg
    assert T == NT * BT
    assert H == Hg * GROUP_RATIO

    # Reshape q, k: [B, T, Hg, K] -> [B, NT, BT, Hg, K] -> [B, NT, Hg, BT, K]
    # -> [B, NT, H, BT, K] via repeat_interleave on the Hg axis.
    q_chunked = q.reshape(B, NT, BT, Hg, K).permute(0, 1, 3, 2, 4).contiguous()
    q_chunked = q_chunked.repeat_interleave(GROUP_RATIO, dim=2)   # [B, NT, H, BT, K]

    k_chunked = k.reshape(B, NT, BT, Hg, K).permute(0, 1, 3, 2, 4).contiguous()
    k_chunked = k_chunked.repeat_interleave(GROUP_RATIO, dim=2)   # [B, NT, H, BT, K]

    # v: [B, T, H, V] -> [B, NT, BT, H, V] -> [B, NT, H, BT, V]
    v_chunked = v.reshape(B, NT, BT, H, V).permute(0, 1, 3, 2, 4).contiguous()

    # g: [B, T, H] -> [B, NT, BT, H] -> [B, NT, H, BT]
    g_chunked = g.reshape(B, NT, BT, H).permute(0, 1, 3, 2).contiguous()

    # h is already [B, NT, H, V, K].
    assert h.shape == (B, NT, H, V, K)

    # ----------------------------------------------------------------------
    # Term 1: o_h = einsum("bnhiK,bnhVK->bnhiV", q_chunked, h) — one shot.
    # Then scale by exp(g) per chunk row.
    # ----------------------------------------------------------------------
    # f32 einsum; bf16 inputs are promoted via .float() (lossless).
    o_h = torch.einsum(
        "bnhiK,bnhVK->bnhiV",
        q_chunked.float(),
        h.float(),
    )                                                              # [B, NT, H, BT, V] f32
    o_h = o_h * torch.exp(g_chunked).unsqueeze(-1)                 # broadcast on V

    # ----------------------------------------------------------------------
    # Term 2: A[b, n, h, i, j] = q_chunked[..., i, :] · k_chunked[..., j, :]
    # All chunks, all heads in one batched contraction.
    # ----------------------------------------------------------------------
    A = torch.einsum(
        "bnhiK,bnhjK->bnhij",
        q_chunked.float(),
        k_chunked.float(),
    )                                                              # [B, NT, H, BT, BT]

    # Gate (post-dot, FLA :120).
    A = A * torch.exp(g_chunked.unsqueeze(-1) - g_chunked.unsqueeze(-2))

    # Causal+diag mask (`>=`, INCLUSIVE; FLA :124).
    o_t = torch.arange(BT)
    m_A = (o_t[:, None] >= o_t[None, :])                           # [BT, BT] bool
    A = torch.where(m_A, A, torch.zeros_like(A))                   # broadcast

    # bf16 round-trip (FLA :137 — full-tensor cast, post-mask).
    A_bf16 = A.to(torch.bfloat16)

    # Closing dot — A · v.
    o_a = torch.einsum(
        "bnhij,bnhjV->bnhiV",
        A_bf16.float(),
        v_chunked.float(),
    )                                                              # [B, NT, H, BT, V]

    # ----------------------------------------------------------------------
    # Combine: o = scale * (o_h + o_a). Two `* scale` in spec, but
    # mathematically a single scalar — we apply once.
    # ----------------------------------------------------------------------
    o_chunked = scale * (o_h + o_a)                                # [B, NT, H, BT, V] f32

    # Reshape back to [B, T, H, V] and cast to bf16.
    o = o_chunked.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(torch.bfloat16).contiguous()
    return o


def main():
    meta = load_meta()
    B = meta["B"]; T = meta["T"]; Hg = meta["Hg"]; H = meta["H"]
    K = meta["K"]; V = meta["V"]; BT = meta["BT"]; NT = meta["NT"]
    scale = meta["scale"]

    q = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_o_input_q.bin"),
                  B * T * Hg * K).view(B, T, Hg, K)
    k = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_o_input_k.bin"),
                  B * T * Hg * K).view(B, T, Hg, K)
    v = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_o_input_v.bin"),
                  B * T * H * V).view(B, T, H, V)
    h = read_bf16(os.path.join(HERE, "gated_delta_net_chunk_o_input_h.bin"),
                  B * NT * H * V * K).view(B, NT, H, V, K)
    g = read_f32(os.path.join(HERE, "gated_delta_net_chunk_o_input_g.bin"),
                 B * T * H).view(B, T, H)

    o_oracle = chunk_fwd_o_oracle(q, k, v, h, g, scale, B, T, Hg, H, K, V, BT)

    o_existing = read_bf16(
        os.path.join(HERE, "gated_delta_net_chunk_o_o_ref.bin"),
        B * T * H * V).view(B, T, H, V)

    max_err = (o_oracle.float() - o_existing.float()).abs().max().item()

    tol = 1e-6
    print("chunk_fwd_o oracle vs existing reference fixture:")
    print(f"  max_o_err = {max_err:.6e}   (tol {tol:.1e})")

    failed = max_err > tol
    if failed:
        print()
        print("ORACLE FAIL: chunk_fwd_o reference fixture deviates from FLA spec.")
        sys.exit(1)
    print()
    print("ORACLE PASS: chunk_fwd_o reference fixture matches FLA spec ordering.")
    sys.exit(0)


if __name__ == "__main__":
    main()
