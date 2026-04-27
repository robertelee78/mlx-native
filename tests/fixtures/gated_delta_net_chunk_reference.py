#!/usr/bin/env python3
"""Reference fixture generator for the chunk-parallel inter-chunk
state-recurrence kernel (Wave 5b.1 iter 1).

Produces the bit-for-bit reference output for
`kernel_gated_delta_net_chunk_inter_state_bf16`, which is a
straight port of FLA's
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
(/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-298).

This is a pure-PyTorch re-implementation of that same algorithm; we do NOT
import FLA (it's bundled inside vLLM behind heavy distributed deps that
won't load standalone). The math is straight from chunk_delta_h.py:84-298,
translated 1:1 to PyTorch vector ops. The sequential T-loop is identical
to the Triton kernel's i_t loop, with the same f32 accumulation policy
(b_h1..b_h4 are f32; b_w/b_v/b_k are bf16 for storage but cast to f32
for the dots).

Inputs (the inter-chunk kernel takes post-WY-projection w & u as inputs;
we generate them synthetically — this isolates step 5 of the chunk pipeline):

  k:    [B, T, Hg, K]   bf16   — per-token K (post-L2norm if applicable)
  w:    [B, T, H,  K]   bf16   — WY-projected K (= T_n_full @ (β·k·exp(G_n)))
  u:    [B, T, H,  V]   bf16   — WY-projected V (= T_n_full @ (β·v))
  g:    [B, T, H]       f32    — log-decay accumulator (already cumsumed)
  h0:   [B, H, V, K]    f32    — initial state (zero in this fixture)

Outputs:

  h:           [B, NT, H, V, K]  bf16  — state at start of each chunk
  v_new:       [B, T, H, V]      bf16  — `b_v` after subtracting w·h^T
  final_state: [B, H, V, K]      f32   — state after last chunk

Shape: B=1, T=128, Hg=2, H=4, K=128, V=128, BT=64 → NT=2.

Determinism: fixed seed 0xC0FFEE on torch.Generator (CPU).

The fixture writes:
  gated_delta_net_chunk_state_ref.bin  — h[B, NT, H, V, K] bf16 raw bytes
  gated_delta_net_chunk_v_new_ref.bin  — v_new[B, T, H, V] bf16 raw bytes
  gated_delta_net_chunk_final_ref.bin  — final_state[B, H, V, K] f32 raw bytes
  gated_delta_net_chunk_inputs.bin     — concatenated bf16 inputs:
       offset 0      : k[B,T,Hg,K]
       offset k_end  : w[B,T,H,K]
       offset w_end  : u[B,T,H,V]
       offset u_end  : g[B,T,H] (f32)
  gated_delta_net_chunk_meta.json      — shape/dtype/seed/version metadata

Reference: arXiv 2412.06464 §4 (chunkwise parallelization);
FLA chunk_delta_h.py:43-298 (canonical Triton kernel; we implement the
identical math without using Triton).
"""

# SPDX-License-Identifier: Apache-2.0
# Implementation reference: flash-linear-attention (FLA)
# `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at
# /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-298,
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT license).
# The math here is a direct re-derivation of the recurrence printed in
# arXiv 2412.06464 §4, validated against the Triton kernel's structure.

import json
import os
import struct
import sys

import torch


# ---------------------------------------------------------------------------
# Constants (kept identical to FLA defaults; matches llama.cpp's CS=64).
# ---------------------------------------------------------------------------
FIXTURE_VERSION = "wave5b.1-iter1-v1"
SEED = 0xC0FFEE
B = 1            # batch
T = 128          # seq len (must be a multiple of BT for this fixture)
Hg = 2           # K-head count
H = 4            # V-head count (Hg divides H; group_ratio = H // Hg = 2)
K = 128          # head_dim_k
V = 128          # head_dim_v
BT = 64          # chunk size (= FLA_CHUNK_SIZE)

assert T % BT == 0, "fixture assumes T multiple of BT (no padding edge case)"
NT = T // BT
assert H % Hg == 0
GROUP_RATIO = H // Hg


# ---------------------------------------------------------------------------
# Reference implementation — direct PyTorch port of chunk_delta_h.py:84-298.
# ---------------------------------------------------------------------------
def chunk_h_reference(
    k: torch.Tensor,    # [B, T, Hg, K] bf16
    w: torch.Tensor,    # [B, T, H,  K] bf16
    u: torch.Tensor,    # [B, T, H,  V] bf16
    g: torch.Tensor,    # [B, T, H]     f32  (log-decay, already cumsumed
                        #                     within each chunk via local cumsum
                        #                     producing g_cumsum)
    h0: torch.Tensor,   # [B, H, V, K]  f32  (initial state)
    BT: int,
):
    """Inter-chunk state recurrence — the kernel with no Apple Metal precedent.

    Performs steps equivalent to FLA's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`:

    For each batch sample × head:
        b_h := h0  (f32 [V, K])
        for chunk i_t in 0..NT:
            store b_h to h[i_n, i_t, i_h]
            b_w  := w[bos+i_t*BT : bos+(i_t+1)*BT, i_h, :]   # [BT, K] bf16
            b_v  := u[bos+i_t*BT : bos+(i_t+1)*BT, i_h, :]
                  - b_w @ b_h^T                              # [BT, V] bf16
            store b_v -> v_new[bos+i_t*BT:..., i_h]

            last  = (i_t+1)*BT - 1
            g_last = g[bos+last, i_h]                        # f32
            g_blk  = g[bos+i_t*BT:..., i_h]                  # f32 [BT]
            b_v   *= exp(g_last - g_blk)[:, None]            # mask zero past T
            b_h   *= exp(g_last)
            b_k   := k[bos+i_t*BT:..., i_h // GROUP_RATIO, :]  # [BT, K] bf16
            b_h   += b_k.T @ b_v                             # outer accumulate

        store b_h -> final_state[i_h]

    All dot products go bf16 -> f32 accum -> stay in f32 for the running b_h.
    """
    assert k.shape == (B, T, Hg, K), f"k {tuple(k.shape)} != ({B},{T},{Hg},{K})"
    assert w.shape == (B, T, H, K), f"w {tuple(w.shape)} != ({B},{T},{H},{K})"
    assert u.shape == (B, T, H, V), f"u {tuple(u.shape)} != ({B},{T},{H},{V})"
    assert g.shape == (B, T, H), f"g {tuple(g.shape)} != ({B},{T},{H})"
    assert h0.shape == (B, H, V, K), f"h0 {tuple(h0.shape)} != ({B},{H},{V},{K})"
    assert k.dtype == torch.bfloat16
    assert w.dtype == torch.bfloat16
    assert u.dtype == torch.bfloat16
    assert g.dtype == torch.float32
    assert h0.dtype == torch.float32

    NT_local = T // BT

    h_out = torch.zeros((B, NT_local, H, V, K), dtype=torch.bfloat16)
    v_new = torch.zeros((B, T, H, V), dtype=torch.bfloat16)
    final_state = torch.zeros((B, H, V, K), dtype=torch.float32)

    for b in range(B):
        for i_h in range(H):
            kh = i_h // GROUP_RATIO
            b_h = h0[b, i_h].clone()  # [V, K] f32

            for i_t in range(NT_local):
                t_start = i_t * BT
                t_end = t_start + BT

                # 1. Snapshot b_h at chunk start.  (bf16 store; f32 keep.)
                h_out[b, i_t, i_h] = b_h.to(torch.bfloat16)

                # 2. b_v = u_chunk - w_chunk @ b_h^T   (post-WY new V)
                b_w = w[b, t_start:t_end, i_h, :].float()        # [BT, K]
                b_u = u[b, t_start:t_end, i_h, :].float()        # [BT, V]
                # b_v = u - w @ h^T -> [BT, V]  (f32 accum; bf16 store)
                b_v_f32 = b_u - b_w @ b_h.T                      # [BT, V]
                v_new[b, t_start:t_end, i_h] = b_v_f32.to(torch.bfloat16)

                # 3. Apply gate to b_v and b_h.  (FLA chunk_delta_h.py:205-221)
                last = t_end - 1
                g_last = g[b, last, i_h]                          # f32 scalar
                g_blk = g[b, t_start:t_end, i_h]                  # f32 [BT]
                # Note: chunk_delta_h.py masks elements past T with `0`, but
                # T is a multiple of BT here so no masking needed.
                # Cast b_v -> bf16 -> f32 (matches `b_v = b_v.to(k.dtype)`
                # at line 255, then f32 reduction in the dot below).
                b_v_f32 = b_v_f32.to(torch.bfloat16).float()
                b_v_f32 = b_v_f32 * torch.exp(g_last - g_blk).unsqueeze(-1)
                b_h = b_h * torch.exp(g_last)

                # 4. b_h += k^T @ b_v   (outer accumulate)
                b_k = k[b, t_start:t_end, kh, :].float()         # [BT, K]
                # Cast b_v back to bf16 (per chunk_delta_h.py:255), then mat dot
                # gives outer in K x V layout:
                #   tl.dot(b_k_KxBT, b_v_BTxV) -> [K, V]  ;  trans -> [V, K]
                b_v_bf16 = b_v_f32.to(torch.bfloat16).float()
                # b_h is [V, K]; we want b_h += (k^T @ v_new)^T = v_new^T @ k
                b_h = b_h + b_v_bf16.T @ b_k                     # [V, K]

            final_state[b, i_h] = b_h

    return h_out, v_new, final_state


# ---------------------------------------------------------------------------
# Generate inputs deterministically.
# ---------------------------------------------------------------------------
def make_inputs():
    g_rng = torch.Generator(device="cpu")
    g_rng.manual_seed(SEED)

    # Random k, w, u in bf16 — small magnitudes to keep f32 accumulators stable.
    k = (torch.randn((B, T, Hg, K), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    w = (torch.randn((B, T, H, K), generator=g_rng, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    u = (torch.randn((B, T, H, V), generator=g_rng, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    # g: log-decay (cumsumed within each chunk).  Generate raw per-token g
    # in the small negative range (so exp(g) is in (0, 1]), then cumsum
    # within chunks.  (FLA's input is g_cumsum already; we mirror that.)
    # `torch.empty` does not accept a generator — use rand shifted into [-0.05, -0.005].
    g_raw = torch.rand((B, T, H), generator=g_rng, dtype=torch.float32) * 0.045 - 0.05
    g = g_raw.clone()
    for i_t in range(NT):
        s, e = i_t * BT, (i_t + 1) * BT
        g[:, s:e, :] = torch.cumsum(g_raw[:, s:e, :], dim=1)

    h0 = torch.zeros((B, H, V, K), dtype=torch.float32)

    return k, w, u, g, h0


def write_bin(path, t):
    """Write tensor's raw little-endian bytes."""
    arr = t.contiguous()
    if arr.dtype == torch.bfloat16:
        # bf16 = 2 bytes; PyTorch stores as truncated f32 high-bits,
        # which is the IEEE bf16 layout -> write as i16-pun bytes.
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

    k, w, u, g, h0 = make_inputs()
    h_out, v_new, final_state = chunk_h_reference(k, w, u, g, h0, BT)

    # Write reference outputs.
    write_bin(os.path.join(here, "gated_delta_net_chunk_state_ref.bin"), h_out)
    write_bin(os.path.join(here, "gated_delta_net_chunk_v_new_ref.bin"), v_new)
    write_bin(os.path.join(here, "gated_delta_net_chunk_final_ref.bin"), final_state)

    # Write input tensors so the Rust test loads identical bytes.
    write_bin(os.path.join(here, "gated_delta_net_chunk_input_k.bin"), k)
    write_bin(os.path.join(here, "gated_delta_net_chunk_input_w.bin"), w)
    write_bin(os.path.join(here, "gated_delta_net_chunk_input_u.bin"), u)
    write_bin(os.path.join(here, "gated_delta_net_chunk_input_g.bin"), g)
    write_bin(os.path.join(here, "gated_delta_net_chunk_input_h0.bin"), h0)

    # Sidecar: shape/dtype/seed/version metadata.
    meta = {
        "version": FIXTURE_VERSION,
        "seed": SEED,
        "B": B,
        "T": T,
        "Hg": Hg,
        "H": H,
        "K": K,
        "V": V,
        "BT": BT,
        "NT": NT,
        "group_ratio": GROUP_RATIO,
        "fla_reference": (
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64 at "
            "/opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-298"
        ),
        "fla_commit_note": (
            "vendored copy in /opt/vllm; original FLA repo "
            "github.com/sustcsonglin/flash-linear-attention. "
            "vllm /opt/vllm head: "
            + os.popen("cd /opt/vllm && git rev-parse --short HEAD 2>/dev/null").read().strip()
        ),
        "inputs": {
            "k":  {"shape": [B, T, Hg, K], "dtype": "bf16",
                   "file":  "gated_delta_net_chunk_input_k.bin"},
            "w":  {"shape": [B, T, H,  K], "dtype": "bf16",
                   "file":  "gated_delta_net_chunk_input_w.bin"},
            "u":  {"shape": [B, T, H,  V], "dtype": "bf16",
                   "file":  "gated_delta_net_chunk_input_u.bin"},
            "g":  {"shape": [B, T, H],     "dtype": "f32",
                   "file":  "gated_delta_net_chunk_input_g.bin"},
            "h0": {"shape": [B, H, V, K],  "dtype": "f32",
                   "file":  "gated_delta_net_chunk_input_h0.bin"},
        },
        "outputs": {
            "h":           {"shape": [B, NT, H, V, K], "dtype": "bf16",
                            "file":  "gated_delta_net_chunk_state_ref.bin"},
            "v_new":       {"shape": [B, T, H, V],     "dtype": "bf16",
                            "file":  "gated_delta_net_chunk_v_new_ref.bin"},
            "final_state": {"shape": [B, H, V, K],     "dtype": "f32",
                            "file":  "gated_delta_net_chunk_final_ref.bin"},
        },
        "tolerance": {
            "h_atol":     5e-3,  "h_rtol":     2e-2,
            "v_new_atol": 5e-3,  "v_new_rtol": 2e-2,
            "final_atol": 5e-3,  "final_rtol": 2e-2,
        },
    }
    with open(os.path.join(here, "gated_delta_net_chunk_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote fixtures to {here}")
    print(f"  h_out shape  = {tuple(h_out.shape)}      ({h_out.numel() * 2} bytes)")
    print(f"  v_new shape  = {tuple(v_new.shape)}        ({v_new.numel() * 2} bytes)")
    print(f"  final shape  = {tuple(final_state.shape)} ({final_state.numel() * 4} bytes)")
    # Quick stat sanity print.
    print(f"  h.abs().max()       = {h_out.float().abs().max().item():.6f}")
    print(f"  v_new.abs().max()   = {v_new.float().abs().max().item():.6f}")
    print(f"  final.abs().max()   = {final_state.abs().max().item():.6f}")


if __name__ == "__main__":
    main()
