#!/usr/bin/env python3
"""Independent reference fixture for chunk_tri_solve_invert (Wave 5b.1 iter 4.5 T1).

This fixture exists to give the dispatch_chunk_tri_solve_invert kernel a DIRECT
unit test asserting the (I + A_strict)^-1 semantics, completely orthogonal to
the end-to-end chunk_gated_delta_rule_fwd path.

Codex iter-4 audit (missed-test): the chunk_tri_solve_invert kernel had only
end-to-end coverage. A sign-flip ((I - A)^-1 vs (I + A)^-1) or stride bug
could be masked by downstream stages. This fixture closes that gap.

Spec source:
  - FLA solve_tril semantics:
    /opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:506-530
    "Compute the inverse of the matrix I + A. A should be strictly lower
     triangular, i.e., A.triu() == 0."

Math: For each [BT, BT] strict-lower matrix A_strict (zeros on diagonal and
above), the kernel computes A_inv = numpy.linalg.inv(I + A_strict).

We use BT=64 (FIXED_BT in the kernel), B=4 (4 independent random matrices),
H=1, T=BT=64 (NT=1 chunk per batch, simplest geometry).

Layout matches the kernel's [B, T, H, BT] = [4, 64, 1, 64] f32 layout. With
H=1, NT=1 the strides collapse to: A_strict[b, i, 0, j] at index
(b*64 + i)*1*64 + j = b*4096 + i*64 + j.

Random seed: 0xC0FFEE (deterministic). Magnitudes are bounded so I+A is
well-conditioned (we want to test correctness, not numerical stability of
forward substitution).

Outputs:
  chunk_tri_solve_invert_input_a_strict.bin : [B, T, H, BT] f32 (BT=64, B=4, H=1, T=64)
  chunk_tri_solve_invert_a_inv_ref.bin      : same layout, reference (I + A_strict)^-1

Plus a json metadata file with B, T, H, BT.
"""

# SPDX-License-Identifier: Apache-2.0
# Reference: numpy.linalg.inv (LAPACK getrf+getri, completely independent
# of the kernel's hand-rolled forward-substitution algorithm).

import json
import os

import numpy as np


SEED = 0xC0FFEE
B = 4
T = 64
H = 1
BT = 64


def make_strict_lower_inputs(rng):
    """Generate B random strict-lower-triangular [BT, BT] matrices.

    Magnitudes bounded to ~0.1 so I+A is diagonally dominant (well-conditioned).
    """
    a_strict = np.zeros((B, T, H, BT), dtype=np.float32)
    for b in range(B):
        # Random [BT, BT] f32, then mask to strict-lower (zero diag + upper).
        m = rng.standard_normal((BT, BT)).astype(np.float32) * 0.1
        m_strict = np.tril(m, k=-1)  # strict lower (k=-1 zeros diag)
        # Place into [T, H, BT] = [64, 1, 64] block; T spans the BT rows.
        a_strict[b, :, 0, :] = m_strict
    return a_strict


def reference_inv(a_strict):
    """Compute A_inv = (I + A_strict)^-1 per [BT, BT] block via numpy.linalg.inv."""
    a_inv = np.zeros_like(a_strict)
    for b in range(B):
        m_strict = a_strict[b, :, 0, :]    # [BT, BT]
        m_full = np.eye(BT, dtype=np.float32) + m_strict
        m_inv = np.linalg.inv(m_full).astype(np.float32)
        a_inv[b, :, 0, :] = m_inv
    return a_inv


def main():
    rng = np.random.default_rng(SEED)

    a_strict = make_strict_lower_inputs(rng)
    a_inv_ref = reference_inv(a_strict)

    # Sanity: assert strict-lower property holds on input.
    for b in range(B):
        block = a_strict[b, :, 0, :]
        # diagonal must be zero
        assert np.all(np.diag(block) == 0.0), f"input {b}: diag not zero"
        # upper triangle (strict, k=1) must be zero
        upper = np.triu(block, k=0)
        # k=0 includes diag — already checked. k=1 is strictly above.
        upper_strict = np.triu(block, k=1)
        assert np.all(upper_strict == 0.0), f"input {b}: strict upper not zero"

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Write inputs + reference.
    a_strict.tofile(os.path.join(out_dir, "chunk_tri_solve_invert_input_a_strict.bin"))
    a_inv_ref.tofile(os.path.join(out_dir, "chunk_tri_solve_invert_a_inv_ref.bin"))

    meta = {
        "B": B,
        "T": T,
        "H": H,
        "BT": BT,
        "seed": f"0x{SEED:08X}",
        "spec_source": "FLA solve_tril (vllm/.../solve_tril.py:506-530)",
        "reference_impl": "numpy.linalg.inv(I + A_strict)",
    }
    with open(os.path.join(out_dir, "chunk_tri_solve_invert_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"chunk_tri_solve_invert reference: B={B} T={T} H={H} BT={BT} "
        f"max(|A_inv|) = {np.abs(a_inv_ref).max():.4f}"
    )


if __name__ == "__main__":
    main()
