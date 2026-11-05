#!/usr/bin/env python3
"""Extract the checked-in Gemma routing-witness sample used by
`tests/gemma_gelu_epilogue_bitwise.rs` (ADR-063 Q4-GEMMA-DECODE T-N3,
design section 2.4 witness rows).

Source: the Q1 routing witness of the accepted greedy144 dataset,
`q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl`
(4 lanes x 144 positions x 30 layers, top-8 of 128). Positions 0-127 are the
batched-prefill rows, 128-143 the greedy continuation (decode-like).

Each group is one layer and four consecutive positions; `ids[i][lane]` holds
the 8 selected expert ids (selection order) of lane `lane` at
`positions[i]`. The test replays one position as an n_tokens = 4 pattern (the
four lanes), two as 8 and four as 16.

Usage: gemma_routing_witness_extract.py <routing-witness.jsonl> <out.json>
"""

import hashlib
import json
import sys

LANES = 4
DECODE = range(128, 144)
# Decode-like rows: first and last group of four, at four spread layers.
DECODE_GROUPS = [(layer, start) for layer in (0, 11, 23, 29) for start in (128, 140)]
# Rows 29-127 (lanes partly identical): first and last group, three layers.
PREFILL_GROUPS = [(layer, start) for layer in (2, 14, 27) for start in (29, 124)]


def main(src, out):
    digest = hashlib.sha256()
    rows = {}
    with open(src, "rb") as f:
        for raw in f:
            digest.update(raw)
            o = json.loads(raw)
            if o["kind"] != "routing_row":
                continue
            at = o["at"]
            rows[(at["layer"], at["row"]["position"], at["row"]["lane"])] = o["selected_ids"]

    def u(layer, position):
        return len({e for lane in range(LANES) for e in rows[(layer, position, lane)]})

    # The decode rows with the fewest and the most distinct experts, each as
    # the start of a group that stays inside 128-143 (skipped if already
    # sampled above).
    decode_rows = sorted(
        ((u(layer, p), layer, p) for layer in range(30) for p in DECODE),
        key=lambda t: (t[0], t[1], t[2]),
    )
    extremes = []
    for _, layer, p in (decode_rows[0], decode_rows[-1]):
        group = (layer, min(p, DECODE.stop - 4))
        if group not in DECODE_GROUPS and group not in extremes:
            extremes.append(group)

    groups = []
    for kind, selection in (
        ("decode", DECODE_GROUPS),
        ("decode_extreme_u", extremes),
        ("prefill", PREFILL_GROUPS),
    ):
        for layer, start in selection:
            positions = list(range(start, start + 4))
            groups.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "positions": positions,
                    "u": [u(layer, p) for p in positions],
                    "ids": [[rows[(layer, p, lane)] for lane in range(LANES)] for p in positions],
                }
            )

    doc = {
        "source": "q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl",
        "source_sha256": digest.hexdigest(),
        "lanes": LANES,
        "top_k": 8,
        "n_experts": 128,
        "groups": groups,
    }
    with open(out, "w") as f:
        # One group per line keeps the file small and diffable.
        f.write("{\n")
        for key in ("source", "source_sha256", "lanes", "top_k", "n_experts"):
            f.write(f'  "{key}": {json.dumps(doc[key])},\n')
        f.write('  "groups": [\n')
        f.write(",\n".join("    " + json.dumps(g, separators=(",", ":")) for g in groups))
        f.write("\n  ]\n}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
