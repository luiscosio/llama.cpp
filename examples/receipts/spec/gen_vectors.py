#!/usr/bin/env python3
"""Produce test vectors for the Lean spec from a real trace and the verifier's GGUF.

    python3 gen_vectors.py receipt.json --model model.gguf --out vectors.json

For a few weight rows of an opened Q4_K matmul and one Q6_K matmul, records the raw block
bytes, the activation row as float32 bit patterns, and what the Python verifier computes:
the Q8_K quants and scales, the per-block integer sums, and the float row value. The Lean
checker recomputes all of it from the bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import verify_trace as vt  # noqa: E402


def pick(tdoc, qtype):
    for o in tdoc["openings"]:
        leaf = tdoc["leaves"][o["index"]]
        if leaf["op"] == "MUL_MAT" and leaf["srcs"][0].get("kind") == "weight" and leaf["srcs"][0]["base"]["type"].lower() == qtype and leaf["out"]["ne"][1] == 1:
            return o, leaf
    raise SystemExit(f"no opened one-row MUL_MAT with a {qtype} weight")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("receipt")
    p.add_argument("--model", required=True)
    p.add_argument("--trace")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--out", default="vectors.json")
    a = p.parse_args(argv)
    tpath = Path(a.trace) if a.trace else Path(a.receipt).with_name(Path(a.receipt).name.removesuffix(".json") + ".trace.json")
    tdoc = json.loads(tpath.read_text())
    store = vt.WeightStore(a.model)
    out = {"q4k": [], "q6k": []}

    o, leaf = pick(tdoc, "q4_k")
    name = leaf["srcs"][0]["base"]["name"]
    K = leaf["srcs"][0]["base"]["ne"][0]
    asrc = leaf["srcs"][1]
    act = vt.tensor_from_bytes(vt.decode_blob(o["srcs"][1], o.get("enc", "zlib+base64")), asrc["type"], asrc["ne"], asrc["nb"], asrc["offset"]).reshape(-1, K)[0].astype(np.float32)
    d_a, q8, bsums = vt.quantize_q8_k(act[None, :])
    blocks = store.blocks(name)
    d_w, dmin, sc, mn, q4 = vt.unpack_q4_k(blocks[: a.rows])
    nb = K // 256
    for r in range(a.rows):
        q4b = q4[r].reshape(nb, 8, 32).astype(np.int64)
        q8b = q8[0].reshape(nb, 8, 32).astype(np.int64)
        s1 = np.einsum("ijl,ijl->ij", q4b, q8b)
        s1 = (s1 * sc[r].astype(np.int64)).sum(axis=1)
        s2 = (bsums[0].astype(np.int64) * np.repeat(mn[r].astype(np.int64), 2, axis=-1)).sum(axis=1)
        row = float((d_a[0].astype(np.float64) * d_w[r].astype(np.float64) * s1 - d_a[0].astype(np.float64) * dmin[r].astype(np.float64) * s2).sum())
        out["q4k"].append({"weight": name, "row": r, "blocks_hex": blocks[r].tobytes().hex(),
                           "act_bits": act.view(np.uint32).tolist(), "d_bits": d_a[0].view(np.uint32).tolist(),
                           "q8": q8[0].reshape(-1).astype(int).tolist(), "bsums": bsums[0].reshape(-1).astype(int).tolist(),
                           "s1": s1.tolist(), "s2": s2.tolist(), "row_value": row})

    o, leaf = pick(tdoc, "q6_k")
    name = leaf["srcs"][0]["base"]["name"]
    K = leaf["srcs"][0]["base"]["ne"][0]
    asrc = leaf["srcs"][1]
    act = vt.tensor_from_bytes(vt.decode_blob(o["srcs"][1], o.get("enc", "zlib+base64")), asrc["type"], asrc["ne"], asrc["nb"], asrc["offset"]).reshape(-1, K)[0].astype(np.float32)
    d_a, q8, bsums = vt.quantize_q8_k(act[None, :])
    blocks = store.blocks(name)
    d_w, scales, q6 = vt.unpack_q6_k(blocks[: a.rows])
    nb = K // 256
    for r in range(a.rows):
        q6b = q6[r].reshape(nb, 16, 16).astype(np.int64)
        q8b = q8[0].reshape(nb, 16, 16).astype(np.int64)
        dot = (np.einsum("ijl,ijl->ij", q6b, q8b) * scales[r].astype(np.int64)).sum(axis=1)
        out["q6k"].append({"weight": name, "row": r, "blocks_hex": blocks[r].tobytes().hex(),
                           "act_bits": act.view(np.uint32).tolist(), "dot": dot.tolist()})
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {len(out['q4k'])} q4_K rows of {out['q4k'][0]['weight']}, {len(out['q6k'])} q6_K rows of {out['q6k'][0]['weight']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
