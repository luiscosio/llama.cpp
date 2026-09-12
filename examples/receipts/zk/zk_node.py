#!/usr/bin/env python3
"""Prove one traced matmul node with Expander and check it against the receipt.

    python3 zk_node.py receipt.json --model model.gguf [--index N] [--config orion|raw] [--out dir]

Picks a MUL_MAT opening whose weight is Q4_K (the first decode-graph one unless --index
is given), extracts the witness from the opened bytes and the verifier's own GGUF:

    q4, sc, mn   the weight's nibbles, sub-block scales and mins (public, checked from the GGUF)
    q8           the activation row quantized to Q8_K exactly as ggml does
    s1, s2       the per-block integer sums the circuit must reproduce

then runs `receipts-zk prove`, `receipts-zk verify`, and finally applies the per-block
float scales to s1 and s2 the way ggml's kernel does and compares with the node's
opened output. The proof covers the integer core; the float step is a few thousand
multiply-adds the receipt verifier does itself.

Requires the receipts-zk binary (cargo build --release in this directory) and numpy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import verify_trace as vt  # noqa: E402

QK_K = 256


def pick_opening(tdoc: dict, index: int | None) -> dict:
    leaves = tdoc["leaves"]
    for o in tdoc["openings"]:
        leaf = leaves[o["index"]]
        if index is not None and o["index"] != index:
            continue
        if leaf["op"] != "MUL_MAT" or leaf["srcs"][0].get("kind") != "weight":
            continue
        if leaf["srcs"][0]["base"]["type"].lower() != "q4_k":
            continue
        if index is None and (leaf["g"] == 0 or leaf["out"]["ne"][1] != 1):
            continue  # first proof of concept: one activation row (a decode graph)
        return o
    raise SystemExit("no opened decode-graph MUL_MAT with a Q4_K weight in this trace")


def integer_sums(q4: np.ndarray, sc: np.ndarray, mn: np.ndarray, q8: np.ndarray, bsums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """s1[m,i] and s2[m,i] as the circuit defines them (exact int64)."""
    m, nb = sc.shape[0], sc.shape[1]
    q4b = q4.reshape(m, nb, 8, 32).astype(np.int64)
    q8b = q8.reshape(nb, 8, 32).astype(np.int64)
    sub = np.einsum("mijl,ijl->mij", q4b, q8b)  # (m, nb, 8)
    s1 = np.einsum("mij,mij->mi", sub, sc.astype(np.int64))
    mins16 = np.repeat(mn.astype(np.int64), 2, axis=-1)  # (m, nb, 16)
    s2 = np.einsum("ij,mij->mi", bsums.astype(np.int64), mins16)
    return s1, s2


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("receipt")
    p.add_argument("--model", required=True)
    p.add_argument("--trace")
    p.add_argument("--index", type=int, help="leaf index of the opening to prove")
    p.add_argument("--config", default="orion", choices=["orion", "raw"])
    p.add_argument("--out", default="zk-out")
    p.add_argument("--binary", default=str(HERE / "target" / "release" / "receipts-zk"))
    p.add_argument("--verify-only", action="store_true", help="verify an existing public.json and proof.bin without creating a proof")
    a = p.parse_args(argv)

    rec = json.loads(Path(a.receipt).read_text())
    tpath = Path(a.trace) if a.trace else Path(a.receipt).with_name(Path(a.receipt).name.removesuffix(".json") + ".trace.json")
    tdoc = json.loads(tpath.read_text())
    o = pick_opening(tdoc, a.index)
    leaf = tdoc["leaves"][o["index"]]
    wsrc, asrc = leaf["srcs"][0], leaf["srcs"][1]
    name = wsrc["base"]["name"]
    K, M = wsrc["base"]["ne"][0], wsrc["base"]["ne"][1]
    nb = K // QK_K
    print(f"node: leaf {o['index']} graph {leaf['g']} {leaf['name']} = {name} [{K} x {M}] q4_K times activation {asrc['ne'][:2]}")

    # weights from the verifier's GGUF, checked against the leaf's committed hash
    store = vt.WeightStore(a.model)
    assert store.hashes[name] == wsrc["sha256"], "weight hash in the leaf does not match this GGUF"
    blocks = store.blocks(name)
    d_w, dmin, sc, mn, q4 = vt.unpack_q4_k(blocks)  # (M,nb) (M,nb) (M,nb,8) (M,nb,8) (M,nb,256)

    # activation and output from the opened bytes, checked against the leaf's hashes
    enc = o.get("enc", "zlib+base64")
    out_blob = vt.decode_blob(o["out"], enc)
    act_blob = vt.decode_blob(o["srcs"][1], enc)
    assert vt.sha256(act_blob) == asrc["base"]["sha256"] and vt.sha256(out_blob) == leaf["out"]["base"]["sha256"]
    act = vt.tensor_from_bytes(act_blob, asrc["type"], asrc["ne"], asrc["nb"], asrc["offset"]).reshape(-1, K)
    out = vt.tensor_from_bytes(out_blob, leaf["out"]["type"], leaf["out"]["ne"], leaf["out"]["nb"], leaf["out"]["offset"]).reshape(-1, M)
    assert act.shape[0] == 1 and out.shape[0] == 1, "this proof of concept covers one activation row"
    d_a, q8, bsums = vt.quantize_q8_k(act.astype(np.float32))  # (1,nb) (1,nb,256) (1,nb,16)
    q8 = q8[0].reshape(-1).astype(np.int32)
    bsums = bsums[0]

    s1, s2 = integer_sums(q4.reshape(M, -1), sc, mn, q8, bsums)
    witness = {"m": M, "k": K, "q4": q4.reshape(-1).astype(int).tolist(), "sc": sc.reshape(-1).astype(int).tolist(),
               "mn": mn.reshape(-1).astype(int).tolist(), "q8": q8.tolist(), "s1": s1.reshape(-1).tolist(), "s2": s2.reshape(-1).tolist()}
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"statement: {M * K} nibbles, {M * nb * 8} scales, {M * nb * 8} mins, {K} activations, {2 * M * nb} sums public")

    # the float step the receipt verifier performs itself, as ggml's kernel does after the integer core
    ref = (d_a[0][None, :] * d_w * s1 - d_a[0][None, :] * dmin * s2).sum(axis=1)
    err = vt.normalized_error(out[0], ref)
    print(f"float step: max|out - ref| / max|ref| = {err:.2e} against the opened output")

    prove_stats = None
    t_prove = 0.0
    if not a.verify_only:
        (outdir / "witness.json").write_text(json.dumps(witness))
        t = time.time()
        r = subprocess.run([a.binary, "prove", str(outdir / "witness.json"), str(outdir), a.config], capture_output=True, text=True)
        print(r.stderr.strip())
        if r.returncode != 0:
            print("prove failed", r.stdout)
            return 1
        prove_stats = json.loads(r.stdout.strip().splitlines()[-1])
        t_prove = time.time() - t

    public = json.loads((outdir / "public.json").read_text())
    for key in ("q4", "sc", "mn", "q8", "s1", "s2"):
        if public[key] != witness[key]:
            raise RuntimeError(f"proof public input {key} differs from the value derived from the GGUF and trace")

    t = time.time()
    v = subprocess.run([a.binary, "verify", str(outdir / "public.json"), str(outdir / "proof.bin"), a.config], capture_output=True, text=True)
    print(v.stderr.strip())
    verify_stats = json.loads(v.stdout.strip().splitlines()[-1]) if v.stdout.strip() else {"accept": False}
    t_verify = time.time() - t

    # a tampered public sum must not verify
    bad = json.loads((outdir / "public.json").read_text())
    bad["s1"][0] += 1
    (outdir / "public-tampered.json").write_text(json.dumps(bad))
    b = subprocess.run([a.binary, "verify", str(outdir / "public-tampered.json"), str(outdir / "proof.bin"), a.config], capture_output=True, text=True)
    tampered_accept = b.returncode == 0

    summary = {"leaf": o["index"], "node": leaf["name"], "weight": name, "m": M, "k": K, "config": a.config,
               "float_step_error": err, "prove": prove_stats, "prove_wall_seconds": round(t_prove, 3),
               "verify": verify_stats, "verify_wall_seconds": round(t_verify, 3), "tampered_sum_accepted": tampered_accept}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    ok = verify_stats.get("accept") and not tampered_accept and err < 1e-5
    print("RESULT:", "proof verifies, tampered sum rejected, float step matches" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
