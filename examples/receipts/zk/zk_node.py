#!/usr/bin/env python3
"""Prove one traced matmul node with Expander and check it against the receipt.

    python3 zk_node.py receipt.json --model model.gguf [--index N] [--config orion|raw] [--out dir]

Picks a MUL_MAT opening whose weight is Q4_K (the first decode-graph one unless --index
is given), extracts the witness from the opened bytes and the GGUF:

    q4, sc, mn   the weight's nibbles, sub-block scales and mins (private, under a commitment)
    q8           the activation row quantized to Q8_K exactly as ggml does (public)
    s1, s2       the per-block integer sums the circuit must reproduce (public)

Three roles, run here one after the other:

    register   `receipts-zk commit` on the weights from the GGUF: the tensor's commitment,
               which a deployment would publish once per model (registered.hex);
    prove      `receipts-zk prove` on the witness: proof.bin, public.json, commitment.hex;
    verify     `receipts-zk verify --commitment registered.hex`: the verifier never sees the
               weights, only the registered commitment, the public inputs and the proof.

Then the negative cases: a tampered public sum, a proof made with different weights (whose
commitment therefore differs), and the registered commitment of another tensor. Finally
the per-block float scales are applied to s1 and s2 the way ggml's kernel does and the
result is compared with the node's opened output. The proof covers the integer core; the
float step is a few thousand multiply-adds the receipt verifier does itself.

Not zero-knowledge yet: the commitment binds but does not hide, and Expander's GKR has no
masking. What is established is model binding with a verifier that holds no weights.

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
    p.add_argument("--verify-only", action="store_true", help="verify an existing proof against the registered commitment without proving")
    p.add_argument("--manifest", help="registration manifest (register.py); the registered commitment is taken from it instead of being computed here")
    a = p.parse_args(argv)
    if a.config == "raw":
        print("the raw configuration ships the witness inside the proof; use it only to debug the circuit")

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
    weights = {"m": M, "k": K, "q4": q4.reshape(-1).astype(int).tolist(), "sc": sc.reshape(-1).astype(int).tolist(),
               "mn": mn.reshape(-1).astype(int).tolist()}
    witness = {**weights, "q8": q8.tolist(), "s1": s1.reshape(-1).tolist(), "s2": s2.reshape(-1).tolist()}
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"statement: {M * K} nibbles, {M * nb * 8} scales, {M * nb * 8} mins private under the registered commitment; "
          f"{K} activations, {2 * M * nb} sums public")

    # the float step the receipt verifier performs itself, as ggml's kernel does after the integer core
    ref = (d_a[0][None, :] * d_w * s1 - d_a[0][None, :] * dmin * s2).sum(axis=1)
    err = vt.normalized_error(out[0], ref)
    print(f"float step: max|out - ref| / max|ref| = {err:.2e} against the opened output")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([a.binary, *args], capture_output=True, text=True)

    def last_json(r: subprocess.CompletedProcess) -> dict:
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"accept": False}

    prove_stats = commit_stats = None
    t_prove = t_commit = 0.0
    committing = a.config != "raw"
    manifest_commitment = None
    if a.manifest:
        manifest = json.loads(Path(a.manifest).read_text())
        entry = next((t for t in manifest["tensors"] if t["name"] == name), None)
        if entry is None or "commitment" not in entry:
            raise SystemExit(f"{name} has no commitment in {a.manifest}")
        if entry["sha256"] != wsrc["sha256"]:
            raise SystemExit(f"{name}: the leaf's weight hash is not the registered one")
        manifest_commitment = entry["commitment"]["value"]
        (outdir / "registered").mkdir(parents=True, exist_ok=True)
        (outdir / "registered" / "commitment.hex").write_text(manifest_commitment + "\n")
        print(f"registered commitment from manifest {manifest['manifest_id'][:16]}... ({manifest['model']['name']}): {manifest_commitment[:16]}...")
    if not a.verify_only:
        if committing and manifest_commitment is None:
            # registration: the tensor's commitment from the GGUF alone (here on the same machine; a
            # deployment publishes it once per model and the verifier pins it, see register.py)
            (outdir / "weights.json").write_text(json.dumps(weights))
            t = time.time()
            r = run("commit", str(outdir / "weights.json"), str(outdir / "registered"))
            print(r.stderr.strip())
            if r.returncode != 0:
                print("commit failed", r.stdout)
                return 1
            commit_stats = last_json(r)
            t_commit = time.time() - t
        (outdir / "witness.json").write_text(json.dumps(witness))
        t = time.time()
        r = run("prove", str(outdir / "witness.json"), str(outdir), a.config)
        print(r.stderr.strip())
        if r.returncode != 0:
            print("prove failed", r.stdout)
            return 1
        prove_stats = last_json(r)
        t_prove = time.time() - t

    public = json.loads((outdir / "public.json").read_text())
    for key in ("q8", "s1", "s2"):
        if public[key] != witness[key]:
            raise RuntimeError(f"proof public input {key} differs from the value derived from the trace")
    registered = (outdir / "registered" / "commitment.hex").read_text().strip() if committing else ""
    commit_args = ["--commitment", registered] if committing else []
    if committing:
        print(f"registered commitment {registered[:16]}..., proof carries {public['commitment'][:16]}...")

    t = time.time()
    v = run("verify", str(outdir / "public.json"), str(outdir / "proof.bin"), a.config, *commit_args)
    print(v.stderr.strip())
    verify_stats = last_json(v)
    t_verify = time.time() - t

    # negative 1: a tampered public sum must not verify
    bad = dict(public)
    bad["s1"] = list(public["s1"])
    bad["s1"][0] += 1
    (outdir / "public-tampered.json").write_text(json.dumps(bad))
    tampered_accept = run("verify", str(outdir / "public-tampered.json"), str(outdir / "proof.bin"), a.config, *commit_args).returncode == 0

    other_weights_accept = other_tensor_accept = False
    if committing and not a.verify_only:
        # negative 2: a prover with different weights. One nibble changed and the sums recomputed
        # so its circuit is satisfied; its commitment differs from the registered one.
        q4_bad = q4.reshape(M, -1).copy()
        q4_bad[0, 0] = (q4_bad[0, 0] + 1) % 16
        s1_bad, s2_bad = integer_sums(q4_bad, sc, mn, q8, bsums)
        wit_bad = {**witness, "q4": q4_bad.reshape(-1).astype(int).tolist(), "s1": s1_bad.reshape(-1).tolist(), "s2": s2_bad.reshape(-1).tolist()}
        (outdir / "other-weights").mkdir(exist_ok=True)
        (outdir / "other-weights" / "witness.json").write_text(json.dumps(wit_bad))
        r = run("prove", str(outdir / "other-weights" / "witness.json"), str(outdir / "other-weights"), a.config)
        if r.returncode != 0:
            print("other-weights prove failed", r.stdout)
            return 1
        other = run("verify", str(outdir / "other-weights" / "public.json"), str(outdir / "other-weights" / "proof.bin"), a.config, *commit_args)
        other_weights_accept = other.returncode == 0
        print(other.stderr.strip().splitlines()[-1])
        # negative 3: the honest proof against the registered commitment of another tensor of the
        # same shape (the same weight in the next layer)
        layer = int(name.split(".")[1])
        other_name = name.replace(f"blk.{layer}.", f"blk.{layer + 1}.")
        if other_name not in store.hashes:
            other_name = name.replace(f"blk.{layer}.", f"blk.{layer - 1}.")
        _, _, sc_o, mn_o, q4_o = vt.unpack_q4_k(store.blocks(other_name))
        (outdir / "other-tensor").mkdir(exist_ok=True)
        (outdir / "other-tensor" / "weights.json").write_text(json.dumps({"m": M, "k": K, "q4": q4_o.reshape(-1).astype(int).tolist(),
                                                                          "sc": sc_o.reshape(-1).astype(int).tolist(), "mn": mn_o.reshape(-1).astype(int).tolist()}))
        r = run("commit", str(outdir / "other-tensor" / "weights.json"), str(outdir / "other-tensor"))
        if r.returncode != 0:
            print("other-tensor commit failed", r.stdout)
            return 1
        other_reg = (outdir / "other-tensor" / "commitment.hex").read_text().strip()
        ot = run("verify", str(outdir / "public.json"), str(outdir / "proof.bin"), a.config, "--commitment", other_reg)
        other_tensor_accept = ot.returncode == 0
        print(f"{other_name}: {ot.stderr.strip().splitlines()[-1]}")

    summary = {"leaf": o["index"], "node": leaf["name"], "weight": name, "m": M, "k": K, "config": a.config,
               "float_step_error": err, "commit": commit_stats, "commit_wall_seconds": round(t_commit, 3),
               "prove": prove_stats, "prove_wall_seconds": round(t_prove, 3),
               "verify": verify_stats, "verify_wall_seconds": round(t_verify, 3),
               "tampered_sum_accepted": tampered_accept, "other_weights_accepted": other_weights_accept,
               "other_tensor_commitment_accepted": other_tensor_accept}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    ok = verify_stats.get("accept") and not tampered_accept and not other_weights_accept and not other_tensor_accept and err < 1e-5
    print("RESULT:", "proof verifies against the registered commitment; tampered sum, other weights and other tensor rejected; float step matches"
          if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
