#!/usr/bin/env python3
"""Zero-knowledge proof of one traced Q4_K matmul node with Groth16 (circom, snarkjs).

    python3 groth16_node.py receipt.json --model model.gguf --index N [--rows 64] [--groups 0,1]
                            [--build build/r64_k1536] [--manifest manifest.json] [--out dir]

The node's weight rows are proven in groups of ROWS rows. For each group the circuit
(qdot_rows.circom) proves ggml's Q4_K x Q8_K integer core with the weights private and bound
to a Poseidon commitment (public output); the activation quants and the per-block sums are
public inputs. Groth16 hides the private witness beyond this full public statement;
activation quants and sums remain public. Registration publishes one commitment per group (register.py --groth16);
the verifier compares the proof's commitment with the registered one for that group.

Negative cases, per group: a tampered public sum, a proof made with one changed nibble and
consistent sums (its commitment differs), and the honest proof against another group's
registered commitment.
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
sys.path.insert(0, str(HERE.parents[1]))
import verify_trace as vt  # noqa: E402
from zk_node import integer_sums, pick_opening  # noqa: E402

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
NODE = ["node", "--max-old-space-size=16384"]
SNARKJS = str(HERE / "node_modules" / "snarkjs" / "build" / "cli.cjs")


def fe(x: int) -> str:
    return str(int(x) % P)


def bits(values: np.ndarray, n: int) -> list[str]:
    v = values.reshape(-1).astype(np.int64)
    return [str((int(x) >> b) & 1) for x in v for b in range(n)]


def commitment_js(q4: np.ndarray, sc: np.ndarray, mn: np.ndarray, salt: int) -> str:
    doc = {"salt": str(salt), "q4": q4.reshape(-1).astype(int).tolist(), "sc": sc.reshape(-1).astype(int).tolist(), "mn": mn.reshape(-1).astype(int).tolist()}
    r = subprocess.run(["node", str(HERE / "commit.js")], input=json.dumps(doc), capture_output=True, text=True, check=True)
    return r.stdout.strip()


def snarkjs(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([*NODE, SNARKJS, *args], capture_output=True, text=True, cwd=cwd)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("receipt")
    p.add_argument("--model", required=True)
    p.add_argument("--trace")
    p.add_argument("--index", type=int, required=True, help="leaf index of the opened MUL_MAT")
    p.add_argument("--rows", type=int, help="rows per proof (default: register.py's rule for this K); must match the compiled circuit")
    p.add_argument("--groups", help="comma-separated group indices to prove (default: all)")
    p.add_argument("--build", help="directory with main_js/main.wasm, main_final.zkey, verification_key.json (default: build/r{rows}_k{K})")
    p.add_argument("--manifest", help="registration manifest with groth16 group commitments; the verifier takes the registered value from it")
    p.add_argument("--salt", type=int, default=0)
    p.add_argument("--out", default="groth16-out")
    p.add_argument("--progress-json", action="store_true", help="emit stage events for a local interface")
    a = p.parse_args(argv)

    def progress(stage, group):
        if a.progress_json:
            print(json.dumps({"event": "progress", "stage": stage, "group": group}), flush=True)

    rec = json.loads(Path(a.receipt).read_text())
    tpath = Path(a.trace) if a.trace else Path(a.receipt).with_name(Path(a.receipt).name.removesuffix(".json") + ".trace.json")
    tdoc = json.loads(tpath.read_text())
    o = pick_opening(tdoc, a.index)
    leaf = tdoc["leaves"][o["index"]]
    wsrc, asrc = leaf["srcs"][0], leaf["srcs"][1]
    name = wsrc["base"]["name"]
    K, M = wsrc["base"]["ne"][0], wsrc["base"]["ne"][1]
    nb = K // 256
    if a.rows is None:
        from register import groth16_rows
        a.rows = groth16_rows(K)
    assert M % a.rows == 0, f"{M} rows is not a multiple of {a.rows}"
    n_groups = M // a.rows
    print(f"node: leaf {o['index']} graph {leaf['g']} {leaf['name']} = {name} [{K} x {M}] q4_K; {n_groups} groups of {a.rows} rows")

    store = vt.WeightStore(a.model)
    assert store.hashes[name] == wsrc["sha256"], "weight hash in the leaf does not match this GGUF"
    d_w, dmin, sc, mn, q4 = vt.unpack_q4_k(store.blocks(name))
    enc = o.get("enc", "zlib+base64")
    act_blob = vt.decode_blob(o["srcs"][1], enc)
    assert vt.sha256(act_blob) == asrc["base"]["sha256"]
    act = vt.tensor_from_bytes(act_blob, asrc["type"], asrc["ne"], asrc["nb"], asrc["offset"]).reshape(-1, K)
    assert act.shape[0] == 1, "one activation row"
    d_a, q8, bsums = vt.quantize_q8_k(act.astype(np.float32))
    q8 = q8[0].reshape(-1).astype(np.int32)
    bsums = bsums[0]
    s1, s2 = integer_sums(q4.reshape(M, -1), sc, mn, q8, bsums)

    build = Path(a.build) if a.build else HERE / "build" / f"r{a.rows}_k{K}"
    wasm, zkey, vkey = build / "main_js" / "main.wasm", build / "main_final.zkey", build / "verification_key.json"
    for f in (wasm, zkey, vkey):
        assert f.exists(), f"missing {f}; run setup.sh"
    registered = None
    if a.manifest:
        man = json.loads(Path(a.manifest).read_text())
        from registration_schema import validate_manifest
        validate_manifest(man)
        entry = next((t for t in man["tensors"] if t["name"] == name), None)
        if entry is None or "groth16" not in entry:
            raise SystemExit(f"{name} has no groth16 commitments in {a.manifest}")
        assert entry["groth16"]["rows_per_group"] == a.rows and entry["sha256"] == wsrc["sha256"]
        registered = entry["groth16"]["groups"]
        print(f"registered commitments from manifest {man['manifest_id'][:16]}...")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = [int(g) for g in a.groups.split(",")] if a.groups else list(range(n_groups))
    if not groups or len(set(groups)) != len(groups) or any(g < 0 or g >= n_groups for g in groups):
        raise SystemExit("groups must be distinct indices inside the tensor")
    results = []
    for g in groups:
        rows = slice(g * a.rows, (g + 1) * a.rows)
        q4g, scg, mng = q4.reshape(M, -1)[rows], sc[rows], mn[rows]
        gdir = outdir / f"group{g}"
        gdir.mkdir(exist_ok=True)
        inp = {"q4bits": bits(q4g, 4), "scbits": bits(scg, 6), "mnbits": bits(mng, 6), "salt": str(a.salt),
               "q8": [fe(x) for x in q8], "s1": [fe(x) for x in s1[rows].reshape(-1)], "s2": [fe(x) for x in s2[rows].reshape(-1)]}
        (gdir / "input.json").write_text(json.dumps(inp))
        progress("witness", g)
        t = time.time()
        r = snarkjs("wtns", "calculate", str(wasm), str(gdir / "input.json"), str(gdir / "witness.wtns"))
        if r.returncode != 0:
            print(f"group {g}: witness generation failed (the inputs do not satisfy the circuit)\n{r.stderr[-400:]}")
            return 1
        t_wit = time.time() - t
        progress("proving", g)
        t = time.time()
        r = snarkjs("groth16", "prove", str(zkey), str(gdir / "witness.wtns"), str(gdir / "proof.json"), str(gdir / "public.json"))
        if r.returncode != 0:
            print(f"group {g}: prove failed\n{r.stderr[-400:]}")
            return 1
        t_prove = time.time() - t
        (gdir / "witness.wtns").unlink()
        (gdir / "input.json").unlink()  # the private inputs never leave the prover
        public = json.loads((gdir / "public.json").read_text())
        commitment = public[0]
        expected = registered[g] if registered else commitment_js(q4g, scg, mng, a.salt)
        bound = commitment == expected
        progress("checking", g)
        t = time.time()
        if a.manifest:
            v = subprocess.run(["node", str(HERE / "verify_package.cjs"), str(vkey), a.manifest, name, str(g), str(gdir)], capture_output=True, text=True)
            accept = v.returncode == 0 and json.loads(v.stdout).get("accept") is True
        else:
            v = snarkjs("groth16", "verify", str(vkey), str(gdir / "public.json"), str(gdir / "proof.json"))
            accept = v.returncode == 0 and "OK" in v.stdout
        t_verify = time.time() - t
        # negative 1: tampered public sum
        bad = list(public)
        bad[1 + K] = fe(int(bad[1 + K]) + 1)
        (gdir / "public-tampered.json").write_text(json.dumps(bad))
        tampered = snarkjs("groth16", "verify", str(vkey), str(gdir / "public-tampered.json"), str(gdir / "proof.json")).returncode == 0
        # negative 2: another group's registered commitment
        other = (registered[(g + 1) % n_groups] if registered else commitment_js(q4.reshape(M, -1)[((g + 1) % n_groups) * a.rows:((g + 1) % n_groups + 1) * a.rows],
                                                                                    sc[((g + 1) % n_groups) * a.rows:((g + 1) % n_groups + 1) * a.rows],
                                                                                    mn[((g + 1) % n_groups) * a.rows:((g + 1) % n_groups + 1) * a.rows], a.salt))
        other_bound = commitment == other
        proof_bytes = (gdir / "proof.json").stat().st_size + (gdir / "public.json").stat().st_size
        print(f"group {g}: witness {t_wit:.1f}s, prove {t_prove:.1f}s, verify {t_verify:.2f}s, package {proof_bytes} bytes; "
              f"proof {'ACCEPT' if accept else 'REJECT'}; commitment {'matches' if bound else 'DIFFERS FROM'} the {'registered' if registered else 'locally recomputed'} one; "
              f"tampered sum {'accepted' if tampered else 'rejected'}; other group's commitment {'accepted' if other_bound else 'rejected'}")
        results.append({"group": g, "witness_seconds": round(t_wit, 2), "prove_seconds": round(t_prove, 2), "verify_seconds": round(t_verify, 3),
                        "package_bytes": proof_bytes, "accept": accept, "commitment": commitment, "bound": bound,
                        "tampered_sum_accepted": tampered, "other_group_accepted": other_bound})
    ok = all(r["accept"] and r["bound"] and not r["tampered_sum_accepted"] and not r["other_group_accepted"] for r in results)
    summary = {"leaf": o["index"], "node": leaf["name"], "weight": name, "m": M, "k": K, "rows_per_group": a.rows, "groups": results,
               "float_step_error": float(vt.normalized_error(
                   vt.tensor_from_bytes(vt.decode_blob(o["out"], enc), leaf["out"]["type"], leaf["out"]["ne"], leaf["out"]["nb"], leaf["out"]["offset"]).reshape(-1, M)[0],
                   (d_a[0][None, :] * d_w * s1 - d_a[0][None, :] * dmin * s2).sum(axis=1)))}
    print(f"float step: max|out - ref| / max|ref| = {summary['float_step_error']:.2e}")
    summary["verification_policy"] = "registered shared policy" if a.manifest else "pairing and local commitment only; shared registration/range policy skipped"
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("RESULT:", ("proofs pass the shared registration and range policy; negatives rejected" if a.manifest else
                       "pairings and locally recomputed commitments match; shared registration/range policy SKIPPED") if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
