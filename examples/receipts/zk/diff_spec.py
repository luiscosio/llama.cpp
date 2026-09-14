#!/usr/bin/env python3
"""Differential test: the proof circuit against the Lean specification, on the same vectors.

    python3 spec/gen_vectors.py receipt.json --model model.gguf --out vectors.json
    spec-check vectors vectors.json --emit lean.json
    python3 zk/diff_spec.py vectors.json lean.json [--binary target/release/receipts-zk] [--config orion]

For every Q4_K row in the vectors, the weight integers come from the row's block bytes and the
activation quants and the per-block sums come from what the Lean spec computed. The circuit
must accept exactly those sums and reject them once one is changed. Python's numbers are not
used, so this checks the circuit against the specification, not against the verifier.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import verify_trace as vt  # noqa: E402


def run(binary: str, *args: str) -> tuple[int, str]:
    r = subprocess.run([binary, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("vectors")
    p.add_argument("lean", help="what spec-check vectors --emit wrote")
    p.add_argument("--binary", default=str(HERE / "target" / "release" / "receipts-zk"))
    p.add_argument("--config", default="orion", choices=["orion", "raw"])
    a = p.parse_args(argv)
    vectors = json.loads(Path(a.vectors).read_text())["q4k"]
    lean = json.loads(Path(a.lean).read_text())["q4k"]
    if len(vectors) != len(lean):
        print(f"{len(vectors)} vector rows but {len(lean)} Lean rows")
        return 2
    ok_all = True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (vec, spec) in enumerate(zip(vectors, lean)):
            if vec["blocks_hex"] != spec["blocks_hex"]:
                print(f"row {i}: Lean output is for different block bytes")
                return 2
            blocks = np.frombuffer(bytes.fromhex(vec["blocks_hex"]), dtype=np.uint8).reshape(1, -1)
            _, _, sc, mn, q4 = vt.unpack_q4_k(blocks)
            K = int(q4[0].size)  # nibbles of the row, in the kernel's read order
            witness = {"m": 1, "k": K, "q4": q4.reshape(-1).astype(int).tolist(), "sc": sc.reshape(-1).astype(int).tolist(),
                       "mn": mn.reshape(-1).astype(int).tolist(), "q8": [int(x) for x in spec["q8"]],
                       "s1": [int(x) for x in spec["s1"]], "s2": [int(x) for x in spec["s2"]]}
            out = Path(tmp) / f"row{i}"
            out.mkdir()
            (out / "witness.json").write_text(json.dumps(witness))
            code, log = run(a.binary, "prove", str(out / "witness.json"), str(out), a.config)
            if code != 0:
                print(f"row {i}: circuit rejects the spec's sums at proving time\n{log.strip()[-400:]}")
                ok_all = False
                continue
            code, log = run(a.binary, "verify", str(out / "public.json"), str(out / "proof.bin"), a.config)
            accept = code == 0
            bad = json.loads((out / "public.json").read_text())
            bad["s1"][0] += 1
            (out / "public-bad.json").write_text(json.dumps(bad))
            code, _ = run(a.binary, "verify", str(out / "public-bad.json"), str(out / "proof.bin"), a.config)
            reject = code != 0
            print(f"row {i}: k={K}, {len(spec['s1'])} blocks; Lean's s1, s2 accepted by the circuit: {accept}; s1[0]+1 rejected: {reject}")
            ok_all = ok_all and accept and reject
    print("diff_spec:", "ok" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
