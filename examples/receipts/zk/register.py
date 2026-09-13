#!/usr/bin/env python3
"""Registration manifest for one GGUF: what a verifier pins before it checks any proof.

    python3 register.py model.gguf --out manifest.json [--commit] [--source-url URL --source-sha256 HEX]
                        [--quantize-cmd "..."] [--llama-rev REV] [--limit N]
    python3 register.py --check manifest.json --model model.gguf [--commit]

The manifest records the model file (digest, size, architecture and its hyperparameters, file
type), every tensor (name, type, shape, bytes, SHA-256), the tokenizer material (a digest over
tokens, merges, token types and the pre-tokenizer name), the execution specification the
statement fixes, the proof system's identity, and, with --commit, the Orion commitment of every
Q4_K tensor as the circuit of `receipts-zk` lays it out: the value `receipts-zk verify
--commitment` will demand. The manifest's id is the SHA-256 of its canonical JSON.

--check recomputes everything from another copy of the GGUF and reports what differs, so a
registration can be validated independently. Changing a weight, a tensor's type or shape, the
tokenizer or the execution specification changes the id.

Not a hiding commitment: Orion binds the weights but does not hide them. The manifest names
the scheme so a later hiding scheme can be added as another entry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(1, str(HERE.parents[3] / "gguf-py"))
import verify_trace as vt  # noqa: E402
from gguf import GGUFReader  # noqa: E402

STATEMENT_VERSION = "stmt/v0"
COMMITMENT_SCHEME = "expander-orion/m31x16/receipts-zk-qdot"


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def field_scalar(reader: GGUFReader, name: str):
    f = reader.fields.get(name)
    if f is None or not f.data:
        return None
    part = f.parts[f.data[0]]
    if part.dtype.kind == "u" and part.dtype.itemsize == 1 and f.types and f.types[0].name == "STRING":
        return bytes(part).decode("utf-8", "replace")
    if len(f.data) == 1:
        v = part.tolist()
        return v[0] if isinstance(v, list) and len(v) == 1 else v
    return None


def field_strings(reader: GGUFReader, name: str) -> list[str] | None:
    f = reader.fields.get(name)
    if f is None:
        return None
    return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]


def field_ints(reader: GGUFReader, name: str) -> list[int] | None:
    f = reader.fields.get(name)
    if f is None:
        return None
    return [int(f.parts[i][0]) for i in f.data]


def tokenizer_identity(reader: GGUFReader) -> dict:
    doc = {"model": field_scalar(reader, "tokenizer.ggml.model"), "pre": field_scalar(reader, "tokenizer.ggml.pre"),
           "tokens": field_strings(reader, "tokenizer.ggml.tokens"), "merges": field_strings(reader, "tokenizer.ggml.merges"),
           "token_type": field_ints(reader, "tokenizer.ggml.token_type"),
           "bos": field_scalar(reader, "tokenizer.ggml.bos_token_id"), "eos": field_scalar(reader, "tokenizer.ggml.eos_token_id")}
    return {"model": doc["model"], "pre": doc["pre"], "n_tokens": len(doc["tokens"] or []), "n_merges": len(doc["merges"] or []),
            "bos": doc["bos"], "eos": doc["eos"], "sha256": hashlib.sha256(canonical(doc)).hexdigest(),
            "covers": "canonical JSON of model, pre, tokens, merges, token_type, bos, eos"}


def hparams(reader: GGUFReader, arch: str) -> dict:
    out = {}
    for name in reader.fields:
        if name.startswith(arch + "."):
            v = field_scalar(reader, name)
            if v is not None:
                out[name[len(arch) + 1:]] = v
    return out


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def commit_tensor(binary: str, store: vt.WeightStore, name: str, m: int, k: int, tmp: Path) -> tuple[str, float]:
    _, _, sc, mn, q4 = vt.unpack_q4_k(store.blocks(name))
    weights = {"m": m, "k": k, "q4": q4.reshape(-1).astype(int).tolist(), "sc": sc.reshape(-1).astype(int).tolist(), "mn": mn.reshape(-1).astype(int).tolist()}
    wpath = tmp / "weights.json"
    wpath.write_text(json.dumps(weights))
    t = time.time()
    r = subprocess.run([binary, "commit", str(wpath), str(tmp / "out")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"commit failed for {name}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])["commitment"], time.time() - t


def build(a) -> dict:
    model = Path(a.model)
    store = vt.WeightStore(str(model))
    file_hash = vt.sha256_file(str(model))
    reader = GGUFReader(str(model))
    arch = field_scalar(reader, "general.architecture")
    tensors = []
    commit_total = 0.0
    n_committed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for t in reader.tensors:
            shape = [int(x) for x in t.shape]
            entry = {"name": t.name, "type": t.tensor_type.name, "shape": shape, "n_bytes": int(t.n_bytes), "sha256": store.hashes[t.name]}
            if a.commit and t.tensor_type.name == "Q4_K" and len(shape) == 2 and shape[0] % 256 == 0 and (a.limit is None or n_committed < a.limit):
                k, m = shape[0], shape[1]
                value, secs = commit_tensor(a.binary, store, t.name, m, k, Path(tmp))
                entry["commitment"] = {"scheme": COMMITMENT_SCHEME, "circuit": f"qdot(m={m},k={k})", "value": value, "seconds": round(secs, 2)}
                commit_total += secs
                n_committed += 1
                print(f"  committed {t.name} ({m} x {k}) in {secs:.1f}s: {value[:16]}...", file=sys.stderr)
            tensors.append(entry)
    manifest = {
        "manifest_version": "registration/v0",
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "name": field_scalar(reader, "general.name"), "architecture": arch, "file_type": field_scalar(reader, "general.file_type"),
            "file_sha256": file_hash, "file_bytes": model.stat().st_size, "n_tensors": len(tensors), "hparams": hparams(reader, arch),
            "source": {"url": a.source_url, "sha256": a.source_sha256, "quantize_cmd": a.quantize_cmd, "llama_cpp_rev": a.llama_rev or git_rev(HERE)},
        },
        "tensors": tensors,
        "tokenizer": tokenizer_identity(reader),
        "execution": {"statement": STATEMENT_VERSION, "llama_cpp_rev": a.llama_rev or git_rev(HERE), "backend": "cpu", "threads": 8,
                      "flash_attention": "as built (FLASH_ATTN_EXT)", "max_prompt_tokens": 64, "decoding": "greedy, lowest token id wins ties",
                      "logits": "last position only", "context": "empty at start"},
        "proof_system": {"prover": "Expander (GKR over Mersenne-31) via ExpanderCompilerCollection", "expander_rev": "096581e via luiscosio/Expander macos-build b07edf4",
                         "compiler_rev": "b5a94702d0fc3304f117d0cdba307517b8b3f6b8", "receipts_zk_rev": git_rev(HERE),
                         "commitment": {"scheme": COMMITMENT_SCHEME, "hiding": False, "parameters": "expander_pcs_init_testing_only, fixed test RNG, one commitment per Q4_K tensor over the circuit's private input layer (16 identical SIMD lanes)"},
                         "zero_knowledge": False},
        "registration": {"registrar": platform.node(), "committed_tensors": n_committed, "commit_seconds": round(commit_total, 1)},
    }
    manifest["manifest_id"] = hashlib.sha256(canonical(manifest)).hexdigest()
    return manifest


def check(a) -> int:
    manifest = json.loads(Path(a.check).read_text())
    mid = manifest.pop("manifest_id")
    problems = []
    if hashlib.sha256(canonical(manifest)).hexdigest() != mid:
        problems.append("manifest_id does not match the manifest's content")
    store = vt.WeightStore(a.model)
    file_hash = vt.sha256_file(a.model)
    reader = GGUFReader(a.model)
    if file_hash != manifest["model"]["file_sha256"]:
        problems.append(f"file digest {file_hash[:16]} vs registered {manifest['model']['file_sha256'][:16]}")
    by_name = {t.name: t for t in reader.tensors}
    if len(by_name) != len(manifest["tensors"]):
        problems.append(f"{len(by_name)} tensors in the file, {len(manifest['tensors'])} registered")
    n_commit_checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for entry in manifest["tensors"]:
            t = by_name.get(entry["name"])
            if t is None:
                problems.append(f"{entry['name']}: missing from the file"); continue
            if t.tensor_type.name != entry["type"] or [int(x) for x in t.shape] != entry["shape"]:
                problems.append(f"{entry['name']}: type or shape differs")
            if store.hashes[t.name] != entry["sha256"]:
                problems.append(f"{entry['name']}: digest differs")
            c = entry.get("commitment")
            if c and a.commit and (a.limit is None or n_commit_checked < a.limit):
                k, m = entry["shape"][0], entry["shape"][1]
                value, _ = commit_tensor(a.binary, store, t.name, m, k, Path(tmp))
                n_commit_checked += 1
                if value != c["value"]:
                    problems.append(f"{entry['name']}: commitment differs")
    if tokenizer_identity(reader)["sha256"] != manifest["tokenizer"]["sha256"]:
        problems.append("tokenizer material differs")
    print(f"checked {len(manifest['tensors'])} tensors, {n_commit_checked} commitments recomputed")
    for p in problems:
        print("  PROBLEM:", p)
    print("manifest:", "valid" if not problems else "INVALID")
    return 0 if not problems else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?")
    p.add_argument("--out")
    p.add_argument("--check", help="validate this manifest against --model")
    p.add_argument("--model", dest="model_opt")
    p.add_argument("--commit", action="store_true", help="also compute the Orion commitment of every Q4_K tensor (or check them)")
    p.add_argument("--limit", type=int, help="commit or check only the first N Q4_K tensors (testing)")
    p.add_argument("--binary", default=str(HERE / "target" / "release" / "receipts-zk"))
    p.add_argument("--source-url")
    p.add_argument("--source-sha256")
    p.add_argument("--quantize-cmd")
    p.add_argument("--llama-rev")
    a = p.parse_args(argv)
    if a.check:
        a.model = a.model_opt or a.model
        if not a.model:
            p.error("--check needs --model")
        return check(a)
    a.model = a.model or a.model_opt
    if not a.model or not a.out:
        p.error("model and --out are required")
    manifest = build(a)
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    m = manifest["model"]
    print(f"manifest {manifest['manifest_id'][:16]}... for {m['name']} ({m['architecture']}, {m['n_tensors']} tensors, file {m['file_sha256'][:16]}...), "
          f"{manifest['registration']['committed_tensors']} Q4_K tensors committed in {manifest['registration']['commit_seconds']}s, written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
