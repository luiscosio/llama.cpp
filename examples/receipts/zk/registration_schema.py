"""Structural validation of public registrations; GGUF correspondence is checked by register.py."""
import hashlib
import json
import re

VERSION = "registration/v1"
GROTH16_SCHEME = "groth16-poseidon/bn254/qdot_rows"
ORION_SCHEME = "expander-orion/m31x16/receipts-zk-qdot"
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hex_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def scalar(value):
    return isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]{0,76}", value) is not None and int(value) < P


def validate_manifest(manifest):
    try:
        require(isinstance(manifest, dict) and manifest.get("manifest_version") == VERSION, "unsupported registration version")
        content = {k: v for k, v in manifest.items() if k != "manifest_id"}
        require(hex_digest(manifest.get("manifest_id")) and hashlib.sha256(canonical(content)).hexdigest() == manifest["manifest_id"], "manifest_id differs from content")
        model, tensors = manifest["model"], manifest["tensors"]
        require(isinstance(tensors, list) and len(tensors) > 0, "missing tensor table")
        names = [t["name"] for t in tensors]
        require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names), "duplicate or invalid tensor identity")
        require(model["n_tensors"] == len(tensors) and hex_digest(model["file_sha256"]), "invalid model identity")
        ex = manifest["execution"]
        require(ex["statement"] == "stmt/v0" and ex["backend"] == "cpu" and type(ex["threads"]) is int and ex["threads"] == 8,
                "unsupported execution definition")
        require(ex["max_prompt_tokens"] == 64 and ex["decoding"] == "greedy, lowest token id wins ties" and ex["context"] == "empty at start" and
                ex["logits"] == "last position only" and ex["flash_attention"] == "as built (FLASH_ATTN_EXT)", "unsupported execution policy")
        require(isinstance(ex["llama_cpp_rev"], str) and re.fullmatch(r"[0-9a-f]{7,40}", ex["llama_cpp_rev"]) is not None, "execution revision must be pinned")
        require(hex_digest(manifest["tokenizer"]["sha256"]), "invalid tokenizer digest")
        proof_system = manifest["proof_system"]
        require(proof_system["zero_knowledge"] is False and proof_system["commitment"]["scheme"] == ORION_SCHEME and
                proof_system["commitment"]["hiding"] is False, "unsupported Expander proof definition")
        require(proof_system["compiler_rev"] == "b5a94702d0fc3304f117d0cdba307517b8b3f6b8" and
                proof_system["expander_rev"] == "096581e via luiscosio/Expander macos-build b07edf4", "unsupported proof backend revision")
        circuits = set()
        for t in tensors:
            shape = t["shape"]
            require(isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape), "invalid tensor shape")
            require(type(t["n_bytes"]) is int and t["n_bytes"] > 0 and hex_digest(t["sha256"]), "invalid tensor bytes or digest")
            if "commitment" in t:
                c = t["commitment"]
                require(t["type"] == "Q4_K" and len(shape) == 2 and shape[0] % 256 == 0, "unsupported Orion tensor")
                require(c["scheme"] == ORION_SCHEME and c["circuit"] == f"qdot(m={shape[1]},k={shape[0]})" and hex_digest(c["value"]), "invalid Orion definition")
            if "groth16" in t:
                g = t["groth16"]
                require(t["type"] == "Q4_K" and len(shape) == 2 and shape[0] in (1024, 1536, 2048, 3072), "unsupported Groth16 tensor")
                rows = 16 if shape[0] <= 1536 else 8
                require(type(g["rows_per_group"]) is int and g["rows_per_group"] == rows and shape[1] % rows == 0, "invalid row grouping")
                circuit = f"r{rows}_k{shape[0]}"
                require(g["scheme"] == GROTH16_SCHEME and g["circuit"] == circuit and scalar(g["salt"]), "invalid Groth16 definition")
                require(isinstance(g["groups"], list) and len(g["groups"]) == shape[1] // rows and all(scalar(c) for c in g["groups"]), "invalid group commitments")
                circuits.add(circuit)
        if circuits:
            ps = manifest["proof_system"]["groth16"]
            require(ps["zero_knowledge"] is True and ps["circuit"] == "groth16/qdot_rows.circom", "unsupported Groth16 proof definition")
            require(ps["scheme"] == GROTH16_SCHEME and set(ps["verification_keys"]) == circuits, "missing or extra registered verification keys")
            require(all(hex_digest(v) for v in ps["verification_keys"].values()), "invalid key digest")
            require(all(hex_digest(ps[k]) for k in ("circuit_sha256", "setup_sha256", "verifier_sha256")), "missing proof material digests")
        reg = manifest["registration"]
        require(reg["committed_tensors"] == sum("commitment" in t for t in tensors), "incorrect Orion coverage count")
        require(reg.get("groth16_tensors", 0) == sum("groth16" in t for t in tensors), "incorrect Groth16 coverage count")
    except (KeyError, TypeError, AttributeError, OverflowError) as e:
        raise ValueError("malformed registration") from e
