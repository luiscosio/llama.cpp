# receipts-spec: the receipt protocol and its arithmetic, in Lean 4

A machine-readable statement of what `llama-receipts` commits to and what `verify_trace.py` checks, written as executable Lean 4 definitions with proofs of the properties that do not depend on floating point. It is the specification a proof system for these receipts would have to satisfy, and it is checked against real receipts produced by the C++ tool.

No Mathlib. Lean 4.33.1, `lake build` takes seconds.

## What is specified

`ReceiptsSpec/Quant.lean`, the arithmetic of one matmul node:

- Byte-exact block layouts for `block_q4_K` (144 bytes), `block_q6_K` (210 bytes) and `block_q8_K`, including the packed 6-bit scale and min fields (`get_scale_min_k4`) and the nibble order the kernels consume.
- `quantizeQ8K`: ggml's `quantize_row_q8_K_ref` in `Float32`, first-occurrence maximum, `iscale = -127 / max`, round half to even, `d = 1 / iscale`.
- The integer core: `s1 = Σ_j scale_j · Σ_l q4·q8`, `s2 = Σ_j bsums_j · min_{j/2}`, and the Q6_K dot. This is what the GKR circuit in `../zk` constrains.
- The float rim: the per-block combination `d·d_a·s1 − dmin·d_a·s2`, evaluated in `Float`.

`ReceiptsSpec/Merkle.lean`, the commitment and the challenge:

- `leafHash`, `nodeHash`, `pairUp`, `root`: the Merkle tree with `H(0x00 ‖ leaf)`, `H(0x01 ‖ l ‖ r)`, odd levels duplicating the last node, over an abstract hash `H`.
- `verifyPath`: audit path verification.
- `deriveIndices`: the Fiat-Shamir sample, counter mode over `H(domain ‖ root ‖ binding ‖ seed ‖ counter)`, duplicates skipped, sorted.

`ReceiptsSpec/Protocol.lean`, the trace:

- `Leaf` and `Src` as parsed from a sidecar, with `canonicalBytes` being sorted-key compact JSON (the bytes nlohmann's `dump()` and Python's `json.dumps(sort_keys=True, separators=(",", ":"))` produce; Lean's `Json.compress` gives the same).
- `traceRoot`, `checkEdges` (every data input equals its producer's output hash, or the zeroed KV cache), `checkInputs` (token and position inputs regenerated from the receipt), `challengeIndices`.

`ReceiptsSpec/Sha256.lean`: FIPS 180-4 over `ByteArray`, so the checker needs no foreign library. Established by test vectors, not by proof; the theorems take `H` as a parameter.

## What is proved

All theorems are Mathlib-free and `sorry`-free.

| Theorem | Statement |
|---|---|
| `nibble_lt_16`, `packedLo_lt_64`, `packedHi_lt_64`, `sixBits_lt_64` | every decoded bit field is in range |
| `scaleMin_fst_lt_64`, `scaleMin_snd_lt_64`, `q4_lt_16`, `q6_bounds` | scales and mins are 6-bit, Q4_K values 4-bit, Q6_K values in `-32..31` |
| `mul_bound`, `foldl_add_bound` | products and left folds of bounded terms are bounded |
| `subDot_bound`, `s1_bound`, `s1_lt_2_pow_25`, `s2_bound` | for activations in `±127`, every `s1` is below `2^25` in magnitude and every `s2` below `2,048,256`, so Mersenne-31 field arithmetic in the circuit never wraps |
| `pairUp_length` | one Merkle level up has `⌈n/2⌉` nodes |
| `root_singleton`, `root_pair`, `verifyPath_singleton`, `verifyPath_pair_left`, `verifyPath_pair_right` | roots and audit paths for one- and two-leaf trees |
| `deriveIndices_lt`, `deriveIndices_nodup` | every sampled index is a valid leaf index and no leaf is sampled twice |

Not proved here, stated as the target: the sampling bound. If the verifier accepts and a fraction `f` of leaves is inconsistent, the probability that none of `k` sampled indices hit one is at most `(1 − f)^k`, assuming `H` is collision resistant and the challenge is a random oracle. That statement needs a probability model and would build on Mathlib. General audit-path completeness for trees of any size is also left open.

## Checking the spec against reality

```bash
lake build
./.lake/build/bin/spec-check sha256                                 # FIPS test vectors
./.lake/build/bin/spec-check trace receipt.json receipt.trace.json TRUSTED_TOPOLOGY_SHA256
python3 gen_vectors.py receipt.json --model model.gguf --out vectors.json
./.lake/build/bin/spec-check vectors vectors.json                  # Q8_K quants, scales, s1, s2, dot, row value, canonical JSON
./.lake/build/bin/spec-check vectors vectors.json --emit lean.json # also write what the spec computed, for ../zk/diff_spec.py
```

The topology digest must come from a separately trusted reference run for the same model, engine settings and request shape. On a receipt from `llama-receipts` (qwen2.5 1.5B Q4_K_M, CPU, 12 tokens, 32 openings, Sep 11, 2026):

| Check | Result |
|---|---|
| Merkle root over 8,086 canonical leaves | equals the receipt's committed root |
| Topology | equals the verifier's pinned digest |
| Content commitments | token arrays and displayed text recompute |
| Fiat-Shamir sample | equals the 32 opened indices and meets policy |
| Data edges | 0 of 10,621 inconsistent |
| Token inputs, position inputs | 13 of 13 and 728 of 728 match the receipt's tokens |
| Q4_K rows: Q8_K quants, `d` bit patterns, `s1`, `s2` | all equal to the Python verifier's |
| Canonical JSON, 14 documents with control characters and non-ASCII | byte-identical to Python's `json.dumps` |
| The proof circuit (`../zk/diff_spec.py`) on the spec's `s1`, `s2` | accepted for every row; a one-off change rejected |
| Q4_K row value in `Float` | relative error 0 against the Python float64 reference |
| Q6_K dot | equal |

The trace check runs in under half a second, so the Lean checker is a usable independent verifier of the hash-level protocol, not only a document.

## How this relates to the other pieces

- `../receipts.cpp` produces the artifacts; `../verify_trace.py` is the full verifier (it also re-executes opened nodes, which this spec does not).
- `../zk` proves the integer core of one node with Expander. `s1_lt_2_pow_25` and `s2_bound` are the reason its M31 circuit is sound as integer arithmetic.
- `Float32` operations are opaque to Lean's logic. The float rim is specified by executable code and checked by differential testing; a proof about it would need a formal IEEE 754 model.
