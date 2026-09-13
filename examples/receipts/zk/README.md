# receipts-zk: a GKR proof of one traced matmul node

Proof of concept for the next rung of `llama-receipts`: instead of re-executing an opened node in numpy, the verifier checks a cryptographic proof that the node's integer arithmetic was done correctly. The prover is [Expander](https://github.com/PolyhedraZK/Expander) (GKR over the Mersenne-31 field) through its circuit compiler; the statement is exactly ggml's CPU arithmetic for a `Q4_K` weight against a `Q8_K`-quantized activation, as specified in `../spec`.

## The statement

For an output row `m` and a 256-wide block `i` of the weight:

```
s1[m,i] = Σ_{j<8}  sc[m,i,j] · Σ_{l<32} q4[m,i,32j+l] · q8[i,32j+l]
s2[m,i] = Σ_{j<16} bsums[i,j] · mn[m,i,j/2]        bsums[i,j] = Σ_{l<16} q8[i,16j+l]
```

The weight's nibbles `q4` (0..15), sub-block scales `sc` and mins `mn` (0..63) are private inputs. Expander commits to the circuit's private input layer and writes that commitment into the proof; `receipts-zk commit` computes the same commitment from the weights alone, which is what a registration publishes once per tensor, and `receipts-zk verify --commitment` refuses a proof whose commitment is not the registered one before any sumcheck runs. The activation quants `q8` (-127..127) and the per-block sums `s1`, `s2` are public and range-checked by prover and verifier before field conversion. The verifier then applies the per-block float scales itself, `sum_i d_a[i] * (d[m,i] * s1[m,i] - dmin[m,i] * s2[m,i])`, and compares with the node's opened output. The integer core, millions of multiplications, is what the proof covers.

Not zero-knowledge yet. Orion's commitment is a hash-based root without blinding, so it binds the weights but does not hide them, and Expander's GKR has no masking. What this establishes is model binding with a verifier that holds no weights. Weight ranges need no in-circuit constraints: the registration encodes the GGUF bytes, and nibbles and six-bit fields are in range by construction; the binding to the registered commitment is what keeps a prover from substituting values.

Every per-block sum is below 2^25 in magnitude (proved in `../spec`, `s1_lt_2_pow_25` and `s2_bound`), so arithmetic in the M31 field equals integer arithmetic and negatives are read back as `p − |x|`.

## Build and run

Rust nightly 2025-05-17 (pinned in `rust-toolchain.toml`) and an MPI library (`brew install open-mpi` on macOS; Expander links against MPI).

```bash
cd examples/receipts/zk
cargo build --release

# register, prove and verify a node from a receipt's trace, run the negative cases, check the float step
python3 zk_node.py receipt.json --model model.gguf --out zk-out            # first decode-graph Q4_K matmul
python3 zk_node.py receipt.json --model model.gguf --index 779 --out zk-out # a specific opened leaf
python3 zk_node.py receipt.json --model model.gguf --index 779 --out zk-out --verify-only # verify existing files against the registered commitment

# the three roles by hand
receipts-zk commit weights.json registered/            # registration: registered/commitment.hex from {m, k, q4, sc, mn}
receipts-zk prove  witness.json zk-out                 # proof.bin, public.json {q8, s1, s2}, commitment.hex
receipts-zk verify zk-out/public.json zk-out/proof.bin orion --commitment $(cat registered/commitment.hex)

# the circuit against the Lean specification on the same vectors
python3 ../spec/gen_vectors.py receipt.json --model model.gguf --out vectors.json
../spec/.lake/build/bin/spec-check vectors vectors.json --emit lean.json
python3 diff_spec.py vectors.json lean.json
```

`zk_node.py` reads the opening's bytes and the GGUF, quantizes the activation to `Q8_K` with the same code the verifier uses, registers the tensor's commitment from the weights alone (`registered/commitment.hex`; in a deployment this happens once per model, elsewhere), writes `witness.json`, runs `receipts-zk prove`, runs `receipts-zk verify --commitment` with the registered value, and then the negative cases: a tampered public sum, a proof made with one changed nibble and consistent sums (its commitment differs), and the honest proof against the registered commitment of the same tensor in the next layer. Finally it applies the float step. `--verify-only` repeats the verification of an existing directory. `diff_spec.py` feeds the circuit the sums the Lean spec computed (`spec-check vectors --emit`) instead of Python's, so the circuit is checked against the specification. Receipts must come from the CPU backend for the integer statement to describe the kernel that ran; Metal's kernels use float32 activations.

`orion` is the configuration that commits. `raw` ships the witness inside the proof: useful to debug the circuit, meaningless as a proof, and `commit` refuses it.

## Zero-knowledge proofs of the same statement (Groth16)

Expander has no zero-knowledge layer, so `groth16/` proves the same integer core with an established one: a circom circuit (`qdot_rows.circom`) and Groth16 through snarkjs. The weights of a group of rows (64 rows for K up to 1536, 32 for K = 2048) are private inputs as bits, so every nibble and six-bit field is in range by construction; a Poseidon chain over the packed weights and a salt is the public commitment; the activation quants and the per-block sums are public inputs. Groth16 proofs are zero-knowledge and about 800 bytes, and `snarkjs.groth16.verify` runs in a browser.

```bash
cd groth16 && npm install                                   # snarkjs, circomlib, circomlibjs; circom from cargo install --git https://github.com/iden3/circom
# one instance per (rows, K); build/r64_k1536, build/r64_k1024, build/r32_k2048 are the shapes of the two models here
./setup.sh build/r64_k1536 ptau/pot20_final.ptau            # Groth16 setup and one phase-2 contribution: main_final.zkey, verification_key.json
python3 groth16_node.py receipt.json --model model.gguf --index N [--manifest manifest.json] [--groups 0,1] --out groth16-out
python3 ../register.py --augment manifest.json --model model.gguf --groth16   # one Poseidon commitment per row group, per Q4_K tensor
```

`groth16_node.py` proves each row group of the opened node, verifies it, checks the proof's commitment against the registered one (from `--manifest`, or recomputed with `commit.js`), and runs the negatives: a tampered public sum and another group's commitment. The powers of tau are generated locally (`snarkjs powersoftau`, 2^20) because the public Hermez mirrors no longer serve the files; together with the single phase-2 contribution these are proof-of-concept parameters, and a deployment would use a ceremony. What the proof does not cover is unchanged: the float scales, the other operations of the token, and the sampled output.

## Registration

`register.py` writes the manifest a verifier pins before checking any proof: the model file (digest, size, architecture and hyperparameters, file type, source and requantization command), every tensor (name, type, shape, bytes, SHA-256), the tokenizer material (a digest over tokens, merges, token types and the pre-tokenizer name), the execution specification, the proof system's identity, and with `--commit` the Orion commitment of every `Q4_K` tensor as the circuit lays it out. The manifest's id is the SHA-256 of its canonical JSON.

```bash
python3 register.py model.gguf --out manifest.json --commit --source-url URL --source-sha256 HEX --quantize-cmd "llama-quantize ..."
python3 register.py --check manifest.json --model other-copy.gguf [--commit]   # recompute and compare; exit 0 only if nothing differs
python3 zk_node.py receipt.json --model model.gguf --index N --manifest manifest.json --out zk-out   # verifier takes the commitment from the manifest
```

A changed weight, tensor type, shape, tokenizer or execution definition changes the id, and `--check` names the tensor whose digest or commitment differs. The commitments bind but do not hide, and Expander's commitment parameters are its testing-only ones, so the manifest names the scheme and the parameters and a hiding scheme can be added later as another entry.

## Results

Apple M5, qwen2.5 1.5B Q4_K_M, one activation row from a CPU trace, private weights under a registered Orion commitment (Sep 13, 2026):

| Node | Weight | Private inputs | Register | Compile | Prove | Proof | Verify | Float step error | Rejected |
|---|---|---|---|---|---|---|---|---|---|
| `Vcur-22`, 256 x 1536 | `blk.22.attn_v.weight` | 417,792 | 2.0 s | 1.8 s | 0.28 s | 6.5 MB | 0.04 s | 3.8e-7 | tampered sum, other weights, other tensor |

The circuit has four layers. Verification recompiles the circuit from its shape, the compile column; caching the compiled circuit per shape would remove that cost. Expander packs 16 SIMD lanes, so batching 16 different rows or nodes into those lanes is free throughput left on the table. Registration time is the Orion encoding and Merkle tree over the private layer.

## What this establishes

- llama.cpp's native block-quantized arithmetic can be stated as a circuit without dequantizing to float, and proven with an existing GKR prover in seconds for one node.
- The proof is bound to the model: its commitment to the private weights must equal a value registered once per tensor from the GGUF, and the verifier needs no weights, only that value, the public inputs and the proof.
- Prover and verifier check the Q8_K range and the proven sum bounds on every public input before field conversion, so a value that would alias in Mersenne-31 is refused rather than proven. Weight ranges hold by construction of the registration encoding.
- The circuit accepts exactly the sums the Lean spec computes and rejects a one-off change (`diff_spec.py`).
- The sums the circuit outputs are exactly the ones the Lean spec defines and bounds.

Not yet: zero knowledge (masking in the GKR, a hiding commitment), the other ops of a layer, Q6_K matmuls, proving every matmul of a token instead of one node, batching, published commitment parameters instead of Expander's testing-only ones, and a GPU prover. The per-node circuit is also the natural unit for a proof that the circuit matches the Lean spec. The Stage 1 record in the project's `docs/stage1/` lists what each of these needs.

## Dependencies and licence

Expander and its compiler collection are AGPL-3.0. They are pulled by git revision through Cargo, not vendored. The build uses a fork at `luiscosio/Expander`, branch `macos-build`, which differs from upstream by two small changes needed to link on macOS: the huge-page `madvise` call is Linux-only, and the CUDA feature gates are Linux-only with no-op stubs for the GPU symbols elsewhere. `Cargo.lock` pins the exact revisions.
