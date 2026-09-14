# llama-receipts

Inference receipts for llama.cpp, with a Merkle-committed activation trace that a verifier can check without running the model.

Two programs:

- `llama-receipts` (C++). Generates text with a replayable sampler and writes a receipt. With `--trace` it also hashes every tensor the forward pass computes into one Merkle tree and writes a sidecar with the leaves and openings for a sampled set of nodes. It can also replay a receipt by re-execution.
- `verify_trace.py` (Python, numpy, the repo's `gguf-py` and the built native vocabulary checker). Checks a trace sidecar against its receipt and the GGUF file without running inference or using a GPU.

Neither is a zero-knowledge proof. The verifier holds the weights. What this gives is a signed-able record of exactly what ran, a cheap way to catch a prover that ran a different or cheaper model, and the commit-and-open structure a proof system would slot into.

## Build and run

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --target llama-receipts -j

# prove: text, receipt, and (with --trace) the activation trace
./build/bin/llama-receipts -m model.gguf -p "The capital of France is" -n 32 --seed 42 \
    --temp 0.7 --top-k 40 --top-p 0.95 --trace --openings 32 --out receipt.json

# verify the trace without running the model
python3 examples/receipts/verify_trace.py receipt.json --model model.gguf \
    --expected-topology-sha256 TRUSTED_TOPOLOGY_SHA256

# verify by replay instead (re-runs the model; exit code 0 on accept)
./build/bin/llama-receipts -m model.gguf --replay receipt.json
```

`--trace` writes `receipt.trace.json` next to the receipt. Prompts must fit in one batch (`-b`), because the tracer counts one graph per decode call. `--challenge-seed HEX` derives the openings from a seed the verifier supplied instead of from the root alone; the verifier then passes the same `--challenge-seed`. `--claim-model other.gguf` is a test aid: the receipt and leaves describe another model while this one runs, which is how the "cheaper quantization served" rejection below was produced.

The verifier requires a topology digest pinned by the verifier from a separately trusted reference run for the same model, engine build, backend settings, prompt length and response length. The receipt records this digest for discovery, but copying it from the receipt being checked is not a security check. This external policy prevents a prover from submitting a smaller, locally consistent graph that skips model layers. The verifier also requires at least 32 openings by default; `--min-openings` changes that verifier-side policy.

The receipt is not signed by this tool. Sign the file with your own key tooling, for example:

```bash
openssl genpkey -algorithm ed25519 -out prover.pem
openssl pkeyutl -sign -inkey prover.pem -rawin -in receipt.json -out receipt.sig
openssl pkeyutl -verify -pubin -inkey prover.pub -rawin -in receipt.json -sigfile receipt.sig
```

## What the receipt contains

```
receipt_version, created_at
engine:      llama.cpp build and commit, system info, backend, n_gpu_layers, threads, n_ctx, n_batch
model:       file name and size, whole-file SHA-256, Merkle root over per-tensor SHA-256s, tensor count, GGUF metadata
request:     prompt text and tokens, sampler {type, temperature, top_k, top_p, seed}, n_predict
response:    tokens, text, per_token[]: position, token, logprob, u, boundary_distance, logits_sha256, top candidates
commitments: tokens_sha256 over the canonical JSON of prompt and response tokens
trace:       (with --trace) version, root, n_leaves, n_graphs, leaf hash rule, tracer stats
```

The sampler is deterministic by construction. The uniform draw at position `t` is `SHA-256("llama-receipts/u/v1/" || seed || t)`, ties break by token id, and temperature, top-k and top-p run in double precision. Any verifier can reproduce the draw without RNG state, so a receipt shows not just which token was chosen but how far the draw sat from the decision boundary.

## The trace

The tracer sits on `cb_eval`. Layout ops (reshape, view, permute, transpose) compute nothing and are resolved to the tensor they view. Every other node becomes a leaf:

```
g, i        graph index (prompt prefill is 0, then one per token), node index within the graph
name, op    ggml node name and op; unary and GLU ops carry the specific function
params      the op's parameters (rope frequencies, softmax scale, norm eps, ...)
out         type, ne, nb, offset into its base tensor; base {name, type, ne, nb, nbytes, sha256}
srcs[]      "weight": name, type, shape, and the per-tensor SHA-256 from the model commitment
            "data":   type, ne, nb, offset; base {..., sha256}; producer
```

`producer` is the leaf that last wrote the input's base tensor, `-1` for a graph input (tokens, positions, output row ids, KV cell indices, attention mask), or `-2` for persistent state before its first write, which is the KV cache llama.cpp zeroes. Inputs are hashed over their whole base tensor, so a view of the KV cache and the SET_ROWS node that wrote it hash the same bytes and the edge check is a string comparison. Leaves are hashed as `sha256(0x00 || canonical_json(leaf))` into one binary Merkle tree over the whole generation. Weights are never shipped; leaves name them and the verifier uses its own copy.

The prover runs twice. Pass one generates and hashes; the root goes into the receipt. Pass two replays the same tokens with the KV cache cleared, captures the bytes of the sampled leaves, and must reproduce the root bit for bit. It does, on Metal and on CPU. The openings carry each sampled leaf's Merkle path and the raw bytes of its output and data inputs.

The challenge is `k` distinct indices derived from `SHA-256(domain || root || tokens_sha256 || model file sha256 || seed || counter)`. Without a seed this is Fiat-Shamir from the commitment; with `--challenge-seed` an interactive verifier owns the randomness.

## What the verifier checks

Cheapest first. Everything before the openings needs no tensor bytes.

1. Root and content: the leaves hash to the committed root; leaf and graph counts match; token and human-readable text commitments recompute.
2. Topology: the hash-free graph structure matches the verifier's pinned topology digest.
3. Challenge: the openings are exactly the sampled indices and meet the verifier's minimum count.
4. Structure: graph and node indices are contiguous and the graph count fits the token sequence.
5. Inputs: each graph's token and position inputs hash to what the claimed prompt and response say, and the output row selection is as expected. KV cell indices, the causal mask and every other graph input must be classified and match.
6. Edges: every data input's hash equals the output hash of its producer leaf, which comes earlier; only named cache tensors may use zeroed persistent state.
7. Logits: there is one well-formed per-token record per response token, and every sampled token's graph binds to that record's logits hash.
8. Openings: each sits in the tree, its bytes hash to the leaf, its weights match the verifier's own per-tensor hashes of the GGUF by name, type, shape and hash, and re-executing the op in numpy reproduces the output.

Re-execution compares `max|out - ref| / max|ref|` against a per-op limit; CONT and SET_ROWS are compared bit for bit. Matmuls are judged against three references and take the closest: dequantized weight times float32 activation (Metal decode kernels), both operands rounded to half (Metal prefill kernels), and ggml's CPU path for Q4_K and Q6_K weights, where the activation row is quantized to Q8_K and the dot product is integer. That integer path is reproduced exactly, following `quantize_row_q8_K_ref` and `ggml_vec_dot_q4_K_q8_K`. Weights with more than 16,384 output rows (the vocabulary projection) are checked on a challenge-chosen sample of 2,048 rows.

Ops supported: GET_ROWS, RMS_NORM, NORM, ADD, MUL, SCALE, MUL_MAT (F16, F32, Q4_K, Q6_K weights; other quantized types fall back to the float references), ROPE (normal and NeoX, no YaRN), SOFT_MAX, FLASH_ATTN_EXT (no ALiBi), GLU (SwiGLU, GeGLU, ReGLU), SILU, GELU, RELU, SET_ROWS, CONT, CPY. An opened node with an unsupported op fails verification, so the coverage is visible rather than silent.

Calibrated limits (qwen2.5 1.5B Q4_K_M, Apple M5, several hundred openings per backend):

| Op | Metal | CPU | Limit |
|---|---|---|---|
| ADD, MUL, RMS_NORM, SWIGLU | about 1e-7 | about 1e-7 | 1e-5 |
| ROPE | 4e-7 | 7e-7 | 5e-5 |
| GET_ROWS, CONT, SET_ROWS | exact | exact | exact |
| MUL_MAT, decode | 4e-7 (f32) | 8e-7 (Q8_K) | 8e-3 |
| MUL_MAT, prefill | 2.2e-3 (half inputs) | 1e-7 (Q8_K) | 8e-3 |
| FLASH_ATTN_EXT | 1.9e-3 | 7.5e-3 | 2e-2 |

The remaining error is half-precision accumulation inside the kernels, which a float reference cannot reproduce: Metal's prefill matmuls feed half operands to the simdgroup units, and the fused attention kernels keep the softmax and the KV product in half.

## Results

qwen2.5 1.5B Q4_K_M, 8 to 12 tokens, 16 to 32 openings, verifier in trace-only mode (Sep 11, 2026):

| Case | Verdict | What decided it |
|---|---|---|
| Honest, Metal (flash attention on) | accept | all checks pass; worst matmul error 1.2e-3 |
| Honest, CPU, 8 threads | accept | matmuls match the Q8_K integer reference to 5e-7 |
| Replay of the Metal receipt on Metal | accept | bit-exact: every logits hash matches |
| Replay of the Metal receipt on CPU | accept | drift regime: token match 0.75, largest draw gap 0.08 |
| Q2_K model served, receipt claims Q4_K_M (`--claim-model`) | reject | 8 of 8 sampled weight matmuls fail, errors 0.2 to 0.4 |
| Honest Q4_K_M receipt checked against the Q2_K file | reject | model file hash, then weight bindings |
| One response token edited, commitments refreshed | reject | challenge indices differ; token input of the decode graph mismatches |
| Interactive seed, verifier passes the same seed | accept | |
| Interactive seed, verifier passes a different seed | reject | recorded seed is not the verifier's |

A larger matrix on the same design, run with a Python prototype of this tracer on the same kernels, is in the author's notes: 28 of 28 verdicts correct on seven cases, plus one case that documents the sampling bound below.

## Proof of one node

`zk/` proves the integer core of one opened matmul node with Expander (GKR over Mersenne-31). The weight nibbles, scales and mins are private inputs bound to a commitment registered once per tensor from the GGUF (`receipts-zk commit`); the verifier holds no weights and refuses a proof whose commitment differs. The activation quants and the per-block sums are public and range-checked before field conversion, and the receipt verifier applies the float scales itself. Not zero-knowledge yet: the commitment binds but does not hide, and the GKR has no masking. See `zk/README.md`.

## Limits

- **Sampling bound.** A cheat confined to one node with consistent edges is caught only if that node or a consumer is sampled: with `k` openings over `N` leaves the miss probability is about `1 - (1 + consumers) k / N`. Dense cheats (wrong weights, wrong model, wrong arithmetic everywhere) are what the trace is built for; a targeted single-node fabrication needs full replay or a proof system.
- **Topology policy.** A verifier must provision the expected topology digest independently. The digest is request-shape-specific; accepting the value from the receipt under review removes this protection.
- **Sampler tampering is out of scope for the trace.** Temperature, greedy or seed tampering leave every activation honest. The trace ties its logits to the receipt; `--replay` judges the sampler.
- **The verifier holds the weights.** It needs the GGUF to bind weights and re-execute matmuls. Not zero-knowledge.
- **Cost.** Observing every node splits the graph into single-node dispatches with a sync each. Tracing a 12-token generation of this 1.5B model takes about 3 seconds per pass on Metal; hashing the model file once is another 4 seconds. Trace verification takes 2 to 3 seconds and is dominated by dequantizing the opened weights; at this model size a replay is cheaper. The trace verifier's cost is fixed by `k`, replay grows with the model.
- **Sidecar size.** Openings are base64 of raw tensor bytes; a KV cache opening at `-c 1024` is about half a megabyte. A 12-token receipt with 32 openings is around 10 MB.
- **Signatures.** Out of scope here; sign the receipt file externally.

## Formal specification

`spec/` holds the receipt protocol and the Q4_K, Q6_K and Q8_K arithmetic as executable Lean 4 definitions with proofs of the integer-side properties, plus a checker that recomputes a real trace's root, challenge, edges and inputs and reproduces the verifier's arithmetic on test vectors. See `spec/README.md`.

## Relation to other work

Signed receipts with model hash, seed and decode policy, checked by byte-equality replay, exist in several projects. Tolerant recompute with a per-token metric is Token-DiFR (Karvonen, Rinberg et al., 2025); the two-regime rule in `--replay` (strict when logits hashes match, tolerant otherwise) and the hash-derived draws are this tool's variant of it. Commit-and-open verification of a forward pass at layer granularity is "Lightweight Cryptographic Proofs of Inference" (Anchuri, Campanelli, Gennaro et al., SaTML 2026); this is the same shape at GGML node granularity on the engine's real kernels, with every edge of the graph bound and the inputs regenerated from the receipt.

Receipt content checks: replay rejects unknown receipt versions and verifies prompt tokenization and response token pieces (including special tokens). `llama-receipts -m model.gguf --check-content receipt.json` performs only structure and text/token checks using a vocabulary-only load; it does not verify inference or the model file commitment. The Python trace verifier calls this helper and separately checks the GGUF digest and trace. Build the native executable before running `verify_trace.py`.

Response text is the UTF-8 decoding of the complete concatenated token-byte stream, with malformed or incomplete sequences displayed as U+FFFD. Token pieces are combined before decoding, so characters split across tokens remain intact. Exact token IDs are still committed separately. This lets a token-limited generation stop inside a character without crashing JSON serialization; content verification applies the same decoding policy.
