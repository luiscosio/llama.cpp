# receipts-zk: a GKR proof of one traced matmul node

Proof of concept for the next rung of `llama-receipts`: instead of re-executing an opened node in numpy, the verifier checks a cryptographic proof that the node's integer arithmetic was done correctly. The prover is [Expander](https://github.com/PolyhedraZK/Expander) (GKR over the Mersenne-31 field) through its circuit compiler; the statement is exactly ggml's CPU arithmetic for a `Q4_K` weight against a `Q8_K`-quantized activation, as specified in `../spec`.

## The statement

For an output row `m` and a 256-wide block `i` of the weight:

```
s1[m,i] = Σ_{j<8}  sc[m,i,j] · Σ_{l<32} q4[m,i,32j+l] · q8[i,32j+l]
s2[m,i] = Σ_{j<16} bsums[i,j] · mn[m,i,j/2]        bsums[i,j] = Σ_{l<16} q8[i,16j+l]
```

The weight's nibbles `q4` (0..15), sub-block scales `sc` and mins `mn` (0..63) are private inputs, committed by the proof. The activation quants `q8` (−127..127) and the per-block sums `s1`, `s2` are public. The verifier of a receipt then applies the per-block float scales itself, `Σ_i d_a[i]·(d[m,i]·s1[m,i] − dmin[m,i]·s2[m,i])`, a few thousand multiply-adds, and compares with the node's opened output. The integer core, millions of multiplications, is what the proof covers.

Every per-block sum is below 2^25 in magnitude (proved in `../spec`, `s1_lt_2_pow_25` and `s2_bound`), so arithmetic in the M31 field equals integer arithmetic and negatives are read back as `p − |x|`.

## Build and run

Rust nightly 2025-05-17 (pinned in `rust-toolchain.toml`) and an MPI library (`brew install open-mpi` on macOS; Expander links against MPI).

```bash
cd examples/receipts/zk
cargo build --release

# prove and verify a node from a receipt's trace, then check the float step against the opened output
python3 zk_node.py receipt.json --model model.gguf --out zk-out            # first decode-graph Q4_K matmul
python3 zk_node.py receipt.json --model model.gguf --index 779 --out zk-out # a specific opened leaf
```

`zk_node.py` reads the opening's bytes and the verifier's GGUF, quantizes the activation to `Q8_K` with the same code the verifier uses, writes `witness.json`, runs `receipts-zk prove`, runs `receipts-zk verify` with the private inputs zeroed, verifies a tampered public sum is rejected, and applies the float step. Receipts must come from the CPU backend for the integer statement to describe the kernel that ran; Metal's kernels use float32 activations.

Two configurations. `orion` (default) commits to the private inputs with a hash-based polynomial commitment; the verifier never sees the weights. `raw` ships the witness inside the proof; it verifies, but it hides nothing and the proof is twice as large.

## Results

Apple M5, qwen2.5 1.5B Q4_K_M, one activation row from a CPU trace, Orion commitment (Sep 11, 2026):

| Node | Weight | Private inputs | Compile | Witness | Prove | Proof | Verify | Float step error | Tampered sum |
|---|---|---|---|---|---|---|---|---|---|
| `Kcur-0`, 256 x 1536 | `blk.0.attn_k.weight` | 417,792 | 1.7 s | 0.05 s | 0.28 s | 6.5 MB | 0.08 s | 2.7e-7 | rejected |
| `Qcur-7`, 1536 x 1536 | `blk.7.attn_q.weight` | 2,506,752 | 15.2 s | 0.45 s | 2.5 s | 16.0 MB | 0.40 s | 1.8e-7 | rejected |

The circuit has four layers. Verification recompiles the circuit from its shape, which is the 1.7 s and 15 s in the compile column; caching the compiled circuit per shape would remove that. Expander packs 16 SIMD lanes, so the prover proves 16 identical copies of the witness; batching 16 different rows or nodes into those lanes is free throughput left on the table. With the `raw` configuration the `Kcur-0` proof is 34 MB.

## What this establishes

- llama.cpp's native block-quantized arithmetic can be stated as a circuit without dequantizing to float, and proven with an existing GKR prover in seconds for one node.
- The proof slots into the receipt design unchanged: the opening's bytes become the witness, the leaf's committed hashes bind the public inputs, and the verifier's remaining work is the float step.
- The sums the circuit outputs are exactly the ones the Lean spec defines and bounds.

Not yet: the other ops of a layer, proving every matmul of a token instead of one node, batching, a GPU prover, and publishing the weight commitment once per model so the verifier does not need the GGUF. The per-node circuit is also the natural unit for a proof that the circuit matches the Lean spec.

## Dependencies and licence

Expander and its compiler collection are AGPL-3.0. They are pulled by git revision through Cargo, not vendored. The build uses a fork at `luiscosio/Expander`, branch `macos-build`, which differs from upstream by two small changes needed to link on macOS: the huge-page `madvise` call is Linux-only, and the CUDA feature gates are Linux-only with no-op stubs for the GPU symbols elsewhere. `Cargo.lock` pins the exact revisions.
