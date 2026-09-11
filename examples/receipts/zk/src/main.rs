//! receipts-zk: prove one traced llama.cpp matmul node with Expander (GKR).
//!
//! The statement is ggml's CPU arithmetic for a Q4_K weight against a Q8_K-quantized
//! activation row, with the float scaling left outside the circuit:
//!
//!   for every output row m and 256-wide block i:
//!     s1[m,i] = sum_{j<8} sc[m,i,j] * sum_{l<32} q4[m,i,32j+l] * q8[i,32j+l]
//!     s2[m,i] = sum_{j<16} bsums[i,j] * mn[m,i,j/2],   bsums[i,j] = sum_{l<16} q8[i,16j+l]
//!
//! q4 (0..15), sc and mn (0..63) are the weight's integers and are private inputs; q8
//! (-127..127) and the per-block sums s1, s2 are public. The verifier of a receipt
//! applies the per-block float scales to s1 and s2 itself (see zk_node.py), which is
//! the cheap part; the integer core is what the proof covers.
//!
//! Integers live in the Mersenne-31 field. Every per-block sum is bounded by
//! 256 * 127 * 945 < 2^25, far from the modulus, so field arithmetic equals integer
//! arithmetic and negatives are recovered as p - |x|.
//!
//!   receipts-zk prove  witness.json out_dir  [raw|orion]
//!   receipts-zk verify public.json  proof.bin [raw|orion]

use std::fs;
use std::io::{BufReader, BufWriter, Cursor, Read};
use std::path::Path;
use std::time::Instant;

use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::MPIConfig;
use serde::{Deserialize, Serialize};
use serdes::ExpSerde;

const QK_K: usize = 256;

declare_circuit!(QDot {
    m: usize,
    k: usize,
    q4: [Variable],       // m * k nibbles
    sc: [Variable],       // m * nb * 8
    mn: [Variable],       // m * nb * 8
    q8: [PublicVariable], // k
    s1: [PublicVariable], // m * nb
    s2: [PublicVariable], // m * nb
});

fn sum_tree<C: Config, B: RootAPI<C>>(api: &mut B, mut xs: Vec<Variable>) -> Variable {
    if xs.is_empty() {
        return api.constant(0);
    }
    while xs.len() > 1 {
        let mut next = Vec::with_capacity(xs.len().div_ceil(2));
        for pair in xs.chunks(2) {
            next.push(if pair.len() == 2 { api.add(pair[0], pair[1]) } else { pair[0] });
        }
        xs = next;
    }
    xs[0]
}

impl<C: Config> Define<C> for QDot<Variable> {
    fn define<B: RootAPI<C>>(&self, api: &mut B) {
        let (m, k) = (self.m, self.k);
        let nb = k / QK_K;
        // group sums of the activation, shared by every output row
        let mut bsums = Vec::with_capacity(nb * 16);
        for i in 0..nb {
            for j in 0..16 {
                let terms: Vec<Variable> = (0..16).map(|l| self.q8[i * QK_K + j * 16 + l]).collect();
                bsums.push(sum_tree(api, terms));
            }
        }
        for r in 0..m {
            for i in 0..nb {
                let mut scaled = Vec::with_capacity(8);
                for j in 0..8 {
                    let prods: Vec<Variable> = (0..32)
                        .map(|l| {
                            let idx = i * QK_K + j * 32 + l;
                            api.mul(self.q4[r * k + idx], self.q8[idx])
                        })
                        .collect();
                    let sub = sum_tree(api, prods);
                    scaled.push(api.mul(sub, self.sc[(r * nb + i) * 8 + j]));
                }
                let s1 = sum_tree(api, scaled);
                api.assert_is_equal(s1, self.s1[r * nb + i]);
                let mins: Vec<Variable> = (0..16).map(|j| api.mul(bsums[i * 16 + j], self.mn[(r * nb + i) * 8 + j / 2])).collect();
                let s2 = sum_tree(api, mins);
                api.assert_is_equal(s2, self.s2[r * nb + i]);
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct WitnessFile {
    m: usize,
    k: usize,
    q4: Vec<u8>,
    sc: Vec<u8>,
    mn: Vec<u8>,
    q8: Vec<i32>,
    s1: Vec<i64>,
    s2: Vec<i64>,
}

#[derive(Serialize, Deserialize)]
struct PublicFile {
    m: usize,
    k: usize,
    config: String,
    q8: Vec<i32>,
    s1: Vec<i64>,
    s2: Vec<i64>,
}

fn fe<C: Config>(x: i64) -> CircuitField<C> {
    assert!(x.unsigned_abs() < (1u64 << 31), "value {} does not fit the field headroom", x);
    let mag = CircuitField::<C>::from(x.unsigned_abs() as u32);
    if x < 0 {
        CircuitField::<C>::zero() - mag
    } else {
        mag
    }
}

fn shape_circuit(m: usize, k: usize) -> QDot<Variable> {
    let nb = k / QK_K;
    QDot {
        m,
        k,
        q4: vec![Variable::default(); m * k],
        sc: vec![Variable::default(); m * nb * 8],
        mn: vec![Variable::default(); m * nb * 8],
        q8: vec![Variable::default(); k],
        s1: vec![Variable::default(); m * nb],
        s2: vec![Variable::default(); m * nb],
    }
}

fn assignment<C: Config>(w: &WitnessFile, zero_private: bool) -> QDot<CircuitField<C>> {
    let z = |v: i64| if zero_private { CircuitField::<C>::zero() } else { fe::<C>(v) };
    QDot {
        m: w.m,
        k: w.k,
        q4: w.q4.iter().map(|&x| z(x as i64)).collect(),
        sc: w.sc.iter().map(|&x| z(x as i64)).collect(),
        mn: w.mn.iter().map(|&x| z(x as i64)).collect(),
        q8: w.q8.iter().map(|&x| fe::<C>(x as i64)).collect(),
        s1: w.s1.iter().map(|&x| fe::<C>(x)).collect(),
        s2: w.s2.iter().map(|&x| fe::<C>(x)).collect(),
    }
}

fn check_shape(w: &WitnessFile) {
    assert!(w.k % QK_K == 0, "k must be a multiple of 256");
    let nb = w.k / QK_K;
    assert_eq!(w.q4.len(), w.m * w.k);
    assert_eq!(w.sc.len(), w.m * nb * 8);
    assert_eq!(w.mn.len(), w.m * nb * 8);
    assert_eq!(w.q8.len(), w.k);
    assert_eq!(w.s1.len(), w.m * nb);
    assert_eq!(w.s2.len(), w.m * nb);
}

fn prove<C: Config>(w: &WitnessFile, out_dir: &Path, config_name: &str) {
    check_shape(w);
    let nb = w.k / QK_K;
    eprintln!("circuit: m={} k={} blocks={} private inputs={} public inputs={}", w.m, w.k, nb, w.m * w.k + 2 * w.m * nb * 8, w.k + 2 * w.m * nb);

    let t = Instant::now();
    let compiled: CompileResult<C> = compile(&shape_circuit(w.m, w.k), CompileOptions::default()).expect("compile");
    let n_layers = compiled.layered_circuit.layer_ids.len();
    eprintln!("compiled in {:.2}s: {} layers", t.elapsed().as_secs_f64(), n_layers);

    let t = Instant::now();
    let n_pack = SIMDField::<C>::PACK_SIZE;
    let assign = assignment::<C>(w, false);
    let assigns = vec![assign; n_pack];
    let witness = compiled.witness_solver.solve_witnesses(&assigns).expect("witness");
    let ok = compiled.layered_circuit.run(&witness);
    assert!(ok.iter().all(|x| *x), "the witness does not satisfy the circuit: the claimed sums are wrong");
    eprintln!("witness solved in {:.2}s (SIMD pack of {} identical copies), circuit satisfied", t.elapsed().as_secs_f64(), n_pack);

    let t = Instant::now();
    let mut circuit = compiled.layered_circuit.export_to_expander_flatten();
    let mpi = MPIConfig::prover_new(None, None);
    let (simd_input, simd_public) = witness.to_simd();
    circuit.layers[0].input_vals = simd_input;
    circuit.public_input = simd_public;
    circuit.evaluate();
    let (claimed_v, proof) = executor::prove::<C>(&mut circuit, mpi);
    let prove_s = t.elapsed().as_secs_f64();

    fs::create_dir_all(out_dir).expect("out dir");
    let mut bytes = Vec::new();
    proof.serialize_into(&mut bytes).expect("proof");
    claimed_v.serialize_into(&mut bytes).expect("claimed value");
    fs::write(out_dir.join("proof.bin"), &bytes).expect("write proof");
    let public = PublicFile { m: w.m, k: w.k, config: config_name.to_string(), q8: w.q8.clone(), s1: w.s1.clone(), s2: w.s2.clone() };
    serde_json::to_writer(BufWriter::new(fs::File::create(out_dir.join("public.json")).unwrap()), &public).expect("public");
    eprintln!("proved in {:.2}s: proof {} bytes, layers {}", prove_s, bytes.len(), n_layers);
    println!("{{\"prove_seconds\":{:.3},\"proof_bytes\":{},\"layers\":{},\"config\":\"{}\"}}", prove_s, bytes.len(), n_layers, config_name);
}

fn verify<C: Config>(p: &PublicFile, proof_path: &Path, config_name: &str) -> bool {
    let t = Instant::now();
    let compiled: CompileResult<C> = compile(&shape_circuit(p.m, p.k), CompileOptions::default()).expect("compile");
    let compile_s = t.elapsed().as_secs_f64();

    // The verifier knows only the public inputs. Private inputs are set to zero: with the
    // Orion commitment the proof carries what the verifier needs about them; with the raw
    // "commitment" it does not, and verification of a real proof is expected to fail.
    let w = WitnessFile { m: p.m, k: p.k, q4: vec![0; p.m * p.k], sc: vec![0; p.m * (p.k / QK_K) * 8], mn: vec![0; p.m * (p.k / QK_K) * 8], q8: p.q8.clone(), s1: p.s1.clone(), s2: p.s2.clone() };
    let n_pack = SIMDField::<C>::PACK_SIZE;
    let assigns = vec![assignment::<C>(&w, true); n_pack];
    let witness = compiled.witness_solver.solve_witnesses(&assigns).expect("witness shape");
    let mut circuit = compiled.layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public) = witness.to_simd();
    circuit.layers[0].input_vals = simd_input;
    circuit.public_input = simd_public;

    let mut bytes = Vec::new();
    fs::File::open(proof_path).expect("proof").read_to_end(&mut bytes).expect("read");
    let mut cursor = Cursor::new(&bytes[..]);
    let proof = gkr_engine::Proof::deserialize_from(&mut cursor).expect("proof bytes");
    let claimed_v = <ChallengeField<C>>::deserialize_from(&mut cursor).expect("claimed value");

    let t = Instant::now();
    let mpi = MPIConfig::prover_new(None, None);
    let ok = executor::verify::<C>(&mut circuit, mpi, &proof, &claimed_v);
    let verify_s = t.elapsed().as_secs_f64();
    eprintln!("compile {:.2}s, verify {:.3}s, config {}: {}", compile_s, verify_s, config_name, if ok { "ACCEPT" } else { "REJECT" });
    println!("{{\"verify_seconds\":{:.3},\"compile_seconds\":{:.3},\"accept\":{},\"config\":\"{}\"}}", verify_s, compile_s, ok, config_name);
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: receipts-zk prove witness.json out_dir [raw|orion]\n       receipts-zk verify public.json proof.bin [raw|orion]");
        std::process::exit(2);
    }
    let config = args.get(4).map(String::as_str).unwrap_or("orion").to_string();
    match args[1].as_str() {
        "prove" => {
            let w: WitnessFile = serde_json::from_reader(BufReader::new(fs::File::open(&args[2]).expect("witness file"))).expect("witness json");
            match config.as_str() {
                "raw" => prove::<M31Config>(&w, Path::new(&args[3]), "raw"),
                _ => prove::<M31OrionConfig>(&w, Path::new(&args[3]), "orion"),
            }
        }
        "verify" => {
            let p: PublicFile = serde_json::from_reader(BufReader::new(fs::File::open(&args[2]).expect("public file"))).expect("public json");
            let cfg = if args.len() > 4 { config.clone() } else { p.config.clone() };
            let ok = match cfg.as_str() {
                "raw" => verify::<M31Config>(&p, Path::new(&args[3]), "raw"),
                _ => verify::<M31OrionConfig>(&p, Path::new(&args[3]), "orion"),
            };
            std::process::exit(if ok { 0 } else { 1 });
        }
        other => {
            eprintln!("unknown command {}", other);
            std::process::exit(2);
        }
    }
}
