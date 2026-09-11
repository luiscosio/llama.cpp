import Lean
import ReceiptsSpec.Sha256
import ReceiptsSpec.Merkle

/-!
# The trace protocol: leaves, edges, inputs, root, challenge

Executable definitions over the sidecar's JSON, byte-exact with `verify_trace.py`. The
checker (`spec-check trace`) runs them on real receipts and traces produced by `llama-receipts`.
-/

open Lean

namespace ReceiptsSpec.Protocol

/-- One input of a node. Weights carry the GGUF tensor hash; data inputs carry the hash of the
    base tensor they view and the leaf that produced it (`-1` graph input, `-2` zeroed KV cache). -/
structure Src where
  kind     : String
  ttype    : String
  ne       : List Nat
  baseSha  : String
  nbytes   : Nat
  producer : Int
  deriving Inhabited, Repr

structure Leaf where
  g         : Nat
  i         : Nat
  op        : String
  name      : String
  outSha    : String
  outNbytes : Nat
  outNe     : List Nat
  srcs      : List Src
  canonical : ByteArray   -- sorted-key compact JSON, what the Merkle leaf hashes
  deriving Inhabited

/-- Canonical JSON bytes, the same bytes `json.dumps(sort_keys=True, separators=(",", ":"))`
    and nlohmann's `dump()` produce for these objects. -/
def canonicalBytes (j : Json) : ByteArray := j.compress.toUTF8

def parseSrc (s : Json) : Except String Src := do
  let kind ← s.getObjValAs? String "kind"
  let ttype ← s.getObjValAs? String "type"
  let ne ← s.getObjValAs? (List Nat) "ne"
  let base ← s.getObjVal? "base"
  let nbytes ← base.getObjValAs? Nat "nbytes"
  if kind == "weight" then
    pure { kind, ttype, ne, baseSha := ← s.getObjValAs? String "sha256", nbytes, producer := 0 }
  else
    pure { kind, ttype, ne, baseSha := ← base.getObjValAs? String "sha256", nbytes, producer := ← s.getObjValAs? Int "producer" }

def parseLeaf (j : Json) : Except String Leaf := do
  let out ← j.getObjVal? "out"
  let base ← out.getObjVal? "base"
  let srcs ← (← j.getObjValAs? (Array Json) "srcs").toList.mapM parseSrc
  pure { g := ← j.getObjValAs? Nat "g", i := ← j.getObjValAs? Nat "i", op := ← j.getObjValAs? String "op",
         name := ← j.getObjValAs? String "name", outSha := ← base.getObjValAs? String "sha256",
         outNbytes := ← base.getObjValAs? Nat "nbytes", outNe := ← out.getObjValAs? (List Nat) "ne",
         srcs, canonical := canonicalBytes j }

/-- Root of the committed trace. -/
def traceRoot (leaves : Array Leaf) : String :=
  Sha256.toHex (Merkle.root Sha256.hash (leaves.toList.map (·.canonical)))

structure EdgeReport where
  total : Nat
  bad   : Nat
  deriving Repr

/-- Every data input names its producer's output hash, or the zeroed initial KV cache. -/
def checkEdges (leaves : Array Leaf) : EdgeReport := Id.run do
  let mut total := 0
  let mut bad := 0
  let mut zeros : List (Nat × String) := []
  for idx in [0:leaves.size] do
    let leaf := leaves[idx]!
    for s in leaf.srcs do
      if s.kind != "data" then continue
      if s.producer ≥ 0 then
        total := total + 1
        let p := s.producer.toNat
        if p ≥ idx ∨ leaves[p]!.outSha != s.baseSha ∨ leaves[p]!.outNbytes != s.nbytes then
          bad := bad + 1
      else if s.producer == -2 then
        total := total + 1
        let z ← match zeros.lookup s.nbytes with
          | some h => pure h
          | none =>
            let h := Sha256.hashHex (ByteArray.mk (Array.replicate s.nbytes 0))
            zeros := (s.nbytes, h) :: zeros
            pure h
        if z != s.baseSha then bad := bad + 1
  return { total, bad }

def int32LE (xs : List Int) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  for x in xs do
    let u : Nat := (x % 4294967296).toNat
    out := out.push (UInt8.ofNat (u % 256)) |>.push (UInt8.ofNat (u / 256 % 256))
             |>.push (UInt8.ofNat (u / 65536 % 256)) |>.push (UInt8.ofNat (u / 16777216 % 256))
  return out

structure InputReport where
  tokensMatch    : Nat
  tokensMismatch : Nat
  posMatch       : Nat
  posMismatch    : Nat
  deriving Repr

/-- Graph `g` consumed the prompt (g = 0) or response token `g - 1`, at positions starting where
    the previous graphs ended. The token input of the embedding lookup and the position input of
    every rotary node must hash to exactly that. -/
def checkInputs (leaves : Array Leaf) (prompt response : List Int) : InputReport := Id.run do
  let mut r : InputReport := { tokensMatch := 0, tokensMismatch := 0, posMatch := 0, posMismatch := 0 }
  for leaf in leaves do
    let toks : Option (List Int) :=
      if leaf.g == 0 then some prompt
      else if leaf.g ≤ response.length then some [response[leaf.g - 1]!] else none
    match toks with
    | none => continue
    | some toks =>
      let start : Int := if leaf.g == 0 then 0 else (prompt.length : Int) + (leaf.g : Int) - 1
      if leaf.op == "GET_ROWS" ∧ leaf.srcs.length == 2 ∧ leaf.srcs[0]!.kind == "weight" then
        let want := Sha256.hashHex (int32LE toks)
        if want == leaf.srcs[1]!.baseSha then r := { r with tokensMatch := r.tokensMatch + 1 }
        else r := { r with tokensMismatch := r.tokensMismatch + 1 }
      if leaf.op == "ROPE" ∧ leaf.srcs.length ≥ 2 then
        let want := Sha256.hashHex (int32LE ((List.range toks.length).map (fun t => start + Int.ofNat t)))
        if want == leaf.srcs[1]!.baseSha then r := { r with posMatch := r.posMatch + 1 }
        else r := { r with posMismatch := r.posMismatch + 1 }
  return r

/-- The Fiat-Shamir sample the receipt implies. -/
def challengeIndices (rootHex tokensShaHex modelShaHex seedHex : String) (n k : Nat) : List Nat :=
  Merkle.deriveIndices Sha256.hash (Sha256.ofHex rootHex) (Sha256.ofHex tokensShaHex ++ Sha256.ofHex modelShaHex)
    (Sha256.ofHex seedHex) n k

end ReceiptsSpec.Protocol
