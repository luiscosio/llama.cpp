import ReceiptsSpec

open Lean ReceiptsSpec

def selfTestSha256 : IO Bool := do
  let cases : List (String × String) := [
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")]
  let mut ok := true
  for (msg, want) in cases do
    let got := Sha256.hashHex (Sha256.ofString msg)
    if got != want then
      IO.println s!"sha256 mismatch for {repr msg}: {got}"
      ok := false
  return ok

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: {e}")

def getOr {α} [Inhabited α] (e : Except String α) : IO α :=
  match e with
  | .ok v => pure v
  | .error m => throw (IO.userError m)

/-- Recompute root, challenge, edges and token/position inputs of a real trace. -/
def checkTrace (receiptPath tracePath : String) : IO Bool := do
  let rec ← readJson receiptPath
  let tr ← readJson tracePath
  let leavesJ ← getOr (tr.getObjValAs? (Array Json) "leaves")
  let mut leaves : Array Protocol.Leaf := Array.mkEmpty leavesJ.size
  for j in leavesJ do
    leaves := leaves.push (← getOr (Protocol.parseLeaf j))
  let trace ← getOr (rec.getObjVal? "trace")
  let signedRoot ← getOr (trace.getObjValAs? String "root")
  let root := Protocol.traceRoot leaves
  IO.println s!"leaves: {leaves.size}"
  IO.println s!"root: {if root == signedRoot then "ok" else "MISMATCH"} ({root.take 16} vs receipt {signedRoot.take 16})"
  -- challenge
  let tokensSha ← getOr ((← getOr (rec.getObjVal? "commitments")).getObjValAs? String "tokens_sha256")
  let modelSha ← getOr ((← getOr (rec.getObjVal? "model")).getObjValAs? String "file_sha256")
  let ch ← getOr (tr.getObjVal? "challenge")
  let k ← getOr (ch.getObjValAs? Nat "k")
  let seed := (ch.getObjValAs? String "seed").toOption.getD ""
  let want := Protocol.challengeIndices signedRoot tokensSha modelSha seed leaves.size k
  let openingsJ ← getOr (tr.getObjValAs? (Array Json) "openings")
  let mut got : List Nat := []
  for o in openingsJ do
    got := (← getOr (o.getObjValAs? Nat "index")) :: got
  let gotSorted := got.mergeSort (fun a b => decide (a ≤ b))
  IO.println s!"challenge: {if gotSorted == want then "ok" else "MISMATCH"} ({gotSorted.length} openings)"
  -- edges
  let e := Protocol.checkEdges leaves
  IO.println s!"edges: {if e.bad == 0 then "ok" else "MISMATCH"} ({e.bad} of {e.total} inconsistent)"
  -- inputs
  let req ← getOr (rec.getObjVal? "request")
  let prompt ← getOr (req.getObjValAs? (List Int) "prompt_tokens")
  let resp ← getOr ((← getOr (rec.getObjVal? "response")).getObjValAs? (List Int) "tokens")
  let inp := Protocol.checkInputs leaves prompt resp
  IO.println s!"inputs: tokens {inp.tokensMatch} match {inp.tokensMismatch} mismatch; positions {inp.posMatch} match {inp.posMismatch} mismatch"
  return root == signedRoot ∧ gotSorted == want ∧ e.bad == 0 ∧ inp.tokensMismatch == 0 ∧ inp.posMismatch == 0 ∧ inp.tokensMatch > 0

/-- Differential test of the arithmetic spec against vectors produced by the Python verifier. -/
def checkVectors (path : String) : IO Bool := do
  let v ← readJson path
  let mut ok := true
  let q4 ← getOr (v.getObjValAs? (Array Json) "q4k")
  for case in q4 do
    let blocksHex ← getOr (case.getObjValAs? String "blocks_hex")
    let bytes := Sha256.ofHex blocksHex
    let nb := bytes.size / 144
    let actBits ← getOr (case.getObjValAs? (Array Nat) "act_bits")
    let acts : Array Float32 := actBits.map (fun b => Float32.ofBits (UInt32.ofNat b))
    let wantQ8 ← getOr (case.getObjValAs? (Array Int) "q8")
    let wantD ← getOr (case.getObjValAs? (Array Nat) "d_bits")
    let wantS1 ← getOr (case.getObjValAs? (Array Int) "s1")
    let wantS2 ← getOr (case.getObjValAs? (Array Int) "s2")
    let wantRow ← getOr (case.getObjValAs? Float "row_value")
    let mut blocks : Array Quant.Q4KBlock := #[]
    let mut aBlocks : Array Quant.Q8KBlock := #[]
    for i in [0:nb] do
      blocks := blocks.push (Quant.parseQ4K bytes (i * 144))
      aBlocks := aBlocks.push (Quant.quantizeQ8K (acts.extract (i * 256) ((i + 1) * 256)))
    let gotQ8 := aBlocks.foldl (fun acc b => acc ++ b.qs) #[]
    let gotD := aBlocks.map (fun b => b.d.toBits.toNat)
    let gotS1 := (Array.range nb).map (fun i => Quant.s1 blocks[i]! aBlocks[i]!)
    let gotS2 := (Array.range nb).map (fun i => Quant.s2 blocks[i]! aBlocks[i]!)
    let gotRow := Quant.rowQ4K blocks aBlocks
    let rel := (gotRow - wantRow).abs / (max wantRow.abs 1e-30)
    let good := gotQ8 == wantQ8 ∧ gotD == wantD ∧ gotS1 == wantS1 ∧ gotS2 == wantS2 ∧ rel < 1e-9
    IO.println s!"q4k row: q8 {gotQ8 == wantQ8}, d {gotD == wantD}, s1 {gotS1 == wantS1}, s2 {gotS2 == wantS2}, row rel err {rel}"
    ok := ok ∧ good
  let q6 ← getOr (v.getObjValAs? (Array Json) "q6k")
  for case in q6 do
    let bytes := Sha256.ofHex (← getOr (case.getObjValAs? String "blocks_hex"))
    let nb := bytes.size / 210
    let acts : Array Float32 := (← getOr (case.getObjValAs? (Array Nat) "act_bits")).map (fun b => Float32.ofBits (UInt32.ofNat b))
    let wantDot ← getOr (case.getObjValAs? (Array Int) "dot")
    let gotDot := (Array.range nb).map (fun i => Quant.dotQ6 (Quant.parseQ6K bytes (i * 210)) (Quant.quantizeQ8K (acts.extract (i * 256) ((i + 1) * 256))))
    IO.println s!"q6k row: dot {gotDot == wantDot}"
    ok := ok ∧ gotDot == wantDot
  return ok

def main (args : List String) : IO UInt32 := do
  match args with
  | ["sha256"] =>
    if ← selfTestSha256 then IO.println "sha256: ok"; return 0 else return 1
  | ["trace", receipt, trace] =>
    if ← checkTrace receipt trace then IO.println "trace: ok"; return 0 else IO.println "trace: FAILED"; return 1
  | ["vectors", path] =>
    if ← checkVectors path then IO.println "vectors: ok"; return 0 else IO.println "vectors: FAILED"; return 1
  | _ =>
    IO.println "usage: spec-check sha256 | trace receipt.json trace.json | vectors vectors.json"
    return 2
