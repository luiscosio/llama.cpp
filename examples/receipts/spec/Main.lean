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
def checkTrace (receiptPath tracePath expectedTopology : String) : IO Bool := do
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
  let signedTopology ← getOr (trace.getObjValAs? String "topology_sha256")
  let sideTopology ← getOr (tr.getObjValAs? String "topology_sha256")
  let topology := Protocol.topologySha256 leavesJ
  let topologyOk := topology == signedTopology ∧ topology == sideTopology ∧ topology == expectedTopology
  IO.println s!"topology: {if topologyOk then "ok" else "MISMATCH"} ({topology.take 16} vs expected {expectedTopology.take 16})"
  -- challenge
  let commitments ← getOr (rec.getObjVal? "commitments")
  let tokensSha ← getOr (commitments.getObjValAs? String "tokens_sha256")
  let contentSha ← getOr (commitments.getObjValAs? String "content_sha256")
  let modelSha ← getOr ((← getOr (rec.getObjVal? "model")).getObjValAs? String "file_sha256")
  let ch ← getOr (tr.getObjVal? "challenge")
  let k ← getOr (ch.getObjValAs? Nat "k")
  let seed := (ch.getObjValAs? String "seed").toOption.getD ""
  let want := Protocol.challengeIndices signedRoot tokensSha contentSha modelSha seed leaves.size k
  let openingsJ ← getOr (tr.getObjValAs? (Array Json) "openings")
  let mut got : List Nat := []
  for o in openingsJ do
    got := (← getOr (o.getObjValAs? Nat "index")) :: got
  let gotSorted := got.mergeSort (fun a b => decide (a ≤ b))
  let openingPolicyOk := gotSorted.length ≥ min 32 leaves.size
  IO.println s!"challenge: {if gotSorted == want ∧ openingPolicyOk then "ok" else "MISMATCH"} ({gotSorted.length} openings)"
  -- edges
  let e := Protocol.checkEdges leaves
  IO.println s!"edges: {if e.bad == 0 then "ok" else "MISMATCH"} ({e.bad} of {e.total} inconsistent)"
  -- inputs
  let req ← getOr (rec.getObjVal? "request")
  let prompt ← getOr (req.getObjValAs? (List Int) "prompt_tokens")
  let responseJ ← getOr (rec.getObjVal? "response")
  let resp ← getOr (responseJ.getObjValAs? (List Int) "tokens")
  let promptText ← getOr (req.getObjValAs? String "prompt_text")
  let responseText ← getOr (responseJ.getObjValAs? String "text")
  let gotTokensSha := Sha256.hashHex (Protocol.canonicalBytes (Json.mkObj [("prompt", toJson prompt), ("response", toJson resp)]))
  let gotContentSha := Sha256.hashHex (Protocol.canonicalBytes (Json.mkObj [("prompt_text", toJson promptText), ("response_text", toJson responseText)]))
  let commitmentsOk := gotTokensSha == tokensSha ∧ gotContentSha == contentSha
  IO.println s!"content commitments: {if commitmentsOk then "ok" else "MISMATCH"}"
  let inp := Protocol.checkInputs leaves prompt resp
  IO.println s!"inputs: tokens {inp.tokensMatch} match {inp.tokensMismatch} mismatch; positions {inp.posMatch} match {inp.posMismatch} mismatch"
  return root == signedRoot ∧ topologyOk ∧ commitmentsOk ∧ gotSorted == want ∧ openingPolicyOk ∧ e.bad == 0 ∧
    inp.tokensMismatch == 0 ∧ inp.posMismatch == 0 ∧ inp.tokensMatch > 0

/-- Differential test of the arithmetic spec against vectors produced by the Python verifier.
    With `emit`, also writes what the spec computed (Q8_K quants, `d` bits, `s1`, `s2`, `Q6_K`
    dots) so that other implementations, such as the proof circuit, can be checked against
    the spec rather than against Python. -/
def checkVectors (path : String) (emit : Option String := none) : IO Bool := do
  let v ← readJson path
  let mut ok := true
  let mut emitQ4 : Array Json := #[]
  let mut emitQ6 : Array Json := #[]
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
    emitQ4 := emitQ4.push (Json.mkObj [("blocks_hex", toJson blocksHex), ("q8", toJson gotQ8), ("d_bits", toJson gotD),
                                        ("s1", toJson gotS1), ("s2", toJson gotS2), ("row_value", toJson gotRow)])
  let q6 ← getOr (v.getObjValAs? (Array Json) "q6k")
  for case in q6 do
    let bytes := Sha256.ofHex (← getOr (case.getObjValAs? String "blocks_hex"))
    let nb := bytes.size / 210
    let acts : Array Float32 := (← getOr (case.getObjValAs? (Array Nat) "act_bits")).map (fun b => Float32.ofBits (UInt32.ofNat b))
    let wantDot ← getOr (case.getObjValAs? (Array Int) "dot")
    let gotDot := (Array.range nb).map (fun i => Quant.dotQ6 (Quant.parseQ6K bytes (i * 210)) (Quant.quantizeQ8K (acts.extract (i * 256) ((i + 1) * 256))))
    IO.println s!"q6k row: dot {gotDot == wantDot}"
    ok := ok ∧ gotDot == wantDot
    emitQ6 := emitQ6.push (Json.mkObj [("dot", toJson gotDot)])
  -- canonical JSON, including string escapes, against Python's json.dumps
  let canon ← getOr (v.getObjValAs? (Array Json) "canonical")
  let mut canonOk := true
  for case in canon do
    let value ← getOr (case.getObjVal? "value")
    let want ← getOr (case.getObjValAs? String "sha256")
    canonOk := canonOk ∧ Sha256.hashHex (Protocol.canonicalBytes value) == want
  IO.println s!"canonical json: {canon.size} documents, {if canonOk then "all match" else "MISMATCH"}"
  ok := ok ∧ canonOk
  if let some out := emit then
    IO.FS.writeFile out (Json.mkObj [("q4k", .arr emitQ4), ("q6k", .arr emitQ6)]).compress
    IO.println s!"wrote {out}"
  return ok

def main (args : List String) : IO UInt32 := do
  match args with
  | ["sha256"] =>
    if ← selfTestSha256 then IO.println "sha256: ok"; return 0 else return 1
  | ["trace", receipt, trace, expectedTopology] =>
    if ← checkTrace receipt trace expectedTopology then IO.println "trace: ok"; return 0 else IO.println "trace: FAILED"; return 1
  | ["vectors", path] =>
    if ← checkVectors path then IO.println "vectors: ok"; return 0 else IO.println "vectors: FAILED"; return 1
  | ["vectors", path, "--emit", out] =>
    if ← checkVectors path (some out) then IO.println "vectors: ok"; return 0 else IO.println "vectors: FAILED"; return 1
  | _ =>
    IO.println "usage: spec-check sha256 | trace receipt.json trace.json expected-topology-sha256 | vectors vectors.json [--emit lean.json]"
    return 2
