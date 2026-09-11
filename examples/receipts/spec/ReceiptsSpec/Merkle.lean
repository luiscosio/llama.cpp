/-!
# Merkle commitment and Fiat-Shamir challenge

Byte-exact with `receipts.cpp` and `verify_trace.py`: leaves hash as `H(0x00 ‖ data)`, nodes as
`H(0x01 ‖ left ‖ right)`, an odd level duplicates its last node, and the empty tree hashes to
`H("")`. The hash is a parameter so the definitions are about any collision-resistant `H`;
`Sha256.hash` is plugged in by the executable checker.
-/

namespace ReceiptsSpec.Merkle

variable (H : ByteArray → ByteArray)

def leafHash (data : ByteArray) : ByteArray := H (ByteArray.mk #[0] ++ data)

def nodeHash (l r : ByteArray) : ByteArray := H (ByteArray.mk #[1] ++ l ++ r)

/-- One level up: pair neighbours, duplicating a trailing odd node. -/
def pairUp : List ByteArray → List ByteArray
  | a :: b :: rest => nodeHash H a b :: pairUp rest
  | [a] => [nodeHash H a a]
  | [] => []

theorem pairUp_length : ∀ (l : List ByteArray), (pairUp H l).length = (l.length + 1) / 2
  | a :: b :: rest => by simp [pairUp, pairUp_length rest]; omega
  | [a] => by simp [pairUp]
  | [] => by simp [pairUp]

/-- Reduce levels until one node is left. `fuel` only bounds the recursion; the tree has at
    most `length` levels, so `rootOfLevel` passes the length. -/
def rootFuel : Nat → List ByteArray → ByteArray
  | _, [] => H ByteArray.empty
  | _, [x] => x
  | 0, _ => H ByteArray.empty
  | fuel + 1, l => rootFuel fuel (pairUp H l)

def rootOfLevel (l : List ByteArray) : ByteArray := rootFuel H l.length l

/-- Root of the tree over `leaves` (already-serialized leaf data). -/
def root (leaves : List ByteArray) : ByteArray :=
  match leaves with
  | [] => H ByteArray.empty
  | _ => rootOfLevel H (leaves.map (leafHash H))

/-- An audit path: for each level, whether the sibling sits to the left, and its hash. -/
def pathRoot (leaf : ByteArray) (path : List (Bool × ByteArray)) : ByteArray :=
  path.foldl (fun acc (p : Bool × ByteArray) => if p.1 then nodeHash H p.2 acc else nodeHash H acc p.2) (leafHash H leaf)

def verifyPath (leaf : ByteArray) (path : List (Bool × ByteArray)) (r : ByteArray) : Bool :=
  decide (pathRoot H leaf path = r)

theorem root_singleton (x : ByteArray) : root H [x] = leafHash H x := by
  simp [root, rootOfLevel, rootFuel]

theorem verifyPath_singleton (x : ByteArray) : verifyPath H x [] (root H [x]) = true := by
  simp [verifyPath, pathRoot, root_singleton]

theorem root_pair (x y : ByteArray) : root H [x, y] = nodeHash H (leafHash H x) (leafHash H y) := by
  simp [root, rootOfLevel, rootFuel, pairUp]

theorem verifyPath_pair_left (x y : ByteArray) :
    verifyPath H x [(false, leafHash H y)] (root H [x, y]) = true := by
  simp [verifyPath, pathRoot, root_pair]

theorem verifyPath_pair_right (x y : ByteArray) :
    verifyPath H y [(true, leafHash H x)] (root H [x, y]) = true := by
  simp [verifyPath, pathRoot, root_pair]

/-! ## Fiat-Shamir sample of leaf indices -/

def challengeDomain : ByteArray := "llama-receipts/trace-challenge/v1/".toUTF8

def be64 (v : Nat) : ByteArray :=
  ByteArray.mk ((Array.range 8).map (fun i => UInt8.ofNat ((v / 256 ^ (7 - i)) % 256)))

def first8BE (b : ByteArray) : Nat :=
  (List.range 8).foldl (fun acc i => acc * 256 + (b.get! i).toNat) 0

/-- Counter-mode sampling: hash `pre ‖ counter`, reduce modulo `n`, keep new values, stop at `k`. -/
def deriveGo (pre : ByteArray) (n k : Nat) : Nat → Nat → List Nat → List Nat
  | 0, _, acc => acc
  | fuel + 1, counter, acc =>
    if acc.length ≥ k ∨ n = 0 then acc
    else
      let idx := first8BE (H (pre ++ be64 counter)) % n
      deriveGo pre n k fuel (counter + 1) (if acc.contains idx then acc else idx :: acc)

def natLe (a b : Nat) : Bool := decide (a ≤ b)

/-- `k` distinct indices below `n` from `H(domain ‖ root ‖ binding ‖ seed ‖ counter)`, sorted. -/
def deriveIndices (root binding seed : ByteArray) (n k : Nat) : List Nat :=
  (deriveGo H (challengeDomain ++ root ++ binding ++ seed) n k (4 * k + 64) 0 []).mergeSort natLe

theorem deriveGo_lt (pre : ByteArray) (n k : Nat) (hn : 0 < n) :
    ∀ fuel counter acc, (∀ i ∈ acc, i < n) → ∀ i ∈ deriveGo H pre n k fuel counter acc, i < n := by
  intro fuel
  induction fuel with
  | zero => intro counter acc hacc i hi; simpa [deriveGo] using hacc i hi
  | succ fuel ih =>
    intro counter acc hacc i hi
    simp only [deriveGo] at hi
    split at hi
    · exact hacc i hi
    · apply ih (counter + 1) _ _ i hi
      intro j hj
      split at hj
      · exact hacc j hj
      · simp only [List.mem_cons] at hj
        rcases hj with rfl | hj
        · exact Nat.mod_lt _ hn
        · exact hacc j hj

theorem deriveGo_nodup (pre : ByteArray) (n k : Nat) :
    ∀ fuel counter acc, acc.Nodup → (deriveGo H pre n k fuel counter acc).Nodup := by
  intro fuel
  induction fuel with
  | zero => intro counter acc h; simpa [deriveGo] using h
  | succ fuel ih =>
    intro counter acc h
    simp only [deriveGo]
    split
    · exact h
    · apply ih
      split
      · exact h
      · rename_i hc
        exact List.nodup_cons.mpr ⟨fun hm => hc (List.contains_iff_mem.mpr hm), h⟩

/-- Every sampled index is a valid leaf index. -/
theorem deriveIndices_lt (root binding seed : ByteArray) (n k : Nat) (hn : 0 < n) :
    ∀ i ∈ deriveIndices H root binding seed n k, i < n := by
  intro i hi
  unfold deriveIndices at hi
  rw [List.mem_mergeSort] at hi
  exact deriveGo_lt H _ n k hn _ _ _ (by simp) i hi

/-- No leaf is opened twice. -/
theorem deriveIndices_nodup (root binding seed : ByteArray) (n k : Nat) :
    (deriveIndices H root binding seed n k).Nodup := by
  unfold deriveIndices
  exact (List.mergeSort_perm _ natLe).nodup_iff.mpr (deriveGo_nodup H _ n k _ _ _ List.nodup_nil)

end ReceiptsSpec.Merkle
