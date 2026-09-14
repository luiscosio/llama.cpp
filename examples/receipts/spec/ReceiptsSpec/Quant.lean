/-!
# ggml's block-quantized matmul, as a specification

This file pins down the arithmetic a proof of a llama.cpp matmul has to attest.
It follows `ggml-quants.c` (`quantize_row_q8_K_ref`) and `ggml-cpu/quants.c`
(`ggml_vec_dot_q4_K_q8_K_generic`, `ggml_vec_dot_q6_K_q8_K_generic`).

Two layers:

* **Integer core** (`s1`, `s2`, `dotQ6`): exact `Int` arithmetic over the block's
  nibbles, scales and mins and the Q8_K activation quants. This is what the GKR circuit
  in `../zk` constrains and what the theorems below are about.
* **Float rim** (`quantizeQ8K`, `combineQ4K`): the activation quantization in `Float32`,
  exactly the operations ggml performs, and the per-block scaling of the integer sums.
  Floats are opaque to Lean's logic, so these are executable definitions checked by
  differential testing against the C++ engine and the Python verifier, not by proof.

Block layouts are byte-exact copies of `block_q4_K`, `block_q6_K` and `block_q8_K`.
-/

namespace ReceiptsSpec.Quant

/-- Elements per super-block. -/
def QK_K : Nat := 256

/-! ## Bit fields, with their bounds -/

/-- Low or high nibble of a byte. -/
def nibble (b : UInt8) (high : Bool) : Nat := if high then b.toNat / 16 else b.toNat % 16

theorem nibble_lt_16 (b : UInt8) (high : Bool) : nibble b high < 16 := by
  unfold nibble
  have := b.toNat_lt
  split <;> omega

/-- `get_scale_min_k4`, upper half: four low bits of one byte and the top two bits of another. -/
def packedLo (a b : UInt8) : Nat := a.toNat % 16 + 16 * (b.toNat / 64)

/-- `get_scale_min_k4`, upper half: four high bits of one byte and the top two bits of another. -/
def packedHi (a b : UInt8) : Nat := a.toNat / 16 + 16 * (b.toNat / 64)

theorem packedLo_lt_64 (a b : UInt8) : packedLo a b < 64 := by
  unfold packedLo
  have := b.toNat_lt
  omega

theorem packedHi_lt_64 (a b : UInt8) : packedHi a b < 64 := by
  unfold packedHi
  have ha := a.toNat_lt
  have hb := b.toNat_lt
  omega

/-- A Q6_K value before the offset: a nibble plus two bits taken from `qh` at `shift`. -/
def sixBits (ql qh : UInt8) (high : Bool) (shift : Nat) : Nat :=
  nibble ql high + 16 * ((qh.toNat / 4 ^ shift) % 4)

theorem sixBits_lt_64 (ql qh : UInt8) (high : Bool) (shift : Nat) : sixBits ql qh high shift < 64 := by
  unfold sixBits
  have h1 := nibble_lt_16 ql high
  have h2 : (qh.toNat / 4 ^ shift) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- Decode a little-endian IEEE half from two bytes into a `Float32`. -/
def halfToFloat32 (lo hi : UInt8) : Float32 :=
  let bits : Nat := hi.toNat * 256 + lo.toNat
  let sign : Float32 := if bits / 32768 = 1 then -1.0 else 1.0
  let exp : Nat := (bits / 1024) % 32
  let frac : Nat := bits % 1024
  if exp = 0 then
    sign * (Float32.ofNat frac) / Float32.ofNat (2 ^ 24)
  else if exp = 31 then
    if frac = 0 then sign * (1.0 / 0.0) else (0.0 / 0.0)
  else
    let mant : Float32 := (Float32.ofNat (1024 + frac)) / 1024.0
    let scale : Float32 := if exp ≥ 15 then Float32.ofNat (2 ^ (exp - 15)) else 1.0 / Float32.ofNat (2 ^ (15 - exp))
    sign * mant * scale

/-! ## Blocks -/

/-- One 144-byte Q4_K block: two halves `d`, `dmin`, twelve packed scale bytes, 128 nibble bytes. -/
structure Q4KBlock where
  d      : Float32
  dmin   : Float32
  scales : Array UInt8   -- 12 bytes
  qs     : Array UInt8   -- 128 bytes
  deriving Inhabited

/-- One 210-byte Q6_K block: 128 low-nibble bytes, 64 high-bit bytes, 16 signed scales, one half `d`. -/
structure Q6KBlock where
  ql     : Array UInt8   -- 128
  qh     : Array UInt8   -- 64
  scales : Array Int     -- 16, each in -128..127
  d      : Float32
  deriving Inhabited

def parseQ4K (b : ByteArray) (off : Nat) : Q4KBlock :=
  { d := halfToFloat32 (b.get! off) (b.get! (off+1))
    dmin := halfToFloat32 (b.get! (off+2)) (b.get! (off+3))
    scales := (Array.range 12).map (fun i => b.get! (off + 4 + i))
    qs := (Array.range 128).map (fun i => b.get! (off + 16 + i)) }

def parseQ6K (b : ByteArray) (off : Nat) : Q6KBlock :=
  { ql := (Array.range 128).map (fun i => b.get! (off + i))
    qh := (Array.range 64).map (fun i => b.get! (off + 128 + i))
    scales := (Array.range 16).map (fun i =>
      let v := (b.get! (off + 192 + i)).toNat
      if v ≥ 128 then (v : Int) - 256 else (v : Int))
    d := halfToFloat32 (b.get! (off + 208)) (b.get! (off + 209)) }

/-- `get_scale_min_k4`: the 6-bit scale and min of sub-block `j` (0..7). -/
def scaleMin (sc : Array UInt8) (j : Nat) : Nat × Nat :=
  if j < 4 then (sc[j]!.toNat % 64, sc[j + 4]!.toNat % 64)
  else (packedLo sc[j + 4]! sc[j - 4]!, packedHi sc[j + 4]! sc[j]!)

/-- Nibble `k` (0..255) of a Q4_K block, in the order the kernel consumes them:
    for each 64-element chunk, the low nibbles of 32 bytes, then their high nibbles. -/
def q4 (blk : Q4KBlock) (k : Nat) : Nat :=
  nibble blk.qs[(k / 64) * 32 + k % 32]! (decide (k % 64 ≥ 32))

/-- Value `k` (0..255) of a Q6_K block, in `-32..31`. -/
def q6 (blk : Q6KBlock) (k : Nat) : Int :=
  -- half = k / 128, group = (k % 128) / 32 in 0..3, lane = k % 32
  (sixBits blk.ql[(k / 128) * 64 + (if (k % 128) / 32 % 2 = 1 then 32 else 0) + k % 32]!
           blk.qh[(k / 128) * 32 + k % 32]! (decide ((k % 128) / 32 ≥ 2)) ((k % 128) / 32) : Int) - 32

/-- A Q8_K activation block: scale, 256 quants in `-127..127`, sixteen group sums. -/
structure Q8KBlock where
  d     : Float32
  qs    : Array Int   -- 256
  bsums : Array Int   -- 16
  deriving Inhabited

/-- Round half to even, as ggml's `nearest_int` does for |x| < 2^22. -/
def rintF32 (x : Float32) : Int :=
  let f := x.floor
  let diff := x - f
  let fi : Int := f.toInt64.toInt
  if diff < 0.5 then fi
  else if diff > 0.5 then fi + 1
  else if fi % 2 = 0 then fi else fi + 1

/-- `quantize_row_q8_K_ref` on one block of 256 `Float32` activations. -/
def quantizeQ8K (x : Array Float32) : Q8KBlock := Id.run do
  -- the element of largest magnitude, first occurrence on ties (strict '>' in the C loop)
  let mut amax : Float32 := 0
  let mut mx : Float32 := 0
  for v in x do
    let av := if v < 0 then -v else v
    if av > amax then
      amax := av
      mx := v
  if amax == 0 then
    return { d := 0, qs := Array.replicate QK_K 0, bsums := Array.replicate 16 0 }
  let iscale : Float32 := (-127.0) / mx
  let qs := x.map (fun v => min (rintF32 (iscale * v)) 127)
  let bsums := (Array.range 16).map (fun j => (List.range 16).foldl (fun acc l => acc + qs[j * 16 + l]!) 0)
  return { d := 1.0 / iscale, qs := qs, bsums := bsums }

/-! ## Integer core -/

/-- Sum of `q4 * q8` over sub-block `j` (32 elements). -/
def subDot (blk : Q4KBlock) (a : Q8KBlock) (j : Nat) : Int :=
  (List.range 32).foldl (fun acc l => acc + ((q4 blk (32 * j + l) : Nat) : Int) * a.qs[32 * j + l]!) 0

/-- `s1 = Σ_j scale_j · Σ_l q4·q8` for one weight block against one activation block. -/
def s1 (blk : Q4KBlock) (a : Q8KBlock) : Int :=
  (List.range 8).foldl (fun acc j => acc + ((scaleMin blk.scales j).1 : Int) * subDot blk a j) 0

/-- `s2 = Σ_{j<16} bsums_j · min_{j/2}`. -/
def s2 (blk : Q4KBlock) (a : Q8KBlock) : Int :=
  (List.range 16).foldl (fun acc j => acc + a.bsums[j]! * ((scaleMin blk.scales (j / 2)).2 : Int)) 0

/-- Q6_K integer core: `Σ_{j<16} scale_j · Σ_{l<16} q6·q8`. -/
def dotQ6 (blk : Q6KBlock) (a : Q8KBlock) : Int :=
  (List.range 16).foldl (fun acc j =>
    acc + blk.scales[j]! * (List.range 16).foldl (fun s l => s + q6 blk (16 * j + l) * a.qs[16 * j + l]!) 0) 0

/-! ## Float rim -/

/-- The kernel's per-block contribution: `d·d_a·s1 − dmin·d_a·s2`. The kernel accumulates in
    `float`; the accumulation order is not part of the statement, hence the tolerance in the verifier. -/
def combineQ4K (blk : Q4KBlock) (a : Q8KBlock) : Float :=
  let da := a.d.toFloat
  blk.d.toFloat * da * (Float.ofInt (s1 blk a)) - blk.dmin.toFloat * da * (Float.ofInt (s2 blk a))

def combineQ6K (blk : Q6KBlock) (a : Q8KBlock) : Float :=
  blk.d.toFloat * a.d.toFloat * (Float.ofInt (dotQ6 blk a))

/-- One output element: a weight row of `nb` blocks against an activation row of `nb` blocks. -/
def rowQ4K (w : Array Q4KBlock) (a : Array Q8KBlock) : Float :=
  (List.range w.size).foldl (fun acc i => acc + combineQ4K w[i]! a[i]!) 0

/-! ## Theorems about the integer core -/

theorem scaleMin_fst_lt_64 (sc : Array UInt8) (j : Nat) : (scaleMin sc j).1 < 64 := by
  unfold scaleMin
  split
  · exact Nat.mod_lt _ (by decide)
  · exact packedLo_lt_64 _ _

theorem scaleMin_snd_lt_64 (sc : Array UInt8) (j : Nat) : (scaleMin sc j).2 < 64 := by
  unfold scaleMin
  split
  · exact Nat.mod_lt _ (by decide)
  · exact packedHi_lt_64 _ _

theorem q4_lt_16 (blk : Q4KBlock) (k : Nat) : q4 blk k < 16 := nibble_lt_16 _ _

theorem sub32_bounds (x : Nat) (hx : x < 64) : -32 ≤ (x : Int) - 32 ∧ (x : Int) - 32 < 32 := by omega

theorem q6_bounds (blk : Q6KBlock) (k : Nat) : -32 ≤ q6 blk k ∧ q6 blk k < 32 := by
  unfold q6
  exact sub32_bounds _ (sixBits_lt_64 _ _ _ _)

/-! ### Bounds: the integer core fits the circuit's field

Every per-block sum is bounded far below 2^31 − 1, the Mersenne-31 modulus the GKR
circuit uses, so field arithmetic there equals integer arithmetic. -/

/-- `0 ≤ w ≤ W` and `-Q ≤ q ≤ Q` give `-(W·Q) ≤ w·q ≤ W·Q`. -/
theorem mul_bound {w W q Q : Int} (hw0 : 0 ≤ w) (hw : w ≤ W) (hQ0 : 0 ≤ Q) (hq : -Q ≤ q) (hq' : q ≤ Q) :
    -(W * Q) ≤ w * q ∧ w * q ≤ W * Q := by
  have h1 : w * q ≤ w * Q := Int.mul_le_mul_of_nonneg_left hq' hw0
  have h2 : w * Q ≤ W * Q := Int.mul_le_mul_of_nonneg_right hw hQ0
  have h3 : w * (-Q) ≤ w * q := Int.mul_le_mul_of_nonneg_left hq hw0
  have h4 : w * (-Q) = -(w * Q) := Int.mul_neg w Q
  have h5 : -(W * Q) ≤ -(w * Q) := Int.neg_le_neg h2
  exact ⟨Int.le_trans h5 (h4 ▸ h3), Int.le_trans h1 h2⟩

/-- A left fold of `n` terms each bounded by `B` is bounded by `n·B`. -/
theorem foldl_add_bound (f : Nat → Int) (B : Int) (hB : ∀ i, -B ≤ f i ∧ f i ≤ B) (n : Nat) :
    -((n : Int) * B) ≤ (List.range n).foldl (fun acc i => acc + f i) 0 ∧
    (List.range n).foldl (fun acc i => acc + f i) 0 ≤ (n : Int) * B := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    have e : ((n + 1 : Nat) : Int) * B = (n : Int) * B + B := by
      rw [Int.natCast_add, Int.natCast_one, Int.add_mul, Int.one_mul]
    have := hB n
    rw [e]
    omega

theorem subDot_bound (blk : Q4KBlock) (a : Q8KBlock) (hq : ∀ (i : Nat), (-127 : Int) ≤ a.qs[i]! ∧ a.qs[i]! ≤ 127) (j : Nat) :
    -(32 * 1905 : Int) ≤ subDot blk a j ∧ subDot blk a j ≤ 32 * 1905 := by
  unfold subDot
  have hB : ∀ l, -(1905 : Int) ≤ ((q4 blk (32 * j + l) : Nat) : Int) * a.qs[32 * j + l]! ∧
      ((q4 blk (32 * j + l) : Nat) : Int) * a.qs[32 * j + l]! ≤ 1905 := by
    intro l
    have hw := q4_lt_16 blk (32 * j + l)
    have hw0 : (0 : Int) ≤ ((q4 blk (32 * j + l) : Nat) : Int) := Int.natCast_nonneg _
    have hw15 : ((q4 blk (32 * j + l) : Nat) : Int) ≤ 15 := by omega
    obtain ⟨lo, hi⟩ := hq (32 * j + l)
    exact mul_bound hw0 hw15 (by decide) lo hi
  exact foldl_add_bound _ 1905 hB 32

/-- `|s1| ≤ 8 · 63 · 32 · 1905 = 30,723,840 < 2^25`. -/
theorem s1_bound (blk : Q4KBlock) (a : Q8KBlock) (hq : ∀ (i : Nat), (-127 : Int) ≤ a.qs[i]! ∧ a.qs[i]! ≤ 127) :
    -(8 * (63 * 60960) : Int) ≤ s1 blk a ∧ s1 blk a ≤ 8 * (63 * 60960) := by
  unfold s1
  have hB : ∀ j, -(63 * 60960 : Int) ≤ ((scaleMin blk.scales j).1 : Int) * subDot blk a j ∧
      ((scaleMin blk.scales j).1 : Int) * subDot blk a j ≤ 63 * 60960 := by
    intro j
    have hs := scaleMin_fst_lt_64 blk.scales j
    have hs0 : (0 : Int) ≤ ((scaleMin blk.scales j).1 : Int) := Int.natCast_nonneg _
    have hs63 : ((scaleMin blk.scales j).1 : Int) ≤ 63 := by omega
    obtain ⟨lo, hi⟩ := subDot_bound blk a hq j
    exact mul_bound hs0 hs63 (by decide) lo hi
  exact foldl_add_bound _ (63 * 60960) hB 8

theorem s1_lt_2_pow_25 (blk : Q4KBlock) (a : Q8KBlock) (hq : ∀ (i : Nat), (-127 : Int) ≤ a.qs[i]! ∧ a.qs[i]! ≤ 127) :
    -(2 ^ 25 : Int) < s1 blk a ∧ s1 blk a < 2 ^ 25 := by
  have := s1_bound blk a hq
  omega

/-- `|s2| ≤ 16 · 63 · 2032 = 2,048,256`. -/
theorem s2_bound (blk : Q4KBlock) (a : Q8KBlock) (hb : ∀ (j : Nat), (-2032 : Int) ≤ a.bsums[j]! ∧ a.bsums[j]! ≤ 2032) :
    -(16 * (63 * 2032) : Int) ≤ s2 blk a ∧ s2 blk a ≤ 16 * (63 * 2032) := by
  unfold s2
  have hB : ∀ j, -(63 * 2032 : Int) ≤ a.bsums[j]! * ((scaleMin blk.scales (j / 2)).2 : Int) ∧
      a.bsums[j]! * ((scaleMin blk.scales (j / 2)).2 : Int) ≤ 63 * 2032 := by
    intro j
    have hm := scaleMin_snd_lt_64 blk.scales (j / 2)
    have hm0 : (0 : Int) ≤ ((scaleMin blk.scales (j / 2)).2 : Int) := Int.natCast_nonneg _
    have hm63 : ((scaleMin blk.scales (j / 2)).2 : Int) ≤ 63 := by omega
    obtain ⟨lo, hi⟩ := hb j
    have h := mul_bound hm0 hm63 (by decide) lo hi
    rw [Int.mul_comm (((scaleMin blk.scales (j / 2)).2 : Nat) : Int) (a.bsums[j]!)] at h
    exact h
  exact foldl_add_bound _ (63 * 2032) hB 16

end ReceiptsSpec.Quant
