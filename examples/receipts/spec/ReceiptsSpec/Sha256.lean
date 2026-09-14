/-!
# SHA-256, executable

A direct transcription of FIPS 180-4 over `ByteArray`, so the spec can recompute
receipt hashes, Merkle roots and challenge indices without a foreign library.
Correctness is established by test vectors (`spec-check sha256`), not by proof:
the protocol theorems treat the hash as an abstract function.
-/

namespace ReceiptsSpec.Sha256

private def K : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

private def H0 : Array UInt32 :=
  #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

@[inline] private def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

/-- Message padding: 0x80, zeros, and the bit length as a 64-bit big-endian integer. -/
def pad (msg : ByteArray) : ByteArray := Id.run do
  let bitLen : UInt64 := (UInt64.ofNat msg.size) * 8
  let mut out := msg.push 0x80
  while out.size % 64 != 56 do
    out := out.push 0
  for i in [0:8] do
    out := out.push (UInt8.ofNat ((bitLen >>> (UInt64.ofNat (8 * (7 - i)))).toNat % 256))
  return out

private def be32 (b : ByteArray) (i : Nat) : UInt32 :=
  ((b.get! i).toUInt32 <<< 24) ||| ((b.get! (i+1)).toUInt32 <<< 16) ||| ((b.get! (i+2)).toUInt32 <<< 8) ||| (b.get! (i+3)).toUInt32

private def schedule (b : ByteArray) (off : Nat) : Array UInt32 := Id.run do
  let mut w : Array UInt32 := Array.mkEmpty 64
  for t in [0:16] do
    w := w.push (be32 b (off + 4 * t))
  for t in [16:64] do
    let s0 := rotr w[t-15]! 7 ^^^ rotr w[t-15]! 18 ^^^ (w[t-15]! >>> 3)
    let s1 := rotr w[t-2]! 17 ^^^ rotr w[t-2]! 19 ^^^ (w[t-2]! >>> 10)
    w := w.push (w[t-16]! + s0 + w[t-7]! + s1)
  return w

private def compress (h : Array UInt32) (w : Array UInt32) : Array UInt32 := Id.run do
  let mut a := h[0]!; let mut b := h[1]!; let mut c := h[2]!; let mut d := h[3]!
  let mut e := h[4]!; let mut f := h[5]!; let mut g := h[6]!; let mut hh := h[7]!
  for t in [0:64] do
    let S1 := rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25
    let ch := (e &&& f) ^^^ ((~~~e) &&& g)
    let t1 := hh + S1 + ch + K[t]! + w[t]!
    let S0 := rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22
    let maj := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
    let t2 := S0 + maj
    hh := g; g := f; f := e; e := d + t1; d := c; c := b; b := a; a := t1 + t2
  return #[h[0]! + a, h[1]! + b, h[2]! + c, h[3]! + d, h[4]! + e, h[5]! + f, h[6]! + g, h[7]! + hh]

/-- The 32-byte digest. -/
def hash (msg : ByteArray) : ByteArray := Id.run do
  let p := pad msg
  let mut h := H0
  let mut off := 0
  while off < p.size do
    h := compress h (schedule p off)
    off := off + 64
  let mut out := ByteArray.empty
  for x in h do
    out := out.push (UInt8.ofNat ((x >>> 24).toNat % 256))
    out := out.push (UInt8.ofNat ((x >>> 16).toNat % 256))
    out := out.push (UInt8.ofNat ((x >>> 8).toNat % 256))
    out := out.push (UInt8.ofNat (x.toNat % 256))
  return out

def hexDigit (n : Nat) : Char := Char.ofNat (if n < 10 then 48 + n else 87 + n)

def toHex (b : ByteArray) : String :=
  b.foldl (fun s x => s.push (hexDigit (x.toNat / 16)) |>.push (hexDigit (x.toNat % 16))) ""

def hexVal (c : Char) : Nat :=
  if c.isDigit then c.toNat - '0'.toNat
  else if 'a' ≤ c ∧ c ≤ 'f' then c.toNat - 'a'.toNat + 10
  else if 'A' ≤ c ∧ c ≤ 'F' then c.toNat - 'A'.toNat + 10 else 0

def ofHex (s : String) : ByteArray := Id.run do
  let cs := s.toList.toArray
  let mut out := ByteArray.empty
  let mut i := 0
  while i + 1 < cs.size do
    out := out.push (UInt8.ofNat (16 * hexVal cs[i]! + hexVal cs[i+1]!))
    i := i + 2
  return out

def hashHex (msg : ByteArray) : String := toHex (hash msg)

def ofString (s : String) : ByteArray := s.toUTF8

end ReceiptsSpec.Sha256
