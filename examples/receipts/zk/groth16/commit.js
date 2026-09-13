// Poseidon commitment of one row group, exactly as qdot_rows.circom computes it:
//   chain over [salt, packed nibbles (62 per element), packed scales (41 per element),
//   packed mins (41 per element)], padded with zeros to 16 + 15k elements;
//   h = P(e[0..16]); h = P(h, next 15) ...
// Usage: node commit.js < elements.json   where elements.json = {"salt": "0", "q4": [..], "sc": [..], "mn": [..]}
// Prints the commitment as a decimal string. Also used by register.py.
const { buildPoseidon } = require("circomlibjs");

function pack(values, per, bits) {
  const out = [];
  for (let e = 0; e * per < values.length; e++) {
    let acc = 0n;
    for (let i = 0; i < per && e * per + i < values.length; i++) {
      acc += BigInt(values[e * per + i]) << BigInt(bits * i);
    }
    out.push(acc);
  }
  return out;
}

async function main() {
  const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
  const poseidon = await buildPoseidon();
  const F = poseidon.F;
  const commit = (g) => {
    const elems = [BigInt(input.salt || 0), ...pack(g.q4, 62, 4), ...pack(g.sc, 41, 6), ...pack(g.mn, 41, 6)];
    const n = elems.length;
    const nchain = n <= 16 ? 16 : 16 + 15 * Math.ceil((n - 16) / 15);
    while (elems.length < nchain) elems.push(0n);
    let h = poseidon(elems.slice(0, 16));
    for (let c = 16; c < nchain; c += 15) {
      h = poseidon([h, ...elems.slice(c, c + 15)]);
    }
    return F.toString(h);
  };
  if (input.groups) {
    console.log(JSON.stringify(input.groups.map(commit)));
  } else {
    console.log(commit(input));
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
