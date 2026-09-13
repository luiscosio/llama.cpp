pragma circom 2.1.6;

include "node_modules/circomlib/circuits/poseidon.circom";

// Zero-knowledge proof of ggml's Q4_K x Q8_K integer core for ROWS output rows of one weight
// tensor, with the weights private and bound to a Poseidon commitment.
//
//   for every row m and 256-wide block i:
//     s1[m,i] = sum_{j<8}  sc[m,i,j] * sum_{l<32} q4[m,i,32j+l] * q8[i,32j+l]
//     s2[m,i] = sum_{j<16} bsums[i,j] * mn[m,i,j/2],   bsums[i,j] = sum_{l<16} q8[i,16j+l]
//
// Private: the bits of the nibbles q4 (4 each), of the scales sc and mins mn (6 each), and a
// salt. Public: the activation quants q8, the sums s1 and s2, and the commitment, which is a
// Poseidon chain over [salt, packed nibbles, packed scales, packed mins]. Bits are the private
// signals so that every nibble and six-bit field is in range by construction (b*(b-1) = 0)
// and the packed elements are linear in the bits. Ranges of q8, s1, s2 are the verifier's job,
// as in the receipts verifier; the field is BN254, far above every magnitude here.

template Bits2Num(n) {
    signal input bits[n];
    signal output out;
    var acc = 0;
    for (var i = 0; i < n; i++) {
        bits[i] * (bits[i] - 1) === 0;
        acc += bits[i] * (1 << i);
    }
    out <== acc;
}

// Poseidon chain: h = P(e[0..16]); then h = P(h, e[16+15k .. 16+15k+15]) for each further chunk.
// n must be 16 + 15 * chunks.
template PoseidonChain(n) {
    signal input in[n];
    signal output out;
    var chunks = (n - 16) \ 15;
    component p[chunks + 1];
    p[0] = Poseidon(16);
    for (var i = 0; i < 16; i++) { p[0].inputs[i] <== in[i]; }
    for (var c = 0; c < chunks; c++) {
        p[c + 1] = Poseidon(16);
        p[c + 1].inputs[0] <== p[c].out;
        for (var i = 0; i < 15; i++) { p[c + 1].inputs[i + 1] <== in[16 + 15 * c + i]; }
    }
    out <== p[chunks].out;
}

template QDotRows(ROWS, K) {
    var NB = K \ 256;
    var NQ4 = ROWS * K;          // nibbles
    var NSC = ROWS * NB * 8;     // scales, and as many mins
    var PER_Q4 = 62;             // nibbles per packed field element (248 bits)
    var PER_SC = 41;             // six-bit values per packed element (246 bits)
    var NP_Q4 = (NQ4 + PER_Q4 - 1) \ PER_Q4;
    var NP_SC = (NSC + PER_SC - 1) \ PER_SC;
    var NE = 1 + NP_Q4 + 2 * NP_SC;                       // salt + packed elements
    var NCHAIN = NE <= 16 ? 16 : 16 + 15 * ((NE - 16 + 14) \ 15); // padded to 16 + 15k

    signal input q4bits[NQ4 * 4];
    signal input scbits[NSC * 6];
    signal input mnbits[NSC * 6];
    signal input salt;
    signal input q8[K];
    signal input s1[ROWS * NB];
    signal input s2[ROWS * NB];
    signal output commitment;

    // values from bits
    component q4b[NQ4];
    signal q4[NQ4];
    for (var n = 0; n < NQ4; n++) {
        q4b[n] = Bits2Num(4);
        for (var b = 0; b < 4; b++) { q4b[n].bits[b] <== q4bits[4 * n + b]; }
        q4[n] <== q4b[n].out;
    }
    component scb[NSC];
    component mnb[NSC];
    signal sc[NSC];
    signal mn[NSC];
    for (var n = 0; n < NSC; n++) {
        scb[n] = Bits2Num(6);
        mnb[n] = Bits2Num(6);
        for (var b = 0; b < 6; b++) { scb[n].bits[b] <== scbits[6 * n + b]; mnb[n].bits[b] <== mnbits[6 * n + b]; }
        sc[n] <== scb[n].out;
        mn[n] <== mnb[n].out;
    }

    // the commitment: Poseidon chain over salt and the packed values
    component chain = PoseidonChain(NCHAIN);
    chain.in[0] <== salt;
    for (var e = 0; e < NP_Q4; e++) {
        var acc = 0;
        for (var i = 0; i < PER_Q4; i++) {
            var idx = e * PER_Q4 + i;
            if (idx < NQ4) { acc += q4[idx] * (1 << (4 * i)); }
        }
        chain.in[1 + e] <== acc;
    }
    for (var e = 0; e < NP_SC; e++) {
        var accs = 0;
        var accm = 0;
        for (var i = 0; i < PER_SC; i++) {
            var idx = e * PER_SC + i;
            if (idx < NSC) { accs += sc[idx] * (1 << (6 * i)); accm += mn[idx] * (1 << (6 * i)); }
        }
        chain.in[1 + NP_Q4 + e] <== accs;
        chain.in[1 + NP_Q4 + NP_SC + e] <== accm;
    }
    for (var e = NE; e < NCHAIN; e++) { chain.in[e] <== 0; }
    commitment <== chain.out;

    // block sums of the activation, linear in the public q8
    var bsums[NB][16];
    for (var i = 0; i < NB; i++) {
        for (var j = 0; j < 16; j++) {
            var acc = 0;
            for (var l = 0; l < 16; l++) { acc += q8[i * 256 + j * 16 + l]; }
            bsums[i][j] = acc;
        }
    }

    // the integer core
    signal prod[ROWS][NB][8][32];
    signal sub[ROWS][NB][8];
    signal t1[ROWS][NB][8];
    signal t2[ROWS][NB][16];
    for (var m = 0; m < ROWS; m++) {
        for (var i = 0; i < NB; i++) {
            var acc1 = 0;
            for (var j = 0; j < 8; j++) {
                var accs = 0;
                for (var l = 0; l < 32; l++) {
                    prod[m][i][j][l] <== q4[m * K + i * 256 + j * 32 + l] * q8[i * 256 + j * 32 + l];
                    accs += prod[m][i][j][l];
                }
                sub[m][i][j] <== accs;
                t1[m][i][j] <== sc[m * NB * 8 + i * 8 + j] * sub[m][i][j];
                acc1 += t1[m][i][j];
            }
            s1[m * NB + i] === acc1;
            var acc2 = 0;
            for (var j = 0; j < 16; j++) {
                t2[m][i][j] <== bsums[i][j] * mn[m * NB * 8 + i * 8 + j \ 2];
                acc2 += t2[m][i][j];
            }
            s2[m * NB + i] === acc2;
        }
    }
}
