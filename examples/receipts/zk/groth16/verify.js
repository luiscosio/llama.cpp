// Shared browser and offline policy for the registered integer-core checkpoint.
(function (root) {
    "use strict";
    const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
    const SCHEME = "groth16-poseidon/bn254/qdot_rows";
    function field(s) {
        if (typeof s !== "string" || !/^(0|[1-9][0-9]{0,76})$/.test(s)) throw Error("public signals must be canonical decimal strings");
        const n = BigInt(s);
        if (n >= P) throw Error("public signal is outside the field");
        return n;
    }
    function bounded(s, bound, label) {
        const n = field(s);
        if (n > bound && n < P - bound) throw Error(label + " is outside its signed integer range");
    }
    function canonical(x) {
        if (Array.isArray(x)) return "[" + x.map(canonical).join(",") + "]";
        if (x && typeof x === "object") return "{" + Object.keys(x).sort().map(k => JSON.stringify(k) + ":" + canonical(x[k])).join(",") + "}";
        return JSON.stringify(x);
    }
    async function digest(x) {
        const crypto = root.crypto || require("node:crypto").webcrypto;
        const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical(x)));
        return Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, "0")).join("");
    }
    function validatePublic(pub, entry, group, vkey) {
        if (!entry || entry.type !== "Q4_K" || !entry.groth16 || entry.groth16.scheme !== SCHEME) throw Error("unsupported tensor registration");
        const [k, m] = entry.shape || [];
        const g = entry.groth16, rows = g.rows_per_group;
        if (entry.shape.length !== 2 || !Number.isSafeInteger(k) || !Number.isSafeInteger(m) || m <= 0 ||
            ![1024, 1536, 2048, 3072].includes(k) || rows !== (k <= 1536 ? 16 : 8) || m % rows ||
            g.circuit !== `r${rows}_k${k}` || !Array.isArray(g.groups) || g.groups.length !== m / rows) throw Error("unsupported circuit shape");
        if (!Number.isSafeInteger(group) || group < 0 || group >= g.groups.length) throw Error("group is outside the registered tensor");
        const blocks = rows * k / 256, count = 1 + k + 2 * blocks;
        if (!vkey || vkey.protocol !== "groth16" || vkey.curve !== "bn128" || vkey.nPublic !== count) throw Error("verification key shape differs from registration");
        if (!Array.isArray(pub) || pub.length !== count) throw Error("incorrect public signal count");
        field(pub[0]); field(g.groups[group]);
        for (let i = 1; i <= k; i++) bounded(pub[i], 127n, "q8");
        for (let i = 1 + k; i < 1 + k + blocks; i++) bounded(pub[i], 30723840n, "s1");
        for (let i = 1 + k + blocks; i < count; i++) bounded(pub[i], 2048256n, "s2");
    }
    async function verify(snarkjs, manifest, entry, group, vkey, pub, proof) {
        const result = { accept: false, pairing: false, bound: false, error: "" };
        try {
            if (manifest.manifest_version !== "registration/v1") throw Error("unsupported registration version");
            validatePublic(pub, entry, group, vkey);
            const pins = manifest.proof_system?.groth16?.verification_keys;
            const expected = pins && pins[entry.groth16.circuit];
            if (!expected || await digest(vkey) !== expected) throw Error("verification key is not pinned by the registration");
            result.bound = pub[0] === entry.groth16.groups[group];
            if (!result.bound) throw Error("commitment differs from the registered group");
            if (!proof || proof.protocol !== "groth16" || proof.curve !== "bn128") throw Error("unsupported proof format");
            result.pairing = await snarkjs.groth16.verify(vkey, pub, proof);
            result.accept = result.pairing && result.bound;
            if (!result.pairing) result.error = "proof verification failed";
        } catch (e) {
            result.error = String(e.message || e).slice(0, 240);
        }
        return result;
    }
    const api = { verify, validatePublic, canonical, digest };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    else root.ReceiptsVerifier = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
