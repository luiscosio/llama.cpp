#!/usr/bin/env node
// Arguments name independently trusted registration materials, never files selected by a proof.
const fs = require("node:fs");
const snarkjs = require("snarkjs");
const verifier = require("./verify.js");
function read(path, limit) {
    if (fs.statSync(path).size > limit) throw Error("package file exceeds size limit");
    return JSON.parse(fs.readFileSync(path, "utf8"));
}
(async () => {
    const [key, registration, tensor, group, directory] = process.argv.slice(2);
    if (process.argv.length !== 7 || !/^(0|[1-9][0-9]*)$/.test(group)) throw Error("usage: verify_package.cjs KEY MANIFEST TENSOR GROUP PACKAGE_DIR");
    const manifest = read(registration, 16 * 1024 * 1024);
    const hash = require("node:crypto").createHash("sha256").update(fs.readFileSync(__dirname + "/verify.js")).digest("hex");
    if (manifest.proof_system?.groth16?.verifier_sha256 !== hash) throw Error("verifier source is not pinned by the registration");
    const entry = manifest.tensors.find(t => t.name === tensor);
    const result = await verifier.verify(snarkjs, manifest, entry, Number(group), read(key, 2 * 1024 * 1024),
        read(directory + "/public.json", 2000000), read(directory + "/proof.json", 8192));
    console.log(JSON.stringify(result));
    process.exit(result.accept ? 0 : 1);
})().catch(e => { console.error(String(e.message || e)); process.exit(2); });
