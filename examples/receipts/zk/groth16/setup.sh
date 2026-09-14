#!/bin/bash
# Groth16 setup for one compiled instance: build/<name>/main.r1cs -> main_final.zkey, verification_key.json
#   ./setup.sh build/r64_k1536 ptau/pot20_final.ptau
set -euo pipefail
BUILD="$1"; PTAU="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
export NODE_OPTIONS=--max-old-space-size=20000
SNARKJS="$(cd "$(dirname "$0")" && pwd)/node_modules/snarkjs/build/cli.cjs"
cd "$BUILD"
node "$SNARKJS" groth16 setup main.r1cs "$PTAU" main_0000.zkey
node "$SNARKJS" zkey contribute main_0000.zkey main_final.zkey --name="receipts-zk phase 2, single contributor (proof of concept)" -e="$(head -c 32 /dev/urandom | xxd -p)"
node "$SNARKJS" zkey export verificationkey main_final.zkey verification_key.json
rm -f main_0000.zkey
ls -la main_final.zkey verification_key.json
