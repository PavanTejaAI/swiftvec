#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --release
bin=target/release/swiftvec
[ -f "$bin.exe" ] && bin="$bin.exe"
common=(--m 32 --ef-construction 400)
"$bin" live --storage f32 "${common[@]}" --ef 32,64,128,256,512 > benchmarks/results/f32-768.txt
"$bin" live --storage int8 "${common[@]}" --ef 32,64,128,256,512 > benchmarks/results/int8-768.txt
"$bin" live --storage int8 --rerank "${common[@]}" --ef 64,128,256,512 > benchmarks/results/int8-768-rerank.txt
"$bin" live --storage int8 --dim 256 "${common[@]}" --ef 32,64,128,256,512 > benchmarks/results/int8-256.txt
"$bin" live --storage int8 --dim 256 --rerank "${common[@]}" --ef 64,128,256,512 > benchmarks/results/int8-256-rerank.txt
echo "all benchmark results written to benchmarks/results/"
