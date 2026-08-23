#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/benchmarks/data"
curl -fSL https://raw.githubusercontent.com/usemoss/moss/main/benchmarks/bench_100k_docs.json -o "$root/benchmarks/data/bench_100k_docs.json"
echo "corpus ready at benchmarks/data/bench_100k_docs.json"
