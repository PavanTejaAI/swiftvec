# swiftvec

[![ci](https://github.com/pavanteja/swiftvec/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanteja/swiftvec/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![crates](https://img.shields.io/badge/crates.io-swiftvec--core-orange)](https://crates.io/crates/swiftvec-core)
[![pypi](https://img.shields.io/badge/pypi-swiftvec-blue)](https://pypi.org/project/swiftvec/)

**A fully on-device vector search engine — with the embedding model, index, keyword search, and persistence built in. No cloud. No server. No account.**

Embed, index, persist, and search entirely on your machine. The embedding model ([MongoDB/mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir), Apache-2.0, quantized ONNX) runs in-process; the HNSW engine, BM25 index, and snapshot format are implemented from scratch in Rust with zero runtime dependencies in the core.

```python
from swiftvec import SwiftVec

db = SwiftVec()
db.add_batch(["a", "b", "c"], [
    "Vector search finds similar content by meaning",
    "Photosynthesis converts sunlight into energy",
    "HNSW graphs enable fast approximate nearest neighbor search",
], metadatas=[{"topic": "ir"}, {"topic": "bio"}, {"topic": "ir"}])

results = db.search("how does fast similarity search work", top_k=3)
hybrid  = db.search("exact keyword: qubits", top_k=3, alpha=0.4)

db.save("my_collection.swiftvec")
db = SwiftVec.load("my_collection.swiftvec")
```

## Why

Most retrieval stacks call a remote vector database — the network round trip dominates the latency budget, and your documents leave the machine to be embedded and indexed by someone else's cloud. swiftvec runs the entire pipeline — embedding, graph construction, keyword index, query — inside your process. Retrieval is a function you call, not a service you query.

| | moss | swiftvec |
|---|---|---|
| index build | their cloud | **on device** |
| embedding | their cloud (closed model) | **on device** (open mdbr-leaf-ir, Apache-2.0) |
| query | local (after cloud sync) | **on device** |
| hybrid search | semantic + keyword | **semantic + BM25 with RRF fusion (`alpha`)** |
| engine source | closed binary | **open (Rust, zero-dep core)** |
| signup / api key | required | **none** |
| recall benchmarks | not published | **published, reproducible from this repo** |
| works offline from scratch | no | **yes** |

## Benchmarks

We run the exact public protocol and corpus shipped in [moss's benchmarks directory](https://github.com/usemoss/moss/tree/main/benchmarks): their `bench_100k_docs.json` (100,000 documents, byte-identical), their fixed 15 queries, 3 warmup rounds, 50 measured rounds (750 measurements), top_k=5, **embedding time included** — plus the one axis they do not publish: **recall@5 against exact brute-force ground truth**.

Their published results (Apple M4 Pro, from the moss README):

| System | P50 | P95 | P99 | Mean |
|--------|-----|-----|-----|------|
| Moss | 3.1 ms | 4.3 ms | 5.4 ms | 3.3 ms |
| Pinecone | 432.6 ms | 732.1 ms | 934.2 ms | 485.8 ms |
| Qdrant | 597.6 ms | 682.0 ms | 771.4 ms | 596.5 ms |
| ChromaDB | 351.8 ms | 423.5 ms | 538.5 ms | 358.0 ms |

Our measured results (Windows x64 laptop CPU, single-threaded search, 4-thread embedding, m=32, ef_construction=400; full output in [`benchmarks/results/`](benchmarks/results/)):

| config | ef | recall@5 | search P50 | e2e P50 | e2e P95 | e2e P99 |
|---|---|---|---|---|---|---|
| f32 768d | 256 | 0.960 | 1.77 ms | 4.26 ms | 5.19 ms | 5.60 ms |
| f32 768d | 512 | **1.000** | 2.33 ms | 4.84 ms | 6.22 ms | 6.89 ms |
| int8 768d + rerank | 512 | 0.973 | 1.50 ms | 3.48 ms | 4.54 ms | 4.89 ms |
| int8 256d (MRL) | 32 | 0.533 | 150 µs | 1.75 ms | 2.65 ms | 3.66 ms |
| **int8 256d (MRL) + rerank** | **512** | **0.960** | 886 µs | **2.54 ms** | **2.82 ms** | **3.31 ms** |

Stated plainly:

- **At recall 0.960, the int8-256+rerank tier is 2.54 ms P50 / 3.31 ms P99 on a regular laptop CPU — faster than moss's published 3.1 ms P50 / 5.4 ms P99 measured on an M4 Pro, with recall actually published.** Their recall at their operating point is unknown — nobody outside their company can know.
- On-device embedding (mdbr-leaf-ir, quantized ONNX, in-process) accounts for ~2.1 ms of the total and beats moss's separately-claimed 3 ms embedding on an M4 Pro.
- Search-only, the int8-256 tier runs at 150–886 µs P50 — 1.4–8× faster than moss's claimed 1.2 ms search.
- Exact recall 1.0 is available (f32, ef=512) — something no moss benchmark demonstrates.
- Their corpus is 100k templated documents with ~12,500 near-duplicates per topic: the hardest recall case we have found, and the axis their benchmark does not show. Their own published figures do not reconcile (3.1 ms total vs. their blog's 3 ms embed + 1.2 ms search + 0.8 ms rerank).
- Hardware differs (M4 Pro vs laptop CPU); the claim is architectural class, not identical silicon.

Reproduce everything: `bash benchmarks/fetch_corpus.sh && bash benchmarks/run.sh` (see [`benchmarks/README.md`](benchmarks/README.md)).

## Install

Python (3.9+):

```bash
uv pip install swiftvec
```

First run fetches the embedding model and ONNX Runtime:

```bash
git clone https://github.com/pavanteja/swiftvec && cd swiftvec
bash tools/fetch_ort.sh
bash tools/fetch_model.sh
python examples/basic.py
```

Rust:

```bash
cargo add swiftvec-core
```

## Engine

- **HNSW from scratch** — heuristic neighbor selection (Algorithm 4), bidirectional links, shrink-on-overflow, deterministic builds
- **CSR-packed serving graph** — flat offsets+targets after `pack()`, zero pointer-chasing per hop, software prefetch of neighbor vectors
- **AVX2+FMA kernels** — f32 and int8 dot/L2 with runtime CPU dispatch and auto-vectorized scalar fallback
- **int8 storage + f32 rerank cascade** — int8 traversal at 4× less memory, exact rerank of candidates recovers recall
- **MRL truncation** — 768→256 dims via the model's Matryoshka capability, 12× memory reduction vs f32-768
- **BM25 + weighted RRF fusion** — hybrid semantic+keyword search in one call (`alpha`)
- **Brute-force exact oracle** — recall ground truth for every benchmark; no mocks anywhere in the test suite
- **Snapshots** — single-file persistence of index, ids, metadata, and texts; databases stay mutable across save/load

## Layout

```
crates/core     zero-dependency engine: hnsw, csr, kernels, int8, bm25, rrf, oracle, snapshots
crates/embed    mdbr-leaf-ir on-device runtime: quantized onnx, rust tokenizers, dense projection
crates/python   pyo3 sdk → pypi package `swiftvec` (author: Pavan Teja)
crates/cli      live (moss-protocol benchmark), bench, embed tools
benchmarks/     protocol, corpus fetch, run scripts, committed results
tools/          model + runtime fetch scripts
```

## Roadmap

- [x] HNSW engine, SIMD kernels, int8 + rerank, CSR packing, filters
- [x] On-device embedding (mdbr-leaf-ir quantized ONNX)
- [x] BM25 + RRF hybrid search
- [x] Snapshots (persistence) + Python SDK
- [x] Moss-protocol benchmark with published recall
- [ ] mmap zero-copy snapshot loading
- [ ] Binary first-pass cascade (1-bit signatures → int8 → f32)
- [ ] Metadata filter operators at query time ($eq / $and / $in)
- [ ] Node.js SDK (napi-rs) and WebAssembly build
- [ ] SIFT1M / GIST1M Pareto suite vs usearch, hnswlib, faiss

## Publishing

```bash
uvx maturin build --release -m crates/python/Cargo.toml -o dist
uv publish dist/*
```

## License

MIT — © Pavan Teja. The bundled embedding model [MongoDB/mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) is Apache-2.0.
