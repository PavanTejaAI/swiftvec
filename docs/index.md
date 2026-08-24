# swiftvec documentation

<p align="center">
  <img src="../assets/banner.svg" alt="swiftvec, fully on-device vector search engine" width="100%">
</p>

swiftvec is a fully on-device vector search engine. The embedding model, the HNSW graph, the BM25 keyword index, hybrid fusion, and persistence all run inside your process, retrieval is a function you call, not a service you query.

## The pipeline

```
text ──▶ mdbr-leaf-ir (quantized ONNX, in-process) ──▶ 768-d vector ──▶ optional MRL truncation ──▶ normalized vector
                                                                                                        │
query text ──▶ same encoder (with query prompt) ───────────────────────────────▶ query vector ─────────┤
                                                                                                        │
                                                       ┌────────────────────────────────────────────────┤
                                                       ▼                                                ▼
                                              HNSW graph search                                BM25 keyword search
                                              (int8 traversal,                                 (postings index over
                                               f32 rerank)                                      the stored texts)
                                                       │                                                │
                                                       └────────── weighted RRF fusion ─────────────────┘
                                                                            │
                                                                            ▼
                                                          metadata filter applied at query time
                                                                            │
                                                                            ▼
                                                              SearchResult(id, score, text, metadata)
```

## Repository layout

| path | role |
|---|---|
| `crates/core` | zero-dependency engine: HNSW, CSR packing, SIMD kernels, int8 quantization, BM25, RRF, brute-force oracle, snapshots |
| `crates/embed` | on-device embedding runtime: quantized ONNX via ONNX Runtime, rust tokenizers, dense projection |
| `crates/python` | PyO3 SDK published to PyPI as `swiftvec` |
| `crates/cli` | `swiftvec` binary: `bench`, `embed`, `live` subcommands |
| `benchmarks/` | moss-protocol benchmark: corpus fetch, run scripts, committed results |
| `docs/` | this documentation set |
| `assets/` | banner and media used by the README |
| `tools/` | model and ONNX Runtime fetch scripts, PyPI release helper |

## Choosing a configuration

The Python SDK ships one tuned configuration by default: int8 storage with exact f32 rerank, Dot metric. You control two levers:

| you want | do this |
|---|---|
| smallest memory footprint | `SwiftVec(dim=256)`, Matryoshka truncation to 256 dims |
| maximum recall | full 768 dims, raise `ef`: `db.search(q, top_k=10, ef=512)` |
| exact-keyword heavy queries | hybrid mode with a lower alpha: `db.search(q, alpha=0.3)` |
| strict metadata scoping | pass `filter=` expressions, see [filters](filters.md) |

## Where to go next

- [Installation](installation.md): pip/uv, source builds, ONNX Runtime setup
- [Python SDK reference](python-sdk.md): every method and parameter
- [Filter grammar](filters.md): operators, semantics, examples
- [Engine internals](engine.md): how the HNSW engine works
- [Embeddings](embeddings.md): model details, prompts, pooling, truncation
- [Hybrid search](hybrid-search.md): BM25 + RRF fusion and tuning
- [Snapshots](snapshots.md): on-disk format specification
- [Benchmarks](benchmarks.md): protocol, methodology, reproduction
- [Publishing](publishing.md): maintainer release guide
