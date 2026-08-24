# Benchmarks

## Protocol

The `live` subcommand reproduces the exact public protocol of [moss's benchmarks directory](https://github.com/usemoss/moss/tree/main/benchmarks):

- corpus: their `bench_100k_docs.json`, 100,000 documents, byte-identical download
- queries: their fixed set of 15 queries (embedded on device with the query prompt)
- 3 warmup rounds, then 50 measured rounds × 15 queries = **750 measurements per configuration**
- top_k = 5, **embedding time included** in every end-to-end measurement
- cold query reported separately
- additionally, the axis moss does not publish: **recall@5 against exact brute-force ground truth** computed over the same embedding space (`swiftvec-core::top_k`)

Search-only timings are reported separately from end-to-end so kernel-level changes are visible.

## Reproduce

```bash
bash tools/fetch_ort.sh
bash tools/fetch_model.sh
bash benchmarks/fetch_corpus.sh
bash benchmarks/run.sh
```

The first run embeds the full corpus on device (~4 min) and caches it to `benchmarks/data/corpus-embeddings.bin`; later runs reuse the cache. Results land in `benchmarks/results/`. All configurations use m=32, ef_construction=400.

## Our results

Measured with this repository: Intel i5 laptop (Windows x64, 16 GB RAM), single-threaded search, 4-thread embedding, m=32, ef_construction=400.

| config | ef | recall@5 | search P50 | e2e P50 | e2e P95 | e2e P99 |
|---|---|---|---|---|---|---|
| f32 768d | 256 | 0.960 | 1.77 ms | 4.26 ms | 5.19 ms | 5.60 ms |
| f32 768d | 512 | **1.000** | 2.33 ms | 4.84 ms | 6.22 ms | 6.89 ms |
| int8 768d + rerank | 512 | 0.973 | 1.50 ms | 3.48 ms | 4.54 ms | 4.89 ms |
| int8 256d (MRL) | 32 | 0.533 | 150 µs | 1.75 ms | 2.65 ms | 3.66 ms |
| **int8 256d (MRL) + rerank** | **512** | **0.960** | **886 µs** | **2.54 ms** | **2.82 ms** | **3.31 ms** |

Headlines:

- **2.54 ms P50 / 3.31 ms P99 at recall 0.960** (int8-256d MRL + rerank, ef=512), embedding included, on a Intel i5 laptop, faster than moss's published 3.1 ms P50 / 5.4 ms P99 on an Apple M4 Pro
- search-only floor of **150-886 µs P50** across the int8 tiers
- exact **recall 1.000** available at f32, ef=512
- every recall number is measured against **exact brute-force ground truth** over the identical embedding space, recall is published, not implied

Raw output for every configuration: [`results/`](results/) (`f32-768.txt`, `int8-768.txt`, `int8-768-rerank.txt`, `int8-256.txt`, `int8-256-rerank.txt`). Each file includes build throughput, pack time, memory footprint, cold-query latency, and the recall/latency table across the ef sweep.

## Leaderboard

Moss protocol: 100k docs, their 15 queries, top_k=5, embedding time included. swiftvec row measured with this repository; other rows as published by moss.dev on an Apple M4 Pro or their own infrastructure.

| System | Hardware | P50 | P95 | P99 | Mean | recall@5 |
|---|---|---|---|---|---|---|
| **swiftvec**: int8 256d MRL + rerank (ef=512) | Intel i5 laptop | **2.54 ms** | **2.82 ms** | **3.31 ms** | **2.57 ms** | **0.960** |
| Moss | Apple M4 Pro | 3.1 ms | 4.3 ms | 5.4 ms | 3.3 ms | not published |
| ChromaDB | per moss's setup | 351.8 ms | 423.5 ms | 538.5 ms | 358.0 ms | not published |
| Pinecone | per moss's setup | 432.6 ms | 732.1 ms | 934.2 ms | 485.8 ms | not published |
| Qdrant | per moss's setup | 597.6 ms | 682.0 ms | 771.4 ms | 596.5 ms | not published |

## Fairness notes

- Latency was measured on different hardware (M4 Pro vs Windows laptop CPU). The claim this repo makes is architectural class, not identical silicon.
- Moss's blog separately claims ~3 ms embed + 1.2 ms search + 0.8 ms rerank (~5 ms total), which does not reconcile with its published 3.1 ms total; neither figure includes recall.
- Their corpus is 100k templated documents with ~12,500 near-duplicates per topic, a genuinely hard recall case, and precisely why recall@5 against exact ground truth is published here alongside latency.
- Recall is measured at the first measured round after warmup, per configuration, over all 15 queries.

## Binary cascade A/B

`--cascade` (int8+rerank, 256-bit Hamming prefilter, tau=128) measured on synthetic data, same seeds:

| dataset | ef | recall (off → on) | p50 (off → on) | p99 (off → on) |
|---|---|---|---|---|
| clustered 50k dim=128 | 64 | 1.000 → 1.000 | 50 µs → 58 µs | 295 µs → **115 µs** |
| clustered 50k dim=128 | 256 | 1.000 → 1.000 | 282 µs → **203 µs** | 1167 µs → **549 µs** |
| uniform 20k dim=64 | 512 | 1.000 → 1.000 | 349 µs → 387 µs | 930 µs → **656 µs** |

Reading: recall is unchanged everywhere; tail latency improves at large `ef`; small-dim workloads pay a little on p50 because an int8 dot at dim=64 costs less than the signature check. Enable per workload, it is a flag, not a default.

## Micro-benchmarks

`swiftvec bench` runs synthetic clustered/uniform datasets with configurable n, dim, metric, storage tier, qrange, filter selectivity, m, ef_construction, and an ef sweep. It reports build rate, layer count, memory split (vectors vs graph), and per-ef p50/p95/p99/qps/recall against brute-force truth:

```bash
swiftvec bench --n 100000 --dim 128 --storage int8 --rerank --ef 64,128,256
```

Every number is reproducible: datasets come from seeded generators and the RNG used by index construction is fixed-seed.
