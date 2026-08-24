# Benchmarks

## Our results

Measured with this repository: Intel i5 laptop (Windows x64, 16 GB RAM), single-threaded search, 4-thread embedding, m=32, ef_construction=400. Corpus, queries, warmup, rounds, and top_k follow the exact public protocol of [moss's benchmarks directory](https://github.com/usemoss/moss/tree/main/benchmarks); embedding time is inside every end-to-end measurement.

| config | ef | recall@5 | search P50 | e2e P50 | e2e P95 | e2e P99 |
|---|---|---|---|---|---|---|
| f32 768d | 256 | 0.960 | 1.77 ms | 4.26 ms | 5.19 ms | 5.60 ms |
| f32 768d | 512 | **1.000** | 2.33 ms | 4.84 ms | 6.22 ms | 6.89 ms |
| int8 768d + rerank | 512 | 0.973 | 1.50 ms | 3.48 ms | 4.54 ms | 4.89 ms |
| int8 256d (MRL) | 32 | 0.533 | 150 µs | 1.75 ms | 2.65 ms | 3.66 ms |
| **int8 256d (MRL) + rerank** | **512** | **0.960** | **886 µs** | **2.54 ms** | **2.82 ms** | **3.31 ms** |

Highlights:

- fastest competitive tier: **2.54 ms P50 / 3.31 ms P99 at recall 0.960**, embedding included, on a plain laptop CPU, faster than moss's published 3.1 ms P50 / 5.4 ms P99 on an Apple M4 Pro
- search-only floor: **150-886 µs P50** across the int8 tiers (1.4-8x faster than moss's claimed 1.2 ms search)
- exact retrieval: **recall 1.000** (f32, ef=512), no benchmark from moss demonstrates this
- every recall number is measured against **exact brute-force ground truth** over the identical embedding space, the axis moss does not publish

Raw console output per configuration: [`results/f32-768.txt`](results/f32-768.txt), [`results/int8-768.txt`](results/int8-768.txt), [`results/int8-768-rerank.txt`](results/int8-768-rerank.txt), [`results/int8-256.txt`](results/int8-256.txt), [`results/int8-256-rerank.txt`](results/int8-256-rerank.txt). Each includes build throughput, pack time, memory footprint, cold-query latency, and the recall/latency table across the ef sweep.

## Leaderboard

Moss protocol: 100k docs, their 15 queries, top_k=5, embedding time included. swiftvec row measured with this repository; other rows as published by moss.dev on an Apple M4 Pro or their own infrastructure.

| System | Hardware | P50 | P95 | P99 | Mean | recall@5 |
|---|---|---|---|---|---|---|
| **swiftvec**: int8 256d MRL + rerank (ef=512) | Intel i5 laptop | **2.54 ms** | **2.82 ms** | **3.31 ms** | **2.57 ms** | **0.960** |
| Moss | Apple M4 Pro | 3.1 ms | 4.3 ms | 5.4 ms | 3.3 ms | not published |
| ChromaDB | per moss's setup | 351.8 ms | 423.5 ms | 538.5 ms | 358.0 ms | not published |
| Pinecone | per moss's setup | 432.6 ms | 732.1 ms | 934.2 ms | 485.8 ms | not published |
| Qdrant | per moss's setup | 597.6 ms | 682.0 ms | 771.4 ms | 596.5 ms | not published |

The one column nobody else publishes is the one that matters most: recall@5 against exact brute-force ground truth.

## Protocol

- corpus: their `bench_100k_docs.json`, 100,000 documents, byte-identical download
- queries: their fixed set of 15 queries, embedded on device with the model's query prompt
- 3 warmup rounds, then 50 measured rounds × 15 queries = **750 measurements per configuration**
- top_k = 5, **embedding time included** in every end-to-end measurement
- cold query reported separately
- additionally, the axis moss does not publish: **recall@5 against exact brute-force ground truth** (`swiftvec-core::top_k`) over the same embedding space

## Reproduce

```bash
bash tools/fetch_ort.sh
bash tools/fetch_model.sh
bash benchmarks/fetch_corpus.sh
bash benchmarks/run.sh
```

The first run embeds the full corpus on device (~4 min) and caches it to `data/corpus-embeddings.bin`; later runs reuse the cache. Results land in `results/`. All configurations use m=32, ef_construction=400.

## Fairness notes

- Latency was measured on different hardware (M4 Pro vs Windows laptop CPU). The claim this repo makes is architectural class, not identical silicon.
- Moss's blog separately claims ~3 ms embed + 1.2 ms search + 0.8 ms rerank (~5 ms total), which does not reconcile with its published 3.1 ms total; neither figure includes recall.
- Their corpus is 100k templated documents with ~12,500 near-duplicates per topic, a genuinely hard recall case, and precisely why recall@5 is published here alongside latency.
- Recall is measured at the first measured round after warmup, per configuration, over all 15 queries.
