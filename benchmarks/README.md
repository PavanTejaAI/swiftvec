# Benchmarks

## Protocol

We benchmark against the exact public protocol and corpus shipped in [moss's benchmarks directory](https://github.com/usemoss/moss/tree/main/benchmarks):

- corpus: `bench_100k_docs.json` — 100,000 real documents (their file, byte-identical)
- queries: their fixed set of 15 queries
- 3 warmup rounds, then 50 measured rounds x 15 queries = 750 measurements per configuration
- top_k = 5, **embedding time included** in every measurement (single-call embed + search)
- cold query reported separately
- addition moss does not publish: **recall@5 against exact brute-force ground truth** over the same embedding space

## Reproduce

```bash
bash tools/fetch_ort.sh
bash tools/fetch_model.sh
bash benchmarks/fetch_corpus.sh
bash benchmarks/run.sh
```

Results land in `results/`. The first run embeds the corpus on-device (~4 min) and caches it to `data/corpus-embeddings.bin`; later runs reuse the cache.

## Results

Windows x64 laptop CPU, single-threaded search, 4-thread embedding, m=32, ef_construction=400. Reference: moss.dev published numbers measured on an Apple M4 Pro (p50 3.1ms / p95 4.3ms / p99 5.4ms, embedding included, **recall unpublished**).

See `results/*.txt` for full output. Summary tables live in the root README.

## Fairness notes

- moss's published latency was measured on different hardware (M4 Pro); ours on a regular Windows laptop CPU. The claim this repo makes is architectural class, not identical silicon.
- moss's blog separately claims local embed 3ms + search 1.2ms + rerank 0.8ms (~5ms total), which does not reconcile with their benchmark's 3.1ms total; neither figure includes recall.
- their corpus is 100k templated documents with ~12,500 near-duplicates per topic — the hardest recall case we have found, and the axis their benchmark does not show.
