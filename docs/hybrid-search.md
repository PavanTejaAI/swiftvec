# Hybrid search

`db.search(..., alpha=a)` runs both retrievers and fuses their rankings in one call.

## The two retrievers

**Vector (semantic).** HNSW beam search over embedded vectors, matches meaning, tolerant to paraphrase, weak on rare exact tokens.

**Keyword (lexical).** BM25 over the stored texts with an inverted postings index:

```
score(q, d) = Σ_t∈q  idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d|/avgdl))
```

with `k1 = 1.2`, `b = 0.75`, ASCII-alphanumeric lowercased tokens. BM25 catches exact identifiers, error codes, part numbers, anything where token identity, not meaning, is the signal.

## Fusion

Weighted Reciprocal Rank Fusion with `RRF_K = 60`:

```
rrf(d) = alpha      / (60 + rank_vector(d))
       + (1 − alpha) / (60 + rank_bm25(d))
```

Ranks are 0-based positions in each retriever's top list. Documents absent from a list simply get no contribution from it. The fused score is rank-based: its magnitude is not comparable across queries, only the ordering within one query matters.

## Choosing alpha

| alpha | behavior |
|---|---|
| `None` (default) | pure semantic search |
| `1.0` | semantic ranking only, but returned through the fusion path |
| `0.7-0.9` | mostly semantic; keyword results break ties on exact terms |
| `0.3-0.5` | balanced; good default for mixed keyword-heavy workloads |
| `0.0-0.2` | mostly lexical; semantics only reorders ties |

## Candidate fetching

In hybrid mode each retriever produces `fetch = max(16, 4 · top_k)` candidates (`ef` defaults to `2 · fetch`, override with `ef=`). If you pass a metadata `filter`, it is applied to **both** candidate lists before fusion, see [filters](filters.md).

## Example

```python
db.search("exact keyword: qubits", top_k=5, alpha=0.4)
db.search("RFC 9110 caching directives", top_k=5, alpha=0.25)
db.search("papers similar to HNSW", top_k=5)          # pure semantic
```

## When hybrid helps

- corpora containing identifiers, code, or rare proper nouns
- queries mixing natural language with exact tokens ("how does function `pack()` work")
- recall-critical retrieval where either signal alone leaves gaps

The benchmark suite measures pure vector latency; hybrid adds one BM25 pass (microseconds for typical corpora) plus fusion.
