# Python SDK reference

Package: `swiftvec` (PyPI). Module exposes two classes: `SwiftVec` and `SearchResult`.

```python
from swiftvec import SwiftVec, SearchResult
```

## SwiftVec

### Constructor

```python
SwiftVec(model_dir="models/leaf-ir", dim=None)
```

| parameter | type | default | description |
|---|---|---|---|
| `model_dir` | `str` | `"models/leaf-ir"` | directory containing the mdbr-leaf-ir model files (see [installation](installation.md)) |
| `dim` | `int \| None` | `None` | truncate embeddings to this many dimensions via Matryoshka truncation. `None` keeps all 768. Valid range 1-768 |

The HNSW index is created lazily on the first `add()`/`add_batch()`, no explicit index-creation call exists or is needed. The default configuration is int8 storage with exact f32 rerank and Dot metric.

### add_batch

```python
db.add_batch(ids, texts, metadatas=None) -> int
```

Embeds `texts` in one batched inference pass and indexes every document.

- `ids`: list of unique string identifiers.
- `texts`: list of document strings; must have equal length to `ids`.
- `metadatas`: optional list of dicts (or `None` entries) with JSON-compatible values (`str`, `int`, `float`, `bool`, `None`, nested `dict`, `list`). Must match the length of `ids`.

Returns the total number of documents in the database.

Metadata values are validated at insert time; unsupported types raise immediately rather than failing at query time.

### add

```python
db.add(id, text, metadata=None) -> int
```

Convenience wrapper around `add_batch` for a single document.

### search

```python
db.search(query, top_k=5, ef=None, alpha=None, filter=None) -> list[SearchResult]
```

| parameter | type | default | description |
|---|---|---|---|
| `query` | `str` | required | raw query text; embedded on device with the model's query prompt |
| `top_k` | `int` | `5` | number of results; `1..=len(db)` |
| `ef` | `int \| None` | semantic: `top_k * 8`; hybrid: `2 * fetch` | HNSW search width. Higher = better recall, slower |
| `alpha` | `float \| None` | `None` | hybrid weight in `[0, 1]`. `1.0` = pure vector ranking, `0.0` = pure BM25 ranking, in between = weighted Reciprocal Rank Fusion |
| `filter` | `dict \| None` | `None` | metadata predicate evaluated at query time, grammar documented in [filters](filters.md) |

Score semantics:

- **semantic mode** (`alpha=None`): `score` is cosine similarity of the full-precision vectors in `[−1, 1]`, higher is better.
- **hybrid mode** (`alpha` set): `score` is the fused RRF weight `alpha / (60 + rank_vector) + (1 − alpha) / (60 + rank_bm25)`. It is a rank-based score, not a similarity; only relative order matters.

In hybrid mode the engine fetches `max(16, 4 * top_k)` candidates from each retriever (`16x top_k` when a filter is present, to compensate for candidates removed by the predicate), applies the metadata filter to both candidate lists, then fuses.

Errors: searching an empty database raises; invalid `top_k`, out-of-range `alpha`, or malformed filters raise with descriptive messages.

### get

```python
db.get(id) -> dict
```

Returns `{"id": ..., "text": ..., "metadata": ...}` for a stored document id. Raises `KeyError` if unknown.

### save / load

```python
db.save(path)                      # writes a single .swiftvec snapshot file
SwiftVec.load(path, model_dir="models/leaf-ir")   # classmethod
```

The snapshot stores the packed-or-unpacked graph, quantized codes, rerank vectors, ids, metadata, and texts (format spec: [snapshots](snapshots.md)). The BM25 postings are rebuilt from texts on load. Loaded databases remain mutable, you can keep calling `add` after `load`.

### introspection

```python
len(db)            # number of documents
db.dim             # embedding dimension used for the index
db.info()          # dict: docs, dim, model_dir, hybrid, filter_ops
repr(db)           # <SwiftVec docs=N dim=D>
```

## SearchResult

| attribute | type | description |
|---|---|---|
| `id` | `str` | caller-supplied document id |
| `score` | `float` | cosine similarity (semantic) or fused RRF score (hybrid) |
| `text` | `str` | original document text |
| `metadata` | `dict \| None` | round-tripped metadata |

```python
for r in db.search("fast similarity search", top_k=3):
    print(r.id, r.score, r.text, r.metadata)
```

## Thread-safety

`SwiftVec` instances hold an ONNX session guarded by a mutex; concurrent calls from threads serialize on it. For parallel workloads create one instance per thread (each loads its own session) sharing nothing but the snapshot file. The GIL is released during embedding and during graph search.

## Not supported today

- deletion / updates of indexed documents (the graph has no tombstones, see roadmap)
- server-side pagination or cursors (single-process library)
- multiple named collections per instance (create one `SwiftVec` per collection)

These are tracked in the root README roadmap.
