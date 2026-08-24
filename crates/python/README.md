# swiftvec

Fully on-device vector search: embeddings, HNSW index, BM25 keyword search, hybrid fusion, metadata filters, and single-file persistence, all in your process. No cloud. No server. No account.

```python
from swiftvec import SwiftVec

db = SwiftVec()
db.add_batch(
    ["a", "b", "c"],
    [
        "Vector search finds similar content by meaning",
        "Photosynthesis converts sunlight into energy",
        "HNSW graphs enable fast approximate nearest neighbor search",
    ],
    metadatas=[{"topic": "ir"}, {"topic": "bio"}, {"topic": "ir"}],
)

db.search("how does fast similarity search work", top_k=3)
db.search("exact keyword: qubits", top_k=3, alpha=0.4)
db.search("similarity search", top_k=3, filter={"topic": "ir"})

db.save("my_collection.swiftvec")
db = SwiftVec.load("my_collection.swiftvec")
```

## Highlights

- **on-device embeddings**: MongoDB/mdbr-leaf-ir (Apache-2.0), quantized ONNX, ~2 ms/query on a laptop CPU
- **zero-dependency engine**: HNSW with heuristic selection, CSR-packed graph, AVX2+FMA kernels
- **int8 + rerank cascade**: 4x less memory at near-f32 recall; optional Matryoshka truncation to 256 dims
- **hybrid search**: semantic + BM25 fused with weighted RRF (`alpha`)
- **metadata filters**: `$eq $ne $gt $gte $lt $lte $in $nin $exists` with `$and/$or/$nor`, pushed into graph traversal
- **snapshots**: one file, fully mutable after load; deterministic builds from fixed seeds

100k-document benchmark with published recall@5: [github.com/PavanTejaAI/swiftvec](https://github.com/PavanTejaAI/swiftvec).

## First-run setup

The wheel bundles ONNX Runtime for your platform, nothing to install beyond the model weights (~130 MB, fetched once, fully offline afterwards):

```bash
git clone https://github.com/PavanTejaAI/swiftvec && cd swiftvec
python tools/fetch_model.py
```

Point `SwiftVec(model_dir="models/leaf-ir")` at the directory. Source builds can fetch a runtime with `bash tools/fetch_ort.sh` or point `ORT_DYLIB_PATH` at any onnxruntime shared library, details in [docs/installation.md](https://github.com/PavanTejaAI/swiftvec/blob/main/docs/installation.md).

## Documentation

| document | contents |
|---|---|
| [Python SDK reference](https://github.com/PavanTejaAI/swiftvec/blob/main/docs/python-sdk.md) | every method and parameter |
| [Filters](https://github.com/PavanTejaAI/swiftvec/blob/main/docs/filters.md) | metadata filter grammar |
| [Hybrid search](https://github.com/PavanTejaAI/swiftvec/blob/main/docs/hybrid-search.md) | alpha tuning |
| [Snapshots](https://github.com/PavanTejaAI/swiftvec/blob/main/docs/snapshots.md) | on-disk format |

Rust crate without Python bindings: `cargo add swiftvec-core`.

MIT © Pavan Teja. Embedding model Apache-2.0 © MongoDB.
