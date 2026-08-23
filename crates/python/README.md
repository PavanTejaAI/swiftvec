# swiftvec python sdk

fully on-device semantic + keyword search with built-in embeddings. no cloud, no server, no account.

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

results = db.search("fast similarity search", top_k=3)
results = db.search("fast similarity search", top_k=3, alpha=0.5)
```

the model directory (mdbr-leaf-ir, apache-2.0) ships with the repo under `models/leaf-ir` via `tools/fetch_model.sh`. snapshots persist index, ids, metadata and texts: `db.save("my_collection.swiftvec")` / `SwiftVec.load("my_collection.swiftvec")`.
