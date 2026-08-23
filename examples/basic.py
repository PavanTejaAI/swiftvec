import time

from swiftvec import SwiftVec

db = SwiftVec()

ids = ["ir-1", "bio-1", "ir-2", "fin-1", "quantum-1"]
docs = [
    "Vector search finds similar content by meaning using embeddings",
    "Photosynthesis converts sunlight into chemical energy in plants",
    "HNSW graphs enable fast approximate nearest neighbor search at scale",
    "The stock market closed higher after the interest rate decision",
    "Quantum computers use qubits to perform certain computations faster",
]
metas = [
    {"topic": "retrieval"},
    {"topic": "biology"},
    {"topic": "retrieval"},
    {"topic": "finance"},
    {"topic": "physics"},
]

t0 = time.perf_counter()
db.add_batch(ids, docs, metadatas=metas)
print(f"indexed {len(db)} docs in {(time.perf_counter() - t0) * 1000:.1f}ms")

db.save("my_collection.swiftvec")
loaded = SwiftVec.load("my_collection.swiftvec")
print(f"snapshot loaded: {len(loaded)} docs")

t0 = time.perf_counter()
results = loaded.search("how does fast similarity search work", top_k=3)
elapsed = (time.perf_counter() - t0) * 1000
print(f"semantic search in {elapsed:.2f}ms (embed + search, fully on-device)")
for r in results:
    print(f"  {r.id}  {r.score:.4f}  {r.metadata}")

hybrid = loaded.search("exact keyword match: qubits", top_k=3, alpha=0.4)
print("hybrid search (alpha=0.4):")
for r in hybrid:
    print(f"  {r.id}  {r.score:.4f}  {r.metadata}")
