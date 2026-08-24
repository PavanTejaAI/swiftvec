from swiftvec import SwiftVec

db = SwiftVec(dim=256)

db.add_batch(
    ["a", "b", "c", "d", "e"],
    [
        "Approximate nearest neighbor search with HNSW graphs",
        "BM25 ranking for exact keyword retrieval",
        "Reciprocal rank fusion combines ranked lists",
        "Quantized int8 vectors shrink memory four times",
        "Matryoshka embeddings truncate without retraining",
    ],
    metadatas=[
        {"topic": "ann", "year": 2025, "lang": "en", "tags": ["hnsw"]},
        {"topic": "bm25", "year": 2024, "lang": "en", "tags": ["lexical"]},
        {"topic": "fusion", "year": 2025, "lang": "de", "tags": ["rrf"]},
        {"topic": "quant", "year": 2023, "lang": "en", "tags": ["int8", "simd"]},
        {"topic": "mrl", "year": 2026, "lang": "en", "tags": ["mrl", "hnsw"]},
    ],
)

print("equality:", [r.id for r in db.search("search", top_k=3, filter={"topic": "ann"})])
print("in:", [r.id for r in db.search("search", top_k=5, filter={"topic": {"$in": ["ann", "fusion"]}})])
print("range:", [r.id for r in db.search("search", top_k=5, filter={"year": {"$gte": 2024}})])
print("exists:", [r.id for r in db.search("search", top_k=5, filter={"lang": {"$exists": True}})])
print("ne:", [r.id for r in db.search("search", top_k=5, filter={"lang": {"$ne": "en"}})])
print("list containment:", [r.id for r in db.search("graph", top_k=5, filter={"tags": "hnsw"})])
print(
    "or:",
    [
        r.id
        for r in db.search(
            "vectors",
            top_k=5,
            filter={"$or": [{"topic": {"$eq": "quant"}}, {"$and": [{"year": 2026}, {"tags": "mrl"}]}]},
        )
    ],
)
print("nor:", [r.id for r in db.search("search", top_k=5, filter={"$nor": [{"lang": "en"}]})])

try:
    db.search("x", top_k=1, filter={"y": {"$>": 3}})
except RuntimeError as e:
    print("rejected:", e)

print(db.info())
