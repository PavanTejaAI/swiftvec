# Metadata filters

Filters are evaluated at query time against the metadata stored with each document. The predicate is compiled once per query (invalid operators and malformed expressions fail immediately with a descriptive error), then pushed into graph traversal, nodes that cannot pass the filter are still traversed for connectivity, but only passing nodes are admitted as results.

## Grammar

A filter is a dict. Each key is either a **field name** or a **combinator**:

```python
{ "field": condition, ... }                  implicit $eq, fields AND-combined
{ "$and": [filter, ...] }                    all sub-filters must match
{ "$or":  [filter, ...] }                    at least one sub-filter must match
{ "$nor": [filter, ...] }                    no sub-filter may match
```

A **condition** is either a literal value (implicit equality) or an operator dict:

```python
{"topic": "ir"}                              topic == "ir"
{"topic": {"$in": ["ir", "rag"]}}            topic is one of the values
{"year": {"$gte": 2024}}                     numeric range
```

## Operators

| operator | meaning | notes |
|---|---|---|
| `$eq` | field == value | implicit when the condition is not an operator dict |
| `$ne` | field != value | missing fields match |
| `$gt` / `$gte` | greater than / >= | numbers compare numerically, strings lexicographically; mixed types do not match |
| `$lt` / `$lte` | less than / <= | same typing rules |
| `$in` | field in value list | value list required to be a `list` |
| `$nin` | field not in value list | missing fields match |
| `$exists` | field present (`true`) or absent (`false`) | boolean required |

Semantics worth knowing:

- **Numbers compare across int/float**: `{"n": 1}` matches stored `1`, `1.0`.
- **List-typed metadata fields** match by containment: `{"tags": "rust"}` matches `["rust", "ai"]`.
- **Missing fields**: comparisons (`$eq`, ranges, `$in`) never match; negations (`$ne`, `$nin`) match, mirroring MongoDB semantics.
- An empty filter `{}` matches everything.
- Multiple fields in one filter are AND-combined: `{"topic": "ir", "year": {"$gte": 2024}}`.

## Examples

```python
from swiftvec import SwiftVec

db = SwiftVec()
db.add_batch(
    ["a", "b", "c", "d"],
    ["doc a", "doc b", "doc c", "doc d"],
    metadatas=[
        {"topic": "ir",    "lang": "en", "year": 2025, "tags": ["ann"]},
        {"topic": "rag",   "lang": "en", "year": 2024, "tags": ["llm"]},
        {"topic": "bio",   "lang": "de", "year": 2023},
        {"topic": "ir",    "lang": "en", "year": 2024, "tags": ["ann", "simd"]},
    ],
)

db.search("query", top_k=3)
# pure semantic over everything

db.search("query", top_k=3, filter={"topic": "ir"})
# only topic == "ir"

db.search("query", top_k=3, filter={"topic": {"$in": ["ir", "rag"]}, "lang": "en"})
# both conditions must hold

db.search("query", top_k=3, filter={"year": {"$gte": 2024}})
# range

db.search("query", top_k=3, filter={"tags": "simd"})
# containment inside a list field

db.search("query", top_k=3, filter={
    "$or": [
        {"topic": "bio"},
        {"$and": [{"topic": "ir"}, {"year": {"$gte": 2025}}]},
    ]
})
# nested combinators

db.search("hybrid keywords alpha", top_k=3, alpha=0.4, filter={"year": 2025})
# filters apply to both the vector and the keyword candidate lists before fusion
```

## Error cases

All of these raise at call time, before any embedding work happens:

```python
db.search("q", filter="topic == 'ir'")          # not a dict
db.search("q", filter={"year": {"$>": 1}})      # unknown operator
db.search("q", filter={"$and": []})             # empty combinator
db.search("q", filter={"tags": {"$in": "ann"}}) # $in requires a list
db.search("q", filter={"x": {"$exists": 1}})    # $exists requires a bool
```

`db.info()["filter_ops"]` lists the supported operators at runtime.
