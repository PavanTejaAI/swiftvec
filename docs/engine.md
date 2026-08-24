# Engine internals

`swiftvec-core` implements the entire retrieval engine with zero runtime dependencies. This document explains how each layer works and why it is shaped the way it is.

## HNSW graph

The index follows Malkov & Yashunin's Hierarchical Navigable Small World graphs:

- every node draws an integer level from an exponential distribution `floor(-ln(u) * mult)` where `mult = 1/ln(m)`; level 0 contains all nodes
- insertion descends greedily from the entry point through layers above the new node's level, then runs beam search (`ef_construction`) on each layer at or below, selecting neighbors with **Algorithm 4** (the heuristic diversity check: a candidate is kept only if it is closer to the new point than to every already-selected neighbor; pruned candidates backfill any remaining capacity)
- links are bidirectional; when a node exceeds its degree cap (`m0 = 2m` at level 0, `m` above), its neighbor list shrinks by re-running Algorithm 4 over the current candidates
- search descends greedily to layer 1, then runs beam search with user-controlled `ef` at layer 0

Determinism: level sampling uses a Xoshiro256** generator seeded from SplitMix64 with a fixed seed (`0x5EED_5EED`), so identical insert sequences produce byte-identical graphs, verified by a dedicated test.

## Storage tiers and the rerank cascade

| tier | stored per vector | traversal | recall recovery |
|---|---|---|---|
| f32 | 768 × f32 (3 KB @768d) | exact distances | n/a |
| int8 | dim × i8 (4x smaller) | quantized distances | optional f32 rerank |
| int8 + MRL | 256 × i8 (12x smaller than f32-768) | quantized distances | f32 rerank of the truncated vectors |

Quantization is symmetric per-dimension-range: codes are `round(x · 127/range)` clamped to ±127. Distances during traversal are integer dot products rescaled by `(range/127)²`. The query vector is quantized once per search into a reusable buffer.

With `rerank_f32` enabled, the original vectors are also retained; after beam search returns `ef` candidates, each candidate's exact full-precision distance replaces the quantized one and the list is re-sorted before truncation to `k`. Traversal stays cheap (int8 kernels); ordering precision is recovered exactly where it matters.

## CSR-packed serving graph

During construction the adjacency lives in `Vec<Vec<Vec<u32>>>` (layer → node → targets), which is convenient for mutation. `pack()` converts this to flat arrays per layer, `offsets[n+1]` + `targets[nnz]`, eliminating pointer chasing on every hop. Packing is terminal: further `add()` calls assert. Search performance in the committed benchmarks is measured on packed indexes; save/load preserves the packed state, so a snapshot taken after `pack()` loads ready-to-serve.

## SIMD kernels

All distance computations dispatch at runtime via `is_x86_feature_detected!`:

- `dot_f32` / `l2sq_f32`: AVX2+FMA intrinsics with four unrolled accumulators over 32-element blocks, falling back to an 8-lane scalar loop that auto-vectorizes, then a tail loop
- `dot_i8`: AVX2 path widens 16-byte loads to 16-bit lanes (`_mm256_cvtepi8_epi16`) and accumulates with `_mm256_madd_epi16` into i32 lanes; i32 accumulation is exact for 768 dimensions well within range
- neighbor rows are prefetched (`_mm_prefetch`, T0 hint) one iteration ahead in both greedy descent and beam search

Correctness of every kernel is asserted against its scalar fallback across odd lengths (1, 7, 16, 17, 31, …) in unit tests, and end-to-end recall is asserted against the brute-force oracle in integration tests.

## Beam search mechanics

- visited-set bookkeeping uses a generation-stamp array (`Visited`): marking a node writes the current generation; `reset()` just increments it, so clearing between queries costs O(1). The stamp generation wraps safely at u32::MAX.
- the frontier is a min-heap keyed by distance; the result set a bounded max-heap of size `ef`. A candidate enters the frontier if it is unvisited and either improves the worst result or the result set is not yet full. Termination occurs when the best frontier distance exceeds the worst result distance.
- filtered searches pass an `Option<&dyn Fn(u32) -> bool>` predicate: non-passing nodes are traversed (to preserve connectivity) but never admitted to the result set. The Python SDK compiles filter dicts into such predicates, see [filters](filters.md).

## BM25 and fusion

The keyword index is classic BM25 (`k1 = 1.2`, `b = 0.75`) over ASCII-alphanumeric tokens, with postings lists `term → [(doc, tf)]` and a bounded heap top-k. Hybrid ranking fuses the two ranked lists with weighted Reciprocal Rank Fusion (`k = 60`):

```
score(d) = w_vector / (60 + rank_vector(d)) + w_text / (60 + rank_text(d))
```

Details and tuning guidance live in [hybrid-search](hybrid-search.md).

## Snapshots

Single-file little-endian format with magic `0x5357_5643` ("SWVC"). Version 2 frames every bulk section with an 8-byte alignment pad (`u32` pad length + zero bytes) so zero-copy readers can cast typed slices safely, and appends the cascade block (flag, hyperplane seed, packed signatures) when enabled. Readers accept version 1 (legacy, unframed, no cascade) and version 2. Full byte-level spec: [snapshots](snapshots.md).

## mmap zero-copy serving

With the default-off `mmap` cargo feature, a packed v2 snapshot can be served straight out of the page cache:

```rust
use swiftvec_core::{Mapping, MappedIndex};

let mapping = Mapping::open("index.swiftvec")?;
let index = MappedIndex::decode(mapping.data())?;
let hits = index.search(&query, 10, 128, None);
```

`Mapping::open` maps the file; `MappedIndex::decode` validates the header and CSR sections and hands out slices over the mapped bytes, levels, vectors, codes, signatures, and graph targets are never copied onto the heap. Load time drops to mmap+validation cost regardless of index size, and the OS shares resident pages across processes. Constraints: the snapshot must be written after `pack()` in v2 format, and big-endian hosts are rejected at decode.

## Binary cascade

The optional cascade adds a first pass of cheap bit tests before int8 distance work:

- on insertion, each vector gets a 256-bit signature: the signs of its dot products with 256 fixed random hyperplanes (seeded deterministically from the config seed, builds stay reproducible)
- during greedy descent and beam search, a neighbor is evaluated only if its Hamming distance to the query signature is ≤ `CASCADE_TAU` (128 bits ≈ angular distance ≈ 90°); farther nodes are marked visited and skipped without any int8 dot
- signatures are stored in the snapshot (32 bytes per node), so saved and reloaded indexes cascade identically

Measured trade-offs (50k clustered / 20k uniform synthetic, int8+rerank):

| scenario | effect |
|---|---|
| clustered, ef=256 | p99 1167 µs → **549 µs**, recall identical |
| clustered, ef=64-128 | ~neutral (baseline recall already 1.0) |
| uniform dim=64 | slightly slower, an int8 dot at low dim is cheaper than the signature check |

Guidance: enable for high-dimensional indexes or when chasing tail latency at large `ef`; leave off for small dims. It is opt-in (`HnswConfig::cascade`, CLI `--cascade`) precisely because it is not a universal win.

## Complexity

| operation | cost |
|---|---|
| add | O(ef_construction · m · log n)-ish graph walk per level |
| search | O(ef · m · d) distance work dominated by kernel throughput |
| memory (int8+MRL+rerank) | dim bytes code + 256×4 B rerank vector per doc, plus ~8·m·4 B links |
