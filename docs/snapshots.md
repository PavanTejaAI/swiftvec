# Snapshot format

A snapshot is a single file containing the HNSW index, document ids, metadata, and texts. Everything is little-endian. Databases loaded from snapshots remain mutable.

## Versions

| version | writer | reader |
|---|---|---|
| 1 | swiftvec ≤ 0.2 | all versions |
| 2 | swiftvec ≥ 0.3 (framed sections, cascade block) | all versions |

Readers reject unknown versions with a clear error; the container never changes silently.

## Container layout (Python SDK, `.swiftvec` files)

```
u64  core_len
[u8 × core_len]        HNSW core snapshot (spec below)
u64  ids_len
[u8 × ids_len]         JSON array of string ids
u64  metas_len
[u8 × metas_len]       JSON array of metadata values (may contain null)
[u8 × rest]            JSON array of document texts (no length prefix; extends to EOF)
```

Row `i` across the three arrays corresponds to graph node id `i`.

## HNSW core format (`swiftvec-core`, magic + version)

```
u32  magic = 0x5357_5643 ("SWVC")
u32  version = 2
u32  dim
u32  m
u32  m0 (= 2m)
u32  ef_construction
u8   metric        (0 = dot, 1 = l2)
u8   storage       (0 = f32, 1 = int8)
u8   rerank_f32    (0/1)
u8   packed        (0/1)
f32  qrange
f32  qscale        ((qrange/127)²)
f32  mult          (1/ln(m))
u32  entry point
u32  max_level
u32  n             (node count; must be > 0 with dim > 0)

pad(8)                                   alignment frame: u32 pad_count + pad bytes
u32 × n            per-node level
pad(8)
u64  vectors_len
f32 × vectors_len  f32 vector store
pad(8)
u64  codes_len
i8  × codes_len    int8 codes

u8   cascade       (0/1, v2 only)
if cascade:
  u64  hyperplane seed
  pad(8)
  u64  sig_lanes (= 4n)
  u64 × sig_lanes    packed 256-bit signatures

if packed:
  u32 layers
  per layer:
    pad(8); u32 offsets_len; u32 × offsets_len
    pad(8); u32 targets_len; u32 × targets_len
else:
  u32 layers
  per layer:
    u32 nodes
    per node:
      u32 degree; u32 × degree
```

`pad(8)` frames appear only in v2 and only before bulk typed arrays; they guarantee every array starts at an 8-byte boundary so zero-copy readers can cast aligned slices. Version-1 files have no padding and no cascade block.

## mmap requirements

`MappedIndex::decode` accepts only **version 2 + `packed = 1`** snapshots. Upper graph layers are sparse, their CSR arrays cover exactly the nodes present on that layer, not all `n` nodes, so readers must bounds-check node ids against each layer's offsets length.

## Notes for tool authors

- The core snapshot carries an explicit version integer. Loaders reject unsupported versions.
- Any change to the public API or to either on-disk format requires a workspace version bump (repository rule in `AGENTS.md`).
- The container adds sections by appending new length-prefixed blocks before EOF; loaders of older versions read only what they know.

## Versioning policy

- The core snapshot carries an explicit version integer. Loaders accept 1 and 2 and reject anything newer.
- Any change to the public API or to either on-disk format requires a workspace version bump (repository rule in `AGENTS.md`).
- New bulk sections are appended as framed blocks so zero-copy alignment guarantees survive every future revision.

## Notes for tool authors

- All integers are LE; floats are IEEE-754 binary32.
- The BM25 postings are not serialized; they are rebuilt from `texts` on load, which keeps files smaller and guarantees the keyword index can never drift from the stored corpus.
- A snapshot saved after `pack()` records `packed = 1` and loads directly into serving mode; a pre-pack snapshot loads into mutable construction mode and can be packed later.
