# swiftvec rules

- rust workspace; `swiftvec-core` carries zero runtime dependencies
- no comments, no doc comments, no doc strings
- optimized-first: SoA layouts, cache-aware access, zero-copy, runtime SIMD dispatch, deterministic seeds
- SOLID via small traits and modules; no god objects
- no mocks ever: correctness is proven against the brute-force oracle in this repo; every benchmark reproduces from a fixed seed
- public API and on-disk format changes require a version bump
- python tooling uses uv, never pip
- conventional commits
