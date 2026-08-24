# Publishing

Maintainer guide for releasing `swiftvec` (PyPI) and `swiftvec-core` (crates.io). Three paths, in order of preference: tag-driven CI, `tools/publish_pypi.py`, or fully manual maturin/uv commands. Releases are wheels-only, see the note under Automated release.

## Versioning

Repository rule: **public API changes and on-disk format changes require a version bump.** Bump once in the workspace root:

```
Cargo.toml            [workspace.package] version = "X.Y.Z"
crates/python/pyproject.toml   version = "X.Y.Z"
```

The Python package version must match the workspace version; maturin reads it from `pyproject.toml` at build time.

## Automated release (preferred)

Push a tag and GitHub Actions does the rest (`.github/workflows/release.yml`):

```bash
git tag v0.2.0
git push origin v0.2.0
```

The workflow builds abi3 wheels (Python 3.9-3.13) for:

- Linux x86_64 + aarch64 (manylinux)
- macOS x86_64 + arm64
- Windows x86_64

and publishes them to PyPI via **trusted publishing**. The release is intentionally **wheels-only**: the workspace uses path dependencies between crates (`crates/python` → `../core`, `../embed`), which maturin cannot package into a buildable sdist. Source installation is documented via `git clone` + `uvx maturin build` instead.

One-time repository setup:

1. Create a PyPI *trusted publisher* for this project: publisher name `github`, owner/repository `pavanteja/swiftvec`, workflow name `release.yml`, environment `pypi`.
2. Create a GitHub environment named `pypi` (require reviewers if you want an approval gate).

No tokens are stored in the repository.

## One-command local release

`tools/publish_pypi.py` validates versions, builds the wheel with maturin, and uploads it with uv:

```bash
export UV_PUBLISH_TOKEN="pypi-..."        # from pypi.org -> account -> API tokens
python tools/publish_pypi.py
```

The script refuses to run when the workspace `Cargo.toml` version and `crates/python/pyproject.toml` version disagree, cleans `dist/`, and prints every artifact it uploads.

## Manual release

```bash
uvx maturin build --release -m crates/python/Cargo.toml -o dist
uv publish dist/*
```

For other platforms either run the same command there or use maturin's cross images:

```bash
uvx maturin build --release -m crates/python/Cargo.toml -o dist --target aarch64-unknown-linux-gnu   # inside manylinux cross env
```

Manual upload with a token:

```bash
uv publish dist/* --token "$UV_PUBLISH_TOKEN"
```

## Pre-flight checklist

1. `cargo test --release` passes on all matrix OSes (CI enforces).
2. Version bumped in both places listed above.
3. `README.md` benchmark tables refreshed if numbers changed (`bash benchmarks/run.sh`).
4. Roadmap items ticked/unticked honestly.
5. `crates/python/README.md` still accurate (it is the PyPI landing page).
6. Tag follows `vX.Y.Z`.

## crates.io

Publish the core crate after the Python release when its API changed:

```bash
cd crates/core && cargo publish
```

`swiftvec-embed`, `swiftvec-python`, and `swiftvec-cli` are path-dependents of the workspace and are published only if needed standalone.

## Post-release

- Verify `pip install swiftvec==X.Y.Z` resolves on a clean machine.
- Check the PyPI page renders (description comes from `crates/python/README.md`).
- Attach release notes to the GitHub tag.
