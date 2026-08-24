# Installation

## Python package (PyPI)

Requires Python 3.9+.

```bash
uv pip install swiftvec
```

The wheel ships the compiled engine and embedding runtime. It does not ship model weights; fetch them once (about 130 MB) and everything afterwards runs fully offline:

```bash
git clone https://github.com/PavanTejaAI/swiftvec && cd swiftvec
bash tools/fetch_model.sh
```

On systems without bash (for example native Windows), use the Python fetcher:

```bash
python tools/fetch_model.py
```

Both scripts populate `models/leaf-ir/` in the repository checkout. Point `SwiftVec` at that directory:

```python
from swiftvec import SwiftVec

db = SwiftVec(model_dir="models/leaf-ir")
```

If you keep the model elsewhere, pass any directory that follows the [mdbr-leaf-ir layout](embeddings.md#model-files).

## ONNX Runtime

**Wheels bundle it, no setup needed.** Every wheel published to PyPI carries the platform-matching ONNX Runtime 1.23.0 shared library (MIT, © Microsoft) inside the package. On import, swiftvec points ONNX Runtime at the bundled library automatically. `uv pip install swiftvec` is genuinely the only step.

Resolution order at load time:

1. the `ORT_DYLIB_PATH` environment variable, if you set it (custom builds, newer runtimes);
2. `onnxruntime.dll` in the current working directory;
3. `vendor/onnxruntime/onnxruntime.dll` relative to the current working directory.

For pip installs, path 1 is prefilled with the bundled library unless you override it.

### Source builds

Source checkouts do not bundle a runtime; fetch one:

```bash
bash tools/fetch_ort.sh
```

downloads ONNX Runtime 1.23.0 for Windows x64 into `vendor/onnxruntime/`. For other platforms, either install the `onnxruntime` pip package and set `ORT_DYLIB_PATH` into its `capi/` folder, or download a release from [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases) and point `ORT_DYLIB_PATH` at its shared library (`onnxruntime.dll`, `libonnxruntime.so`, `libonnxruntime.dylib`).

## Rust crate

```bash
cargo add swiftvec-core
```

`swiftvec-core` has zero dependencies. The embedding stack is a separate crate (`swiftvec-embed`); depend on it only if you embed from Rust.

## Building from source

Requirements: a stable Rust toolchain, Python 3.9+, and [uv](https://docs.astral.sh/uv/) for Python tooling.

```bash
git clone https://github.com/PavanTejaAI/swiftvec && cd swiftvec
cargo build --release
cargo test --release
```

Build the Python wheel with maturin:

```bash
uvx maturin build --release -m crates/python/Cargo.toml -o dist
```

## Verifying the install

Python (pip or source build):

```bash
python examples/basic.py
```

Runs indexing, snapshot save/load, semantic search, hybrid search, and filtered search end to end; the model directory must be reachable (see above).

Rust CLI (source builds only):

```bash
swiftvec embed --text "hello world"
```

Prints per-text dimensions, norms, and inference latency. The binary resolves ONNX Runtime from the current directory or `vendor/onnxruntime/`, and the model from `models/leaf-ir`, so run it from a repository checkout that has been set up with `tools/fetch_ort.sh` and `tools/fetch_model.sh`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `run tools/fetch_model.sh` error on load | model files missing under `models/leaf-ir/onnx/` | run `tools/fetch_model.sh` or `tools/fetch_model.py` |
| load-time panic mentioning ORT / dylib | ONNX Runtime library not found | set `ORT_DYLIB_PATH` as shown above |
| `dim must be in 1..=768` | requested truncation larger than the model output or zero | use 1-768 |
| slow first query | session warm-up happens at load | ignore it; benchmarks report warm numbers separately |
