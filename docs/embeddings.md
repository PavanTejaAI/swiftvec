# Embeddings

swiftvec embeds text entirely on your machine. There is no API call anywhere in the pipeline.

## Model

| property | value |
|---|---|
| model | [MongoDB/mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) |
| license | Apache-2.0 (© MongoDB) |
| architecture | distilled BERT-style encoder with a dense projection head |
| output | 768-d, L2-normalized, Matryoshka-capable (MRL) |
| runtime | quantized ONNX variant (`onnx/model_quantized.onnx`) via ONNX Runtime, loaded dynamically |

Acknowledgement and citation guidance for the model live in the root README. The Matryoshka property is what enables the fast 256-dimension tier: the first 256 output dimensions are trained to be independently useful, so truncating after pooling costs little recall and saves 3x memory plus 3x kernel work.

## Pipeline

1. **prompt**: query texts are prefixed with the model's query prompt (read from `config_sentence_transformers.json`, default `"Represent this sentence for searching relevant passages: "`); documents are embedded unprompted.
2. **tokenize**: HuggingFace `tokenizers` from `tokenizer.json`, batch-truncated to `max_seq_length` (from `sentence_bert_config.json`, default 512), padded to the batch's longest sequence.
3. **infer**: one ONNX session run per batch (`input_ids`, `attention_mask`, `token_type_ids`); graph optimization level 3; intra-op threads configurable, defaults to `min(4, available_parallelism)`.
4. **pool**: mean pooling over unmasked tokens.
5. **project**: dense linear layer `W·pooled + b`, weights parsed directly from `2_Dense/model.safetensors`.
6. **normalize**: L2 normalization.
7. **truncate**: optional MRL truncation to `dim` when `SwiftVec(dim=...)` requests it; queries and documents truncate identically, keeping the two spaces aligned.

The session is warmed with a two-token dummy input at load time so first-query latency does not pay graph-optimization costs.

## Model files

`tools/fetch_model.sh` (bash) and `tools/fetch_model.py` (pure Python, any platform) download the following layout from Hugging Face into `models/leaf-ir/`:

```
models/leaf-ir/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  sentence_bert_config.json
  config_sentence_transformers.json
  1_Pooling/config.json
  2_Dense/config.json
  2_Dense/model.safetensors
  onnx/model_quantized.onnx
  onnx/model_quantized.onnx_data
```

`Embedder::load` validates shape compatibility (`hidden_size` vs `in_features`) at load time and fails with a descriptive error rather than producing garbage embeddings.

## Throughput expectations

Measured during the benchmark runs committed in `benchmarks/results/`: roughly 470-540 docs/s on a laptop CPU with 4 embedding threads at 768 dims, batch size 64, about 2.1 ms of end-to-end query cost attributable to embedding. See [benchmarks](benchmarks.md) for the full protocol.

## Practical notes

- Batch large ingests through `add_batch`, batching amortizes padding and session overhead.
- `dim=256` cuts kernel cost ~3x; combine with int8 storage + rerank for the best latency/recall trade-off measured in this repo.
- Embedding is deterministic for fixed inputs; index builds are deterministic given insertion order (fixed RNG seed), so snapshots are reproducible.
