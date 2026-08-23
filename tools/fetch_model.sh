#!/usr/bin/env bash
set -euo pipefail
base=https://huggingface.co/MongoDB/mdbr-leaf-ir/resolve/main
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/models/leaf-ir"
mkdir -p "$dir/onnx" "$dir/1_Pooling" "$dir/2_Dense"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json sentence_bert_config.json config_sentence_transformers.json 1_Pooling/config.json 2_Dense/config.json 2_Dense/model.safetensors; do
  curl -fSL "$base/$f" -o "$dir/$f"
done
curl -fSL "$base/onnx/model_quantized.onnx" -o "$dir/onnx/model_quantized.onnx"
curl -fSL "$base/onnx/model_quantized.onnx_data" -o "$dir/onnx/model_quantized.onnx_data"
echo "model ready at $dir"
