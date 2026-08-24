import pathlib
import sys
import urllib.request

BASE = "https://huggingface.co/MongoDB/mdbr-leaf-ir/resolve/main"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "onnx/model_quantized.onnx",
    "onnx/model_quantized.onnx_data",
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    dst = root / "models" / "leaf-ir"
    for rel in FILES:
        out = dst / rel
        if out.exists():
            print(f"skip {rel} (present)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel}"
        print(f"fetch {url}")
        urllib.request.urlretrieve(url, out)
    print(f"model ready at {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
