#!/usr/bin/env bash
set -euo pipefail
ver=1.23.0
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$root/target/ort-dl"
mkdir -p "$tmp" "$root/vendor/onnxruntime"
curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ver}/onnxruntime-win-x64-${ver}.zip" -o "$tmp/ort.zip"
powershell -NoProfile -Command "Expand-Archive -Path '$(cygpath -w "$tmp/ort.zip")' -DestinationPath '$(cygpath -w "$tmp")' -Force"
cp "$tmp/onnxruntime-win-x64-${ver}/lib/onnxruntime.dll" "$root/vendor/onnxruntime/onnxruntime.dll"
echo "onnxruntime ${ver} ready at $root/vendor/onnxruntime"
