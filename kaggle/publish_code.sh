#!/usr/bin/env bash
# Build the anode package as a wheel and upload it to the private Kaggle
# dataset mhasbiarz/anode-src, which the benchmark notebooks install from.
set -euo pipefail
cd "$(dirname "$0")/.."
out=kaggle/_code
rm -rf "$out" && mkdir -p "$out"
uv build --wheel -q -o "$out"
cat > "$out/dataset-metadata.json" <<JSON
{"title": "anode-src", "id": "mhasbiarz/anode-src", "licenses": [{"name": "MIT"}]}
JSON
if kaggle datasets status mhasbiarz/anode-src >/dev/null 2>&1; then
  kaggle datasets version -p "$out" -m "${1:-update}" -q
else
  kaggle datasets create -p "$out" -q
fi
