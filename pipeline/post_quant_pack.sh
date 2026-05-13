#!/usr/bin/env bash
#
# Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit
#
# Ship gate wrapper for quantization output.
#
# Purpose:
#   Quantization toolkits (e.g., qwen3.6-nvfp4-toolkit v8-RTN llmcompressor) often produce
#   incomplete output directories missing tokenizer.json, tokenizer_config.json, and other
#   files required by loaders. This silently breaks downstream serving — AutoProcessor /
#   AutoTokenizer fail at runtime, users can't load the model.
#
#   This wrapper:
#     1. Copies missing files from BF16 source dir to quantization output dir
#     2. Runs 3 sanity gates before allowing publication
#     3. Exits 1 if any gate fails (ship blocked)
#
# Usage:
#   ./post_quant_pack.sh <bf16_source_dir> <quant_output_dir>
#
# Example:
#   ./post_quant_pack.sh \
#     /path/to/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged \
#     /path/to/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN
#

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <bf16_source_dir> <quant_output_dir>"
    exit 1
fi

BF16="$1"
OUT="$2"

if [ ! -d "$BF16" ]; then
    echo "❌ BF16 source dir does not exist: $BF16"
    exit 1
fi
if [ ! -d "$OUT" ]; then
    echo "❌ Quant output dir does not exist: $OUT"
    exit 1
fi

echo "=== Post-quant packaging ==="
echo "  BF16 source : $BF16"
echo "  Quant out   : $OUT"
echo

# Files that quantization toolkits often forget to copy
SHIP_FILES=(
    "tokenizer.json"
    "tokenizer_config.json"
    "chat_template.jinja"
    "generation_config.json"
    "model.safetensors.index.json"
    "special_tokens_map.json"
)

for f in "${SHIP_FILES[@]}"; do
    if [ -f "$BF16/$f" ] && [ ! -f "$OUT/$f" ]; then
        cp -v "$BF16/$f" "$OUT/$f"
    elif [ ! -f "$BF16/$f" ]; then
        echo "  ⊘ skip $f (not in BF16 source)"
    fi
done

echo
echo "=== Final ship package ==="
ls -la "$OUT/"

echo
echo "=== Sanity Gates ==="

# Gate 1: required file presence
REQUIRED=(
    "tokenizer.json"
    "tokenizer_config.json"
    "config.json"
    "model.safetensors"
    "generation_config.json"
    "chat_template.jinja"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$OUT/$f" ]; then
        echo "❌ Gate 1 FAIL: missing required file: $f"
        echo "   Ship blocked. Either copy this file manually from BF16 source, or"
        echo "   skip it if your model architecture doesn't use it (rare)."
        exit 1
    fi
done
echo "✓ Gate 1: required files present"

# Gate 2: tokenizer loadable
if ! python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$OUT', trust_remote_code=True)
ids = t.encode('hello world')
assert len(ids) > 0, 'tokenizer encoded empty'
" 2>/dev/null; then
    echo "❌ Gate 2 FAIL: tokenizer.from_pretrained failed"
    echo "   Inspect tokenizer.json + tokenizer_config.json + special_tokens_map.json."
    exit 1
fi
echo "✓ Gate 2: tokenizer loadable"

# Gate 3: index consistency (if present)
if [ -f "$OUT/model.safetensors.index.json" ]; then
    if ! python3 -c "
import json
idx = json.load(open('$OUT/model.safetensors.index.json'))
weight_map = idx.get('weight_map', {})
import os
shards_in_index = set(weight_map.values())
files_present = set(f for f in os.listdir('$OUT') if f.startswith('model-') and f.endswith('.safetensors'))
missing = shards_in_index - files_present
if missing:
    print(f'index references missing shards: {missing}')
    exit(1)
" 2>/dev/null; then
        echo "❌ Gate 3 FAIL: index.json references missing shard files"
        exit 1
    fi
    echo "✓ Gate 3: index ↔ shards consistent"
else
    echo "✓ Gate 3: single-file model (no index needed)"
fi

echo
echo "✅ All 3 gates PASSED — ship clear to proceed"
