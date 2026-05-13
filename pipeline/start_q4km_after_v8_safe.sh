#!/usr/bin/env bash
set -euo pipefail
LOG=/root/autodl-tmp/reports/q4km_build.log
SRC=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged
V8=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN
F16=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-F16.gguf
Q4=/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-Q4_K_M.gguf
LLAMA=/root/autodl-tmp/llama.cpp
mkdir -p /root/autodl-tmp/reports
exec > >(tee -a "$LOG") 2>&1

ts(){ date "+%F %T"; }
log(){ echo "[$(ts)] $*"; }

log "start Q4_K_M build"
if [[ ! -s "$SRC/config.json" ]]; then
  log "missing BF16 source: $SRC"
  exit 10
fi

log "delete local v8 to free disk: $V8"
rm -rf "$V8"
df -h /root/autodl-tmp

log "convert HF BF16 -> F16 GGUF"
rm -f "$F16.tmp" "$Q4.tmp"
/root/autodl-tmp/conda-envs/r6000-eval/bin/python "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outfile "$F16.tmp" --outtype f16
mv "$F16.tmp" "$F16"
ls -lh "$F16"
df -h /root/autodl-tmp

log "quantize F16 GGUF -> Q4_K_M"
"$LLAMA/build/bin/llama-quantize" "$F16" "$Q4.tmp" Q4_K_M
mv "$Q4.tmp" "$Q4"
ls -lh "$Q4"

log "remove F16 intermediate"
rm -f "$F16"
df -h /root/autodl-tmp

log "Q4 smoke"
"$LLAMA/build/bin/llama-cli" -m "$Q4" -p "请用一句话介绍 Lynn-V4-Pro-Distill。" -n 80 --temp 0 --no-display-prompt | tee /root/autodl-tmp/reports/q4km_smoke.txt
log "__Q4KM_BUILD_DONE__"
