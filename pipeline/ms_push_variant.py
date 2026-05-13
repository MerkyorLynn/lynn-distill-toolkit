"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""Push one Lynn-V4-Pro-Distill-Qwen-35B-A3B variant to ModelScope Merkyor namespace.
Token via MS_TOKEN env var. Variant -> repo mapping:
  BF16-merged     -> Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B
  FP8             -> Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-FP8
  NVFP4-modelopt  -> Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-modelopt
  NVFP4-v8-RTN    -> Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN

Usage:
    MS_TOKEN='ms-xxx' python3 ms_push_variant.py --variant BF16-merged
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

VARIANT_TO_REPO = {
    "BF16-merged":     "Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B",
    "FP8":             "Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-FP8",
    "NVFP4-modelopt":  "Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-modelopt",
    "NVFP4-v8-RTN":    "Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN",
}

VARIANT_TO_SRC = {
    "BF16-merged":     "/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged",
    "FP8":             "/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-FP8",
    "NVFP4-modelopt":  "/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-modelopt",
    "NVFP4-v8-RTN":    "/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN",
}

CARDS_DIR = "/root/autodl-tmp/hf_staging/ms-Lynn-V4-Pro-Distill-Qwen-35B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_TO_REPO.keys()))
    args = ap.parse_args()

    token = os.environ.get("MS_TOKEN")
    if not token:
        sys.exit("ERROR: MS_TOKEN env var not set")

    repo_id = VARIANT_TO_REPO[args.variant]
    src = VARIANT_TO_SRC[args.variant]
    print(f"[variant] {args.variant}")
    print(f"[src] {src}")
    print(f"[dst] {repo_id}")

    api = HubApi()
    api.login(token)
    print(f"[login] OK")

    # Create repo
    for attempt in range(1, 6):
        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"[repo] ready (attempt {attempt})")
            break
        except Exception as e:
            print(f"[repo] attempt {attempt}/5 failed: {str(e)[:200]}")
            if attempt < 5:
                time.sleep(15 * attempt)
            else:
                sys.exit("create_repo failed")

    # Stage cards into model dir for unified upload (BF16-merged 主仓只)
    if args.variant == "BF16-merged":
        for fname in ["README.md", "LICENSE", "NOTICE"]:
            src_card = Path(CARDS_DIR) / fname
            dst_card = Path(src) / fname
            if src_card.exists():
                shutil.copy(src_card, dst_card)
                print(f"[stage] {fname} copied")

    # Upload folder (exclude .bak files for quants)
    total_bytes = sum(p.stat().st_size for p in Path(src).rglob("*")
                      if p.is_file() and not p.name.endswith(".bak")
                      and not p.name.endswith(".bak.causal"))
    total_gb = total_bytes / (1024**3)
    print(f"[upload] {args.variant} {total_gb:.1f} GB → {repo_id}")
    t0 = time.time()

    for attempt in range(1, 4):
        try:
            api.upload_folder(
                folder_path=src,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Initial upload: {args.variant} ({total_gb:.1f} GB)",
                ignore_patterns=["*.bak", "*.bak.*", "*.causal.bak", "model.safetensors.causal.bak",
                                 "config.json.causal.bak", "__pycache__", "*.pyc", ".git*", "*.tmp"],
            )
            break
        except Exception as e:
            print(f"[upload] attempt {attempt}/3 failed: {str(e)[:300]}")
            if attempt < 3:
                time.sleep(60 * attempt)
            else:
                sys.exit(f"upload {args.variant} failed")

    dt = time.time() - t0
    print(f"[done] {args.variant} uploaded {total_gb:.1f} GB in {dt/60:.1f} min "
          f"(avg {total_bytes/dt/1e6:.1f} MB/s)")
    print(f"[link] https://modelscope.cn/models/{repo_id}")


if __name__ == "__main__":
    sys.exit(main())
