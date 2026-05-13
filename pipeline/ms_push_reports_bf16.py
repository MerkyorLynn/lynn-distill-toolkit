"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""
Push reports/ subdirectory to MS BF16 repo (additional commit after main BF16 push).
Run on R6000 after PID 216524 (BF16 main push) exits successfully.

Reports published:
- 4gate_results.json (lynn-4gate-v1)
- diff_sanity_lynn-v4-pro-r64.json (lynn-diff-sanity-v1)
- eval_summary.md

Token: hardcoded from memory reference_modelscope_token.md (ms-uuid 39-char).
"""

import os
import sys
import shutil
from pathlib import Path

# ms-uuid 39-char SDK token (memory reference_modelscope_token.md)
MS_TOKEN = os.environ.get("MS_TOKEN") or sys.exit("Set MS_TOKEN env (format: ms-<uuid>)")
REPO_ID = "Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B"

REPORTS_SRC = Path("/root/autodl-tmp/reports")
STAGE_DIR = Path("/tmp/ms_reports_bf16_stage")

REPORT_FILES = [
    "4gate_results.json",
    "diff_sanity_lynn-v4-pro-r64.json",
    "eval_summary.md",
]


def stage_reports():
    """Copy report files to clean staging dir (avoid pushing extra junk from reports/)."""
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    for fname in REPORT_FILES:
        src = REPORTS_SRC / fname
        if not src.exists():
            raise SystemExit(f"Missing report: {src}")
        shutil.copy(src, STAGE_DIR / fname)
    print(f"[stage] Staged {len(REPORT_FILES)} reports in {STAGE_DIR}")
    for f in sorted(STAGE_DIR.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")


def push_to_ms():
    """Upload reports/ subdirectory to MS BF16 repo."""
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(MS_TOKEN)

    print(f"[MS] Pushing {STAGE_DIR}/ → {REPO_ID}/reports/")
    result = api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(STAGE_DIR),
        path_in_repo="reports",
        commit_message="Publish 4-gate ship eval evidence (NET_WIN +40.00pp)",
        repo_type="model",
    )
    print(f"[MS] Result: {result}")
    return result


def main():
    stage_reports()
    push_to_ms()
    print("[done] reports/ pushed to MS BF16 repo")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
