"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""Differential sanity check — PEFT silent-fail killer (memory feedback_lora_multimodal_loading.md).

For Qwen3.6-A3B (Qwen3_5MoeForConditionalGeneration multimodal arch),
LoRA adapter MUST be loaded via AutoModelForImageTextToText (not AutoModelForCausalLM,
which auto-strips language tower and silently produces logits_diff=0).

Test: 5 inputs × (adapter on / adapter off) compare logits mean-abs-diff.
Pass: diff > 0.01 on every prompt (LoRA actually fired).
Fail: diff <= 0.01 → PEFT silent fail or wrong loader/arch.

Usage:
    python3 differential_sanity.py \
        --base /root/autodl-tmp/models/Qwen3.6-35B-A3B-S1-S5v2-S4-BF16 \
        --adapter /root/autodl-tmp/adapters/Lynn-V4-Pro-r64 \
        --out-json /root/autodl-tmp/reports/diff_sanity_<tag>.json
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch


SANITY_PROMPTS = [
    "用一句话解释 Mixture-of-Experts active parameters。",
    "Python 写一个递归阶乘函数。",
    "比较 RoPE 与 ALiBi 的优缺点。",
    "今天北京适合穿什么?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
]

DIFF_THRESHOLD = 0.01
MAX_NEW_TOKENS = 8  # 只取 logits,不需要长生成


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BF16 base model dir (must match training base)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (adapter_model.safetensors)")
    ap.add_argument("--device-map", default="auto", help="device_map for from_pretrained")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out-json", default=None, help="optional JSON output path (consumed by four_gate_eval)")
    args = ap.parse_args()

    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    assert (base_path / "config.json").exists(), f"Base config not found: {base_path}"
    assert (adapter_path / "adapter_config.json").exists(), f"Adapter config not found: {adapter_path}"
    assert (adapter_path / "adapter_model.safetensors").exists(), f"Adapter weights not found"

    # === Step 1: load base via AutoModelForImageTextToText(memory feedback_lora_multimodal_loading.md)===
    print(f"[1/4] Loading base BF16 from {base_path}")
    print(f"      Using AutoModelForImageTextToText (Qwen3_5MoeForConditionalGeneration arch)")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    base = AutoModelForImageTextToText.from_pretrained(
        str(base_path),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    base.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    print(f"      base loaded, dtype={base.dtype}, device_map applied")

    # === Step 2: wrap with PEFT(adapter loaded but can toggle on/off)===
    print(f"[2/4] Loading LoRA adapter from {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(adapter_path), adapter_name="a0", is_trainable=False)
    model.set_adapter("a0")
    print(f"      adapter set: a0 active")

    # === Step 3: run 5 prompts × on/off compare logits ===
    print(f"[3/4] Running {len(SANITY_PROMPTS)} prompts × (adapter on / off) logits diff")
    results = []
    for i, prompt in enumerate(SANITY_PROMPTS, 1):
        # No-think chat template (production-aligned per memory feedback_eval_match_production_inference.md)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        with torch.inference_mode():
            # adapter ON
            model.set_adapter("a0")
            with torch.amp.autocast("cuda", dtype=dtype):
                logits_on = model(input_ids=input_ids).logits[0, -1, :].float()

            # adapter OFF(disable_adapter context)
            with model.disable_adapter():
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits_off = model(input_ids=input_ids).logits[0, -1, :].float()

        diff = (logits_on - logits_off).abs().mean().item()
        passed = diff > DIFF_THRESHOLD
        results.append({"idx": i, "prompt": prompt[:40], "diff": diff, "pass": passed})
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"      [{i}/5] diff={diff:.6f}  {status}  ({prompt[:40]!r})")

    # === Step 4: verdict ===
    print(f"\n[4/4] Verdict")
    all_pass = all(r["pass"] for r in results)
    n_pass = sum(1 for r in results if r["pass"])
    print(f"      pass: {n_pass}/{len(results)} (threshold diff > {DIFF_THRESHOLD})")

    # Emit JSON (consumed by four_gate_eval --sanity-json)
    if args.out_json:
        tz_cst = timezone(timedelta(hours=8))
        report = {
            "schema_version": "lynn-diff-sanity-v1",
            "created_at": datetime.now(tz_cst).isoformat(timespec="seconds"),
            "base": str(base_path),
            "adapter": str(adapter_path),
            "pass": all_pass,
            "n_pass": n_pass,
            "n_total": len(results),
            "diff_threshold": DIFF_THRESHOLD,
            "items": results,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"      JSON written: {out_path}")

    if all_pass:
        print(f"      ✅ DIFFERENTIAL SANITY PASS — LoRA confirmed active, base/arch correct")
        print(f"\n__DIFF_SANITY_PASS__")
        return 0
    else:
        fails = [r for r in results if not r["pass"]]
        print(f"      ❌ DIFFERENTIAL SANITY FAIL — {len(fails)} prompts with diff <= {DIFF_THRESHOLD}")
        print(f"      Likely causes:")
        print(f"        - AutoModelForCausalLM used instead of AutoModelForImageTextToText (silent strip语言塔)")
        print(f"        - LoRA adapter base mismatch (S1-S5v2-S4 vs raw)")
        print(f"        - PEFT adapter not actually applied (config / target_modules)")
        print(f"      → ABORT, do not proceed to 4-gate eval / ship")
        print(f"\n__DIFF_SANITY_FAIL__")
        return 1


if __name__ == "__main__":
    sys.exit(main())
