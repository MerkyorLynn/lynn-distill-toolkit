"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""PEFT merge — runs on R6000(单卡 RTX PRO 6000 96GB)to merge BF16 base + LoRA adapter.

替代 A100 stage_merge:A100 outbound 2 MB/s 限速,67G merged BF16 传输 9h/variant 太慢。
S1-S5v2-S4 base 已 rsync 到 R6000(5/12 night),只需传 85MB adapter,merge 本地做。

Memory lessons applied:
- `feedback_lora_multimodal_loading.md`:Qwen3.6-A3B 是 Qwen3_5MoeForConditionalGeneration 多模态架构,
  必须用 AutoModelForImageTextToText(不能用 AutoModelForCausalLM,会静默剥语言塔)
- `feedback_qwen36_raw_bf16_per_expert_silent_garbage.md`(trap 15):per-expert layout 静默 random init,
  必须 base-only generate coherence check(majority script ratio > 0.6)
- S1-S5v2-S4 base 已验证 FUSED layout(safe to load)

Usage:
    python peft_merge.py \\
        --base /root/autodl-tmp/models/Qwen3.6-35B-A3B-S1-S5v2-S4 \\
        --adapter /root/autodl-tmp/adapters/V_PRO \\
        --output /root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged

ETA: ~30 min on single R6000 96GB(load 5min + merge 20min + save 5min)
"""
import argparse
import os
import re
import shutil
import sys

import torch


def check_base_coherence(model, tokenizer, label: str = "base"):
    """Trap 15 protection: verify base experts are functional via 5-token generate."""
    print(f"\n[coherence check] {label} base-only generate sanity")
    test_prompts = [
        "用一句话解释 Mixture-of-Experts active parameters。",
        "Python 写一个递归阶乘函数。",
    ]
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # Script coherence: ASCII or CJK combined (mixed bilingual safe).
        # Trap 15 garbage typically includes Arabic/symbols outside both classes,
        # which lowers combined ratio — protection preserved.
        # Old "max(ascii, cjk) / total" false-positives on coherent zh-en mixed output.
        ascii_chars = len(re.findall(r"[A-Za-z0-9 ]", gen))
        cjk_chars = len(re.findall(r"[㐀-鿿]", gen))
        punct_chars = len(re.findall(r"[,。.，、:!?;:!?\-—()()\"\"''「」《》<>]", gen))
        total = max(len(gen.strip()), 1)
        script_chars = ascii_chars + cjk_chars + punct_chars
        majority_ratio = script_chars / total
        print(f"    prompt: {prompt[:40]!r}")
        print(f"    gen:    {gen[:80]!r}")
        print(f"    majority_ratio (ascii+cjk+punct): {majority_ratio:.2f}")
        if majority_ratio < 0.6:
            raise SystemExit(
                f"❌ FAIL_COHERENCE: majority_ratio={majority_ratio:.2f} < 0.6\n"
                f"    Likely cause: per-expert random init (trap 15) → output contains\n"
                f"    Arabic/garbled chars outside ascii+cjk+punct classes\n"
                f"    Abort: do not merge on garbage base"
            )
    print(f"  ✅ coherence check PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BF16 base dir(must be FUSED layout,e.g., S1-S5v2-S4)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir(adapter_model.safetensors)")
    ap.add_argument("--output", required=True, help="output dir for merged BF16")
    ap.add_argument("--skip-coherence", action="store_true", help="skip base coherence check(只 dev/test)")
    ap.add_argument("--shard-size", default="5GB", help="safetensors max shard size(default 5GB)")
    args = ap.parse_args()

    # Sanity
    assert os.path.exists(f"{args.base}/config.json"), f"Base config not found: {args.base}"
    assert os.path.exists(f"{args.adapter}/adapter_config.json"), f"Adapter config not found: {args.adapter}"
    assert os.path.exists(f"{args.adapter}/adapter_model.safetensors"), f"Adapter weights not found"

    if os.path.exists(args.output):
        print(f"[clean] removing stale output: {args.output}")
        shutil.rmtree(args.output)

    # === Step 1: load base via AutoModelForImageTextToText(多模态架构强制)===
    print(f"=== Step 1/3: load base BF16 from {args.base} ===")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    base = AutoModelForImageTextToText.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print(f"  base loaded: dtype={base.dtype}, device_map=auto")

    # === Step 1b: trap 15 coherence check ===
    if not args.skip_coherence:
        check_base_coherence(base, tokenizer, label="S1-S5v2-S4")

    # === Step 2: load adapter + merge ===
    print(f"\n=== Step 2/3: load LoRA adapter from {args.adapter} + merge ===")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter)
    print(f"  adapter loaded")
    print(f"  merging (in-place per module)...")
    merged = model.merge_and_unload()
    print(f"  merge done")

    # === Step 3: save merged BF16 ===
    print(f"\n=== Step 3/3: save merged BF16 to {args.output} ===")
    os.makedirs(args.output, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size=args.shard_size)
    tokenizer.save_pretrained(args.output)
    # Also copy chat_template if present
    for fn in ["chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"]:
        sp = f"{args.base}/{fn}"
        if os.path.exists(sp):
            shutil.copy(sp, f"{args.output}/{fn}")
    print(f"  saved")

    # Verify output
    assert os.path.exists(f"{args.output}/config.json"), "output config.json missing"
    total_size = sum(
        os.path.getsize(f"{args.output}/{f}")
        for f in os.listdir(args.output)
        if f.endswith(".safetensors")
    )
    print(f"\n✅ merge complete: {args.output}")
    print(f"   total safetensors size: {total_size / 1024 / 1024 / 1024:.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
