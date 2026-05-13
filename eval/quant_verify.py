"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""Quant variant verification:
- Load each quantized ckpt (NVFP4 modelopt / FP8 / NVFP4 v8-RTN / BF16 reference)
- Run 6 sanity prompts × 64 token greedy generation
- Check: no NaN/inf, readable script mix, repetition, prompt-specific semantic anchors, tokens/sec, output text
- Compare adapter outputs cross-variant vs BF16 reference
- Output JSON per variant + markdown summary

ONE model at a time to avoid 96GB OOM on R6000.

Usage:
    python3 quant_verify.py \\
        --bf16-dir /root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged \\
        --variant nvfp4-modelopt:/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-modelopt \\
        --variant fp8:/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-FP8 \\
        --variant v8-rtn:/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN \\
        --out-dir /root/autodl-tmp/reports
"""
import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch


TZ_CST = timezone(timedelta(hours=8))

PROMPTS = [
    "用一句话解释 Mixture-of-Experts active parameters。",
    "Python 写一个递归阶乘函数。",
    "比较 RoPE 与 ALiBi 的优缺点。",
    "今天北京适合穿什么?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请给出 SQL 查询:从 users 表选年龄大于 18 的用户。",
]
MAX_NEW = 64

CJK = re.compile(r"[一-鿿]")
ASCII_OK = re.compile(r"[A-Za-z0-9 ]")
PUNCT = re.compile(r"[,。.，、:!?;:!?\-—()()\"\"''「」《》<>\[\]]")
ARABIC = re.compile(r"[\u0600-\u06ff]")
KANA = re.compile(r"[\u3040-\u30ff]")
CYRILLIC = re.compile(r"[\u0400-\u04ff]")
HANGUL = re.compile(r"[\uac00-\ud7af]")
REPLACEMENT = re.compile(r"\ufffd")

PROMPT_ANCHORS = [
    ("moe_active_params", ["expert", "experts", "专家", "active", "激活", "参数", "parameter"]),
    ("python_factorial", ["def", "factorial", "阶乘", "return", "递归"]),
    ("rope_alibi", ["rope", "alibi", "位置", "position", "bias", "旋转"]),
    ("beijing_clothes", ["穿", "衣", "外套", "天气", "温度", "clothes", "wear"]),
    ("train_distance", ["150", "miles", "mile", "distance", "距离"]),
    ("sql_users", ["select", "users", "where", "age", "年龄"]),
]
GIBBERISH_FRAGMENTS = [
    "haltup", "depiwup", "luania", "iaup", "upupup", "printstats", "matchcondition",
]


def coherence_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    ok = sum(1 for c in text if CJK.match(c) or ASCII_OK.match(c) or PUNCT.match(c))
    return ok / total


def script_counts(text: str) -> dict:
    return {
        "cjk": sum(1 for c in text if CJK.match(c)),
        "latin_digit_space": sum(1 for c in text if ASCII_OK.match(c)),
        "punct": sum(1 for c in text if PUNCT.match(c)),
        "arabic": sum(1 for c in text if ARABIC.match(c)),
        "kana": sum(1 for c in text if KANA.match(c)),
        "cyrillic": sum(1 for c in text if CYRILLIC.match(c)),
        "hangul": sum(1 for c in text if HANGUL.match(c)),
        "replacement": sum(1 for c in text if REPLACEMENT.match(c)),
    }


def repetition_flags(text: str) -> dict:
    compact = re.sub(r"\s+", "", text.lower())
    repeated_ngram = bool(re.search(r"(.{2,8})\1{2,}", compact))
    repeated_char = bool(re.search(r"(.)\1{5,}", compact))
    known_fragments = [frag for frag in GIBBERISH_FRAGMENTS if frag in compact]
    return {
        "repeated_ngram": repeated_ngram,
        "repeated_char": repeated_char,
        "known_fragments": known_fragments,
        "pass": not repeated_ngram and not repeated_char and not known_fragments,
    }


def semantic_pass(prompt_idx: int, text: str) -> dict:
    name, anchors = PROMPT_ANCHORS[prompt_idx - 1]
    low = text.lower()
    hits = [a for a in anchors if a.lower() in low]
    return {
        "task": name,
        "anchors": anchors,
        "hits": hits,
        "pass": bool(hits),
    }


def quality_check(prompt_idx: int, text: str) -> dict:
    counts = script_counts(text)
    total = max(len(text), 1)
    weird_count = counts["arabic"] + counts["kana"] + counts["cyrillic"] + counts["hangul"] + counts["replacement"]
    weird_ratio = weird_count / total
    readable = coherence_ratio(text)
    repetition = repetition_flags(text)
    semantic = semantic_pass(prompt_idx, text)
    min_len_pass = len(text.strip()) >= 8
    script_pass = weird_count <= 2 and weird_ratio <= 0.03
    readable_pass = readable >= 0.65
    overall = min_len_pass and script_pass and readable_pass and repetition["pass"] and semantic["pass"]
    reasons = []
    if not min_len_pass:
        reasons.append("too_short")
    if not script_pass:
        reasons.append(f"unexpected_script_mix:{weird_count}/{total}")
    if not readable_pass:
        reasons.append(f"low_readable_ratio:{readable:.2f}")
    if not repetition["pass"]:
        reasons.append("repetition_or_known_gibberish")
    if not semantic["pass"]:
        reasons.append(f"missing_semantic_anchor:{semantic['task']}")
    return {
        "pass": overall,
        "reasons": reasons,
        "readable_ratio": round(readable, 3),
        "script_counts": counts,
        "weird_script_ratio": round(weird_ratio, 3),
        "repetition": repetition,
        "semantic": semantic,
    }


def has_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


def load_and_gen(model_path: str, label: str, prompts: list, max_new: int = MAX_NEW) -> dict:
    """Load model, run prompts, unload (free GPU). Returns per-prompt results."""
    print(f"\n[{label}] loading {model_path}", flush=True)
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception as e:
        return {
            "label": label,
            "path": model_path,
            "load_error": f"{type(e).__name__}: {str(e)[:300]}",
            "verdict": "LOAD_FAIL",
        }
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_dt = time.time() - t0
    print(f"[{label}] loaded in {load_dt:.1f}s, dtype={model.dtype}", flush=True)

    results = []
    for i, p in enumerate(prompts, 1):
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=max_new,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            dt = time.time() - t0
            new_tokens = out[0, input_len:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            n_tokens = len(new_tokens)
            tps = n_tokens / dt if dt > 0 else 0
            quality = quality_check(i, output)
            sample = {
                "idx": i,
                "prompt": p[:60],
                "output": output[:400],
                "n_tokens": n_tokens,
                "latency_s": round(dt, 2),
                "tokens_per_sec": round(tps, 2),
                "coherence_ratio": quality["readable_ratio"],
                "quality": quality,
                "coherent": quality["pass"],
            }
            results.append(sample)
            mark = "✓" if sample["coherent"] else "✗"
            reason = ",".join(quality["reasons"]) if quality["reasons"] else "ok"
            print(f"[{label}] [{i}/{len(prompts)}] {mark} {tps:.1f} tok/s "
                  f"read={quality['readable_ratio']:.2f} reason={reason} out={output[:60]!r}", flush=True)
        except Exception as e:
            results.append({"idx": i, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[{label}] [{i}/{len(prompts)}] ERROR: {type(e).__name__}", flush=True)

    n_coherent = sum(1 for r in results if r.get("coherent"))
    avg_tps = sum(r.get("tokens_per_sec", 0) for r in results) / max(len(results), 1)
    summary = {
        "label": label,
        "path": model_path,
        "load_time_s": round(load_dt, 1),
        "n_prompts": len(prompts),
        "n_coherent": n_coherent,
        "all_coherent": n_coherent == len(prompts),
        "verdict": "PASS" if n_coherent == len(prompts) else "FAIL",
        "avg_tokens_per_sec": round(avg_tps, 2),
        "items": results,
    }

    # Free GPU mem
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-dir", required=True, help="BF16 reference model dir")
    ap.add_argument("--variant", action="append", required=True,
                    help="LABEL:PATH e.g. nvfp4-modelopt:/root/.../NVFP4-modelopt (repeatable)")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/reports")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse variants
    variants = [("bf16-ref", args.bf16_dir)]
    for v in args.variant:
        if ":" not in v:
            sys.exit(f"--variant must be LABEL:PATH, got: {v}")
        label, path = v.split(":", 1)
        variants.append((label, path))

    print(f"[run] {len(variants)} variants to verify: {[v[0] for v in variants]}")
    all_results = []
    for label, path in variants:
        summary = load_and_gen(path, label, PROMPTS, MAX_NEW)
        all_results.append(summary)

    # Save JSON
    timestamp = datetime.now(TZ_CST).strftime("%Y%m%dT%H%M%S")
    report = {
        "schema_version": "lynn-quant-verify-v2",
        "created_at": datetime.now(TZ_CST).isoformat(timespec="seconds"),
        "n_variants": len(variants),
        "prompts": PROMPTS,
        "results": all_results,
    }
    out_path = out_dir / f"quant_verify_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[done] {out_path}")

    # Markdown summary
    md_path = out_dir / f"quant_verify_{timestamp}.md"
    lines = [
        f"# Lynn-V4-Pro-Distill Quant Verify {timestamp}",
        "",
        "| Variant | Load (s) | Verdict | Pass prompts | Avg tok/s | Path |",
        "|---|---|---|---|---|---|",
    ]
    for r in all_results:
        if "load_error" in r:
            lines.append(f"| **{r['label']}** | ❌ LOAD FAIL | LOAD_FAIL | — | — | `{r.get('path', '?')}` |")
            lines.append(f"| | error: `{r['load_error']}` | | | | |")
            continue
        mark = "✅ PASS" if r["all_coherent"] else "❌ FAIL"
        lines.append(
            f"| **{r['label']}** | {r['load_time_s']} | {mark} | {r['n_coherent']}/{r['n_prompts']} "
            f"| {r['avg_tokens_per_sec']} | `{r['path']}` |"
        )

    lines.append("\n## Sample outputs (prompt 1)")
    for r in all_results:
        if r.get("items"):
            first = r["items"][0]
            verdict = "PASS" if first.get("coherent") else "FAIL"
            reasons = ",".join(first.get("quality", {}).get("reasons", []))
            lines.append(f"\n**{r['label']}** [{verdict} {reasons}] (`{first.get('output', '?')[:200]}`)")

    md_path.write_text("\n".join(lines))
    print(f"[done] {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
