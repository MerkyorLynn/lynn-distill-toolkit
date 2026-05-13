"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""Four-gate eval — Lynn V4 distill ship gate (canonical schema lynn-4gate-v1).

Final ship gate before Spark cutover. Schema is the single source of truth;
spark_cutover.py only reads top-level `verdict == NET_WIN` and `net_score >= 10`.

Gates:
  g1_v4_style    : V4 distill 主打 — rule-based style on v4_distill_verify_35
  g2_regression  : 回归 — v8 (stage1+4+5 tool-calling) + v9 (holdout+probe math)
                   Each gate runs adapter on AND off, computes delta_pp.
  g3_base_parity : 复用 v4 prompts, adapter on/off win/tie/loss → net_score

Verdict (strict, no WARN):
  NET_WIN if ALL of:
    sanity_pass                        (from --sanity-json or --sanity-pass flag)
    g1.avg_style       >= 4.0
    g1.cliche_free_pct >= 90.0
    g2.v8_delta_pp     >= -5.0
    g2.v9_delta_pp     >= -5.0
    g3.net_score       >= 10.0
  else ABORT (reasons in hard_fail_reasons[])

Usage:
    python3 four_gate_eval.py \\
        --base /root/autodl-tmp/models/.../base \\
        --adapter /root/autodl-tmp/adapters/Lynn-V4-Pro-r64 \\
        --model-tag lynn-v4-pro-r64 \\
        --runner r6000 \\
        --prompts-dir /root/autodl-tmp/eval_prompts \\
        --out-dir /root/autodl-tmp/reports \\
        --sanity-json /root/autodl-tmp/reports/diff_sanity_lynn-v4-pro-r64.json
"""
import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean

import torch


# ─── Constants ───────────────────────────────────────────────────────────────

SCHEMA_VERSION = "lynn-4gate-v1"
TZ_CST = timezone(timedelta(hours=8))

THRESHOLDS = {
    "min_net_score": 10.0,
    "max_regression_pp": -5.0,
    "min_avg_style": 4.0,
    "min_cliche_free_pct": 90.0,
    "min_reference_win_delta": 0.03,
}

CLICHE_STARTS = [
    "I'd be happy to help",
    "Let me break this down",
    "As an AI",
    "I cannot",
    "好的,我来帮",
    "首先,让我",
    "当然可以,",
    "好的,",
    "当然,",
]

CJK_PATTERN = re.compile(r"[一-鿿]")
ASCII_OK = re.compile(r"[A-Za-z0-9\s.,;:!?\-\(\)\[\]\{\}\"'`/\\<>=+*&|%#@$~\^_]")

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_JSON_INLINE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}', re.DOTALL
)


# ─── Prompts loading ─────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    assert path.exists(), f"prompt file missing: {path}"
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def get_prompt_text(record: dict) -> str:
    return record.get("prompt") or record.get("user") or record.get("problem") or ""


def choose_max_new(record: dict) -> int:
    cat = record.get("category", "")
    if "long" in cat or "research" in cat:
        return 1024
    if record.get("verifier") or "math" in record.get("subset", ""):
        return 384
    return 384


# ─── Rule-based scoring (g1, g3) ─────────────────────────────────────────────

def length_ok(text: str, lo: int = 50, hi: int = 8000) -> bool:
    return lo <= len(text) <= hi


def no_tail_repetition(text: str, n: int = 8, max_repeat: int = 5) -> bool:
    if len(text) < n * max_repeat:
        return True
    tail = text[len(text) // 2:]
    counts = {}
    for i in range(len(tail) - n + 1):
        gram = tail[i:i + n]
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= max_repeat:
            return False
    return True


def no_garbage(text: str, threshold: float = 0.6) -> bool:
    if not text:
        return False
    ok = sum(1 for c in text if CJK_PATTERN.match(c) or ASCII_OK.match(c))
    return (ok / len(text)) >= threshold


def no_cliche(text: str) -> bool:
    head = text.lstrip()[:60]
    return not any(head.startswith(c) for c in CLICHE_STARTS)


def format_match(prompt: str, output: str) -> bool:
    p = prompt.lower()
    if any(k in p for k in ["python", "实现", "sql", "bash", "代码"]):
        return ("```" in output) or ("def " in output) or ("SELECT" in output.upper())
    if any(k in prompt for k in ["证明", "多远", "整除", "分数", "转成"]):
        return bool(re.search(r"\d", output))
    if "json" in p:
        return "{" in output and "}" in output
    if "markdown" in p and "表格" in prompt:
        return "|" in output
    return True


RULES = [
    ("length_ok", lambda p, o: length_ok(o)),
    ("no_tail_repetition", lambda p, o: no_tail_repetition(o)),
    ("no_garbage", lambda p, o: no_garbage(o)),
    ("no_cliche", lambda p, o: no_cliche(o)),
    ("format_match", lambda p, o: format_match(p, o)),
]


def score_rule_based(prompt: str, output: str) -> dict:
    rule_results = {name: fn(prompt, output) for name, fn in RULES}
    n_pass = sum(1 for v in rule_results.values() if v)
    return {
        "rules": rule_results,
        "n_rule_pass": n_pass,
        "n_rule_total": len(RULES),
        "rule_pass_ratio": n_pass / len(RULES),
        "overall_pass": n_pass >= 4,  # 4/5 rules
    }


def char_ngrams(text: str, n: int = 3) -> set:
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def f1_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(b)
    return 2 * precision * recall / (precision + recall)


def extract_numbers(text: str) -> set:
    return set(re.findall(r"(?:\d+(?:\.\d+)?%?|\d{4}年|\d+月|\d+日)", text))


def extract_cjk_terms(text: str) -> set:
    # Lightweight semantic proxy without adding tokenizer deps. Long technical nouns
    # carry most of the reference signal for these Chinese distill prompts.
    return set(re.findall(r"[\u4e00-\u9fff]{2,8}", text))


def structure_features(text: str) -> dict:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    headers = sum(1 for x in lines if re.match(r"^(#{1,4}\s+|[一二三四五六七八九十]+[、.．]|[0-9]+[、.．])", x))
    bullets = sum(1 for x in lines if re.match(r"^([-*+]\s+|[0-9]+[.)、]\s+)", x))
    code_blocks = text.count("```") // 2
    tables = sum(1 for x in lines if x.count("|") >= 2)
    paragraphs = max(1, len(re.split(r"\n\s*\n", text.strip()))) if text.strip() else 0
    nums = len(extract_numbers(text))
    return {
        "headers": headers,
        "bullets": bullets,
        "code_blocks": code_blocks,
        "tables": tables,
        "paragraphs": paragraphs,
        "numbers": nums,
    }


def coverage_score(actual: int, target: int) -> float:
    if target <= 0:
        return 1.0 if actual <= 0 else 0.8
    return min(actual / target, 1.0)


def reference_similarity_score(prompt: str, output: str, reference: str, category: str = "") -> dict:
    """Reference-calibrated score for G3.

    This measures the actual distillation objective: adapter/base output closeness
    to the V4-Pro teacher reference. It deliberately avoids rewarding raw length.
    """
    output = output or ""
    reference = reference or ""
    if not reference:
        rb = score_rule_based(prompt, output)
        return {"score": rb["rule_pass_ratio"], "mode": "rule_fallback", "components": rb["rules"]}

    tri = f1_overlap(char_ngrams(output, 3), char_ngrams(reference, 3))
    bi = f1_overlap(char_ngrams(output, 2), char_ngrams(reference, 2))
    seq = SequenceMatcher(None, output[:5000], reference[:5000]).ratio()
    num = f1_overlap(extract_numbers(output), extract_numbers(reference))
    terms = f1_overlap(extract_cjk_terms(output), extract_cjk_terms(reference))

    of = structure_features(output)
    rf = structure_features(reference)
    struct_parts = [
        coverage_score(of["headers"], rf["headers"]),
        coverage_score(of["bullets"], rf["bullets"]),
        coverage_score(of["code_blocks"], rf["code_blocks"]),
        coverage_score(of["tables"], rf["tables"]),
        coverage_score(of["paragraphs"], rf["paragraphs"]),
        coverage_score(of["numbers"], rf["numbers"]),
    ]
    struct = sum(struct_parts) / len(struct_parts)

    # Keep floor checks as penalties, not as the primary score. This prevents
    # "close to reference but degenerate/cliche" from winning.
    floor = score_rule_based(prompt, output)
    floor_penalty = 1.0
    if not floor["rules"]["no_cliche"]:
        floor_penalty -= 0.15
    if not floor["rules"]["no_tail_repetition"]:
        floor_penalty -= 0.15
    if not floor["rules"]["no_garbage"]:
        floor_penalty -= 0.20
    floor_penalty = max(0.5, floor_penalty)

    lexical = 0.60 * tri + 0.40 * bi
    entity = 0.55 * terms + 0.45 * num
    score = (0.45 * lexical + 0.25 * seq + 0.15 * entity + 0.15 * struct) * floor_penalty
    return {
        "score": round(float(score), 4),
        "mode": "reference_similarity",
        "components": {
            "trigram_f1": round(tri, 4),
            "bigram_f1": round(bi, 4),
            "sequence": round(seq, 4),
            "number_f1": round(num, 4),
            "term_f1": round(terms, 4),
            "structure": round(struct, 4),
            "floor_penalty": round(floor_penalty, 4),
        },
        "output_features": of,
        "reference_features": rf,
    }


# ─── Tool-call scoring (g2_v8) ───────────────────────────────────────────────

def extract_tool_calls(output: str) -> list:
    """Structured-first: <tool_call>{...}</tool_call> > inline JSON blob.

    Returns list of {"name": str, "arguments": dict}.
    """
    calls = []
    for m in TOOL_CALL_BLOCK.finditer(output):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                calls.append(obj)
        except json.JSONDecodeError:
            continue
    if not calls:
        for m in TOOL_CALL_JSON_INLINE.finditer(output):
            try:
                obj = json.loads(m.group())
                if isinstance(obj, dict) and obj.get("name"):
                    calls.append(obj)
            except json.JSONDecodeError:
                continue
    return calls


def score_tool_call(output: str, expected: dict) -> dict:
    """Structured-first; substring fallback requires name + at least one param hint."""
    expected_name = expected.get("tool_name", "")
    param_hints = expected.get("param_hints", {})

    structured = extract_tool_calls(output)
    if structured:
        for call in structured:
            if call.get("name") == expected_name:
                # User rule: 至少要求 tool_name 出现在结构化调用块 → pass
                return {"pass": True, "via": "structured", "structured_calls": structured}
        # structured exists but none match expected name → fail (don't fallback)
        return {"pass": False, "via": "structured_mismatch", "structured_calls": structured}

    # No structured block → substring fallback (strict: name + at least 1 param hint)
    has_name = expected_name in output
    has_param = False
    if param_hints:
        for key, hints in param_hints.items():
            if any(h in output for h in hints):
                has_param = True
                break
    else:
        has_param = True  # no param hints required by expected spec
    return {
        "pass": has_name and has_param,
        "via": "substring_fallback",
        "has_name": has_name,
        "has_param": has_param,
    }


# ─── String-match scoring (g2_v9) ────────────────────────────────────────────

def score_string_match(output: str, record: dict) -> dict:
    """v9 holdout/probe: prefer verifier=string_match on gold_answer; fallback rule."""
    gold = record.get("gold_answer")
    verifier = record.get("verifier", "string_match")
    if gold is None:
        rb = score_rule_based(get_prompt_text(record), output)
        return {"pass": rb["overall_pass"], "via": "rule_fallback", "rule_pass_ratio": rb["rule_pass_ratio"]}
    if verifier == "string_match":
        passed = str(gold).strip() in output
    elif verifier == "regex":
        passed = bool(re.search(str(gold), output))
    else:
        passed = str(gold).strip() in output
    return {"pass": passed, "via": verifier, "gold": str(gold)[:80]}


# ─── Generation ──────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new: int = 384, tools: list = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if tools:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


# ─── Gate runners ────────────────────────────────────────────────────────────

def run_g1_v4_style(model, tokenizer, prompts: list) -> tuple:
    """g1: adapter on, rule-based, returns (gate_dict, adapter_outputs_cache)."""
    print(f"\n=== g1_v4_style (adapter on, n={len(prompts)}) ===")
    model.set_adapter("a0")
    items = []
    cache = []
    for i, rec in enumerate(prompts, 1):
        ptext = get_prompt_text(rec)
        max_new = choose_max_new(rec)
        t0 = time.time()
        out = generate(model, tokenizer, ptext, max_new=max_new)
        dt = time.time() - t0
        sc = score_rule_based(ptext, out)
        items.append({
            "idx": i,
            "id": rec.get("id", f"v4_{i:03d}"),
            "category": rec.get("category", ""),
            "prompt": ptext[:200],
            "output_len": len(out),
            "latency_s": round(dt, 2),
            "rule_pass_ratio": sc["rule_pass_ratio"],
            "cliche_free": sc["rules"]["no_cliche"],
            "overall_pass": sc["overall_pass"],
        })
        cache.append({"prompt": ptext, "output": out, "score": sc, "record": rec})
        print(f"  [{i}/{len(prompts)}] {dt:.1f}s rule_pass={sc['n_rule_pass']}/5 "
              f"len={len(out)} {rec.get('category', '')[:20]}")

    rule_ratios = [it["rule_pass_ratio"] for it in items]
    avg_style = mean(rule_ratios) * 5.0  # scale [0,1] → [0,5]
    cliche_free_pct = sum(1 for it in items if it["cliche_free"]) / len(items) * 100
    gate_pass = (avg_style >= THRESHOLDS["min_avg_style"]
                 and cliche_free_pct >= THRESHOLDS["min_cliche_free_pct"])
    gate = {
        "verdict": "PASS" if gate_pass else "FAIL",
        "n": len(prompts),
        "avg_style": round(avg_style, 3),
        "cliche_free_pct": round(cliche_free_pct, 2),
        "items": items,
    }
    print(f"  ⇒ avg_style={avg_style:.2f}  cliche_free={cliche_free_pct:.1f}%  "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return gate, cache


def run_g2_v8(model, tokenizer, prompts: list) -> dict:
    """g2_v8: stage1+4+5 tool-calling regression. Adapter on + base. Delta pp."""
    print(f"\n=== g2_v8 stage regression (n={len(prompts)}, adapter on + base) ===")
    items_adapter, items_base = [], []

    # Adapter pass
    model.set_adapter("a0")
    for i, rec in enumerate(prompts, 1):
        ptext = get_prompt_text(rec)
        tools = rec.get("tools")
        out = generate(model, tokenizer, ptext, max_new=256, tools=tools)
        sc = score_tool_call(out, rec.get("expected", {}))
        items_adapter.append({"id": rec.get("id"), "pass": sc["pass"], "via": sc["via"]})

    # Base pass (disable adapter)
    with model.disable_adapter():
        for i, rec in enumerate(prompts, 1):
            ptext = get_prompt_text(rec)
            tools = rec.get("tools")
            out = generate(model, tokenizer, ptext, max_new=256, tools=tools)
            sc = score_tool_call(out, rec.get("expected", {}))
            items_base.append({"id": rec.get("id"), "pass": sc["pass"], "via": sc["via"]})

    adapter_pct = sum(1 for it in items_adapter if it["pass"]) / len(prompts) * 100
    base_pct = sum(1 for it in items_base if it["pass"]) / len(prompts) * 100
    delta_pp = adapter_pct - base_pct
    print(f"  ⇒ v8: adapter={adapter_pct:.1f}%  base={base_pct:.1f}%  delta={delta_pp:+.2f}pp")
    return {
        "n": len(prompts),
        "adapter_pass_pct": round(adapter_pct, 2),
        "base_pass_pct": round(base_pct, 2),
        "delta_pp": round(delta_pp, 2),
        "items_adapter": items_adapter,
        "items_base": items_base,
    }


def run_g2_v9(model, tokenizer, prompts: list) -> dict:
    """g2_v9: holdout+probe string_match on gold_answer. Adapter on + base."""
    print(f"\n=== g2_v9 holdout/probe regression (n={len(prompts)}, adapter on + base) ===")
    items_adapter, items_base = [], []

    model.set_adapter("a0")
    for i, rec in enumerate(prompts, 1):
        ptext = get_prompt_text(rec)
        out = generate(model, tokenizer, ptext, max_new=512)
        sc = score_string_match(out, rec)
        items_adapter.append({"id": rec.get("id"), "pass": sc["pass"], "via": sc["via"]})

    with model.disable_adapter():
        for i, rec in enumerate(prompts, 1):
            ptext = get_prompt_text(rec)
            out = generate(model, tokenizer, ptext, max_new=512)
            sc = score_string_match(out, rec)
            items_base.append({"id": rec.get("id"), "pass": sc["pass"], "via": sc["via"]})

    adapter_pct = sum(1 for it in items_adapter if it["pass"]) / len(prompts) * 100
    base_pct = sum(1 for it in items_base if it["pass"]) / len(prompts) * 100
    delta_pp = adapter_pct - base_pct
    print(f"  ⇒ v9: adapter={adapter_pct:.1f}%  base={base_pct:.1f}%  delta={delta_pp:+.2f}pp")
    return {
        "n": len(prompts),
        "adapter_pass_pct": round(adapter_pct, 2),
        "base_pass_pct": round(base_pct, 2),
        "delta_pp": round(delta_pp, 2),
        "items_adapter": items_adapter,
        "items_base": items_base,
    }


def run_g3_base_parity(model, tokenizer, g1_cache: list) -> dict:
    """g3: rerun base on g1 prompts, compare adapter/base to teacher reference.

    Categories: adapter_win / base_win / tie. If references are missing, fall back
    to the old floor-pass bucket, but v2 prompts should always include references.
    """
    print(f"\n=== g3_base_parity (base rerun on v4 prompts, n={len(g1_cache)}) ===")
    adapter_win = base_win = tie = 0
    tie_both_pass = tie_both_fail = 0
    items = []
    win_delta = THRESHOLDS["min_reference_win_delta"]

    with model.disable_adapter():
        for i, entry in enumerate(g1_cache, 1):
            ptext = entry["prompt"]
            rec = entry["record"]
            adapter_pass = entry["score"]["overall_pass"]
            max_new = choose_max_new(rec)
            out_base = generate(model, tokenizer, ptext, max_new=max_new)
            base_sc = score_rule_based(ptext, out_base)
            base_pass = base_sc["overall_pass"]

            reference = rec.get("reference_output", "")
            if reference:
                adapter_ref = reference_similarity_score(ptext, entry["output"], reference, rec.get("category", ""))
                base_ref = reference_similarity_score(ptext, out_base, reference, rec.get("category", ""))
                delta = adapter_ref["score"] - base_ref["score"]
                if delta >= win_delta:
                    bucket = "adapter_win"; adapter_win += 1
                elif delta <= -win_delta:
                    bucket = "base_win"; base_win += 1
                else:
                    bucket = "tie"; tie += 1
                items.append({
                    "id": rec.get("id"),
                    "category": rec.get("category", ""),
                    "adapter_pass": adapter_pass,
                    "base_pass": base_pass,
                    "bucket": bucket,
                    "adapter_ref_score": adapter_ref["score"],
                    "base_ref_score": base_ref["score"],
                    "ref_delta": round(delta, 4),
                    "adapter_components": adapter_ref["components"],
                    "base_components": base_ref["components"],
                    "reference_len": len(reference),
                    "adapter_len": len(entry["output"]),
                    "base_len": len(out_base),
                })
                print(f"  [{i}/{len(g1_cache)}] {bucket} "
                      f"adapter_ref={adapter_ref['score']:.4f} base_ref={base_ref['score']:.4f} "
                      f"delta={delta:+.4f}")
            else:
                if adapter_pass and not base_pass:
                    bucket = "adapter_win"; adapter_win += 1
                elif base_pass and not adapter_pass:
                    bucket = "base_win"; base_win += 1
                elif adapter_pass and base_pass:
                    bucket = "tie_both_pass"; tie_both_pass += 1
                else:
                    bucket = "tie_both_fail"; tie_both_fail += 1
                items.append({
                    "id": rec.get("id"),
                    "category": rec.get("category", ""),
                    "adapter_pass": adapter_pass,
                    "base_pass": base_pass,
                    "bucket": bucket,
                    "mode": "rule_fallback_no_reference",
                })
                print(f"  [{i}/{len(g1_cache)}] {bucket} "
                      f"adapter={adapter_pass} base={base_pass}")

    n = len(g1_cache)
    adapter_win_pp = adapter_win / n * 100
    base_win_pp = base_win / n * 100
    net_score = (adapter_win - base_win) / n * 100
    gate_pass = net_score >= THRESHOLDS["min_net_score"]
    print(f"  ⇒ adapter_win={adapter_win} base_win={base_win} "
          f"tie_both_pass={tie_both_pass} tie_both_fail={tie_both_fail} "
          f"net_score={net_score:+.2f}pp  {'PASS' if gate_pass else 'FAIL'}")
    return {
        "verdict": "PASS" if gate_pass else "FAIL",
        "n": n,
        "adapter_win": adapter_win,
        "base_win": base_win,
        "tie": tie,
        "tie_both_pass": tie_both_pass,
        "tie_both_fail": tie_both_fail,
        "adapter_win_pp": round(adapter_win_pp, 2),
        "base_win_pp": round(base_win_pp, 2),
        "net_score": round(net_score, 2),
        "min_reference_win_delta": win_delta,
        "scoring": "reference_similarity_v1",
        "items": items,
    }


# ─── Sanity resolver ─────────────────────────────────────────────────────────

def resolve_sanity(args) -> dict:
    """Priority: --sanity-json (read JSON) > --sanity-pass flag > implicit False."""
    if args.sanity_json:
        p = Path(args.sanity_json)
        if not p.exists():
            return {"pass": False, "source": "json_missing", "report_path": str(p)}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {
                "pass": bool(data.get("pass", False)),
                "source": "json",
                "report_path": str(p),
                "n_pass": data.get("n_pass"),
                "n_total": data.get("n_total"),
                "schema_version": data.get("schema_version"),
            }
        except (json.JSONDecodeError, OSError) as e:
            return {"pass": False, "source": f"json_parse_error:{e}", "report_path": str(p)}
    if args.sanity_pass:
        return {"pass": True, "source": "flag", "report_path": None}
    return {"pass": False, "source": "implicit_false", "report_path": None}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--runner", default="r6000", choices=["r6000", "a100", "local", "remote"])
    ap.add_argument("--prompts-dir", default="/root/autodl-tmp/eval_prompts")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/reports")
    ap.add_argument("--sanity-json", default=None,
                    help="diff_sanity_*.json path (preferred). Read 'pass' field.")
    ap.add_argument("--sanity-pass", action="store_true",
                    help="fallback flag if no --sanity-json. Default False (abort).")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert (base_path / "config.json").exists(), f"Base config missing: {base_path}"
    assert (adapter_path / "adapter_config.json").exists(), f"Adapter config missing"

    # Resolve sanity first (cheap, fail-fast if implicit_false)
    sanity = resolve_sanity(args)
    print(f"[sanity] pass={sanity['pass']} source={sanity['source']}")

    # Load prompts
    print(f"[prompts] dir={prompts_dir}")
    v4_prompt_file = prompts_dir / "v4_distill_verify_35_v2.jsonl"
    if not v4_prompt_file.exists():
        v4_prompt_file = prompts_dir / "v4_distill_verify_35.jsonl"
    v4_prompts = load_jsonl(v4_prompt_file)
    v8_prompts = (load_jsonl(prompts_dir / "stage1_tool_calling.jsonl")
                  + load_jsonl(prompts_dir / "stage4_research.jsonl")
                  + load_jsonl(prompts_dir / "stage5_coding.jsonl"))
    v9_prompts = (load_jsonl(prompts_dir / "v9_holdout.jsonl")
                  + load_jsonl(prompts_dir / "v9_probe_expanded.jsonl"))
    n_v4_refs = sum(1 for x in v4_prompts if x.get("reference_output"))
    print(f"[prompts] v4={len(v4_prompts)} refs={n_v4_refs} file={v4_prompt_file.name} "
          f"v8={len(v8_prompts)} v9={len(v9_prompts)}")

    # Load model
    print(f"[load] base={base_path}")
    print(f"[load] adapter={adapter_path}")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    t0 = time.time()
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(base_path), torch_dtype=dtype, device_map=args.device_map, trust_remote_code=True
    )
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_path),
                                       adapter_name="a0", is_trainable=False)
    model.set_adapter("a0")
    print(f"[load] done in {time.time()-t0:.1f}s")

    # Run gates
    gate_t0 = time.time()
    g1, g1_cache = run_g1_v4_style(model, tokenizer, v4_prompts)
    v8 = run_g2_v8(model, tokenizer, v8_prompts)
    v9 = run_g2_v9(model, tokenizer, v9_prompts)
    g3 = run_g3_base_parity(model, tokenizer, g1_cache)
    gate_dt = time.time() - gate_t0

    # Aggregate g2
    g2_pass = (v8["delta_pp"] >= THRESHOLDS["max_regression_pp"]
               and v9["delta_pp"] >= THRESHOLDS["max_regression_pp"])
    g2 = {
        "verdict": "PASS" if g2_pass else "FAIL",
        "v8_delta_pp": v8["delta_pp"],
        "v9_delta_pp": v9["delta_pp"],
        "v8_adapter_pass_pct": v8["adapter_pass_pct"],
        "v8_base_pass_pct": v8["base_pass_pct"],
        "v9_adapter_pass_pct": v9["adapter_pass_pct"],
        "v9_base_pass_pct": v9["base_pass_pct"],
        "v8_items_adapter": v8["items_adapter"],
        "v8_items_base": v8["items_base"],
        "v9_items_adapter": v9["items_adapter"],
        "v9_items_base": v9["items_base"],
    }

    # Compute verdict — strict, hard_fail_reasons populated
    net_score = g3["net_score"]
    hard_fail_reasons = []
    if not sanity["pass"]:
        hard_fail_reasons.append(f"sanity.pass=False (source={sanity['source']})")
    if g1["avg_style"] < THRESHOLDS["min_avg_style"]:
        hard_fail_reasons.append(
            f"g1.avg_style={g1['avg_style']} < {THRESHOLDS['min_avg_style']}")
    if g1["cliche_free_pct"] < THRESHOLDS["min_cliche_free_pct"]:
        hard_fail_reasons.append(
            f"g1.cliche_free_pct={g1['cliche_free_pct']} < {THRESHOLDS['min_cliche_free_pct']}")
    if v8["delta_pp"] < THRESHOLDS["max_regression_pp"]:
        hard_fail_reasons.append(
            f"g2.v8_delta_pp={v8['delta_pp']} < {THRESHOLDS['max_regression_pp']}")
    if v9["delta_pp"] < THRESHOLDS["max_regression_pp"]:
        hard_fail_reasons.append(
            f"g2.v9_delta_pp={v9['delta_pp']} < {THRESHOLDS['max_regression_pp']}")
    if net_score < THRESHOLDS["min_net_score"]:
        hard_fail_reasons.append(
            f"g3.net_score={net_score} < {THRESHOLDS['min_net_score']}")

    verdict = "NET_WIN" if not hard_fail_reasons else "ABORT"
    created_at = datetime.now(TZ_CST).isoformat(timespec="seconds")

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "runner": args.runner,
        "model_tag": args.model_tag,
        "base": str(base_path),
        "adapter": str(adapter_path),
        "verdict": verdict,
        "net_score": net_score,
        "thresholds": THRESHOLDS,
        "gates": {
            "g1_v4_style": g1,
            "g2_regression": g2,
            "g3_base_parity": g3,
        },
        "deltas": {
            "adapter_vs_base_win_pp": net_score,
            "v8_regression_pp": v8["delta_pp"],
            "v9_regression_pp": v9["delta_pp"],
        },
        "hard_fail_reasons": hard_fail_reasons,
        "sanity": sanity,
        "runtime_s": round(gate_dt, 1),
    }

    # Write timestamped + canonical paths
    ts = datetime.now(TZ_CST).strftime("%Y%m%dT%H%M%S")
    timestamped = out_dir / f"4gate_{args.model_tag}_{ts}.json"
    canonical = out_dir / "4gate_results.json"
    timestamped.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    canonical.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  net_score={net_score:+.2f}pp")
    if hard_fail_reasons:
        print(f"  hard_fail_reasons:")
        for r in hard_fail_reasons:
            print(f"    - {r}")
    print(f"  timestamped: {timestamped}")
    print(f"  canonical:   {canonical}")
    print(f"\n__FOUR_GATE_{'PASS' if verdict == 'NET_WIN' else 'FAIL'}__")

    return 0 if verdict == "NET_WIN" else 2


if __name__ == "__main__":
    sys.exit(main())
