"""
Part of Lynn V4-Pro Distill Toolkit — https://github.com/MerkyorLynn/lynn-distill-toolkit

NOTE: Default paths (/root/autodl-tmp/..., /mnt/data3/...) reflect the R6000/A100
      rental environment where the V4-Pro Distill pipeline was developed.
      Adjust paths for your setup OR use the path constants at top of each script.
"""
#!/usr/bin/env python3
"""Activation profile — collect MoE router stats per layer per expert.

For each forward token in calibration prompts, hooks each `model.layers.{i}.mlp.gate`
Linear router. Captures:
  - top-k expert indices (Qwen3.6-A3B: top_k=8 out of 128 experts)
  - softmax(router_logits) → per-expert routing probabilities

Aggregates per layer:
  - expert_usage_pct: % of tokens routed through each expert
  - avg_gate_prob: average gate probability per expert
  - by_category: same split by prompt category (for expert specialization map)

Output: lynn-activation-profile-v1 JSON. Drives expert prune decisions
(see project_lynn_27b_pruning_engine_strategy_0512.md Phase A).

Usage:
    python3 activation_profile.py \\
        --model /root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged \\
        --prompts-dir /root/autodl-tmp/eval_prompts \\
        --out /root/autodl-tmp/reports/activation_profile.json \\
        --max-tokens 64

Dry-run on raw S1-S5v2-S4 base (verify hooks work before Lynn-Distill):
    python3 activation_profile.py \\
        --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-S1-S5v2-S4 \\
        --prompts-dir /root/autodl-tmp/eval_prompts \\
        --out /root/autodl-tmp/reports/activation_profile_dryrun.json \\
        --max-prompts 5 --max-tokens 16
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F


TZ_CST = timezone(timedelta(hours=8))
SCHEMA_VERSION = "lynn-activation-profile-v1"

ROUTER_PATTERN = re.compile(r"\.layers\.(\d+)\.mlp\.gate$")


def load_prompts(prompts_dir: Path, max_prompts: int = None) -> list:
    """Load all .jsonl files,union prompts with category metadata."""
    items = []
    for jf in sorted(prompts_dir.glob("*.jsonl")):
        category = jf.stem
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = rec.get("prompt") or rec.get("user") or rec.get("problem") or ""
            if prompt:
                items.append({
                    "category": rec.get("category") or category,
                    "source_file": jf.name,
                    "id": rec.get("id"),
                    "prompt": prompt,
                })
        if max_prompts and len(items) >= max_prompts:
            break
    if max_prompts:
        items = items[:max_prompts]
    return items


_state = {}


def install_router_hooks(model, num_layers: int, num_experts: int, top_k: int):
    _state.clear()
    _state["expert_count"] = [[0] * num_experts for _ in range(num_layers)]
    _state["gate_prob_sum"] = [[0.0] * num_experts for _ in range(num_layers)]
    _state["total_tokens"] = [0] * num_layers
    _state["cat_expert_count"] = defaultdict(
        lambda: [[0] * num_experts for _ in range(num_layers)]
    )
    _state["current_category"] = None

    def make_hook(layer_idx):
        def hook(module, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            logits = output.detach().float()
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)  # (1, seq, experts) 防 batchless
            if logits.dim() != 3:
                return
            probs = F.softmax(logits, dim=-1)
            topk_vals, topk_idx = probs.topk(top_k, dim=-1)
            b, s, k = topk_idx.shape
            n_tokens = b * s
            _state["total_tokens"][layer_idx] += n_tokens
            idx_flat = topk_idx.reshape(-1, k).cpu().tolist()
            prob_flat = topk_vals.reshape(-1, k).cpu().tolist()
            cat = _state["current_category"]
            cnt = _state["expert_count"][layer_idx]
            psum = _state["gate_prob_sum"][layer_idx]
            cat_cnt = _state["cat_expert_count"][cat][layer_idx] if cat else None
            for token_i in range(len(idx_flat)):
                for k_i in range(k):
                    e = idx_flat[token_i][k_i]
                    p = prob_flat[token_i][k_i]
                    cnt[e] += 1
                    psum[e] += p
                    if cat_cnt is not None:
                        cat_cnt[e] += 1
        return hook

    handles = []
    matched_layers = set()
    for name, module in model.named_modules():
        m = ROUTER_PATTERN.search(name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx < num_layers:
                handles.append(module.register_forward_hook(make_hook(layer_idx)))
                matched_layers.add(layer_idx)
    print(f"[hooks] installed {len(handles)} router hooks "
          f"({len(matched_layers)}/{num_layers} layers matched)")
    if len(matched_layers) < num_layers:
        missing = set(range(num_layers)) - matched_layers
        print(f"  ⚠️ missing layers: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        print(f"  → router pattern may not match this arch. Inspect with --inspect")
    return handles


def inspect_model(model_path: str):
    """Print module names matching common router patterns. For debug only."""
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"[inspect] config: layers={getattr(cfg, 'num_hidden_layers', '?')} "
          f"experts={getattr(cfg, 'num_experts', '?')} "
          f"n_routed={getattr(cfg, 'n_routed_experts', '?')}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    print(f"[inspect] modules with 'gate' or 'router' in name (first 20):")
    n = 0
    for name, _ in model.named_modules():
        low = name.lower()
        if "gate" in low or "router" in low:
            print(f"  {name}")
            n += 1
            if n >= 20:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--inspect", action="store_true",
                    help="print model module names (debug router pattern)")
    args = ap.parse_args()

    if args.inspect:
        inspect_model(args.model)
        return 0

    prompts = load_prompts(Path(args.prompts_dir), max_prompts=args.max_prompts)
    print(f"[load] {len(prompts)} prompts from {args.prompts_dir}")

    from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)  # multimodal wraps text config
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    num_experts = (getattr(text_cfg, "num_experts", None)
                   or getattr(text_cfg, "n_routed_experts", None)
                   or getattr(cfg, "num_experts", None))
    assert num_layers and num_experts, (
        f"missing config: layers={num_layers} experts={num_experts}. "
        f"Run --inspect to debug arch."
    )
    print(f"[config] layers={num_layers} experts={num_experts} top_k={args.top_k}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[load] done in {time.time()-t0:.1f}s")

    handles = install_router_hooks(model, num_layers, num_experts, args.top_k)
    if len(handles) == 0:
        print("ERROR: no router hooks installed. Check arch with --inspect.")
        return 2

    try:
        gen_t0 = time.time()
        for i, p in enumerate(prompts, 1):
            _state["current_category"] = p["category"]
            messages = [{"role": "user", "content": p["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                _ = model.generate(
                    **inputs, max_new_tokens=args.max_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            if i % 10 == 0 or i == len(prompts) or i <= 3:
                dt = time.time() - gen_t0
                tps = _state["total_tokens"][0] / dt if dt > 0 else 0
                print(f"  [{i}/{len(prompts)}] cat={p['category'][:20]} "
                      f"layer0_tokens={_state['total_tokens'][0]} {tps:.1f}tok/s")
    finally:
        for h in handles:
            h.remove()

    layers_out = []
    for li in range(num_layers):
        total = _state["total_tokens"][li]
        cnt = _state["expert_count"][li]
        psum = _state["gate_prob_sum"][li]
        usage_pct = [c / total * 100 if total > 0 else 0 for c in cnt]
        avg_prob = [s / cnt[e] if cnt[e] > 0 else 0 for e, s in enumerate(psum)]
        layers_out.append({
            "layer_idx": li,
            "n_tokens": total,
            "expert_token_count": cnt,
            "expert_usage_pct": [round(x, 4) for x in usage_pct],
            "avg_gate_prob_when_active": [round(x, 6) for x in avg_prob],
        })

    by_category = {}
    for cat, layer_counts in _state["cat_expert_count"].items():
        cat_layers = []
        for li, cnt in enumerate(layer_counts):
            total_routings = sum(cnt)
            n_tokens = total_routings // max(args.top_k, 1)
            usage_pct = [c / n_tokens * 100 if n_tokens > 0 else 0 for c in cnt]
            cat_layers.append({
                "layer_idx": li,
                "n_tokens": n_tokens,
                "expert_token_count": cnt,
                "expert_usage_pct": [round(x, 4) for x in usage_pct],
            })
        by_category[cat] = cat_layers

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(TZ_CST).isoformat(timespec="seconds"),
        "model": args.model,
        "n_layers": num_layers,
        "n_experts_per_layer": num_experts,
        "top_k": args.top_k,
        "n_prompts": len(prompts),
        "max_tokens_per_prompt": args.max_tokens,
        "categories": sorted(by_category.keys()),
        "layers": layers_out,
        "by_category": by_category,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[done] wrote {out_path}")
    print(f"  layer 0 total tokens: {_state['total_tokens'][0]}")
    print(f"  categories captured:  {len(by_category)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
