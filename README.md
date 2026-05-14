# Lynn Distill Toolkit

> Training/eval/ship pipeline used for **Lynn-V4-Pro-Distill-Qwen-35B-A3B** (and V Flash sibling, V Pro-27B pruning roadmap). Released 2026-05-13.

## What's in here

```
lynn-distill-toolkit/
├── eval/
│   ├── four_gate_eval.py          ⭐ B+ schema lynn-4gate-v1: V4 style + V8 regression + V9 holdout + reference parity
│   ├── differential_sanity.py     LoRA adapter active vs base — logits diff > 0.01 hard gate
│   ├── quant_verify.py            Quantized variant output similarity vs BF16 reference
│   └── prompts/
│       └── v4_distill_verify_35.jsonl  ⭐ 35-prompt public eval set (research/math/tool/general)
│
├── pipeline/
│   ├── peft_merge.py              Multimodal-aware LoRA merge with coherence check
│   ├── post_quant_pack.sh         ⭐ Ship gate wrapper — fixes the "v8-RTN missing tokenizer" bug
│   ├── ms_push_variant.py         Generic ModelScope upload script
│   ├── ms_push_reports_bf16.py    Reports/ subdirectory push for BF16 repo
│   └── start_q4km_after_v8_safe.sh  Q4_K_M 2-step quantization with throttle/disk-safe
│
└── pruning/
    └── activation_profile.py      27B pruning Phase 1: activation profile across 256 experts
```

## Why this toolkit exists

Lynn-V4-Pro-Distill is shipped across **3 quantization variants × 2 platforms (HF + MS)**:

- BF16 merged (canonical, 65.4 GB)
- NVFP4 v8-RTN compressed-tensors (W4A4, 21 GB, Blackwell GPU)
- Q4_K_M GGUF (llama.cpp / Ollama, 22 GB)

Each variant needs:
- **Eval**: same 4-gate framework, comparable scores
- **Sanity**: differential check that LoRA actually does something
- **Quant verify**: quantized output ≥ 70% similar to BF16 reference (chrF + ROUGE-L composite)
- **Ship gate**: file completeness, tokenizer loadable, index consistent

This toolkit standardizes all of the above so V Flash / V5 / V Pro-27B pruning can re-run identical gates without reinventing them.

## Quick start (4-gate eval on Lynn-V4-Pro)

```bash
git clone https://github.com/MerkyorLynn/lynn-distill-toolkit
cd lynn-distill-toolkit

# 4-gate eval against the public 35-prompt set
python eval/four_gate_eval.py \
  --model nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B \
  --prompts eval/prompts/v4_distill_verify_35.jsonl \
  --output reports/my_4gate_results.json
```

Expected verdict for Lynn-V4-Pro: `NET_WIN, net_score +40.00pp` (see [reports/](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B/tree/main/reports) in the model repo).

### Supplementary V8/V9 v4 — 3-way serving comparison (2026-05-14)

After ship we ran a 75-question Lynn daily-mix eval (`v8_tool_calling` 15 + `v9_holdout` 8 + `v9_probe_expanded` 52) against three serving configurations of Lynn-V4-Pro. The grader uses string-match + N-of-M token fallback + LaTeX normalization (v4 grader, [details](eval/)).

| Suite | NVFP4 v8-RTN nothink (production) | NVFP4 v8-RTN thinking=True | Q4_K_M (GGUF default thinking) |
|---|---|---|---|
| V8 tool calling | **15/15 (100.0%)** | 14/15 (93.3%) | 15/15 (100.0%) |
| V9 holdout | 5/8 (62.5%) | 6/8 (75.0%) | **8/8 (100.0%)** |
| V9 expanded | 46/52 (88.5%) | 49/52 (94.2%) | **51/52 (98.1%)** |
| **TOTAL** | **66/75 (88.0%)** | **69/75 (92.0%)** | **74/75 (98.7%)** |

⚠️ **Q4_K_M's 6.7pp lead over NVFP4-thinking is chat_template wrap, not quantization quality.** Same Lynn V4-Pro weights, same prompts, same `temperature=0`, same 4096-token max output. GGUF embeds a more concise thinking template than SGLang's `chat_template.jinja`; within budget, Q4 reaches the answer while NVFP4 hits the ceiling mid-derivation.

Sampled cases where Q4 PASS / NVFP4-think FAIL:

| qid | NVFP4 think tokens | Q4 tokens | What happened |
|---|---|---|---|
| v9_002 (gold 540) | **4096 ⚠️ truncated** | 3868 | NVFP4 stuck on `324cosθ-432sinθ`, never computed `sqrt(324²+432²)` |
| v9_008 (gold 0.48 eV) | **4096 ⚠️ truncated** | **668** ✓ | NVFP4 unwinding `hc/λ`; Q4 reached `K_max=0.4816 eV` cleanly |
| v9p_aime_001 (gold 468) | **4096 ⚠️ truncated** | **1796** ✓ | NVFP4 mid-coordinate; Q4 reached area=468 |
| v9p_fin_005 (gold 957.88) | **4096 ⚠️ truncated** | **929** ✓ | NVFP4 stuck verifying; Q4 computed bond price |

Raw JSONs: [`evaluation/`](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN/tree/main/evaluation) on the v8-RTN HF repo.

### Throughput — TPS suite (Spark GB10 sm_121, 2026-05-14)

| Configuration | Single TPS (avg) | TTFT (avg) | N=4 aggregate | N=16 aggregate |
|---|---|---|---|---|
| NVFP4 v8-RTN @ SGLang dev-cu13 (production, no MTP) | **58.7 tok/s** | **81 ms** | **220 tok/s** | **599 tok/s** |
| NVFP4 v8-RTN @ SGLang + MTP NEXTN | 28.4 tok/s ⚠️ -52% | 175 ms | 81 tok/s | 235 tok/s |
| Q4_K_M @ llama.cpp sm_121 (--parallel 16) | **74.9 tok/s** | 122 ms | 89 tok/s ⚠️ | 252 tok/s |

Notes:
- **NVFP4 wins multi-user serving by 2-3x at N≥4** (SGLang continuous batching) — use for Lynn brain / shared inference
- **Q4_K_M wins single-stream by 27%** but `--parallel` is slot-multiplexing (not true continuous batching); N=4 aggregate regresses below N=2 — use for consumer single-user
- **MTP NEXTN slows V4-Pro by 50-60% across all metrics** because the model was distilled without MTP head weights — drafts are rejected. Production config = no MTP.
- Long-context: NVFP4 32K input → 48.4 tok/s ✓; Q4 32K input fails with HTTP 400 (llama.cpp `ctx-size` handling differs)
- `首先...` Chinese thinking-prefix injection has **no perf impact** (directional only) — same TPS as baseline on both backends

## Ship gate — preventing silent failure

The `pipeline/post_quant_pack.sh` wrapper exists because of a **real ship-blocker** we hit:

> R6000's `v8-rtn-llmcompressor.py` (from NVFP4 toolkit) produces only 5 files in its output dir. Tokenizer (`tokenizer.json`, `tokenizer_config.json`), index JSON, and a few other files are **silently missing**. This makes SGLang's `AutoProcessor` fail at server startup — users download the model and it doesn't load.

The wrapper post-processes quantization output by copying these required files from the BF16 source dir, then runs **3 sanity gates**:

1. File completeness (model.safetensors / config / tokenizer / chat_template / generation_config)
2. Tokenizer loadable via `transformers.AutoTokenizer.from_pretrained`
3. `model.safetensors.index.json` references consistent (if multi-shard)

Any gate fail → exit 1, ship blocked. See script for details.

## Lynn V4 Distill series — model repos

| Variant | HuggingFace | ModelScope |
|---|---|---|
| BF16 merged | [nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B) | [Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B](https://modelscope.cn/models/Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B) |
| NVFP4 v8-RTN | [nerkyor/...-NVFP4-v8-RTN](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN) | [Merkyor/...-NVFP4-v8-RTN](https://modelscope.cn/models/Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN) |
| Q4_K_M GGUF | [nerkyor/...-Q4_K_M](https://huggingface.co/nerkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-Q4_K_M) | [Merkyor/...-Q4_K_M](https://modelscope.cn/models/Merkyor/Lynn-V4-Pro-Distill-Qwen-35B-A3B-Q4_K_M) |

## Related toolkits

- **[MerkyorLynn/qwen3.6-nvfp4-toolkit](https://github.com/MerkyorLynn/qwen3.6-nvfp4-toolkit)** — NVFP4 quantization recipe (v8-RTN compressed-tensors + modelopt_fp4 calibration). Used to produce the v8-RTN variant referenced above.

## Path conventions

Scripts default to R6000 / A100 paths (`/root/autodl-tmp/...`, `/mnt/data3/...`) since that's where the pipeline was developed. You'll need to either:

1. **Edit path constants** at top of each script (most scripts have them clearly marked), or
2. **Set environment variables** where supported (e.g., `MS_TOKEN` for ModelScope SDK access)

This is a research/operations toolkit, not a polished library — paths are documented but not parameterized everywhere yet. PRs welcome.

## License

MIT — see [LICENSE](./LICENSE). Based on prior work under Apache 2.0; full attribution in [NOTICE](./NOTICE) (R1-Distill style Path B).

## Citation

```bibtex
@misc{lynn-distill-toolkit-2026,
  title = {Lynn Distill Toolkit: V4-Pro Distill Pipeline (eval/sanity/ship/pruning)},
  author = {Lynn / MerkyorLynn},
  year = {2026},
  url = {https://github.com/MerkyorLynn/lynn-distill-toolkit}
}
```

## Background reading

- 📝 [Lynn-V4-Pro-Distill 发布日志 (5/13)](https://zhuanlan.zhihu.com/p/2036443846322680848) — V4-Pro Distill Phase 2 + Phase 3.2 + NVFP4 三合一长文
- (link to 5/13 standalone release post will go here when published)

---

**5/14 - 5/17**: V4 Flash sibling model + V Pro-27B pruning work runs on the same toolkit. Watch this space.
