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
