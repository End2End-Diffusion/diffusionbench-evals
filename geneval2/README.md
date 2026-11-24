# GenEval2 Evaluator (Minimal)

A minimal, self-contained reimplementation of the [GenEval2](https://github.com/facebookresearch/GenEval2) (Soft-TIFA) evaluation pipeline for text-to-image generation.

## What this repo does

- Provides a lightweight Python package (`geneval2_evaluator`) that scores generated images against 800 compositional prompts using **Soft-TIFA**.
- Soft-TIFA queries a VQA model (**Qwen3-VL-8B-Instruct**) with each prompt's question/answer pairs and scores every question by the probability the model assigns to the correct answer token.
- Reports all four GenEval2 methods per prompt: **Soft-TIFA AM** (arithmetic mean of soft scores, atom-level), **Soft-TIFA GM** (geometric mean, prompt-level), **TIFA** (hard argmax match), and **VQAScore** (single whole-prompt yes/no query). Aggregates also break down by skill and by compositionality (`atom_count`).
- Exposes simple APIs: `load_models`, `evaluate_image`, `evaluate_batch`, `evaluate_pairs`, `summarize_results`, `fetch_metadata`.
- Ships the GenEval2 benchmark (`geneval2_data.jsonl`, 800 prompts) as package data.

## What changed from the original GenEval2

- Extracted evaluation-only code — no image generation.
- Wrapped the script in an importable API matching the other evaluators in this repo (`load_models` / `evaluate_pairs` / `summarize_results`), instead of a CLI over a `{prompt: filepath}` dict.
- Batches up to 8 questions per image with left padding; use `options={"batch_size": 1}` for reference execution order. Compositionality breakdowns use Soft-TIFA GM, as upstream.
- Runs the Qwen3-VL judge on **transformers v5** (the reference targets 4.57).
- Bundled the benchmark data as package data; no external download for the prompts.

## Installation

```bash
uv pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=geneval2"
```

## Quick start

```python
from PIL import Image
from geneval2_evaluator import load_models, fetch_metadata, evaluate_pairs, summarize_results

models = load_models(device="cuda")          # Qwen3-VL-8B-Instruct (use device="auto" to shard)
metadata = fetch_metadata()                   # list of 800 prompt dicts

# Pair each prompt with the image your T2I model generated for it.
prompts = metadata[:8]
images = [Image.open(f"gen/{i}.png") for i in range(len(prompts))]

results = evaluate_pairs(images, prompts, models)
summary = summarize_results(results)
print(summary["vqascore"], summary["tifa"], summary["soft_tifa_am"], summary["soft_tifa_gm"])
print(summary["skill_scores"])   # {'object': ..., 'attribute': ..., 'count': ..., 'position': ..., 'verb': ...}
```

Scores are fractions in `[0, 1]`; multiply by 100 to compare with the numbers in the paper.

## Benchmark structure

Each entry in `geneval2_data.jsonl` is a dict:

- `prompt` — the text prompt
- `atom_count` — compositionality (number of atoms; "and"/"a" do not count)
- `vqa_list` — list of `[question, answer]` pairs, one per atom
- `skills` — skill tag per VQA pair (`object`, `attribute`, `count`, `position`, `verb`)

## Project structure

```
geneval2_evaluator/
  __init__.py          # Public API
  evaluator.py         # Soft-TIFA scoring, evaluation, aggregation
  constants.py         # VQA model id, answer-list helpers, options
  geneval2_data.jsonl  # GenEval2 benchmark (800 prompts)
```

## Acknowledgements

This project is based on [GenEval2](https://github.com/facebookresearch/GenEval2) by Amita Kamath et al. (Meta FAIR, UW, UCLA, AI2).
