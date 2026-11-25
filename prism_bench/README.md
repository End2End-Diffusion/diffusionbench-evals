# PRISM-Bench Evaluator (Minimal)

A minimal, self-contained reimplementation of the [PRISM-Bench](https://github.com/rongyaofang/prism-bench) evaluation pipeline for text-to-image generation, with both the open-source **Qwen2.5-VL** judge and the official **GPT-4.1** API judge.

## What this repo does

- Provides a lightweight Python package (`prism_bench_evaluator`) that scores generated images across PRISM-Bench's **7 tracks** (`imagination`, `entity`, `text_rendering`, `style`, `affection`, `composition`, `long_text`), 100 prompts each.
- Each image is judged under **two rubrics**: `alignment` (track-specific) and `aesthetic` (shared). A **Qwen-VL or GPT-4.1** judge reads the image + prompt and emits JSON `{justification, score}` with score in 0–10.
- Reports per-track alignment / aesthetic / combined averages plus overall alignment, aesthetic, and final-total scores.
- Exposes one self-contained wrapper class, `PrismBenchEvaluator` (judge backend + templates + methods), with `evaluate_image` / `evaluate_batch` / `evaluate_pairs` and the static `fetch_metadata` / `summarize_results` helpers.
- Ships the English PRISM-Bench prompts and the judge templates as package data.

## What changed from the original PRISM-Bench

- Extracted evaluation-only code — no image generation, no CLI.
- Wrapped it in an importable API matching the other evaluators in this repo (`load_models` / `evaluate_pairs` / `summarize_results`).
- **Batched, left-padded judge inference** — the reference runs one image at a time; this batches (`batch_size`) with `padding_side="left"`, which is required for correct decoder-only batched generation.
- The judge **prompt templates are byte-identical** to the reference scripts.
- The GPT-4.1 backend matches `eval_gpt41.py`: Chat Completions, `gpt-4.1-2025-04-14`, text then base64 image input, `max_tokens=4096`, temperature 0, top-p 1, and seed 42.
- English only (the `--zh` track is not bundled).

## Installation

```bash
uv pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=prism_bench"
```

## Quick start

```python
from PIL import Image
from prism_bench_evaluator import PrismBenchEvaluator

# Qwen2.5-VL-72B is the leaderboard judge; pass a smaller model_name
# (e.g. "Qwen/Qwen2.5-VL-7B-Instruct") to fit a single GPU.
judge = PrismBenchEvaluator(device="cuda")   # default qwen2.5-vl-72b; or model_name="qwen3-vl-8b" etc.

# Simplest: just (image, prompt) pairs of the PRISM-Bench prompts. The track is looked up
# from the prompt, and both rubrics (alignment + aesthetic) are scored per image.
prompts = list(PrismBenchEvaluator.prompt_to_track())   # the 700 PRISM prompts (or your own list)
images = [Image.open(f"images/{i}.png") for i in range(len(prompts))]

results = judge.evaluate(images, prompts, batch_size=8)   # 2 rows per image (alignment + aesthetic)
summary = PrismBenchEvaluator.summarize_results(results)
print(summary["final_total_average"], summary["overall_alignment"], summary["overall_aesthetic"])
print(summary["combined_by_track"])
```

Scores are on PRISM-Bench's 0–10 scale. If your prompts aren't the bundled PRISM ones
(so the track can't be looked up), build metadata yourself and call
`evaluate_pairs(images, metadata)`, where each entry is
`{"track": ..., "eval_type": "alignment"|"aesthetic", "prompt": ...}`.

### GPT-4.1 judge

Pass either `gpt-4.1` or `gpt4.1`; both select the pinned snapshot from the reference script. The standard OpenAI endpoint is used when `base_url` is omitted.

```python
import os

from prism_bench_evaluator import PrismBenchEvaluator

judge = PrismBenchEvaluator(
    model_name="gpt-4.1",
    api_key=os.environ["OPENAI_API_KEY"],
    # base_url="https://your-openai-compatible-endpoint/v1",  # optional
)

results = judge.evaluate(images, prompts)
summary = judge.summarize_results(results)
```

For a custom OpenAI-compatible model name, pass `backend="openai"`. A preconfigured compatible client can also be supplied with `client=...`. API requests use bounded concurrency (default 8), preserving result order and per-item payloads. Set `options={"batch_size": 1}` for sequential requests.

### Judge models

Local judges are loaded via `AutoModelForImageTextToText`, so any Qwen-VL family works — pass a key from `HF_JUDGE_MODELS` (or a full HF / local path) as `model_name`: `qwen2.5-vl-{3b,7b,32b,72b}`, `qwen3-vl-{2b,4b,8b,32b,30b-a3b}`, `qwen3.5-{4b,9b,27b,35b-a3b}`, `qwen3.6-{27b,35b-a3b}`, `qwen3.8-27b`. The default remains **Qwen2.5-VL-72B** for backward compatibility. GPT-4.1 is available through `gpt-4.1`, `gpt4.1`, or the pinned `gpt-4.1-2025-04-14` model name.

### Reference protocol

GPT-4.1 uses the official pinned snapshot, rubrics and request settings. Qwen2.5-VL uses the reference vision preprocessing, input dtype and generation defaults. Local judging defaults to batches of 8; use `batch_size=1` for reference execution order. Batching and alternative judges require separate numerical validation. Neither a seed nor batch size 1 guarantees identical results across hardware or API runs.

Only numeric JSON scores are counted. Missing or malformed verdicts remain missing, never zero. Save returned per-item `raw` responses and check `total_parsed / total_judged` before comparing runs; this library does not implement durable resume.

## Benchmark structure

```
prism_bench_evaluator/
  __init__.py            # Public API
  evaluator.py           # Qwen/GPT-4.1 judges, scoring, aggregation
  constants.py           # tracks, judge model ids, options
  judge_templates.json   # alignment (per-track) + aesthetic judge prompts
  prism_bench_data.jsonl # English benchmark: one entry per prompt {track, index, prompt} (700)
```

## Acknowledgements

This project is based on [PRISM-Bench](https://github.com/rongyaofang/prism-bench) (FLUX-Reason-6M & PRISM-Bench, Fang et al.).
