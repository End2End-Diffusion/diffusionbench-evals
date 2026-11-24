# qwen_image_bench_evaluator

Minimal [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) evaluator for text-to-image generation.

This is a stripped-down fork of [QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) that keeps only the judging + scoring logic and **replaces the ms-swift inference backend with a vanilla `transformers` one** — no `ms-swift` dependency, no CLI scripts, just an importable API.

## What's included

- A Qwen-VL **judge** loaded with `transformers` (`AutoModelForImageTextToText` + `AutoProcessor`)
- The full Qwen-Image-Bench checklist hierarchy and scoring (L3 → L2 → L1 → Total)
- Bundled bench metadata (1000 prompt IDs → evaluation dimensions)

## What's changed (vs upstream)

- **Backend:** `backends/ms_swift_backend.py` (ms-swift `PtEngine`) → `backends/transformers_backend.py` (`TransformersJudge`), same `generate_batch(items)` interface
- **No CLI:** `judge.py` / `compute_scores.py` argparse scripts replaced by `evaluate()` / `evaluate_batch()` library functions (no file IO, no Excel)
- Standardized on `pyproject.toml` + uv; the consuming project chooses the `torch`/`torchvision` build

## Scoring

Each level-1 dimension (Quality, Aesthetics, Alignment, Real-world Fidelity, Creative Generation) is judged against a checklist. Facet ratings map **0 → 0, 1 → 60, 2 → 100** (`N/A` ignored) and average upward: L3 facets → L2 sub-dimensions → L1 dimensions → overall Total. Inference is greedy (`temperature=0`), `repetition_penalty=1.05`, `max_new_tokens=4096`, Qwen3 thinking mode on — mirroring the released dataset's settings.

## Usage

Input is **(image, prompt)** pairs. Each prompt must be one of the 1000 bundled
Qwen-Image-Bench prompts (English or Chinese) — it is looked up to recover the
prompt ID and the evaluation dimensions that apply. An out-of-set prompt raises
`KeyError`.

```python
from qwen_image_bench_evaluator import QwenImageBench
from PIL import Image

# 1. Construct the evaluator — loads the judge model once (default
#    "Qwen/Qwen-Image-Bench", sharded across GPUs). Pass model=<id/path> to override.
evaluator = QwenImageBench()

# 2. Your generated images and the bundled prompts used to generate them
images  = [Image.open("0.png"), Image.open("1.png")]
prompts = ["<a bundled bench prompt>", "<another bundled bench prompt>"]

# 3a. Per-sample results (calling the evaluator)
results = evaluator(images, prompts)
# results[i] = {"ID", "total", "dim_scores", "score_json", "raw_outputs", "valid"}

# 3b. Aggregated bench-level scores
bench = evaluator.evaluate(images, prompts)
# {"level1": {"Quality": ..., ...}, "level2": {...}, "total": 73.4}
```

The functional API (`evaluate` / `evaluate_batch` taking an explicit `judge`) and the
low-level `TransformersJudge` are still exported for advanced use.

The bundled prompt set (`metadata/bench_metadata.json`: 1000 × `{ID, prompt_en,
prompt_cn, dims_en}`) is used by default. To score a custom set, pass your own via
`metadata=load_bench_metadata(path)`.

## Acknowledgements

A minimal fork of [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) by the Qwen team.
