# t2v_metrics

Minimal [VQAScore](https://github.com/linzhiqiu/t2v_metrics) implementation for text-to-image evaluation.

This is a stripped-down fork of [linzhiqiu/t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) that keeps only what's needed to compute VQAScore — no video support, no extra metrics, no heavy dependencies.

## What's included

- **VQAScore** computation for image-text alignment
- Batched forward pass support for efficient evaluation
- Model families:
  - **CLIP-FlanT5**: `clip-flant5-xxl`, `clip-flant5-xl` (+ prompting-ablation variants `clip-flant5-xxl-no-system`, `clip-flant5-xxl-no-system-no-user`)
  - **Qwen2.5-VL**: `qwen2.5-vl-3b`, `qwen2.5-vl-7b`, `qwen2.5-vl-32b`, `qwen2.5-vl-72b`
  - **Qwen3-VL**: `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b`, `qwen3-vl-32b`, `qwen3-vl-30b-a3b` (MoE), `qwen3-vl-235b-a22b` (MoE)
  - **Qwen3.5**: `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-27b`, `qwen3.5-35b-a3b` (MoE), `qwen3.5-122b-a10b` (MoE)
  - **Qwen3.6**: `qwen3.6-27b`, `qwen3.6-35b-a3b` (MoE)

The largest BF16 checkpoints (`qwen3-vl-235b-a22b` and `qwen3.5-122b-a10b`) are automatically distributed across the available devices with `device_map="auto"`. Smaller checkpoints are loaded directly onto the requested `device`.

### Protocol and batching

The GenAI-Bench reference uses `clip-flant5-xxl`, the default question template and answer `Yes`. Other judges define different metrics. This fork returns matched-pair scores, unlike upstream's cross-product API. PIL inputs are converted to RGB, as file inputs are upstream.

Batched BF16 inference can change scores. No zero-mean-error or exact-equivalence guarantee is made; use the same batch size and evaluator revision across comparisons.

## What's removed (vs upstream)

- Video scoring (CLIPScore, PickScore, ImageReward, etc.)
- LLaVA-based models
- Dataset downloading / GenAI-Bench evaluation harness
- All non-VQAScore metrics

## Installation

```bash
pip install "git+https://github.com/G-REPA/nanogen-evals-dev.git#subdirectory=t2v_metrics"
```

## Usage

```python
import t2v_metrics

score_func = t2v_metrics.VQAScore(model="clip-flant5-xxl")

# Single pair
score = score_func(images="image.png", texts="a cat sitting on a couch")

# Batch of paired images and texts
scores = score_func(
    images=["img1.png", "img2.png"],
    texts=["a cat sitting on a couch", "a dog in the park"],
)
```

## Acknowledgements

This project is a minimal fork of [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) by [Zhiqiu Lin](https://github.com/linzhiqiu).
