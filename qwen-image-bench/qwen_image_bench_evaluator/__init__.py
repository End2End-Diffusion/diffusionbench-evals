"""
Qwen-Image-Bench Evaluator for text-to-image generation.

A minimal, importable fork of QwenLM/Qwen-Image-Bench that swaps the upstream
ms-swift inference backend for a vanilla ``transformers`` one. The judge is a
Qwen-VL model that scores each generated image against per-dimension checklists
(Quality, Aesthetics, Alignment, Real-world Fidelity, Creative Generation);
facet scores 0/1/2 map to 0/60/100 and average up L3 → L2 → L1 → Total.

Usage:
    >>> from qwen_image_bench_evaluator import QwenImageBench
    >>> from PIL import Image
    >>>
    >>> # 1. Construct the evaluator (loads the judge model once; defaults to
    >>> #    "Qwen/Qwen-Image-Bench", sharded across GPUs). Pass model=<path> to override.
    >>> evaluator = QwenImageBench()
    >>>
    >>> # 2. Prepare your generated images and the prompts used to make them.
    >>> #    Each prompt MUST be one of the 1000 bundled Qwen-Image-Bench prompts
    >>> #    (English or Chinese) — it is looked up to recover the eval dimensions.
    >>> images  = [Image.open("0.png"), Image.open("1.png")]
    >>> prompts = ["<a bundled bench prompt>", "<another bundled bench prompt>"]
    >>>
    >>> # 3a. Per-sample results (calling the evaluator)
    >>> results = evaluator(images, prompts)
    >>> # results[i] = {"ID", "total", "dim_scores", "score_json", "raw_outputs", "valid"}
    >>>
    >>> # 3b. Aggregated bench-level scores
    >>> bench = evaluator.evaluate(images, prompts)
    >>> # bench = {"level1": {...}, "level2": {...}, "total": float}

An out-of-set prompt raises ``KeyError``. The functional API (``evaluate`` /
``evaluate_batch`` taking an explicit ``judge``) and the low-level
``TransformersJudge`` remain available for advanced use.
"""

from .backends.transformers_backend import TransformersJudge
from .evaluator import (
    QwenImageBench,
    aggregate_bench_scores,
    build_prompt_index,
    evaluate,
    evaluate_batch,
    load_and_resize_image,
    load_bench_metadata,
    resolve_prompt_id,
)
from .score_utils import (
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

__all__ = [
    "QwenImageBench",
    "TransformersJudge",
    "evaluate",
    "evaluate_batch",
    "aggregate_bench_scores",
    "load_bench_metadata",
    "build_prompt_index",
    "resolve_prompt_id",
    "load_and_resize_image",
    "compute_dimension_score",
    "aggregate_total_score",
    "extract_json_from_response",
    "fix_score_json",
]
