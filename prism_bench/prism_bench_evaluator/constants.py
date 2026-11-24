"""Constants for the PRISM-Bench Qwen and GPT-4.1 judges."""

import json
import os

# The 7 PRISM-Bench tracks.
TRACKS = [
    "imagination", "entity", "text_rendering", "style",
    "affection", "composition", "long_text",
]

# Each image is judged under two rubrics.
EVAL_TYPES = ["alignment", "aesthetic"]

# Registry of convenient local judge names -> HF model id. Pass any of these short names
# (or a full HF / local path) to PrismBenchEvaluator. The model class is resolved
# automatically by AutoModelForImageTextToText, so new Qwen-VL families work without
# code changes (just add an entry here). Mirrors the t2v_metrics model naming.
HF_JUDGE_MODELS = {
    # Qwen2.5-VL -- the PRISM-Bench reference judge (72B is the leaderboard model).
    "qwen2.5-vl-3b":  "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    # Qwen3-VL
    "qwen3-vl-2b":  "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b":  "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b":  "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3-vl-30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",  # MoE (auto-resolved)
    "qwen3-vl-235b-a22b": "Qwen/Qwen3-VL-235B-A22B-Instruct",  # MoE
    # Qwen3.5 / Qwen3.6 (same paths as t2v_metrics)
    "qwen3.5-4b":  "Qwen/Qwen3.5-4B",
    "qwen3.5-9b":  "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",  # MoE
    "qwen3.5-122b-a10b": "Qwen/Qwen3.5-122B-A10B",  # MoE
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",  # MoE
}

# OpenAI-compatible API judges. Both spellings are accepted for convenience, while the
# value stays pinned to the snapshot used by PRISM-Bench's reference eval_gpt41.py.
OPENAI_JUDGE_MODELS = {
    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-2025-04-14": "gpt-4.1-2025-04-14",
    "gpt5.5": "gpt-5.5-2026-04-23",
    "gpt-5.5": "gpt-5.5-2026-04-23",
    "gpt-5.5-2026-04-23": "gpt-5.5-2026-04-23",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-sol": "gpt-5.6-sol",
}

JUDGE_MODELS = {**HF_JUDGE_MODELS, **OPENAI_JUDGE_MODELS}

# Default judge: the PRISM-Bench leaderboard model (paper: Qwen2.5-VL-72B).
DEFAULT_JUDGE_MODEL = "qwen2.5-vl-72b"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4.1-2025-04-14"

DEFAULT_OPTIONS = {
    "max_new_tokens": 1024,
    # Qwen2.5-VL dynamic-resolution bounds, matching the reference.
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    # Batched for throughput; use 1 for reference execution order.
    # Generated verdicts can change with batching; no zero-mean guarantee.
    "batch_size": 8,
}

# Reference GPT-4.1 payload settings; batch_size bounds concurrent requests.
# Set batch_size=1 for the reference execution order.
DEFAULT_OPENAI_OPTIONS = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "batch_size": 8,
}


def load_judge_templates():
    """Load the alignment (per-track) + aesthetic judge prompt templates."""
    with open(os.path.join(os.path.dirname(__file__), "judge_templates.json")) as f:
        return json.load(f)
