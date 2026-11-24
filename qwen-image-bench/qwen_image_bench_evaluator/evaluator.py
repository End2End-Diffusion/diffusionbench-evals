"""
Qwen-Image-Bench metric computation — library API.

This is the importable equivalent of the upstream ``judge.py`` script, with all
argparse / file-IO / Excel reporting stripped out. You bring a judge model (see
``TransformersJudge``) and call :func:`evaluate` (aggregated bench scores) or
:func:`evaluate_batch` (per-sample results).

Input is **(image, prompt)** pairs. Each prompt must be one of the 1000
Qwen-Image-Bench prompts (English or Chinese) — it is looked up in the bundled
prompt set to recover the prompt ID and the evaluation dimensions that apply.
An unknown prompt raises ``KeyError``.

Hierarchy: each level-1 dimension (Quality, Aesthetics, Alignment, Real-world
Fidelity, Creative Generation) is scored by the judge against a checklist; raw
facet scores (0/1/2) map to 0/60/100 and average upward L3 → L2 → L1 → Total.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from PIL import Image

from .checklists import (
    DIM_TO_CHECKLIST,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_dims_by_level1,
)
from .score_utils import (
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

ImageInput = Union[str, Image.Image]
_METADATA_PATH = Path(__file__).parent / "metadata" / "bench_metadata.json"


# --------------------------------------------------------------------------- #
# Metadata / prompt set
# --------------------------------------------------------------------------- #
def load_bench_metadata(path: Optional[str] = None) -> Dict[int, dict]:
    """Load the bundled bench metadata as ``{ID: {prompt_en, prompt_cn, dims_en}}``.

    ``dims_en`` encodes the level1/2/3 dimensions that apply to each prompt, e.g.
    ``"Quality / Realism / Physical Logic; Aesthetics / Color Harmony / Color Harmony"``.
    """
    src = Path(path) if path else _METADATA_PATH
    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {
        rec["ID"]: {
            "prompt_en": rec.get("prompt_en", ""),
            "prompt_cn": rec.get("prompt_cn", ""),
            "dims_en": rec.get("dims_en", ""),
        }
        for rec in records
    }


def build_prompt_index(metadata: Dict[int, dict]) -> Dict[str, int]:
    """Build a ``{prompt_text: ID}`` lookup over both English and Chinese prompts."""
    index: Dict[str, int] = {}
    for row_id, rec in metadata.items():
        for key in ("prompt_en", "prompt_cn"):
            text = (rec.get(key) or "").strip()
            if text:
                index[text] = row_id
    return index


def resolve_prompt_id(prompt: str, prompt_index: Dict[str, int]) -> int:
    """Resolve a prompt string to its bench ID, raising ``KeyError`` if unknown."""
    pid = prompt_index.get(prompt.strip())
    if pid is None:
        raise KeyError(
            f"Prompt is not in the Qwen-Image-Bench prompt set: {prompt[:80]!r}. "
            f"Inputs must use one of the 1000 bundled prompts (English or Chinese)."
        )
    return pid


def load_and_resize_image(img: ImageInput, max_size: int = 1024) -> Image.Image:
    """Open (if a path) and normalize an image the way the judge expects.

    Matches upstream behaviour: convert to RGB and, if either side exceeds
    ``max_size``, resize to a ``max_size`` square with LANCZOS resampling.
    """
    out = img if isinstance(img, Image.Image) else Image.open(img)
    if out.mode != "RGB":
        out = out.convert("RGB")
    if max(out.size) > max_size:
        out = out.resize((max_size, max_size), Image.LANCZOS)
    out.load()
    return out


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #
def _parse_output_to_scores(output_text: str, level1_dim: str):
    """Raw judge text → fixed score_json (or ``None`` on parse failure)."""
    score_json = extract_json_from_response(output_text)
    if score_json is None:
        return None
    return fix_score_json(score_json, level1_dim)


def _build_tasks(images, prompts, row_ids, metadata, resize):
    """Expand (image, prompt, id) rows into per-dimension judge tasks.

    Returns ``(tasks, task_meta, skipped)`` where each task is the dict the judge
    backend consumes and ``task_meta[i] = (row_idx, level1_dim)``. Rows whose
    image fails to load are recorded in ``skipped``.
    """
    tasks, task_meta, skipped = [], [], set()

    for row_idx, (img_in, prompt, row_id) in enumerate(zip(images, prompts, row_ids)):
        try:
            img = load_and_resize_image(img_in, max_size=1024 if resize else 10**9)
        except Exception:
            skipped.add(row_idx)
            continue

        dims_en = metadata[row_id]["dims_en"]
        for level1_dim in parse_dims_by_level1(dims_en):
            if level1_dim not in DIM_TO_CHECKLIST:
                continue
            user_text = USER_PROMPT_TEMPLATE.format(
                prompt=prompt,
                level1_dim=level1_dim,
                format_checklist=DIM_TO_CHECKLIST[level1_dim],
            )
            tasks.append({
                "system_prompt": SYSTEM_PROMPT,
                "user_text": user_text,
                "image": img,
            })
            task_meta.append((row_idx, level1_dim))

    return tasks, task_meta, skipped


def evaluate_batch(
    images: Sequence[ImageInput],
    prompts: Sequence[str],
    judge,
    metadata: Optional[Dict[int, dict]] = None,
    batch_size: int = 24,
    resize: bool = True,
    show_progress: bool = True,
) -> List[dict]:
    """Judge a batch of generated images and return per-sample results.

    Args:
        images: PIL images or file paths, one per sample.
        prompts: the prompt used to generate each image. **Each must be one of
            the bundled Qwen-Image-Bench prompts** (English or Chinese); an
            unknown prompt raises ``KeyError``.
        judge: an object exposing ``generate_batch(items) -> List[str]`` (e.g.
            :class:`TransformersJudge`). ``items`` are
            ``{"system_prompt", "user_text", "image"}`` dicts.
        metadata: ``{ID: {...}}`` from :func:`load_bench_metadata`; defaults to
            the bundled bench metadata.
        batch_size: judge inference batch size.
        resize: apply the upstream RGB / 1024² normalization to each image.
        show_progress: show a tqdm progress bar over judge batches.

    Returns:
        One result dict per input sample, in input order::

            {
              "ID": int,                      # resolved bench prompt ID
              "total": float | None,          # overall score (0–100)
              "dim_scores": {L1: {level1_score, level2_scores, level3_scores}},
              "score_json": {L1: fixed_score_json | None},
              "raw_outputs": {L1: raw judge text},
              "valid": bool,                  # False if image failed to load
            }
    """
    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same length")
    if metadata is None:
        metadata = load_bench_metadata()

    # Resolve every prompt up front so an out-of-set prompt fails loudly before
    # any (expensive) inference runs.
    prompt_index = build_prompt_index(metadata)
    row_ids = [resolve_prompt_id(p, prompt_index) for p in prompts]

    tasks, task_meta, skipped = _build_tasks(images, prompts, row_ids, metadata, resize)

    # Run judge inference in batches.
    outputs: List[str] = []
    batch_starts = range(0, len(tasks), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            batch_starts = tqdm(batch_starts, desc="Judging")
        except ImportError:
            pass
    for i in batch_starts:
        outputs.extend(judge.generate_batch(tasks[i:i + batch_size]))

    # Regroup outputs by source row.
    row_score_json: Dict[int, dict] = defaultdict(dict)
    row_raw_outputs: Dict[int, dict] = defaultdict(dict)
    for (row_idx, level1_dim), output_text in zip(task_meta, outputs):
        row_raw_outputs[row_idx][level1_dim] = output_text
        row_score_json[row_idx][level1_dim] = _parse_output_to_scores(output_text, level1_dim)

    # Build per-sample result records.
    results = []
    for row_idx, row_id in enumerate(row_ids):
        if row_idx in skipped:
            results.append({
                "ID": row_id, "total": None, "dim_scores": {},
                "score_json": {}, "raw_outputs": {}, "valid": False,
            })
            continue

        score_json = row_score_json.get(row_idx, {})
        dim_scores = {
            l1: compute_dimension_score(sj)
            for l1, sj in score_json.items()
            if sj is not None
        }
        results.append({
            "ID": row_id,
            "total": aggregate_total_score(dim_scores),
            "dim_scores": dim_scores,
            "score_json": score_json,
            "raw_outputs": row_raw_outputs.get(row_idx, {}),
            "valid": True,
        })

    return results


def aggregate_bench_scores(results: List[dict]) -> dict:
    """Aggregate per-sample results into bench-level scores.

    Mean across samples (None skipped) for each level-1 dimension, its level-2
    sub-dimensions, and the overall total.

    Returns ``{"level1": {dim: score}, "level2": {dim: {sub: score}}, "total": float}``.
    """
    l1_accum = defaultdict(list)
    l2_accum = defaultdict(lambda: defaultdict(list))
    total_accum = []

    for r in results:
        if r.get("total") is not None:
            total_accum.append(r["total"])
        for l1_dim, dim_data in r.get("dim_scores", {}).items():
            if dim_data["level1_score"] is not None:
                l1_accum[l1_dim].append(dim_data["level1_score"])
            for l2_name, l2_score in dim_data["level2_scores"].items():
                if l2_score is not None:
                    l2_accum[l1_dim][l2_name].append(l2_score)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "level1": {d: _mean(v) for d, v in l1_accum.items()},
        "level2": {d: {l2: _mean(v) for l2, v in l2d.items()} for d, l2d in l2_accum.items()},
        "total": _mean(total_accum),
    }


def evaluate(
    images: Sequence[ImageInput],
    prompts: Sequence[str],
    judge,
    metadata: Optional[Dict[int, dict]] = None,
    batch_size: int = 24,
    resize: bool = True,
    show_progress: bool = True,
) -> dict:
    """Judge a batch and return aggregated bench-level scores.

    Convenience wrapper over :func:`evaluate_batch` + :func:`aggregate_bench_scores`.
    Returns ``{"level1": {...}, "level2": {...}, "total": float}``.
    """
    results = evaluate_batch(
        images, prompts, judge,
        metadata=metadata, batch_size=batch_size,
        resize=resize, show_progress=show_progress,
    )
    return aggregate_bench_scores(results)


# --------------------------------------------------------------------------- #
# Evaluator wrapper
# --------------------------------------------------------------------------- #
class QwenImageBench:
    """Qwen-Image-Bench evaluator — a thin wrapper around a :class:`TransformersJudge`.

    Mirrors the other evaluators in this repo (e.g. ``t2v_metrics.VQAScore``):
    construct it once with a judge model, then call it on ``(images, prompts)``::

        >>> from qwen_image_bench_evaluator import QwenImageBench
        >>> evaluator = QwenImageBench()                    # default judge model
        >>> per_sample = evaluator(images, prompts)         # list of per-sample dicts
        >>> bench      = evaluator.evaluate(images, prompts)  # aggregated bench scores

    Each prompt must be one of the bundled bench prompts (English or Chinese); an
    unknown prompt raises ``KeyError``. The judge model and bench metadata are
    loaded once at construction and reused across calls.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        batch_size: int = 24,
        metadata: Optional[Dict[int, dict]] = None,
        **judge_kwargs,
    ):
        # Lazy import keeps the module-level functions importable without torch.
        from .backends.transformers_backend import TransformersJudge

        if model is None:
            model = TransformersJudge.DEFAULT_MODEL
        self.model_name = model
        self.batch_size = batch_size
        self.metadata = metadata if metadata is not None else load_bench_metadata()
        self.judge = TransformersJudge(model, max_batch_size=batch_size, **judge_kwargs)

    def evaluate_batch(
        self,
        images: Sequence[ImageInput],
        prompts: Sequence[str],
        batch_size: Optional[int] = None,
        resize: bool = True,
        show_progress: bool = True,
    ) -> List[dict]:
        """Per-sample results for each (image, prompt). See :func:`evaluate_batch`."""
        return evaluate_batch(
            images, prompts, self.judge,
            metadata=self.metadata,
            batch_size=batch_size or self.batch_size,
            resize=resize, show_progress=show_progress,
        )

    def evaluate(
        self,
        images: Sequence[ImageInput],
        prompts: Sequence[str],
        batch_size: Optional[int] = None,
        resize: bool = True,
        show_progress: bool = True,
    ) -> dict:
        """Aggregated bench-level scores. See :func:`evaluate`."""
        return aggregate_bench_scores(self.evaluate_batch(
            images, prompts,
            batch_size=batch_size, resize=resize, show_progress=show_progress,
        ))

    # Calling the evaluator returns per-sample results (t2v_metrics-style).
    __call__ = evaluate_batch
