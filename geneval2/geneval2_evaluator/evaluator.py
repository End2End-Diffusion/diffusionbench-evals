"""GenEval2 (Soft-TIFA) evaluation functions.

Soft-TIFA queries a VQA model with each prompt's list of question/answer pairs and
scores every question by the probability mass the model puts on the correct answer
token. Per prompt we report the arithmetic mean (Soft-TIFA AM, atom-level) and the
geometric mean (Soft-TIFA GM, prompt-level) of those soft scores; the hard argmax
match gives the original TIFA score. See https://github.com/facebookresearch/GenEval2.
"""

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.stats import gmean
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .constants import (
    DEFAULT_VQA_MODEL, DEFAULT_OPTIONS, YES_ANSWERS, VQASCORE_TEMPLATE, build_answer_list,
)


def load_models(device="cuda", model_name=None):
    """Load the Qwen3-VL VQA judge.

    Args:
        device: "cuda"/"cpu" (model is moved there), or "auto" to shard across GPUs.
        model_name: HuggingFace id of the VQA judge (defaults to Qwen3-VL-8B-Instruct).
    """
    model_name = model_name or DEFAULT_VQA_MODEL
    if device == "auto":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto"
        ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    return {"model": model, "processor": processor, "model_name": model_name}


@torch.no_grad()
def _query_batch(models, image, texts, answer_lists):
    """Score matched questions in one left-padded generation step."""
    processor, model = models["processor"], models["model"]
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]}] for text in texts]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        padding=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    probs = torch.softmax(outputs.scores[0], dim=-1).cpu()
    predictions = processor.batch_decode(probs.argmax(-1).unsqueeze(-1))
    results = []
    for pred, row, answers in zip(predictions, probs, answer_lists):
        # Preserve the reference's token spellings and probability summation.
        soft = sum(row[processor.tokenizer.encode(ans)[0]].item() for ans in answers)
        results.append((pred, soft))
    return results


def _query(models, image, text, answer_list):
    return _query_batch(models, image, [text], [answer_list])[0]


def evaluate_image(image, metadata, models, options=None):
    """Run Soft-TIFA on a single (image, metadata) pair.

    Args:
        image: PIL.Image.Image or np.ndarray.
        metadata: dict with keys `prompt`, `vqa_list`, `skills`, `atom_count`.
        models: dict returned by load_models.
        options: overrides for DEFAULT_OPTIONS.

    Returns:
        result dict with per-question soft scores, hard matches, and the per-prompt
        Soft-TIFA AM / Soft-TIFA GM / TIFA aggregates.
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    suffix = options.get("question_suffix", "")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    vqa_list = metadata.get("vqa_list", [])
    prompt = metadata.get("prompt", "")
    answer_lists = [build_answer_list(q, a) for q, a in vqa_list] + [YES_ANSWERS]
    texts = [f"{question} {suffix}" for question, _ in vqa_list]
    texts.append(VQASCORE_TEMPLATE.format(prompt))
    batch_size = int(options["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    answers = []
    for start in range(0, len(texts), batch_size):
        answers.extend(_query_batch(models, image, texts[start:start + batch_size],
                                    answer_lists[start:start + batch_size]))
    soft_scores = [soft for _, soft in answers[:-1]]
    hard_scores = [int(pred.lower() in accepted)
                   for (pred, _), accepted in zip(answers[:-1], answer_lists[:-1])]
    vqascore = answers[-1][1]

    return {
        "prompt": prompt,
        "atom_count": metadata.get("atom_count", len(vqa_list)),
        "skills": json.dumps(metadata.get("skills", [])),
        "vqa_soft": json.dumps(soft_scores),
        "vqa_hard": json.dumps(hard_scores),
        "vqascore": vqascore,
        "soft_tifa_am": float(np.mean(soft_scores)) if soft_scores else 0.0,
        "soft_tifa_gm": float(gmean(soft_scores)) if soft_scores else 0.0,
        "tifa": float(np.mean(hard_scores)) if hard_scores else 0.0,
        "metadata": json.dumps(metadata),
    }


def evaluate_batch(images, metadata_list, models, options=None, show_progress=True):
    """Evaluate grouped images (list-of-lists; multiple samples per prompt allowed).

    Args:
        images: list (per prompt) of lists of images.
        metadata_list: list of metadata dicts (same length as images).
        models: dict returned by load_models.

    Returns:
        list of result dicts with prompt_idx / sample_idx plus the evaluate_image fields.
    """
    from tqdm import tqdm

    if len(images) != len(metadata_list):
        raise ValueError(f"images and metadata_list must have the same length, got {len(images)} vs {len(metadata_list)}")

    total_images = sum(len(img_list) for img_list in images)

    def iterate_images():
        for prompt_idx, (image_list, metadata) in enumerate(zip(images, metadata_list)):
            for sample_idx, image in enumerate(image_list):
                yield prompt_idx, sample_idx, image, metadata

    iterator = iterate_images()
    if show_progress:
        iterator = tqdm(iterator, total=total_images, desc="Evaluating")

    results = []
    for prompt_idx, sample_idx, image, metadata in iterator:
        result = evaluate_image(image, metadata, models, options)
        result = {"prompt_idx": prompt_idx, "sample_idx": sample_idx, **result}
        results.append(result)
    return results


def evaluate_pairs(images, metadata_list, models, options=None, show_progress=True):
    """Evaluate a list of (image, metadata) pairs (1 image per metadata)."""
    if len(images) != len(metadata_list):
        raise ValueError(f"images and metadata_list must have the same length, got {len(images)} vs {len(metadata_list)}")
    grouped_images = [[img] for img in images]
    return evaluate_batch(grouped_images, metadata_list, models=models, options=options, show_progress=show_progress)


def summarize_results(results):
    """Aggregate per-prompt results into benchmark-level Soft-TIFA / TIFA scores.

    Returns a dict with:
      - total_prompts
      - vqascore / tifa / soft_tifa_am / soft_tifa_gm  (mean over prompts, fraction in [0, 1])
      - skill_scores: {skill: mean soft score over all questions of that skill}
      - atom_scores: {atom_count: mean Soft-TIFA GM over prompts of that compositionality}
      - overall_score: alias of soft_tifa_am (the headline atom-level metric)
    """
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if df.empty:
        return {
            "total_prompts": 0,
            "vqascore": 0.0, "tifa": 0.0, "soft_tifa_am": 0.0, "soft_tifa_gm": 0.0,
            "skill_scores": {}, "atom_scores": {}, "overall_score": 0.0,
        }

    vqascore = float(df["vqascore"].mean()) if "vqascore" in df.columns else 0.0
    soft_tifa_am = float(df["soft_tifa_am"].mean())
    soft_tifa_gm = float(df["soft_tifa_gm"].mean())
    tifa = float(df["tifa"].mean())

    # Per-skill: flatten every question's soft score against its skill.
    skill_totals, skill_counts = {}, {}
    for _, row in df.iterrows():
        soft = json.loads(row["vqa_soft"])
        skills = json.loads(row["skills"])
        for s, skill in zip(soft, skills):
            skill_totals[skill] = skill_totals.get(skill, 0.0) + s
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    skill_scores = {k: skill_totals[k] / skill_counts[k] for k in sorted(skill_totals)}

    # Official per_atomicity_analysis uses the per-prompt geometric mean.
    atom_scores = {
        int(atom): float(g["soft_tifa_gm"].mean())
        for atom, g in df.groupby("atom_count")
    }

    return {
        "total_prompts": int(len(df)),
        "vqascore": vqascore,
        "tifa": tifa,
        "soft_tifa_am": soft_tifa_am,
        "soft_tifa_gm": soft_tifa_gm,
        "skill_scores": skill_scores,
        "atom_scores": atom_scores,
        "overall_score": soft_tifa_am,
    }


def fetch_metadata():
    """Load the bundled GenEval2 benchmark (list of 800 prompt dicts)."""
    path = os.path.join(os.path.dirname(__file__), "geneval2_data.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
