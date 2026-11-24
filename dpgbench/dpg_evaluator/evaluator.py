"""
DPG-Bench metric computation functions.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Callable

from PIL import Image


def compute_vqa_scores(
    qid2question: Dict[int, str],
    qid2dependency: Dict[int, List[int]],
    vqa_fn: Callable[[List[Image.Image], List[str]], List[str]],
    image: Image.Image,
    question_batch_size: int = 16,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, bool]]:
    """
    Compute VQA scores for all questions on an image.

    Returns:
        Tuple of (raw_scores, adjusted_scores, validity)
    """
    raw_scores = {}

    # Fixed-length FP32 batches preserve the reference input representation.
    # Set question_batch_size=1 for the original execution order.
    if question_batch_size < 1:
        raise ValueError("question_batch_size must be positive")
    questions = list(qid2question.items())
    for start in range(0, len(questions), question_batch_size):
        batch = questions[start:start + question_batch_size]
        answers = vqa_fn([image] * len(batch), [q for _, q in batch])
        if len(answers) != len(batch):
            raise ValueError("VQA must return one answer per question")
        for (qid, _), answer in zip(batch, answers):
            raw_scores[qid] = float(answer == 'yes')

    adjusted_scores = raw_scores.copy()
    validity = {}

    for qid, parent_ids in qid2dependency.items():
        any_parent_no = any(
            adjusted_scores[pid] == 0
            for pid in parent_ids
            if pid != 0
        )
        if any_parent_no:
            adjusted_scores[qid] = 0.0
            validity[qid] = False
        else:
            validity[qid] = True

    return raw_scores, adjusted_scores, validity


def compute_sample_score(
    image: Image.Image,
    prompt: str,
    prompt2id: Dict[str, str],
    question_dict: Dict[str, Dict],
    vqa_fn: Callable[[List[Image.Image], List[str]], List[str]],
    question_batch_size: int = 16,
) -> Tuple[float, Dict[int, str], Dict[int, float]]:
    """
    Compute DPG score for a single (image, prompt) pair.

    Returns:
        Tuple of (score, qid2tuple, raw_scores)
    """
    item_id = prompt2id.get(prompt)
    if item_id is None:
        raise KeyError(f"No item_id found for prompt: {prompt[:50]}...")

    metadata = question_dict.get(item_id)
    if metadata is None:
        raise KeyError(f"No metadata found for item_id: {item_id}")

    qid2tuple = metadata['qid2tuple']
    qid2question = metadata['qid2question']
    qid2dependency = metadata['qid2dependency']

    raw_scores, adjusted_scores, _ = compute_vqa_scores(
        qid2question, qid2dependency, vqa_fn, image, question_batch_size
    )
    score = sum(adjusted_scores.values()) / len(adjusted_scores)

    return score, qid2tuple, raw_scores


def evaluate_batch(
    images: List[Image.Image],
    prompts: List[str],
    prompt2id: Dict[str, str],
    question_dict: Dict[str, Dict],
    vqa_fn: Callable[[List[Image.Image], List[str]], List[str]],
    question_batch_size: int = 16,
) -> List[Dict]:
    """
    Evaluate a batch of (image, prompt) pairs and return per-sample results.

    Args:
        images: List of PIL images
        prompts: List of prompts (same length as images)
        prompt2id: Mapping from prompt text to item_id
        question_dict: Metadata dict from load_dpg_metadata()
        vqa_fn: Batch function (images, questions) -> answers

    Returns:
        List of dicts, each containing: score, qid2tuple, raw_scores
    """
    assert len(images) == len(prompts), "images and prompts must have same length"

    results = []
    for image, prompt in zip(images, prompts):
        score, qid2tuple, raw_scores = compute_sample_score(
            image, prompt, prompt2id, question_dict, vqa_fn, question_batch_size
        )
        results.append({
            'score': score,
            'qid2tuple': qid2tuple,
            'raw_scores': raw_scores,
        })

    return results


def aggregate_results(
    results: List[Dict],
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Aggregate per-sample results into category scores.

    Args:
        results: List of result dicts from evaluate_batch

    Returns:
        Tuple of (mean_score, l1_category_scores, l2_category_scores)
    """
    if not results:
        return 0.0, {}, {}

    all_scores = [r['score'] for r in results]
    mean_score = sum(all_scores) / len(all_scores)

    category2scores = defaultdict(list)
    for r in results:
        for qid, tuple_str in r['qid2tuple'].items():
            category = tuple_str.split('(')[0].strip()
            category2scores[category].append(r['raw_scores'][qid])

    # Aggregate to L1 categories
    l1_scores = defaultdict(list)
    for category, scores in category2scores.items():
        l1_cat = category.split('-')[0].strip()
        l1_scores[l1_cat].extend(scores)

    l1_category_scores = {k: sum(v) / len(v) for k, v in l1_scores.items()}
    l2_category_scores = {k: sum(v) / len(v) for k, v in category2scores.items()}

    return mean_score, l1_category_scores, l2_category_scores


def evaluate(
    images: List[Image.Image],
    prompts: List[str],
    prompt2id: Dict[str, str],
    question_dict: Dict[str, Dict],
    vqa_fn: Callable[[List[Image.Image], List[str]], List[str]],
    question_batch_size: int = 16,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Evaluate a list of (image, prompt) pairs.

    Args:
        images: List of PIL images
        prompts: List of prompts (same length as images)
        prompt2id: Mapping from prompt text to item_id
        question_dict: Metadata dict from load_dpg_metadata()
        vqa_fn: Batch function (images, questions) -> answers

    Returns:
        Tuple of (mean_score, l1_category_scores, l2_category_scores)
    """
    results = evaluate_batch(images, prompts, prompt2id, question_dict, vqa_fn, question_batch_size)
    return aggregate_results(results)
