"""
Quick sample-wise scoring over a folder of ``<name>.png`` + ``<name>.txt`` pairs.

Each image is scored against its sibling ``.txt`` prompt (which must be one of the
bundled Qwen-Image-Bench prompts). Writes two files next to the images:

  - ``sample_scores.csv``  : one row per image (file, ID, total, per-L1 dimension)
  - ``sample_scores.jsonl``: full per-sample detail (dim/level2/level3 scores)

Usage:
    python score_folder.py --images images/gpt-image-2 --model Qwen/Qwen-Image-Bench
    python score_folder.py --images images/gpt-image-2 --model <local_path> --limit 20
"""

import argparse
import csv
import glob
import json
import os

from qwen_image_bench_evaluator import QwenImageBench, TransformersJudge, aggregate_bench_scores

L1_DIMS = ["Quality", "Aesthetics", "Alignment", "Real-world Fidelity", "Creative Generation"]
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else ""


def load_pairs(folder):
    """Collect (name, image_path, prompt) for every image with a sibling .txt."""
    names, images, prompts, missing = [], [], [], []
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(os.path.join(folder, ext)))
    for img_path in paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            missing.append(os.path.basename(img_path))
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        names.append(os.path.basename(img_path))
        images.append(img_path)
        prompts.append(prompt)
    return names, images, prompts, missing


def score_folder(folder, evaluator, batch_size=24, out=None, limit=None, keep_raw=False, show_progress=True):
    """Score every image+prompt pair in ``folder`` and write CSV + JSONL.

    ``evaluator`` is a :class:`QwenImageBench` (or anything exposing
    ``evaluate_batch(images, prompts, batch_size=, show_progress=)``).

    Returns ``(results, out_csv, out_jsonl)``.
    """
    names, images, prompts, missing = load_pairs(folder)
    if missing:
        print(f"WARNING: {len(missing)} image(s) had no sibling .txt and were skipped "
              f"(e.g. {missing[:3]})")
    if limit:
        names, images, prompts = names[:limit], images[:limit], prompts[:limit]
    print(f"Scoring {len(images)} image+prompt pairs from {folder}")

    results = evaluator.evaluate_batch(images, prompts, batch_size=batch_size, show_progress=show_progress)

    out_csv = out or os.path.join(folder, "sample_scores.csv")
    out_jsonl = os.path.splitext(out_csv)[0] + ".jsonl"

    # CSV: one compact row per sample (total + per-L1-dimension level1 scores).
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "ID", "total", *L1_DIMS, "valid"])
        for name, r in zip(names, results):
            dim = r["dim_scores"]
            row = [name, r["ID"], _fmt(r["total"])]
            row += [_fmt(dim[d]["level1_score"]) if d in dim else "" for d in L1_DIMS]
            row.append(r["valid"])
            w.writerow(row)

    # JSONL: full detail (raw judge text dropped unless --keep-raw).
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for name, r in zip(names, results):
            rec = {"file": name, **r}
            if not keep_raw:
                rec.pop("raw_outputs", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Quick summary to the terminal.
    bench = aggregate_bench_scores(results)
    n_valid = sum(1 for r in results if r["total"] is not None)
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_jsonl}")
    print(f"\nMean total over {n_valid}/{len(results)} scored samples: {_fmt(bench['total']) or 'N/A'}")
    for d in L1_DIMS:
        print(f"  {d:22s}: {_fmt(bench['level1'].get(d)) or 'N/A'}")
    return results, out_csv, out_jsonl


def main():
    ap = argparse.ArgumentParser(description="Sample-wise Qwen-Image-Bench scoring over a folder")
    ap.add_argument("--images", required=True, help="Folder of <name>.png + <name>.txt pairs")
    ap.add_argument("--model", default=TransformersJudge.DEFAULT_MODEL,
                    help="HF model ID or local path for the judge (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None, help="Output CSV path (default: <images>/sample_scores.csv)")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N images (quick test)")
    ap.add_argument("--keep-raw", action="store_true", help="Keep raw judge text in the JSONL")
    args = ap.parse_args()

    evaluator = QwenImageBench(
        model=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    score_folder(
        args.images, evaluator,
        batch_size=args.batch_size, out=args.out,
        limit=args.limit, keep_raw=args.keep_raw,
    )


if __name__ == "__main__":
    main()
