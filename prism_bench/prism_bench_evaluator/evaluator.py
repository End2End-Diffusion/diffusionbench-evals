"""PRISM-Bench evaluator (Qwen2.5-VL or GPT-4.1 judge).

PRISM-Bench scores generated images on 7 tracks, each image judged under two rubrics:
`alignment` (track-specific) and `aesthetic` (shared). A multimodal judge reads the
image + the track's prompt and emits a JSON `{justification, score}` with score in
0-10. The judge can run locally through Transformers (Qwen-VL) or through an
OpenAI-compatible Chat Completions API (GPT-4.1). See
https://github.com/rongyaofang/prism-bench.

This port keeps the judge prompts byte-identical to the reference scripts. Local
judging uses **left-padded batched generation** (the reference processes one image at
a time; decoder-only batching requires left padding to stay correct). GPT-4.1 uses
the request shape and pinned model snapshot from `eval_gpt41.py`.

Everything lives on one wrapper class:

    judge = PrismBenchEvaluator(device="cuda")        # loads the Qwen2.5-VL judge
    judge = PrismBenchEvaluator(model_name="gpt-4.1", api_key="...")
    metadata = judge.fetch_metadata()                  # 7 tracks x 100 x {alignment, aesthetic}
    results = judge.evaluate_pairs(images, metadata)   # batched, left-padded
    summary = judge.summarize_results(results)
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from .constants import (
    TRACKS, EVAL_TYPES, JUDGE_MODELS, OPENAI_JUDGE_MODELS,
    DEFAULT_JUDGE_MODEL, DEFAULT_OPENAI_JUDGE_MODEL,
    DEFAULT_OPTIONS, DEFAULT_OPENAI_OPTIONS, load_judge_templates,
)


class PrismBenchEvaluator:
    """Self-contained PRISM-Bench judge for local Qwen or API-backed GPT-4.1."""

    TRACKS = TRACKS
    EVAL_TYPES = EVAL_TYPES

    def __init__(
        self,
        model_name=None,
        device="cuda",
        options=None,
        api_key=None,
        base_url=None,
        client=None,
        backend=None,
    ):
        """Load a local Qwen-VL judge or configure an OpenAI-compatible API judge.

        Args:
            model_name: a key in JUDGE_MODELS (e.g. "qwen2.5-vl-72b", "qwen3-vl-8b",
                "qwen3.5-4b", or "gpt-4.1"), a full HF/local path, or an API model
                name. Defaults to the PRISM-Bench Qwen2.5-VL-72B judge. With
                backend="openai", defaults to the reference GPT-4.1 snapshot.
            device: device the local model is moved to ("cuda"/"cpu"/"cuda:0"/...).
            options: local overrides for DEFAULT_OPTIONS, or API overrides for
                DEFAULT_OPENAI_OPTIONS.
            api_key: OpenAI API key. If omitted, the OpenAI client uses its environment
                configuration.
            base_url: optional OpenAI-compatible endpoint URL.
            client: optional preconfigured OpenAI-compatible client (also useful for tests).
            backend: "huggingface" or "openai". Inferred from GPT-4.1 aliases, API
                arguments, or client when omitted.
        """
        api_alias = model_name in OPENAI_JUDGE_MODELS if model_name is not None else False
        if backend is None:
            backend = (
                "openai"
                if api_alias or client is not None or api_key is not None or base_url is not None
                else "huggingface"
            )
        backend = {"hf": "huggingface", "api": "openai"}.get(backend, backend)
        if backend not in {"huggingface", "openai"}:
            raise ValueError("backend must be 'huggingface' or 'openai'")

        self.backend = backend
        self.device = device
        self.templates = load_judge_templates()

        if backend == "openai":
            opts = {**DEFAULT_OPENAI_OPTIONS, **(options or {})}
            self.model_name = model_name or DEFAULT_OPENAI_JUDGE_MODEL
            self.model_id = OPENAI_JUDGE_MODELS.get(self.model_name, self.model_name)
            self.batch_size = int(opts["batch_size"])
            self.max_tokens = int(opts["max_tokens"])
            self.temperature = float(opts["temperature"])
            self.top_p = float(opts["top_p"])
            self.seed = int(opts["seed"])
            self.model = None
            self.processor = None

            if client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise ImportError(
                        "The openai package is required for the GPT-4.1 judge"
                    ) from exc
                client_options = {}
                if api_key is not None:
                    client_options["api_key"] = api_key
                if base_url:
                    client_options["base_url"] = base_url
                client = OpenAI(**client_options)
            self.client = client
            return

        opts = {**DEFAULT_OPTIONS, **(options or {})}
        self.model_name = model_name or DEFAULT_JUDGE_MODEL
        self.model_id = JUDGE_MODELS.get(self.model_name, self.model_name)
        self.batch_size = int(opts["batch_size"])
        self.max_new_tokens = int(opts["max_new_tokens"])
        self.client = None

        # Big judges (e.g. 72B, ~145 GB bf16) don't fit one GPU -> shard across all visible
        # GPUs with device_map="auto"; smaller ones load onto `device`. Pull the param count
        # robustly from the id: the first "<N>B" token (handles "...-72B-Instruct",
        # "...-35B-A3B", and short names); default 0 -> single device.
        m = re.search(r"(\d+)\s*b", self.model_id, re.IGNORECASE)
        model_size = int(m.group(1)) if m else 0
        if model_size > 60:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype="auto", device_map="auto"
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype="auto"
            ).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=opts["min_pixels"], max_pixels=opts["max_pixels"]
        )
        # REQUIRED for correct batched generation with a decoder-only model: pad on the left
        # so generation continues from real content, not from trailing pad tokens.
        self.processor.tokenizer.padding_side = "left"

    # --- prompt construction -------------------------------------------------

    def full_prompt(self, metadata):
        """Pick the judge template for this item's eval_type/track and insert the prompt."""
        if metadata["eval_type"] == "alignment":
            template = self.templates["alignment"][metadata["track"]]
        else:
            template = self.templates["aesthetic"]
        return template.format(text_prompt=metadata.get("prompt", ""))

    # --- inference -----------------------------------------------------------

    def _judge_batch(self, images, prompts):
        """Run the selected judge on (image, full_prompt) pairs."""
        if self.backend == "openai":
            return self._judge_openai_batch(images, prompts)
        return self._judge_hf_batch(images, prompts)

    @torch.no_grad()
    def _judge_hf_batch(self, images, prompts):
        """Run the local judge on a batch; return raw response strings."""
        texts = []
        reference_qwen = "qwen2.5-vl" in self.model_id.lower()
        image_inputs = []
        for img, prompt in zip(images, prompts):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            # add_vision_id=True matches the reference (prepends a "Picture N:" marker
            # before each image in the rendered prompt). enable_thinking=False keeps
            # thinking-capable judges (Qwen3.5 / Qwen3.6) from emitting chain-of-thought
            # instead of the JSON verdict; it is an unused template var (harmless) for the
            # non-thinking judges (Qwen2.5-VL / Qwen3-VL).
            template_options = {} if reference_qwen else {"enable_thinking": False}
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
                **template_options))
            if reference_qwen:
                from qwen_vl_utils import process_vision_info
                processed, _ = process_vision_info(messages)
                image_inputs.extend(processed)
            else:
                image_inputs.append(img)

        inputs = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        if reference_qwen:
            # eval_qwen25.py casts floating inputs to FP16 and uses the checkpoint's
            # generation defaults. Alternative judges use explicit greedy decoding.
            inputs = inputs.to(torch.float16)
        generation_options = {} if reference_qwen else {"do_sample": False}
        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, **generation_options)
        # With left padding every input row has the same length, so slicing off len(input)
        # leaves exactly the generated tokens for each sample.
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @staticmethod
    def _encode_image_to_base64(image):
        """Encode a PIL image as the data payload expected by the reference script."""
        buffered = io.BytesIO()
        image.save(buffered, format=image.format or "PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _judge_openai_batch(self, images, prompts):
        """Issue bounded concurrent requests, preserving input/result correspondence."""
        def judge_one(pair):
            image, prompt = pair
            image_data = self._encode_image_to_base64(image)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                },
            ]
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": False,
            }
            if self.model_id.startswith("gpt-5"):
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens
            # GPT-5.5 and GPT-5.6 only accept the default sampling behavior.
            # Omitting these fields avoids 400s from non-default temperature
            # values (and keeps the request compatible across GPT-5 variants).
            if not self.model_id.startswith("gpt-5"):
                payload["temperature"] = self.temperature
                payload["top_p"] = self.top_p
                if "gemini" not in self.model_id:
                    payload["seed"] = self.seed
            completion = self.client.chat.completions.create(**payload)
            return completion.choices[0].message.content or ""

        pairs = list(zip(images, prompts))
        if len(pairs) <= 1:
            return [judge_one(pair) for pair in pairs]
        with ThreadPoolExecutor(max_workers=min(self.batch_size, len(pairs))) as pool:
            return list(pool.map(judge_one, pairs))

    def _result(self, metadata, raw, index):
        parsed = self.clean_and_parse_json(raw)
        parsed = parsed if isinstance(parsed, dict) else {}
        score = parsed.get("score")
        # Match the official summary: JSON numbers only, never numeric strings.
        if not isinstance(score, (int, float)):
            score = None
        return {
            "index": index,
            "track": metadata.get("track", ""),
            "eval_type": metadata.get("eval_type", ""),
            "prompt": metadata.get("prompt", ""),
            "score": score,
            "justification": parsed.get("justification", ""),
            "raw": raw,
        }

    # --- public evaluation API ----------------------------------------------

    def evaluate_image(self, image, metadata):
        """Judge a single (image, metadata) pair. metadata needs `track`, `eval_type`, `prompt`."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        raw = self._judge_batch([image], [self.full_prompt(metadata)])[0]
        return self._result(metadata, raw, 0)

    def evaluate_batch(self, images, metadata_list, show_progress=True, batch_size=None):
        """Judge many (image, metadata) pairs with left-padded batched generation.

        Args:
            images: list of PIL.Image.Image / np.ndarray (one per metadata entry).
            metadata_list: list of dicts with `track`, `eval_type`, `prompt`.
            batch_size: judge images per forward pass (defaults to self.batch_size).

        For local judges, batched generation is NOT bit-identical to batch_size=1 (the
        judge generates text; bf16 + non-batch-invariant kernels can flip a token).
        GPT-4.1 uses bounded concurrent requests with unchanged per-item payloads.
        """
        from tqdm import tqdm

        if len(images) != len(metadata_list):
            raise ValueError(f"images and metadata_list must have the same length, got {len(images)} vs {len(metadata_list)}")
        bs = batch_size or self.batch_size

        results = []
        rng = range(0, len(images), bs)
        if show_progress:
            rng = tqdm(rng, desc="Judging", total=(len(images) + bs - 1) // bs)
        for start in rng:
            batch_imgs, batch_prompts, batch_md = [], [], []
            for j in range(start, min(start + bs, len(images))):
                img = images[j]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                batch_imgs.append(img.convert("RGB"))
                batch_prompts.append(self.full_prompt(metadata_list[j]))
                batch_md.append((j, metadata_list[j]))
            raws = self._judge_batch(batch_imgs, batch_prompts)
            for (idx, md), raw in zip(batch_md, raws):
                results.append(self._result(md, raw, idx))
        return results

    def evaluate_pairs(self, images, metadata_list, show_progress=True, batch_size=None):
        """Alias of evaluate_batch (1 image per metadata); kept for interface parity."""
        return self.evaluate_batch(images, metadata_list,
                                   show_progress=show_progress, batch_size=batch_size)

    def evaluate(self, images, prompts, eval_types=None, show_progress=True, batch_size=None):
        """Judge plain (image, prompt) pairs of the bundled PRISM-Bench prompts.

        Looks up each prompt's track from the bundled benchmark and runs every eval_type
        (alignment + aesthetic by default) -- so you only supply image+prompt pairs and
        get the scores back. Returns len(images) * len(eval_types) result rows.

        For prompts that are not part of PRISM-Bench, use evaluate_pairs with an explicit
        `track` in each metadata dict instead.
        """
        if len(images) != len(prompts):
            raise ValueError(f"images and prompts must have the same length, got {len(images)} vs {len(prompts)}")
        eval_types = eval_types or EVAL_TYPES
        lookup = self.prompt_to_track()
        exp_images, metadata = [], []
        for img, prompt in zip(images, prompts):
            track = lookup.get(prompt)
            if track is None:
                raise KeyError(
                    "prompt is not in PRISM-Bench (use evaluate_pairs with an explicit "
                    f"track for custom prompts): {prompt[:80]!r}")
            for eval_type in eval_types:
                exp_images.append(img)
                metadata.append({"track": track, "eval_type": eval_type, "prompt": prompt})
        return self.evaluate_batch(exp_images, metadata, show_progress=show_progress, batch_size=batch_size)

    # --- stateless helpers (also usable without loading a model) -------------

    @staticmethod
    def clean_and_parse_json(text):
        """Parse the judge's JSON output, tolerating code fences / trailing commas.

        Faithful port of the reference, with a demjson3 fallback. Returns {} if unparseable.
        """
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        s = re.sub(r",\s*(?=[}\]])", "", s)  # drop trailing commas
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                import demjson3
                return demjson3.decode(s)
            except Exception:
                return {}

    @staticmethod
    def prompt_to_track():
        """Map each bundled PRISM-Bench prompt -> its track (for prompt-only evaluation)."""
        path = os.path.join(os.path.dirname(__file__), "prism_bench_data.jsonl")
        mapping = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    mapping[row["prompt"]] = row["track"]
        return mapping

    @staticmethod
    def fetch_metadata(eval_types=None):
        """Load the bundled PRISM-Bench (English) benchmark as a flat metadata list.

        Reads the single file `prism_bench_data.jsonl` (one entry per prompt, each tagged
        with its `track` and `index`) and expands it across eval_types. Returns one entry
        per (track, index, eval_type):
            {"track", "index", "eval_type", "prompt"}
        Pair each entry with the generated image for (track, index) — i.e.
        `<image_root>/<track>/<index>.png` — when calling evaluate_pairs. Each image's
        `index` appears once per eval_type (alignment + aesthetic) by default.
        """
        eval_types = eval_types or EVAL_TYPES
        path = os.path.join(os.path.dirname(__file__), "prism_bench_data.jsonl")
        metadata = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for eval_type in eval_types:
                    metadata.append({
                        "track": row["track"], "index": row["index"],
                        "eval_type": eval_type, "prompt": row["prompt"],
                    })
        return metadata

    @staticmethod
    def summarize_results(results):
        """Aggregate judge scores into PRISM-Bench's per-track and overall averages.

        Returns a dict with:
          - total_judged / total_parsed
          - alignment_by_track / aesthetic_by_track / combined_by_track  ({track: mean score})
          - overall_alignment / overall_aesthetic / final_total_average
          - overall_score: alias of final_total_average
        Scores are on PRISM-Bench's 0-10 scale; rows the judge failed to parse are ignored.
        """
        df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        empty = {
            "total_judged": 0, "total_parsed": 0,
            "alignment_by_track": {}, "aesthetic_by_track": {}, "combined_by_track": {},
            "overall_alignment": 0.0, "overall_aesthetic": 0.0,
            "final_total_average": 0.0, "overall_score": 0.0,
        }
        if df.empty:
            return empty
        valid = df[df["score"].notna()]
        if valid.empty:
            return {**empty, "total_judged": int(len(df))}

        alignment_by_track, aesthetic_by_track, combined_by_track = {}, {}, {}
        all_alignment, all_aesthetic = [], []
        for track in TRACKS:
            a = valid[(valid["eval_type"] == "alignment") & (valid["track"] == track)]["score"].tolist()
            e = valid[(valid["eval_type"] == "aesthetic") & (valid["track"] == track)]["score"].tolist()
            alignment_by_track[track] = float(np.mean(a)) if a else 0.0
            aesthetic_by_track[track] = float(np.mean(e)) if e else 0.0
            combined_by_track[track] = float(np.mean(a + e)) if (a or e) else 0.0
            all_alignment += a
            all_aesthetic += e

        overall_alignment = float(np.mean(all_alignment)) if all_alignment else 0.0
        overall_aesthetic = float(np.mean(all_aesthetic)) if all_aesthetic else 0.0
        final_total = float(np.mean(all_alignment + all_aesthetic)) if (all_alignment or all_aesthetic) else 0.0

        return {
            "total_judged": int(len(df)),
            "total_parsed": int(len(valid)),
            "alignment_by_track": alignment_by_track,
            "aesthetic_by_track": aesthetic_by_track,
            "combined_by_track": combined_by_track,
            "overall_alignment": overall_alignment,
            "overall_aesthetic": overall_aesthetic,
            "final_total_average": final_total,
            "overall_score": final_total,
        }
