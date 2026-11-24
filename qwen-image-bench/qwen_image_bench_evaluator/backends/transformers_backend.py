"""
transformers inference backend for the Qwen-Image-Bench judge.

Drop-in replacement for the upstream ms-swift PtEngine backend: it exposes the
same ``generate_batch(items)`` interface, where each item is::

    {"system_prompt": str, "user_text": str, "image": PIL.Image}

and the return value is a list of generated text strings (one per item).

The judge is loaded with vanilla ``transformers`` (``AutoModelForImageTextToText``
+ ``AutoProcessor``), so the model class is resolved from the checkpoint config
and any Qwen-VL judge (Qwen3-VL / Qwen2.5-VL / the fine-tuned Qwen-Image-Bench
judge) works without code changes.

Inference settings mirror the swift command used to produce the released
dataset's ``*_response_*`` fields::

    swift infer --infer_backend pt --max_batch_size 24 --seed 42 \
                --temperature 0 --top_k 1 --top_p 1 \
                --repetition_penalty 1.05 --max_new_tokens 4096 \
                --enable_thinking true

temperature=0 / top_k=1 is greedy decoding, so we set ``do_sample=False`` and
keep ``repetition_penalty=1.05``; the remaining sampling knobs are inert under
greedy decoding.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# The upstream user prompt embeds a literal "<image>" marker showing where the
# generated image belongs in the conversation. transformers' chat template
# instead represents the image as a structured content block, so we split the
# text on this marker and re-insert the image at exactly that position.
IMAGE_PLACEHOLDER = "<image>"


class TransformersJudge:
    """Qwen-Image-Bench judge backed by a raw transformers model."""

    DEFAULT_MODEL = "Qwen/Qwen-Image-Bench"

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        max_batch_size=24,
        max_new_tokens=4096,
        dtype=torch.bfloat16,
        enable_thinking=True,
        seed=42,
    ):
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        # Greedy decoding is deterministic, but seed anyway so any incidental
        # randomness (e.g. dropout left enabled) matches the reference run.
        torch.manual_seed(seed)

        self.processor = AutoProcessor.from_pretrained(model_path)
        # Left padding is required for correct batched generation with
        # decoder-only models.
        self.processor.tokenizer.padding_side = "left"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(device)

    def _build_messages(self, item):
        """Convert a judge item into a transformers chat-message list.

        The image is placed where the upstream ``<image>`` marker sat; if no
        marker is present it is prepended to the user turn.
        """
        before, sep, after = item["user_text"].partition(IMAGE_PLACEHOLDER)
        content = []
        if not sep:
            # No placeholder found: lead with the image, then the full text.
            content.append({"type": "image", "image": item["image"]})
            content.append({"type": "text", "text": item["user_text"]})
        else:
            if before.strip():
                content.append({"type": "text", "text": before})
            content.append({"type": "image", "image": item["image"]})
            if after.strip():
                content.append({"type": "text", "text": after})
        return [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": content},
        ]

    def _apply_template(self, messages):
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Processor/template variants that don't accept enable_thinking.
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    @torch.inference_mode()
    def generate_batch(self, items):
        """Batch inference for multiple items.

        Each item: ``{"system_prompt": str, "user_text": str, "image": PIL.Image}``.
        Returns a list of generated text strings (raw judge output, including any
        ``<think>`` section, which the score parser strips downstream).
        """
        messages_list = [self._build_messages(it) for it in items]
        texts = [self._apply_template(m) for m in messages_list]
        images = [it["image"] for it in items]

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
        # Drop the prompt tokens, keep only the newly generated continuation.
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
