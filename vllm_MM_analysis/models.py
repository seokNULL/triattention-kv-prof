"""Vision-language models used by vllm_MM_analysis and their prompt formats."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cli import MODELS_DIR  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
DATA_DIR = REPO_ROOT / "data"

# template: prompt with {media} and {question}; None means the tokenizer's chat template.
# placeholder: text put in the prompt per modality.
# vision_token: token vLLM expands the placeholder into (one per vision-token position).
# temporal_patch: frames merged into one group of video tokens (default 1).
VLM_SPECS = {
    "LLaVA-OneVision-7B": {
        "repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "template": "<|im_start|>user {media}\n{question}<|im_end|><|im_start|>assistant\n",
        "placeholder": {"video": "<video>", "image": "<image>"},
        "vision_token": {"video": "<video>", "image": "<image>"},
    },
    "Qwen2.5-VL-7B": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "template": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|>{media}<|vision_end|>"
            "{question}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "placeholder": {"video": "<|video_pad|>", "image": "<|image_pad|>"},
        "vision_token": {"video": "<|video_pad|>", "image": "<|image_pad|>"},
        "temporal_patch": 2,  # two frames share one set of video tokens
        # Frames keep their resolution up to the image limit (~16k tokens per frame
        # pair), so a 1080p clip overflows the context; cap pixels per frame.
        "max_pixels": 448 * 448,  # ~256 video tokens per frame pair
    },
    "InternVL3-8B": {
        "repo": "OpenGVLab/InternVL3-8B",
        "template": None,
        "placeholder": {"video": "<video>", "image": "<image>"},
        "vision_token": {"video": "<IMG_CONTEXT>", "image": "<IMG_CONTEXT>"},
        "trust_remote_code": True,
        "stop_tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"],
    },
}


def model_path(name: str) -> Path:
    path = MODELS_DIR / name
    if not (path / "config.json").exists():
        raise SystemExit(f"model not found: {path}\nRun: python prepare.py --models {name}")
    return path


def engine_extra(name: str, max_model_len: int | None = None, max_pixels: int | None = None) -> dict:
    """Model-specific LLM() arguments.

    Args:
        name: Model name in VLM_SPECS.
        max_model_len: Context limit; None keeps the model's own maximum.
        max_pixels: Pixels per frame for models that resize by area (Qwen2.5-VL);
            None uses the default in VLM_SPECS.
    """
    spec = VLM_SPECS[name]
    extra = {"trust_remote_code": True} if spec.get("trust_remote_code") else {}
    if max_model_len:
        extra["max_model_len"] = max_model_len
    pixels = max_pixels or spec.get("max_pixels")
    if pixels:
        extra["mm_processor_kwargs"] = {"max_pixels": pixels}
    return extra


def build_prompt(name: str, modality: str, question: str, tokenizer) -> str:
    spec = VLM_SPECS[name]
    media = spec["placeholder"][modality]
    if spec["template"] is None:
        messages = [{"role": "user", "content": f"{media}\n{question}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return spec["template"].format(media=media, question=question)


def stop_token_ids(name: str, tokenizer) -> list[int] | None:
    tokens = VLM_SPECS[name].get("stop_tokens")
    if not tokens:
        return None
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    return [i for i in ids if i is not None and i != tokenizer.unk_token_id]


def vision_token_id(name: str, modality: str, tokenizer) -> int:
    return tokenizer.convert_tokens_to_ids(VLM_SPECS[name]["vision_token"][modality])
