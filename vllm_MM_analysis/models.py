"""Vision-language models used by vllm_MM_analysis and their prompt formats."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cli import MODELS_DIR  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent / "results"

# name -> (HF repo, prompt template with {media} and {question}, placeholder per modality)
VLM_SPECS = {
    "LLaVA-OneVision-7B": {
        "repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "template": "<|im_start|>user {media}\n{question}<|im_end|><|im_start|>assistant\n",
        "placeholder": {"video": "<video>", "image": "<image>"},
    },
    "Qwen2.5-VL-7B": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "template": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|>{media}<|vision_end|>"
            "{question}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "placeholder": {"video": "<|video_pad|>", "image": "<|image_pad|>"},
    },
}


def model_path(name: str) -> Path:
    path = MODELS_DIR / name
    if not (path / "config.json").exists():
        raise SystemExit(
            f"model not found: {path}\n"
            f"Run: python prepare_models.py --models {name}"
        )
    return path


def build_prompt(name: str, modality: str, question: str) -> str:
    spec = VLM_SPECS[name]
    return spec["template"].format(media=spec["placeholder"][modality], question=question)
