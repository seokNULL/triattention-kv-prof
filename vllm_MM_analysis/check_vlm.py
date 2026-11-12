#!/usr/bin/env python3
"""Step 1: check that a vision-language model runs on this vLLM with video and image input.

Sends one video prompt and one image prompt, prints the answers and where the
vision tokens sit in each prompt (needed later to analyze their KV separately).
Without --video/--image a synthetic clip (a red square moving right over a
static background) and a synthetic image are used, so no dataset is needed.

Example (from vllm_MM_analysis/):
    python check_vlm.py --model LLaVA-OneVision-7B
    python check_vlm.py --model InternVL3-8B --video clip.mp4 --image cat.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from models import (OUTPUT_ROOT, VLM_SPECS, build_prompt, engine_extra, model_path,
                    stop_token_ids, vision_token_id)
from video_io import load_video

VIDEO_QUESTION = "Describe what happens in this video in one sentence."
IMAGE_QUESTION = "Describe this image in one sentence."


def synthetic_video(num_frames: int, size: int = 384) -> np.ndarray:
    """Red square moving left to right over a fixed gradient: mostly static frames."""
    ramp = np.linspace(40, 200, size, dtype=np.uint8)
    background = np.stack([np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size)),
                           np.full((size, size), 120, np.uint8)], axis=-1)
    frames = np.repeat(background[None], num_frames, axis=0)
    box = size // 6
    for t in range(num_frames):
        x = int((size - box) * t / max(num_frames - 1, 1))
        frames[t, size // 2 - box // 2:size // 2 + box // 2, x:x + box] = (230, 20, 20)
    return frames


def synthetic_image(size: int = 384) -> Image.Image:
    img = np.full((size, size, 3), 245, np.uint8)
    img[size // 4:3 * size // 4, size // 4:3 * size // 4] = (20, 60, 220)
    return Image.fromarray(img)


def token_runs(token_ids: list[int], target: int) -> list[list[int]]:
    """Contiguous [start, end) ranges where token_ids == target."""
    runs, start = [], None
    for i, tok in enumerate(token_ids + [None]):
        if tok == target and start is None:
            start = i
        elif tok != target and start is not None:
            runs.append([start, i])
            start = None
    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(VLM_SPECS))
    p.add_argument("--video", help="Video file (default: synthetic clip).")
    p.add_argument("--image", help="Image file (default: synthetic image).")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Context limit (default: the model's maximum).")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Pixels per frame for Qwen2.5-VL (default 448*448); others resize to a fixed size.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # The TriAttention plugin targets text models with 1D RoPE; keep it out.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import vllm
    from vllm import LLM, SamplingParams

    path = model_path(args.model)
    video = load_video(args.video, args.num_frames) if args.video else synthetic_video(args.num_frames)
    image = Image.open(args.image).convert("RGB") if args.image else synthetic_image()
    print(f"vllm {vllm.__version__} | model {path}")
    print(f"video {args.video or 'synthetic'}: {video.shape} | image {args.image or 'synthetic'}: {image.size}")

    t0 = time.time()
    llm = LLM(model=str(path),
              limit_mm_per_prompt={"video": 1, "image": 1},
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, seed=0,
              **engine_extra(args.model, args.max_model_len, args.max_pixels))
    print(f"engine ready in {time.time() - t0:.0f}s")

    tokenizer = llm.get_tokenizer()
    requests = [
        ("video", {"prompt": build_prompt(args.model, "video", VIDEO_QUESTION, tokenizer),
                   "multi_modal_data": {"video": video}}),
        ("image", {"prompt": build_prompt(args.model, "image", IMAGE_QUESTION, tokenizer),
                   "multi_modal_data": {"image": image}}),
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            stop_token_ids=stop_token_ids(args.model, tokenizer))
    t0 = time.time()
    outputs = llm.generate([r for _, r in requests], params)
    print(f"generated in {time.time() - t0:.1f}s")

    report = {"vllm": vllm.__version__, "model": args.model,
              "video_source": args.video or "synthetic", "num_frames": int(video.shape[0]),
              "image_source": args.image or "synthetic", "results": []}
    for (modality, _), out in zip(requests, outputs):
        placeholder = VLM_SPECS[args.model]["vision_token"][modality]
        placeholder_id = vision_token_id(args.model, modality, tokenizer)
        ids = list(out.prompt_token_ids)
        runs = token_runs(ids, placeholder_id)
        n_vision = sum(e - s for s, e in runs)
        text = out.outputs[0].text.strip()
        print(f"\n[{modality}] prompt tokens {len(ids)}, {placeholder} tokens {n_vision} in runs {runs}")
        print(f"[{modality}] answer: {text}")
        report["results"].append({"modality": modality, "prompt_tokens": len(ids),
                                  "placeholder": placeholder, "placeholder_id": placeholder_id,
                                  "vision_tokens": n_vision, "vision_runs": runs, "answer": text})

    out_dir = OUTPUT_ROOT / args.model / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "check_vlm.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir / 'check_vlm.json'}")


if __name__ == "__main__":
    main()
