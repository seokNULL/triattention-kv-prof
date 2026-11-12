"""Video-MME (lmms-lab/Video-MME): 900 videos x 3 four-choice questions, split by length.

Layout after ``prepare.py --datasets videomme``: the question parquet and the
extracted video archives (<videoID>.mp4) anywhere under data/videomme/.
Items are grouped ("task") by video length: short (< 2 min), medium
(4-15 min), long (30-60 min). Subtitles are not used.
"""

from __future__ import annotations

import re

from models import DATA_DIR
from video_io import VideoIndex

ROOT = DATA_DIR / "videomme"
HF_REPO = "lmms-lab/Video-MME"
# The full video set is very large; fetch chunks selectively with --patterns.
DEFAULT_PATTERNS = ["*.parquet", "videos_chunked_*.zip"]
TASKS = ["short", "medium", "long"]
LETTERS = "ABCD"

PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    "{question}\n{options}\nThe best answer is:"
)


def _option_text(option: str) -> str:
    return re.sub(r"^\s*\(?[A-D][.)]\s*", "", option)


def load(tasks: list[str] | None = None, limit_per_task: int | None = None) -> tuple[list[dict], dict]:
    """Return (items, missing_count_per_length). Each item has an id, task, video path and prompt text."""
    import pandas as pd

    parquets = sorted(ROOT.rglob("*.parquet")) if ROOT.is_dir() else []
    if not parquets:
        raise SystemExit(f"Video-MME questions not found under {ROOT}\n"
                         "Run: python prepare.py --skip-models --datasets videomme")
    for task in tasks or []:
        if task not in TASKS:
            raise SystemExit(f"unknown Video-MME length {task!r}; choose from {TASKS}")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    index = VideoIndex(ROOT)
    items, missing = [], {}
    for task in tasks or TASKS:
        rows = df[df["duration"] == task]
        if limit_per_task:
            rows = rows.head(limit_per_task)
        missing[task] = 0
        for row in rows.itertuples():
            path = index.find(f"{row.videoID}.mp4")
            if path is None:
                missing[task] += 1
                continue
            options = list(row.options)
            items.append({
                "id": str(row.question_id),
                "task": task,
                "task_type": row.task_type,
                "video": str(path),
                "bound": None,
                "question": PROMPT.format(question=row.question, options="\n".join(options)),
                "answer": str(row.answer).strip(),
                "candidates": [_option_text(o) for o in options],
            })
    return items, missing
