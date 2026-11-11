"""MVBench (OpenGVLab/MVBench): 20 multiple-choice video tasks.

Layout after ``prepare.py --datasets mvbench``: data/mvbench/json/<task>.json and
the extracted video archives under data/mvbench/video/. Each json entry has
video, question, candidates, answer and, for some tasks, start/end seconds.
Videos are found by file name, preferring the source folder the official
evaluation uses for the task.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from models import DATA_DIR

ROOT = DATA_DIR / "mvbench"
HF_REPO = "OpenGVLab/MVBench"

# task -> (json file, source folder used by the official evaluation, uses start/end)
TASKS = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation", False),
    "Action Localization": ("action_localization.json", "sta/sta_video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video", False),
    "Action Count": ("action_count.json", "perception/videos", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation", False),
    "State Change": ("state_change.json", "perception/videos", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd", False),
    "Character Order": ("character_order.json", "perception/videos", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation", False),
}

INSTRUCTION = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question."
)
LETTERS = "ABCDEFGH"


class VideoIndex:
    """Map a video name from the json to a file or frame folder under video_root."""

    def __init__(self, video_root: Path):
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(video_root):
            for name in filenames + dirnames:
                self.by_name[name].append(Path(dirpath) / name)

    def find(self, video: str, source: str) -> Path | None:
        paths = self.by_name.get(Path(video).name, [])
        for preferred in (f"{source}/{video}", video, source):
            hits = [p for p in paths if p.as_posix().endswith(preferred) or preferred in p.as_posix()]
            if hits:
                return hits[0]
        return paths[0] if paths else None


def format_question(item: dict) -> str:
    options = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(item["candidates"]))
    return (f"{INSTRUCTION}\nQuestion: {item['question']}\nOptions:\n{options}\n"
            "Answer with the option's letter from the given choices directly.")


def load(tasks: list[str] | None = None, limit_per_task: int | None = None) -> tuple[list[dict], dict]:
    """Return (items, missing_count_per_task). Each item has an id, task, path and prompt text."""
    json_dir, video_root = ROOT / "json", ROOT / "video"
    if not json_dir.is_dir():
        raise SystemExit(f"MVBench not found at {ROOT}\nRun: python prepare.py --datasets mvbench")
    index = VideoIndex(video_root)
    items, missing = [], {}
    for task in tasks or TASKS:
        json_name, source, bounded = TASKS[task]
        rows = json.loads((json_dir / json_name).read_text())
        if limit_per_task:
            rows = rows[:limit_per_task]
        missing[task] = 0
        for i, row in enumerate(rows):
            path = index.find(row["video"], source)
            if path is None:
                missing[task] += 1
                continue
            items.append({
                "id": f"{json_name.removesuffix('.json')}/{i}",
                "task": task,
                "video": str(path),
                "bound": (row["start"], row["end"]) if bounded and "start" in row else None,
                "question": format_question(row),
                "answer": LETTERS[row["candidates"].index(row["answer"])],
                "candidates": row["candidates"],
            })
    return items, missing
