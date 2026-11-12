"""MVBench (OpenGVLab/MVBench): 20 multiple-choice video tasks.

Layout after ``prepare.py --datasets mvbench``: data/mvbench/json/<task>.json and
the extracted video archives under data/mvbench/video/. Each json entry has
video, question, candidates, answer and, for some tasks, start/end seconds.
Videos are found by file name, preferring the source folder the official
evaluation uses for the task.
"""

from __future__ import annotations

import json

from models import DATA_DIR
from video_io import VideoIndex

ROOT = DATA_DIR / "mvbench"
HF_REPO = "OpenGVLab/MVBench"
DEFAULT_PATTERNS = ["json/*", "video/*"]

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


def format_question(item: dict) -> str:
    options = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(item["candidates"]))
    return (f"{INSTRUCTION}\nQuestion: {item['question']}\nOptions:\n{options}\n"
            "Answer with the option's letter from the given choices directly.")


def load(tasks: list[str] | None = None, limit_per_task: int | None = None) -> tuple[list[dict], dict]:
    """Return (items, missing_count_per_task). Each item has an id, task, video path and prompt text."""
    json_dir, video_root = ROOT / "json", ROOT / "video"
    if not json_dir.is_dir():
        raise SystemExit(f"MVBench not found at {ROOT}\nRun: python prepare.py --datasets mvbench")
    index = VideoIndex(video_root)
    items, missing = [], {}
    for task in tasks or TASKS:
        if task not in TASKS:
            raise SystemExit(f"unknown MVBench task {task!r}; choose from {list(TASKS)}")
        json_name, source, bounded = TASKS[task]
        if not (json_dir / json_name).exists():
            print(f"[warn] {json_dir / json_name} missing, task {task!r} skipped")
            continue
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
