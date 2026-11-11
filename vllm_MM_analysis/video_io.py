"""Uniform frame sampling from a video file or a folder of frame images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_indices(total: int, num_frames: int, first: int = 0, last: int | None = None) -> list[int]:
    """Centers of num_frames equal segments of [first, last]."""
    last = total - 1 if last is None else min(last, total - 1)
    first = max(0, min(first, last))
    seg = (last - first + 1) / num_frames
    return sorted({min(last, first + int(seg * (i + 0.5))) for i in range(num_frames)})


def _bound_frames(bound, fps: float, total: int) -> tuple[int, int | None]:
    if not bound or fps <= 0:
        return 0, None
    start, end = bound
    return max(0, round(start * fps)), min(total - 1, round(end * fps))


def load_video(path: str | Path, num_frames: int, bound=None, frame_fps: float = 3.0) -> np.ndarray:
    """Return [F, H, W, 3] uint8 RGB frames.

    Args:
        path: Video file, or a folder of frame images (sorted by name).
        num_frames: Frames to sample uniformly.
        bound: Optional (start_sec, end_sec) to sample within.
        frame_fps: Frame rate assumed for a folder of images.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no frame images in {path}")
        first, last = _bound_frames(bound, frame_fps, len(files))
        from PIL import Image

        return np.stack([np.asarray(Image.open(files[i]).convert("RGB"))
                         for i in sample_indices(len(files), num_frames, first, last)])

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # Frame counts in webm/gif headers are unreliable: count by grabbing.
    total = 0
    while cap.grab():
        total += 1
    if total == 0:
        raise ValueError(f"no frames in {path}")
    first, last = _bound_frames(bound, fps, total)
    wanted = set(sample_indices(total, num_frames, first, last))
    cap.release()
    cap = cv2.VideoCapture(str(path))
    frames = []
    for i in range(max(wanted) + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)
