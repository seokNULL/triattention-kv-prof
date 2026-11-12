"""Uniform frame sampling from a video file or a folder of frame images."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Containers whose header frame count and seeking are reliable enough for long videos.
SEEKABLE = {".mp4", ".mkv", ".mov", ".avi"}


class VideoIndex:
    """Find a video file or frame folder by name anywhere under root."""

    def __init__(self, root: Path):
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames + dirnames:
                self.by_name[name].append(Path(dirpath) / name)

    def find(self, video: str, prefer: str = "") -> Path | None:
        """Path named like video; among several, one whose path contains prefer."""
        paths = self.by_name.get(Path(video).name, [])
        for preferred in (f"{prefer}/{video}", video, prefer):
            hits = [p for p in paths if preferred and preferred in p.as_posix()]
            if hits:
                return hits[0]
        return paths[0] if paths else None


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
    header_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if path.suffix.lower() in SEEKABLE and header_total > 0:
        wanted = sample_indices(header_total, num_frames, *_bound_frames(bound, fps, header_total))
        frames = _read_by_seek(cap, wanted)
        cap.release()
        if len(frames) == len(wanted):
            return np.stack(frames)
        cap = cv2.VideoCapture(str(path))
    frames = _read_by_grab(cap, path, num_frames, bound, fps)
    cap.release()
    return np.stack(frames)


def _read_by_seek(cap, indices: list[int]) -> list[np.ndarray]:
    import cv2

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _read_by_grab(cap, path: Path, num_frames: int, bound, fps: float) -> list[np.ndarray]:
    """Count frames by grabbing (webm/gif headers are unreliable), then decode the sampled ones."""
    import cv2

    total = 0
    while cap.grab():
        total += 1
    if total == 0:
        raise ValueError(f"no frames in {path}")
    wanted = set(sample_indices(total, num_frames, *_bound_frames(bound, fps, total)))
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
    return frames
