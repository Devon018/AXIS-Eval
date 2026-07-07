from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class VideoLogger:
    def __init__(self, output_path: str | Path, fps: int = 10, enabled: bool = True):
        self.output_path = Path(output_path)
        self.fps = fps
        self.enabled = enabled
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray | None) -> None:
        if not self.enabled or frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Video frame must have shape HxWx3, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.frames.append(np.ascontiguousarray(arr))

    def save(self) -> str | None:
        if not self.enabled or not self.frames:
            return None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self.output_path, self.frames, fps=self.fps)
        return str(self.output_path)
