"""Video file frame source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from data.base import FrameSource

logger = logging.getLogger(__name__)


class VideoSource(FrameSource):
    """Yields ``(frame_id, bgr_frame)`` from a video file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                yield frame_id, frame
        finally:
            cap.release()

        logger.info("Video exhausted after %d frames", frame_id)
