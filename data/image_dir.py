"""Image-directory frame source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Set, Tuple

import cv2
import numpy as np

from data.base import FrameSource

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDirSource(FrameSource):
    """Yields ``(frame_id, bgr_frame)`` from sorted images in a directory."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self._path}")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        image_files = sorted(
            p for p in self._path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            raise FileNotFoundError(f"No images found in {self._path}")

        logger.info("Found %d images in %s", len(image_files), self._path)

        for frame_id, img_path in enumerate(image_files, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue
            yield frame_id, frame
