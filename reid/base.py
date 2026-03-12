from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseReID(ABC):
    """Abstract base class for Re-Identification feature extractors."""

    @abstractmethod
    def extract(self, img_bgr: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract appearance features for cropped detections.

        Args:
            img_bgr: (H, W, 3) uint8 BGR full frame.
            bboxes:  (N, 4) float32 ``[x1, y1, x2, y2]`` boxes.

        Returns:
            (N, D) float32 feature matrix.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass for JIT / CUDA warm-up."""
        ...
