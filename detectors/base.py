from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for object detectors.

    All concrete detectors must implement ``detect`` and return an (N, 6)
    float32 numpy array with columns ``[x1, y1, x2, y2, score, class_id]``
    in original-image pixel coordinates.
    """

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run detection on a single BGR frame.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image (OpenCV convention).

        Returns:
            (N, 6) float32 array ``[x1, y1, x2, y2, score, class_id]``.
            Returns ``np.empty((0, 6), dtype=np.float32)`` when no
            detections pass the confidence threshold.
        """
        ...
