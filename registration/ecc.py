from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from registration.base import BaseRegistration


class ECCRegistration(BaseRegistration):
    """ECC (Enhanced Correlation Coefficient) based registration."""

    def __init__(
        self,
        downscale: int = 2,
        num_iterations: int = 5000,
        eps: float = 1e-6,
    ) -> None:
        self._downscale = max(1, int(downscale))
        self._warp_mode = cv2.MOTION_EUCLIDEAN
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            num_iterations,
            eps,
        )
        self._prev_frame: Optional[np.ndarray] = None

    def apply(
        self,
        raw_frame: np.ndarray,
        detections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self._downscale > 1:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(
                frame, (width // self._downscale, height // self._downscale),
            )

        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return H

        try:
            _, H = cv2.findTransformECC(
                self._prev_frame, frame, H,
                self._warp_mode, self._criteria, None, 1,
            )
        except cv2.error:
            pass

        self._prev_frame = frame.copy()
        return H

    def reset(self) -> None:
        self._prev_frame = None
