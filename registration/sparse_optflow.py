from __future__ import annotations

import copy
from typing import Optional

import cv2
import numpy as np

from registration.base import BaseRegistration


class SparseOptFlowRegistration(BaseRegistration):
    """Sparse optical flow (Lucas-Kanade) based registration."""

    def __init__(self, downscale: int = 2) -> None:
        self._downscale = max(1, int(downscale))
        self._feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_keypoints = None

    def apply(
        self,
        raw_frame: np.ndarray,
        detections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        if self._downscale > 1:
            frame = cv2.resize(
                frame, (width // self._downscale, height // self._downscale),
            )

        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self._feature_params)

        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            self._prev_keypoints = copy.copy(keypoints)
            return H

        matched_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, frame, self._prev_keypoints, None,
        )

        prev_points = []
        curr_points = []
        for i in range(len(status)):
            if status[i]:
                prev_points.append(self._prev_keypoints[i])
                curr_points.append(matched_keypoints[i])

        prev_points = np.array(prev_points)
        curr_points = np.array(curr_points)

        if prev_points.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(
                prev_points, curr_points, cv2.RANSAC,
            )
            if H is None:
                H = np.eye(2, 3)
            elif self._downscale > 1:
                H[0, 2] *= self._downscale
                H[1, 2] *= self._downscale

        self._prev_frame = frame.copy()
        self._prev_keypoints = copy.copy(keypoints)
        return H

    def reset(self) -> None:
        self._prev_frame = None
        self._prev_keypoints = None
