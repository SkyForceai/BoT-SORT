from __future__ import annotations

import copy
from typing import Optional

import cv2
import numpy as np

from registration.base import BaseRegistration


class FeatureRegistration(BaseRegistration):
    """Feature-based registration (ORB or SIFT) with spatial filtering."""

    def __init__(self, method: str = "orb", downscale: int = 2) -> None:
        self._downscale = max(1, int(downscale))

        if method == "orb":
            self._detector = cv2.FastFeatureDetector_create(20)
            self._extractor = cv2.ORB_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "sift":
            self._detector = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20,
            )
            self._extractor = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20,
            )
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            raise ValueError(
                f"Unknown feature method: '{method}'. Available: orb, sift",
            )

        self._prev_frame: Optional[np.ndarray] = None
        self._prev_keypoints = None
        self._prev_descriptors = None

    # ------------------------------------------------------------------

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
            width = width // self._downscale
            height = height // self._downscale

        mask = np.zeros_like(frame)
        mask[int(0.02 * height):int(0.98 * height),
             int(0.02 * width):int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self._downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self._detector.detect(frame, mask)
        keypoints, descriptors = self._extractor.compute(frame, keypoints)

        if self._prev_frame is None:
            self._store(frame, keypoints, descriptors)
            return H

        if self._prev_descriptors is None or descriptors is None:
            self._store(frame, keypoints, descriptors)
            return H

        knn_matches = self._matcher.knnMatch(self._prev_descriptors, descriptors, 2)
        if len(knn_matches) == 0:
            self._store(frame, keypoints, descriptors)
            return H

        matches, spatial_distances = self._filter_matches(
            knn_matches, keypoints, width, height,
        )

        if len(matches) > 0:
            H = self._estimate_transform(matches, keypoints, spatial_distances)

        self._store(frame, keypoints, descriptors)
        return H

    # ------------------------------------------------------------------

    def _filter_matches(self, knn_matches, keypoints, width, height):
        max_spatial_dist = 0.25 * np.array([width, height])
        matches = []
        spatial_distances = []

        for m, n in knn_matches:
            if m.distance < 0.9 * n.distance:
                prev_pt = self._prev_keypoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt
                sd = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
                if abs(sd[0]) < max_spatial_dist[0] and abs(sd[1]) < max_spatial_dist[1]:
                    spatial_distances.append(sd)
                    matches.append(m)

        return matches, spatial_distances

    def _estimate_transform(self, matches, keypoints, spatial_distances):
        H = np.eye(2, 3)
        sd_arr = np.array(spatial_distances)
        mean_sd = np.mean(sd_arr, axis=0)
        std_sd = np.std(sd_arr, axis=0)
        inliers = (sd_arr - mean_sd) < 2.5 * std_sd

        prev_points = []
        curr_points = []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                prev_points.append(self._prev_keypoints[matches[i].queryIdx].pt)
                curr_points.append(keypoints[matches[i].trainIdx].pt)

        prev_points = np.array(prev_points)
        curr_points = np.array(curr_points)

        if prev_points.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(
                prev_points, curr_points, cv2.RANSAC,
            )
            if H is None:
                return np.eye(2, 3)
            if self._downscale > 1:
                H[0, 2] *= self._downscale
                H[1, 2] *= self._downscale

        return H

    # ------------------------------------------------------------------

    def _store(self, frame, keypoints, descriptors):
        self._prev_frame = frame.copy()
        self._prev_keypoints = copy.copy(keypoints)
        self._prev_descriptors = copy.copy(descriptors)

    def reset(self) -> None:
        self._prev_frame = None
        self._prev_keypoints = None
        self._prev_descriptors = None
