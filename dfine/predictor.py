"""
DFINEPredictor: drop-in replacement for mc_demo's YOLOX Predictor.

Wraps DFINEDetector to expose the same ``inference(img, timer)`` interface
that mc_demo.image_demo / imageflow_demo expect.
"""

from __future__ import annotations

import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dfine.detector import DFINEDetector


class DFINEPredictor:
    """
    Mimics the YOLOX ``Predictor`` interface used by ``mc_demo.py``.

    ``inference(img, timer)`` returns ``(detections, img_info)`` where

    * ``detections`` – ``np.ndarray`` of shape ``(N, 6)`` with columns
      ``[x1, y1, x2, y2, score, class_id]`` in original-image pixel
      coordinates, **or** an empty list when there are no detections.
    * ``img_info``   – dict with keys ``height``, ``width``, ``raw_img``,
      ``file_name``.
    """

    def __init__(self, detector: DFINEDetector) -> None:
        self.detector = detector

    def inference(
        self,
        img,
        timer,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run D-FINE detection.

        Args:
            img: file path (str) or BGR numpy array.
            timer: Timer object – ``timer.tic()`` is called before
                   detection, ``timer.toc()`` by the caller.

        Returns:
            (detections, img_info)
        """
        img_info: Dict[str, Any] = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"]  = width
        img_info["raw_img"] = img

        timer.tic()
        detections = self.detector.detect(img)      # (N, 6) numpy

        if len(detections) == 0:
            detections = []

        return detections, img_info
