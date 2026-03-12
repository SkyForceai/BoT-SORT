"""FastReID wrapper conforming to :class:`BaseReID`.

Delegates to ``fast_reid.fast_reid_interfece.FastReIDInterface`` for the
actual model loading and feature extraction.  The ``fast_reid`` library
lives under ``reid/fast_reid/`` and uses ``from fast_reid.fastreid.*``
imports internally, so we add ``reid/`` to ``sys.path`` once at import
time so those resolve correctly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_REID_DIR = str(Path(__file__).resolve().parent)
if _REID_DIR not in sys.path:
    sys.path.insert(0, _REID_DIR)

from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: E402

from reid.base import BaseReID

logger = logging.getLogger(__name__)


class FastReIDExtractor(BaseReID):
    """ReID feature extractor backed by the FastReID library.

    Args:
        config_path:  Path to a FastReID YAML config (e.g.
                      ``reid/fast_reid/configs/MOT20/sbs_S50.yml``).
        weights_path: Path to a ``.pth`` checkpoint.
        device:       ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cuda",
    ) -> None:
        self._encoder = FastReIDInterface(config_path, weights_path, device)
        logger.info(
            "FastReID loaded  config=%s  weights=%s  device=%s",
            config_path, weights_path, device,
        )

    # ------------------------------------------------------------------
    # BaseReID contract
    # ------------------------------------------------------------------

    def extract(self, img_bgr: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract appearance features for detected boxes.

        Args:
            img_bgr: (H, W, 3) uint8 BGR full frame.
            bboxes:  (N, 4) float32 ``[x1, y1, x2, y2]`` boxes.

        Returns:
            (N, D) float32 feature matrix (L2-normalised).
            Returns ``np.zeros((0, 2048))`` when *bboxes* is empty.
        """
        if bboxes is None or len(bboxes) == 0:
            return np.zeros((0, 2048), dtype=np.float32)

        return self._encoder.inference(img_bgr, bboxes)

    def warmup(self) -> None:
        dummy = np.zeros((128, 64, 3), dtype=np.uint8)
        dummy_box = np.array([[0, 0, 64, 128]], dtype=np.float32)
        self._encoder.inference(dummy, dummy_box)
        logger.info("FastReID warm-up done")
