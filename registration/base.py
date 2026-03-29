from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseRegistration(ABC):
    """Abstract base class for frame-to-frame image registration (GMC).

    Concrete subclasses compute a 2×3 affine transform that maps pixel
    coordinates from the *previous* frame to the *current* frame, enabling
    global motion compensation in the tracker.
    """

    @abstractmethod
    def apply(
        self,
        raw_frame: np.ndarray,
        detections: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate the affine warp from the previous frame to *raw_frame*.

        Parameters
        ----------
        raw_frame:
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
        detections:
            Optional ``(N, 4+)`` array of detections in *tlbr* format.
            Moving-object regions can be masked out to improve estimation.

        Returns
        -------
        np.ndarray
            2×3 affine transformation matrix (dtype ``float64``).
        """
        ...

    def reset(self) -> None:
        """Clear internal state (call between sequences)."""
