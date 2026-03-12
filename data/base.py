"""Abstract base class for frame sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import numpy as np


class FrameSource(ABC):
    """Iterable that yields ``(frame_id, bgr_frame)`` pairs.

    Every concrete source (video file, image directory, …) must implement
    ``__iter__``.  The ``frame_id`` is a 1-based sequential integer.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        ...
