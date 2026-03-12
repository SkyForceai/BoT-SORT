"""Data-source package — public API for frame iteration."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple, Type

import numpy as np

from data.base import FrameSource
from data.image_dir import ImageDirSource
from data.video import VideoSource

_REGISTRY: Dict[str, Type[FrameSource]] = {
    "video": VideoSource,
    "image_dir": ImageDirSource,
}


def build_source(data_cfg: Dict) -> FrameSource:
    """Instantiate the appropriate :class:`FrameSource` from *data_cfg*.

    Required keys in *data_cfg*:
      - ``source_type``: key into the source registry (e.g. ``"video"``).
      - ``path``: filesystem path forwarded to the source constructor.
    """
    source_type = data_cfg["source_type"]
    cls = _REGISTRY.get(source_type)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown source_type '{source_type}'. Supported: {supported}"
        )
    return cls(path=data_cfg["path"])


def iter_frames(data_cfg: Dict) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_id, bgr_frame)`` from the configured data source."""
    return iter(build_source(data_cfg))


__all__ = ["FrameSource", "build_source", "iter_frames"]
