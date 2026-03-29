"""Data-source package — public API for frame iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Type

import numpy as np

from data.base import FrameSource
from data.image_dir import IMAGE_EXTENSIONS, ImageDirSource
from data.video import VideoSource

_REGISTRY: Dict[str, Type[FrameSource]] = {
    "video": VideoSource,
    "image_dir": ImageDirSource,
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


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


def resolve_sequences(data_cfg: Dict) -> List[Tuple[str, Dict]]:
    """Resolve ``data_cfg`` into a list of ``(seq_name, per_seq_cfg)`` pairs.

    Optional ``data_cfg["paths"]`` (non-empty list of roots) runs the same
    resolution as ``path`` for each entry and concatenates results. When
    ``paths`` is set, ``path`` is ignored for discovery (per-sequence cfgs
    still carry a concrete ``path`` string).

    Single-vs-multi-sequence mode is auto-detected from each root in
    ``paths`` or from ``data_cfg["path"]``:

    * **image_dir**: if *path* contains image files it is a single sequence;
      if it contains only subdirectories each subdirectory is a sequence.
    * **video**: if *path* is a video file it is a single sequence;
      if it is a directory every video file inside is a sequence.
    """
    extra_roots = data_cfg.get("paths")
    if extra_roots is not None:
        if not isinstance(extra_roots, (list, tuple)):
            raise TypeError('data.paths must be a list (or tuple) of path strings')
        if len(extra_roots) == 0:
            raise ValueError("data.paths is empty — omit it and use data.path instead")
        merged: List[Tuple[str, Dict]] = []
        base_cfg = {k: v for k, v in data_cfg.items() if k != "paths"}
        for root in extra_roots:
            merged.extend(resolve_sequences({**base_cfg, "path": str(root)}))
        return merged

    p = Path(data_cfg["path"])
    source_type = data_cfg["source_type"]

    if source_type == "image_dir":
        if not p.is_dir():
            raise FileNotFoundError(f"Path is not a directory: {p}")
        has_images = any(
            c.suffix.lower() in IMAGE_EXTENSIONS for c in p.iterdir() if c.is_file()
        )
        if has_images:
            return [(p.name, data_cfg)]
        subdirs = sorted(d for d in p.iterdir() if d.is_dir())
        if not subdirs:
            raise FileNotFoundError(f"No images or subdirectories in: {p}")
        return [(d.name, {**data_cfg, "path": str(d)}) for d in subdirs]

    if source_type == "video":
        if p.is_file():
            return [(p.stem, data_cfg)]
        if p.is_dir():
            videos = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not videos:
                raise FileNotFoundError(f"No video files in: {p}")
            return [(v.stem, {**data_cfg, "path": str(v)}) for v in videos]
        raise FileNotFoundError(f"Video path not found: {p}")

    raise ValueError(f"Cannot resolve sequences for source_type '{source_type}'")


__all__ = ["FrameSource", "build_source", "iter_frames", "resolve_sequences"]
