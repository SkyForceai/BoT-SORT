"""
CSVExporter: write per-run detections.csv and tracks.csv files.

Format
------
detections.csv  (one row per detection, all detections across every frame)
    frame_id, x1, y1, x2, y2, score, class_id [, feat_0, ..., feat_N]
    • Minimum 7 columns.
    • Columns 7+ are feature-vector values if features were present.
    • Boxes in original-image pixel coords (tlbr / xyxy).

tracks.csv  (one row per active track output, all frames)
    frame_id, track_id, x1, y1, x2, y2, score, class_id [, feat_0, ..., feat_N]
    • Minimum 8 columns.
    • Feature columns present iff detection features were supplied.
    • Boxes in original-image pixel coords (tlbr / xyxy).

Both files use comma-separated values and no header row.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


class CSVExporter:
    """
    Accumulate per-frame detections and tracks, then write two CSV files.

    Usage::

        exporter = CSVExporter("/path/to/output_dir")

        # inside the per-frame loop:
        exporter.add_detections(frame_id, detections)  # (N, 6+) or []
        exporter.add_tracks(frame_id, online_targets)  # list[STrack]

        # after the loop:
        exporter.save()  # writes detections.csv and tracks.csv
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = str(output_dir)
        self._det_rows:   List[List] = []
        self._track_rows: List[List] = []
        self._has_features: bool = False   # set True when feature cols seen

    # ------------------------------------------------------------------
    # Per-frame collection
    # ------------------------------------------------------------------

    def add_detections(self, frame_id: int, detections) -> None:
        """
        Record raw detector outputs for one frame.

        Args:
            frame_id:   1-based frame index.
            detections: ``(N, 6+)`` numpy array with columns
                        ``[x1, y1, x2, y2, score, class_id, *features]``,
                        or an empty list / empty array when there are no
                        detections.
        """
        if not hasattr(detections, '__len__') or len(detections) == 0:
            return

        arr = np.asarray(detections)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return

        if arr.shape[1] > 6:
            self._has_features = True

        for row in arr:
            x1, y1, x2, y2, score, cls = row[:6]
            entry: List = [int(frame_id), float(x1), float(y1),
                           float(x2), float(y2), float(score), float(cls)]
            if arr.shape[1] > 6:
                entry.extend(float(v) for v in row[6:])
            self._det_rows.append(entry)

    def add_tracks(self, frame_id: int, online_targets) -> None:
        """
        Record tracker outputs for one frame.

        Args:
            frame_id:       1-based frame index.
            online_targets: list of ``STrack`` objects returned by
                            ``tracker.update()``.  Each object must expose
                            ``.tlbr`` (4-float array), ``.track_id``,
                            ``.score``, ``.cls``, and optionally
                            ``.curr_feat`` (numpy array or None).
        """
        for t in online_targets:
            tlbr = t.tlbr
            x1, y1, x2, y2 = float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])
            entry: List = [int(frame_id), int(t.track_id),
                           x1, y1, x2, y2,
                           float(t.score), float(t.cls)]

            feat = getattr(t, "curr_feat", None)
            if feat is not None and self._has_features:
                entry.extend(float(v) for v in feat)

            self._track_rows.append(entry)

    # ------------------------------------------------------------------
    # Flush to disk
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write ``detections.csv`` and ``tracks.csv`` to ``output_dir``."""
        os.makedirs(self.output_dir, exist_ok=True)

        det_path   = Path(self.output_dir) / "detections.csv"
        track_path = Path(self.output_dir) / "tracks.csv"

        self._write_csv(det_path, self._det_rows)
        self._write_csv(track_path, self._track_rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: List[List]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
