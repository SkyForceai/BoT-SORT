"""Parser for MOT-style CSV files.

Expected CSV layout (no header)::

    frame_id, object_id, x1, y1, x2, y2, score, class_id

Optional trailing column::

    * 9 columns: ``occlusion`` — VisDrone-style level (0 none, 1 partial, 2 heavy).

``score`` conventions for **ground truth**:

* ``1.0`` — active annotation, used for TP/FP/FN counting.
* ``0.0`` — ignored annotation (don't-care zone).  Predictions overlapping
  an ignored GT box are not penalised as false positives.  Ignored GT never
  counts towards false negatives.

For **ground truth** only, ``occlusion`` is converted to ``Detection.visibility`` using
:func:`evaluation.schema.visibility_from_visdrone_occlusion` so PORR can use visibility.
Prediction CSVs keep ``visibility`` unset even if a 9th column is present.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np

from evaluation.parsers import ParserBase, register_parser
from evaluation.schema import (
    AnnotationMask,
    Detection,
    FrameData,
    SequenceData,
    visibility_from_visdrone_occlusion,
)


def _read_csv_rows(
    path: Path,
    *,
    for_ground_truth: bool,
) -> dict[int, list[Detection]]:
    """Read a CSV file and group detections by frame_id."""
    per_frame: dict[int, list[Detection]] = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            frame_id = int(float(row[0]))
            object_id = int(float(row[1]))
            bbox = np.array([float(row[2]), float(row[3]),
                             float(row[4]), float(row[5])], dtype=np.float64)
            score = float(row[6])
            class_id = int(float(row[7]))
            occ: int | None = None
            vis: float | None = None
            if len(row) >= 9 and row[8].strip() != "":
                occ = int(float(row[8]))
                if for_ground_truth:
                    vis = visibility_from_visdrone_occlusion(occ)
            det = Detection(
                object_id=object_id,
                bbox_xyxy=bbox,
                score=score,
                class_id=class_id,
                visibility=vis,
                occlusion=occ,
            )
            per_frame[frame_id].append(det)
    return dict(per_frame)


@register_parser("mot_csv")
class MotCsvParser(ParserBase):
    """Parse MOT CSV files into canonical evaluation types."""

    def parse_predictions(self, path: Path) -> SequenceData:
        per_frame = _read_csv_rows(path, for_ground_truth=False)
        frames = {
            fid: FrameData(frame_id=fid, detections=dets)
            for fid, dets in per_frame.items()
        }
        return SequenceData(sequence_id=path.stem, frames=frames)

    def parse_ground_truth(
        self,
        path: Path,
    ) -> Tuple[SequenceData, AnnotationMask]:
        per_frame = _read_csv_rows(path, for_ground_truth=True)
        frames = {
            fid: FrameData(frame_id=fid, detections=dets)
            for fid, dets in per_frame.items()
        }
        seq = SequenceData(sequence_id=path.stem, frames=frames)
        mask = AnnotationMask.from_gt_data(seq)
        return seq, mask
