"""Size-bin, class-group, and evaluation-slice filtering.

The evaluation pipeline evaluates metrics on **slices** of the data.
A slice is a named combination of a :class:`ClassGroup` (which class IDs
to include) and a :class:`SizeBin` (which object sizes to include).

Filtering is applied **after** matching so that the matcher runs once on
all objects, and the result is narrowed per slice before metrics consume it.

Slicing always uses the **product** layout: full cross-product of
class groups x size bins, plus per-class totals, per-size totals,
and a global ``all``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from evaluation.schema import (
    IGNORED_CLASS_ID,
    Detection,
    FrameData,
    FrameResult,
    MatchedPair,
    SequenceData,
)


# ------------------------------------------------------------------
# Size bin
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SizeBin:
    """Half-open interval ``[min_val, max_val)`` on a size measure."""

    name: str
    min_val: float  # inclusive
    max_val: float  # exclusive

    def contains(self, value: float) -> bool:
        return self.min_val <= value < self.max_val


ALL_BIN = SizeBin("all", 0.0, math.inf)

# Default PORR size rows (narrow side, pixels) when ``evaluation.size_bins`` is empty.
DEFAULT_PORR_SIZE_BINS: Tuple[SizeBin, ...] = (
    SizeBin("T", 0.0, 20.0),
    SizeBin("S", 20.0, 40.0),
    SizeBin("M", 40.0, 60.0),
    SizeBin("L", 60.0, math.inf),
)


def porr_metric_row_slug(bin_name: str) -> str:
    """Stable token for PORR metric keys / JSON (alphanumeric and underscores)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", bin_name.strip())
    return s.strip("_") or "bin"


def size_bins_for_porr(cfg: Dict[str, list]) -> List[SizeBin]:
    """Size bins for :class:`~evaluation.metrics.porr.PostOcclusionRecoveryRate` rows.

    Uses ``evaluation.size_bins`` from YAML, same as the rest of the pipeline, but:

    * Drops the catch-all ``all`` bin (case-insensitive name).
    * Drops composite bins whose name contains ``"+"`` (overlapping aggregates).

    Remaining bins are sorted by ``min_val``.  If nothing remains, returns
    :data:`DEFAULT_PORR_SIZE_BINS` (legacy T/S/M/L on ``min_side``).
    """
    if not cfg:
        return list(DEFAULT_PORR_SIZE_BINS)
    bins = build_size_bins(cfg)
    filtered: List[SizeBin] = []
    for b in bins:
        core = re.split(r"[\s\-\[\(]+", b.name.strip(), maxsplit=1)[0].lower()
        if core == "all":
            continue
        if "+" in b.name:
            continue
        filtered.append(b)
    filtered.sort(key=lambda x: (x.min_val, x.max_val))
    if not filtered:
        return list(DEFAULT_PORR_SIZE_BINS)
    return filtered


# ------------------------------------------------------------------
# Class group
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ClassGroup:
    """A named set of class IDs to evaluate together.

    An **empty** ``class_ids`` set means "all classes" (no filtering).
    """

    name: str
    class_ids: frozenset[int]

    def contains(self, class_id: int) -> bool:
        return not self.class_ids or class_id in self.class_ids


ALL_CLASS_GROUP = ClassGroup("all", frozenset())


# ------------------------------------------------------------------
# Evaluation slice  (class group + size bin)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSlice:
    """Named evaluation partition combining a class filter and a size filter."""

    class_group: ClassGroup
    size_bin: SizeBin

    @property
    def name(self) -> str:
        cname = self.class_group.name
        sname = self.size_bin.name
        if cname == "all" and sname == "all":
            return "all"
        if cname == "all":
            return sname
        if sname == "all":
            return cname
        return f"{cname} / {sname}"


# ------------------------------------------------------------------
# Building bins / groups from config
# ------------------------------------------------------------------

def build_size_bins(cfg: Dict[str, list]) -> List[SizeBin]:
    """Build a list of :class:`SizeBin` from the YAML config dict.

    Expected format::

        {"small": [0, 16], "medium": [16, 32], ...}

    Values may use ``inf`` / ``.inf`` / ``1e5`` for the upper bound.
    """
    bins: List[SizeBin] = []
    for name, (lo, hi) in cfg.items():
        bins.append(SizeBin(name=name, min_val=float(lo), max_val=float(hi)))
    return bins


def build_class_groups(cfg: Dict[str, List[int]]) -> List[ClassGroup]:
    """Build a list of :class:`ClassGroup` from the YAML config dict.

    Expected format::

        {"person": [1], "car": [2, 3, 4], ...}
    """
    groups: List[ClassGroup] = []
    for name, ids in cfg.items():
        groups.append(ClassGroup(name=name, class_ids=frozenset(ids)))
    return groups


def _is_class_agnostic_ignored(det: Detection) -> bool:
    """True for ignored regions that should survive any class filter."""
    return det.is_ignored and det.class_id == IGNORED_CLASS_ID


def filter_sequence_data(
    seq_data: SequenceData,
    class_group: ClassGroup,
    size_bin: SizeBin,
) -> SequenceData:
    """Filter detections at the :class:`SequenceData` level.

    Applied **before** passing to TrackEval so that metric computation
    only considers objects in the desired class/size slice.  Both GT
    and prediction sequences should be filtered the same way.

    Class-agnostic ignored GT (``class_id == IGNORED_CLASS_ID``, ``score <= 0``)
    always passes the class filter so it can suppress false positives for
    any class group.
    """
    if not class_group.class_ids and size_bin.name == "all":
        return seq_data

    filtered_frames: dict[int, FrameData] = {}
    for fid, frame in seq_data.frames.items():
        filtered_dets = [
            d
            for d in frame.detections
            if (class_group.contains(d.class_id) or _is_class_agnostic_ignored(d))
            and size_bin.contains(d.min_side)
        ]
        filtered_frames[fid] = FrameData(
            frame_id=fid, detections=filtered_dets,
        )

    return SequenceData(
        sequence_id=seq_data.sequence_id, frames=filtered_frames,
    )


def build_eval_slices(
    size_bins: List[SizeBin],
    class_groups: List[ClassGroup],
) -> List[EvalSlice]:
    """Build evaluation slices (product layout).

    Produces the full cross-product of *class_groups* x *size_bins*,
    plus per-class totals, per-size totals, and a global ``all`` slice
    (always last).

    Parameters
    ----------
    size_bins:
        User-defined size bins (excluding the implicit ``ALL_BIN``).
    class_groups:
        User-defined class groups (excluding the implicit ``ALL_CLASS_GROUP``).

    Returns
    -------
    List[EvalSlice]
        Ordered list of slices.  The global ``all`` slice is always last.
    """
    slices: List[EvalSlice] = []

    for cg in class_groups:
        for sb in size_bins:
            slices.append(EvalSlice(cg, sb))
        slices.append(EvalSlice(cg, ALL_BIN))
    for sb in size_bins:
        slices.append(EvalSlice(ALL_CLASS_GROUP, sb))

    slices.append(EvalSlice(ALL_CLASS_GROUP, ALL_BIN))
    return slices


# ------------------------------------------------------------------
# Filtering a FrameResult by size bin
# ------------------------------------------------------------------

def filter_frame_result(
    result: FrameResult,
    size_bin: SizeBin,
) -> FrameResult:
    """Narrow *result* to objects whose size falls in *size_bin*.

    Size is measured by :attr:`Detection.min_side` (narrow side in pixels).

    - Matched pairs are kept only if the **GT** object is in the bin.
    - Unmatched GT entries are kept only if in the bin.
    - Unmatched predictions are kept only if the **prediction** is in
      the bin (prevents cross-bin FP inflation).
    """
    filtered_matched: List[MatchedPair] = []

    for mp in result.matched:
        if size_bin.contains(mp.gt.min_side):
            filtered_matched.append(mp)

    filtered_unmatched_gt = [
        d for d in result.unmatched_gt if size_bin.contains(d.min_side)
    ]

    filtered_unmatched_pred = [
        d for d in result.unmatched_pred if size_bin.contains(d.min_side)
    ]

    return FrameResult(
        frame_id=result.frame_id,
        gt_detections=result.gt_detections,
        pred_detections=result.pred_detections,
        matched=filtered_matched,
        unmatched_gt=filtered_unmatched_gt,
        unmatched_pred=filtered_unmatched_pred,
    )
