"""Canonical data classes for the MOT evaluation pipeline.

Every component in the pipeline operates on these types.  ``Detection``
is the atomic unit; ``FrameData`` groups detections by frame;
``SequenceData`` groups frames by video/sequence.  ``FrameResult``
carries the output of the matching step and is consumed by metrics.

Result containers (``BinResult``, ``SequenceResult``, ``EvalReport``)
are the output side: produced by the pipeline for each (sequence, bin)
pair and consumed by reporters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from evaluation.config import EvalConfig

IGNORED_CLASS_ID: int = -1
"""Sentinel class ID for class-agnostic ignored regions.

GT detections with this class ID (and ``score <= 0``) represent spatial
"don't-care" zones.  They survive class-group filtering and suppress
false-positive penalties for predictions that overlap them.
"""


def visibility_from_visdrone_occlusion(occlusion: int) -> float:
    """Map VisDrone occlusion label to a visibility fraction for PORR and related metrics.

    Convention (occlusion = fraction of the instance that is occluded):

    * **0** — no occlusion (0% occluded) → visibility **1.0**
    * **1** — ~25% occluded → visibility **0.75**
    * **2** — more than ~50% occluded → visibility **0.25** (representative hint)

    PORR treats frames with visibility **≤** ``visibility_occluded_max`` (default 0.75) as
    low-visibility / occluded; values are chosen so 0 is clearly above that threshold and
    1–2 are at or below it (``0.75`` is not ``> 0.75`` in the clear-visibility loop).
    """
    if occlusion == 0:
        return 1.0
    if occlusion == 1:
        return 0.75
    if occlusion == 2:
        return 0.25
    raise ValueError(f"VisDrone occlusion must be 0, 1, or 2; got {occlusion!r}")


# ------------------------------------------------------------------
# Atomic types
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """Single object instance in a frame (GT or prediction).

    Attributes:
        object_id: track_id for predictions, gt_id for ground truth.
        bbox_xyxy:  (4,) array in *x1 y1 x2 y2* format (absolute pixels).
        score:      Confidence score.  Conventionally 1.0 for GT; 0.0 for
            ignored annotations that should not be evaluated (predictions
            overlapping ignored GT are not penalised as false positives).
        class_id:   Semantic class label.  ``-1`` (:data:`IGNORED_CLASS_ID`)
            marks a class-agnostic ignored region.
        visibility: Optional GT visibility in ``(0, 1]`` (fraction of bbox area visible).
            For GT loaded from MOT CSV with a VisDrone ``occlusion`` column, this is set
            via :func:`visibility_from_visdrone_occlusion`.  Otherwise set in code; ``None``
            means unknown (e.g. 8-column CSV).
        occlusion: Optional VisDrone-style occlusion level (0 none, 1 partial, 2 heavy)
            from the optional 9th MOT CSV column (see ``mot_csv``).
    """

    object_id: int
    bbox_xyxy: np.ndarray
    score: float
    class_id: int
    visibility: float | None = None
    occlusion: int | None = None

    @cached_property
    def is_ignored(self) -> bool:
        """True when this GT detection should be excluded from TP/FN counts."""
        return self.score <= 0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @cached_property
    def area(self) -> float:
        w = float(self.bbox_xyxy[2] - self.bbox_xyxy[0])
        h = float(self.bbox_xyxy[3] - self.bbox_xyxy[1])
        return w * h

    @cached_property
    def min_side(self) -> float:
        w = float(self.bbox_xyxy[2] - self.bbox_xyxy[0])
        h = float(self.bbox_xyxy[3] - self.bbox_xyxy[1])
        return min(w, h)

    def visible_fraction(self) -> float:
        """Fraction of bbox visible; 1.0 when visibility is unknown."""
        if self.visibility is None:
            return 1.0
        return float(self.visibility)


# ------------------------------------------------------------------
# Frame / sequence containers
# ------------------------------------------------------------------

@dataclass
class FrameData:
    """All detections in a single frame."""

    frame_id: int
    detections: List[Detection] = field(default_factory=list)


@dataclass
class SequenceData:
    """All frames for a single video / sequence."""

    sequence_id: str
    frames: dict[int, FrameData] = field(default_factory=dict)

    @property
    def frame_ids(self) -> set[int]:
        return set(self.frames.keys())


# ------------------------------------------------------------------
# Annotation mask
# ------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationMask:
    """Marks which frames carry ground-truth annotations.

    Explicit masks evaluate exactly the listed frames. The default mask
    produced from GT rows means "GT-observed frames"; the evaluation
    pipeline expands it to the full observed frame span unless an
    explicit sidecar mask is provided.
    """

    annotated_frame_ids: frozenset[int]
    is_explicit: bool = True

    def is_annotated(self, frame_id: int) -> bool:
        return frame_id in self.annotated_frame_ids

    @classmethod
    def from_gt_data(cls, gt: SequenceData) -> AnnotationMask:
        """Construct the default GT-observed mask from CSV rows."""
        return cls(frozenset(gt.frames.keys()), is_explicit=False)

    @classmethod
    def from_frame_ids(cls, frame_ids) -> AnnotationMask:
        """Construct from an explicit iterable of frame IDs."""
        return cls(frozenset(frame_ids), is_explicit=True)

    @classmethod
    def from_file(cls, path: str | Path) -> AnnotationMask:
        """Load annotated frame IDs from a plain-text sidecar file.

        The file may contain one frame ID per line or comma/whitespace-
        separated IDs. Inline ``#`` comments are ignored.
        """
        frame_ids: set[int] = set()
        with open(path) as fh:
            for raw_line in fh:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                for token in line.replace(",", " ").split():
                    frame_ids.add(int(float(token)))
        return cls(frozenset(frame_ids), is_explicit=True)

    def resolve_frame_ids(
        self,
        ground_truth: SequenceData,
        predictions: SequenceData,
    ) -> frozenset[int]:
        """Return the frame IDs that should actually be evaluated."""
        if self.is_explicit:
            return self.annotated_frame_ids

        observed_ids = ground_truth.frame_ids | predictions.frame_ids
        if not observed_ids:
            return frozenset()
        return frozenset(range(1, max(observed_ids) + 1))


# ------------------------------------------------------------------
# Evaluation input bundle
# ------------------------------------------------------------------

@dataclass
class EvalSequence:
    """One sequence ready for evaluation."""

    sequence_id: str
    predictions: SequenceData
    ground_truth: SequenceData
    annotation_mask: AnnotationMask


# ------------------------------------------------------------------
# Matching output
# ------------------------------------------------------------------

@dataclass
class MatchedPair:
    """A GT detection matched to a predicted detection."""

    gt: Detection
    pred: Detection
    iou: float


@dataclass
class FrameResult:
    """Complete per-frame evaluation record produced by the matcher.

    Carries *both* the match result and all raw detections so that
    metrics which need custom matching can access the original data.
    """

    frame_id: int
    gt_detections: List[Detection]
    pred_detections: List[Detection]
    matched: List[MatchedPair]
    unmatched_gt: List[Detection]
    unmatched_pred: List[Detection]


# ------------------------------------------------------------------
# Result containers (consumed by reporters)
# ------------------------------------------------------------------

@dataclass
class BinResult:
    """Metric values for a single evaluation slice."""

    bin_name: str
    metric_values: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SequenceResult:
    """All bin results for one sequence, plus annotation metadata."""

    sequence_id: str
    num_annotated_frames: int
    num_total_frames: int
    bins: Dict[str, BinResult] = field(default_factory=dict)
    never_matched_gt: List[Dict[str, Any]] = field(default_factory=list)
    """GT objects never matched in any frame.
    Each entry: ``{"frame_id": int, "object_id": int, "class_id": int}``."""


@dataclass
class EvalReport:
    """Top-level evaluation report."""

    sequences: Dict[str, SequenceResult]
    aggregated_bins: Dict[str, BinResult]
    overall: Dict[str, Dict[str, float]]
    config: EvalConfig
    density_aggregated: Dict[str, Dict[str, BinResult]] = field(
        default_factory=dict,
    )
    """Per-density-bin aggregation (frame-level).  Outer key is density
    bin name (e.g. ``"low"``), inner dict has the same structure as
    :attr:`aggregated_bins` (eval-slice name -> BinResult).  Each frame
    is assigned to a bin by its GT object count."""
