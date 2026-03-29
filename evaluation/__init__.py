"""MOT Evaluation Pipeline.

Quick-start::

    from evaluation import EvaluationPipeline, EvalConfig

    config = EvalConfig.from_dict(yaml_cfg["evaluation"])
    pipeline = EvaluationPipeline(config)
    report = pipeline.evaluate_and_report(sequences, output_dir="eval_results")

Metric computation is delegated to `TrackEval
<https://github.com/JonathonLuiten/TrackEval>`_.  This package provides
the CSV adapter, filtering, aggregation, and reporting layers.

Evaluation slices are the full cross-product of ``class_groups`` x
``size_bins``, plus per-class totals, per-size totals, and a global
``all`` slice.  See :class:`~evaluation.filtering.EvalSlice` for details.
"""

from pathlib import Path
from typing import Union

from evaluation.config import EvalConfig
from evaluation.filtering import ClassGroup, EvalSlice, size_bins_for_porr
from evaluation.pipeline import EvaluationPipeline
from evaluation.schema import (
    AnnotationMask,
    Detection,
    EvalSequence,
    FrameData,
    FrameResult,
    MatchedPair,
    SequenceData,
    visibility_from_visdrone_occlusion,
)


def resolve_annotation_mask(
    gt_path: Path,
    seq_name: str,
    annotation_frames_cfg: str | None,
    default_mask: AnnotationMask,
) -> AnnotationMask:
    """Resolve the annotation mask for *seq_name* with per-sequence support.

    Resolution order:

    1. **Co-located sidecar** — ``{seq_name}_annotated.txt`` in the same
       directory as *gt_path*.  Requires no config; just drop the file.
    2. **Config directory** — if *annotation_frames_cfg* is a directory,
       look for ``{seq_name}_annotated.txt`` then ``{seq_name}.txt``.
    3. **Config file** — if *annotation_frames_cfg* is a file, use it
       globally for all sequences (backwards-compatible).
    4. **Default** — return *default_mask* (the GT-observed mask from
       the parser).
    """
    gt_dir = gt_path.parent

    colocated = gt_dir / f"{seq_name}_annotated.txt"
    if colocated.is_file():
        return AnnotationMask.from_file(colocated)

    if annotation_frames_cfg:
        cfg_path = Path(annotation_frames_cfg)
        if cfg_path.is_dir():
            for candidate_name in (
                f"{seq_name}_annotated.txt",
                f"{seq_name}.txt",
            ):
                candidate = cfg_path / candidate_name
                if candidate.is_file():
                    return AnnotationMask.from_file(candidate)
        elif cfg_path.is_file():
            return AnnotationMask.from_file(cfg_path)

    return default_mask


def _resolve_gt_path_one(gt: Path, seq_name: str) -> Path:
    if gt.is_file():
        return gt
    if gt.is_dir():
        candidate = gt / f"{seq_name}_gt.csv"
        if candidate.is_file():
            return candidate
        candidate = gt / f"{seq_name}.csv"
        if candidate.is_file():
            return candidate
    return gt / f"{seq_name}_gt.csv"


def resolve_gt_path(
    gt_csv_or_dir: Union[str, Path, list, tuple],
    seq_name: str,
) -> Path:
    """Resolve a GT CSV path for *seq_name*.

    * If *gt_csv_or_dir* is a file, return it as-is (single-sequence mode).
    * If it is a directory, look for ``{seq_name}_gt.csv`` first, then
      ``{seq_name}.csv`` as a fallback.
    * If it is a ``list`` or ``tuple`` of directory paths (e.g. from YAML),
      try each in order and return the first existing match; if none match,
      return the conventional path under the first root (for logging).
    """
    if isinstance(gt_csv_or_dir, (list, tuple)):
        roots = list(gt_csv_or_dir)
        if not roots:
            raise ValueError("gt_csv path list is empty")
        first = Path(roots[0])
        for root in roots:
            candidate = _resolve_gt_path_one(Path(root), seq_name)
            if candidate.is_file():
                return candidate
        return _resolve_gt_path_one(first, seq_name)

    return _resolve_gt_path_one(Path(gt_csv_or_dir), seq_name)


__all__ = [
    "ClassGroup",
    "EvalConfig",
    "EvalSlice",
    "size_bins_for_porr",
    "EvaluationPipeline",
    "AnnotationMask",
    "Detection",
    "EvalSequence",
    "FrameData",
    "FrameResult",
    "MatchedPair",
    "SequenceData",
    "resolve_annotation_mask",
    "resolve_gt_path",
    "visibility_from_visdrone_occlusion",
]
