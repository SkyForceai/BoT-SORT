"""Evaluation pipeline configuration dataclasses.

All evaluation behaviour is driven by :class:`EvalConfig`, which is
built from a dict (typically parsed from YAML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportingConfig:
    formats: List[str] = field(default_factory=lambda: ["console", "json"])
    output_dir: str = "eval_results"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReportingConfig:
        return cls(
            formats=d.get("formats", ["console", "json"]),
            output_dir=d.get("output_dir", cls.output_dir),
        )


@dataclass
class EvalConfig:
    """Top-level evaluation configuration.

    Typically constructed via ``EvalConfig.from_dict(yaml_section)``.

    Attributes
    ----------
    iou_threshold:
        Similarity threshold used by CLEAR (MOTA) and Identity (IDF1)
        metrics.  HOTA always evaluates at 19 thresholds from 0.05
        to 0.95 regardless of this setting.  Default: ``0.5``.
    class_groups:
        Named groups of class IDs for per-class evaluation.
        Format: ``{"person": [1], "car": [2, 3, 4]}``.
        An empty dict means no class-based slicing.
    clearml:
        When ``True`` the ``"clearml"`` reporter is automatically added
        to :attr:`reporting.formats`, sending metrics, bar-charts and
        summary tables to the active ClearML task.
    clearml_slice_scalars:
        When ``True`` (and ``clearml`` is enabled), also log per-slice
        headline metrics as **Scalars** (e.g. ``PerClass/{metric}``) for
        tabular/CSV comparison across tasks.
    annotation_frames_path:
        Optional plain-text sidecar listing annotated frame IDs. When not
        provided, evaluation assumes the full observed frame span is
        annotated. Use this to opt into partial-frame annotation.
    fps:
        Sequence frame rate used to convert frame counts into seconds
        or minutes (e.g. for ID instability rate).  Default: ``30.0``.
    """

    parser_format: str = "mot_csv"
    iou_threshold: float = 0.5
    size_bins: Dict[str, List[float]] = field(default_factory=dict)
    class_groups: Dict[str, List[int]] = field(default_factory=dict)
    density_bins: Dict[str, List[float]] = field(default_factory=dict)
    """Frame-level density bins (GT objects per frame).  Format same as
    size_bins: ``{"low": [0, 15], "medium": [15, 30], "high": [30, 1e5]}``.
    Each frame is assigned to a bin by its GT count; metrics are computed
    on the frame subsets and aggregated across sequences.
    Empty dict disables density-based aggregation."""
    metrics: List[str] = field(default_factory=list)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    clearml: bool = False
    clearml_slice_scalars: bool = False
    annotation_frames_path: str | None = None
    fps: float = 24.0
    porr: Optional[Dict[str, Any]] = None
    """Optional PORR settings (visibility threshold, pre-occlusion frame count).

    Size bins for PORR come from :attr:`size_bins` via
    :func:`evaluation.filtering.size_bins_for_porr`, not from this dict.
    """

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvalConfig:
        reporting_raw = d.get("reporting", {})
        reporting = ReportingConfig.from_dict(reporting_raw)

        # Support old ``matching.iou_threshold`` and new ``iou_threshold``.
        iou_threshold = d.get("iou_threshold", 0.5)
        matching = d.get("matching", {})
        if isinstance(matching, dict) and "iou_threshold" in matching:
            iou_threshold = matching["iou_threshold"]

        use_clearml = d.get("clearml", False)
        if use_clearml and "clearml" not in reporting.formats:
            reporting.formats = list(reporting.formats) + ["clearml"]

        return cls(
            parser_format=d.get("parser_format", cls.parser_format),
            iou_threshold=iou_threshold,
            size_bins=d.get("size_bins", {}),
            class_groups=d.get("class_groups", {}),
            density_bins=d.get("density_bins", {}),
            metrics=d.get("metrics", []),
            reporting=reporting,
            clearml=use_clearml,
            clearml_slice_scalars=bool(d.get("clearml_slice_scalars", False)),
            annotation_frames_path=d.get("annotation_frames_path"),
            fps=float(d.get("fps", 24.0)),
            porr=d.get("porr"),
        )
