"""Main evaluation pipeline using TrackEval for metric computation.

:class:`EvaluationPipeline` wires together parsing, filtering,
the TrackEval adapter, metric computation, aggregation, and reporting
into a single ``evaluate`` call.

The pipeline evaluates metrics on **slices** of the data, where each
slice is a combination of a class group and a size bin.  See
:func:`~evaluation.filtering.build_eval_slices` for the available
slicing modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from evaluation.adapter import (
    build_trackeval_data,
    build_trackeval_data_from_frame_results,
    global_match_sequence,
)
from evaluation.schema import BinResult, EvalReport, SequenceResult
from evaluation.config import EvalConfig
from evaluation.filtering import (
    ALL_BIN,
    ALL_CLASS_GROUP,
    ClassGroup,
    EvalSlice,
    SizeBin,
    build_class_groups,
    build_eval_slices,
    build_size_bins,
    filter_frame_result,
    filter_sequence_data,
    size_bins_for_porr,
)
from evaluation.parsers import build_parser
from evaluation.reporting import build_reporter
from evaluation.metrics.porr import format_porr_metrics
from evaluation.schema import EvalSequence, SequenceData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# GT track median size
# ------------------------------------------------------------------

def _compute_gt_median_sizes(
    gt: SequenceData,
) -> Dict[int, float]:
    """Return ``{object_id: median_min_side}`` for every GT track.

    Each GT track's ``min_side`` is recorded in every frame it appears.
    The median is used as the track's representative size so it can be
    assigned to exactly one size bin (avoiding double-counting across
    overlapping bins).
    """
    from collections import defaultdict

    track_sizes: dict[int, list[float]] = defaultdict(list)
    for frame in gt.frames.values():
        for det in frame.detections:
            track_sizes[det.object_id].append(det.min_side)
    return {
        oid: float(np.median(sizes))
        for oid, sizes in track_sizes.items()
    }


def _find_never_matched_gt(
    match_results: list,
) -> List[Dict[str, Any]]:
    """Identify GT objects that were never matched in any frame.

    Returns a sorted list of ``{"frame_id", "object_id", "class_id"}``
    dicts — one entry per frame appearance of each never-matched object.
    """
    ever_matched: set[int] = set()
    gt_appearances: dict[int, list[tuple[int, int]]] = {}

    for fr in match_results:
        for mp in fr.matched:
            ever_matched.add(mp.gt.object_id)
        all_gt = [mp.gt for mp in fr.matched] + list(fr.unmatched_gt)
        for g in all_gt:
            gt_appearances.setdefault(g.object_id, []).append(
                (fr.frame_id, g.class_id),
            )

    rows: List[Dict[str, Any]] = []
    for oid in sorted(gt_appearances):
        if oid in ever_matched:
            continue
        for fid, cid in sorted(gt_appearances[oid]):
            rows.append({"frame_id": fid, "object_id": oid, "class_id": cid})
    return rows


# ------------------------------------------------------------------
# TrackEval metric wiring
# ------------------------------------------------------------------

_METRIC_CLASS_MAP: Dict[str, tuple[str, str]] = {
    "hota": ("trackeval.metrics.hota", "HOTA"),
    "mota": ("trackeval.metrics.clear", "CLEAR"),
    "clear": ("trackeval.metrics.clear", "CLEAR"),
    "idf1": ("trackeval.metrics.identity", "Identity"),
    "identity": ("trackeval.metrics.identity", "Identity"),
    "coverage": ("evaluation.metrics.coverage", "TrackCoverage"),
    "pd": ("evaluation.metrics.pd", "ProbabilityOfDetection"),
    "id_instability": ("evaluation.metrics.id_instability", "IDInstabilityRate"),
    "realtime_kpi": ("evaluation.metrics.realtime_kpi", "RealTimeKPI"),
}

_CLASS_TO_OUR_NAME: Dict[str, str] = {
    "HOTA": "hota",
    "CLEAR": "mota",
    "Identity": "idf1",
    "TrackCoverage": "coverage",
    "ProbabilityOfDetection": "pd",
    "IDInstabilityRate": "id_instability",
    "RealTimeKPI": "realtime_kpi",
}


def _import_metric_class(module_path: str, class_name: str):
    """Dynamically import a TrackEval metric class."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_trackeval_metrics(
    metric_names: List[str],
    iou_threshold: float,
    fps: float = 24.0,
) -> list:
    """Instantiate TrackEval metric objects from config metric names."""
    metrics = []
    seen_classes: set[str] = set()

    for name in metric_names:
        if name == "porr":
            continue
        if name not in _METRIC_CLASS_MAP:
            raise ValueError(
                f"Unknown metric: '{name}'. "
                f"Available: {list(_METRIC_CLASS_MAP)} + ['porr']"
            )
        module_path, class_name = _METRIC_CLASS_MAP[name]
        if class_name in seen_classes:
            continue
        seen_classes.add(class_name)

        cls = _import_metric_class(module_path, class_name)
        config: Dict[str, Any] = {"PRINT_CONFIG": False}
        if class_name in ("CLEAR", "Identity", "TrackCoverage",
                          "ProbabilityOfDetection", "IDInstabilityRate",
                          "RealTimeKPI"):
            config["THRESHOLD"] = iou_threshold
        if class_name in ("IDInstabilityRate", "RealTimeKPI"):
            config["FPS"] = fps
        metrics.append(cls(config))

    return metrics


# ------------------------------------------------------------------
# Result formatting (TrackEval dicts -> our flat float dicts)
# ------------------------------------------------------------------

_HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)


def _format_results(metric, raw: dict) -> Dict[str, float]:
    """Convert a TrackEval result dict to our flat ``{key: float}`` dict."""
    class_name = type(metric).__name__

    if class_name == "HOTA":
        result: Dict[str, float] = {
            "hota": float(np.mean(raw["HOTA"])),
            "deta": float(np.mean(raw["DetA"])),
            "assa": float(np.mean(raw["AssA"])),
            "loca": float(np.mean(raw["LocA"])),
        }
        for ai, alpha in enumerate(_HOTA_ALPHAS):
            tag = f"{alpha:.2f}"
            result[f"hota@{tag}"] = float(raw["HOTA"][ai])
            result[f"deta@{tag}"] = float(raw["DetA"][ai])
            result[f"assa@{tag}"] = float(raw["AssA"][ai])
        return result

    if class_name == "CLEAR":
        gt_total = float(raw["CLR_TP"] + raw["CLR_FN"])
        num_gt_objects = float(raw["MT"] + raw["PT"] + raw["ML"])
        num_frames = float(raw.get("CLR_Frames", 0))
        num_fp = float(raw["CLR_FP"])
        far = num_fp / num_frames if num_frames > 0 else 0.0
        return {
            "mota": float(raw["MOTA"]),
            "motp": float(raw["MOTP"]),
            "recall": float(raw["CLR_Re"]),
            "precision": float(raw["CLR_Pr"]),
            "num_tp": float(raw["CLR_TP"]),
            "num_fp": num_fp,
            "num_fn": float(raw["CLR_FN"]),
            "num_idsw": float(raw["IDSW"]),
            "frag": float(raw["Frag"]),
            "mt": float(raw["MT"]),
            "pt": float(raw["PT"]),
            "ml": float(raw["ML"]),
            "num_gt": gt_total,
            "num_gt_objects": num_gt_objects,
            "num_frames": num_frames,
            "far": far,
        }

    if class_name == "Identity":
        return {
            "idf1": float(raw["IDF1"]),
            "idp": float(raw["IDP"]),
            "idr": float(raw["IDR"]),
            "idtp": float(raw["IDTP"]),
            "idfp": float(raw["IDFP"]),
            "idfn": float(raw["IDFN"]),
        }

    if class_name == "TrackCoverage":
        return {
            "coverage": float(raw["coverage"]) / 100.0,
            "coverage_per_track": float(raw["coverage_per_track"]) / 100.0,
            "num_covered_frames": float(raw["num_covered_frames"]),
            "num_visible_frames": float(raw["num_visible_frames"]),
            "num_gt_tracks": float(raw["num_gt_tracks"]),
        }

    if class_name == "ProbabilityOfDetection":
        return {
            "pd": float(raw["pd"]) / 100.0,
            "num_detected": float(raw["num_detected"]),
            "num_gt_objects": float(raw["num_gt_objects"]),
        }

    if class_name == "IDInstabilityRate":
        return {
            "id_instability": float(raw["id_instability_rate"]),
            "num_gt_objects": float(raw["num_gt_objects"]),
            "total_idsw": float(raw["total_idsw"]),
        }

    if class_name == "RealTimeKPI":
        return {
            "tid_mean_frames": float(raw["tid_mean_frames"]),
            "tid_mean_sec": float(raw["tid_mean_sec"]),
            "tid_median_frames": float(raw["tid_median_frames"]),
            "tid_ratio_immediate": float(raw["tid_ratio_immediate"]),
            "tid_ratio_within_1s": float(raw["tid_ratio_within_1s"]),
            "num_gt_objects": float(raw["num_gt_objects"]),
            "num_never_matched": float(raw["num_never_matched"]),
        }

    raise ValueError(f"Unknown metric class: {class_name}")


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class EvaluationPipeline:
    """End-to-end MOT evaluation using TrackEval metrics."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.parser = build_parser(config.parser_format)

        size_bins = build_size_bins(config.size_bins)
        class_groups = (
            build_class_groups(config.class_groups)
            if config.class_groups
            else []
        )
        self.eval_slices: List[EvalSlice] = build_eval_slices(
            size_bins, class_groups,
        )

        self.density_bins: List[SizeBin] = (
            build_size_bins(config.density_bins)
            if config.density_bins
            else []
        )

        self.metric_names: List[str] = list(config.metrics)
        self._trackeval_metrics = _build_trackeval_metrics(
            self.metric_names, config.iou_threshold, fps=config.fps,
        )
        self._porr_metric = None
        if "porr" in self.metric_names:
            from evaluation.metrics.porr import PostOcclusionRecoveryRate

            self._porr_metric = PostOcclusionRecoveryRate(
                dict(config.porr or {}),
                size_bins=size_bins_for_porr(config.size_bins),
            )
        self.reporter = build_reporter(config.reporting.formats)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, sequences: List[EvalSequence]) -> EvalReport:
        """Run evaluation on all *sequences* and return a report."""
        seq_results: List[SequenceResult] = []
        raw_by_seq: Dict[str, Dict[str, Dict[str, dict]]] = {}
        density_raw_by_seq: Dict[
            str, Dict[str, Dict[str, Dict[str, dict]]]
        ] = {}

        for seq in sequences:
            sr, raw_per_slice, density_raw = self._evaluate_sequence(seq)
            seq_results.append(sr)
            raw_by_seq[seq.sequence_id] = raw_per_slice
            density_raw_by_seq[seq.sequence_id] = density_raw

        aggregated = self._aggregate(seq_results, raw_by_seq)

        overall_bin = aggregated.get("all")
        overall = overall_bin.metric_values if overall_bin else {}

        density_aggregated: Dict[str, Dict[str, BinResult]] = {}
        if self.density_bins:
            for dbin in self.density_bins:
                dbin_raw = {
                    sr.sequence_id: density_raw_by_seq.get(
                        sr.sequence_id, {},
                    ).get(dbin.name, {})
                    for sr in seq_results
                }
                dbin_srs = [
                    sr for sr in seq_results
                    if density_raw_by_seq.get(
                        sr.sequence_id, {},
                    ).get(dbin.name)
                ]
                if dbin_srs:
                    density_aggregated[dbin.name] = self._aggregate(
                        dbin_srs, dbin_raw,
                    )
                    logger.info(
                        "Density bin '%s': %d sequences with matching frames",
                        dbin.name, len(dbin_srs),
                    )

        return EvalReport(
            sequences={sr.sequence_id: sr for sr in seq_results},
            aggregated_bins=aggregated,
            overall=overall,
            config=self.config,
            density_aggregated=density_aggregated,
        )

    def evaluate_and_report(
        self,
        sequences: List[EvalSequence],
        output_dir: Path | str,
    ) -> EvalReport:
        """Evaluate and write the report in one call."""
        report = self.evaluate(sequences)
        self.reporter.report(report, Path(output_dir))
        return report

    # ------------------------------------------------------------------
    # Per-sequence logic
    # ------------------------------------------------------------------

    def _evaluate_sequence(
        self,
        seq: EvalSequence,
    ) -> tuple[
        SequenceResult,
        Dict[str, Dict[str, dict]],
        Dict[str, Dict[str, Dict[str, dict]]],
    ]:
        """Evaluate one sequence across all configured slices.

        Uses the **match-once-then-slice** strategy: for each class
        group, matching is performed once on all sizes using CLEAR-style
        Hungarian assignment.  The match results are then sliced by GT
        size bin so that cross-size TPs are preserved and FPs are
        attributed by prediction size.

        Returns
        -------
        sr : SequenceResult
        raw_per_slice : {slice_name: {metric_name: raw_dict}}
        density_raw : {density_bin: {slice_name: {metric_name: raw_dict}}}
            Frame-level density evaluation raw results.
        """
        cg_to_slices: dict[ClassGroup, list[EvalSlice]] = {}
        for es in self.eval_slices:
            cg_to_slices.setdefault(es.class_group, []).append(es)

        bin_results: Dict[str, BinResult] = {}
        raw_per_slice: Dict[str, Dict[str, dict]] = {}
        density_raw: Dict[str, Dict[str, Dict[str, dict]]] = {}
        global_match_results: list | None = None

        for class_group, slices in cg_to_slices.items():
            class_gt = filter_sequence_data(
                seq.ground_truth, class_group, ALL_BIN,
            )
            class_pred = filter_sequence_data(
                seq.predictions, class_group, ALL_BIN,
            )

            match_results = global_match_sequence(
                class_gt, class_pred, seq.annotation_mask,
                iou_threshold=self.config.iou_threshold,
            )

            if class_group == ALL_CLASS_GROUP:
                global_match_results = match_results

            gt_median_sizes = _compute_gt_median_sizes(class_gt)

            for eval_slice in slices:
                sliced = [
                    filter_frame_result(fr, eval_slice.size_bin)
                    for fr in match_results
                ]

                data = build_trackeval_data_from_frame_results(sliced)

                metric_values: Dict[str, Dict[str, float]] = {}
                raw_for_slice: Dict[str, dict] = {}

                for metric in self._trackeval_metrics:
                    raw = metric.eval_sequence(data)
                    name = _CLASS_TO_OUR_NAME[type(metric).__name__]
                    metric_values[name] = _format_results(metric, raw)
                    raw_for_slice[name] = raw

                exclusive_count = sum(
                    1 for median in gt_median_sizes.values()
                    if eval_slice.size_bin.contains(median)
                )
                raw_for_slice["_exclusive_gt_objects"] = exclusive_count
                if "mota" in metric_values:
                    metric_values["mota"]["num_gt_objects"] = float(
                        exclusive_count,
                    )

                if (
                    self._porr_metric is not None
                    and eval_slice.size_bin == ALL_BIN
                ):
                    raw_porr = self._porr_metric.eval_from_match_results(
                        match_results,
                        fps=self.config.fps,
                    )
                    raw_for_slice["porr"] = raw_porr
                    metric_values["porr"] = format_porr_metrics(raw_porr)

                bin_results[eval_slice.name] = BinResult(
                    bin_name=eval_slice.name,
                    metric_values=metric_values,
                )
                raw_per_slice[eval_slice.name] = raw_for_slice

            # -- Frame-level density bin evaluation -----------------------
            for dbin in self.density_bins:
                density_frames = [
                    fr for fr in match_results
                    if dbin.contains(len(fr.gt_detections))
                ]
                if not density_frames:
                    continue

                density_gt_oids = {
                    d.object_id
                    for fr in density_frames
                    for d in fr.gt_detections
                }

                for eval_slice in slices:
                    sliced = [
                        filter_frame_result(fr, eval_slice.size_bin)
                        for fr in density_frames
                    ]
                    data = build_trackeval_data_from_frame_results(sliced)

                    raw_for_dslice: Dict[str, dict] = {}
                    for metric in self._trackeval_metrics:
                        raw = metric.eval_sequence(data)
                        name = _CLASS_TO_OUR_NAME[type(metric).__name__]
                        raw_for_dslice[name] = raw

                    exclusive_count = sum(
                        1 for oid, median in gt_median_sizes.items()
                        if oid in density_gt_oids
                        and eval_slice.size_bin.contains(median)
                    )
                    raw_for_dslice["_exclusive_gt_objects"] = exclusive_count

                    density_raw.setdefault(
                        dbin.name, {},
                    )[eval_slice.name] = raw_for_dslice

        annotated_ids = seq.annotation_mask.resolve_frame_ids(
            seq.ground_truth, seq.predictions,
        )
        total_frame_ids = (
            seq.predictions.frame_ids
            | seq.ground_truth.frame_ids
            | set(annotated_ids)
        )

        never_matched: List[Dict[str, Any]] = []
        if global_match_results is not None:
            never_matched = _find_never_matched_gt(global_match_results)

        sr = SequenceResult(
            sequence_id=seq.sequence_id,
            num_annotated_frames=len(annotated_ids),
            num_total_frames=max(total_frame_ids, default=0),
            bins=bin_results,
            never_matched_gt=never_matched,
        )
        return sr, raw_per_slice, density_raw

    # ------------------------------------------------------------------
    # Cross-sequence aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        seq_results: List[SequenceResult],
        raw_by_seq: Dict[str, Dict[str, Dict[str, dict]]],
    ) -> Dict[str, BinResult]:
        """Micro-aggregate results across sequences.

        Sums raw TrackEval counts (TP, FP, FN, etc.) across sequences
        via each metric's ``combine_sequences``, then computes final
        ratios from the totals.  This is the standard MOTChallenge /
        TrackEval aggregation and guarantees that per-bin metrics are
        consistent with the all-bin metric (no Simpson's paradox).
        """
        if not seq_results:
            return {}

        all_bin_names: List[str] = []
        seen: set[str] = set()
        for sr in seq_results:
            for bn in sr.bins:
                if bn not in seen:
                    all_bin_names.append(bn)
                    seen.add(bn)

        aggregated: Dict[str, BinResult] = {}
        for bin_name in all_bin_names:
            metric_values: Dict[str, Dict[str, float]] = {}

            for metric in self._trackeval_metrics:
                name = _CLASS_TO_OUR_NAME[type(metric).__name__]
                all_res: Dict[str, dict] = {}

                for sr in seq_results:
                    raw_slice = raw_by_seq.get(sr.sequence_id, {}).get(
                        bin_name,
                    )
                    if raw_slice and name in raw_slice:
                        all_res[sr.sequence_id] = raw_slice[name]

                if all_res:
                    combined = metric.combine_sequences(all_res)
                    metric_values[name] = _format_results(metric, combined)
                else:
                    metric_values[name] = {}

            exclusive_total = sum(
                raw_by_seq.get(sr.sequence_id, {})
                .get(bin_name, {})
                .get("_exclusive_gt_objects", 0)
                for sr in seq_results
            )
            if "mota" in metric_values and metric_values["mota"]:
                metric_values["mota"]["num_gt_objects"] = float(
                    exclusive_total,
                )

            if self._porr_metric is not None:
                porr_all: Dict[str, dict] = {}
                for sr in seq_results:
                    raw_slice = raw_by_seq.get(sr.sequence_id, {}).get(
                        bin_name,
                    )
                    if raw_slice and "porr" in raw_slice:
                        porr_all[sr.sequence_id] = raw_slice["porr"]
                if porr_all:
                    combined_porr = self._porr_metric.combine_sequences(
                        porr_all,
                    )
                    metric_values["porr"] = format_porr_metrics(combined_porr)

            aggregated[bin_name] = BinResult(
                bin_name=bin_name,
                metric_values=metric_values,
            )

        return aggregated
