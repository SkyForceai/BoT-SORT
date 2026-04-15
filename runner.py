"""BoT-SORT tracking runner.

Supports single-sequence and multi-sequence runs (auto-detected from
``data.path`` or ``data.paths`` — see :func:`data.resolve_sequences`).

Usage::

    python runner.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from data import iter_frames, resolve_sequences
from detectors import build_detector
from recorder import RunRecorder
from registration import build_registration
from reid import build_reid
from tracker.mc_bot_sort import BoTSORT

logger = logging.getLogger(__name__)


def _apply_cmc(tracks, warp: np.ndarray) -> None:
    """Apply full affine CMC to Kalman states (mean **and** covariance).

    State layout: [cx, cy, w, h, vx, vy, vw, vh].
    The 2×3 warp (similarity / partial-affine) is decomposed into a 2×2
    rotation+scale matrix *R* and a translation *t*.  Width / height and
    their velocities are uniformly scaled by *s = sqrt(|det(R)|)*.
    """
    R = warp[:2, :2]
    t = warp[:2, 2]
    s = np.sqrt(np.abs(np.linalg.det(R)))

    A = np.eye(8)
    A[0:2, 0:2] = R    # cx, cy
    A[2, 2] = s         # w
    A[3, 3] = s         # h
    A[4:6, 4:6] = R    # vx, vy
    A[6, 6] = s         # vw
    A[7, 7] = s         # vh

    t8 = np.zeros(8)
    t8[:2] = t

    for strack in tracks:
        if strack.mean is not None:
            strack.mean = A @ strack.mean + t8
            strack.covariance = A @ strack.covariance @ A.T


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_tracker_args(tracker_cfg: Dict, reid_enabled: bool) -> types.SimpleNamespace:
    """Build the attribute-access object that ``BoTSORT.__init__`` expects.

    ``with_reid`` is derived from ``reid.enabled``, not from the tracker
    section, so there is a single source of truth.
    """
    return types.SimpleNamespace(
        track_high_thresh=tracker_cfg.get("track_high_thresh", 0.6),
        track_low_thresh=tracker_cfg.get("track_low_thresh", 0.1),
        new_track_thresh=tracker_cfg.get("new_track_thresh", 0.7),
        track_buffer=tracker_cfg.get("track_buffer", 240),
        match_thresh=tracker_cfg.get("match_thresh", 0.8),
        proximity_thresh=tracker_cfg.get("proximity_thresh", 0.5),
        appearance_thresh=tracker_cfg.get("appearance_thresh", 0.25),
        with_reid=reid_enabled,

        second_match_thresh=tracker_cfg.get("second_match_thresh", 0.5),
        unconfirmed_match_thresh=tracker_cfg.get("unconfirmed_match_thresh", 0.7),
        duplicate_iou_thresh=tracker_cfg.get("duplicate_iou_thresh", 0.15),

        kalman_filter=tracker_cfg.get("kalman_filter", {}),

        #  New Feature: Birth Logic: forward confirmation config to BoTSORT 
        confirmation=tracker_cfg.get("confirmation", {}),
    )


# ------------------------------------------------------------------
# Per-sequence tracking
# ------------------------------------------------------------------

def _track_sequence(
    config: Dict,
    data_cfg: Dict,
    output_dir: Path,
    detector,
    reid_model,
) -> None:
    """Track a single sequence, writing results to *output_dir*.

    A fresh tracker and registration module are created each call
    (they carry per-sequence state).  The *detector* and *reid_model*
    are shared across sequences (expensive GPU resources, stateless).
    """
    reid_enabled = reid_model is not None
    min_box_axis = config["tracker"].get("min_box_axis", 0.0)
    save_video = config["output"].get("save_video", False)
    video_fps = config["output"].get("video_fps", 24.0)

    recorder = RunRecorder(output_dir, save_video=save_video, video_fps=video_fps)
    registration = build_registration(config.get("registration", {}))
    tracker = BoTSORT(make_tracker_args(config["tracker"], reid_enabled))

    t_start = time.time()
    frame_count = 0

    for frame_id, frame_bgr in iter_frames(data_cfg):
        frame_count += 1

        detections = detector.detect(frame_bgr)

        if reid_model is not None and len(detections) > 0:
            features = reid_model.extract(frame_bgr, detections[:, :4])
            detections = np.concatenate(
                [detections, features.astype(np.float32)], axis=1,
            )

        if registration is not None:
            dets_for_reg = detections[:, :4] if len(detections) > 0 else None
            warp = registration.apply(frame_bgr, dets_for_reg)

            _apply_cmc(tracker.tracked_stracks + tracker.lost_stracks, warp)

        online_targets = tracker.update(detections)

        recorder.add_detections(frame_id, detections)
        recorder.add_tracks(frame_id, online_targets, min_box_axis)
        recorder.write_video_frame(frame_bgr, online_targets, frame_id, min_box_axis)

        if frame_count % 20 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / max(elapsed, 1e-6)
            logger.info(
                "Frame %d | %d dets | %d tracks | %.1f fps",
                frame_id, len(detections), len(online_targets), fps,
            )

    elapsed = time.time() - t_start
    logger.info(
        "Done. %d frames in %.1fs (%.1f fps)",
        frame_count, elapsed, frame_count / max(elapsed, 1e-6),
    )

    recorder.save()


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def _evaluate(
    config: Dict,
    sequences: List[Tuple[str, Dict]],
    base_output_dir: Path,
    multi: bool,
) -> None:
    """Run evaluation over all tracked sequences (if enabled)."""
    eval_section = config.get("evaluation", {})
    if not eval_section.get("enabled", False):
        return

    gt_csv = eval_section.get("gt_csv", "")
    if not gt_csv:
        logger.warning("evaluation.enabled is true but gt_csv is empty — skipping")
        return

    from evaluation import (
        EvalConfig,
        EvalSequence,
        EvaluationPipeline,
        resolve_annotation_mask,
        resolve_gt_path,
    )

    eval_cfg = EvalConfig.from_dict(eval_section)
    pipeline = EvaluationPipeline(eval_cfg)

    eval_seqs: list[EvalSequence] = []
    for seq_name, _ in sequences:
        gt_path = resolve_gt_path(gt_csv, seq_name)
        seq_dir = base_output_dir / seq_name if multi else base_output_dir
        pred_csv = eval_section.get("pred_csv", "")
        pred_path = Path(pred_csv) if pred_csv else seq_dir / "tracks.csv"

        if not gt_path.is_file():
            logger.warning("GT CSV not found for %s: %s — skipping", seq_name, gt_path)
            continue
        if not pred_path.is_file():
            logger.warning("Pred CSV not found for %s: %s — skipping", seq_name, pred_path)
            continue

        gt_data, default_mask = pipeline.parser.parse_ground_truth(gt_path)
        mask = resolve_annotation_mask(
            gt_path, seq_name, eval_cfg.annotation_frames_path, default_mask,
        )
        pred_data = pipeline.parser.parse_predictions(pred_path)

        eval_seqs.append(EvalSequence(
            sequence_id=seq_name,
            predictions=pred_data,
            ground_truth=gt_data,
            annotation_mask=mask,
        ))

    if not eval_seqs:
        logger.warning("No valid sequences for evaluation")
        return

    eval_out = Path(eval_section.get("output_dir", base_output_dir / "eval"))
    pipeline.evaluate_and_report(eval_seqs, output_dir=eval_out)
    logger.info("Evaluation report saved to %s", eval_out)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def init_clearml(config: Dict) -> None:
    """Initialise a ClearML Task when ``evaluation.clearml`` is enabled."""
    if not config.get("evaluation", {}).get("clearml", False):
        return
    try:
        from clearml import Task

        task = Task.init(
            project_name="MOT-Tracking",
            task_name=config["output"].get("experiment_name", "run"),
        )
        task.connect(config, name="config")
    except Exception:
        logger.warning("ClearML initialisation failed; continuing without it.", exc_info=True)


def run(config: Dict) -> None:
    init_clearml(config)

    base_output_dir = Path(config["output"]["dir"]) / config["output"]["experiment_name"]
    base_output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(config["detector"])
    reid_model = build_reid(config.get("reid", {}))

    sequences = resolve_sequences(config["data"])
    multi = len(sequences) > 1

    for idx, (seq_name, data_cfg) in enumerate(sequences, 1):
        if multi:
            logger.info("=== Sequence %d/%d: %s ===", idx, len(sequences), seq_name)
        seq_dir = base_output_dir / seq_name if multi else base_output_dir
        _track_sequence(config, data_cfg, seq_dir, detector, reid_model)

    _evaluate(config, sequences, base_output_dir, multi)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BoT-SORT tracking runner")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
