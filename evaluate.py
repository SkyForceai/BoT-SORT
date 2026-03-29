"""Standalone MOT evaluation entry point.

Single-sequence and multi-sequence modes are auto-detected:

* ``--gt-csv`` is a **file** → single-sequence evaluation.
* ``--gt-csv`` is a **directory** → batch evaluation over all GT CSVs
  found inside (naming convention: ``{seq_name}_gt.csv``).

Usage::

    # Single sequence
    python evaluate.py -c configs/default.yaml \\
        --gt-csv /path/to/seq_gt.csv --pred-csv /path/to/tracks.csv

    # Batch (directory of GT CSVs + directory of per-sequence pred dirs)
    python evaluate.py -c configs/default.yaml \\
        --gt-csv /path/to/gt_csv/ --pred-csv /path/to/outputs/run/

When ``output.save_video`` is true in the config (same as ``runner.py``), prediction
boxes are burned into ``output.mp4`` under ``-o`` (one folder per sequence in batch mode).
Frames are read from ``data`` using the same ``resolve_sequences`` names as the runner.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from data import iter_frames, resolve_sequences
from evaluation import (
    EvalConfig,
    EvalSequence,
    EvaluationPipeline,
    resolve_annotation_mask,
)
from recorder import RunRecorder

logger = logging.getLogger(__name__)

_GT_SUFFIX = "_gt"


def _build_sequences_from_dir(
    gt_dir: Path,
    pred_path: Path,
    pipeline: EvaluationPipeline,
    eval_cfg: EvalConfig,
) -> List[EvalSequence]:
    """Build :class:`EvalSequence` objects for every GT CSV in *gt_dir*."""
    gt_files = sorted(gt_dir.glob("*.csv"))
    if not gt_files:
        print(f"Error: no CSV files found in gt directory: {gt_dir}", file=sys.stderr)
        sys.exit(1)

    sequences: list[EvalSequence] = []
    for gt_file in gt_files:
        seq_name = gt_file.stem.removesuffix(_GT_SUFFIX)

        if pred_path.is_dir():
            seq_pred = pred_path / seq_name / "tracks.csv"
            if not seq_pred.is_file():
                seq_pred = pred_path / f"{seq_name}.csv"
        else:
            seq_pred = pred_path

        if not seq_pred.is_file():
            logger.warning("Pred CSV not found for %s — skipping", seq_name)
            continue

        gt_data, default_mask = pipeline.parser.parse_ground_truth(gt_file)
        mask = resolve_annotation_mask(
            gt_file, seq_name, eval_cfg.annotation_frames_path, default_mask,
        )
        pred_data = pipeline.parser.parse_predictions(seq_pred)

        sequences.append(EvalSequence(
            sequence_id=seq_name,
            predictions=pred_data,
            ground_truth=gt_data,
            annotation_mask=mask,
        ))

    return sequences


def _build_single_sequence(
    gt_csv: Path,
    pred_csv: Path,
    pipeline: EvaluationPipeline,
    eval_cfg: EvalConfig,
) -> List[EvalSequence]:
    """Build a single :class:`EvalSequence`."""
    if not gt_csv.is_file():
        print(f"Error: gt_csv not found: {gt_csv}", file=sys.stderr)
        sys.exit(1)
    if not pred_csv.is_file():
        print(f"Error: pred_csv not found: {pred_csv}", file=sys.stderr)
        sys.exit(1)

    seq_name = gt_csv.stem.removesuffix(_GT_SUFFIX)
    gt_data, default_mask = pipeline.parser.parse_ground_truth(gt_csv)
    mask = resolve_annotation_mask(
        gt_csv, seq_name, eval_cfg.annotation_frames_path, default_mask,
    )
    pred_data = pipeline.parser.parse_predictions(pred_csv)

    return [EvalSequence(
        sequence_id=seq_name,
        predictions=pred_data,
        ground_truth=gt_data,
        annotation_mask=mask,
    )]


def _rows_for_frame(
    seq: EvalSequence,
    frame_id: int,
) -> List[Tuple[float, float, float, float, int, float, int]]:
    """Build track rows ``(x1,y1,x2,y2,id,score,cls)`` for one frame."""
    fd = seq.predictions.frames.get(frame_id)
    if not fd:
        return []
    rows: list[Tuple[float, float, float, float, int, float, int]] = []
    for d in fd.detections:
        b = d.bbox_xyxy
        rows.append(
            (
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
                int(d.object_id),
                float(d.score),
                int(d.class_id),
            ),
        )
    return rows


def write_eval_videos(
    cfg: dict,
    sequences: List[EvalSequence],
    output_dir: Path,
) -> None:
    """When ``output.save_video`` is set, render ``output.mp4`` per sequence from ``data`` + predictions."""
    out_cfg = cfg.get("output") or {}
    if not out_cfg.get("save_video", False):
        return

    data_root = cfg.get("data")
    if not data_root:
        logger.warning(
            "output.save_video is true but config has no data section — skipping eval video",
        )
        return

    try:
        resolved = resolve_sequences(data_root)
    except Exception:
        logger.warning(
            "Could not resolve data paths for eval video; skipping.",
            exc_info=True,
        )
        return

    by_name = {name: dc for name, dc in resolved}
    video_fps = float(out_cfg.get("video_fps", 30.0))
    min_box_axis = float(cfg.get("tracker", {}).get("min_box_axis", 0.0))
    multi = len(sequences) > 1
    out_base = Path(output_dir)

    for seq in sequences:
        data_cfg = by_name.get(seq.sequence_id)
        if data_cfg is None:
            logger.warning(
                "No data source entry for sequence %r — skipping eval video",
                seq.sequence_id,
            )
            continue

        vid_dir = out_base / seq.sequence_id if multi else out_base
        vid_dir.mkdir(parents=True, exist_ok=True)

        recorder = RunRecorder(
            vid_dir,
            save_video=True,
            video_fps=video_fps,
            persist_csv=False,
            configure_logging=False,
        )
        try:
            for frame_id, frame_bgr in iter_frames(data_cfg):
                rows = _rows_for_frame(seq, frame_id)
                recorder.write_video_frame_from_track_rows(
                    frame_bgr, frame_id, min_box_axis, rows,
                )
        finally:
            recorder.save()

        logger.info("Eval visualization video: %s", vid_dir / "output.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOT evaluation")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--gt-csv", type=str, default=None,
                        help="GT CSV file or directory of GT CSVs (auto-detected)")
    parser.add_argument("--pred-csv", type=str, default=None,
                        help="Pred CSV file or output directory with per-sequence subdirs (auto-detected)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Override evaluation.output_dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_section = cfg.get("evaluation", {})

    gt_csv = Path(args.gt_csv or eval_section.get("gt_csv", ""))
    pred_csv = Path(args.pred_csv or eval_section.get("pred_csv", ""))

    clearml_task = None
    if eval_section.get("clearml", False):
        try:
            from clearml import Task

            # Same as runner.py: ClearML task name follows output.experiment_name.
            task_name = cfg.get("output", {}).get("experiment_name", "run")
            clearml_task = Task.init(
                project_name="MOT-Tracking",
                task_name=task_name,
            )
            clearml_task.connect(cfg, name="config")
        except Exception:
            logger.warning("ClearML initialisation failed; continuing without it.", exc_info=True)

    output_dir = args.output_dir or eval_section.get("output_dir", "eval_results")

    eval_cfg = EvalConfig.from_dict(eval_section)
    pipeline = EvaluationPipeline(eval_cfg)

    if gt_csv.is_dir():
        sequences = _build_sequences_from_dir(gt_csv, pred_csv, pipeline, eval_cfg)
    else:
        sequences = _build_single_sequence(gt_csv, pred_csv, pipeline, eval_cfg)

    if not sequences:
        print("Error: no valid sequences found for evaluation", file=sys.stderr)
        sys.exit(1)

    pipeline.evaluate_and_report(sequences, output_dir=output_dir)

    write_eval_videos(cfg, sequences, Path(output_dir))

    if clearml_task is not None:
        try:
            clearml_task.flush(wait_for_uploads=True)
            page = clearml_task.get_output_log_web_page()
            print(
                "\nClearML: KPIs are under the experiment's Scalars, Plots, and Table tabs "
                f"(not only the Console log).\nClearML results: {page}",
                file=sys.stderr,
            )
        except Exception:
            logger.warning("ClearML final flush failed.", exc_info=True)


if __name__ == "__main__":
    main()
